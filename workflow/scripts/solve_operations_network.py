import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.linopf import define_constraints
from pypsa.linopf import define_variables
from pypsa.linopf import get_var
from pypsa.linopf import ilopf
from pypsa.linopf import join_exprs
from pypsa.linopf import linexpr
from pypsa.linopf import network_lopf
from vresutils.benchmark import memory_logger

logger = logging.getLogger(__name__)


def prepare_network(n, solve_opts):

    if "clip_p_max_pu" in solve_opts:
        for df in (n.generators_t.p_max_pu, n.storage_units_t.inflow):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    load_shedding = solve_opts.get("load_shedding")
    if load_shedding:
        buses_i = n.buses.query("carrier == 'AC'").index.to_list()
        add_load_shedding(n, buses=buses_i, marginal_cost=load_shedding)

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components(n.one_port_components):
            # if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if "marginal_cost" in t.df:
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.random.random(len(t.df)) - 0.5
                )

        for t in n.iterate_components(["Line", "Link"]):
            t.df["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)
            ) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    return n


def add_SAFE_constraints(n, config):
    peakdemand = (
        1.0 + config["electricity"]["SAFE_reservemargin"]
    ) * n.loads_t.p_set.sum(axis=1).max()
    conv_techs = config["plotting"]["conv_techs"]
    exist_conv_caps = n.generators.query(
        "~p_nom_extendable & carrier in @conv_techs",
    ).p_nom.sum()
    ext_gens_i = n.generators.query("carrier in @conv_techs & p_nom_extendable").index
    lhs = linexpr((1, get_var(n, "Generator", "p_nom")[ext_gens_i])).sum()
    rhs = peakdemand - exist_conv_caps
    define_constraints(n, lhs, ">=", rhs, "Safe", "mintotalcap")


def add_operational_reserve_margin_constraint(n, config):

    reserve_config = config["electricity"]["operational_reserve"]
    EPSILON_LOAD = reserve_config["epsilon_load"]
    EPSILON_VRES = reserve_config["epsilon_vres"]
    CONTINGENCY = reserve_config["contingency"]

    # Reserve Variables
    reserve = get_var(n, "Generator", "r")
    lhs = linexpr((1, reserve)).sum(1)

    # Share of extendable renewable capacities
    ext_i = n.generators.query("p_nom_extendable").index
    vres_i = n.generators_t.p_max_pu.columns
    if not ext_i.empty and not vres_i.empty:
        capacity_factor = n.generators_t.p_max_pu[vres_i.intersection(ext_i)]
        renewable_capacity_variables = get_var(n, "Generator", "p_nom")[
            vres_i.intersection(ext_i)
        ]
        lhs += linexpr(
            (-EPSILON_VRES * capacity_factor, renewable_capacity_variables),
        ).sum(1)

    # Total demand at t
    demand = n.loads_t.p.sum(1)

    # VRES potential of non extendable generators
    capacity_factor = n.generators_t.p_max_pu[vres_i.difference(ext_i)]
    renewable_capacity = n.generators.p_nom[vres_i.difference(ext_i)]
    potential = (capacity_factor * renewable_capacity).sum(1)

    # Right-hand-side
    rhs = EPSILON_LOAD * demand + EPSILON_VRES * potential + CONTINGENCY

    define_constraints(n, lhs, ">=", rhs, "Reserve margin")


def add_operational_reserve_margin(n, sns, config):
    """
    Build reserve margin constraints based on the formulation given in
    https://genxproject.github.io/GenX/dev/core/#Reserves.
    """

    define_variables(n, 0, np.inf, "Generator", "r", axes=[sns, n.generators.index])

    add_operational_reserve_margin_constraint(n, config)

    update_capacity_constraint(n)


def add_battery_constraints(n):
    nodes = n.buses.index[n.buses.carrier == "battery"]
    if nodes.empty or ("Link", "p_nom") not in n.variables.index:
        return
    link_p_nom = get_var(n, "Link", "p_nom")
    lhs = linexpr(
        (1, link_p_nom[nodes + " charger"]),
        (
            -n.links.loc[nodes + " discharger", "efficiency"].values,
            link_p_nom[nodes + " discharger"].values,
        ),
    )
    define_constraints(n, lhs, "=", 0, "Link", "charger_ratio")


def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to
    ``pypsa.linopf.network_lopf``.

    If you want to enforce additional custom constraints, this is a good location to add them.
    The arguments ``opts`` and ``snakemake.config`` are expected to be attached to the network.
    """
    opts = n.opts
    config = n.config
    if "BAU" in opts and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, config)
    if "SAFE" in opts and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, config)
    if "CCL" in opts and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, config)
    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)
    for o in opts:
        if "EQ" in o:
            add_EQ_constraints(n, o)
    add_battery_constraints(n)


def solve_network(n, config, opts="", **kwargs):
    solver_options = config["solving"]["solver"].copy()
    solver_name = solver_options.pop("name")
    cf_solving = config["solving"]["options"]
    track_iterations = cf_solving.get("track_iterations", False)
    min_iterations = cf_solving.get("min_iterations", 4)
    max_iterations = cf_solving.get("max_iterations", 6)

    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    skip_iterations = cf_solving.get("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    if skip_iterations:
        network_lopf(
            n,
            solver_name=solver_name,
            solver_options=solver_options,
            extra_functionality=extra_functionality,
            **kwargs,
        )
    else:
        ilopf(
            n,
            solver_name=solver_name,
            solver_options=solver_options,
            track_iterations=track_iterations,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            extra_functionality=extra_functionality,
            **kwargs,
        )
    return n


def solve_operations_model(n, solve_opts, solver_options):
    load_shedding = True
    solver_name = solver_options.pop("name")
    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
    set_all_extendable_to(n, False)
    if load_shedding:
        buses_i = n.buses.query("carrier == 'AC'").index.to_list()
        add_load_shedding(n, buses=buses_i, marginal_cost=load_shedding)
    n.optimize(n.snapshots, solver_name=solver_name, solver_options=solver_options)
    return n


def set_all_extendable_to(n, val):
    for component in n.iterate_components():
        if hasattr(component.df, "p_nom_extendable"):
            component.df.p_nom_extendable = val
        if hasattr(component.df, "s_nom_extendable"):
            component.df.s_nom_extendable = val


def add_load_shedding(n, buses=None, marginal_cost=10000, p_nom=1e6):
    # intersect between macroeconomic and surveybased
    # willingness to pay
    # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full)
    if marginal_cost == True:
        marginal_cost = 10000  # $/MWh
    elif not isinstance(marginal_cost, (int, float)):
        marginal_cost = 10000  # $/MWh
    buses = buses if buses else n.buses.index
    n.optimize.add_load_shedding(
        buses=buses,
        sign=1,  # keep everything in MW
        marginal_cost=marginal_cost,  # Cost of load shedding
        p_nom=p_nom,  # Max load shedding
        suffix=" load",
    )
    n.carriers.at["Load", "color"] = "#dd2e23"
    n.carriers.at["Load", "nice_name"] = "Load Shedding"
    n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_network",
            interconnect="western",
            # simpl='',
            clusters="30",
            ll="vopt",
            opts="Co2L0.75",
        )
    configure_logging(snakemake)

    tmpdir = snakemake.config["solving"].get("tmpdir")
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
    opts = snakemake.wildcards.opts.split("-")
    solve_opts = snakemake.config["solving"]["options"]
    solver_options = snakemake.config["solving"]["solver"]

    fn = getattr(snakemake.log, "memory", None)
    with memory_logger(filename=fn, interval=30.0) as mem:
        n = pypsa.Network(snakemake.input[0])
        n = solve_operations_model(n, solve_opts, solver_options)
        # n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
        n.export_to_netcdf(snakemake.output[0])

    logger.info(f"Maximum memory usage: {mem.mem_usage}")

    # n.optimize.fix_optimal_capacities()
    # n = prepare_network(n, solve_opts, config=snakemake.config)
    # n = solve_network(
    #     n, config=snakemake.config, opts=opts, log_fn=snakemake.log.solver
    # )
