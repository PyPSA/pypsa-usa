"""
Solves optimal operation and capacity for a network with the option to
iteratively optimize while updating line reactances.

This script is used for optimizing the electrical network as well as the
sector coupled network.

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.

The optimization is based on the :func:`network.optimize` function.
Additionally, some extra constraints specified in :mod:`solve_network` are added.

.. note::

    The rules ``solve_elec_networks`` and ``solve_sector_networks`` run
    the workflow for all scenarios in the configuration file (``scenario:``)
    based on the rule :mod:`solve_network`.
"""

import logging

import numpy as np
import pypsa
import yaml
from _helpers import (
    configure_logging,
    update_config_from_wildcards,
    update_config_with_sector_opts,
)
from opts.content import (
    add_BAU_constraints,
    add_EQ_constraints,
    add_regional_co2limit,
    add_RPS_constraints,
    add_technology_capacity_target_constraints,
)
from opts.land import add_land_use_constraints
from opts.reserves import add_ERM_constraints, add_operational_reserve_margin, add_PRM_constraints
from opts.sector import (
    add_cooling_heat_pump_constraints,
    add_demand_response_constraint,
    add_ev_generation_constraint,
    add_gshp_capacity_constraint,
    add_ng_import_export_limits,
    add_sector_co2_constraints,
    add_sector_demand_response_constraints,
    add_water_heater_constraints,
)

logger_gurobi = logging.getLogger("gurobipy")
logger_gurobi.propagate = False

logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)


def prepare_network(n, solve_opts=None):
    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,
            n.storage_units_t.inflow,
        ):
            df = df.where(df > solve_opts["clip_p_max_pu"], other=0.0)

    load_shedding = solve_opts.get("load_shedding")
    if load_shedding:
        # intersect between macroeconomic and surveybased willingness to pay
        # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
        # TODO: retrieve color and nice name from config
        logger.warning("Adding load shedding generators.")
        n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
        buses_i = n.buses.query("carrier == 'AC'").index
        if not np.isscalar(load_shedding):
            # TODO: do not scale via sign attribute (use Eur/MWh instead of Eur/kWh)
            load_shedding = 1e2  # Eur/kWh

        n.madd(
            "Generator",
            buses_i,
            " load",
            bus=buses_i,
            carrier="load",
            sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=load_shedding,  # Eur/kWh
            p_nom=1e9,  # kW
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components():
            if "marginal_cost" in t.df:
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (np.random.random(len(t.df)) - 0.5)

        for t in n.iterate_components(["Line", "Link"]):
            t.df["capital_cost"] += (1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    return n


def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to
    ``pypsa.optimization.optimize``.

    If you want to enforce additional custom constraints, this is a good
    location to add them. The arguments ``opts`` and
    ``snakemake.config`` are expected to be attached to the network.
    """
    opts = n.opts
    config = n.config
    sector_enabled = "sector" in opts

    # Make snakemake available in function scope if it exists in global scope
    global_snakemake = globals().get("snakemake")

    # Define constraint application functions in a registry
    # Each function should take network and necessary parameters
    constraint_registry = {
        "RPS": lambda: add_RPS_constraints(n, config, sector_enabled, global_snakemake)
        if n.generators.p_nom_extendable.any()
        else None,
        "REM": lambda: add_regional_co2limit(n, snapshots, config, global_snakemake)
        if n.generators.p_nom_extendable.any()
        else None,
        "BAU": lambda: add_BAU_constraints(n, config) if n.generators.p_nom_extendable.any() else None,
        "PRM": lambda: add_PRM_constraints(n, config) if n.generators.p_nom_extendable.any() else None,
        "ERM": lambda: add_ERM_constraints(n, config) if n.generators.p_nom_extendable.any() else None,
        "TCT": lambda: add_technology_capacity_target_constraints(n, config)
        if n.generators.p_nom_extendable.any()
        else None,
        "EQ": lambda: add_EQ_constraints(n, config),
    }

    # Apply constraints based on options
    for opt in opts:
        if opt in constraint_registry:
            constraint_registry[opt]()

    # Always apply land use constraints
    add_land_use_constraints(n)

    # Apply operational reserve if configured
    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)

    # Apply demand response if configured
    dr_config = config["electricity"].get("demand_response", {})
    if dr_config:
        add_demand_response_constraint(n, config, sector_enabled)

    # Apply sector-specific constraints if sector is enabled
    if sector_enabled:
        apply_sector_constraints(n, config, global_snakemake)


def apply_sector_constraints(n, config, global_snakemake):
    """Apply all sector-specific constraints to the network."""
    # Heat pump constraints
    add_cooling_heat_pump_constraints(n, config)

    # Apply GSHP capacity constraint if urban/rural not split
    if not config["sector"]["service_sector"].get("split_urban_rural", False):
        add_gshp_capacity_constraint(n, config, global_snakemake)

    # CO2 constraints for sectors
    if config["sector"]["co2"].get("policy", {}):
        add_sector_co2_constraints(n, config)

    # Natural gas import/export constraints
    if config["sector"]["natural_gas"].get("imports", False):
        add_ng_import_export_limits(n, config)

    # Water heater constraints
    water_config = config["sector"]["service_sector"].get("water_heating", {})
    if not water_config.get("simple_storage", True):
        add_water_heater_constraints(n, config)

    # EV generation constraints
    if config["sector"]["transport_sector"]["investment"]["ev_policy"]:
        if not config["sector"]["transport_sector"]["investment"]["exogenous"]:
            add_ev_generation_constraint(n, config, global_snakemake)

    # Sector demand response constraints
    add_sector_demand_response_constraints(n, config)


def solve_network(n, config, solving, opts="", **kwargs):
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    if "sector" not in opts:
        kwargs["multi_investment_periods"] = config["foresight"] == "perfect"

    kwargs["solver_options"] = solving["solver_options"][set_of_options] if set_of_options else {}
    kwargs["solver_name"] = solving["solver"]["name"]
    kwargs["extra_functionality"] = extra_functionality
    kwargs["transmission_losses"] = cf_solving.get("transmission_losses", False)
    kwargs["linearized_unit_commitment"] = cf_solving.get(
        "linearized_unit_commitment",
        False,
    )
    kwargs["assign_all_duals"] = cf_solving.get("assign_all_duals", False)

    rolling_horizon = cf_solving.pop("rolling_horizon", False)
    skip_iterations = cf_solving.pop("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    # add to network for additional_constraints
    n.config = config
    n.opts = opts

    if rolling_horizon:
        kwargs["horizon"] = cf_solving.get("horizon", 365)
        kwargs["overlap"] = cf_solving.get("overlap", 0)
        n.optimize.optimize_with_rolling_horizon(**kwargs)
        status, condition = "", ""
    elif skip_iterations:
        status, condition = n.optimize(**kwargs)
    else:
        kwargs["track_iterations"] = (cf_solving.get("track_iterations", False),)
        kwargs["min_iterations"] = (cf_solving.get("min_iterations", 4),)
        kwargs["max_iterations"] = (cf_solving.get("max_iterations", 6),)
        status, condition = n.optimize.optimize_transmission_expansion_iteratively(
            **kwargs,
        )

    if status != "ok" and not rolling_horizon:
        logger.warning(
            f"Solving status '{status}' with termination condition '{condition}'",
        )
    if "infeasible" in condition:
        # n.model.print_infeasibilities()
        raise RuntimeError("Solving status 'infeasible'")

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_network",
            interconnect="western",
            simpl="11",
            clusters="4m",
            ll="v1.0",
            opts="4h",
            sector="E-G",
            planning_horizons="2030",
        )
    configure_logging(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)
    if "sector_opts" in snakemake.wildcards.keys():
        update_config_with_sector_opts(
            snakemake.config,
            snakemake.wildcards.sector_opts,
        )

    opts = snakemake.wildcards.opts
    if "sector_opts" in snakemake.wildcards.keys():
        opts += "-" + snakemake.wildcards.sector_opts
    opts = [o for o in opts.split("-") if o != ""]
    solve_opts = snakemake.params.solving["options"]

    # sector specific co2 options
    if snakemake.wildcards.sector != "E":
        # sector co2 limits applied via config file, not through Co2L
        opts = [x for x in opts if not x.startswith("Co2L")]
        opts.append("sector")

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)

    n = prepare_network(
        n,
        solve_opts,
    )

    n = solve_network(
        n,
        config=snakemake.config,
        solving=snakemake.params.solving,
        opts=opts,
        log_fn=snakemake.log.solver,
    )
    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])

    with open(snakemake.output.config, "w") as file:
        yaml.dump(
            n.meta,
            file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
