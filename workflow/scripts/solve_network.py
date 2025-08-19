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

import copy
import logging

import numpy as np
import pandas as pd
import pypsa
import yaml
from _helpers import (
    configure_logging,
    update_config_from_wildcards,
)
from opts.bidirectional_link import add_bidirectional_link_constraints
from opts.interchange import add_interchange_constraints
from opts.land import add_land_use_constraints
from opts.policy import (
    add_regional_co2limit,
    add_RPS_constraints,
    add_technology_capacity_target_constraints,
)
from opts.reserves import (
    add_ERM_constraints,
    add_operational_reserve_margin,
    add_PRM_constraints,
    store_ERM_duals,
)
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

    if solve_opts.get("noisy_costs"):  ##random noise to costs of generators
        for t in n.iterate_components():
            if "marginal_cost" in t.df:
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (np.random.random(len(t.df)) - 0.5)

        for t in n.iterate_components(["Line", "Link"]):
            t.df["capital_cost"] += (1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        # Get first nhours for each level of the multi-index
        first_nhours = pd.MultiIndex.from_tuples(
            [
                snap
                for year in n.snapshots.get_level_values(0).unique()
                for snap in n.snapshots[n.snapshots.get_level_values(0) == year][:nhours]
            ],
            names=n.snapshots.names,
        )
        n.set_snapshots(first_nhours)
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
        "REM": lambda: add_regional_co2limit(n, config) if n.generators.p_nom_extendable.any() else None,
        "PRM": lambda: add_PRM_constraints(n, config, global_snakemake)
        if n.generators.p_nom_extendable.any()
        else None,
        "ERM": lambda: add_ERM_constraints(n, config, global_snakemake)
        if n.generators.p_nom_extendable.any()
        else None,
        "TCT": lambda: add_technology_capacity_target_constraints(n, config)
        if n.generators.p_nom_extendable.any()
        else None,
    }

    # Apply constraints based on options
    for opt in opts:
        if opt in constraint_registry:
            constraint_registry[opt]()

    # Always apply land use constraints
    add_land_use_constraints(n)

    # Always apply bidirectional link constraints
    add_bidirectional_link_constraints(n)

    # Apply operational reserve if configured
    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)

    # Apply demand response if configured
    dr_config = config["electricity"].get("demand_response", {})
    if dr_config:
        add_demand_response_constraint(n, config, sector_enabled)

    # Apply interchange constraints if configured
    if config["electricity"].get("imports", {}).get("enable", False):
        if config["electricity"].get("imports", {}).get("volume_limit", False):
            add_interchange_constraints(n, config, "imports")

    # Apply interchange constraints if configured
    if config["electricity"].get("exports", {}).get("enable", False):
        if config["electricity"].get("exports", {}).get("volume_limit", False):
            add_interchange_constraints(n, config, "exports")

    # Apply sector-specific constraints if sector is enabled
    if sector_enabled:
        # Heat pump constraints
        add_cooling_heat_pump_constraints(n, config)

        # Apply GSHP capacity constraint if urban/rural not split
        if not config["sector"]["service_sector"].get("split_urban_rural", False):
            add_gshp_capacity_constraint(n, config, global_snakemake)

        # CO2 constraints for sectors
        if "REMsec" in opts:
            add_sector_co2_constraints(n, config)

        # Natural gas import/export constraints
        if config["sector"]["natural_gas"].get("imports", False):
            add_ng_import_export_limits(n, config)

        # Water heater constraints
        water_config = config["sector"]["service_sector"].get("water_heating", {})
        if not water_config.get("simple_storage", True):
            add_water_heater_constraints(n, config)

        # EV generation constraints
        if config["sector"]["transport_sector"].get("ev_policy", {}):
            add_ev_generation_constraint(n, config, global_snakemake)

        # Sector demand response constraints
        add_sector_demand_response_constraints(n, config)


def run_optimize(n, rolling_horizon, skip_iterations, cf_solving, **kwargs):
    """Initiate the correct type of pypsa.optimize function."""
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


def prepare_brownfield(n, planning_horizon):
    """Prepare the network for the next planning horizon by setting up brownfield constraints.
    Used for myopic foresight.

    This function:
    1. Sets minimum capacities for transmission lines and DC links
    2. Updates generator, link, and storage unit capacities
    3. Handles time-dependent data transfer between planning periods
    """
    # electric transmission grid set optimised capacities of previous as minimum
    n.lines.s_nom_min = n.lines.s_nom_opt  # for lines
    dc_i = n.links[n.links.carrier == "DC"].index
    n.links.loc[dc_i, "p_nom_min"] = n.links.loc[dc_i, "p_nom_opt"]  # for links

    for c in n.iterate_components(["Generator", "Link", "StorageUnit"]):
        nm = c.name
        # limit our components that we remove/modify to those prior to this time horizon
        c_lim = c.df.loc[n.get_active_assets(nm, planning_horizon)]

        logger.info(f"Preparing brownfield for the component {nm}")
        # attribute selection for naming convention
        attr = "p"
        # copy over asset sizing from previous period
        c_lim[f"{attr}_nom"] = c_lim[f"{attr}_nom_opt"]
        c_lim[f"{attr}_nom_extendable"] = False
        df = copy.deepcopy(c_lim)
        time_df = copy.deepcopy(c.pnl)

        for c_idx in c_lim.index:
            n.remove(nm, c_idx)

        for df_idx in df.index:
            if nm == "Generator":
                n.madd(
                    nm,
                    [df_idx],
                    carrier=df.loc[df_idx].carrier,
                    bus=df.loc[df_idx].bus,
                    p_nom_min=df.loc[df_idx].p_nom_min,
                    p_nom=df.loc[df_idx].p_nom,
                    p_nom_max=df.loc[df_idx].p_nom_max,
                    p_nom_extendable=df.loc[df_idx].p_nom_extendable,
                    ramp_limit_up=df.loc[df_idx].ramp_limit_up,
                    ramp_limit_down=df.loc[df_idx].ramp_limit_down,
                    efficiency=df.loc[df_idx].efficiency,
                    marginal_cost=df.loc[df_idx].marginal_cost,
                    capital_cost=df.loc[df_idx].capital_cost,
                    build_year=df.loc[df_idx].build_year,
                    lifetime=df.loc[df_idx].lifetime,
                    heat_rate=df.loc[df_idx].heat_rate,
                    fuel_cost=df.loc[df_idx].fuel_cost,
                    vom_cost=df.loc[df_idx].vom_cost,
                    carrier_base=df.loc[df_idx].carrier_base,
                    p_min_pu=df.loc[df_idx].p_min_pu,
                    p_max_pu=df.loc[df_idx].p_max_pu,
                    land_region=df.loc[df_idx].land_region,
                )
            else:
                n.add(nm, df_idx, **df.loc[df_idx])
        logger.info(n.consistency_check())

        # copy time-dependent
        selection = n.component_attrs[nm].type.str.contains("series")
        for tattr in n.component_attrs[nm].index[selection]:
            n.import_series_from_dataframe(time_df[tattr], nm, tattr)

    # roll over the last snapshot of time varying storage state of charge to be the state_of_charge_initial for the next time period
    n.storage_units.loc[:, "state_of_charge_initial"] = n.storage_units_t.state_of_charge.loc[planning_horizon].iloc[-1]


def solve_network(n, config, solving, opts="", **kwargs):
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    foresight = snakemake.params.foresight
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

    match foresight:
        case "perfect":
            run_optimize(n, rolling_horizon, skip_iterations, cf_solving, **kwargs)
        case "myopic":
            for i, planning_horizon in enumerate(n.investment_periods):
                sns_horizon = n.snapshots[n.snapshots.get_level_values(0) == planning_horizon]
                kwargs["snapshots"] = sns_horizon

                run_optimize(n, rolling_horizon, skip_iterations, cf_solving, **kwargs)

                if i == len(n.investment_periods) - 1:
                    logger.info(f"Final time horizon {planning_horizon}")
                    continue
                logger.info(f"Preparing brownfield from {planning_horizon}")
                prepare_brownfield(n, planning_horizon)

        case _:
            raise ValueError(f"Invalid foresight option: '{foresight}'. Must be 'perfect' or 'myopic'.")

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_network",
            interconnect="western",
            simpl="12",
            clusters="4m",
            ll="v1.0",
            opts="4h-REM",
            sector="E",
            planning_horizons="2030",
        )
    configure_logging(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    opts = snakemake.wildcards.opts
    opts = [o for o in opts.split("-") if o != ""]
    solve_opts = snakemake.params.solving["options"]

    # sector specific co2 options
    if snakemake.wildcards.sector != "E":
        opts = ["REMsec" if x == "REM" else x for x in opts]
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

    if "ERM" in opts:
        store_ERM_duals(n)

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
