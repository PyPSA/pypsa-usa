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
import pandas as pd
import pypsa
import yaml
from _helpers import (
    configure_logging,
    log_network_schema,
    update_config_from_wildcards,
)
from constants import HOURS_PER_YEAR
from opts.bidirectional_link import add_bidirectional_link_constraints
from opts.interchange import add_interchange_constraints
from opts.land import add_land_use_constraints
from opts.policy import (
    add_regional_co2limit,
    add_RPS_constraints,
    add_RPS_constraints_sector,
    add_technology_capacity_target_constraints,
    apply_forced_retirements,
)
from opts.reserves import (
    add_ERM_constraints,
    add_operational_reserve_margin,
    store_ERM_duals,
)
from opts.sector import (
    add_cooling_heat_pump_constraints,
    add_demand_response_constraint,
    add_ev_generation_constraint,
    add_fossil_generation_constraint,
    add_gshp_capacity_constraint,
    add_ng_import_export_limits,
    add_sector_co2_constraints,
    add_sector_demand_response_constraints,
    add_water_heater_constraints,
)

logger_gurobi = logging.getLogger("gurobipy")
logger_gurobi.propagate = False

logger = logging.getLogger(__name__)
logging.getLogger("pypsa.network.power_flow").setLevel(logging.WARNING)


def prepare_network(n, solve_opts=None):
    if "clip_p_max_pu" in solve_opts:
        df = n.generators_t.p_max_pu
        n.generators_t.p_max_pu = df.where(df > solve_opts["clip_p_max_pu"], other=0.0)
        df = n.generators_t.p_min_pu
        n.generators_t.p_min_pu = df.where(df > solve_opts["clip_p_max_pu"], other=0.0)

        df = n.links_t.p_max_pu
        n.links_t.p_max_pu = df.where(df > solve_opts["clip_p_max_pu"], other=0.0)
        df = n.links_t.p_min_pu
        n.links_t.p_min_pu = df.where(df > solve_opts["clip_p_max_pu"], other=0.0)

        df = n.storage_units_t.inflow
        n.storage_units_t.inflow = df.where(df > solve_opts["clip_p_max_pu"], other=0.0)
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

        n.add(
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
        for t in n.components:
            if "marginal_cost" in t.static:
                t.static["marginal_cost"] += 1e-2 + 2e-3 * (np.random.random(len(t.static)) - 0.5)

        for t in (n.components[c] for c in ["Line", "Link"]):
            t.static["capital_cost"] += (1e-1 + 2e-2 * (np.random.random(len(t.static)) - 0.5)) * t.static["length"]

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
        n.snapshot_weightings[:] = HOURS_PER_YEAR / nhours

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
        "RPS": lambda: (
            add_RPS_constraints(n, config, global_snakemake) if n.generators.p_nom_extendable.any() else None
        ),
        "REM": lambda: add_regional_co2limit(n, config) if n.generators.p_nom_extendable.any() else None,
        "ERM": lambda: (
            add_ERM_constraints(n, snapshots, config, global_snakemake) if n.generators.p_nom_extendable.any() else None
        ),
        "TCT": lambda: (
            add_technology_capacity_target_constraints(n, config) if n.generators.p_nom_extendable.any() else None
        ),
    }

    # Some constraints have different logic for sector networks
    if sector_enabled:
        constraint_registry["RPS"] = lambda: (
            add_RPS_constraints_sector(n, config, global_snakemake) if n.generators.p_nom_extendable.any() else None
        )
        constraint_registry["REM"] = lambda: (
            add_sector_co2_constraints(n, config) if n.generators.p_nom_extendable.any() else None
        )

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
            add_interchange_constraints(n, config, "imports", sector_enabled)

    # Apply interchange constraints if configured
    if config["electricity"].get("exports", {}).get("enable", False):
        if config["electricity"].get("exports", {}).get("volume_limit", False):
            add_interchange_constraints(n, config, "exports", sector_enabled)

    # Apply sector-specific constraints if sector is enabled
    if sector_enabled:
        # Heat pump constraints
        add_cooling_heat_pump_constraints(n, config)

        # Apply GSHP capacity constraint if urban/rural not split
        if not config["sector"]["service_sector"].get("split_urban_rural", False):
            add_gshp_capacity_constraint(n, config, global_snakemake)

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

        # Fossil generation constraints
        add_fossil_generation_constraint(n, config)


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
        n.model.print_infeasibilities()
        raise RuntimeError("Solving status 'infeasible'")


def _stash_original_nominal(n: pypsa.Network) -> None:
    """Snapshot the pre-solve p_nom / e_nom into `*_initial` once, on first call.

    PyPSA's statistics use `n.components[c].static[p_nom]` as the "installed" baseline. The
    myopic freeze loop overwrites p_nom with p_nom_opt between horizons, which
    erases the original baseline. Stashing it lets us restore it after the
    final horizon so downstream tools can recover (p_nom_opt - p_nom_initial)
    as "what was actually built across all horizons".
    """
    for c in (n.components[name] for name in ["Generator", "Link", "StorageUnit", "Store"]):
        attr = "e_nom" if c.name == "Store" else "p_nom"
        col = f"{attr}_initial"
        if col not in c.static.columns:
            c.static[col] = c.static[attr]


def _restore_original_nominal(n: pypsa.Network) -> None:
    """Restore stashed pre-solve p_nom / e_nom so statistics see original baseline."""
    for c in (n.components[name] for name in ["Generator", "Link", "StorageUnit", "Store"]):
        attr = "e_nom" if c.name == "Store" else "p_nom"
        col = f"{attr}_initial"
        if col in c.static.columns:
            c.static[attr] = c.static[col]


def freeze_prior_periods(n: pypsa.Network, prior_period: int):
    renewable_carriers = set(n.config["electricity"].get("renewable_carriers", []))
    for c in (n.components[name] for name in ["Generator", "Link", "StorageUnit", "Store"]):
        # empty components carry an int64 index under pypsa v1, which breaks .str
        if c.static.empty:
            continue
        attr = "e_nom" if c.name == "Store" else "p_nom"

        prior = c.static.build_year <= prior_period
        # Only assets explicitly tagged "existing" in their name (split out by
        # attach_multihorizon_existing_generators in add_extra_components.py) AND
        # not on a renewable carrier are eligible for economic retirement. Renewables
        # attrite via lifetime, not economics, so they're excluded even if their name
        # happens to match.
        existing = c.static.index.str.contains("existing", case=False, na=False)
        not_renewable = ~c.static["carrier"].isin(renewable_carriers)
        retirable = prior & existing & not_renewable

        # lock in the optimized capacity from the prior period as the starting point
        # for the next period — without this, p_nom still holds the pre-solve value
        # (e.g. 0 for a new-build), so the next period's dispatch constraints would
        # see the wrong installed capacity
        c.static.loc[prior, attr] = c.static.loc[prior, attr + "_opt"]

        # freeze all prior-period assets by default; the optimizer cannot add more
        # capacity through assets that have already been built
        c.static.loc[prior, attr + "_extendable"] = False

        # "existing" vintage assets carry p_nom_min=0 already (set by the split in
        # add_extra_components.py), so flipping them back to extendable lets the
        # optimizer retire them by shrinking p_nom toward zero; p_nom_max is capped
        # at the locked-in capacity so no new capacity can be added through this asset
        c.static.loc[retirable, attr + "_extendable"] = True
        c.static.loc[retirable, attr + "_max"] = c.static.loc[retirable, attr]


def solve_network(n, config, solving, opts="", **kwargs):
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    foresight = snakemake.params.foresight
    kwargs["multi_investment_periods"] = True

    kwargs["solver_options"] = solving["solver_options"][set_of_options] if set_of_options else {}
    kwargs["solver_name"] = solving["solver"]["name"]
    kwargs["extra_functionality"] = extra_functionality
    kwargs["transmission_losses"] = cf_solving.get("transmission_losses", False)
    kwargs["linearized_unit_commitment"] = cf_solving.get(
        "linearized_unit_commitment",
        False,
    )
    kwargs["assign_all_duals"] = cf_solving.get("assign_all_duals", False)

    sns_portion = cf_solving.get("snapshot_portion", None)
    if sns_portion:
        logger.info(f"Optimizing over snapshots from {sns_portion['start']} to {sns_portion['end']}")
        sns_portion = pd.date_range(start=sns_portion["start"], end=sns_portion["end"], freq="h")
        sns = n.snapshots
        sns_portion = sns[sns.get_level_values(1).isin(sns_portion)]
        sns_portion.name = "snapshot"
        kwargs["snapshots"] = sns_portion

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
            _stash_original_nominal(n)
            for i, planning_horizon in enumerate(n.investment_periods):
                sns_horizon = n.snapshots[n.snapshots.get_level_values(0) == planning_horizon]
                kwargs["snapshots"] = sns_horizon

                if "TCT" in opts:
                    apply_forced_retirements(n, planning_horizon, config)

                run_optimize(n, rolling_horizon, skip_iterations, cf_solving, **kwargs)

                logger.info(f"Preparing brownfield from {planning_horizon}")
                freeze_prior_periods(n, planning_horizon)
            # Restore the pre-solve p_nom/e_nom baseline so downstream statistics
            # and plots can compute (p_nom_opt - p_nom) = what was actually built
            # across the myopic horizons. Without this, the inter-period freeze
            # leaves p_nom == p_nom_opt and the "new capacity" signal is zero.
            _restore_original_nominal(n)
        case _:
            raise ValueError(f"Invalid foresight option: '{foresight}'. Must be 'perfect' or 'myopic'.")

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_network",
            interconnect="eastern",
            simpl="120",
            clusters="6m",
            ll="v1.0",
            opts="1h-TCT",
            sector="E-G",
            planning_horizons="2030",
        )
    configure_logging(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    opts = snakemake.wildcards.opts
    opts = [o for o in opts.split("-") if o != ""]
    solve_opts = snakemake.params.solving["options"]

    # sector specific co2 options
    if snakemake.wildcards.sector != "E":
        opts.append("sector")

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)
    schema_entry = log_network_schema(n, stage="entry")

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
    log_network_schema(n, stage="exit", baseline=schema_entry)
    n.export_to_netcdf(snakemake.output[0])
    with open(snakemake.output.config, "w") as file:
        yaml.dump(
            n.meta,
            file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
