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
from pathlib import Path

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
    add_RPS_constraints_sector,
    add_technology_capacity_target_constraints,
    apply_forced_retirements,
    apply_zero_emission_retirements,
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
pypsa.pf.logger.setLevel(logging.WARNING)


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


def _log_gurobi_iis(n) -> None:
    """Log the human-readable linopy names of every constraint in the Gurobi IIS.

    linopy's print_infeasibilities() prints to stdout (not captured by the Python
    logging handler), so this function mirrors the IIS report via logger.warning so
    it always appears in the python log. computeIIS() is idempotent if linopy
    already called it.

    Strategy: use linopy's compute_infeasibilities() to get integer labels and map
    them to names. If that returns nothing (can happen if the tmp ILP file is empty
    due to OS-level buffering), fall back to writing the ILP file ourselves and
    parsing it directly.
    """
    import os
    import tempfile

    grb_model = getattr(getattr(n, "model", None), "solver_model", None)
    if grb_model is None:
        logger.warning("IIS extraction: solver_model not available.")
        return

    # Ensure IIS is computed (idempotent)
    try:
        grb_model.computeIIS()
    except Exception as exc:
        logger.warning(f"IIS extraction: computeIIS() failed: {exc}")
        return

    # --- Path A: use linopy's label mapping ---
    labels = []
    try:
        labels = n.model.compute_infeasibilities()
    except Exception as exc:
        logger.warning(f"IIS extraction: linopy compute_infeasibilities failed: {exc}")

    if labels:
        try:
            from linopy.common import print_single_constraint

            names = []
            for lbl in labels:
                try:
                    names.append(print_single_constraint(n.model, lbl))
                except Exception:
                    names.append(f"<label {lbl}>")
            logger.warning(f"IIS constraints ({len(names)}):\n" + "\n".join(names))
            return
        except Exception as exc:
            logger.warning(f"IIS label lookup failed: {exc}; raw labels: {labels}")

    # --- Path B: write ILP ourselves and log raw content ---
    # Only 2 constraints in the IIS (confirmed by Gurobi log), so file is tiny.
    logger.warning("IIS: falling back to direct ILP file write.")
    try:
        fd, ilp_path = tempfile.mkstemp(suffix=".ilp", prefix="iis_")
        os.close(fd)
        grb_model.write(ilp_path)
        with open(ilp_path) as f:
            ilp_text = f.read()
        os.unlink(ilp_path)
        logger.warning(f"IIS report (raw ILP):\n{ilp_text}")
    except Exception as exc:
        logger.warning(f"IIS extraction (ILP fallback) failed: {exc}")


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
        _log_gurobi_iis(n)
        raise RuntimeError("Solving status 'infeasible'")


def _stash_original_nominal(n: pypsa.Network) -> None:
    """Snapshot the pre-solve p_nom / e_nom into `*_initial` once, on first call.

    PyPSA's statistics use `n.df(c)[p_nom]` as the "installed" baseline. The
    myopic freeze loop overwrites p_nom with p_nom_opt between horizons, which
    erases the original baseline. Stashing it lets us restore it after the
    final horizon so downstream tools can recover (p_nom_opt - p_nom_initial)
    as "what was actually built across all horizons".
    """
    for c in n.iterate_components(["Generator", "Link", "StorageUnit", "Store"]):
        attr = "e_nom" if c.name == "Store" else "p_nom"
        col = f"{attr}_initial"
        if col not in c.df.columns:
            c.df[col] = c.df[attr]


def _restore_original_nominal(n: pypsa.Network) -> None:
    """Restore stashed pre-solve p_nom / e_nom so statistics see original baseline."""
    for c in n.iterate_components(["Generator", "Link", "StorageUnit", "Store"]):
        attr = "e_nom" if c.name == "Store" else "p_nom"
        col = f"{attr}_initial"
        if col in c.df.columns:
            c.df[attr] = c.df[col]


def _mask_future_generators(n: pypsa.Network, planning_horizon: int) -> None:
    """Suppress components with build_year > planning_horizon before a period's solve.

    Generators/links/stores with future build years must not be visible to the
    optimizer during the current period: with multi_investment_periods=True, their
    p_nom decision variables are still created by linopy but have no dispatch
    constraints and no capital-cost objective term (the investment-period weighting
    for a future period is 0 when only current-period snapshots are passed). This
    leaves the variable effectively unbounded, which causes Gurobi to report
    infeasible_or_unbounded even when the actual constraint set is feasible.

    Original values are stashed on n._future_mask_stash and restored by
    _restore_future_generators at the start of the next freeze_prior_periods call.
    """
    stash: dict = {}
    for c in n.iterate_components(["Generator", "Link", "StorageUnit", "Store"]):
        attr = "e_nom" if c.name == "Store" else "p_nom"
        future = c.df.build_year > planning_horizon
        if not future.any():
            continue
        stash[c.name] = {
            "index": c.df.index[future].copy(),
            f"{attr}_extendable": c.df.loc[future, f"{attr}_extendable"].copy(),
            f"{attr}_max": c.df.loc[future, f"{attr}_max"].copy(),
            f"{attr}_min": c.df.loc[future, f"{attr}_min"].copy(),
            attr: c.df.loc[future, attr].copy(),
        }
        c.df.loc[future, f"{attr}_extendable"] = False
        c.df.loc[future, f"{attr}_max"] = 0.0
        c.df.loc[future, f"{attr}_min"] = 0.0
        c.df.loc[future, attr] = 0.0
    n._future_mask_stash = stash


def _restore_future_generators(n: pypsa.Network) -> None:
    """Restore components previously suppressed by _mask_future_generators."""
    if not hasattr(n, "_future_mask_stash"):
        return
    for c in n.iterate_components(["Generator", "Link", "StorageUnit", "Store"]):
        if c.name not in n._future_mask_stash:
            continue
        attr = "e_nom" if c.name == "Store" else "p_nom"
        entry = n._future_mask_stash[c.name]
        idx = entry["index"]
        c.df.loc[idx, f"{attr}_extendable"] = entry[f"{attr}_extendable"]
        c.df.loc[idx, f"{attr}_max"] = entry[f"{attr}_max"]
        c.df.loc[idx, f"{attr}_min"] = entry[f"{attr}_min"]
        c.df.loc[idx, attr] = entry[attr]
    del n._future_mask_stash


def freeze_prior_periods(n: pypsa.Network, prior_period: int):
    # Restore components that were masked before this period's solve, then
    # re-mask generators beyond the next planning horizon so the next solve
    # also has no unbounded future-vintage variables.
    _restore_future_generators(n)

    periods = list(n.investment_periods)
    period_idx = periods.index(prior_period)
    next_period = periods[period_idx + 1] if period_idx + 1 < len(periods) else None

    renewable_carriers = set(n.config["electricity"].get("renewable_carriers", []))
    for c in n.iterate_components(["Generator", "Link", "StorageUnit", "Store"]):
        attr = "e_nom" if c.name == "Store" else "p_nom"

        prior = c.df.build_year <= prior_period
        # Only assets explicitly tagged "existing" in their name (split out by
        # attach_multihorizon_existing_generators in add_extra_components.py) AND
        # not on a renewable carrier are eligible for economic retirement. Renewables
        # attrite via lifetime, not economics, so they're excluded even if their name
        # happens to match.
        existing = c.df.index.str.contains("existing", case=False, na=False)
        not_renewable = ~c.df["carrier"].isin(renewable_carriers)
        retirable = prior & existing & not_renewable

        # lock in the optimized capacity from the prior period as the starting point
        # for the next period — without this, p_nom still holds the pre-solve value
        # (e.g. 0 for a new-build), so the next period's dispatch constraints would
        # see the wrong installed capacity. Clip to zero: solver numerical noise can
        # return tiny negative p_nom_opt values that make p_max_pu*p_nom < 0 in the
        # next period, causing an immediate infeasibility.
        c.df.loc[prior, attr] = c.df.loc[prior, attr + "_opt"].clip(lower=0)

        # freeze all prior-period assets by default; the optimizer cannot add more
        # capacity through assets that have already been built
        c.df.loc[prior, attr + "_extendable"] = False

        # "existing" vintage assets carry p_nom_min=0 already (set by the split in
        # add_extra_components.py), so flipping them back to extendable lets the
        # optimizer retire them by shrinking p_nom toward zero; p_nom_max is capped
        # at the locked-in capacity so no new capacity can be added through this asset
        c.df.loc[retirable, attr + "_extendable"] = True
        c.df.loc[retirable, attr + "_max"] = c.df.loc[retirable, attr]

    if next_period is not None:
        _mask_future_generators(n, next_period)


def update_egs_p_nom_max(n, planning_horizon):
    """Cap EGS p_nom_max by subtracting capacity already built in prior periods.

    The land use constraint only covers extendable generators, so non-extendable
    prior-period EGS (frozen by freeze_prior_periods) is invisible to it. Without
    this, the optimizer can build the full geological supply curve in every period.
    """
    current_egs = n.generators.index[
        n.generators["carrier"].str.contains("EGS", case=False)
        & n.generators["p_nom_extendable"]
        & (n.generators["build_year"] == planning_horizon)
    ]
    if current_egs.empty:
        return
    locked_egs = n.generators[
        n.generators["carrier"].str.contains("EGS", case=False) & ~n.generators["p_nom_extendable"]
    ]
    locked_by_bus = locked_egs.groupby("bus")["p_nom"].sum()
    for gen_idx in current_egs:
        bus = n.generators.loc[gen_idx, "bus"]
        already_built = locked_by_bus.get(bus, 0.0)
        n.generators.loc[gen_idx, "p_nom_max"] = max(0.0, n.generators.loc[gen_idx, "p_nom_max"] - already_built)


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
            output_path = Path(snakemake.output[0])
            periods_list = list(n.investment_periods)

            def _checkpoint_path(horizon):
                return output_path.parent / f"{output_path.stem}_checkpoint_{horizon}.nc"

            # Resume from the latest available checkpoint so a failed run doesn't
            # have to re-solve periods that already converged.
            resume_after = None
            for ph in reversed(periods_list):
                cp = _checkpoint_path(ph)
                if cp.exists():
                    logger.info(f"Resuming from checkpoint after {ph}: loading {cp}")
                    n = pypsa.Network(str(cp))
                    n.config = config
                    n.opts = opts
                    resume_after = ph
                    break

            # _stash_original_nominal is idempotent: if p_nom_initial already exists
            # in the loaded checkpoint it does nothing, preserving the original stash.
            _stash_original_nominal(n)

            if resume_after is None:
                # Fresh start: mask everything beyond the first planning horizon.
                _mask_future_generators(n, periods_list[0])
            else:
                # Checkpoints are saved in the *unmasked* state (all future generators
                # restored to their original values). Re-apply the mask for the next
                # period that will actually be solved so the optimizer doesn't see
                # generators it shouldn't yet.
                resume_idx = periods_list.index(resume_after)
                next_after_resume = periods_list[resume_idx + 1] if resume_idx + 1 < len(periods_list) else None
                if next_after_resume is not None:
                    _mask_future_generators(n, next_after_resume)

            for i, planning_horizon in enumerate(periods_list):
                if resume_after is not None and planning_horizon <= resume_after:
                    logger.info(f"Skipping {planning_horizon} — already solved in checkpoint")
                    continue

                sns_horizon = n.snapshots[n.snapshots.get_level_values(0) == planning_horizon]
                kwargs["snapshots"] = sns_horizon

                if "TCT" in opts:
                    apply_forced_retirements(n, planning_horizon, config)

                apply_zero_emission_retirements(n, planning_horizon, config)
                update_egs_p_nom_max(n, planning_horizon)
                run_optimize(n, rolling_horizon, skip_iterations, cf_solving, **kwargs)

                logger.info(f"Preparing brownfield from {planning_horizon}")
                freeze_prior_periods(n, planning_horizon)

                # Save checkpoint after each completed period so the next period can
                # resume without re-solving earlier ones.  Checkpoints are saved in
                # the *unmasked* state: call _restore_future_generators to undo the
                # re-mask that freeze_prior_periods just applied, export, then re-mask
                # so the in-memory network is correct for the next loop iteration.
                period_idx = periods_list.index(planning_horizon)
                next_period = periods_list[period_idx + 1] if period_idx + 1 < len(periods_list) else None
                if next_period is not None:
                    _restore_future_generators(n)  # unmask for clean checkpoint
                    cp = _checkpoint_path(planning_horizon)
                    logger.info(f"Saving checkpoint to {cp}")
                    n.export_to_netcdf(str(cp))
                    _mask_future_generators(n, next_period)  # re-mask for next iteration

            # Remove checkpoints now that the full run succeeded.
            for ph in periods_list:
                cp = _checkpoint_path(ph)
                if cp.exists():
                    cp.unlink()
                    logger.info(f"Deleted checkpoint {cp}")

            # Restore the pre-solve p_nom/e_nom baseline so downstream statistics
            # and plots can compute (p_nom_opt - p_nom) = what was actually built
            # across the myopic horizons.
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
