"""
Energy Reserve Margin (ERM) constraints for PyPSA-USA.

This module contains functions for implementing capacity adequacy constraints,
including energy reserve margins (ERM).
"""

import logging

import numpy as np
import pandas as pd
import pypsa
from linopy import merge
from opts._helpers import get_region_buses
from pypsa.descriptors import nominal_attrs
from xarray import DataArray

logger = logging.getLogger(__name__)


def _get_zero_emission_periods(n, config):
    """Return the set of planning horizons where the national (all-region) CO2 limit is 0.

    Used to exclude emitting generators from ERM capacity credit in periods
    where fossil dispatch is completely prohibited.
    """
    try:
        co2_lims = pd.read_csv(config["electricity"]["regional_Co2_limits"])
    except (KeyError, FileNotFoundError, TypeError):
        return set()
    mask = (co2_lims["regions"].str.strip() == "all") & (co2_lims["limit"] == 0)
    return set(co2_lims.loc[mask, "planning_horizon"])


def define_SU_reserve_constraints(n, sns):
    """Energy-balance constraints for the StorageUnit RESERVES shadow variables.

    Mirrors pypsa v1.3's internal ``define_storage_unit_constraints`` (xarray
    model-space via ``n.optimize._window`` and ``c.da``) so that MultiIndex
    snapshots align under xarray >= 2026; only the variable names differ
    (``*_RESERVES``) and there is no spill term in the shadow system.
    """
    m = n.model
    component = "StorageUnit"
    dim = "snapshot"
    window = n.optimize._window.subset(sns)
    c_obj = n.components[component]

    if c_obj.static.empty:
        return

    active = c_obj.da.active.sel(snapshot=sns, name=c_obj.active_assets)

    eh = window.snapshot_weightings("stores")

    eff_stand = (1 - c_obj.da.standing_loss.sel(snapshot=sns, name=c_obj.active_assets)) ** eh
    eff_dispatch = c_obj.da.efficiency_dispatch.sel(snapshot=sns, name=c_obj.active_assets)
    eff_store = c_obj.da.efficiency_store.sel(snapshot=sns, name=c_obj.active_assets)

    soc = m[f"{component}-state_of_charge_RESERVES"]

    lhs = [
        (-1, soc),
        (-1 / eff_dispatch * eh, m[f"{component}-p_dispatch_RESERVES"]),
        (eff_store * eh, m[f"{component}-p_store_RESERVES"]),
    ]

    # mask `include_previous_soc` excludes the first snapshot for non-cyclic assets
    noncyclic_b = ~c_obj.da.cyclic_state_of_charge.sel(name=c_obj.active_assets)
    include_previous_soc = (active.cumsum(dim) != 1).where(noncyclic_b, True)

    previous_soc = soc.where(active).ffill(dim).roll(snapshot=1).ffill(dim)

    # inflow and initial soc for noncyclic assets go to rhs
    soc_init = c_obj.da.state_of_charge_initial.sel(name=c_obj.active_assets)
    rhs = -c_obj.da.inflow.sel(snapshot=sns, name=c_obj.active_assets) * eh

    if n._multi_invest:
        # per-period cycling / initial-value reset, exactly as pypsa v1 does it
        per_period = c_obj.da.cyclic_state_of_charge_per_period.sel(
            name=c_obj.active_assets,
        ) | c_obj.da.state_of_charge_initial_per_period.sel(name=c_obj.active_assets)

        previous_soc_pp = window.roll_within_periods(soc)

        within_period = ~window.period_start_mask()
        include_previous_soc_pp = active & (
            within_period | c_obj.da.cyclic_state_of_charge_per_period.sel(name=c_obj.active_assets)
        )

        # the per-period inclusion is carried by the `include_previous_soc`
        # coefficient (not by masking `previous_soc_pp` to NaN, which v1 reads
        # as an absent term and would drop the period-start energy-balance row)
        previous_soc = previous_soc.where(~per_period, previous_soc_pp)
        include_previous_soc = include_previous_soc_pp.where(
            per_period,
            include_previous_soc,
        )

    lhs += [(eff_stand * include_previous_soc, previous_soc)]

    lhs = m.linexpr(*lhs)
    rhs = rhs.where(include_previous_soc, rhs - soc_init)

    m.add_constraints(lhs, "=", rhs, name=f"{component}-energy_balance_RESERVES", mask=active)


def define_operational_constraints_for_extendables(
    n: pypsa.Network,
    sns: pd.Index,
    c: str,
    attr: str,
) -> None:
    """
    Sets power dispatch constraints for extendable devices for a given
    component and a given attribute.

    Parameters
    ----------
    n : pypsa.Network
    sns : pd.Index
        Snapshots of the constraint.
    c : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'
    """
    lhs_lower: DataArray | tuple
    lhs_upper: DataArray | tuple

    c_obj = n.components[c]
    ext_i = c_obj.extendables

    if ext_i.empty:
        return
    if isinstance(ext_i, pd.MultiIndex):
        ext_i = ext_i.unique(level="name")

    min_pu, max_pu = c_obj.get_bounds_pu(attr=attr)
    min_pu = min_pu.sel(name=ext_i)
    max_pu = max_pu.sel(name=ext_i)
    if "snapshot" in min_pu.dims:
        min_pu = min_pu.sel(snapshot=sns)
        max_pu = max_pu.sel(snapshot=sns)

    dispatch = n.model[f"{c}-{attr}_RESERVES"].sel(name=ext_i)
    capacity = n.model[f"{c}-{nominal_attrs[c]}"].sel(name=ext_i)

    active = c_obj.da.active.sel(name=ext_i, snapshot=sns)

    lhs_lower = dispatch - min_pu * capacity
    lhs_upper = dispatch - max_pu * capacity

    n.model.add_constraints(
        lhs_lower,
        ">=",
        0,
        name=f"{c}-ext-{attr}-lower_RESERVES",
        mask=active,
    )
    n.model.add_constraints(
        lhs_upper,
        "<=",
        0,
        name=f"{c}-ext-{attr}-upper_RESERVES",
        mask=active,
    )


def define_operational_constraints_for_non_extendables(
    n: pypsa.Network,
    sns: pd.Index,
    c: str,
    attr: str,
) -> None:
    """
    Sets power dispatch constraints for non-extendable and non-commitable
    assets for a given component and a given attribute.

    Parameters
    ----------
    n : pypsa.Network
    sns : pd.Index
        Snapshots of the constraint.
    c : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'
    """
    dispatch_lower: DataArray | tuple
    dispatch_upper: DataArray | tuple

    c_obj = n.components[c]
    fix_i = c_obj.fixed.difference(c_obj.committables)

    if fix_i.empty:
        return

    nominal_fix = c_obj.da[nominal_attrs[c]].sel(name=fix_i)
    min_pu, max_pu = c_obj.get_bounds_pu(attr=attr)
    min_pu = min_pu.sel(name=fix_i)
    max_pu = max_pu.sel(name=fix_i)
    if "snapshot" in min_pu.dims:
        min_pu = min_pu.sel(snapshot=sns)
        max_pu = max_pu.sel(snapshot=sns)

    lower = min_pu * nominal_fix
    upper = max_pu * nominal_fix

    active = c_obj.da.active.sel(name=fix_i, snapshot=sns)

    dispatch_lower = n.model[f"{c}-{attr}_RESERVES"].sel(name=fix_i)
    dispatch_upper = n.model[f"{c}-{attr}_RESERVES"].sel(name=fix_i)

    n.model.add_constraints(
        dispatch_lower,
        ">=",
        lower,
        name=f"{c}-fix-{attr}-lower_RESERVES",
        mask=active,
    )
    n.model.add_constraints(
        dispatch_upper,
        "<=",
        upper,
        name=f"{c}-fix-{attr}-upper_RESERVES",
        mask=active,
    )


def _get_regional_demand(n, region_buses):
    """
    Calculate hourly demand for a specific region.

    Parameters
    ----------
    n : pypsa.Network
    region_buses : pd.DataFrame
        DataFrame containing buses in the region

    Returns
    -------
    pd.Series
        Hourly demand series for the region
    """
    rhs = (
        (-n.get_switchable_as_dense("Load", "p_set", n.snapshots) * n.loads.sign)
        .T.groupby(n.loads.bus)
        .sum()
        .T.reindex(columns=region_buses.index, fill_value=0)
    )

    return rhs


def define_erm_nodal_balance_constraints(
    n,
    snapshots,
    erm,
    region_name,
    region_buses,
    zero_emission_periods=None,
    emitting_carriers=None,
):
    """
    Define ERM nodal balance constraints for a given region across all investment periods.

    Creates a single constraint per region that spans all snapshots (including all
    investment periods). Uses activity masking to zero out generator contributions
    in periods when they are inactive (e.g., retired or not yet built).

    Emitting generators are excluded from the ERM capacity credit in any planning
    horizon whose national CO2 limit is zero, since they cannot legally dispatch
    and should not count as firm capacity.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : pd.Index
        Snapshots of the constraint.
    erm : float
        Energy reserve margin as a fraction (e.g., 0.15 for 15%)
    region_name : str
        Name of the region for constraint naming
    region_buses : pd.DataFrame
        DataFrame containing buses in the region
    zero_emission_periods : set, optional
        Planning horizons (int years) where the national CO2 limit is 0.
        Emitting generators receive no ERM capacity credit in these periods.
    emitting_carriers : set, optional
        Carrier names with co2_emissions > 0. Required if zero_emission_periods is set.
    """
    sns = snapshots
    m = n.model
    buses = region_buses.index

    # RHS: demand * (1 + erm) over ALL snapshots
    regional_demand = _get_regional_demand(n, region_buses).loc[sns]
    planning_reserve = regional_demand * (1.0 + erm)

    # Build a per-snapshot boolean mask: True = snapshot is in a zero-emission period.
    # Emitting generators will be excluded from ERM capacity credit for these snapshots.
    if zero_emission_periods and emitting_carriers:
        period_per_snap = sns.get_level_values(0) if isinstance(sns, pd.MultiIndex) else sns
        snap_is_zero = pd.Series(
            [p in zero_emission_periods for p in period_per_snap],
            index=sns,
            dtype=bool,
        )
    else:
        snap_is_zero = pd.Series(False, index=sns, dtype=bool)

    def _activity_da(component):
        mask = n.components[component].get_activity_mask(sns)
        mask.index.name = "snapshot"
        return DataArray(mask)

    # LHS expressions for storage/transmission with activity masking
    su_activity = _activity_da("StorageUnit") if not n.storage_units.empty else None
    line_activity = _activity_da("Line") if not n.lines.empty else None
    link_activity = _activity_da("Link") if not n.links.empty else None

    link_efficiency = n.get_switchable_as_dense("Link", "efficiency", sns)
    link_efficiency.index.name = "snapshot"

    args = [
        ["StorageUnit", "p_dispatch_RESERVES", "bus", 1, su_activity],
        ["StorageUnit", "p_store_RESERVES", "bus", -1, su_activity],
        ["Line", "s_RESERVES", "bus0", -1, line_activity],
        ["Line", "s_RESERVES", "bus1", 1, line_activity],
        ["Link", "p_RESERVES", "bus0", -1, link_activity],
        ["Link", "p_RESERVES", "bus1", link_efficiency, link_activity],
    ]

    exprs = []

    for c, attr, column, sign, activity in args:
        if n.components[c].static.empty:
            continue

        if "sign" in n.components[c].static:
            sign = sign * n.components[c].static.sign

        expr = DataArray(sign) * m[f"{c}-{attr}"]
        df = n.components[c].static
        # For components with both bus0 and bus1, require both to be in buses
        if "bus0" in df.columns and "bus1" in df.columns:
            mask = df["bus0"].isin(buses) & df["bus1"].isin(buses)
            cbuses = df.loc[mask, column].rename("Bus")
        else:
            cbuses = df[column][lambda ds: ds.isin(buses)].rename("Bus")

        expr = expr.sel(name=cbuses.index)

        if expr.size:
            if activity is not None:
                expr = expr.where(activity.sel(name=cbuses.index))
            exprs.append(expr.groupby(cbuses).sum())

    # Extendable generators on LHS: p_nom * p_max_pu * activity_mask
    region_gens = n.generators.bus.isin(buses)
    extendable_gens = n.generators.p_nom_extendable
    region_ext_gens = n.generators[region_gens & extendable_gens]

    if not region_ext_gens.empty:
        ext_p_nom = m["Generator-p_nom"].loc[region_ext_gens.index]
        ext_p_max_pu = n.get_switchable_as_dense("Generator", "p_max_pu", sns, inds=region_ext_gens.index)

        ext_p_max_pu.index.name = "snapshot"
        ext_p_max_pu.columns.name = "name"
        # wrap in DataArray so linopy keeps the flat 'snapshot' dim of the
        # MultiIndex frames instead of unstacking to period x timestep
        ext_contribution = ext_p_nom * DataArray(ext_p_max_pu)

        # Use .where() to remove terms for inactive periods (sets var labels to -1)
        # rather than zeroing coefficients, which leaves orphaned variable references
        activity = n.components["Generator"].get_activity_mask(sns)[region_ext_gens.index]
        activity.index.name = "snapshot"
        activity.columns.name = "name"

        # Exclude emitting generators from ERM credit in zero-emission periods
        if snap_is_zero.any() and emitting_carriers:
            fossil_cols = region_ext_gens.index[region_ext_gens.carrier.isin(emitting_carriers)]
            if not fossil_cols.empty:
                activity.loc[snap_is_zero, fossil_cols] = False
                # pandas .loc boolean assignment can silently reset index/column names
                activity.index.name = "snapshot"
                activity.columns.name = "name"
                logger.debug(
                    f"Excluded {len(fossil_cols)} emitting extendable generators from ERM "
                    f"in zero-emission snapshots for region {region_name}.",
                )

        ext_contribution = ext_contribution.where(DataArray(activity))

        gen_buses = DataArray(
            region_ext_gens.bus.values,
            dims=["name"],
            coords={"name": region_ext_gens.index.values},
            name="Bus",
        )
        exprs.append(ext_contribution.groupby(gen_buses).sum())

    lhs = merge(exprs, join="outer").reindex(Bus=buses)

    # Non-extendable generators on RHS: p_nom * p_max_pu * activity_mask
    region_nonext_gens = n.generators[region_gens & ~extendable_gens]
    if not region_nonext_gens.empty:
        nonext_activity = n.components["Generator"].get_activity_mask(sns)[region_nonext_gens.index]
        nonext_activity.index.name = "snapshot"

        # Exclude emitting generators from ERM credit in zero-emission periods
        if snap_is_zero.any() and emitting_carriers:
            fossil_nonext_cols = region_nonext_gens.index[region_nonext_gens.carrier.isin(emitting_carriers)]
            if not fossil_nonext_cols.empty:
                nonext_activity.loc[snap_is_zero, fossil_nonext_cols] = False
                logger.debug(
                    f"Excluded {len(fossil_nonext_cols)} emitting non-extendable generators "
                    f"from ERM credit in zero-emission snapshots for region {region_name}.",
                )

        nonext_p_max_pu = n.get_switchable_as_dense("Generator", "p_max_pu", sns, inds=region_nonext_gens.index)
        nonext_p_max_pu.index.name = "snapshot"
        nonext_p_max_pu = nonext_p_max_pu * nonext_activity
        rhs_existing = region_nonext_gens.p_nom * nonext_p_max_pu
        rhs_existing.index = sns
        bus_rhs_capacity = rhs_existing.T.groupby(region_nonext_gens.bus).sum().T
        bus_rhs_capacity = bus_rhs_capacity.reindex(columns=buses, fill_value=0)
        planning_reserve = planning_reserve - bus_rhs_capacity

    rhs = planning_reserve
    rhs.index.name = "snapshot"
    # under pypsa v1 the bus index is named "name"; the lhs groupbys use "Bus"
    rhs.columns.name = "Bus"

    # Constraint over ALL snapshots
    empty_nodal_balance = (lhs.vars == -1).all("_term")
    rhs = DataArray(rhs)
    if empty_nodal_balance.any():
        if (empty_nodal_balance & (rhs != 0)).any().item():
            raise ValueError("Empty LHS with non-zero RHS in nodal balance constraint.")
        mask = ~empty_nodal_balance
    else:
        mask = None

    n.model.add_constraints(
        lhs,
        ">=",
        rhs,
        name=f"GlobalConstraint-{region_name}_ERM",
        mask=mask,
    )


def add_ERM_constraints(n, snapshots, config=None, snakemake=None, regional_erm_data=None):
    """
    Add Energy Reserve Margin (ERM) constraints for regional capacity adequacy.

    This function enforces that each region has sufficient firm capacity to meet
    peak demand plus a reserve margin. These resources must be "energy-backed" meaning
    resources like storage devices must have the state of charge to meet the reserve
    to contribute to the ERM.

    Creates one constraint per region spanning all investment periods, using activity
    masking to handle generator retirements and build years.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object
    config : dict, optional
        Configuration dictionary containing electricity.erm dict.
        Required if regional_erm_data not provided.
    snakemake : snakemake object, optional
        Not used in the new implementation, kept for API compatibility.
    regional_erm_data : dict, optional
        Direct input of ERM requirements as dict {region_name: erm_value}.
        If provided, this takes precedence over config data.
    """
    model = n.model

    # Get ERM data: dict {region_name: erm_value}
    # Default to 15% reserve margin for all regions if not specified
    default_erm = {"all": 0.15}

    if regional_erm_data is not None:
        erm_dict = regional_erm_data
    elif config is not None and config.get("electricity", {}).get("erm"):
        erm_dict = config["electricity"]["erm"]
    else:
        logger.info("No ERM configuration provided. Using default: {'all': 0.15}")
        erm_dict = default_erm

    # Identify planning horizons where fossil cannot count toward ERM (CO2 limit = 0)
    zero_emission_periods = _get_zero_emission_periods(n, config)
    emitting_carriers = set(n.carriers.index[n.carriers.co2_emissions.fillna(0) > 0])
    if zero_emission_periods:
        logger.info(
            f"Zero-emission periods detected {zero_emission_periods}. "
            f"Emitting carriers {emitting_carriers} will receive no ERM capacity credit "
            f"in those periods.",
        )

    for region_name, erm_value in erm_dict.items():
        region_list = [region_name.strip()]
        region_buses = get_region_buses(n, region_list)

        if region_buses.empty:
            continue

        logger.info(f"Adding ERM constraint for {region_name} with reserve level {erm_value}")

        # Create model variables to track storage contributions (only once)
        c = "StorageUnit"
        if not n.storage_units.empty and f"{c}-p_dispatch_RESERVES" not in model.variables:
            model.add_variables(
                -np.inf,
                model.variables["StorageUnit-p_dispatch"].upper,
                name=f"{c}-p_dispatch_RESERVES",
            )
            model.add_variables(
                -np.inf,
                model.variables["StorageUnit-p_store"].upper,
                name=f"{c}-p_store_RESERVES",
            )
            model.add_variables(
                -np.inf,
                model.variables["StorageUnit-state_of_charge"].upper,
                name=f"{c}-state_of_charge_RESERVES",
            )
            define_SU_reserve_constraints(n, snapshots)
            define_operational_constraints_for_extendables(n, snapshots, c, "state_of_charge")
            define_operational_constraints_for_extendables(n, snapshots, c, "p_dispatch")
            define_operational_constraints_for_extendables(n, snapshots, c, "p_store")
            define_operational_constraints_for_non_extendables(n, snapshots, c, "state_of_charge")
            define_operational_constraints_for_non_extendables(n, snapshots, c, "p_dispatch")
            define_operational_constraints_for_non_extendables(n, snapshots, c, "p_store")

        # Create model variables to track transmission contributions (only once)
        if not n.lines.empty and "Line-s_RESERVES" not in model.variables:
            model.add_variables(-np.inf, model.variables["Line-s"].upper, name="Line-s_RESERVES")
            define_operational_constraints_for_extendables(n, snapshots, "Line", "s")
            define_operational_constraints_for_non_extendables(n, snapshots, "Line", "s")

        if not n.links.empty and "Link-p_RESERVES" not in model.variables:
            model.add_variables(-np.inf, model.variables["Link-p"].upper, name="Link-p_RESERVES")
            define_operational_constraints_for_extendables(n, snapshots, "Link", "p")
            define_operational_constraints_for_non_extendables(n, snapshots, "Link", "p")

        define_erm_nodal_balance_constraints(
            n,
            snapshots,
            erm_value,
            region_name,
            region_buses,
            zero_emission_periods=zero_emission_periods,
            emitting_carriers=emitting_carriers,
        )
        logger.info(f"Added ERM constraint for {region_name}")


def add_operational_reserve_margin(n, sns, config):
    """
    Build reserve margin constraints based on the formulation given in
    https://genxproject.github.io/GenX/dev/core/#Reserves.

    Parameters
    ----------
        n : pypsa.Network
        sns: pd.DatetimeIndex
        config : dict

    Example:
    --------
    config.yaml requires to specify operational_reserve:
    operational_reserve: # like https://genxproject.github.io/GenX/dev/core/#Reserves
        activate: true
        epsilon_load: 0.02 # percentage of load at each snapshot
        epsilon_vres: 0.02 # percentage of VRES at each snapshot
        contingency: 400000 # MW
    """
    reserve_config = config["electricity"]["operational_reserve"]
    eps_load = reserve_config["epsilon_load"]
    eps_vres = reserve_config["epsilon_vres"]
    contingency = reserve_config["contingency"]

    # Reserve Variables
    n.model.add_variables(
        0,
        np.inf,
        coords=[sns, n.generators.index],
        name="Generator-r",
    )
    reserve = n.model["Generator-r"]
    summed_reserve = reserve.sum("name")

    # Share of extendable renewable capacities
    ext_i = n.generators.query("p_nom_extendable").index
    vres_i = n.generators_t.p_max_pu.columns
    if not ext_i.empty and not vres_i.empty:
        capacity_factor = n.generators_t.p_max_pu[vres_i.intersection(ext_i)]
        p_nom_vres = n.model["Generator-p_nom"].loc[vres_i.intersection(ext_i)]
        lhs = summed_reserve + (p_nom_vres * DataArray(-eps_vres * capacity_factor)).sum(
            "name",
        )
    else:  # if no extendable VRES
        lhs = summed_reserve

    # Total demand per t
    demand = n.get_switchable_as_dense("Load", "p_set").sum(axis=1)

    # VRES potential of non extendable generators
    capacity_factor = n.generators_t.p_max_pu[vres_i.difference(ext_i)]
    renewable_capacity = n.generators.p_nom[vres_i.difference(ext_i)]
    potential = (capacity_factor * renewable_capacity).sum(axis=1)

    # Right-hand-side
    rhs = eps_load * demand + eps_vres * potential + contingency

    n.model.add_constraints(lhs >= rhs, name="reserve_margin")

    # additional constraint that capacity is not exceeded
    gen_i = n.generators.index
    ext_i = n.generators.query("p_nom_extendable").index
    fix_i = n.generators.query("not p_nom_extendable").index

    dispatch = n.model["Generator-p"]
    reserve = n.model["Generator-r"]

    capacity_fixed = n.generators.p_nom[fix_i]

    p_max_pu = n.get_switchable_as_dense("Generator", "p_max_pu")

    if not ext_i.empty:
        capacity_variable = n.model["Generator-p_nom"]
        lhs = dispatch + reserve - capacity_variable * DataArray(p_max_pu[ext_i])
    else:
        lhs = dispatch + reserve

    rhs = (p_max_pu[fix_i] * capacity_fixed).reindex(columns=gen_i, fill_value=0)

    n.model.add_constraints(lhs <= rhs, name="Generator-p-reserve-upper")


def store_ERM_duals(n):
    """
    Store Energy Reserve Margin (ERM) data if ERM constraints are activated.

    This function checks if the model contains ERM-specific variables and if so,
    extracts and stores this data in the network object for later analysis.
    """
    logger.info("Storing ERM data from optimization results")
    model = n.model
    erm_constraints = [c for c in model.constraints if "ERM" in c]

    if erm_constraints:
        n.buses_t["erm_price"] = pd.DataFrame(index=n.snapshots, columns=n.buses.index)

        for constraint in erm_constraints:
            erm_dual = model.dual[constraint]
            # Store mean ERM price as time series for each bus
            # Automatically detect the ERM global constraint name
            global_constraint_columns = [col for col in erm_dual.to_dataframe().columns if col.endswith("_ERM")]

            if not global_constraint_columns:
                raise ValueError("No ERM global constraint dual found in model results.")
            erm_col = global_constraint_columns[0]
            erm_dual_df = (
                erm_dual.to_dataframe()[erm_col].reset_index().set_index(["period", "timestep"]).pivot(columns="Bus")
            )
            erm_dual_df.columns = erm_dual_df.columns.get_level_values(1)
            n.buses_t["erm_price"].update(erm_dual_df)

        # if "StorageUnit-p_dispatch_RESERVES" in model.solution:
        #     n.storage_units_t["p_dispatch_reserves"] = model.solution["StorageUnit-p_dispatch_RESERVES"].to_pandas()

        # # Get the reserve storage for storage units
        # if "StorageUnit-p_store_RESERVES" in model.solution:
        #     n.storage_units_t["p_store_reserves"] = model.solution["StorageUnit-p_store_RESERVES"].to_pandas()

        # # Get the state of charge for reserve operation
        # if "StorageUnit-state_of_charge_RESERVES" in model.solution:
        #     n.storage_units_t["state_of_charge_reserves"] = model.solution[
        #         "StorageUnit-state_of_charge_RESERVES"
        #     ].to_pandas()

        # # Get the line flow reserves
        # if "Line-s_RESERVES" in model.solution:
        #     n.lines_t["s_reserves"] = model.solution["Line-s_RESERVES"].to_pandas()

        # if "Link-p_RESERVES" in model.solution:
        #     n.links_t["p_reserves"] = model.solution["Link-p_RESERVES"].to_pandas()
