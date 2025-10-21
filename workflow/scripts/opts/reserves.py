"""
Energy Reserve Margin (ERM) and Planning Reserve Margin (PRM) constraints for PyPSA-USA.

This module contains functions for implementing capacity adequacy constraints,
including energy reserve margins (ERM) and planning reserve margins (PRM).
"""

import logging

import linopy
import numpy as np
import pandas as pd
import pypsa
from pypsa.descriptors import (
    expand_series,
    get_activity_mask,
    get_bounds_pu,
    nominal_attrs,
)
from pypsa.descriptors import (
    get_switchable_as_dense as get_as_dense,
)
from pypsa.optimization.common import reindex
from xarray import DataArray, concat

logger = logging.getLogger(__name__)


def define_SU_reserve_constraints(n):
    """Sets energy balance constraints for storage units."""
    sns = n.snapshots
    m = n.model
    c = "StorageUnit"
    dim = "snapshot"
    assets = n.df(c)
    active = DataArray(get_activity_mask(n, c, sns))

    if assets.empty:
        return

    # elapsed hours
    eh = expand_series(n.snapshot_weightings.stores[sns], assets.index)
    # efficiencies
    eff_stand = (1 - get_as_dense(n, c, "standing_loss", sns)).pow(eh)
    eff_dispatch = get_as_dense(n, c, "efficiency_dispatch", sns)
    eff_store = get_as_dense(n, c, "efficiency_store", sns)

    soc = m[f"{c}-state_of_charge_RESERVES"]

    lhs = [
        (-1, soc),
        (-1 / eff_dispatch * eh, m[f"{c}-p_dispatch_RESERVES"]),
        (eff_store * eh, m[f"{c}-p_store_RESERVES"]),
    ]

    # We create a mask `include_previous_soc` which excludes the first snapshot
    # for non-cyclic assets.
    noncyclic_b = ~assets.cyclic_state_of_charge.to_xarray()
    include_previous_soc = (active.cumsum(dim) != 1).where(noncyclic_b, True)

    previous_soc = soc.where(active).ffill(dim).roll(snapshot=1).ffill(dim).where(include_previous_soc)

    # We add inflow and initial soc for noncyclic assets to rhs
    soc_init = assets.state_of_charge_initial.to_xarray()
    rhs = DataArray(-get_as_dense(n, c, "inflow", sns).mul(eh))

    if isinstance(sns, pd.MultiIndex):
        # If multi-horizon optimizing, we update the previous_soc and the rhs
        # for all assets which are cyclid/non-cyclid per period.
        periods = soc.coords["period"]
        per_period = (
            assets.cyclic_state_of_charge_per_period.to_xarray() | assets.state_of_charge_initial_per_period.to_xarray()
        )

        # We calculate the previous soc per period while cycling within a period
        # Normally, we should use groupby, but is broken for multi-index
        # see https://github.com/pydata/xarray/issues/6836
        ps = sns.unique("period")
        sl = slice(None)
        previous_soc_pp_list = [soc.data.sel(snapshot=(p, sl)).roll(snapshot=1) for p in ps]
        previous_soc_pp = concat(previous_soc_pp_list, dim="snapshot")

        # We create a mask `include_previous_soc_pp` which excludes the first
        # snapshot of each period for non-cyclic assets.
        include_previous_soc_pp = active & (periods == periods.shift(snapshot=1))
        include_previous_soc_pp = include_previous_soc_pp.where(noncyclic_b, True)
        # We take values still to handle internal xarray multi-index difficulties
        previous_soc_pp = previous_soc_pp.where(
            include_previous_soc_pp.values,
            linopy.variables.FILL_VALUE,
        )

        # update the previous_soc variables and right hand side
        previous_soc = previous_soc.where(~per_period, previous_soc_pp)
        include_previous_soc = include_previous_soc_pp.where(
            per_period,
            include_previous_soc,
        )
    lhs += [(eff_stand, previous_soc)]
    rhs = rhs.where(include_previous_soc, rhs - soc_init)
    m.add_constraints(lhs, "=", rhs, name=f"{c}-energy_balance_RESERVES", mask=active)


def define_operational_constraints_for_extendables(
    n: pypsa.Network,
    sns: pd.Index,
    c: str,
    attr: str,
    transmission_losses: int,
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

    ext_i = n.get_extendable_i(c)

    if ext_i.empty:
        return

    min_pu, max_pu = map(DataArray, get_bounds_pu(n, c, sns, ext_i, attr))

    dispatch = reindex(n.model[f"{c}-{attr}_RESERVES"], c, ext_i)
    capacity = n.model[f"{c}-{nominal_attrs[c]}"]

    active = get_activity_mask(n, c, sns, ext_i)

    lhs_lower = (1, dispatch), (-min_pu, capacity)
    lhs_upper = (1, dispatch), (-max_pu, capacity)

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
    transmission_losses: int,
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

    fix_i = n.get_non_extendable_i(c)
    fix_i = fix_i.difference(n.get_committable_i(c)).rename(fix_i.name)

    if fix_i.empty:
        return

    nominal_fix = n.df(c)[nominal_attrs[c]].reindex(fix_i)
    min_pu, max_pu = get_bounds_pu(n, c, sns, fix_i, attr)
    lower = min_pu.mul(nominal_fix)
    upper = max_pu.mul(nominal_fix)

    active = get_activity_mask(n, c, sns, fix_i)

    dispatch_lower = reindex(n.model[f"{c}-{attr}_RESERVES"], c, fix_i)
    dispatch_upper = reindex(n.model[f"{c}-{attr}_RESERVES"], c, fix_i)

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
        (-get_as_dense(n, "Load", "p_set", n.snapshots) * n.loads.sign)
        .T.groupby(n.loads.bus)
        .sum()
        .T.reindex(columns=region_buses.index, fill_value=0)
    )

    return rhs


def _calculate_capacity_accredidation(n, planning_horizon, region_buses, specific_hour=None):
    """Calculate capacity accreditation for extendable and non-extendable generators."""
    # Get active generators during this planning period
    active_gens = n.get_active_assets("Generator", planning_horizon)
    extendable_gens = n.generators.p_nom_extendable
    region_gens = n.generators.bus.isin(region_buses.index)

    # Extendable capacity with capacity credit
    region_active_ext_gens = region_gens & active_gens & extendable_gens
    region_active_ext_gens = n.generators[region_active_ext_gens]

    if not region_active_ext_gens.empty:
        ext_p_nom = n.model["Generator-p_nom"].loc[region_active_ext_gens.index]

        ext_p_max_pu = get_as_dense(n, "Generator", "p_max_pu", inds=region_active_ext_gens.index)
        ext_p_max_pu = ext_p_max_pu.loc[planning_horizon]
        ext_p_max_pu.T.index.name = "Generator-ext"

        ext_contribution = ext_p_nom * ext_p_max_pu
    else:
        ext_contribution = 0

    # Non-extendable existing capacity which contributes to the reserve margin
    region_active_nonext_gens = region_gens & active_gens & ~extendable_gens
    region_active_nonext_gens = n.generators[region_active_nonext_gens]

    if not region_active_nonext_gens.empty:
        non_ext_p_max_pu = get_as_dense(n, "Generator", "p_max_pu", inds=region_active_nonext_gens.index)
        non_ext_p_max_pu = non_ext_p_max_pu.loc[planning_horizon]

        non_ext_p_nom = region_active_nonext_gens.p_nom
        non_ext_contribution = non_ext_p_nom * non_ext_p_max_pu
    else:
        non_ext_contribution = 0
    if specific_hour is not None:
        ext_contribution = ext_contribution.loc[specific_hour]
        non_ext_contribution = non_ext_contribution.loc[specific_hour]

    return ext_contribution, non_ext_contribution


def _get_combined_prm_requirements(n, config=None, snakemake=None, regional_prm_data=None):
    """
    Combine PRM requirements from different sources into a single dataframe.

    Parameters
    ----------
    n : pypsa.Network
    config : dict, optional
        If provided, will read PRM requirements from config files
    regional_prm_data : pd.DataFrame, optional
        Direct input of PRM requirements with columns: name, region, prm, planning_horizon

    Returns
    -------
    pd.DataFrame
        Combined PRM requirements with columns: name, region, prm, planning_horizon
    """
    if regional_prm_data is not None:
        return regional_prm_data

    # Load user-defined PRM requirements
    regional_prm = pd.read_csv(
        config["electricity"]["SAFE_regional_reservemargins"],
        index_col=[0],
    )

    # Process ReEDS PRM data if available
    reeds_prm = pd.read_csv(snakemake.input.safer_reeds, index_col=[0])

    # Map NERC regions to ReEDS zones
    nerc_memberships = (
        n.buses.groupby("nerc_reg")["reeds_zone"]
        .apply(
            lambda x: ", ".join(x),
        )
        .to_dict()
    )

    reeds_prm["region"] = reeds_prm.index.map(nerc_memberships)
    reeds_prm = reeds_prm.dropna(subset="region")
    reeds_prm = reeds_prm.drop(
        columns=["none", "ramp2025_20by50", "ramp2025_25by50", "ramp2025_30by50"],
    )
    reeds_prm = reeds_prm.rename(columns={"static": "prm", "t": "planning_horizon"})

    # Combine both data sources
    regional_prm = pd.concat([regional_prm, reeds_prm])

    # Filter for relevant planning horizons
    return regional_prm[regional_prm.planning_horizon.isin(n.investment_periods)]


def define_erm_nodal_balance_constraints(n, erm, region_buses):
    """
    Define ERM nodal balance constraints for a given region and planning horizon.

    This function refactors the original implementation to leverage PyPSA's
    define_nodal_balance_constraints function while maintaining ERM-specific
    functionality for reserve margin calculations.

    Parameters
    ----------
    n : pypsa.Network
    erm : pd.Series
    region_buses : pd.Series
        Series containing buses in the region
    """
    # Get capacity contribution from resources
    lhs_capacity, rhs_existing = _calculate_capacity_accredidation(
        n,
        erm.planning_horizon,
        region_buses,
    )

    # Calculate peak demand and required reserve margin for a bus
    regional_demand = _get_regional_demand(n, region_buses)
    planning_reserve = regional_demand * (1.0 + erm.prm)

    _define_unified_nodal_balance_constraints(
        n,
        erm,
        region_buses.index,
        planning_reserve,
        n.snapshots,
        "",
        rhs_existing,
    )


def _define_unified_nodal_balance_constraints(
    n,
    erm,
    buses,
    planning_reserve,
    sns,
    suffix,
    rhs_existing,
) -> None:
    from linopy import merge
    from xarray import DataArray

    m = n.model
    if buses is None:
        buses = n.buses.index

    args = [
        ["StorageUnit", "p_dispatch_RESERVES", "bus", 1],
        ["StorageUnit", "p_store_RESERVES", "bus", -1],
        ["Line", "s_RESERVES", "bus0", -1],
        ["Line", "s_RESERVES", "bus1", 1],
        ["Link", "p_RESERVES", "bus0", -1],
        ["Link", "p_RESERVES", "bus1", get_as_dense(n, "Link", "efficiency", sns)],
    ]

    exprs = []
    for arg in args:
        c, attr, column, sign = arg

        if n.df(c).empty:
            continue

        if "sign" in n.df(c):
            # additional sign necessary for branches in reverse direction
            sign = sign * n.df(c).sign

        expr = DataArray(sign) * m[f"{c}-{attr}"]
        cbuses = n.df(c)[column][lambda ds: ds.isin(buses)].rename("Bus")

        expr = expr.sel({c: cbuses.index})

        if expr.size:
            exprs.append(expr.groupby(cbuses).sum())

    lhs = merge(exprs, join="outer").reindex(Bus=buses)

    rhs = planning_reserve
    rhs.index.name = "snapshot"

    # # Calculate RHS for each bus (planning reserve minus existing capacity)
    bus_gens_non_ext = n.generators[
        (n.generators.bus.isin(buses)) & ~n.generators.p_nom_extendable & get_activity_mask(n, "Generator", sns).all()
    ]
    if not bus_gens_non_ext.empty:
        rhs_existing.index = sns
        bus_rhs_capacity = rhs_existing.loc[sns, bus_gens_non_ext.index]
        bus_rhs_capacity = bus_rhs_capacity.T.groupby(bus_gens_non_ext.bus).sum().T.reindex(columns=buses, fill_value=0)
        rhs = rhs - bus_rhs_capacity

    empty_nodal_balance = (lhs.vars == -1).all("_term")
    rhs = DataArray(rhs)
    if empty_nodal_balance.any():
        if (empty_nodal_balance & (rhs != 0)).any().item():
            raise ValueError("Empty LHS with non-zero RHS in nodal balance constraint.")

        mask = ~empty_nodal_balance
    else:
        mask = None

    if suffix:
        lhs = lhs.rename(Bus=f"Bus{suffix}")
        rhs = rhs.rename(Bus=f"Bus{suffix}")
        if mask is not None:
            mask = mask.rename(Bus=f"Bus{suffix}")
    n.model.add_constraints(
        lhs,
        "=",
        rhs,
        name=f"GlobalConstraint-{erm.name}_ERM",
        mask=mask,
    )


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
    summed_reserve = reserve.sum("Generator")

    # Share of extendable renewable capacities
    ext_i = n.generators.query("p_nom_extendable").index
    vres_i = n.generators_t.p_max_pu.columns
    if not ext_i.empty and not vres_i.empty:
        capacity_factor = n.generators_t.p_max_pu[vres_i.intersection(ext_i)]
        p_nom_vres = n.model["Generator-p_nom"].loc[vres_i.intersection(ext_i)].rename({"Generator-ext": "Generator"})
        lhs = summed_reserve + (p_nom_vres * (-eps_vres * capacity_factor)).sum(
            "Generator",
        )
    else:  # if no extendable VRES
        lhs = summed_reserve

    # Total demand per t
    demand = get_as_dense(n, "Load", "p_set").sum(axis=1)

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

    p_max_pu = get_as_dense(n, "Generator", "p_max_pu")

    if not ext_i.empty:
        capacity_variable = n.model["Generator-p_nom"].rename(
            {"Generator-ext": "Generator"},
        )
        lhs = dispatch + reserve - capacity_variable * p_max_pu[ext_i]
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
    model = n.model
    duals = model.dual
    logger.info("Storing ERM data from optimization results")
    # Check if ERM constraints are activated by looking for the ERM reserve variables
    if "StorageUnit-p_dispatch_RESERVES" in model.variables:
        logger.info("Storing ERM data from optimization results")

        # Get the reserve dispatch for storage units
        if "StorageUnit-p_dispatch_RESERVES" in model.solution:
            n.storage_units_t["p_dispatch_reserves"] = model.solution["StorageUnit-p_dispatch_RESERVES"].to_pandas()

        # Get the reserve storage for storage units
        if "StorageUnit-p_store_RESERVES" in model.solution:
            n.storage_units_t["p_store_reserves"] = model.solution["StorageUnit-p_store_RESERVES"].to_pandas()

        # Get the state of charge for reserve operation
        if "StorageUnit-state_of_charge_RESERVES" in model.solution:
            n.storage_units_t["state_of_charge_reserves"] = model.solution[
                "StorageUnit-state_of_charge_RESERVES"
            ].to_pandas()

        # Get the line flow reserves
        if "Line-s_RESERVES" in model.solution:
            n.lines_t["s_reserves"] = model.solution["Line-s_RESERVES"].to_pandas()

        if "Link-p_RESERVES" in model.solution:
            n.links_t["p_reserves"] = model.solution["Link-p_RESERVES"].to_pandas()

        # Calculate and store the ERM price (shadow price of the ERM constraint)
        erm_constraints = [c for c in model.constraints if "ERM_hr" in c]
        if erm_constraints:
            # Get the dual values (shadow prices) of ERM constraints
            # For xarray Dataset, we need to use dictionary-based indexing
            erm_prices = {}
            for constraint in erm_constraints:
                # Extract bus name from constraint name (format: GlobalConstraint-{name}_{horizon}_ERM_hr{hour}_bus{bus})
                parts = constraint.split("_")
                bus_part = parts[-1]
                bus = bus_part.replace("bus", "")

                # Store the dual value
                try:
                    dual_value = duals[constraint].item()
                    if bus not in erm_prices:
                        erm_prices[bus] = [dual_value]
                    else:
                        erm_prices[bus].append(dual_value)
                except (KeyError, ValueError):
                    # Skip constraints without dual values
                    continue

            # Create a Series with bus index - averaging values for each bus
            if erm_prices:
                # Calculate average price for each bus
                for bus in erm_prices:
                    erm_prices[bus] = sum(erm_prices[bus]) / len(erm_prices[bus])

                erm_price_series = pd.Series(erm_prices)

                # Store in network
                if "ERM_price" not in n.buses:
                    n.buses["ERM_price"] = 0.0  # Initialize as float
                n.buses.loc[erm_price_series.index, "ERM_price"] = erm_price_series
