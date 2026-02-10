"""
Energy Reserve Margin (ERM) constraints for PyPSA-USA.

This module contains functions for implementing capacity adequacy constraints,
including energy reserve margins (ERM).
"""

import logging

import linopy
import numpy as np
import pandas as pd
import pypsa
from linopy import merge
from opts._helpers import get_region_buses
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


def define_SU_reserve_constraints(n, sns):
    """Sets energy balance constraints for storage units."""
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


def define_erm_nodal_balance_constraints(n, snapshots, erm, region_name, region_buses):
    """
    Define ERM nodal balance constraints for a given region across all investment periods.

    Creates a single constraint per region that spans all snapshots (including all
    investment periods). Uses activity masking to zero out generator contributions
    in periods when they are inactive (e.g., retired or not yet built).

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
    """
    sns = snapshots
    m = n.model
    buses = region_buses.index

    # RHS: demand * (1 + erm) over ALL snapshots
    regional_demand = _get_regional_demand(n, region_buses).loc[sns]
    planning_reserve = regional_demand * (1.0 + erm)

    # LHS expressions for storage/transmission with activity masking
    su_activity = DataArray(get_activity_mask(n, "StorageUnit", sns)) if not n.storage_units.empty else None
    line_activity = DataArray(get_activity_mask(n, "Line", sns)) if not n.lines.empty else None
    link_activity = DataArray(get_activity_mask(n, "Link", sns)) if not n.links.empty else None

    args = [
        ["StorageUnit", "p_dispatch_RESERVES", "bus", 1, su_activity],
        ["StorageUnit", "p_store_RESERVES", "bus", -1, su_activity],
        ["Line", "s_RESERVES", "bus0", -1, line_activity],
        ["Line", "s_RESERVES", "bus1", 1, line_activity],
        ["Link", "p_RESERVES", "bus0", -1, link_activity],
        ["Link", "p_RESERVES", "bus1", get_as_dense(n, "Link", "efficiency", sns), link_activity],
    ]

    exprs = []
    for c, attr, column, sign, activity in args:
        if n.df(c).empty:
            continue

        if "sign" in n.df(c):
            sign = sign * n.df(c).sign

        expr = DataArray(sign) * m[f"{c}-{attr}"]
        cbuses = n.df(c)[column][lambda ds: ds.isin(buses)].rename("Bus")

        expr = expr.sel({c: cbuses.index})

        if expr.size:
            if activity is not None:
                expr = expr.where(activity.sel({c: cbuses.index}))
            exprs.append(expr.groupby(cbuses).sum())

    # Extendable generators on LHS: p_nom * p_max_pu * activity_mask
    region_gens = n.generators.bus.isin(buses)
    extendable_gens = n.generators.p_nom_extendable
    region_ext_gens = n.generators[region_gens & extendable_gens]

    if not region_ext_gens.empty:
        ext_p_nom = m["Generator-p_nom"].loc[region_ext_gens.index]
        ext_p_max_pu = get_as_dense(n, "Generator", "p_max_pu", sns, inds=region_ext_gens.index)

        ext_p_max_pu.columns.name = "Generator-ext"
        ext_contribution = ext_p_nom * ext_p_max_pu

        # Use .where() to remove terms for inactive periods (sets var labels to -1)
        # rather than zeroing coefficients, which leaves orphaned variable references
        activity = get_activity_mask(n, "Generator", sns)[region_ext_gens.index]
        activity.columns.name = "Generator-ext"
        ext_contribution = ext_contribution.where(DataArray(activity))

        gen_buses = DataArray(
            region_ext_gens.bus.values,
            dims=["Generator-ext"],
            coords={"Generator-ext": region_ext_gens.index.values},
            name="Bus",
        )
        exprs.append(ext_contribution.groupby(gen_buses).sum())

    lhs = merge(exprs, join="outer").reindex(Bus=buses)

    # Non-extendable generators on RHS: p_nom * p_max_pu * activity_mask
    region_nonext_gens = n.generators[region_gens & ~extendable_gens]
    if not region_nonext_gens.empty:
        nonext_activity = get_activity_mask(n, "Generator", sns)[region_nonext_gens.index]
        nonext_p_max_pu = get_as_dense(n, "Generator", "p_max_pu", sns, inds=region_nonext_gens.index)
        nonext_p_max_pu = nonext_p_max_pu * nonext_activity
        rhs_existing = region_nonext_gens.p_nom * nonext_p_max_pu
        rhs_existing.index = sns
        bus_rhs_capacity = rhs_existing.T.groupby(region_nonext_gens.bus).sum().T
        bus_rhs_capacity = bus_rhs_capacity.reindex(columns=buses, fill_value=0)
        planning_reserve = planning_reserve - bus_rhs_capacity

    rhs = planning_reserve
    rhs.index.name = "snapshot"

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

        define_erm_nodal_balance_constraints(n, snapshots, erm_value, region_name, region_buses)
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
