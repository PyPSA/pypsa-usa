import logging

import linopy
import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


#: Bus columns a region name in a policy file can refer to
REGION_COLUMNS = ("country", "reeds_zone", "reeds_state", "nerc_reg", "region")

_warned_missing_region_columns: set[tuple[str, ...]] = set()


def get_region_buses(n, region_list):
    """
    Buses that belong to any region of ``region_list``.

    A region is matched against the bus name, the interconnect and the columns
    of ``REGION_COLUMNS``. Networks built with ``topological_boundaries: state``
    carry no ``reeds_zone`` or ``nerc_reg`` (a state spans several of each), so a
    column that is absent matches nothing; regions given at that resolution
    then select no buses.
    """
    region_list = list(region_list)
    buses = n.buses
    if "all" in region_list:
        return buses

    missing = tuple(col for col in REGION_COLUMNS if col != "region" and col not in buses.columns)
    if missing and missing not in _warned_missing_region_columns:
        _warned_missing_region_columns.add(missing)
        logger.warning(
            "The buses have no %s column(s); policy regions given at that resolution match no buses.",
            ", ".join(missing),
        )

    mask = buses.index.isin(region_list)
    for col in REGION_COLUMNS:
        if col in buses.columns:
            mask |= buses[col].isin(region_list)
    if "interconnect" in buses.columns:
        mask |= buses.interconnect.astype(str).str.lower().isin(region_list)
    return buses[mask]


def filter_components(
    n: pypsa.Network,
    component_type: str,
    planning_horizon: str | int,
    carrier_list: list[str],
    region_buses: pd.Index,
    extendable: bool,
):
    """
    Filter components based on common criteria.

    Parameters
    ----------
    - n: pypsa.Network
        The PyPSA network object.
    - component_type: str
        The type of component (e.g., "Generator", "StorageUnit").
    - planning_horizon: str or int
        The planning horizon to filter active assets.
    - carrier_list: list
        List of carriers to filter.
    - region_buses: pd.Index
        Index of region buses to filter.
    - extendable: bool, optional
        If specified, filters by extendable or non-extendable assets.

    Returns
    -------
    - pd.DataFrame
        Filtered assets.
    """
    component = n.components[component_type].static
    if planning_horizon != "all":
        ph = int(planning_horizon)
        iv = n.investment_periods

        # Check if there are any investment periods >= planning horizon
        valid_periods = iv[iv >= ph]
        if len(valid_periods) > 0:
            period = valid_periods[0]
            active_components = n.components[component_type].get_active_assets(investment_period=period)
        else:
            # Instead of empty index, create a boolean Series with all False values
            active_components = pd.Series(False, index=component.index)
    else:
        active_components = component.index

    # Links will throw the following attribute error, as we must specify bus0
    # AttributeError: 'DataFrame' object has no attribute 'bus'. Did you mean: 'bus0'?
    bus_name = "bus0" if component_type.lower() == "link" else "bus"

    # Handle both Series (boolean mask) and Index types of active_components
    if isinstance(active_components, pd.Series):
        filtered = component.loc[
            active_components
            & component.carrier.isin(carrier_list)
            & component[bus_name].isin(region_buses)
            & (component.p_nom_extendable == extendable)
        ]
    else:
        filtered = component.loc[
            component.index.isin(active_components)
            & component.carrier.isin(carrier_list)
            & component[bus_name].isin(region_buses)
            & (component.p_nom_extendable == extendable)
        ]

    return filtered


def ceil_precision(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)


def floor_precision(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)


def get_model_horizon(model: linopy.Model) -> list[int]:
    return model.variables.indexes["snapshot"].get_level_values(0).unique()


def get_last_snapshot(n: pypsa.Network, period: int) -> tuple[int, pd.Timestamp]:
    """
    Last snapshot of an investment period.

    The emission stores start every period empty, so their level there is the
    emissions of that period alone.
    """
    return n.snapshots[n.snapshots.get_level_values("period") == period][-1]
