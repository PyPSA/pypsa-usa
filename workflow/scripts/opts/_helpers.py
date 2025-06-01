import logging

import linopy
import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


def get_region_buses(n, region_list):
    return n.buses[
        (
            n.buses.country.isin(region_list)
            | n.buses.reeds_zone.isin(region_list)
            | n.buses.reeds_state.isin(region_list)
            | n.buses.interconnect.str.lower().isin(region_list)
            | n.buses.nerc_reg.isin(region_list)
            | n.buses.index.isin(region_list)
            | (1 if "all" in region_list else 0)
        )
    ]


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
    component = n.df(component_type)
    if planning_horizon != "all":
        ph = int(planning_horizon)
        iv = n.investment_periods

        # Check if there are any investment periods >= planning horizon
        valid_periods = iv[iv >= ph]
        if len(valid_periods) > 0:
            period = valid_periods[0]
            active_components = n.get_active_assets(component.index.name, period)
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
