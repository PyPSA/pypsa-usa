"""
Module for building state and sector level co2 tracking.
"""

import itertools
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


def build_co2_tracking(
    n: pypsa.Network,
    config: Optional[dict[str, Any]] = None,
) -> None:
    """
    Main funtion to interface with.
    """

    states = n.buses.STATE.unique()

    sectors = ["pwr", "trn", "res", "com", "ind"]

    if not config:
        config = {}

    if "co2" not in n.carriers:
        _add_co2_carrier(n, config)

    _build_co2_bus(n, states, sectors)
    _build_co2_store(n, states, sectors)


def build_ch4_tracking(
    n: pypsa.Network,
    gwp: float,
    leakage_rate: float,
    config: Optional[dict[str, Any]] = None,
) -> None:
    """
    Builds CH4 tracking.

    Natural gas network must already be constructed
    """

    states = [x for x in n.buses.STATE.dropna().unique() if x != np.nan]

    if not config:
        config = {}

    if "ch4" not in n.carriers:
        _add_ch4_carrier(n, config)

    _build_ch4_bus(n, states)
    _build_ch4_store(n, states)
    _build_ch4_links(n, states, gwp, leakage_rate)


def _add_co2_carrier(n, config: dict[Any]):
    try:
        nice_name = config["plotting"]["nice_names"]["co2"]
    except KeyError:
        nice_name = "CO2"
    try:
        color = config["plotting"]["tech_colors"]["co2"]
    except KeyError:
        color = "#000000"  # black

    n.add("Carrier", "co2", nice_name=nice_name, color=color)


def _build_co2_bus(n: pypsa.Network, states: list[str], sectors: list[str]):
    """
    Builds state level co2 bus per sector.
    """

    df = pd.DataFrame(itertools.product(states, sectors), columns=["state", "sector"])
    df.index = df.state + " " + df.sector

    n.madd("Bus", df.index, suffix="-co2", carrier="co2", STATE=df.state)


def _build_co2_store(n: pypsa.Network, states: list[str], sectors: list[str]):
    """
    Builds state level co2 stores per sector.
    """

    df = pd.DataFrame(itertools.product(states, sectors), columns=["state", "sector"])
    df.index = df.state + " " + df.sector

    n.madd(
        "Store",
        df.index,
        suffix="-co2",
        bus=df.index + "-co2",
        e_nom_extendable=False,
        marginal_cost=0,
        e_nom=np.inf,
        e_initial=0,
        e_cyclic=False,
        e_cyclic_per_period=False,
        standing_loss=0,
        e_min_pu=0,
        e_max_pu=1,
        carrier="co2",
    )


def _add_ch4_carrier(n, config: dict[Any]):
    try:
        nice_name = config["plotting"]["nice_names"]["ch4"]
    except KeyError:
        nice_name = "CH4"
    try:
        color = config["plotting"]["tech_colors"]["co2"]
    except KeyError:
        color = "#000000"  # black

    n.add("Carrier", "ch4", nice_name=nice_name, color=color)


def _build_ch4_bus(n: pypsa.Network, states: list[str]):
    """
    Builds state level co2 bus per sector.
    """

    df = pd.DataFrame(states, columns=["state"])
    df.index = df.state

    n.madd("Bus", df.index, suffix=" gas-ch4", carrier="ch4", STATE=df.state)


def _build_ch4_store(n: pypsa.Network, states: list[str]):
    """
    Builds state level co2 stores per sector.
    """

    df = pd.DataFrame(states, columns=["state"])
    df.index = df.state

    n.madd(
        "Store",
        df.index,
        suffix=" gas-ch4",
        bus=df.index + " gas-ch4",
        e_nom_extendable=False,
        marginal_cost=0,
        e_nom=np.inf,
        e_initial=0,
        e_cyclic=False,
        e_cyclic_per_period=False,
        standing_loss=0,
        e_min_pu=0,
        e_max_pu=1,
        carrier="ch4",
    )


def _build_ch4_links(n, states: list[str], gwp: float, leakage_rate: float):
    """
    Modifies existing gas production links.
    """

    # first extract out exising gas production links

    gas_production = [f"{x} gas production" for x in states]
    links = n.links[n.links.index.isin(gas_production)].index

    # calculate co2e value per unit injected to the ng system
    emissions = gwp * leakage_rate

    # append the connection to methane stores

    n.links.loc[links, "bus2"] = n.links.loc[links,].bus1 + "-ch4"  # 'CA gas-ch4'
    n.links.loc[links, "efficiency2"] = emissions
