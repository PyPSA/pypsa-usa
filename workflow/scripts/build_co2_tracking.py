"""
Module for building state and sector level co2 tracking.
"""

import itertools
from typing import Any, Optional

import numpy as np
import pandas as pd
import pypsa


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

    build_co2_bus(n, states, sectors)
    build_co2_store(n, states, sectors)


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


def build_co2_bus(n: pypsa.Network, states: list[str], sectors: list[str]):
    """
    Builds state level co2 bus per sector.
    """

    df = pd.DataFrame(itertools.product(states, sectors), columns=["state", "sector"])
    df.index = df.state + " " + df.sector

    n.madd("Bus", df.index, suffix="-co2", carrier="co2", STATE=df.state)


def build_co2_store(n: pypsa.Network, states: list[str], sectors: list[str]):
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
    )
