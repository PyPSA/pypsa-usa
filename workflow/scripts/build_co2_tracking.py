"""
Module for building state and sector level co2 tracking.
"""

import itertools

import numpy as np
import pandas as pd
import pypsa

TECHS_PER_SECTOR = {
    "pwr": ["ccgt", "ocgt", "coal", "oil"],
    "res": ["gas-furnace"],
    "com": ["gas-furnace"],
    "ind": ["gas-furnace", "coal-boiler"],
    "trn": ["lpg"],
}


def build_co2_tracking(
    n: pypsa.Network,
) -> None:
    """
    Main funtion to interface with.
    """

    states = n.buses.STATE.unique()

    sectors = ["pwr", "trn", "res", "com", "ind"]

    build_co2_bus(n, states, sectors)
    build_co2_store(n, states, sectors)


def build_co2_bus(n: pypsa.Network, states: list[str], sectors: list[str]):
    """
    Builds state level co2 bus per sector.
    """

    if "co2" not in n.carriers:
        n.add("Carrier", "co2", nice_name="CO2")

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
