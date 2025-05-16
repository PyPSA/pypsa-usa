"""Module for building state and sector level co2 tracking."""

import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


def build_co2_tracking(
    n: pypsa.Network,
    config: dict[str, Any] | None = None,
) -> None:
    """Main funtion to interface with."""
    states = n.buses.STATE.unique()

    sectors = ["pwr", "trn", "res", "com", "ind"]

    if not config:
        config = {}

    if "co2" not in n.carriers:
        _add_co2_carrier(n, config)

    _build_co2_bus(n, states, sectors)
    _build_co2_store(n, states, sectors)


def build_co2_storage(n: pypsa.Network, co2_storage_csv: str):
    """Builds node level CO2 (underground) storage."""

    # get node level CO2 storage potential and cost
    co2_storage = pd.read_csv(co2_storage_csv).set_index("node")

    # add node level bus to represent CO2 captured by different processes
    n.madd("Bus",
        co2_storage.index,
        suffix = " co2 capture",
        carrier = "co2",
    )

    # add node level store to represent CO2 (underground) storage
    n.madd("Store",
        co2_storage.index,
        suffix = " co2 storage",
        bus = co2_storage.index + " co2 capture",
        e_nom_extendable = True,
        e_nom_max = co2_storage["potential [MtCO2]"] * 1e6,
        marginal_cost = co2_storage["cost [USD/tCO2]"],
        carrier = "co2",
    )


def build_ch4_tracking(
    n: pypsa.Network,
    gwp: float,
    upstream_leakage_rate: float,
    downstream_leakage_rate: float,
    plotting_config: dict[str, Any] | None = None,
) -> None:
    """
    Builds CH4 tracking.

    Natural gas network must already be constructed
    """
    states = [x for x in n.buses.STATE.dropna().unique() if x != np.nan]

    if not plotting_config:
        plotting_config = {}

    if "ch4" not in n.carriers:
        _add_ch4_carrier(n, plotting_config)

    _build_ch4_bus(n, states)
    _build_ch4_store(n, states)
    _build_ch4_upstream(n, gwp, upstream_leakage_rate)
    _build_ch4_downstream(n, gwp, downstream_leakage_rate)

    # supress pypsa warnings
    n.links["bus3"] = n.links.bus3.fillna("")
    n.links["efficiency3"] = n.links.efficiency3.fillna(0)


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
    """Builds state level co2 bus per sector."""
    df = pd.DataFrame(itertools.product(states, sectors), columns=["state", "sector"])
    df.index = df.state + " " + df.sector

    n.madd("Bus", df.index, suffix="-co2", carrier="co2", STATE=df.state)


def _build_co2_store(n: pypsa.Network, states: list[str], sectors: list[str]):
    """Builds state level co2 stores per sector."""
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
    """Builds state level co2 bus per sector."""
    df = pd.DataFrame(states, columns=["state"])
    df.index = df.state

    n.madd("Bus", df.index, suffix=" gas-ch4", carrier="ch4", STATE=df.state)


def _build_ch4_store(n: pypsa.Network, states: list[str]):
    """Builds state level co2 stores per sector."""
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


def _build_ch4_upstream(n, gwp: float, leakage_rate: float):
    """Modifies existing gas production links."""
    # first extract out exising gas production links
    links = n.links[n.links.carrier == "gas production"].index

    # calculate co2e value per unit injected to the ng system
    emissions = gwp * leakage_rate

    # append the connection to methane stores

    if "bus3" in n.links.columns:
        assert all(n.links.loc[links].bus3.isna())
    if "efficiency3" in n.links.columns:
        assert all(n.links.loc[links].efficiency3.isna())

    n.links.loc[links, "bus3"] = n.links.loc[links,].bus1 + "-ch4"  # 'CA gas-ch4'
    n.links.loc[links, "efficiency3"] = emissions


def _build_ch4_downstream(n, gwp: float, leakage_rate: float):
    """Modifies existing gas consuming links."""
    # want all gas links that originate at the state and are not trade or storage related

    gas_buses = n.buses[n.buses.carrier == "gas"]
    gas_users = n.links[(n.links.bus0.isin(gas_buses.index)) & ~(n.links.carrier.isin(["gas storage", "gas trade"]))]

    links = gas_users.index

    # calculate co2e value per unit injected to the ng system
    emissions = gwp * leakage_rate

    # append the connection to methane stores

    assert all(n.links.loc[links].bus3.isna())
    assert all(n.links.loc[links].efficiency3.isna())

    n.links.loc[links, "bus3"] = n.links.loc[links,].bus0 + "-ch4"  # 'CA gas-ch4'
    n.links.loc[links, "efficiency3"] = emissions
