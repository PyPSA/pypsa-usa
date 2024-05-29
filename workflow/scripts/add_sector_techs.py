"""
Module to add sector conversion technologies.
"""

import numpy as np
import pandas as pd
import pypsa


def add_electrical_distribution(n: pypsa.Network) -> None:
    """
    Adds links to connect node electricity to end-use demand.

    Will only create the links for "com-", "res-", and "ind-"; not
    "trn-"
    """

    def replace_link_names(s: str) -> str:
        repl = {
            "res-elec": "res-elec-dist",
            "com-elec": "com-elec-dist",
            "ind-elec": "ind-elec-dist",
        }
        for k, v in repl.items():
            s = s.replace(k, v)
        return s

    links = n.loads[
        (n.loads.carrier.str.endswith("-elec"))
        & ~(n.loads.carrier.str.endswith("trn-elec"))
    ][["bus", "carrier"]].rename(columns={"bus": "bus1"})
    links.index = links.index.map(replace_link_names)
    links["bus0"] = links.bus1.map(lambda x: x[: x.rfind(" ")])  # remove carrier

    n.madd(
        "Link",
        links.index,
        bus0=links.bus0,
        bus1=links.bus1,
        p_nom_extendable=True,
        p_nom_min=0,
        p_nom_max=np.inf,
        p_min_pu=-1,
        p_max_pu=1,
        carrier=links.carrier,
        efficiency=1,
        lifetime=100,
        capital_cost=0,
    )


def add_evs(n: pypsa.Network) -> None:
    """
    Adds EV connections.
    """

    def replace_link_names(s: str) -> str:
        repl = {
            "trn-elec": "trn-evs",
        }
        for k, v in repl.items():
            s = s.replace(k, v)
        return s

    links = n.loads[n.loads.carrier.str.endswith("trn-elec")][
        ["bus", "carrier"]
    ].rename(columns={"bus": "bus1"})
    links.index = links.index.map(replace_link_names)
    links["bus0"] = links.bus1.map(lambda x: x[: x.rfind(" ")])  # remove carrier

    n.madd(
        "Link",
        links.index,
        bus0=links.bus0,
        bus1=links.bus1,
        p_nom_extendable=True,
        p_nom_min=0,
        p_nom_max=np.inf,
        p_min_pu=-1,
        p_max_pu=1,
        carrier=links.carrier,
        efficiency=1,
        lifetime=100,
        capital_cost=0,
    )


def add_air_cons(n: pypsa.Network) -> None:
    """
    Adds air conditioners to the network.
    """

    def replace_bus_names(s: str) -> str:
        repl = {"res-cool": "res-elec", "com-cool": "com-elec"}
        for k, v in repl.items():
            s = s.replace(k, v)
        return s

    def replace_link_names(s: str) -> str:
        repl = {"res-cool": "res-air-con", "com-cool": "com-air-con"}
        for k, v in repl.items():
            s = s.replace(k, v)
        return s

    links = n.loads[
        (n.loads.carrier.str.endswith("res-cool"))
        | (n.loads.carrier.str.endswith("com-cool"))
    ][["bus", "carrier"]].rename(columns={"bus": "bus1"})

    links["bus0"] = links.bus1.map(replace_bus_names)
    links.index = links.index.map(replace_link_names)

    n.madd(
        "Link",
        links.index,
        bus0=links.bus0,
        bus1=links.bus1,
        p_nom_extendable=True,
        p_nom_min=0,
        p_nom_max=np.inf,
        p_min_pu=-1,
        p_max_pu=1,
        carrier=links.carrier,
        efficiency=1,
        lifetime=100,
        capital_cost=0,
    )


def add_heat_pumps(n: pypsa.Network):
    """
    Adds heat pumps to the network.
    """

    def replace_bus_names(s: str) -> str:
        repl = {"res-heat": "res-elec", "com-heat": "com-elec", "ind-heat": "ind-elec"}
        for k, v in repl.items():
            s = s.replace(k, v)
        return s

    def replace_link_names(s: str) -> str:
        repl = {
            "res-heat": "res-heat-pump",
            "com-heat": "com-heat-pump",
            "ind-heat": "ind-heat-pump",
        }
        for k, v in repl.items():
            s = s.replace(k, v)
        return s

    links = n.loads[
        (n.loads.carrier.str.endswith("res-heat"))
        | (n.loads.carrier.str.endswith("com-heat"))
        | (n.loads.carrier.str.endswith("ind-heat"))
    ][["bus", "carrier"]].rename(columns={"bus": "bus1"})

    links["bus0"] = links.bus1.map(replace_bus_names)
    links.index = links.index.map(replace_link_names)

    n.madd(
        "Link",
        links.index,
        bus0=links.bus0,
        bus1=links.bus1,
        p_nom_extendable=True,
        p_nom_min=0,
        p_nom_max=np.inf,
        p_min_pu=-1,
        p_max_pu=1,
        carrier="heat-pump",
        efficiency=1,
        lifetime=100,
        capital_cost=0,
    )


def add_gas_furnaces(n: pypsa.Network) -> None:
    """
    Adds natural gas based furnaces to the network.
    """

    def replace_link_names(s: str) -> str:
        repl = {
            "res-heat": "res-furnace",
            "com-heat": "com-furnace",
            "ind-heat": "ind-furnace",
        }
        for k, v in repl.items():
            s = s.replace(k, v)
        return s

    links = n.loads[
        (n.loads.carrier.str.endswith("res-heat"))
        | (n.loads.carrier.str.endswith("com-heat"))
        | (n.loads.carrier.str.endswith("ind-heat"))
    ][["bus", "carrier"]].rename(columns={"bus": "bus1"})

    links["bus0"] = links.bus1.map(n.buses.STATE)
    links["bus0"] = links.bus0.map(lambda x: f"{x} gas")

    links.index = links.index.map(replace_link_names)

    n.madd(
        "Link",
        links.index,
        bus0=links.bus0,
        bus1=links.bus1,
        p_nom_extendable=True,
        p_nom_min=0,
        p_nom_max=np.inf,
        p_min_pu=-1,
        p_max_pu=1,
        carrier="gas-furnace",
        efficiency=1,
        lifetime=100,
        capital_cost=0,
    )


def add_coal_furnaces(n: pypsa.Network) -> None:
    """
    Adds natural gas based furnaces to the network.
    """

    def replace_link_names(s: str) -> str:
        repl = {"ind-heat": "ind-boiler"}
        for k, v in repl.items():
            s = s.replace(k, v)
        return s

    links = n.loads[(n.loads.carrier.str.endswith("ind-heat"))][
        ["bus", "carrier"]
    ].rename(columns={"bus": "bus1"})

    links["bus0"] = links.bus1.map(n.buses.STATE)
    links["bus0"] = links.bus0.map(lambda x: f"{x} coal")

    links.index = links.index.map(replace_link_names)

    n.madd(
        "Link",
        links.index,
        bus0=links.bus0,
        bus1=links.bus1,
        p_nom_extendable=True,
        p_nom_min=0,
        p_nom_max=np.inf,
        p_min_pu=-1,
        p_max_pu=1,
        carrier="coal-furnace",
        efficiency=1,
        lifetime=100,
        capital_cost=0,
    )
