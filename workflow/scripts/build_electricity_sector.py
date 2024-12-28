"""Functions for building electricity infrastructure in sector studies"""

from typing import Optional

import numpy as np
import pandas as pd
import pypsa
from constants_sector import SecCarriers


def build_electricty(
    n: pypsa.Network,
    sector: str,
    split_urban_rural: Optional[bool] = None,
    pop_layout_path: Optional[pd.DataFrame] = None,
) -> None:
    """Adds electricity sector infrastructre data"""

    if sector in ("res", "com", "srv"):
        pop_layout = pd.read_csv(pop_layout_path).set_index("name")
        if split_urban_rural:
            _split_urban_rural_load(n, sector, pop_layout)
        else:
            _format_total_load(n, sector)
    elif sector == "ind":
        add_electricity_infrastructure(n, sector)
    else:
        raise ValueError

    add_electricity_stores(n, sector)


def add_electricity_infrastructure(n: pypsa.Network, sector: str):
    """
    Adds links to connect electricity nodes.

    For example, will build the link between "p480 0" and "p480 0 ind-
    elec"
    """

    elec = SecCarriers.ELECTRICITY.value

    df = n.loads[n.loads.index.str.endswith(f"{sector}-{elec}")].copy()

    df["bus0"] = df.apply(lambda row: row.bus.split(f" {row.carrier}")[0], axis=1)
    df["bus1"] = df.bus
    df["sector"] = df.carrier.map(lambda x: x.split("-")[0])
    df.index = df["bus0"] + " " + df["sector"]
    df["carrier"] = df["sector"] + f"-{elec}"

    n.madd(
        "Link",
        df.index,
        suffix=f"-{elec}",
        bus0=df.bus0,
        bus1=df.bus1,
        carrier=df.carrier,
        efficiency=1,
        capital_cost=0,
        p_nom_extendable=True,
        lifetime=np.inf,
    )


def add_electricity_stores(
    n: pypsa.Network,
    sector: str,
) -> None:
    """
    Adds stores to the network to use for demand response.
    """

    elec = SecCarriers.ELECTRICITY.value

    df = n.loads[n.loads.index.str.endswith(f"-{elec}") & n.loads.index.str.contains(f"{sector}-")].copy()
    df["x"] = df.bus.map(n.buses.x)
    df["y"] = df.bus.map(n.buses.y)
    df["carrier"] = df["carrier"]
    df = df.set_index("bus")

    n.madd(
        "Bus",
        df.index,
        suffix=f"-store",
        x=df.x,
        y=df.y,
        carrier=df.carrier,
        unit="MWh",
    )

    # p_nom set to zero
    # demand response config will override this setting

    n.madd(
        "Link",
        df.index,
        suffix=f"-charger",
        bus0=df.index,
        bus1=df.index + "-store",
        efficiency=1,
        carrier=df.carrier,
        p_nom_extendable=False,
        p_nom=0,
    )

    n.madd(
        "Link",
        df.index,
        suffix=f"-discharger",
        bus0=df.index + "-store",
        bus1=df.index,
        efficiency=1,
        carrier=df.carrier,
        p_nom_extendable=False,
        p_nom=0,
    )

    n.madd(
        "Store",
        df.index,
        bus=df.index + "-store",
        e_cyclic=True,
        e_nom_extendable=False,
        e_nom=np.inf,
        carrier=df.carrier,
        standing_loss=0,
        capital_cost=0,
        lifetime=np.inf,
    )


def _split_urban_rural_load(
    n: pypsa.Network,
    sector: str,
    ratios: pd.DataFrame,
) -> None:
    """
    Splits a combined load into urban/rural loads.

    Takes a load (such as "p600 0 com-elec") and converts it into two
    loads ("p600 0 com-urban-elec" and "p600 0 com-rural-elec"). The
    buses for these loads are also added (under the name, for example
    "p600 0 com-urban-elec" and "p600 0 com-rural-elec" at the same
    location as "p600 0").
    """

    assert sector in ("com", "res")

    fuel = SecCarriers.ELECTRICITY.value

    load_names = n.loads[n.loads.carrier == f"{sector}-{fuel}"].index.to_list()

    for system in ("urban", "rural"):

        # add buses to connect the new loads to
        new_buses = pd.DataFrame(index=load_names)
        new_buses.index = new_buses.index.map(n.loads.bus)
        new_buses["x"] = new_buses.index.map(n.buses.x)
        new_buses["y"] = new_buses.index.map(n.buses.y)
        new_buses["country"] = new_buses.index.map(n.buses.country)
        new_buses["interconnect"] = new_buses.index.map(n.buses.interconnect)
        new_buses["STATE"] = new_buses.index.map(n.buses.STATE)
        new_buses["STATE_NAME"] = new_buses.index.map(n.buses.STATE_NAME)
        new_buses["country"] = new_buses.index.map(n.buses.country)
        new_buses["reeds_zone"] = new_buses.index.map(n.buses.reeds_zone)
        new_buses["reeds_ba"] = new_buses.index.map(n.buses.reeds_ba)

        # strip out the 'res-heat' and 'com-heat' to add in 'rural' and 'urban'
        new_buses.index = new_buses.index.str.rstrip(f" {sector}-{fuel}")

        n.madd(
            "Bus",
            new_buses.index,
            suffix=f" {sector}-{system}-{fuel}",
            x=new_buses.x,
            y=new_buses.y,
            carrier=f"{sector}-{system}-{fuel}",
            country=new_buses.country,
            interconnect=new_buses.interconnect,
            STATE=new_buses.STATE,
            STATE_NAME=new_buses.STATE_NAME,
        )

        # get rural or urban loads
        loads_t = n.loads_t.p_set[load_names]
        loads_t = loads_t.rename(
            columns={x: x.rstrip(f" {sector}-{fuel}") for x in loads_t.columns},
        )
        loads_t = loads_t.mul(ratios[f"{system}_fraction"])

        n.madd(
            "Load",
            new_buses.index,
            suffix=f" {sector}-{system}-{fuel}",
            bus=new_buses.index + f" {sector}-{system}-{fuel}",
            p_set=loads_t,
            carrier=f"{sector}-{system}-{fuel}",
        )

    # remove old combined loads from the network
    n.mremove("Load", load_names)
    n.mremove("Bus", load_names)


def _format_total_load(
    n: pypsa.Network,
    sector: str,
) -> None:
    """
    Formats load with 'total' prefix to match urban/rural split.
    """

    assert sector in ("com", "res", "srv")

    fuel = SecCarriers.ELECTRICITY.value

    load_names = n.loads[n.loads.carrier == f"{sector}-{fuel}"].index.to_list()

    # add buses to connect the new loads to
    new_buses = pd.DataFrame(index=load_names)
    new_buses.index = new_buses.index.map(n.loads.bus)
    new_buses["x"] = new_buses.index.map(n.buses.x)
    new_buses["y"] = new_buses.index.map(n.buses.y)
    new_buses["country"] = new_buses.index.map(n.buses.country)
    new_buses["interconnect"] = new_buses.index.map(n.buses.interconnect)
    new_buses["STATE"] = new_buses.index.map(n.buses.STATE)
    new_buses["STATE_NAME"] = new_buses.index.map(n.buses.STATE_NAME)
    new_buses["country"] = new_buses.index.map(n.buses.country)
    new_buses["reeds_zone"] = new_buses.index.map(n.buses.reeds_zone)
    new_buses["reeds_ba"] = new_buses.index.map(n.buses.reeds_ba)

    # strip out the 'res-heat' and 'com-heat' to add in 'rural' and 'urban'
    new_buses.index = new_buses.index.str.rstrip(f" {sector}-{fuel}")

    n.madd(
        "Bus",
        new_buses.index,
        suffix=f" {sector}-total-{fuel}",
        x=new_buses.x,
        y=new_buses.y,
        carrier=f"{sector}-total-{fuel}",
        country=new_buses.country,
        interconnect=new_buses.interconnect,
        STATE=new_buses.STATE,
        STATE_NAME=new_buses.STATE_NAME,
    )

    # get rural or urban loads
    loads_t = n.loads_t.p_set[load_names]
    loads_t = loads_t.rename(
        columns={x: x.rstrip(f" {sector}-{fuel}") for x in loads_t.columns},
    )

    n.madd(
        "Load",
        new_buses.index,
        suffix=f" {sector}-total-{fuel}",
        bus=new_buses.index + f" {sector}-total-{fuel}",
        p_set=loads_t,
        carrier=f"{sector}-total-{fuel}",
    )

    # remove old combined loads from the network
    n.mremove("Load", load_names)
    n.mremove("Bus", load_names)
