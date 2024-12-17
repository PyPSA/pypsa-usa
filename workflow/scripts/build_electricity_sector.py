"""Functions for building electricity infrastructure in sector studies"""

from typing import Any, Optional

import numpy as np
import pypsa
from constants_sector import SecCarriers


def build_electricty(
    n: pypsa.Network,
    sector: str,
    demand_response: Optional[dict[str, Any]] = None,
) -> None:
    """Adds electricity sector infrastructre data"""

    if not demand_response:
        demand_response = {}
    dr_shift = demand_response.get("shift", 0)

    add_electricity_infrastructure(n, sector)
    add_electricity_stores(n, sector, dr_shift)


def add_electricity_infrastructure(n: pypsa.Network, sector: str):
    """
    Adds links to connect electricity nodes.

    For example, will build the link between "p480 0" and "p480 0 res-
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
    dr_shift: Optional[int | float] = None,
) -> None:
    """
    Adds stores to the network to use for demand response.
    """

    elec = SecCarriers.ELECTRICITY.value

    df = n.loads[n.loads.index.str.endswith(f"{sector}-{elec}")].copy()
    df["x"] = df.bus.map(n.buses.x)
    df["y"] = df.bus.map(n.buses.y)
    df["carrier"] = df["carrier"]
    df = df.set_index("bus")
    df["p_nom"] = n.loads_t["p_set"][df.index].max().round(2)

    if not dr_shift:
        dr_shift = 0

    # apply shiftable load via p_max_pu
    # first calc the raw max shiftable load per timestep
    # normalize agaist the max load value
    # ie. if shiftable load is 10%
    #   p_max_mu.max() will return a vector of all '0.10' values

    p_max_pu = n.loads_t["p_set"][df.index].mul(dr_shift).div(n.loads_t["p_set"][df.index].max()).round(4)

    n.madd(
        "Bus",
        df.index,
        suffix=f"-store",
        x=df.x,
        y=df.y,
        carrier=df.carrier,
        unit="MWh",
    )

    # charging discharing limits added in custom constraints

    # by default, no demand response
    n.madd(
        "Link",
        df.index,
        suffix=f"-charger",
        bus0=df.index,
        bus1=df.index + "-store",
        efficiency=1,
        carrier=df.carrier,
        p_nom_extendable=True,
        capital_cost=0,
    )

    n.madd(
        "Link",
        df.index,
        suffix=f"-discharger",
        bus0=df.index + "-store",
        bus1=df.index,
        efficiency=1,
        carrier=df.carrier,
        p_nom_extendable=True,
        p_nom_extendable=False,
        p_nom=df.p_nom,
        p_max_pu=p_max_pu,
    )

    n.madd(
        "Store",
        df.index,
        bus=df.index + "-store",
        e_cyclic=True,
        e_nom_extendable=True,
        carrier=df.carrier,
        standing_loss=0,
        capital_cost=0,
        lifetime=np.inf,
    )
