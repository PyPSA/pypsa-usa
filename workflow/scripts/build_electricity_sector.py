"""Functions for building electricity infrastructure in sector studies."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import pypsa
from constants_sector import SecCarriers

logger = logging.getLogger(__name__)


def build_electricty(
    n: pypsa.Network,
    sector: str,
    pop_layout_path: pd.DataFrame | None = None,
    options: dict[str, Any] | None = None,
) -> None:
    """Adds electricity sector infrastructre data."""
    if sector in ("res", "com", "srv"):
        split_urban_rural = options.get("split_urban_rural", False)

        if split_urban_rural:
            pop_layout = pd.read_csv(pop_layout_path).set_index("name")
            _split_urban_rural_load(n, sector, pop_layout)
            for system in ["urban", "rural"]:
                add_electricity_infrastructure(n, sector, system)

        else:
            _format_total_load(n, sector)
            add_electricity_infrastructure(n, sector, "total")

    elif sector == "ind":
        add_electricity_infrastructure(n, sector)

    else:
        raise ValueError

    dr_config = options.get("demand_response", {})
    if dr_config:
        add_electricity_dr(n, sector, dr_config)


def add_electricity_infrastructure(
    n: pypsa.Network,
    sector: str,
    suffix: str | None = None,
):
    """
    Adds links to connect electricity nodes.

    For example, will build the link between "p480 0" and "p480 0 ind-
    elec"
    """
    elec = SecCarriers.ELECTRICITY.value

    if suffix:
        suffix = f"-{suffix}-{elec}"
    else:
        suffix = f"-{elec}"

    df = n.loads[n.loads.index.str.endswith(f"{sector}{suffix}")].copy()

    df["bus0"] = df.apply(lambda row: row.bus.split(f" {row.carrier}")[0], axis=1)
    df["bus1"] = df.bus
    df["sector"] = df.carrier.map(lambda x: x.split("-")[0])
    df.index = df["bus0"] + " " + df["sector"]
    df["carrier"] = df["sector"] + f"-{elec}"

    n.madd(
        "Link",
        df.index,
        suffix=suffix,
        bus0=df.bus0,
        bus1=df.bus1,
        carrier=df.carrier,
        efficiency=1,
        capital_cost=0,
        p_nom_extendable=True,
        lifetime=np.inf,
        build_year=n.investment_periods[0],
    )


def add_electricity_dr(
    n: pypsa.Network,
    sector: str,
    dr_config: dict[str, Any],
) -> None:
    """Adds stores to the network to use for demand response."""
    # check if dr is applied at a per-carrier level
    dr_config = dr_config.get(sector, dr_config)

    by_carrier = dr_config.get("by_carrier", False)
    if by_carrier:
        dr_config = dr_config.get("elec", {})

    shift = dr_config.get("shift", 0)
    if shift == 0:
        logger.info(f"DR not applied to {sector} as allowable sift is {shift}")
        return

    # assign marginal cost value

    marginal_cost_storage = dr_config.get("marginal_cost", 0)
    if marginal_cost_storage == 0:
        logger.warning(f"No cost applied to demand response for {sector}")

    elec = SecCarriers.ELECTRICITY.value

    df = n.loads[n.loads.index.str.endswith(f"-{elec}") & n.loads.index.str.contains(f"{sector}-")].copy()
    df["x"] = df.bus.map(n.buses.x)
    df["y"] = df.bus.map(n.buses.y)
    df["STATE"] = df.bus.map(n.buses.STATE)
    df["STATE_NAME"] = df.bus.map(n.buses.STATE_NAME)
    df["carrier"] = df["carrier"] + "-dr"
    df = df.set_index("bus")

    # two buses for forward and backwards load shifting

    n.madd(
        "Bus",
        df.index,
        suffix="-fwd-dr",
        x=df.x,
        y=df.y,
        carrier=df.carrier,
        unit="MWh",
        STATE=df.STATE,
        STATE_NAME=df.STATE_NAME,
    )

    n.madd(
        "Bus",
        df.index,
        suffix="-bck-dr",
        x=df.x,
        y=df.y,
        carrier=df.carrier,
        unit="MWh",
        STATE=df.STATE,
        STATE_NAME=df.STATE_NAME,
    )

    # seperate charging/discharging links to follow conventions

    n.madd(
        "Link",
        df.index,
        suffix="-fwd-dr-charger",
        bus0=df.index,
        bus1=df.index + "-fwd-dr",
        carrier=df.carrier,
        p_nom_extendable=False,
        p_nom=np.inf,
        lifetime=np.inf,
        build_year=n.investment_periods[0],
    )

    n.madd(
        "Link",
        df.index,
        suffix="-fwd-dr-discharger",
        bus0=df.index + "-fwd-dr",
        bus1=df.index,
        carrier=df.carrier,
        p_nom_extendable=False,
        p_nom=np.inf,
        lifetime=np.inf,
        build_year=n.investment_periods[0],
    )

    n.madd(
        "Link",
        df.index,
        suffix="-bck-dr-charger",
        bus0=df.index,
        bus1=df.index + "-bck-dr",
        carrier=df.carrier,
        p_nom_extendable=False,
        p_nom=np.inf,
        lifetime=np.inf,
        build_year=n.investment_periods[0],
    )

    n.madd(
        "Link",
        df.index,
        suffix="-bck-dr-discharger",
        bus0=df.index + "-bck-dr",
        bus1=df.index,
        carrier=df.carrier,
        p_nom_extendable=False,
        p_nom=np.inf,
        lifetime=np.inf,
        build_year=n.investment_periods[0],
    )

    # backward stores have positive marginal cost storage and postive e
    # forward stores have negative marginal cost storage and negative e

    n.madd(
        "Store",
        df.index,
        suffix="-bck-dr",
        bus=df.index + "-bck-dr",
        e_cyclic=True,
        e_nom_extendable=False,
        e_nom=np.inf,
        e_min_pu=0,
        e_max_pu=1,
        carrier=df.carrier,
        marginal_cost_storage=marginal_cost_storage,
        lifetime=np.inf,
        build_year=n.investment_periods[0],
    )

    n.madd(
        "Store",
        df.index,
        suffix="-fwd-dr",
        bus=df.index + "-fwd-dr",
        e_cyclic=True,
        e_nom_extendable=False,
        e_nom=np.inf,
        e_min_pu=-1,
        e_max_pu=0,
        carrier=df.carrier,
        marginal_cost_storage=marginal_cost_storage * (-1),
        lifetime=np.inf,
        build_year=n.investment_periods[0],
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
    """Formats load with 'total' prefix to match urban/rural split."""
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
