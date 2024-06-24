"""
Module for building heating and cooling infrastructure.
"""

from typing import Optional

import pandas as pd
import pypsa
import xarray as xr
from add_electricity import load_costs


def build_heat(
    n: pypsa.Network,
    costs_path: str,
    pop_layout_path: str,
    cop_ashp_path: str,
    cop_gshp_path: str,
) -> None:
    """
    Main funtion to interface with.
    """

    costs = load_costs(costs_path)

    pop_layout = pd.read_csv(pop_layout_path).set_index("name")

    ashp_cop = xr.open_dataarray(cop_ashp_path).to_pandas().reindex(index=n.snapshots)

    gshp_cop = xr.open_dataarray(cop_gshp_path).to_pandas().reindex(index=n.snapshots)

    for sector in ("res", "com"):
        add_service_heat(n, sector, pop_layout, costs, ashp_cop, gshp_cop)
        add_service_cooling(n, sector, costs)

    for sector in "ind":
        add_industrial_heat(n, sector, costs)


def add_industrial_heat(
    n: pypsa.Network,
    sector: str,
    costs: pd.DataFrame,
    **kwargs,
) -> None:

    assert sector == "ind"

    add_industrial_furnace(n, costs)
    add_industrial_boiler(n, costs)


def add_service_heat(
    n: pypsa.Network,
    sector: str,
    pop_layout: pd.DataFrame,
    costs: pd.DataFrame,
    ashp_cop: Optional[pd.DataFrame] = None,
    gshp_cop: Optional[pd.DataFrame] = None,
    **kwargs,
):
    """
    Adds heating links for residential and commercial sectors.

    Costs and efficiencies are taken from:
    - https://www.eia.gov/analysis/studies/buildings/equipcosts/pdf/full.pdf
    - https://www.nrel.gov/docs/fy18osti/70485.pdf (Fig. 24)
        - https://data.nrel.gov/submissions/78 (Data)
    """

    assert sector in ("res", "com")

    heat_systems = ("rural", "urban")

    # seperates total heat load to urban/rural
    # note, this is different than pypsa-eur implementation, as we add all load before
    # clustering; we are not adding load here, rather just splitting it up
    _split_urban_rural_load(n, sector, pop_layout)

    # add heat pumps
    for heat_system in heat_systems:

        name_type = "central"  # no district heating (see PyPSA-Eur for how to add)

        heat_pump_type = "air" if heat_system == "urban" else "ground"

        cop = {"air": ashp_cop, "ground": gshp_cop}

        efficiency = cop[heat_pump_type]

        add_service_heat_pumps(
            n,
            sector,
            heat_system,
            name_type,
            heat_pump_type,
            costs,
            efficiency,
        )

        add_service_gas_furnaces(n, sector, heat_system, costs)

        add_service_elec_furnaces(n, sector, heat_system, costs)


def add_service_cooling(
    n: pypsa.Network,
    sector: str,
    costs: pd.DataFrame,
    **kwargs,
):

    assert sector in ("res", "com")

    plotting = kwargs.get("plotting", None)

    add_air_cons(n, sector, costs)


def add_air_cons(
    n: pypsa.Network,
    sector: str,
    costs: pd.DataFrame,
) -> None:
    """
    Adds gas furnaces to the system.
    """

    assert sector in ("com", "res")

    if sector == "res":
        costs_name = "Residential Central Air Conditioner"
    elif sector == "com":
        costs_name = "Rooftop Air Conditioners"

    capex = costs.at[costs_name, "capex"]
    efficiency = costs.at[costs_name, "efficiency"]
    lifetime = costs.at[costs_name, "lifetime"]

    carrier_name = f"{sector}-cool"

    loads = n.loads[
        (n.loads.carrier == carrier_name) & (n.loads.bus.str.contains("urban"))
    ]

    acs = pd.DataFrame(index=loads.bus)
    acs["bus0"] = acs.index
    acs["bus1"] = acs.index.map(lambda x: f"{x} {sector}-cool")
    acs["carrier"] = carrier_name

    n.madd(
        "Link",
        acs.index,
        suffix=f" {sector} air-con",
        bus0=acs.bus0,
        bus1=acs.bus1,
        carrier=acs.carrier,
        efficiency=efficiency,
        capital_cost=capex,
        p_nom_extendable=True,
        lifetime=lifetime,
    )


def _split_urban_rural_load(
    n: pypsa.Network,
    sector: str,
    ratios: pd.DataFrame,
) -> None:
    """
    Splits a combined load into urban/rural loads.

    Takes a load (such as "p600 0 com-heat") and converts it into two
    loads ("p600 0 com-urban-heat" and "p600 0 com-rural-heat"). The
    buses for these loads are also added (under the name, for example
    "p600 0 com-urban-heat" and "p600 0 com-rural-heat" at the same
    location as "p600 0").
    """

    assert sector in ("com", "res")

    load_names = n.loads[n.loads.carrier == f"{sector}-heat"].index.to_list()

    for system in ("urban", "rural"):

        # add buses to connect the new loads to
        new_buses = pd.DataFrame(index=load_names)
        new_buses.index = new_buses.index.map(n.loads.buses)
        new_buses["x"] = new_buses.index.map(n.buses.x)
        new_buses["y"] = new_buses.index.map(n.buses.y)
        new_buses["country"] = new_buses.index.map(n.buses.country)
        new_buses["interconnect"] = new_buses.index.map(n.buses.interconnect)
        new_buses["STATE"] = new_buses.index.map(n.buses.STATE)
        new_buses["STATE_NAME"] = new_buses.index.map(n.buses.STATE_NAME)

        n.madd(
            "Bus",
            new_buses.index,
            suffix=f" {sector}-{system}-heat",
            x=new_buses.buses.x,
            y=new_buses.buses.y,
            carrier=f"{sector}-heat",
            country=new_buses.buses.country,
            interconnect=new_buses.buses.interconnect,
            STATE=new_buses.buses.STATE,
            STATE_NAME=new_buses.buses.STATE_NAME,
        )

        # add system specific loads to the network
        loads_t = n.loads_t.p_set[load_names]
        loads_t.columns = loads_t.columns.map(n.loads.bus)
        loads_t = loads_t.mul(ratios[f"{system}_fraction"]).rename(
            columns={x: f"{x} {sector}-{system}-heat" for x in loads_t.columns},
        )

        n.madd(
            "Load",
            new_buses.index,
            bus=new_buses.index + f" {sector}-{system}-heat",
            p_set=loads_t[new_buses.index],
            suffix=f" {sector}-{system}-heat",
            carrier=f"{sector}-heat",
        )

    # remove old combined loads from the network
    n.mremove("Load", new_buses.index)


def add_service_gas_furnaces(
    n: pypsa.Network,
    sector: str,
    heat_system: str,
    costs: pd.DataFrame,
) -> None:
    """
    Adds gas furnaces to the system.

    n: pypsa.Network
    sector: str
        ("com" or "res")
    heat_system: str
        ("rural" or "urban")
    costs: pd.DataFrame
    """

    assert sector in ("com", "res")
    assert heat_system in ("urban", "rural")

    if sector == "res":
        costs_name = "Residential Gas-Fired Furnaces"
    elif sector == "com":
        costs_name = "Commercial Gas-Fired Furnaces"

    capex = costs.at[costs_name, "capex"]
    efficiency = costs.at[costs_name, "efficiency"]
    lifetime = costs.at[costs_name, "lifetime"]

    carrier_name = f"{sector}-heat"

    loads = n.loads[
        (n.loads.carrier == carrier_name) & (n.loads.bus.str.contains(heat_system))
    ]

    furnaces = pd.DataFrame(index=loads.bus)
    furnaces["bus0"] = furnaces.index.map(n.buses.STATE)
    furnaces["bus0"] = furnaces.index.map(lambda x: f"{x} gas")
    furnaces["bus1"] = furnaces.index.map(lambda x: f"{x} {sector}-{heat_system}")
    furnaces["carrier"] = furnaces.index.map(lambda x: f"{x} {sector}-gas")

    n.madd(
        "Link",
        furnaces.index,
        suffix=f" {sector} gas furnace",
        bus0=furnaces.bus0,
        bus1=furnaces.bus1,
        carrier=furnaces.carrier,
        efficiency=efficiency,
        capital_cost=capex,
        p_nom_extendable=True,
        lifetime=lifetime,
    )


def add_service_elec_furnaces(
    n: pypsa.Network,
    sector: str,
    heat_system: str,
    costs: pd.DataFrame,
) -> None:
    """
    Adds gas furnaces to the system.

    n: pypsa.Network
    sector: str
        ("com" or "res")
    heat_system: str
        ("rural" or "urban")
    costs: pd.DataFrame
    """

    assert sector in ("com", "res")
    assert heat_system in ("urban", "rural")

    if sector == "res":
        costs_name = "Residential Electric Resistance Heaters"
    elif sector == "com":
        costs_name = "Commercial Electric Resistance Heaters"

    capex = costs.at[costs_name, "capex"]
    efficiency = costs.at[costs_name, "efficiency"]
    lifetime = costs.at[costs_name, "lifetime"]

    carrier_name = f"{sector}-heat"

    loads = n.loads[
        (n.loads.carrier == carrier_name) & (n.loads.bus.str.contains(heat_system))
    ]

    furnaces = pd.DataFrame(index=loads.bus)
    furnaces["bus0"] = furnaces.index.map(n.buses.country)
    furnaces["bus1"] = furnaces.index.map(lambda x: f"{x} {sector}-{heat_system}")
    furnaces["carrier"] = "AC"

    n.madd(
        "Link",
        furnaces.index,
        suffix=f" {sector} elec furnace",
        bus0=furnaces.bus0,
        bus1=furnaces.bus1,
        carrier=furnaces.carrier,
        efficiency=efficiency,
        capital_cost=capex,
        p_nom_extendable=True,
        lifetime=lifetime,
    )


def add_service_heat_pumps(
    n: pypsa.Network,
    sector: str,
    heat_system: str,
    name_type: str,
    hp_type: str,
    costs: pd.DataFrame,
    cop: Optional[pd.DataFrame] = None,
) -> None:
    """
    Adds heat pumps to the system.

    n: pypsa.Network
    sector: str
        ("com" or "res")
    heat_system: str
        ("rural" or "urban")
    name_type: str
        ("central" or "decentral")
    heat_pump_type: str
        ("air" or "ground")
    costs: pd.DataFrame
    cop: pd.DataFrame
        If not provided, uses eff in costs
    """

    hp_type = hp_type.capitalize()

    assert sector in ("com", "res")
    assert hp_type in ("Air", "Ground")
    assert heat_system in ("urban", "rural")

    carrier_name = f"{sector}-heat"

    if sector == "res":
        costs_name = f"Residential {hp_type}-Sourced Heat Pump"
    elif sector == "com":
        costs_name = f"Commercial {hp_type}-Sourced Heat Pump"

    loads = n.loads[
        (n.loads.carrier == carrier_name) & (n.loads.bus.str.contains(heat_system))
    ]

    hps = pd.DataFrame(index=loads.bus)
    hps["bus0"] = hps.index.map(lambda x: f"{x} {sector}")
    hps["bus1"] = hps.index.map(lambda x: f"{x} {sector}-{heat_system.lower()}")
    hps["carrier"] = hps.index.map(lambda x: f"{x} {sector}-{heat_system.lower()}")

    if isinstance(cop, pd.DataFrame):
        efficiency = cop[loads.index.to_list()]
    else:
        efficiency = costs.at[costs_name, "efficiency"]

    n.madd(
        "Link",
        hps.index,
        suffix=f" {sector} {hp_type.lower()} heat pump",
        bus0=hps.bus0,
        bus1=hps.bus1,
        carrier=hps.carrier,
        efficiency=efficiency,
        capital_cost=costs.at[costs_name, "efficiency"] * costs.at[costs_name, "fixed"],
        p_nom_extendable=True,
        lifetime=costs.at[costs_name, "lifetime"],
    )


def add_industrial_furnace(n: pypsa.Network, costs: pd.DataFrame) -> None:

    sector = "ind"

    # Estimates found online, and need to update!
    capex = 0.04  # $ / BTU ######################################### UPDATE!!
    efficiency = 0.80
    lifetime = 25

    carrier_name = f"{sector}-heat"

    loads = n.loads[(n.loads.carrier == carrier_name)]

    furnaces = pd.DataFrame(index=loads.bus)
    furnaces["bus0"] = furnaces.index.map(n.buses.STATE)
    furnaces["bus0"] = furnaces.index.map(lambda x: f"{x} gas")
    furnaces["bus1"] = furnaces.index.map(lambda x: f"{x} {sector}")
    furnaces["carrier"] = furnaces.index.map(lambda x: f"{x} {sector}-gas")

    n.madd(
        "Link",
        furnaces.index,
        suffix=f" {sector} gas furnace",
        bus0=furnaces.bus0,
        bus1=furnaces.bus1,
        carrier=furnaces.carrier,
        efficiency=efficiency,
        capital_cost=capex,
        p_nom_extendable=True,
        lifetime=lifetime,
    )


def add_industrial_boiler(n: pypsa.Network, costs: pd.DataFrame) -> None:

    sector = "ind"

    # Estimates found online, and need to update!
    capex = 0.04  # $ / BTU ######################################### UPDATE!!
    efficiency = 0.80
    lifetime = 25

    carrier_name = f"{sector}-heat"

    loads = n.loads[(n.loads.carrier == carrier_name)]

    furnaces = pd.DataFrame(index=loads.bus)
    furnaces["bus0"] = furnaces.index.map(n.buses.STATE)
    furnaces["bus0"] = furnaces.index.map(lambda x: f"{x} coal")
    furnaces["bus1"] = furnaces.index.map(lambda x: f"{x} {sector}")
    furnaces["carrier"] = furnaces.index.map(lambda x: f"{x} {sector}-coal")

    n.madd(
        "Link",
        furnaces.index,
        suffix=f" {sector} coal boiler",
        bus0=furnaces.bus0,
        bus1=furnaces.bus1,
        carrier=furnaces.carrier,
        efficiency=efficiency,
        capital_cost=capex,
        p_nom_extendable=True,
        lifetime=lifetime,
    )
