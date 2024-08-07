"""
Module for building heating and cooling infrastructure.
"""

from typing import Optional

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from add_electricity import load_costs


def build_heat(
    n: pypsa.Network,
    costs: pd.DataFrame,
    pop_layout_path: str,
    cop_ashp_path: str,
    cop_gshp_path: str,
) -> None:
    """
    Main funtion to interface with.
    """

    sns = n.snapshots

    pop_layout = pd.read_csv(pop_layout_path).set_index("name")

    ashp_cop = xr.open_dataarray(cop_ashp_path)
    gshp_cop = xr.open_dataarray(cop_gshp_path)

    ashp_cop = reindex_cop(sns, ashp_cop)
    gshp_cop = reindex_cop(sns, gshp_cop)

    for sector in ("res", "com"):
        add_service_heat(n, sector, pop_layout, costs, ashp_cop, gshp_cop)
        add_service_cooling(n, sector, costs)

    for sector in ["ind"]:
        add_industrial_heat(n, sector, costs)


def reindex_cop(sns: pd.MultiIndex, da: xr.DataArray) -> pd.DataFrame:
    """
    Reindex a COP dataarray.

    This will allign snapshots to match the planning horizon. This will
    also calcualte the mean COP for each period if tsa has occured
    """

    cop = da.to_pandas()
    investment_years = sns.get_level_values(0).unique()

    # use first investment year, as weather profiles dont change between years
    cop.index = cop.index.map(lambda x: x.replace(year=investment_years[0]))

    # need to account for tsa with variable lengths per snapshot
    sns_per_year = sns.get_level_values(1).unique()
    groups = {snapshot: group for group, snapshot in enumerate(sns_per_year)}
    cop["group"] = cop.index.map(groups)
    cop["group"] = cop.group.ffill().astype(int)
    cop = cop.groupby("group").mean()
    cop.index = sns_per_year

    # index to match network multiindex
    cops = []
    for investment_year in investment_years:
        cop_investment_year = cop.copy()
        cop_investment_year.index = cop_investment_year.index.map(
            lambda x: x.replace(year=investment_year),
        )
        cop_investment_year["year"] = investment_year
        cops.append(cop_investment_year.set_index(["year", cop_investment_year.index]))

    return pd.concat(cops).reindex(sns)


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

        # if heat_system == "urban":
        #     add_service_gas_furnaces(n, sector, heat_system, costs)

        add_service_gas_furnaces(n, sector, heat_system, costs)

        add_service_lpg_furnaces(n, sector, heat_system, costs)

        add_service_elec_furnaces(n, sector, heat_system, costs)

        add_service_heat_stores(n, sector, heat_system, costs)


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

    match sector:
        case "res" | "Res" | "residential" | "Residential":
            costs_name = "Residential Central Air Conditioner"
        case "com" | "Com" | "commercial" | "Commercial":
            costs_name = "Commercial Rooftop Air Conditioners"
        case _:
            raise NotImplementedError

    capex = costs.at[costs_name, "capital_cost"].round(1)
    efficiency = costs.at[costs_name, "efficiency"].round(1)
    lifetime = costs.at[costs_name, "lifetime"]

    carrier_name = f"{sector}-cool"

    loads = n.loads[n.loads.carrier == carrier_name]

    acs = pd.DataFrame(index=loads.bus)
    acs["bus0"] = acs.index.map(lambda x: x.split(f" {sector}-cool")[0])
    acs["bus1"] = acs.index
    acs["carrier"] = f"{sector}-air-con"
    acs.index = acs.bus0

    n.madd(
        "Link",
        acs.index,
        suffix=f" {sector}-air-con",
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
        new_buses.index = new_buses.index.map(n.loads.bus)
        new_buses["x"] = new_buses.index.map(n.buses.x)
        new_buses["y"] = new_buses.index.map(n.buses.y)
        new_buses["country"] = new_buses.index.map(n.buses.country)
        new_buses["interconnect"] = new_buses.index.map(n.buses.interconnect)
        new_buses["STATE"] = new_buses.index.map(n.buses.STATE)
        new_buses["STATE_NAME"] = new_buses.index.map(n.buses.STATE_NAME)

        # strip out the 'res-heat' and 'com-heat' to add in 'rural' and 'urban'
        new_buses.index = new_buses.index.str.strip(f" {sector}-heat")

        n.madd(
            "Bus",
            new_buses.index,
            suffix=f" {sector}-{system}-heat",
            x=new_buses.x,
            y=new_buses.y,
            carrier=f"{sector}-{system}-heat",
            country=new_buses.country,
            interconnect=new_buses.interconnect,
            STATE=new_buses.STATE,
            STATE_NAME=new_buses.STATE_NAME,
        )

        # get rural or urban loads
        loads_t = n.loads_t.p_set[load_names]
        loads_t = loads_t.rename(
            columns={x: x.strip(f" {sector}-heat") for x in loads_t.columns},
        )
        loads_t = loads_t.mul(ratios[f"{system}_fraction"])

        n.madd(
            "Load",
            new_buses.index,
            suffix=f" {sector}-{system}-heat",
            bus=new_buses.index + f" {sector}-{system}-heat",
            p_set=loads_t,
            carrier=f"{sector}-{system}-heat",
        )

    # remove old combined loads from the network
    n.mremove("Load", load_names)
    n.mremove("Bus", load_names)


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

    assert heat_system in ("urban", "rural")

    match sector:
        case "res" | "Res" | "residential" | "Residential":
            costs_name = "Residential Gas-Fired Furnaces"
        case "com" | "Com" | "commercial" | "Commercial":
            costs_name = "Commercial Gas-Fired Furnaces"
        case _:
            raise NotImplementedError

    capex = costs.at[costs_name, "capital_cost"].round(1)
    efficiency = costs.at[costs_name, "efficiency"].round(1)
    lifetime = costs.at[costs_name, "lifetime"]

    carrier_name = f"{sector}-{heat_system}-heat"

    loads = n.loads[
        (n.loads.carrier == carrier_name) & (n.loads.bus.str.contains(heat_system))
    ]

    furnaces = pd.DataFrame(index=loads.bus)
    furnaces["bus0"] = furnaces.index.map(n.buses.STATE).map(lambda x: f"{x} gas")
    furnaces["bus1"] = furnaces.index
    furnaces["carrier"] = f"{sector}-{heat_system}-gas-furnace"
    furnaces.index = furnaces.bus1.map(
        lambda x: x.split(f" {sector}-{heat_system}-heat")[0],
    )
    furnaces["bus2"] = furnaces.index.map(n.buses.STATE) + f" {sector}-co2"
    furnaces["efficiency2"] = costs.at["gas", "co2_emissions"]

    n.madd(
        "Link",
        furnaces.index,
        suffix=f" {sector}-{heat_system}-gas-furnace",
        bus0=furnaces.bus0,
        bus1=furnaces.bus1,
        bus2=furnaces.bus2,
        carrier=furnaces.carrier,
        efficiency=efficiency,
        efficiency2=furnaces.efficiency2,
        capital_cost=capex,
        p_nom_extendable=True,
        lifetime=lifetime,
    )


def add_service_lpg_furnaces(
    n: pypsa.Network,
    sector: str,
    heat_system: str,
    costs: pd.DataFrame,
) -> None:
    """
    Adds oil furnaces to the system.

    n: pypsa.Network
    sector: str
        ("com" or "res")
    heat_system: str
        ("rural" or "urban")
    costs: pd.DataFrame
    """

    assert heat_system in ("urban", "rural")

    match sector:
        case "res" | "Res" | "residential" | "Residential":
            costs_name = "Residential Oil-Fired Furnaces"
        case "com" | "Com" | "commercial" | "Commercial":
            costs_name = "Commercial Oil-Fired Furnaces"
        case _:
            raise NotImplementedError

    capex = costs.at[costs_name, "capital_cost"].round(1)
    efficiency = costs.at[costs_name, "efficiency"].round(1)
    lifetime = costs.at[costs_name, "lifetime"]

    carrier_name = f"{sector}-{heat_system}-heat"

    loads = n.loads[
        (n.loads.carrier == carrier_name) & (n.loads.bus.str.contains(heat_system))
    ]

    furnaces = pd.DataFrame(index=loads.bus)
    furnaces["bus0"] = furnaces.index.map(n.buses.STATE).map(lambda x: f"{x} oil")
    furnaces["bus1"] = furnaces.index
    furnaces["carrier"] = f"{sector}-{heat_system}-lpg-furnace"
    furnaces.index = furnaces.bus1.map(
        lambda x: x.split(f" {sector}-{heat_system}-heat")[0],
    )
    furnaces["bus2"] = furnaces.index.map(n.buses.STATE) + f" {sector}-co2"
    furnaces["efficiency2"] = costs.at["oil", "co2_emissions"]

    n.madd(
        "Link",
        furnaces.index,
        suffix=f" {sector}-{heat_system}-lpg-furnace",
        bus0=furnaces.bus0,
        bus1=furnaces.bus1,
        bus2=furnaces.bus2,
        carrier=furnaces.carrier,
        efficiency=efficiency,
        efficiency2=furnaces.efficiency2,
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

    assert heat_system in ("urban", "rural")

    match sector:
        case "res" | "Res" | "residential" | "Residential":
            costs_name = "Residential Electric Resistance Heaters"
        case "com" | "Com" | "commercial" | "Commercial":
            costs_name = "Commercial Electric Resistance Heaters"
        case _:
            raise NotImplementedError

    capex = costs.at[costs_name, "capital_cost"].round(1)
    efficiency = costs.at[costs_name, "efficiency"].round(1)
    lifetime = costs.at[costs_name, "lifetime"]

    carrier_name = f"{sector}-{heat_system}-heat"

    loads = n.loads[
        (n.loads.carrier == carrier_name) & (n.loads.bus.str.contains(heat_system))
    ]

    furnaces = pd.DataFrame(index=loads.bus)
    furnaces["bus0"] = furnaces.index.map(
        lambda x: x.split(f" {sector}-{heat_system}-heat")[0],
    )
    furnaces["bus1"] = furnaces.index
    furnaces["carrier"] = f"{sector}-{heat_system}-elec-furnace"
    furnaces.index = furnaces.bus0

    n.madd(
        "Link",
        furnaces.index,
        suffix=f" {sector}-{heat_system}-elec-furnace",
        bus0=furnaces.bus0,
        bus1=furnaces.bus1,
        carrier=furnaces.carrier,
        efficiency=efficiency,
        capital_cost=capex,
        p_nom_extendable=True,
        lifetime=lifetime,
    )


def add_service_heat_stores(
    n: pypsa.Network,
    sector: str,
    heat_system: str,
    costs: pd.DataFrame,
) -> None:
    """
    Adds end-use thermal storage to the system.

    Will add a heat-store-bus. Two uni-directional links to connect the heat-
    store-bus to the heat-bus. A store to connect to the heat-store-bus.

    Parameters are average of all storage technologies.

    n: pypsa.Network
    sector: str
        ("com" or "res")
    heat_system: str
        ("rural" or "urban")
    costs: pd.DataFrame
    """

    assert heat_system in ("urban", "rural")

    match sector:
        case "res" | "Res" | "residential" | "Residential":
            costs_names = [
                "Residential Gas-Fired Storage Water Heaters",
                "Residential Electric-Resistance Storage Water Heaters",
            ]
        case "com" | "Com" | "commercial" | "Commercial":
            costs_names = [
                "Commercial Electric Resistance Storage Water Heaters",
                "Commercial Gas-Fired Storage Water Heaters",
            ]
        case _:
            raise NotImplementedError

    capex = round(
        sum([costs.at[x, "capital_cost"] for x in costs_names]) / len(costs_names),
        1,
    )
    efficiency = round(
        sum([costs.at[x, "efficiency"] for x in costs_names]) / len(costs_names),
        1,
    )
    lifetime = round(
        sum([costs.at[x, "lifetime"] for x in costs_names]) / len(costs_names),
        1,
    )
    tau = 3 if heat_system == "rural" else 180
    standing_loss = (1 - np.exp(-1 / 24 / tau),)

    carrier_name = f"{sector}-{heat_system}-heat"

    # must be run after rural/urban load split
    buses = n.buses[n.buses.carrier == carrier_name]

    therm_store = pd.DataFrame(index=buses.index)
    therm_store["bus0"] = therm_store.index
    therm_store["bus1"] = therm_store.index + "-store"
    therm_store["x"] = therm_store.index.map(n.buses.x)
    therm_store["y"] = therm_store.index.map(n.buses.y)
    therm_store["carrier"] = f"{sector}-{heat_system}-heat-store"

    n.madd(
        "Bus",
        therm_store.index,
        suffix="-store",
        x=therm_store.x,
        y=therm_store.y,
        carrier=therm_store.carrier,
        unit="MWh",
    )

    n.madd(
        "Link",
        therm_store.index,
        suffix="-store-charger",
        bus0=therm_store.bus0,
        bus1=therm_store.bus1,
        efficiency=efficiency,
        carrier=therm_store.carrier,
        p_nom_extendable=True,
    )

    n.madd(
        "Link",
        therm_store.index,
        suffix="-store-discharger",
        bus0=therm_store.bus1,
        bus1=therm_store.bus0,
        efficiency=1,  # efficiency in first link is round trip
        carrier=therm_store.carrier,
        p_nom_extendable=True,
    )

    n.madd(
        "Store",
        therm_store.index,
        suffix="-store",
        bus=therm_store.bus1,
        e_cyclic=True,
        e_nom_extendable=True,
        carrier=therm_store.carrier,
        standing_loss=standing_loss,
        capital_cost=capex,
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

    carrier_name = f"{sector}-{heat_system}-heat"

    if sector == "res":
        costs_name = f"Residential {hp_type}-Source Heat Pump"
    elif sector == "com":
        if hp_type == "Ground":
            costs_name = "Commercial Ground-Source Heat Pump"
        else:
            costs_name = "Commercial Rooftop Heat Pumps"

    hp_abrev = "ashp" if heat_system == "urban" else "gshp"

    loads = n.loads[
        (n.loads.carrier == carrier_name) & (n.loads.bus.str.contains(heat_system))
    ]

    hps = pd.DataFrame(index=loads.bus)
    hps["bus0"] = hps.index.map(lambda x: x.split(f" {sector}-{heat_system}-heat")[0])
    hps["bus1"] = hps.index
    hps["carrier"] = f"{sector}-{heat_system}-{hp_abrev}"
    hps.index = hps.bus0  # just node name (ie. p480 0)

    if isinstance(cop, pd.DataFrame):
        efficiency = cop[hps.index.to_list()]
    else:
        efficiency = costs.at[costs_name, "efficiency"].round(1)

    capex = costs.at[costs_name, "capital_cost"].round(1)
    lifetime = costs.at[costs_name, "lifetime"]

    n.madd(
        "Link",
        hps.index,
        suffix=f" {sector}-{heat_system}-{hp_abrev}",
        bus0=hps.bus0,
        bus1=hps.bus1,
        carrier=hps.carrier,
        efficiency=efficiency,
        capital_cost=capex,
        p_nom_extendable=True,
        lifetime=lifetime,
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
    furnaces["bus0"] = furnaces.index.map(lambda x: x.split(f" {sector}-heat")[0]).map(
        n.buses.STATE,
    )
    furnaces["bus2"] = furnaces.bus0 + " ind-co2"
    furnaces["bus0"] = furnaces.bus0 + " gas"
    furnaces["bus1"] = furnaces.index
    furnaces["carrier"] = f"{sector}-gas-furnace"
    furnaces.index = furnaces.index.map(lambda x: x.split("-heat")[0])
    furnaces["efficiency2"] = costs.at["gas", "co2_emissions"]

    n.madd(
        "Link",
        furnaces.index,
        suffix="-gas-furnace",  #'ind' included in index already
        bus0=furnaces.bus0,
        bus1=furnaces.bus1,
        bus2=furnaces.bus2,
        carrier=furnaces.carrier,
        efficiency=efficiency,
        efficiency2=furnaces.efficiency2,
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

    boiler = pd.DataFrame(index=loads.bus)
    boiler["bus0"] = boiler.index.map(lambda x: x.split(f" {sector}-heat")[0]).map(
        n.buses.STATE,
    )
    boiler["bus2"] = boiler.bus0 + " ind-co2"
    boiler["bus0"] = boiler.bus0 + " coal"
    boiler["bus1"] = boiler.index
    boiler["carrier"] = f"{sector}-coal-boiler"
    boiler.index = boiler.index.map(lambda x: x.split("-heat")[0])
    boiler["efficiency2"] = costs.at["coal", "co2_emissions"]

    n.madd(
        "Link",
        boiler.index,
        suffix="-coal-boiler",  # 'ind' included in index already
        bus0=boiler.bus0,
        bus1=boiler.bus1,
        bus2=boiler.bus2,
        carrier=boiler.carrier,
        efficiency=efficiency,
        efficiency2=boiler.efficiency2,
        capital_cost=capex,
        p_nom_extendable=True,
        lifetime=lifetime,
    )
