"""
Module for building heating and cooling infrastructure.
"""

from typing import Optional

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from add_electricity import load_costs
from constants import NG_MWH_2_MMCF, STATE_2_CODE, COAL_dol_ton_2_MWHthermal
from eia import FuelCosts


def build_heat(
    n: pypsa.Network,
    costs: pd.DataFrame,
    pop_layout_path: str,
    cop_ashp_path: str,
    cop_gshp_path: str,
    dynamic_pricing: bool = False,
    eia: Optional[str] = None,  # for dynamic pricing
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

    for sector in ("res", "com", "ind"):

        if sector in ("res", "com"):

            if dynamic_pricing:
                assert eia
                gas_costs = _get_dynamic_marginal_costs(n, "gas", eia, sector=sector)
                heating_oil_costs = _get_dynamic_marginal_costs(n, "heating_oil", eia)
            else:
                gas_costs = costs.at["gas", "fuel"]
                heating_oil_costs = costs.at["oil", "fuel"]

            # NOTE: Cooling MUST come first, as HPs attach to cooling buses
            add_service_cooling(n, sector, pop_layout, costs)
            add_service_heat(
                n,
                sector,
                pop_layout,
                costs,
                ashp_cop=ashp_cop,
                gshp_cop=gshp_cop,
                marginal_gas=gas_costs,
                marginal_oil=heating_oil_costs,
            )

            assert not n.links_t.p_set.isna().any().any()

        elif sector == "ind":

            if dynamic_pricing:
                assert eia
                gas_costs = _get_dynamic_marginal_costs(n, "gas", eia, sector=sector)
                coal_costs = _get_dynamic_marginal_costs(n, "coal", eia)
            else:
                gas_costs = costs.at["gas", "fuel"]
                coal_costs = costs.at["coal", "fuel"]

            add_industrial_heat(
                n,
                sector,
                costs,
                marginal_gas=gas_costs,
                marginal_coal=coal_costs,
            )


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


def _get_dynamic_marginal_costs(
    n: pypsa.Network,
    fuel: str,
    eia: str,
    sector: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Gets end-use fuel costs at a state level.
    """

    sector_mapper = {
        "res": "residential",
        "com": "commercial",
        "pwr": "power",
        "ind": "industrial",
        "trn": "transport",
    }

    assert fuel in ("gas", "lpg", "coal", "heating_oil")

    year = n.investment_periods[0]

    match fuel:
        case "gas":
            assert sector in ("res", "com", "ind", "pwr")
            raw = (
                FuelCosts(fuel, year, eia, industry=sector_mapper[sector]).get_data(
                    pivot=True,
                )
                * 1000
                / NG_MWH_2_MMCF
            )  # $/MCF -> $/MWh
        case "coal":
            raw = (
                FuelCosts(fuel, year, eia, industry="power").get_data(pivot=True)
                * COAL_dol_ton_2_MWHthermal
            )  # $/Ton -> $/MWh
        case "lpg":
            # https://afdc.energy.gov/fuels/properties
            btu_per_gallon = 112000
            wh_per_btu = 0.29307
            raw = (
                FuelCosts(fuel, year, eia, grade="total").get_data(pivot=True)
                * (1 / btu_per_gallon)
                * (1 / wh_per_btu)
                * (1000000)
            )  # $/gal -> $/MWh
        case "heating_oil":
            # https://www.eia.gov/energyexplained/units-and-calculators/british-thermal-units.php
            btu_per_gallon = 138500
            wh_per_btu = 0.29307
            raw = (
                FuelCosts("heating_oil", year, eia).get_data(pivot=True)
                * (1 / btu_per_gallon)
                * (1 / wh_per_btu)
                * (1000000)
            )  # $/gal -> $/MWh
        case _:
            raise NotImplementedError

    # may have to convert full state name to abbreviated state name
    # should probably change the EIA module to be consistent on what it returns...
    raw = raw.rename(columns=STATE_2_CODE)

    raw.index = pd.DatetimeIndex(raw.index)

    hourly_index = pd.date_range(
        start=f"{year}-01-01",
        end=f"{year}-12-31 23:00:00",
        freq="H",
    )

    # need ffill and bfill as some data is not provided at the resolution or
    # timeframe required
    costs_hourly = raw.reindex(hourly_index)
    costs_hourly = costs_hourly.ffill().bfill()

    return costs_hourly[costs_hourly.index.isin(n.snapshots.get_level_values(1))]


def get_link_marginal_costs(
    n: pypsa.Network,
    links: pd.DataFrame,
    dynamic_costs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Gets dynamic marginal costs dataframe to add to the system.
    """
    assert len(dynamic_costs) == len(n.snapshots.get_level_values(1))

    if "USA" not in dynamic_costs.columns:
        dynamic_costs["USA"] = dynamic_costs.mean(axis=1)
    mc = pd.DataFrame(index=dynamic_costs.index)

    # there is probably a way to vectorize this operation. But right now
    # its fast enough
    for link in links.index:
        try:
            mc[link] = dynamic_costs[links.at[link, "state"]]
        except KeyError:  # USA average
            mc[link] = dynamic_costs["USA"]

    mc.index.name = "timestep"

    # reindex for multi-period investment
    dfs = []
    for year in n.investment_periods:
        df = mc.copy()
        df["period"] = year
        df = df.set_index(["period", df.index])
        dfs.append(df)

    return pd.concat(dfs)


def add_industrial_heat(
    n: pypsa.Network,
    sector: str,
    costs: pd.DataFrame,
    marginal_gas: Optional[pd.DataFrame | float] = None,
    marginal_coal: Optional[pd.DataFrame | float] = None,
    **kwargs,
) -> None:

    assert sector == "ind"

    add_industrial_furnace(n, costs, marginal_gas)
    add_industrial_boiler(n, costs, marginal_coal)
    add_indusrial_heat_pump(n, costs)


def add_service_heat(
    n: pypsa.Network,
    sector: str,
    pop_layout: pd.DataFrame,
    costs: pd.DataFrame,
    ashp_cop: Optional[pd.DataFrame] = None,
    gshp_cop: Optional[pd.DataFrame] = None,
    marginal_gas: Optional[pd.DataFrame | float] = None,
    marginal_oil: Optional[pd.DataFrame | float] = None,
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
    _split_urban_rural_load(n, sector, "heat", pop_layout)

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

        add_service_gas_furnaces(n, sector, heat_system, costs, marginal_gas)

        add_service_lpg_furnaces(n, sector, heat_system, costs, marginal_oil)

        add_service_elec_furnaces(n, sector, heat_system, costs)

        add_service_heat_stores(n, sector, heat_system, costs)


def add_service_cooling(
    n: pypsa.Network,
    sector: str,
    pop_layout: pd.DataFrame,
    costs: pd.DataFrame,
    **kwargs,
):

    assert sector in ("res", "com")

    heat_systems = ("rural", "urban")

    # seperates total heat load to urban/rural
    # note, this is different than pypsa-eur implementation, as we add all load before
    # clustering; we are not adding load here, rather just splitting it up
    _split_urban_rural_load(n, sector, "cool", pop_layout)

    # add heat pumps
    for heat_system in heat_systems:

        add_air_cons(n, sector, heat_system, costs)


def add_air_cons(
    n: pypsa.Network,
    sector: str,
    heat_system: str,
    costs: pd.DataFrame,
) -> None:
    """
    Adds gas furnaces to the system.
    """

    assert heat_system in ("urban", "rural")

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

    carrier_name = f"{sector}-{heat_system}-cool"

    loads = n.loads[
        (n.loads.carrier == carrier_name) & (n.loads.bus.str.contains(heat_system))
    ]

    acs = pd.DataFrame(index=loads.bus)
    acs["bus0"] = acs.index.map(lambda x: x.split(f" {sector}-{heat_system}-cool")[0])
    acs["bus1"] = acs.index
    acs["carrier"] = f"{sector}-{heat_system}-air-con"
    acs.index = acs.bus0

    n.madd(
        "Link",
        acs.index,
        suffix=f" {sector}-{heat_system}-air-con",
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
    fuel: str,
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
    assert fuel in ("heat", "cool")

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

        # strip out the 'res-heat' and 'com-heat' to add in 'rural' and 'urban'
        new_buses.index = new_buses.index.str.strip(f" {sector}-{fuel}")

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
            columns={x: x.strip(f" {sector}-{fuel}") for x in loads_t.columns},
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


def add_service_gas_furnaces(
    n: pypsa.Network,
    sector: str,
    heat_system: str,
    costs: pd.DataFrame,
    marginal_cost: Optional[pd.DataFrame | float] = None,
) -> None:
    """
    Adds gas furnaces to the system.

    n: pypsa.Network
    sector: str
        ("com" or "res")
    heat_system: str
        ("rural" or "urban")
    costs: pd.DataFrame
    marginal_cost: Optional[pd.DataFrame | float] = None
        Fuel costs at a state level
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
    furnaces["state"] = furnaces.index.map(n.buses.STATE)
    furnaces["bus0"] = furnaces.index.map(n.buses.STATE).map(lambda x: f"{x} gas")
    furnaces["bus1"] = furnaces.index
    furnaces["carrier"] = f"{sector}-{heat_system}-gas-furnace"
    furnaces.index = furnaces.bus1.map(
        lambda x: x.split(f" {sector}-{heat_system}-heat")[0],
    )
    furnaces["bus2"] = furnaces.index.map(n.buses.STATE) + f" {sector}-co2"
    furnaces["efficiency2"] = costs.at["gas", "co2_emissions"]

    if isinstance(marginal_cost, pd.DataFrame):
        assert "state" in furnaces.columns
        mc = get_link_marginal_costs(n, furnaces, marginal_cost)
    elif isinstance(marginal_cost, (int, float)):
        mc = marginal_cost
    elif isinstance(marginal_cost, None):
        mc = 0
    else:
        raise TypeError

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
        marginal_cost=mc,
    )


def add_service_lpg_furnaces(
    n: pypsa.Network,
    sector: str,
    heat_system: str,
    costs: pd.DataFrame,
    marginal_cost: Optional[pd.DataFrame | float] = None,
) -> None:
    """
    Adds oil furnaces to the system.

    n: pypsa.Network
    sector: str
        ("com" or "res")
    heat_system: str
        ("rural" or "urban")
    costs: pd.DataFrame
    marginal_cost: Optional[pd.DataFrame | float] = None
        Fuel costs at a state level
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
    furnaces["state"] = furnaces.index.map(n.buses.STATE)
    furnaces["bus0"] = furnaces.index.map(n.buses.STATE).map(lambda x: f"{x} oil")
    furnaces["bus1"] = furnaces.index
    furnaces["carrier"] = f"{sector}-{heat_system}-lpg-furnace"
    furnaces.index = furnaces.bus1.map(
        lambda x: x.split(f" {sector}-{heat_system}-heat")[0],
    )
    furnaces["bus2"] = furnaces.index.map(n.buses.STATE) + f" {sector}-co2"
    furnaces["efficiency2"] = costs.at["oil", "co2_emissions"]

    if isinstance(marginal_cost, pd.DataFrame):
        assert "state" in furnaces.columns
        mc = get_link_marginal_costs(n, furnaces, marginal_cost)
    elif isinstance(marginal_cost, (int, float)):
        mc = marginal_cost
    elif isinstance(marginal_cost, None):
        mc = 0
    else:
        raise TypeError

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
        marginal_cost=mc,
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


def add_industrial_furnace(
    n: pypsa.Network,
    costs: pd.DataFrame,
    marginal_cost: Optional[pd.DataFrame | float] = None,
) -> None:

    sector = "ind"

    capex = costs.at["direct firing gas", "capital_cost"].round(1)
    efficiency = costs.at["direct firing gas", "efficiency"].round(1)
    lifetime = costs.at["direct firing gas", "lifetime"]

    carrier_name = f"{sector}-heat"

    loads = n.loads[(n.loads.carrier == carrier_name)]

    furnaces = pd.DataFrame(index=loads.bus)

    furnaces["state"] = furnaces.index.map(n.buses.STATE)
    furnaces["bus0"] = furnaces.index.map(lambda x: x.split(f" {sector}-heat")[0]).map(
        n.buses.STATE,
    )
    furnaces["bus2"] = furnaces.bus0 + " ind-co2"
    furnaces["bus0"] = furnaces.bus0 + " gas"
    furnaces["bus1"] = furnaces.index
    furnaces["carrier"] = f"{sector}-gas-furnace"
    furnaces.index = furnaces.index.map(lambda x: x.split("-heat")[0])
    furnaces["efficiency2"] = costs.at["gas", "co2_emissions"]

    if isinstance(marginal_cost, pd.DataFrame):
        assert "state" in furnaces.columns
        mc = get_link_marginal_costs(n, furnaces, marginal_cost)
    elif isinstance(marginal_cost, (int, float)):
        mc = marginal_cost
    elif isinstance(marginal_cost, None):
        mc = 0
    else:
        raise TypeError

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
        marginal_cost=mc,
    )


def add_industrial_boiler(
    n: pypsa.Network,
    costs: pd.DataFrame,
    marginal_cost: Optional[pd.DataFrame | float] = None,
) -> None:

    sector = "ind"

    # performance charasteristics taken from (Table 311.1a)
    # https://ens.dk/en/our-services/technology-catalogues/technology-data-industrial-process-heat
    # same source as tech-data, but its just not in latest version

    # capex approximated based on NG to incorporate fixed costs
    capex = costs.at["direct firing gas", "capital_cost"].round(1) * 2.5
    efficiency = 0.90
    lifetime = 25

    carrier_name = f"{sector}-heat"

    loads = n.loads[(n.loads.carrier == carrier_name)]

    boiler = pd.DataFrame(index=loads.bus)
    boiler["state"] = boiler.index.map(n.buses.STATE)
    boiler["bus0"] = boiler.index.map(lambda x: x.split(f" {sector}-heat")[0]).map(
        n.buses.STATE,
    )
    boiler["bus2"] = boiler.bus0 + " ind-co2"
    boiler["bus0"] = boiler.bus0 + " coal"
    boiler["bus1"] = boiler.index
    boiler["carrier"] = f"{sector}-coal-boiler"
    boiler.index = boiler.index.map(lambda x: x.split("-heat")[0])
    boiler["efficiency2"] = costs.at["coal", "co2_emissions"]

    if isinstance(marginal_cost, pd.DataFrame):
        assert "state" in boiler.columns
        mc = get_link_marginal_costs(n, boiler, marginal_cost)
    elif isinstance(marginal_cost, (int, float)):
        mc = marginal_cost
    elif isinstance(marginal_cost, None):
        mc = 0
    else:
        raise TypeError

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
        marginal_cost=mc,
    )


def add_indusrial_heat_pump(
    n: pypsa.Network,
    costs: pd.DataFrame,
) -> None:

    sector = "ind"

    capex = costs.at["industrial heat pump high temperature", "capital_cost"].round(1)
    efficiency = costs.at["industrial heat pump high temperature", "efficiency"].round(
        1,
    )
    lifetime = costs.at["industrial heat pump high temperature", "lifetime"].round(1)

    carrier_name = f"{sector}-heat"

    loads = n.loads[(n.loads.carrier == carrier_name)]

    hp = pd.DataFrame(index=loads.bus)
    hp["state"] = hp.index.map(n.buses.STATE)
    hp["bus0"] = hp.index.map(lambda x: x.split(f" {sector}-heat")[0])
    hp["bus1"] = hp.index
    hp["carrier"] = f"{sector}-heat-pump"
    hp.index = hp.index.map(lambda x: x.split("-heat")[0])

    n.madd(
        "Link",
        hp.index,
        suffix="-heat-pump",  # 'ind' included in index already
        bus0=hp.bus0,
        bus1=hp.bus1,
        carrier=hp.carrier,
        efficiency=efficiency,
        capital_cost=capex,
        p_nom_extendable=True,
        lifetime=lifetime,
    )
