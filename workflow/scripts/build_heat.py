"""
Module for building heating and cooling infrastructure.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from constants import NG_MWH_2_MMCF, STATE_2_CODE, COAL_dol_ton_2_MWHthermal
from constants_sector import SecCarriers, SecNames
from eia import FuelCosts

logger = logging.getLogger(__name__)

VALID_HEAT_SYSTEMS = ("urban", "rural", "total")


def build_heat(
    n: pypsa.Network,
    costs: pd.DataFrame,
    sector: str,
    pop_layout_path: str,
    cop_ashp_path: str,
    cop_gshp_path: str,
    eia: Optional[str] = None,  # for dynamic pricing
    year: Optional[int] = None,  # for dynamic pricing
    options: Optional[dict[str, str | bool | float]] = None,
    **kwargs,
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

    if not options:
        options = {}

    dynamic_costs = options.get("dynamic_costs", False)
    if dynamic_costs:
        assert eia and year, "Must supply EIA API and costs year for dynamic fuel costs"

    demand_response = options.get("demand_response", {})

    if sector in ("res", "com", "srv"):

        split_urban_rural = options.get("split_urban_rural", False)
        technologies = options.get("technologies")
        water_heating_config = options.get("water_heating", {})

        if dynamic_costs:
            # gas_costs = _get_dynamic_marginal_costs(
            #     n,
            #     "gas",
            #     eia,
            #     year,
            #     sector=sector,
            # )
            heating_oil_costs = _get_dynamic_marginal_costs(
                n,
                "heating_oil",
                eia,
                year,
            )
        else:
            # gas_costs = costs.at["gas", "fuel_cost"]
            heating_oil_costs = costs.at["oil", "fuel_cost"]

        # gas costs are endogenous!
        gas_costs = 0

        add_service_heat(
            n=n,
            sector=sector,
            pop_layout=pop_layout,
            costs=costs,
            split_urban_rural=split_urban_rural,
            technologies=technologies,
            ashp_cop=ashp_cop,
            gshp_cop=gshp_cop,
            marginal_gas=gas_costs,
            marginal_oil=heating_oil_costs,
            water_heating_config=water_heating_config,
            demand_response=demand_response,
        )
        add_service_cooling(
            n=n,
            sector=sector,
            pop_layout=pop_layout,
            costs=costs,
            split_urban_rural=split_urban_rural,
            technologies=technologies,
            demand_response=demand_response,
        )

        assert not n.links_t.p_set.isna().any().any()

    elif sector == SecNames.INDUSTRY.value:

        if dynamic_costs:
            gas_costs = _get_dynamic_marginal_costs(
                n,
                "gas",
                eia,
                year,
                sector=sector,
            )
            coal_costs = _get_dynamic_marginal_costs(n, "coal", eia, year)
        else:
            gas_costs = costs.at["gas", "fuel_cost"]
            coal_costs = costs.at["coal", "fuel_cost"]

        # gas costs are endogenous!
        gas_costs = 0

        add_industrial_heat(
            n,
            sector,
            costs,
            marginal_gas=gas_costs,
            marginal_coal=coal_costs,
        )


def combined_heat(n: pypsa.Network, sector: str) -> bool:
    """
    Searches loads for combined or split heat loads.

    Returns:
        True - If only '-heat' is used in load indexing
        False - If '-water-heat' and '-space-heat' is used in load indexing
    """

    assert sector in ("res", "com")

    loads = n.loads.index.to_list()

    water_loads = [x for x in loads if ("-water-heat" in x) and (sector in x)]
    space_loads = [x for x in loads if ("-space-heat" in x) and (sector in x)]
    combined_loads = [x for x in loads if ("-heat" in x) and (sector in x)]

    if water_loads or space_loads:
        assert len(water_loads) + len(space_loads) == len(combined_loads)
        return False
    else:
        return True


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
    year: int,
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

    match fuel:
        case "gas":
            assert sector in ("res", "com", "ind", "pwr")
            if year < 2024:  # get actual monthly values
                raw = FuelCosts(
                    fuel,
                    year,
                    eia,
                    industry=sector_mapper[sector],
                ).get_data(
                    pivot=True,
                )
                raw = raw * 1000 / NG_MWH_2_MMCF  # $/MCF -> $/MWh
            else:  # scale monthly values according to AEO
                act = FuelCosts(
                    fuel,
                    2023,
                    eia,
                    industry=sector_mapper[sector],
                ).get_data(
                    pivot=True,
                )
                proj = FuelCosts(fuel, year, eia, industry=sector_mapper[sector]).get_data(
                    pivot=True,
                )

                actual_year_mean = act.mean().at["U.S."]
                proj_year_mean = proj.at[year, "U.S."]
                scaler = proj_year_mean / actual_year_mean

                raw = act * scaler * 1000 / NG_MWH_2_MMCF  # $/MCF -> $/MWh

        case "coal":
            raw = (
                FuelCosts(fuel, year, eia, industry="power").get_data(pivot=True) * COAL_dol_ton_2_MWHthermal
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

    investment_year = n.investment_periods[0]

    hourly_index = pd.date_range(
        start=f"{year}-01-01",
        end=f"{year}-12-31 23:00:00",
        freq="H",
    )

    # need ffill and bfill as some data is not provided at the resolution or
    # timeframe required
    costs_hourly = raw.reindex(hourly_index)
    costs_hourly = costs_hourly.ffill().bfill()
    costs_hourly.index = costs_hourly.index.map(
        lambda x: x.replace(year=investment_year),
    )

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

    assert sector == SecNames.INDUSTRY.value

    add_industrial_gas_furnace(n, costs, marginal_gas)
    add_industrial_coal_furnace(n, costs, marginal_coal)
    add_indusrial_heat_pump(n, costs)


def add_service_heat(
    n: pypsa.Network,
    sector: str,
    pop_layout: pd.DataFrame,
    costs: pd.DataFrame,
    split_urban_rural: bool,
    technologies: Optional[dict[str, str | bool | float]] = None,
    ashp_cop: Optional[pd.DataFrame] = None,
    gshp_cop: Optional[pd.DataFrame] = None,
    marginal_gas: Optional[pd.DataFrame | float] = None,
    marginal_oil: Optional[pd.DataFrame | float] = None,
    water_heating_config: Optional[dict[str, Any]] = None,
    demand_response: Optional[dict[str, Any]] = None,
):
    """
    Adds heating links for residential and commercial sectors.
    """

    assert sector in ("res", "com", "srv")

    if not technologies:
        technologies = {}
    space_heating_techs = technologies.get("space_heating", {})
    water_heating_techs = technologies.get("water_heating", {})
    standing_losses = technologies.get("standing_losses", {})

    if split_urban_rural:
        heat_systems = ("rural", "urban")
        _split_urban_rural_load(n, sector, "space-heat", pop_layout)
        _split_urban_rural_load(n, sector, "water-heat", pop_layout)
    else:
        heat_systems = ["total"]
        _format_total_load(n, sector, "space-heat")
        _format_total_load(n, sector, "water-heat")

    if not water_heating_config:
        water_heating_config = {}

    split_space_water = water_heating_config.get("split_space_water", False)

    heat_carrier = "space-heat" if split_space_water else "heat"

    include_hps = space_heating_techs.get("heat_pump", True)
    include_elec_furnace = space_heating_techs.get("elec_furnace", True)
    include_gas_furnace = space_heating_techs.get("gas_furnace", True)
    include_oil_furnace = space_heating_techs.get("oil_furnace", True)
    include_elec_water_furnace = water_heating_techs.get("elec_water_tank", True)
    include_gas_water_furnace = water_heating_techs.get("gas_water_tank", True)
    include_oil_water_furnace = water_heating_techs.get("oil_water_tank", True)

    standing_loss_space_heat = standing_losses.get("space", 0)
    standing_loss_water_heat = standing_losses.get("water", 0)

    # add heat pumps
    for heat_system in heat_systems:

        if (heat_system in ["urban", "total"]) and include_hps:

            heat_pump_type = "air"

            cop = ashp_cop

            add_service_heat_pumps(
                n,
                sector,
                heat_system,
                heat_carrier,
                heat_pump_type,
                costs,
                cop,
            )

        if (heat_system in ["rural", "total"]) and include_hps:

            heat_pump_type = "ground"

            cop = gshp_cop

            add_service_heat_pumps(
                n,
                sector,
                heat_system,
                heat_carrier,
                heat_pump_type,
                costs,
                cop,
            )

        if include_elec_furnace:
            add_service_furnace(n, sector, heat_system, heat_carrier, "elec", costs)

        if include_gas_furnace:
            add_service_furnace(
                n,
                sector,
                heat_system,
                heat_carrier,
                "gas",
                costs,
                marginal_gas,
            )

        if include_oil_furnace:
            add_service_furnace(
                n,
                sector,
                heat_system,
                heat_carrier,
                "lpg",
                costs,
                marginal_oil,
            )

        if not demand_response:
            dr_shift = 0
        else:
            dr_shift = demand_response.get("shift", 0)

        add_service_heat_stores(
            n=n,
            sector=sector,
            heat_system=heat_system,
            heat_carrier=heat_carrier,
            costs=costs,
            standing_loss=standing_loss_space_heat,
            dr_shift=dr_shift,
        )

        # check if water heat is needed
        if split_space_water:

            simple_storage = water_heating_config.get("simple_storage", False)
            n_hours = water_heating_config.get("n_hours", None)

            elec_extendable = True if include_elec_water_furnace else False
            gas_extendable = True if include_gas_water_furnace else False
            lpg_extendable = True if include_oil_water_furnace else False

            add_service_water_store(
                n=n,
                sector=sector,
                heat_system=heat_system,
                fuel="elec",
                costs=costs,
                standing_loss=standing_loss_water_heat,
                extendable=elec_extendable,
                simple_storage=simple_storage,
                n_hours=n_hours,
            )
            add_service_water_store(
                n=n,
                sector=sector,
                heat_system=heat_system,
                fuel="gas",
                costs=costs,
                marginal_cost=marginal_gas,
                standing_loss=standing_loss_water_heat,
                extendable=gas_extendable,
                simple_storage=simple_storage,
                n_hours=n_hours,
            )
            add_service_water_store(
                n=n,
                sector=sector,
                heat_system=heat_system,
                fuel="lpg",
                costs=costs,
                marginal_cost=marginal_gas,
                standing_loss=standing_loss_water_heat,
                extendable=lpg_extendable,
                simple_storage=simple_storage,
                n_hours=n_hours,
            )


def add_service_cooling(
    n: pypsa.Network,
    sector: str,
    pop_layout: pd.DataFrame,
    costs: pd.DataFrame,
    split_urban_rural: Optional[bool] = True,
    technologies: Optional[dict[str, bool]] = None,
    demand_response: Optional[dict[str, Any]] = None,
    **kwargs,
):

    assert sector in ("res", "com", "srv")

    if not technologies:
        technologies = {}

    if split_urban_rural:
        heat_systems = ("rural", "urban")
        _split_urban_rural_load(n, sector, "cool", pop_layout)
    else:
        heat_systems = ["total"]
        _format_total_load(n, sector, "cool")

    if not demand_response:
        dr_shift = 0
    else:
        dr_shift = demand_response.get("shift", 0)

    # add cooling technologies
    for heat_system in heat_systems:
        if technologies.get("air_con", True):
            add_air_cons(n, sector, heat_system, costs)
        if technologies.get("heat_pump", True):
            add_service_heat_pumps_cooling(n, sector, heat_system, "cool")

        add_service_heat_stores(n, sector, heat_system, "cool", dr_shift)


def add_air_cons(
    n: pypsa.Network,
    sector: str,
    heat_system: str,
    costs: pd.DataFrame,
) -> None:
    """
    Adds gas furnaces to the system.
    """

    assert heat_system in ("urban", "rural", "total")

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

    loads = n.loads[(n.loads.carrier == carrier_name) & (n.loads.bus.str.contains(heat_system))]

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


def add_service_heat_pumps_cooling(
    n: pypsa.Network,
    sector: str,
    heat_system: str,
    heat_carrier: str,
) -> None:
    """
    Adds heat pumps to the system for cooling. These heat pumps copy attributes
    from the heating sector. Custom constraints are added to enforce capacity
    and operational limits to capture heating/cooling behaviour.

    n: pypsa.Network
    sector: str
        ("com" or "res")
    heat_system: str
        ("rural" or "urban")
    """

    assert sector in ("com", "res")
    assert heat_system in ("urban", "rural", "total")
    assert heat_carrier in ["cool"]

    # get list of existing heat based hps
    carriers = [f"{sector}-{heat_system}-ashp", f"{sector}-{heat_system}-gshp"]
    heat_links = n.links[n.links.carrier.isin(carriers) & ~(n.links.carrier.index.str.endswith("existing"))]

    cool_links = heat_links.copy()
    cool_links_cop = n.links_t["efficiency"][cool_links.index]

    index_mapper = {x: x + "-cool" for x in cool_links.index}

    cool_links = cool_links.rename(index=index_mapper)
    cool_links_cop = cool_links_cop.rename(columns=index_mapper)

    cool_links["bus1"] = cool_links["bus0"].map(lambda x: x.split(f" {sector}")[0])  # node code
    cool_links["bus1"] = cool_links["bus1"] + f" {sector}-{heat_system}-{heat_carrier}"

    # carrier_name = f"{sector}-{heat_system}-{heat_carrier}"
    # cool_links["carrier"] = carrier_name

    cool_links["capex"] = 0  # capacity is constrained to match heating hps

    cool_links = cool_links[["bus0", "bus1", "carrier", "capex", "lifetime"]]

    # use suffix to retain COP profiles
    n.madd(
        "Link",
        cool_links.index,
        bus0=cool_links.bus0,
        bus1=cool_links.bus1,
        carrier=cool_links.carrier,
        efficiency=cool_links_cop,
        capital_cost=cool_links.capex,
        p_nom_extendable=True,
        lifetime=cool_links.lifetime,
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

    seperates total heat load to urban/rural note, this is different
    than pypsa-eur implementation, as we add all load before clustering;
    we are not adding load here, rather just splitting it up
    """

    assert sector in ("com", "res")
    assert fuel in ("heat", "cool", "space-heat", "water-heat")

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
    fuel: str,
) -> None:
    """
    Formats load with 'total' prefix to match urban/rural split.
    """

    assert sector in ("com", "res", "srv")
    assert fuel in ("heat", "cool", "space-heat", "water-heat")

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


def add_service_furnace(
    n: pypsa.Network,
    sector: str,
    heat_system: str,
    heat_carrier: str,
    fuel: str,
    costs: pd.DataFrame,
    marginal_cost: Optional[pd.DataFrame | float] = None,
) -> None:
    """
    Adds direct furnace heating to the system.

    Parameters are average of all storage technologies.

    n: pypsa.Network
    sector: str
        ("com" or "res")
    heat_system: str
        ("rural" or "urban")
    heat_carrier: str
        ("heat" or "space-heat")
    costs: pd.DataFrame
    """
    assert heat_system in ("urban", "rural", "total")
    assert heat_carrier in ("heat", "space-heat")

    match sector:
        case "res" | "Res" | "residential" | "Residential":
            if fuel == "lpg":
                costs_name = "Residential Oil-Fired Furnaces"
            elif fuel == "gas":
                costs_name = "Residential Gas-Fired Furnaces"
            elif fuel == "elec":
                costs_name = "Residential Electric Resistance Heaters"
        case "com" | "Com" | "commercial" | "Commercial":
            if fuel == "lpg":
                costs_name = "Commercial Oil-Fired Furnaces"
            elif fuel == "gas":
                costs_name = "Commercial Gas-Fired Furnaces"
            elif fuel == "elec":
                costs_name = "Commercial Electric Resistance Heaters"
        case _:
            raise NotImplementedError

    capex = costs.at[costs_name, "capital_cost"].round(1)
    efficiency = costs.at[costs_name, "efficiency"].round(1)
    lifetime = costs.at[costs_name, "lifetime"]

    carrier_name = f"{sector}-{heat_system}-{heat_carrier}"

    loads = n.loads[(n.loads.carrier == carrier_name) & (n.loads.bus.str.contains(heat_system))]

    df = pd.DataFrame(index=loads.bus)
    df["state"] = df.index.map(n.buses.STATE)
    df["bus1"] = df.index
    if heat_carrier == "heat":
        new_carrier = f"{sector}-{heat_system}-{fuel}-furnace"
    else:
        new_carrier = f"{sector}-{heat_system}-space-{fuel}-furnace"
    df["carrier"] = new_carrier
    df.index = df.bus1.map(
        lambda x: x.split(f" {sector}-{heat_system}-{heat_carrier}")[0],
    )
    df["bus2"] = df.index.map(n.buses.STATE) + f" {sector}-co2"

    if fuel == "elec":
        df["bus0"] = df.index.map(
            lambda x: x.split(f" {sector}-{heat_system}-{heat_carrier}")[0],
        )
    else:
        fuel_name = "oil" if fuel == "lpg" else fuel
        df["bus0"] = df.state + " " + fuel_name
        df["efficiency2"] = costs.at[fuel_name, "co2_emissions"]

    if isinstance(marginal_cost, pd.DataFrame):
        assert "state" in df.columns
        mc = get_link_marginal_costs(n, df, marginal_cost)
    elif isinstance(marginal_cost, (int, float)):
        mc = marginal_cost
    else:
        mc = 0

    if fuel == "elec":
        n.madd(
            "Link",
            df.index,
            suffix=f" {new_carrier}",
            bus0=df.bus0,
            bus1=df.bus1,
            carrier=df.carrier,
            efficiency=efficiency,
            capital_cost=capex,
            p_nom_extendable=True,
            lifetime=lifetime,
        )
    else:
        n.madd(
            "Link",
            df.index,
            suffix=f" {new_carrier}",
            bus0=df.bus0,
            bus1=df.bus1,
            bus2=df.bus2,
            carrier=df.carrier,
            efficiency=efficiency,
            efficiency2=df.efficiency2,
            capital_cost=capex,
            p_nom_extendable=True,
            lifetime=lifetime,
            marginal_cost=mc,
        )


def add_service_heat_stores(
    n: pypsa.Network,
    sector: str,
    heat_system: str,
    heat_carrier: str,
    costs: pd.DataFrame,
    standing_loss: Optional[float] = None,
    dr_shift: Optional[int | float] = None,
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
    heat_carrier: str
        ("heat" or "space-heat")
    costs: pd.DataFrame
    """

    assert heat_system in ("urban", "rural", "total")
    assert heat_carrier in ("heat", "space-heat", "cool")

    # changed stores to be demand response where metrics are exogenously defined

    """
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

    if not standing_loss:
        standing_loss = 0

    if heat_carrier == "heat":

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

    elif heat_carrier == "space-heat":

        capex = 0
        efficiency = 1
        lifetime = np.inf
    """

    capex = 0
    efficiency = 1
    lifetime = np.inf

    carrier_name = f"{sector}-{heat_system}-{heat_carrier}"

    # must be run after rural/urban load split
    buses = n.buses[n.buses.carrier == carrier_name]

    therm_store = pd.DataFrame(index=buses.index)
    therm_store["bus0"] = therm_store.index
    therm_store["bus1"] = therm_store.index + "-store"
    therm_store["x"] = therm_store.index.map(n.buses.x)
    therm_store["y"] = therm_store.index.map(n.buses.y)
    therm_store["carrier"] = f"{sector}-{heat_system}-{heat_carrier}"

    therm_store["p_nom"] = n.loads_t["p_set"][therm_store.index].max().round(2)

    if not dr_shift:
        dr_shift = 0

    # apply shiftable load via p_max_pu
    # first calc the raw max shiftable load per timestep
    # normalize agaist the max load value
    # ie. if shiftable load is 10%
    #   p_max_mu.max() will return a vector of all '0.10' values

    p_max_pu = (
        n.loads_t["p_set"][therm_store.index].mul(dr_shift).div(n.loads_t["p_set"][therm_store.index].max()).round(4)
    )

    n.madd(
        "Bus",
        therm_store.index,
        suffix="-store",
        x=therm_store.x,
        y=therm_store.y,
        carrier=therm_store.carrier,
        unit="MWh",
    )

    # by default, no demand response
    n.madd(
        "Link",
        therm_store.index,
        suffix="-charger",
        bus0=therm_store.bus0,
        bus1=therm_store.bus1,
        efficiency=efficiency,
        carrier=therm_store.carrier,
        p_nom_extendable=False,
        p_nom=therm_store.p_nom,
        p_max_pu=p_max_pu,
    )

    n.madd(
        "Link",
        therm_store.index,
        suffix="-discharger",
        bus0=therm_store.bus1,
        bus1=therm_store.bus0,
        efficiency=efficiency,
        carrier=therm_store.carrier,
        p_nom_extendable=True,
    )

    n.madd(
        "Store",
        therm_store.index,
        bus=therm_store.bus1,
        e_cyclic=True,
        e_nom_extendable=True,
        carrier=therm_store.carrier,
        standing_loss=standing_loss,
        capital_cost=capex,
        lifetime=lifetime,
    )


def add_service_water_store(
    n: pypsa.Network,
    sector: str,
    heat_system: str,
    fuel: str,
    costs: pd.DataFrame,
    marginal_cost: Optional[pd.DataFrame | float] = None,
    standing_loss: Optional[float] = None,
    extendable: Optional[bool] = True,
    simple_storage: Optional[bool] = True,
    n_hours: Optional[int | float] = None,
) -> None:
    """
    Adds end-use water heat storage system.

    Will add a watner-heat-store-bus, a oe-way link to connect the heat-
    store-bus to the water-heat-bus. A store to connect to the heat-store-bus.

    These are non-investable, non-restrictied components. Cost parameters are
    attached to the links going from primary energy carrier to the heat-store-bus.

    simple_storage:
        If False, costs are applied to store and energy flows are directed through
        stores. If True, costs are applied to the discharging link based on
        4hr storage capacity.
    """

    assert sector in ("res", "com")
    assert heat_system in ("urban", "rural", "total")

    heat_carrier = "water-heat"

    carrier_name = f"{sector}-{heat_system}-{heat_carrier}"

    match fuel:
        case "elec":
            if sector == "res":
                cost_name = "Residential Electric-Resistance Storage Water Heaters"
            elif sector == "com":
                cost_name = "Commercial Electric Resistance Storage Water Heaters"
        case "gas":
            if sector == "res":
                cost_name = "Residential Gas-Fired Storage Water Heaters"
            elif sector == "com":
                cost_name = "Commercial Gas-Fired Storage Water Heaters"
        case "lpg":
            if sector == "res":
                cost_name = "Residential Oil-Fired Storage Water Heaters"
            elif sector == "com":
                cost_name = "Commercial Oil-Fired Storage Water Heaters"
        case _:
            raise NotImplementedError

    # must be run after rural/urban load split
    buses = n.buses[n.buses.carrier == carrier_name]

    df = pd.DataFrame(index=buses.index)
    df["state"] = df.index.map(n.buses.STATE)
    df["x"] = df.index.map(n.buses.x)
    df["y"] = df.index.map(n.buses.y)
    df.index = df.index.str.replace("-heat", "")
    df["bus1"] = df.index + f"-{fuel}-heater"
    df["bus2"] = df.index + "-heat"
    df["bus3"] = df.state + f" {sector}-co2"
    df["carrier"] = f"{sector}-{heat_system}-water-{fuel}"

    if fuel == "elec":
        df["bus0"] = df.index.map(
            lambda x: x.split(f" {sector}-{heat_system}-water")[0],
        )
    else:
        fuel_name = "oil" if fuel == "lpg" else fuel
        df["bus0"] = df.state + " " + fuel_name
        efficiency2 = costs.at[fuel_name, "co2_emissions"]

    if isinstance(marginal_cost, pd.DataFrame):
        assert "state" in df.columns
        mc = get_link_marginal_costs(n, df, marginal_cost)
    elif isinstance(marginal_cost, (int, float)):
        mc = marginal_cost
    else:
        mc = 0

    if not standing_loss:
        standing_loss = 0

    if simple_storage:
        if not n_hours:
            logger.info("Setting water storage capacity costs to 2 hours")
            n_hours = 2
        link_capex = costs.at[cost_name, "capital_cost"] / n_hours
        store_capex = 0
    else:
        link_capex = 0
        store_capex = costs.at[cost_name, "capital_cost"]

    buses = df.copy().set_index("bus1")
    n.madd(
        "Bus",
        buses.index,
        x=buses.x,
        y=buses.y,
        carrier=buses.carrier,
        unit="MWh",
    )

    # limitless one directional link from primary energy to water store
    if fuel == "elec":
        n.madd(
            "Link",
            df.index,
            suffix=f"-{fuel}-heater-charger",
            bus0=df.bus0,
            bus1=df.bus1,
            efficiency=1,
            carrier=df.carrier,
            p_nom_extendable=extendable,
            capital_cost=0,
            marginal_cost=mc,
            lifetime=costs.at[cost_name, "lifetime"],
        )
    else:  # emission tracking
        n.madd(
            "Link",
            df.index,
            suffix=f"-{fuel}-heater-charger",
            bus0=df.bus0,
            bus1=df.bus1,
            bus2=df.bus3,
            efficiency=1,
            efficiency2=efficiency2,
            carrier=df.carrier,
            p_nom_extendable=extendable,
            capital_cost=0,
            marginal_cost=mc,
            lifetime=costs.at[cost_name, "lifetime"],
        )

    # limitless one directional link from water store to water demand
    n.madd(
        "Link",
        df.index,
        suffix=f"-{fuel}-heater-discharger",
        bus0=df.bus1,
        bus1=df.bus2,
        efficiency=costs.at[cost_name, "efficiency"],
        carrier=df.carrier,
        p_nom_extendable=extendable,
        capital_cost=link_capex,
    )

    # limitless water store.
    n.madd(
        "Store",
        df.index,
        suffix=f"-{fuel}-heater",
        bus=df.bus1,
        e_cyclic=True,
        e_nom_extendable=extendable,
        carrier=df.carrier,
        standing_loss=standing_loss,
        efficiency=1,
        capital_cost=store_capex,
        lifetime=costs.at[cost_name, "lifetime"],
    )


def add_service_heat_pumps(
    n: pypsa.Network,
    sector: str,
    heat_system: str,
    heat_carrier: str,
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
    heat_carrier: str
        ("heat" or "space-heat")
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
    assert heat_system in ("urban", "rural", "total")
    assert heat_carrier in ("heat", "space-heat")

    carrier_name = f"{sector}-{heat_system}-{heat_carrier}"

    if sector == "res":
        costs_name = f"Residential {hp_type}-Source Heat Pump"
    elif sector == "com":
        if hp_type == "Ground":
            costs_name = "Commercial Ground-Source Heat Pump"
        else:
            costs_name = "Commercial Rooftop Heat Pumps"

    hp_abrev = "ashp" if hp_type == "Air" else "gshp"

    loads = n.loads[(n.loads.carrier == carrier_name) & (n.loads.bus.str.contains(heat_system))]

    hps = pd.DataFrame(index=loads.bus)
    hps["bus0"] = hps.index.map(
        lambda x: x.split(f" {sector}-{heat_system}-{heat_carrier}")[0],
    )
    hps["bus1"] = hps.index
    hps["carrier"] = f"{sector}-{heat_system}-{hp_abrev}"
    hps.index = hps.bus0  # just node name (ie. p480 0)

    if isinstance(cop, pd.DataFrame):
        efficiency = cop[hps.index.to_list()]
    else:
        efficiency = costs.at[costs_name, "efficiency"].round(1)

    capex = costs.at[costs_name, "capital_cost"].round(1)
    lifetime = costs.at[costs_name, "lifetime"]

    if heat_carrier == "space-heat":
        suffix = f" {sector}-{heat_system}-space-{hp_abrev}"
    else:
        suffix = f" {sector}-{heat_system}-{hp_abrev}"

    # use suffix to retain COP profiles
    n.madd(
        "Link",
        hps.index,
        suffix=suffix,
        bus0=hps.bus0,
        bus1=hps.bus1,
        carrier=hps.carrier,
        efficiency=efficiency,
        capital_cost=capex,
        p_nom_extendable=True,
        lifetime=lifetime,
    )


def add_industrial_gas_furnace(
    n: pypsa.Network,
    costs: pd.DataFrame,
    marginal_cost: Optional[pd.DataFrame | float] = None,
) -> None:

    sector = SecNames.INDUSTRY.value

    capex = costs.at["direct firing gas", "capital_cost"].round(1)
    # efficiency = costs.at["direct firing gas", "efficiency"].round(1)
    efficiency = 0.95  # source defaults to 100%
    lifetime = costs.at["direct firing gas", "lifetime"]

    carrier_name = f"{sector}-heat"

    loads = n.loads[(n.loads.carrier == carrier_name)]

    furnaces = pd.DataFrame(index=loads.bus)

    furnaces["state"] = furnaces.index.map(n.buses.STATE)
    furnaces["bus0"] = furnaces.index.map(lambda x: x.split(f" {sector}-heat")[0]).map(
        n.buses.STATE,
    )
    furnaces["bus2"] = furnaces.bus0 + f" {sector}-co2"
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
    else:
        mc = 0

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


def add_industrial_coal_furnace(
    n: pypsa.Network,
    costs: pd.DataFrame,
    marginal_cost: Optional[pd.DataFrame | float] = None,
) -> None:

    sector = SecNames.INDUSTRY.value

    # performance charasteristics taken from (Table 311.1a)
    # https://ens.dk/en/our-services/technology-catalogues/technology-data-industrial-process-heat
    # same source as tech-data, but its just not in latest version

    # capex approximated based on NG to incorporate fixed costs
    capex = costs.at["direct firing coal", "capital_cost"].round(1)
    # efficiency = costs.at["direct firing coal", "efficiency"].round(1)
    efficiency = 0.95  # source defaults to 100%
    lifetime = capex = costs.at["direct firing coal", "lifetime"].round(1)

    carrier_name = f"{sector}-heat"

    loads = n.loads[(n.loads.carrier == carrier_name)]

    furnace = pd.DataFrame(index=loads.bus)
    furnace["state"] = furnace.index.map(n.buses.STATE)
    furnace["bus0"] = furnace.index.map(lambda x: x.split(f" {sector}-heat")[0]).map(
        n.buses.STATE,
    )
    furnace["bus2"] = furnace.bus0 + " ind-co2"
    furnace["bus0"] = furnace.bus0 + " coal"
    furnace["bus1"] = furnace.index
    furnace["carrier"] = f"{sector}-coal-furnace"
    furnace.index = furnace.index.map(lambda x: x.split("-heat")[0])
    furnace["efficiency2"] = costs.at["coal", "co2_emissions"]

    if isinstance(marginal_cost, pd.DataFrame):
        assert "state" in furnace.columns
        mc = get_link_marginal_costs(n, furnace, marginal_cost)
    elif isinstance(marginal_cost, (int, float)):
        mc = marginal_cost
    else:
        mc = 0

    n.madd(
        "Link",
        furnace.index,
        suffix="-coal-furnace",  # 'ind' included in index already
        bus0=furnace.bus0,
        bus1=furnace.bus1,
        bus2=furnace.bus2,
        carrier=furnace.carrier,
        efficiency=efficiency,
        efficiency2=furnace.efficiency2,
        capital_cost=capex,
        p_nom_extendable=True,
        lifetime=lifetime,
        marginal_cost=mc,
    )


def add_indusrial_heat_pump(
    n: pypsa.Network,
    costs: pd.DataFrame,
) -> None:

    sector = SecNames.INDUSTRY.value

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
