"""Module for building heating and cooling infrastructure."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from constants_sector import SecCarriers, SecNames

logger = logging.getLogger(__name__)

VALID_HEAT_SYSTEMS = ("urban", "rural", "total")


def build_heat(
    n: pypsa.Network,
    costs: pd.DataFrame,
    sector: str,
    pop_layout_path: str,
    cop_ashp_path: str,
    cop_gshp_path: str,
    eia: str | None = None,  # for dynamic pricing
    year: int | None = None,  # for dynamic pricing
    options: dict[str, str | bool | int | float] | None = None,
    **kwargs,
) -> None:
    """Main funtion to interface with."""
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

    dr_config = options.get("demand_response", {})
    dr_config = dr_config.get(sector, dr_config)

    if sector in ("res", "com", "srv"):
        split_urban_rural = options.get("split_urban_rural", False)
        technologies = options.get("technologies")
        water_heating_config = options.get("water_heating", {})

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
            # marginal_oil=heating_oil_costs,
            water_heating_config=water_heating_config,
            dr_config=dr_config,
        )
        add_service_cooling(
            n=n,
            sector=sector,
            pop_layout=pop_layout,
            costs=costs,
            split_urban_rural=split_urban_rural,
            technologies=technologies,
            dr_config=dr_config,
        )

        assert not n.links_t.p_set.isna().any().any()

    elif sector == SecNames.INDUSTRY.value:
        # gas costs are endogenous!
        gas_costs = 0

        # coal costs tracked at state level store
        coal_costs = 0

        add_industrial_heat(
            n,
            sector,
            costs,
            marginal_gas=gas_costs,
            marginal_coal=coal_costs,
            dr_config=dr_config,
        )


def combined_heat(n: pypsa.Network, sector: str) -> bool:
    """
    Searches loads for combined or split heat loads.

    Returns
    -------
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


def get_link_marginal_costs(
    n: pypsa.Network,
    links: pd.DataFrame,
    dynamic_costs: pd.DataFrame,
) -> pd.DataFrame:
    """Gets dynamic marginal costs dataframe to add to the system."""
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
    marginal_gas: pd.DataFrame | float | None = None,
    marginal_coal: pd.DataFrame | float | None = None,
    dr_config: dict[str, Any] | None = None,
    **kwargs,
) -> None:
    assert sector == SecNames.INDUSTRY.value

    add_industrial_gas_furnace(n, costs, marginal_gas)
    add_industrial_coal_furnace(n, costs, marginal_coal)
    add_indusrial_heat_pump(n, costs)

    if dr_config:
        add_heat_dr(n, sector, dr_config)


def add_service_heat(
    n: pypsa.Network,
    sector: str,
    pop_layout: pd.DataFrame,
    costs: pd.DataFrame,
    split_urban_rural: bool,
    technologies: dict[str, str | bool | float] | None = None,
    ashp_cop: pd.DataFrame | None = None,
    gshp_cop: pd.DataFrame | None = None,
    marginal_gas: pd.DataFrame | float | None = None,
    marginal_oil: pd.DataFrame | float | None = None,
    water_heating_config: dict[str, Any] | None = None,
    dr_config: dict[str, Any] | None = None,
):
    """Adds heating links for residential and commercial sectors."""
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
            )

        if include_oil_furnace:
            add_service_furnace(
                n,
                sector,
                heat_system,
                heat_carrier,
                "oil",
                costs,
            )

        if dr_config:
            add_heat_dr(
                n=n,
                sector=sector,
                heat_system=heat_system,
                heat_carrier=heat_carrier,
                dr_config=dr_config,
                standing_loss=standing_loss_space_heat,
            )

        # check if water heat is needed
        if split_space_water:
            simple_storage = water_heating_config.get("simple_storage", False)
            n_hours = water_heating_config.get("n_hours", None)

            elec_extendable = True if include_elec_water_furnace else False
            gas_extendable = True if include_gas_water_furnace else False
            oil_extendable = True if include_oil_water_furnace else False

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
                fuel="oil",
                costs=costs,
                marginal_cost=marginal_gas,
                standing_loss=standing_loss_water_heat,
                extendable=oil_extendable,
                simple_storage=simple_storage,
                n_hours=n_hours,
            )


def add_service_cooling(
    n: pypsa.Network,
    sector: str,
    pop_layout: pd.DataFrame,
    costs: pd.DataFrame,
    split_urban_rural: bool | None = True,
    technologies: dict[str, bool] | None = None,
    dr_config: dict[str, Any] | None = None,
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

    # add cooling technologies
    for heat_system in heat_systems:
        if technologies.get("air_con", True):
            add_air_cons(n, sector, heat_system, costs)
        if technologies.get("heat_pump", True):
            add_service_heat_pumps_cooling(n, sector, heat_system, "cool")

        if dr_config:
            add_heat_dr(
                n=n,
                sector=sector,
                heat_system=heat_system,
                heat_carrier="cool",
                dr_config=dr_config,
            )


def add_air_cons(
    n: pypsa.Network,
    sector: str,
    heat_system: str,
    costs: pd.DataFrame,
) -> None:
    """Adds gas furnaces to the system."""
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
    build_year = n.investment_periods[0]

    carrier_name = f"{sector}-{heat_system}-cool"

    loads = n.loads[(n.loads.carrier == carrier_name) & (n.loads.bus.str.contains(heat_system))]

    acs = pd.DataFrame(index=loads.bus)
    acs["bus0"] = acs.index.map(lambda x: f"{x.split('-cool')[0]}-{SecCarriers.ELECTRICITY.value}")
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
        build_year=build_year,
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

    cool_links["bus1"] = cool_links["bus0"].map(
        lambda x: x.split(f" {sector}")[0],
    )  # node code
    cool_links["bus1"] = cool_links["bus1"] + f" {sector}-{heat_system}-{heat_carrier}"

    # carrier_name = f"{sector}-{heat_system}-{heat_carrier}"
    # cool_links["carrier"] = carrier_name

    cool_links["capex"] = 0  # capacity is constrained to match heating hps

    cool_links = cool_links[["bus0", "bus1", "carrier", "capex", "lifetime"]]

    build_year = n.investment_periods[0]

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
        build_year=build_year,
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
    """Formats load with 'total' prefix to match urban/rural split."""
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

    if sector in ("res", "residential", "Residential"):
        if fuel == "oil":
            costs_name = "Residential Oil-Fired Furnaces"
        elif fuel == "gas":
            costs_name = "Residential Gas-Fired Furnaces"
        elif fuel == "elec":
            costs_name = "Residential Electric Resistance Heaters"
        else:
            raise ValueError(f"Unexpected fuel of {fuel}")
    elif sector in ("com", "commercial", "Commercial"):
        if fuel == "oil":
            costs_name = "Commercial Oil-Fired Furnaces"
        elif fuel == "gas":
            costs_name = "Commercial Gas-Fired Furnaces"
        elif fuel == "elec":
            costs_name = "Commercial Electric Resistance Heaters"
        else:
            raise ValueError(f"Unexpected fuel of {fuel}")
    else:
        raise ValueError(f"Unexpected sector of {sector}")

    capex = costs.at[costs_name, "capital_cost"].round(1)
    efficiency = costs.at[costs_name, "efficiency"].round(1)
    lifetime = costs.at[costs_name, "lifetime"]
    build_year = n.investment_periods[0]

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
        df["bus0"] = df.index.map(lambda x: f"{x} {sector}-{heat_system}-{SecCarriers.ELECTRICITY.value}")
    else:
        df["bus0"] = df.state + " " + fuel
        df["efficiency2"] = costs.at[fuel, "co2_emissions"]

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
            build_year=build_year,
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
            build_year=build_year,
            # marginal_cost=mc,
        )


def add_heat_dr(
    n: pypsa.Network,
    sector: str,
    dr_config: dict[str, Any],
    heat_system: str | None = None,
    heat_carrier: str | None = None,
    standing_loss: float | None = None,
) -> None:
    """Adds end-use thermal demand response."""
    by_carrier = dr_config.get("by_carrier", False)

    # check if dr is applied at a per-carrier level

    if sector in ["res", "com"]:
        assert heat_system in ("urban", "rural", "total")
        assert heat_carrier in ("heat", "space-heat", "cool")

        carrier_name = f"{sector}-{heat_system}-{heat_carrier}"

        if by_carrier:
            dr_config = dr_config.get(heat_carrier, {})

    elif sector == "ind":
        carrier_name = "ind-heat"

        if by_carrier:
            dr_config = dr_config.get("heat", {})

    else:
        raise ValueError(f"{sector} not valid dr option")

    # check if demand response is applied

    shift = dr_config.get("shift", 0)
    if shift == 0:
        logger.info(f"DR not applied to {sector} as allowable sift is {shift}")
        return

    # assign marginal cost value

    marginal_cost_storage = dr_config.get("marginal_cost", 0)
    if marginal_cost_storage == 0:
        logger.warning(f"No cost applied to demand response for {sector}")

    # get components to add
    # MUST BE RUN AFTER URBAN/RURAL SPLIT

    buses = n.buses[n.buses.carrier == carrier_name]

    df = pd.DataFrame(index=buses.index)
    df["x"] = df.index.map(n.buses.x)
    df["y"] = df.index.map(n.buses.y)
    df["carrier"] = carrier_name + "-dr"
    df["STATE"] = df.index.map(n.buses.STATE)
    df["STATE_NAME"] = df.index.map(n.buses.STATE_NAME)

    lifetime = np.inf
    build_year = n.investment_periods[0]

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
        lifetime=lifetime,
        build_year=build_year,
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
        lifetime=lifetime,
        build_year=build_year,
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
        lifetime=lifetime,
        build_year=build_year,
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
        lifetime=lifetime,
        build_year=build_year,
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
        standing_loss=standing_loss,
        marginal_cost_storage=marginal_cost_storage,
        lifetime=lifetime,
        build_year=build_year,
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
        standing_loss=standing_loss,
        marginal_cost_storage=marginal_cost_storage * (-1),
        lifetime=lifetime,
        build_year=build_year,
    )


def add_service_water_store(
    n: pypsa.Network,
    sector: str,
    heat_system: str,
    fuel: str,
    costs: pd.DataFrame,
    marginal_cost: pd.DataFrame | float | None = None,
    standing_loss: float | None = None,
    extendable: bool | None = True,
    simple_storage: bool | None = True,
    n_hours: int | float | None = None,
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

    if fuel == "elec":
        if sector == "res":
            cost_name = "Residential Electric-Resistance Storage Water Heaters"
        elif sector == "com":
            cost_name = "Commercial Electric Resistance Storage Water Heaters"
    elif fuel == "gas":
        if sector == "res":
            cost_name = "Residential Gas-Fired Storage Water Heaters"
        elif sector == "com":
            cost_name = "Commercial Gas-Fired Storage Water Heaters"
    elif fuel == "oil":
        if sector == "res":
            cost_name = "Residential Oil-Fired Storage Water Heaters"
        elif sector == "com":
            cost_name = "Commercial Oil-Fired Storage Water Heaters"
    else:
        raise ValueError(f"Unexpected fuel of {fuel}")

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
        df["bus0"] = df.index.map(lambda x: f"{x.split('-water')[0]}-{SecCarriers.ELECTRICITY.value}")
    else:
        fuel_name = "oil" if fuel == "lpg" else fuel
        df["bus0"] = df.state + " " + fuel_name
        efficiency2 = costs.at[fuel_name, "co2_emissions"]

    if isinstance(marginal_cost, pd.DataFrame):
        assert "state" in df.columns
        mc = get_link_marginal_costs(n, df, marginal_cost)
    elif isinstance(marginal_cost, int | float):
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

    lifetime = (costs.at[cost_name, "lifetime"],)
    build_year = n.investment_periods[0]

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
            lifetime=lifetime,
            build_year=build_year,
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
            lifetime=lifetime,
            build_year=build_year,
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
        lifetime=lifetime,
        build_year=build_year,
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
        lifetime=lifetime,
        build_year=build_year,
    )


def add_service_heat_pumps(
    n: pypsa.Network,
    sector: str,
    heat_system: str,
    heat_carrier: str,
    hp_type: str,
    costs: pd.DataFrame,
    cop: pd.DataFrame | None = None,
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
        lambda x: f"{x.split(f'-{heat_carrier}')[0]}-{SecCarriers.ELECTRICITY.value}",
    )
    hps["bus1"] = hps.index
    hps["carrier"] = f"{sector}-{heat_system}-{hp_abrev}"
    hps.index = hps.index.map(lambda x: x.split(" ")[0])  # just node name (ie. p480 0)

    if heat_carrier == "space-heat":
        suffix = f"{sector}-{heat_system}-space-{hp_abrev}"
    else:
        suffix = f"{sector}-{heat_system}-{hp_abrev}"

    hps.index = hps.index.map(lambda x: f"{x} {suffix}")

    if isinstance(cop, pd.DataFrame):
        cop_mapper = {x: f"{x} {suffix}" for x in cop.columns}
        cop = cop.rename(columns=cop_mapper)
        efficiency = cop[hps.index.to_list()]
    else:
        efficiency = costs.at[costs_name, "efficiency"].round(1)

    capex = costs.at[costs_name, "capital_cost"].round(1)
    lifetime = costs.at[costs_name, "lifetime"]
    build_year = n.investment_periods[0]

    # use suffix to retain COP profiles
    n.madd(
        "Link",
        hps.index,
        # suffix=suffix,
        bus0=hps.bus0,
        bus1=hps.bus1,
        carrier=hps.carrier,
        efficiency=efficiency,
        capital_cost=capex,
        p_nom_extendable=True,
        lifetime=lifetime,
        build_year=build_year,
    )


def add_industrial_gas_furnace(
    n: pypsa.Network,
    costs: pd.DataFrame,
    marginal_cost: pd.DataFrame | float | None = None,
) -> None:
    sector = SecNames.INDUSTRY.value

    capex = costs.at["direct firing gas", "capital_cost"].round(1)
    efficiency = costs.at["direct firing gas", "efficiency"].round(1)
    lifetime = costs.at["direct firing gas", "lifetime"]
    build_year = n.investment_periods[0]

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
    elif isinstance(marginal_cost, int | float):
        mc = marginal_cost
    else:
        mc = 0

    n.madd(
        "Link",
        furnaces.index,
        suffix="-gas-furnace",  # 'ind' included in index already
        bus0=furnaces.bus0,
        bus1=furnaces.bus1,
        bus2=furnaces.bus2,
        carrier=furnaces.carrier,
        efficiency=efficiency,
        efficiency2=furnaces.efficiency2,
        capital_cost=capex,
        p_nom_extendable=True,
        marginal_cost=mc,
        lifetime=lifetime,
        build_year=build_year,
    )


def add_industrial_coal_furnace(
    n: pypsa.Network,
    costs: pd.DataFrame,
    marginal_cost: pd.DataFrame | float | None = None,
) -> None:
    sector = SecNames.INDUSTRY.value

    # performance charasteristics taken from (Table 311.1a)
    # https://ens.dk/en/our-services/technology-catalogues/technology-data-industrial-process-heat
    # same source as tech-data, but its just not in latest version

    # capex approximated based on NG to incorporate fixed costs
    capex = costs.at["direct firing coal", "capital_cost"].round(1)
    efficiency = costs.at["direct firing coal", "efficiency"].round(1)
    lifetime = costs.at["direct firing coal", "lifetime"].round(1)
    build_year = n.investment_periods[0]

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
    elif isinstance(marginal_cost, int | float):
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
        p_nom_extendable=False,
        marginal_cost=mc,
        lifetime=lifetime,
        build_year=build_year,
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
    build_year = n.investment_periods[0]

    carrier_name = f"{sector}-heat"

    loads = n.loads[(n.loads.carrier == carrier_name)]

    hp = pd.DataFrame(index=loads.bus)
    hp["state"] = hp.index.map(n.buses.STATE)
    hp["bus0"] = hp.index.str.replace("-heat", f"-{SecCarriers.ELECTRICITY.value}")
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
        build_year=build_year,
    )
