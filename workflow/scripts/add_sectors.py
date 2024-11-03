"""
Generic module to add a new energy network.

Reads in the sector wildcard and will call corresponding scripts. In the
future, it would be good to integrate this logic into snakemake
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)
import sys
from typing import Optional

from _helpers import configure_logging, get_snapshots, load_costs
from add_electricity import sanitize_carriers
from build_emission_tracking import build_ch4_tracking, build_co2_tracking
from build_heat import build_heat
from build_natural_gas import StateGeometry, build_natural_gas
from build_stock_data import (
    add_road_transport_brownfield,
    add_service_brownfield,
    get_commercial_stock,
    get_residential_stock,
    get_transport_stock,
)
from build_transportation import apply_exogenous_ev_policy, build_transportation
from constants import STATE_2_CODE, STATES_INTERCONNECT_MAPPER
from constants_sector import RoadTransport
from shapely.geometry import Point

CODE_2_STATE = {v: k for k, v in STATE_2_CODE.items()}


def assign_bus_2_state(
    n: pypsa.Network,
    shp: str,
    states_2_include: list[str] = None,
    state_2_state_name: dict[str, str] = None,
) -> None:
    """
    Adds a state column to the network buses dataframe.

    The shapefile must be the counties shapefile
    """

    buses = n.buses[["x", "y"]].copy()
    buses["geometry"] = buses.apply(lambda x: Point(x.x, x.y), axis=1)
    buses = gpd.GeoDataFrame(buses, crs="EPSG:4269")

    states = gpd.read_file(shp).dissolve("STUSPS")["geometry"]
    states = gpd.GeoDataFrame(states)
    if states_2_include:
        states = states[states.index.isin(states_2_include)]

    # project to avoid CRS warning from geopandas
    buses_projected = buses.to_crs("EPSG:3857")
    states_projected = states.to_crs("EPSG:3857")
    gdf = gpd.sjoin_nearest(buses_projected, states_projected, how="left")

    n.buses["STATE"] = n.buses.index.map(gdf.STUSPS)

    if state_2_state_name:
        n.buses["STATE_NAME"] = n.buses.STATE.map(state_2_state_name)


def add_sector_foundation(
    n: pypsa.Network,
    carrier: str,
    add_supply: bool = True,
    costs: Optional[pd.DataFrame] = pd.DataFrame(),
    center_points: Optional[pd.DataFrame] = pd.DataFrame(),
) -> None:
    """
    Adds carrier, state level bus and store for the energy carrier.

    If add_supply, the store to supply energy will be added. If false,
    only the bus is created and no energy supply will be added to the
    state level bus.
    """

    match carrier:
        case "gas":
            carrier_kwargs = {"color": "#d35050", "nice_name": "Natural Gas"}
        case "coal":
            carrier_kwargs = {"color": "#d35050", "nice_name": "Coal"}
        case "oil" | "lpg":
            carrier_kwargs = {"color": "#d35050", "nice_name": "Liquid Petroleum Gas"}
        case _:
            carrier_kwargs = {}

    try:
        carrier_kwargs["co2_emissions"] = costs.at[carrier, "co2_emissions"]
    except KeyError:
        pass

    # make primary energy carriers

    if carrier not in n.carriers.index:
        n.add("Carrier", carrier, **carrier_kwargs)

    # make state level primary energy carrier buses

    states = n.buses.STATE.dropna().unique()

    zero_center_points = pd.DataFrame(
        index=states,
        columns=["x", "y"],
        dtype=float,
    ).fillna(0)
    zero_center_points.index.name = "STATE"

    if not center_points.empty:
        points = center_points.loc[states].copy()
        points = (
            pd.concat([points, zero_center_points])
            .reset_index(names=["STATE"])
            .drop_duplicates(keep="first", subset="STATE")
            .set_index("STATE")
        )
    else:
        points = zero_center_points.copy()

    points["name"] = points.index.map(CODE_2_STATE)
    points["interconnect"] = points.index.map(STATES_INTERCONNECT_MAPPER)

    buses_to_create = [f"{x} {carrier}" for x in points.index]
    existing = n.buses[n.buses.index.isin(buses_to_create)].STATE.dropna().unique()

    points = points[~points.index.isin(existing)]

    n.madd(
        "Bus",
        names=points.index,
        suffix=f" {carrier}",
        x=points.x,
        y=points.y,
        carrier=carrier,
        unit="MWh_th",
        interconnect=points.interconnect,
        country=points.index,  # for consistency
        STATE=points.index,
        STATE_NAME=points.name,
    )

    if add_supply:

        n.madd(
            "Store",
            names=points.index,
            suffix=f" {carrier}",
            bus=[f"{x} {carrier}" for x in points.index],
            e_nom=0,
            e_nom_extendable=True,
            capital_cost=0,
            e_nom_min=0,
            e_nom_max=np.inf,
            e_min_pu=-1,
            e_max_pu=0,
            e_cyclic_per_period=False,
            carrier=carrier,
            unit="MWh_th",
        )


def convert_generators_2_links(
    n: pypsa.Network,
    carrier: str,
    bus0_suffix: str,
    co2_intensity: float = 0,
):
    """
    Replace Generators with a link connecting to a state level primary energy.

    NOTE: THIS WILL ACCOUNT EMISSIONS TOWARDS THE PWR SECTOR

    Links bus1 are the bus the generator is attached to. Links bus0 are state
    level followed by the suffix (ie. "WA gas" if " gas" is the bus0_suffix)

    n: pypsa.Network,
    carrier: str,
        carrier of the generator to convert to a link
    bus0_suffix: str,
        suffix to attach link to
    """

    plants = n.generators[n.generators.carrier == carrier].copy()

    if plants.empty:
        return

    plants["STATE"] = plants.bus.map(n.buses.STATE)

    pnl = {}

    # copy over pnl parameters
    for c in n.iterate_components(["Generator"]):
        for param, df in c.pnl.items():
            # skip result vars
            if param not in (
                "p_min_pu",
                "p_max_pu",
                "p_set",
                "q_set",
                "marginal_cost",
                "marginal_cost_quadratic",
                "efficiency",
                "stand_by_cost",
            ):
                continue
            cols = [p for p in plants.index if p in df.columns]
            if cols:
                pnl[param] = df[cols]

    n.madd(
        "Link",
        names=plants.index,
        bus0=plants.STATE + bus0_suffix,
        bus1=plants.bus,
        bus2=plants.STATE + " pwr-co2",
        carrier=plants.carrier,
        p_nom_min=plants.p_nom_min / plants.efficiency,
        p_nom=plants.p_nom / plants.efficiency,  # links rated on input capacity
        p_nom_max=plants.p_nom_max / plants.efficiency,
        p_nom_extendable=plants.p_nom_extendable,
        ramp_limit_up=plants.ramp_limit_up,
        ramp_limit_down=plants.ramp_limit_down,
        efficiency=plants.efficiency,
        efficiency2=co2_intensity,
        marginal_cost=plants.marginal_cost * plants.efficiency,  # fuel costs rated at delievered
        capital_cost=plants.capital_cost * plants.efficiency,  # links rated on input capacity
        lifetime=plants.lifetime,
    )

    for param, df in pnl.items():
        n.links_t[param] = n.links_t[param].join(df, how="inner")

    # remove generators
    n.mremove("Generator", plants.index)


def split_loads_by_carrier(n: pypsa.Network):
    """
    Splits loads by carrier.

    At this point, all loads (ie. com-elec, com-heat, com-cool) will be
    nested under the elec bus. This function will create a new bus-load
    pair for each energy carrier that is NOT electricity.

    Note: This will break the flow of energy in the model! You must add a
    new link between the new bus and old bus if you want to retain the flow
    """

    for bus in n.buses.index.unique():
        df = n.loads[n.loads.bus == bus][["bus", "carrier"]]

        n.madd(
            "Bus",
            df.index,
            v_nom=1,
            x=n.buses.at[bus, "x"],
            y=n.buses.at[bus, "y"],
            carrier=df.carrier,
            country=n.buses.at[bus, "country"],
            interconnect=n.buses.at[bus, "interconnect"],
            STATE=n.buses.at[bus, "STATE"],
            STATE_NAME=n.buses.at[bus, "STATE_NAME"],
        )

    n.loads["bus"] = n.loads.index


def build_electricity_infra(n: pypsa.Network):
    """
    Adds links to connect electricity nodes.

    For example, will build the link between "p480 0" and "p480 0 res-
    elec"
    """

    df = n.loads[n.loads.index.str.endswith("-elec")].copy()

    df["bus0"] = df.apply(lambda row: row.bus.split(f" {row.carrier}")[0], axis=1)
    df["bus1"] = df.bus
    df["sector"] = df.carrier.map(lambda x: x.split("-")[0])
    df.index = df["bus0"] + " " + df["sector"]
    df["carrier"] = df["sector"] + "-elec-infra"

    n.madd(
        "Link",
        df.index,
        suffix="-elec-infra",
        bus0=df.bus0,
        bus1=df.bus1,
        carrier=df.carrier,
        efficiency=1,
        capital_cost=0,
        p_nom_extendable=True,
        lifetime=np.inf,
    )


def get_pwr_co2_intensity(carrier: str, costs: pd.DataFrame) -> float:
    """
    Gets co2 intensity to apply to pwr links.

    Spereate function, as there is some odd logic to account for
    different names in translation to a sector study.
    """

    # the ccs case are a hack solution

    match carrier:
        case "gas":
            return 0
        case "CCGT" | "OCGT" | "ccgt" | "ocgt":
            return costs.at["gas", "co2_emissions"]
        case "lpg":
            return costs.at["oil", "co2_emissions"]
        case "CCGT-95CCS" | "CCGT-97CCS":
            base = costs.at["gas", "co2_emissions"]
            ccs_level = int(carrier.split("-")[1].replace("CCS", ""))
            return (1 - ccs_level / 100) * base
        case "coal-95CCS" | "coal-99CCS":
            base = costs.at["gas", "co2_emissions"]
            ccs_level = int(carrier.split("-")[1].replace("CCS", ""))
            return (1 - ccs_level / 100) * base
        case _:
            return costs.at[carrier, "co2_emissions"]


print("y")

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_sectors",
            interconnect="western",
            simpl="33",
            clusters="4m",
            ll="v1.0",
            opts="2190SEG",
            sector="E-G",
        )
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)

    eia_api = snakemake.params.api["eia"]

    sectors = snakemake.wildcards.sector.split("-")

    # exit if only electricity network
    if all(s == "E" for s in sectors):
        n.export_to_netcdf(snakemake.output.network)
        sys.exit()

    # map states to each clustered bus

    if snakemake.wildcards.interconnect == "usa":
        states_2_map = [x for x, y in STATES_INTERCONNECT_MAPPER.items() if y in ("western", "eastern", "texas")]
    else:
        states_2_map = [x for x, y in STATES_INTERCONNECT_MAPPER.items() if y == snakemake.wildcards.interconnect]

    assign_bus_2_state(n, snakemake.input.county, states_2_map, CODE_2_STATE)

    sns = get_snapshots(snakemake.params.snapshots)

    costs = load_costs(snakemake.input.tech_costs)

    ###
    # Sector addition starts here
    ###

    # add sector specific emission tracking
    build_co2_tracking(n)

    # break out loads into sector specific buses
    split_loads_by_carrier(n)

    # add primary energy carriers for each state
    center_points = StateGeometry(snakemake.input.county).state_center_points.set_index(
        "STATE",
    )
    for carrier in ("oil", "coal", "gas"):
        add_supply = False if carrier == "gas" else True  # gas added in build_ng()
        add_sector_foundation(n, carrier, add_supply, costs, center_points)

    for carrier in ("OCGT", "CCGT", "CCGT-95CCS", "CCGT-97CCS"):
        co2_intensity = get_pwr_co2_intensity(carrier, costs)
        convert_generators_2_links(n, carrier, f" gas", co2_intensity)

    for carrier in ("coal", "coal-95CCS", "coal-99CCS"):
        co2_intensity = get_pwr_co2_intensity(carrier, costs)
        convert_generators_2_links(n, carrier, f" coal", co2_intensity)

    for carrier in ["oil"]:
        co2_intensity = get_pwr_co2_intensity(carrier, costs)
        convert_generators_2_links(n, carrier, f" oil", co2_intensity)

    ng_options = snakemake.params.sector["natural_gas"]

    # add natural gas infrastructure and data
    build_natural_gas(
        n=n,
        year=sns[0].year,
        api=eia_api,
        interconnect=snakemake.wildcards.interconnect,
        county_path=snakemake.input.county,
        pipelines_path=snakemake.input.pipeline_capacity,
        pipeline_shape_path=snakemake.input.pipeline_shape,
        options=ng_options,
    )

    # add methane tracking - if leakage rate is included
    # this must happen after natural gas system is built
    methane_options = snakemake.params.sector["methane"]
    leakage_rate = methane_options.get("leakage_rate", 0)
    if leakage_rate > 0.00001:
        gwp = methane_options.get("gwp", 1)
        build_ch4_tracking(n, gwp, leakage_rate)

    pop_layout_path = snakemake.input.clustered_pop_layout
    cop_ashp_path = snakemake.input.cop_air_total
    cop_gshp_path = snakemake.input.cop_soil_total

    # add electricity infrastructure
    build_electricity_infra(n=n)

    dynamic_cost_year = sns.year.min()

    # add heating and cooling
    split_res_com = snakemake.params.sector["service_sector"].get(
        "split_res_com",
        False,
    )
    heat_sectors = ["res", "com", "ind"] if split_res_com else ["srv", "ind"]
    for heat_sector in heat_sectors:
        if heat_sector == "srv":
            raise NotImplementedError
        elif heat_sector in ["res", "com"]:
            options = snakemake.params.sector["service_sector"]
        elif heat_sector == "ind":
            options = snakemake.params.sector["industrial_sector"]
        else:
            logger.warning(f"No config options found for {heat_sector}")
            options = {}
        build_heat(
            n=n,
            costs=costs,
            sector=heat_sector,
            pop_layout_path=pop_layout_path,
            cop_ashp_path=cop_ashp_path,
            cop_gshp_path=cop_gshp_path,
            options=options,
            eia=eia_api,
            year=dynamic_cost_year,
        )

    # add transportation
    ev_policy = pd.read_csv(snakemake.input.ev_policy, index_col=0)
    apply_exogenous_ev_policy(n, ev_policy)
    build_transportation(
        n=n,
        costs=costs,
        dynamic_pricing=True,
        eia=eia_api,
        year=dynamic_cost_year,
    )

    # check for end-use brownfield requirements

    if all(n.investment_periods > 2023):
        # this is quite crude assumption and should get updated
        # assume a 0.5% energy growth per year
        # https://www.eia.gov/todayinenergy/detail.php?id=56040
        base_year = 2023
        growth_multiplier = 1 - (min(n.investment_periods) - 2023) * (0.005)
    else:
        base_year = min(n.investment_periods)
        growth_multiplier = 1

    if snakemake.params.sector["transport_sector"]["brownfield"]:
        ratios = get_transport_stock(snakemake.params.api["eia"], base_year)
        for vehicle in RoadTransport:
            add_road_transport_brownfield(
                n,
                vehicle.value,
                growth_multiplier,
                ratios,
                costs,
            )

    if snakemake.params.sector["service_sector"]["brownfield"]:

        res_stock_dir = snakemake.input.residential_stock
        com_stock_dir = snakemake.input.commercial_stock

        if snakemake.params.sector["service_sector"]["water_heating"]["split_space_water"]:
            fuels = ["space_heating", "water_heating", "cooling"]
        else:
            fuels = ["heating", "cooling"]
        for fuel in fuels:

            # residential sector
            ratios = get_residential_stock(res_stock_dir, fuel)
            ratios.index = ratios.index.map(STATE_2_CODE)
            ratios = ratios.dropna()  # na is USA
            add_service_brownfield(n, "res", fuel, growth_multiplier, ratios, costs)

            # commercial sector
            ratios = get_commercial_stock(com_stock_dir, fuel)
            ratios.index = ratios.index.map(STATE_2_CODE)
            ratios = ratios.dropna()  # na is USA
            add_service_brownfield(n, "com", fuel, growth_multiplier, ratios, costs)

    # Needed as loads may be split off to urban/rural
    sanitize_carriers(n, snakemake.config)

    n.export_to_netcdf(snakemake.output.network)
