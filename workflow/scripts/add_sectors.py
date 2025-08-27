"""
Generic module to add a new energy network.

Reads in the sector wildcard and will call corresponding scripts. In the
future, it would be good to integrate this logic into snakemake
"""

import logging
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging, get_snapshots, load_costs
from add_electricity import sanitize_carriers
from add_extra_components import add_co2_network, add_co2_storage, add_dac
from build_electricity_sector import build_electricty
from build_emission_tracking import build_ch4_tracking, build_co2_tracking
from build_heat import build_heat
from build_natural_gas import StateGeometry, build_natural_gas
from build_stock_data import (
    add_industrial_brownfield,
    add_road_transport_brownfield,
    add_service_brownfield,
    get_commercial_stock,
    get_industrial_stock,
    get_residential_stock,
    get_transport_stock,
)
from build_transportation import build_transportation
from constants import CODE_2_STATE, NG_MWH_2_MMCF, STATE_2_CODE, STATES_INTERCONNECT_MAPPER
from constants_sector import RoadTransport
from eia import FuelCosts
from shapely.geometry import Point

logger = logging.getLogger(__name__)


def assign_bus_2_state(
    n: pypsa.Network,
    shp: str,
    states_2_include: list[str] | None = None,
    state_2_state_name: dict[str, str] | None = None,
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
    costs: pd.DataFrame | None = pd.DataFrame(),
    center_points: pd.DataFrame | None = pd.DataFrame(),
    eia_api: str | None = None,
) -> None:
    """
    Adds carrier, state level bus and store for the energy carrier.

    If add_supply, the store to supply energy will be added. If false,
    only the bus is created and no energy supply will be added to the
    state level bus.
    """
    co2_carrier = "carrier"
    if carrier == "gas":
        carrier_kwargs = {"color": "#d35050", "nice_name": "Natural Gas"}
    elif carrier == "coal":
        carrier_kwargs = {"color": "#d35050", "nice_name": "Coal"}
    elif carrier == "oil":  # heating oil
        carrier_kwargs = {"color": "#d35050", "nice_name": "Heating Oil"}
    elif carrier == "lpg":
        carrier_kwargs = {"color": "#d35050", "nice_name": "Liquid Petroleum Gas"}
        co2_carrier = "oil"
    else:
        raise ValueError(f"Unknown carrier of {carrier}")

    try:
        carrier_kwargs["co2_emissions"] = costs.at[co2_carrier, "co2_emissions"]
    except KeyError:
        pass

    # make primary energy carriers

    if carrier not in n.carriers.index:
        n.add("Carrier", carrier, **carrier_kwargs)

    # make state level primary energy carrier buses

    states = [x for x in n.buses.reeds_state.dropna().unique() if x]

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

    if eia_api:
        year = n.investment_periods[0]
        eia_carrier = "heating_oil" if carrier == "oil" else carrier
        dyanmic_cost = get_dynamic_marginal_costs(n, eia_carrier, eia_api, year, "power")
        dyanmic_cost = dyanmic_cost.set_index([dyanmic_cost.index.year, dyanmic_cost.index])

        if "USA" not in dyanmic_cost.columns:
            dyanmic_cost["USA"] = dyanmic_cost.mean(axis=1)

        marginal_cost = pd.DataFrame(index=dyanmic_cost.index)
        for state in n.buses.reeds_state.fillna(False).unique():
            if not state:
                continue
            try:
                marginal_cost[state] = dyanmic_cost[state]
            except KeyError:  # use USA average
                marginal_cost[state] = dyanmic_cost["USA"]

    else:
        marginal_cost = 0

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
            marginal_cost=marginal_cost,
            lifetime=np.inf,
            build_year=n.investment_periods[0],
        )


def get_dynamic_marginal_costs(
    n: pypsa.Network,
    fuel: str,
    eia: str,
    year: int,
    sector: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Gets end-use fuel costs at a state level."""
    sector_mapper = {
        "res": "residential",
        "com": "commercial",
        "pwr": "power",
        "ind": "industrial",
        "trn": "transport",
    }

    assert fuel in ("gas", "lpg", "coal", "heating_oil")

    if fuel == "gas":
        assert sector in ("res", "com", "ind", "pwr")
        if year < 2024:  # get actual monthly values
            raw = FuelCosts(fuel, year, eia, industry=sector_mapper[sector]).get_data(pivot=True)
            raw = raw * 1000 / NG_MWH_2_MMCF  # $/MCF -> $/MWh
        else:  # scale monthly values according to AEO
            act = FuelCosts(fuel, 2023, eia, industry=sector_mapper[sector]).get_data(pivot=True)
            proj = FuelCosts(fuel, year, eia, industry=sector_mapper[sector]).get_data(pivot=True)

            # actual comes in $/MCF, while projected comes in $/MMBTU
            # https://www.eia.gov/totalenergy/data/browser/index.php?tbl=TA4#/?f=A
            # 1.036 BTU / MCF
            act_mmbtu = act / 1.036

            if "USA" not in act.columns:
                act["USA"] = act.mean(axis=1)
                act_mmbtu["USA"] = act_mmbtu.mean(axis=1)

            actual_year_mean = act_mmbtu.mean().at["U.S."]
            proj_year_mean = proj.at[year, "USA"]
            scaler = proj_year_mean / actual_year_mean

            raw = act * scaler * 1000 / NG_MWH_2_MMCF  # $/MCF -> $/MWh
    elif fuel == "coal":
        # https://www.eia.gov/tools/faqs/faq.php?id=72&t=2
        # 19.18 MMBTU per short ton
        mmbtu_per_ston = 19.18
        wh_per_btu = 0.29307  # same as mwh_per_mmbtu

        # no industry = industrial, so use industry = power
        if year < 2024:  # get actual monthly values
            raw = FuelCosts(fuel, year, eia, industry="power").get_data(pivot=True)
            raw *= 1 / mmbtu_per_ston / wh_per_btu  # $/Ton -> $/MWh
        else:
            # idk why, but there is a weird issue from AEO actual costs (ie 2023) dont
            # seem to match actual reported value (or maybe more likely I am interpreteing
            # something wrong). I am taking the profile, then applying the value to the 2024
            # prices, and scaling from that.

            act = FuelCosts(fuel, 2023, eia, industry="power").get_data(pivot=True)
            proj_2024 = FuelCosts(fuel, 2024, eia, industry="power").get_data(pivot=True)
            proj = FuelCosts(fuel, year, eia, industry="power").get_data(pivot=True)

            act *= 1 / mmbtu_per_ston / wh_per_btu  # $/Ton -> $/MWh
            proj *= 1 / wh_per_btu  # $/MMBTU -> $/MWh
            proj_2024 *= 1 / wh_per_btu  # $/MMBTU -> $/MWh

            if "USA" not in act.columns:
                act["USA"] = act.mean(axis=1)

            present_day_scale = proj_2024.at[2024, "USA"] / act.mean().at["USA"]
            act_adjusted = act * present_day_scale

            proj_year_mean = proj.at[year, "USA"]
            scaler = proj_year_mean / act_adjusted.mean().at["USA"]

            raw = act_adjusted * scaler

            """
            act = FuelCosts(fuel, 2023, eia, industry="power").get_data(pivot=True)
            proj = FuelCosts(fuel, year, eia, industry="power").get_data(pivot=True)

            # actual comes in $/ton, while projected comes in $/MMBTU
            proj *= (1 / wh_per_btu) # $/MMBTU -> $/MWh
            act *= (1 / mmbtu_per_ston / wh_per_btu) # $/Ton -> $/MWh

            if "USA" not in act.columns:
                act["USA"] = act.mean(axis=1)

            # actual_year_mean = act.mean().at["USA"]
            proj_year_mean = proj.at[year, "USA"]
            scaler = proj_year_mean / actual_year_mean

            raw = act * scaler
            """

    elif fuel == "lpg":
        # https://www.eia.gov/energyexplained/units-and-calculators/
        btu_per_gallon = 120214
        wh_per_btu = 0.29307
        if year < 2024:
            raw = (
                FuelCosts(fuel, year, eia, grade="regular").get_data(pivot=True)
                * (1 / btu_per_gallon)
                * (1 / wh_per_btu)
                * (1000000)
            )  # $/gal -> $/MWh
        else:
            act = FuelCosts(fuel, 2023, eia, grade="regular").get_data(pivot=True)
            proj = FuelCosts(fuel, year, eia, grade="regular").get_data(pivot=True)

            # actual comes in $/gal, while projected comes in $/MMBTU
            proj *= btu_per_gallon / 1000000

            if "USA" not in act.columns:
                act["USA"] = act.mean(axis=1)

            actual_year_mean = act.mean().at["USA"]
            proj_year_mean = proj.at[year, "USA"]
            scaler = proj_year_mean / actual_year_mean

            # $/gal -> $/MWh
            raw = act * scaler * (1 / btu_per_gallon) * (1 / wh_per_btu) * (1000000)
    elif fuel == "heating_oil":
        # https://www.eia.gov/energyexplained/units-and-calculators/british-thermal-units.php
        btu_per_gallon = 138500
        wh_per_btu = 0.29307
        if year < 2024:
            raw = (
                FuelCosts("heating_oil", year, eia).get_data(pivot=True)
                * (1 / btu_per_gallon)
                * (1 / wh_per_btu)
                * (1000000)
            )  # $/gal -> $/MWh
        else:
            act = FuelCosts("heating_oil", 2023, eia).get_data(pivot=True)
            proj = FuelCosts("heating_oil", year, eia).get_data(pivot=True)

            # actual comes in $/gal, while projected comes in $/MMBTU
            proj *= btu_per_gallon / 1000000

            if "USA" not in act.columns:
                act["USA"] = act.mean(axis=1)

            actual_year_mean = act.mean().at["USA"]
            proj_year_mean = proj.at[year, "USA"]
            scaler = proj_year_mean / actual_year_mean

            # $/gal -> $/MWh
            raw = act * scaler * (1 / btu_per_gallon) * (1 / wh_per_btu) * (1000000)
    else:
        raise KeyError(f"{fuel} not recognized for dynamic fuel costs.")

    # may have to convert full state name to abbreviated state name
    # should probably change the EIA module to be consistent on what it returns...
    raw = raw.rename(columns=STATE_2_CODE)

    raw.index = pd.DatetimeIndex(raw.index)
    raw.index = raw.index.map(lambda x: x.replace(year=year))

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

    If eia api is provided, dynamic marginal costs are brought in from the API. This will
    match the method end-use techs bring in dynamic marginal costs.
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
        marginal_cost=0,
        # marginal_cost = plants.marginal_cost * plants.efficiency, # fuel costs rated at delievered
        capital_cost=plants.capital_cost,  # links rated on input capacity
        lifetime=plants.lifetime,
        build_year=plants.build_year,
    )

    for param, df in pnl.items():
        n.links_t[param] = n.links_t[param].join(df, how="inner")

    n.mremove("Generator", plants.index)

    # existing links will give a 'nan in efficiency2' warning
    n.links["efficiency2"] = n.links.efficiency2.fillna(0)


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


def get_pwr_co2_intensity(carrier: str, costs: pd.DataFrame) -> float:
    """
    Gets co2 intensity to apply to pwr links.

    Separate function, as there is some odd logic to account for
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


def add_elec_import_emission(n: pypsa.Network):
    """Adds emission tracking for electricity imports."""
    try:
        emissions = n.carriers.at["imports", "co2_emissions"]
    except KeyError:
        logger.info("No electrical imports found, skipping emission tracking")
        return

    import_links = n.links[n.links.carrier == "imports"]
    buses = n.buses[(n.buses.reeds_zone.isin(import_links.bus1)) & (n.buses.carrier == "AC")]
    bus_to_state = buses.set_index("reeds_zone")["reeds_state"].to_dict()

    for bus, state in bus_to_state.items():
        import_links_by_node = import_links[import_links.bus1 == bus]
        n.links.loc[import_links_by_node.index, "efficiency2"] = emissions
        n.links.loc[import_links_by_node.index, "bus2"] = state + " pwr-co2"


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_sectors",
            interconnect="western",
            simpl="40",
            clusters="4m",
            ll="v1.0",
            opts="1h-REM",
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
    # oil in this context is heating_oil
    for carrier in ("oil", "lpg", "coal", "gas"):
        add_supply = False if carrier == "gas" else True  # gas added in build_ng()
        api = None if carrier == "gas" else eia_api  # ng cost endogenously defined
        add_sector_foundation(n, carrier, add_supply, costs, center_points, api)

    for carrier in ("OCGT", "CCGT", "CCGT-95CCS", "CCGT-97CCS"):
        co2_intensity = get_pwr_co2_intensity(carrier, costs)
        convert_generators_2_links(n, carrier, " gas", co2_intensity)

    for carrier in ("coal", "coal-95CCS", "coal-99CCS"):
        co2_intensity = get_pwr_co2_intensity(carrier, costs)
        convert_generators_2_links(n, carrier, " coal", co2_intensity)

    # oil in this context is lpg for ppts
    for carrier in ["oil"]:
        co2_intensity = get_pwr_co2_intensity(carrier, costs)
        convert_generators_2_links(n, carrier, " lpg", co2_intensity)

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

    pop_layout_path = snakemake.input.clustered_pop_layout
    cop_ashp_path = snakemake.input.cop_air_total
    cop_gshp_path = snakemake.input.cop_soil_total

    split_res_com = snakemake.params.sector["service_sector"].get(
        "split_res_com",
        False,
    )

    # add electricity infrastructure
    # transport added seperatly to account for different mode demands
    elec_sectors = ["res", "com", "ind"] if split_res_com else ["srv", "ind"]
    for elec_sector in elec_sectors:
        if elec_sector in ["res", "com"]:
            options = snakemake.params.sector["service_sector"]
            build_electricty(
                n=n,
                sector=elec_sector,
                pop_layout_path=pop_layout_path,
                options=options,
            )
        else:
            options = snakemake.params.sector["industrial_sector"]
            build_electricty(n=n, sector=elec_sector, options=options)

    dynamic_cost_year = sns.year.min()

    # add heating and cooling
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
    trn_options = snakemake.params.sector["transport_sector"]
    must_run_evs = trn_options.get("must_run_evs", True)
    dr_config = trn_options.get("demand_response", {})
    build_transportation(
        n=n,
        costs=costs,
        must_run_evs=must_run_evs,
        dr_config=dr_config,
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
            if fuel == "water_heating":
                simple_storage = snakemake.params.sector["service_sector"]["water_heating"].get("simple_storage", False)
            else:
                simple_storage = None

            # residential sector
            ratios = get_residential_stock(res_stock_dir, fuel)
            ratios.index = ratios.index.map(STATE_2_CODE)
            ratios = ratios.dropna()  # na is USA
            add_service_brownfield(
                n=n,
                sector="res",
                fuel=fuel,
                growth_multiplier=growth_multiplier,
                ratios=ratios,
                costs=costs,
                simple_storage=simple_storage,
            )

            # commercial sector
            ratios = get_commercial_stock(com_stock_dir, fuel)
            ratios.index = ratios.index.map(STATE_2_CODE)
            ratios = ratios.dropna()  # na is USA
            add_service_brownfield(
                n=n,
                sector="com",
                fuel=fuel,
                growth_multiplier=growth_multiplier,
                ratios=ratios,
                costs=costs,
                simple_storage=simple_storage,
            )

    if snakemake.params.sector["industrial_sector"]["brownfield"]:
        mecs_file = snakemake.input.industrial_stock
        ratios = get_industrial_stock(mecs_file)

        fuels = ["heat"]

        for fuel in fuels:
            ratio = ratios.loc[fuel]
            add_industrial_brownfield(
                n=n,
                fuel=fuel,
                growth_multiplier=growth_multiplier,
                ratios=ratio,
                costs=costs,
            )

    # add methane tracking - if leakage rate is included
    # this must happen after all technologies and nat gas is built
    methane_options = snakemake.params.sector["methane"]
    upstream_leakage_rate = methane_options.get("upstream_leakage_rate", 0)
    downstream_leakage_rate = methane_options.get("downstream_leakage_rate", 0)
    gwp = methane_options.get("gwp", 0)
    build_ch4_tracking(n, gwp, upstream_leakage_rate, downstream_leakage_rate)

    # Needed as loads may be split off to urban/rural
    sanitize_carriers(n, snakemake.config)

    # add node level CO2 (underground) storage
    if snakemake.config["co2"]["storage"]:
        logger.info("Adding node level CO2 (underground) storage")
        add_co2_storage(n, snakemake.config, snakemake.input.co2_storage, costs, True)

    # add CO2 (transportation) network
    if snakemake.config["co2"]["network"]["enable"]:
        if snakemake.config["co2"]["storage"]:
            logger.info("Adding CO2 (transportation) network")
            add_co2_network(n, snakemake.config)
        else:
            logger.warning(
                "Not adding CO2 (transportation) network given that CO2 (underground) storage is not enabled",
            )

    # add node level DAC capabilities
    if snakemake.config["dac"]["enable"]:
        raise ValueError("DAC is not supported for Sector Studies. See https://github.com/PyPSA/pypsa-usa/issues/652")
        if snakemake.config["co2"]["storage"]:
            logger.info("Adding DAC capabilities")
            add_dac(n, snakemake.config, True)
        else:
            logger.warning("Not adding DAC capabilities given that CO2 (underground) storage is not enabled")

    # emission tracking for electricity imports
    add_elec_import_emission(n)

    n.export_to_netcdf(snakemake.output.network)
