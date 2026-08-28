################# ----------- Rules to Build Network ---------- #################

from itertools import chain


rule build_shapes:
    params:
        source_offshore_shapes=config_provider("offshore_shape"),
        offwind_params=config_provider("renewable", "offwind"),
    input:
        zone=DATA + "breakthrough_network/base_grid/zone.csv",
        nerc_shapes="repo_data/geospatial/NERC_Regions/NERC_Regions_Subregions.shp",
        reeds_shapes="repo_data/geospatial/Reeds_Shapes/rb_and_ba_areas.shp",
        onshore_shapes="repo_data/geospatial/BA_shapes_new/Modified_BE_BA_Shapes.shp",
        offshore_shapes_ca_osw="repo_data/geospatial/BOEM_CA_OSW_GIS/CA_OSW_BOEM_CallAreas.shp",
        offshore_shapes_eez=DATA + "eez/conus_eez.shp",
        county_shapes=DATA + "counties/cb_2020_us_county_500k.shp",
    output:
        country_shapes=GEOSPATIAL + "{interconnect}/country_shapes.geojson",
        onshore_shapes=GEOSPATIAL + "{interconnect}/onshore_shapes.geojson",
        offshore_shapes=GEOSPATIAL + "{interconnect}/offshore_shapes.geojson",
        state_shapes=GEOSPATIAL + "{interconnect}/state_boundaries.geojson",
        reeds_shapes=GEOSPATIAL + "{interconnect}/reeds_shapes.geojson",
        county_shapes=GEOSPATIAL + "{interconnect}/county_shapes.geojson",
    log:
        "logs/build_shapes/{interconnect}.log",
    threads: 1
    resources:
        walltime=config_provider("walltime", "build_shapes", default="00:30:00"),
        mem_mb=5000,
    script:
        "../scripts/build_shapes.py"


def bus_allocation_data(wildcards):
    """2020 census county populations, only when bus_allocation=population."""
    method = config["electricity"]["demand"].get("bus_allocation", "population")
    if method == "population":
        return {"county_population": DATA + "population/DECENNIALDHC2020.P1-Data.csv"}
    return {}


rule build_base_network:
    params:
        build_offshore_network=config_provider("offshore_network"),
        bus_allocation=config_provider(
            "electricity", "demand", "bus_allocation", default="population"
        ),
        model_topology=config_provider("model_topology", "include"),
        topological_boundaries=config_provider(
            "model_topology", "topological_boundaries"
        ),
        length_factor=config["lines"]["length_factor"],
    input:
        unpack(bus_allocation_data),
        buses=DATA + "breakthrough_network/base_grid/bus.csv",
        lines=DATA + "breakthrough_network/base_grid/branch.csv",
        links=DATA + "breakthrough_network/base_grid/dcline.csv",
        bus2sub=DATA + "breakthrough_network/base_grid/bus2sub.csv",
        sub=DATA + "breakthrough_network/base_grid/sub.csv",
        onshore_shapes=GEOSPATIAL + "{interconnect}/onshore_shapes.geojson",
        offshore_shapes=GEOSPATIAL + "{interconnect}/offshore_shapes.geojson",
        state_shapes=GEOSPATIAL + "{interconnect}/state_boundaries.geojson",
        reeds_shapes=GEOSPATIAL + "{interconnect}/reeds_shapes.geojson",
        county_shapes=GEOSPATIAL + "{interconnect}/county_shapes.geojson",
        reeds_memberships="repo_data/ReEDS_Constraints/membership.csv",
    output:
        bus2sub=BUSMAPS + "{interconnect}/bus2sub.csv",
        sub=BUSMAPS + "{interconnect}/sub.csv",
        bus_gis=GEOSPATIAL + "{interconnect}/bus_gis.csv",
        lines_gis=GEOSPATIAL + "{interconnect}/lines_gis.csv",
        network=NETWORKS + "{interconnect}/elec_base_network.nc",
    log:
        "logs/create_network/{interconnect}.log",
    threads: 1
    resources:
        mem_mb=5000,
        walltime=config_provider("walltime", "build_base_network", default="00:30:00"),
    script:
        "../scripts/build_base_network.py"


rule build_bus_regions:
    params:
        topological_boundaries=config_provider(
            "model_topology", "topological_boundaries"
        ),
        model_topology_include=config_provider(
            "model_topology", "include", default=None
        ),
    input:
        country_shapes=GEOSPATIAL + "{interconnect}/country_shapes.geojson",
        county_shapes=GEOSPATIAL + "{interconnect}/county_shapes.geojson",
        state_shapes=GEOSPATIAL + "{interconnect}/state_boundaries.geojson",
        ba_region_shapes=GEOSPATIAL + "{interconnect}/onshore_shapes.geojson",
        reeds_shapes=GEOSPATIAL + "{interconnect}/reeds_shapes.geojson",
        offshore_shapes=GEOSPATIAL + "{interconnect}/offshore_shapes.geojson",
        base_network=NETWORKS + "{interconnect}/elec_base_network.nc",
        bus2sub=BUSMAPS + "{interconnect}/bus2sub.csv",
        sub=BUSMAPS + "{interconnect}/sub.csv",
    output:
        regions_onshore=GEOSPATIAL + "{interconnect}/regions_onshore.geojson",
        regions_offshore=GEOSPATIAL + "{interconnect}/regions_offshore.geojson",
    log:
        "logs/build_bus_regions/{interconnect}.log",
    threads: 1
    resources:
        mem_mb=3000,
        walltime=config_provider("walltime", "build_bus_regions", default="00:30:00"),
    script:
        "../scripts/build_bus_regions.py"


rule build_cost_data:
    params:
        aeo=config_provider("costs", "aeo"),
        pudl_path=config_provider("pudl_path"),
    input:
        efs_tech_costs="repo_data/costs/EFS_Technology_Data.xlsx",
        efs_icev_costs="repo_data/costs/efs_icev_costs.csv",
        eia_tech_costs="repo_data/costs/eia_tech_costs.csv",
        egs_costs="repo_data/costs/egs_costs.csv",
        additional_costs="repo_data/costs/additional_costs.csv",
    output:
        tech_costs=COSTS + "costs_{year}.csv",
        sector_costs=COSTS + "sector_costs_{year}.csv",
    log:
        LOGS + "costs_{year}.log",
    threads: 1
    resources:
        mem_mb=5000,
        walltime=config_provider("walltime", "build_cost_data", default="00:30:00"),
    script:
        "../scripts/build_cost_data.py"


ATLITE_NPROCESSES = config["atlite"].get("nprocesses", 4)

if config["enable"].get("build_cutout", False):

    rule build_cutout:
        params:
            snapshots=config_provider("snapshots"),
            cutouts=config_provider("atlite", "cutouts"),
            interconnects=config_provider("atlite", "interconnects"),
        input:
            regions_onshore=GEOSPATIAL + "{interconnect}/country_shapes.geojson",
            regions_offshore=GEOSPATIAL + "{interconnect}/offshore_shapes.geojson",
        output:
            protected("cutouts/" + CDIR + "{interconnect}_{cutout}.nc"),
        log:
            "logs/" + CDIR + "build_cutout/{interconnect}_{cutout}.log",
        benchmark:
            "benchmarks/" + CDIR + "build_cutout_{interconnect}_{cutout}"
        threads: ATLITE_NPROCESSES
        resources:
            mem_mb=ATLITE_NPROCESSES * 5000,
            walltime=config_provider("walltime", "build_cutout", default="10:30:00"),
        script:
            "../scripts/build_cutout.py"


# Only use planning_horizon for GoDEEEP future scenarios
godeeep_planning_horizon = (
    config.get("renewable", {}).get("dataset") == "godeeep"
    and config["renewable_scenarios"][0] != "historical"
)


def nrel_exclusion_artifact(kind):
    """Input function for the NREL exclusion artifact of the given ``kind``.

    ``kind`` is either ``avail`` or ``caps``; the artifact is named
    ``{kind}_{tech}_{access}{_cec}{_boem}.nc``. Returns no input when
    ``renewable_land_access`` is unset.
    """

    def _inner(w):
        access = config.get("renewable_land_access")
        if not access:
            return []
        cec = (
            "_cec"
            if config.get("apply_cec_basescreen")
            and w.technology in ("onwind", "solar")
            else ""
        )
        boem = (
            "_boem"
            if config.get("apply_boem_osw") and w.technology.startswith("offwind")
            else ""
        )
        return (
            DATA
            + f"nrel_exclusion/derived/{kind}_{w.technology}_{access}{cec}{boem}.nc"
        )

    return _inner


rule build_renewable_profiles:
    params:
        renewable=config_provider("renewable"),
        snapshots=config_provider("snapshots"),
        planning_horizon=lambda w: (
            int(w.planning_horizon) if godeeep_planning_horizon else None
        ),
        renewable_scenarios=config_provider("renewable_scenarios"),
        mapping_cache_dir=PROFILES + "{interconnect}/nrel_mapping_cache",
    input:
        corine=ancient(
            DATA
            + "copernicus/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_USA_EPSG-4326.tif"
        ),
        natura=lambda w: (
            DATA + "natura.tiff" if config["renewable"][w.technology]["natura"] else []
        ),
        gebco=ancient(
            lambda w: (
                DATA + "gebco/gebco_2023_n55.0_s10.0_w-126.0_e-65.0.tif"
                if config["renewable"][w.technology].get("max_depth")
                else []
            )
        ),
        country_shapes=GEOSPATIAL + "{interconnect}/country_shapes.geojson",
        offshore_shapes=GEOSPATIAL + "{interconnect}/offshore_shapes.geojson",
        cec_onwind="repo_data/geospatial/CEC_GIS/CEC_Wind_BaseScreen_epsg3310.tif",
        cec_solar="repo_data/geospatial/CEC_GIS/CEC_Solar_BaseScreen_epsg3310.tif",
        boem_osw="repo_data/geospatial/boem_osw_planning_areas.tif",
        regions=lambda w: (
            GEOSPATIAL + "{interconnect}/regions_onshore_s{simpl}.geojson"
            if w.technology in ("onwind", "solar")
            else GEOSPATIAL + "{interconnect}/regions_offshore_s{simpl}.geojson"
        ),
        nrel_avail=nrel_exclusion_artifact("avail"),
        nrel_caps=nrel_exclusion_artifact("caps"),
        cutout=lambda wildcards: (
            expand(
                "cutouts/"
                + CDIR
                + "usa_"
                + config["renewable"][wildcards.technology]["cutout"]
                + "_{renewable_weather_year}"
                + ".nc",
                renewable_weather_year=config["renewable_weather_years"],
            )
            if config["renewable"]["dataset"] == "atlite"
            else []
        ),
        busmap_s=BUSMAPS + "{interconnect}/busmap_s{simpl}.csv",
    output:
        profile=(
            PROFILES
            + "{interconnect}/{planning_horizon}/profile_{technology}_s{simpl}.nc"
            if godeeep_planning_horizon
            else PROFILES + "{interconnect}/profile_{technology}_s{simpl}.nc"
        ),
    log:
        LOGS
        + "{interconnect}/build_renewable_profile_{technology}_{planning_horizon}_s{simpl}.log"
        if godeeep_planning_horizon
        else LOGS + "{interconnect}/build_renewable_profile_{technology}_s{simpl}.log",
    benchmark:
        (
            BENCHMARKS
            + "{interconnect}/build_renewable_profiles_{technology}_{planning_horizon}_s{simpl}"
            if godeeep_planning_horizon
            else BENCHMARKS
            + "{interconnect}/build_renewable_profiles_{technology}_s{simpl}"
        )
    threads: ATLITE_NPROCESSES
    resources:
        mem_mb=lambda wildcards, input, attempt: (
            ATLITE_NPROCESSES * input.size // 2000000
        )
        * 2.5,
        walltime=config_provider(
            "walltime", "build_renewable_profiles", default="02:30:00"
        ),
    wildcard_constraints:
        technology="(?!hydro|EGS).*",  # Any technology other than hydro
    script:
        "../scripts/build_renewable_profiles.py"


# eastern broken out just to aviod awful formatting issues
# texas in western due to spillover of interconnect
INTERCONNECT_2_STATE = {
    "eastern": ["AL", "AR", "CT", "DE", "FL", "GA", "IL", "IN", "IA", "KS", "KY", "LA"],
    "western": ["AZ", "CA", "CO", "ID", "MT", "NV", "NM", "OR", "UT", "WA", "WY", "TX"],
    "texas": ["TX"],
}
INTERCONNECT_2_STATE["eastern"].extend(["ME", "MD", "MA", "MI", "MN", "MS", "MO", "NE"])
INTERCONNECT_2_STATE["eastern"].extend(["NH", "NJ", "NY", "NC", "ND", "OH", "OK", "PA"])
INTERCONNECT_2_STATE["eastern"].extend(["RI", "SC", "SD", "TN", "VT", "VA", "WV", "WI"])
INTERCONNECT_2_STATE["usa"] = sum(INTERCONNECT_2_STATE.values(), [])

EER_DEMAND_FILES = (
    "demand_EER2025_100by2050.h5",
    "demand_EER2025_Baseline_AEO2023.h5",
    "demand_EER2025_IRAlow.h5",
)


def eer_demand_file():
    filename = config["electricity"]["demand"]["scenario"].get(
        "eer_file",
        "demand_EER2025_100by2050.h5",
    )
    if filename not in EER_DEMAND_FILES:
        raise ValueError(
            "electricity.demand.scenario.eer_file must be one of "
            f"{EER_DEMAND_FILES}; received {filename}."
        )
    return filename


def demand_raw_data(wildcards):
    # get profile to use
    end_use = wildcards.end_use
    if end_use == "power":
        profile = config["electricity"]["demand"]["profile"]
    elif end_use == "residential":
        eulp_sector = "res"
        profile = "eulp"
    elif end_use == "commercial":
        eulp_sector = "com"
        profile = "eulp"
    elif end_use == "transport":
        vehicle = wildcards.get("vehicle", None)
        if vehicle:  # non-road transport
            profile = "transport_aeo"
        else:
            profile = "transport_efs_aeo"
    elif end_use == "industry":
        profile = "cliu"

    # get required input data based on profile
    if profile == "eia":
        return DATA + "GridEmissions/EIA_DMD_2018_2024.csv"
    elif profile == "efs":
        efs_case = config["electricity"]["demand"]["scenario"]["efs_case"].capitalize()
        efs_speed = config["electricity"]["demand"]["scenario"][
            "efs_speed"
        ].capitalize()
        return DATA + f"nrel_efs/EFSLoadProfile_{efs_case}_{efs_speed}.csv"
    elif profile == "eer":
        return DATA + f"eer/{eer_demand_file()}"
    elif profile == "ferc":
        return [
            DATA + "pudl/out_ferc714__hourly_estimated_state_demand.parquet",
            DATA + "pudl/censusdp1tract.sqlite",
        ]
    elif profile == "eulp":
        return [
            DATA + f"eulp/{eulp_sector}/{state}.csv"
            for state in INTERCONNECT_2_STATE[wildcards.interconnect]
        ]
    elif profile == "cliu":
        return [
            DATA + "industry_load/2014_update_20170910-0116.csv",  # cliu data
            DATA + "industry_load/epri_industrial_loads.csv",  # epri data
            DATA + "industry_load/table3_2.xlsx",  # mecs data
            DATA + "industry_load/fips_codes.csv",  # fips data
        ]
    elif profile == "transport_efs_aeo":
        efs_case = config["electricity"]["demand"]["scenario"]["efs_case"].capitalize()
        efs_speed = config["electricity"]["demand"]["scenario"][
            "efs_speed"
        ].capitalize()
        return [
            DATA + f"nrel_efs/EFSLoadProfile_{efs_case}_{efs_speed}.csv",
            "repo_data/sectors/transport_ratios.csv",
        ]
    elif profile == "transport_aeo":
        return [
            "repo_data/sectors/transport_ratios.csv",
        ]
    else:
        return []


def demand_disaggregate_data(wildcards):
    """CLIU county-level industrial loads are the only disaggregation input.

    All other end uses disaggregate by population and need no extra file.
    """
    if wildcards.end_use != "industry":
        return []
    return DATA + "industry_load/2014_update_20170910-0116.csv"


def demand_scaling_data(wildcards):

    end_use = wildcards.end_use
    if end_use == "power":
        profile = config["electricity"]["demand"]["profile"]
    else:
        profile = "eia"

    if profile == "efs":
        efs_case = config["electricity"]["demand"]["scenario"]["efs_case"].capitalize()
        efs_speed = config["electricity"]["demand"]["scenario"][
            "efs_speed"
        ].capitalize()
        return DATA + f"nrel_efs/EFSLoadProfile_{efs_case}_{efs_speed}.csv"
    elif profile == "eia":
        return []
    elif profile == "ferc":
        return []
    elif profile == "eer":
        return []
    else:
        return ""


rule build_electrical_demand:
    wildcard_constraints:
        end_use="power",  # added for consistency in build_demand.py
    params:
        demand_params=config["electricity"]["demand"],
        eia_api=config["api"]["eia"],
        profile_year=pd.to_datetime(config["snapshots"]["start"]).year,
        planning_horizons=config["scenario"]["planning_horizons"],
        renewable_weather_years=config["renewable_weather_years"],
        snapshots=config["snapshots"],
        pudl_path=config_provider("pudl_path"),
    input:
        network=NETWORKS + "{interconnect}/elec_s{simpl}.nc",
        demand_files=demand_raw_data,
        demand_scaling_file=demand_scaling_data,
    output:
        elec_demand=DEMAND + "{interconnect}/{end_use}_electricity_s{simpl}.csv",
    log:
        LOGS + "{interconnect}/{end_use}_build_demand_s{simpl}.log",
    benchmark:
        BENCHMARKS + "{interconnect}/{end_use}_build_demand_s{simpl}"
    threads: 2
    resources:
        mem_mb=lambda wildcards, input, attempt: (input.size // 100000) * attempt * 2,
        walltime=config_provider(
            "walltime", "build_electrical_demand", default="00:50:00"
        ),
    script:
        "../scripts/build_demand.py"


rule build_service_demand:
    wildcard_constraints:
        end_use="residential|commercial",
    params:
        planning_horizons=config_provider("scenario", "planning_horizons"),
        profile_year=pd.to_datetime(config["snapshots"]["start"]).year,
        eia_api=config_provider("api", "eia"),
        snapshots=config_provider("snapshots"),
    input:
        network=NETWORKS + "{interconnect}/elec_s{simpl}.nc",
        demand_files=demand_raw_data,
        dissagregate_files=demand_disaggregate_data,
        demand_scaling_file=demand_scaling_data,
    output:
        electricity=DEMAND + "{interconnect}/{end_use}_electricity_s{simpl}.pkl",
        space_heat=DEMAND + "{interconnect}/{end_use}_space-heating_s{simpl}.pkl",
        water_heat=DEMAND + "{interconnect}/{end_use}_water-heating_s{simpl}.pkl",
        cool=DEMAND + "{interconnect}/{end_use}_cooling_s{simpl}.pkl",
    log:
        LOGS + "{interconnect}/demand/{end_use}_build_demand_s{simpl}.log",
    benchmark:
        BENCHMARKS + "{interconnect}/demand/{end_use}_build_demand_s{simpl}"
    threads: 2
    resources:
        mem_mb=lambda wildcards, input, attempt: (input.size // 70000) * attempt * 2,
    script:
        "../scripts/build_demand.py"


rule build_industry_demand:
    wildcard_constraints:
        end_use="industry",
    params:
        planning_horizons=config_provider("scenario", "planning_horizons"),
        profile_year=pd.to_datetime(config["snapshots"]["start"]).year,
        eia_api=config_provider("api", "eia"),
        snapshots=config_provider("snapshots"),
        pudl_path=config_provider("pudl_path"),
    input:
        network=NETWORKS + "{interconnect}/elec_s{simpl}.nc",
        demand_files=demand_raw_data,
        dissagregate_files=demand_disaggregate_data,
        demand_scaling_file=demand_scaling_data,
    output:
        electricity=DEMAND + "{interconnect}/{end_use}_electricity_s{simpl}.pkl",
        heat=DEMAND + "{interconnect}/{end_use}_heating_s{simpl}.pkl",
    log:
        LOGS + "{interconnect}/demand/{end_use}_build_demand_s{simpl}.log",
    benchmark:
        BENCHMARKS + "{interconnect}/demand/{end_use}_build_demand_s{simpl}"
    threads: 2
    resources:
        mem_mb=lambda wildcards, input, attempt: (input.size // 70000) * attempt * 2,
        walltime=config_provider("walltime", "build_sector_demand", default="00:50:00"),
    script:
        "../scripts/build_demand.py"


rule build_transport_road_demand:
    wildcard_constraints:
        end_use="transport",
    params:
        planning_horizons=config_provider("scenario", "planning_horizons"),
        profile_year=pd.to_datetime(config["snapshots"]["start"]).year,
        eia_api=config_provider("api", "eia"),
        snapshots=config_provider("snapshots"),
    input:
        network=NETWORKS + "{interconnect}/elec_s{simpl}.nc",
        demand_files=demand_raw_data,
        dissagregate_files=demand_disaggregate_data,
        demand_scaling_file=demand_scaling_data,
    output:
        light_duty=DEMAND + "{interconnect}/{end_use}_light-duty_s{simpl}.pkl",
        med_duty=DEMAND + "{interconnect}/{end_use}_med-duty_s{simpl}.pkl",
        heavy_duty=DEMAND + "{interconnect}/{end_use}_heavy-duty_s{simpl}.pkl",
        bus=DEMAND + "{interconnect}/{end_use}_bus_s{simpl}.pkl",
    log:
        LOGS + "{interconnect}/demand/{end_use}_build_demand_s{simpl}.log",
    benchmark:
        BENCHMARKS + "{interconnect}/demand/{end_use}_build_demand_s{simpl}"
    threads: 2
    resources:
        mem_mb=lambda wildcards, input, attempt: (input.size // 70000) * attempt * 2,
        walltime=config_provider(
            "walltime", "build_transport_road_demand", default="00:50:00"
        ),
    script:
        "../scripts/build_demand.py"


rule build_transport_other_demand:
    wildcard_constraints:
        end_use="transport",
        vehicle="boat-shipping|air|rail-shipping|rail-passenger",
    params:
        planning_horizons=config_provider("scenario", "planning_horizons"),
        eia_api=config_provider("api", "eia"),
        snapshots=config_provider("snapshots"),
    input:
        network=NETWORKS + "{interconnect}/elec_s{simpl}.nc",
        demand_files=demand_raw_data,
        dissagregate_files=demand_disaggregate_data,
    output:
        DEMAND + "{interconnect}/{end_use}_{vehicle}_s{simpl}.pkl",
    log:
        LOGS + "{interconnect}/demand/{end_use}_{vehicle}_build_demand_s{simpl}.log",
    benchmark:
        BENCHMARKS
        +"{interconnect}/demand/{end_use}_{vehicle}_build_demand_s{simpl}"
    threads: 2
    resources:
        mem_mb=lambda wildcards, input, attempt: (input.size // 70000) * attempt * 2,
    script:
        "../scripts/build_demand.py"


def demand_to_add(wildcards):

    if config["scenario"]["sector"] == "E":
        return DEMAND + "{interconnect}/power_electricity_s{simpl}.csv"
    else:
        # service demand
        services = ["residential", "commercial"]
        if config["sector"]["service_sector"]["split_space_water_heating"]:
            fuels = ["electricity", "cooling", "space-heating", "water-heating"]
        else:
            fuels = ["electricity", "cooling", "heating"]
        service_demands = [
            DEMAND + "{interconnect}/" + service + "_" + fuel + "_s{simpl}.pkl"
            for service in services
            for fuel in fuels
        ]
        # industrial demand
        fuels = ["electricity", "heating"]
        industrial_demands = [
            DEMAND + "{interconnect}/industry_" + fuel + "_s{simpl}.pkl"
            for fuel in fuels
        ]
        # road transport demands
        vehicles = ["light-duty", "med-duty", "heavy-duty", "bus"]
        road_demand = [
            DEMAND + "{interconnect}/transport_" + vehicle + "_s{simpl}.pkl"
            for vehicle in vehicles
        ]

        # other transport demands
        vehicles = ["boat-shipping", "rail-shipping", "rail-passenger", "air"]
        non_road_demand = [
            DEMAND + "{interconnect}/transport_" + vehicle + "_s{simpl}.pkl"
            for vehicle in vehicles
        ]

        return chain(service_demands, industrial_demands, road_demand, non_road_demand)


rule add_demand:
    params:
        sectors=config["scenario"]["sector"],
        planning_horizons=config_provider("scenario", "planning_horizons"),
        snapshots=config_provider("snapshots"),
    input:
        network=NETWORKS + "{interconnect}/elec_s{simpl}.nc",
        demand=demand_to_add,
    output:
        network=NETWORKS + "{interconnect}/elec_s{simpl}_dem.nc",
    log:
        LOGS + "{interconnect}/elec_s{simpl}_add_demand.log",
    benchmark:
        BENCHMARKS + "{interconnect}/elec_s{simpl}_add_demand"
    resources:
        mem_mb=lambda wildcards, input, attempt: (input.size // 70000) * attempt * 2,
        walltime=config_provider("walltime", "add_demand", default="00:50:00"),
    script:
        "../scripts/add_demand.py"


def ba_gas_dynamic_fuel_price_files(wildcards):
    files = []
    if wildcards.interconnect in ("usa", "western"):
        files.append(DATA + "costs/caiso_ng_power_prices.csv")
    return files


rule build_fuel_prices:
    params:
        snapshots=config["snapshots"],
        api_eia=config["api"]["eia"],
        pudl_path=config_provider("pudl_path"),
    input:
        gas_balancing_area=ba_gas_dynamic_fuel_price_files,
    output:
        state_ng_fuel_prices=PRICES + "{interconnect}/state_ng_power_prices.csv",
        state_coal_fuel_prices=PRICES + "{interconnect}/state_coal_power_prices.csv",
        ba_ng_fuel_prices=PRICES + "{interconnect}/ba_ng_power_prices.csv",
        pudl_fuel_costs=PRICES + "{interconnect}/pudl_fuel_costs.csv",
    log:
        LOGS + "{interconnect}/build_fuel_prices.log",
    benchmark:
        BENCHMARKS + "{interconnect}/build_fuel_prices"
    threads: 1
    retries: 3
    resources:
        mem_mb=30000,
        walltime=config_provider("walltime", "build_fuel_prices", default="00:20:00"),
    script:
        "../scripts/build_fuel_prices.py"


def dynamic_fuel_price_files(wildcards):
    if config["conventional"]["dynamic_fuel_price"]["wholesale"]:
        return {
            "state_ng_fuel_prices": PRICES + "{interconnect}/state_ng_power_prices.csv",
            "state_coal_fuel_prices": PRICES
            + "{interconnect}/state_coal_power_prices.csv",
            "ba_ng_fuel_prices": PRICES + "{interconnect}/ba_ng_power_prices.csv",
        }
    else:
        return {}


rule build_powerplants:
    params:
        pudl_path=config_provider("pudl_path"),
    input:
        wecc_ads="repo_data/WECC_ADS_public",
        eia_ads_generator_mapping="repo_data/WECC_ADS_public/eia_ads_generator_mapping_updated.csv",
        fuel_costs="repo_data/plants/fuelCost22.csv",
        cems="repo_data/plants/cems_heat_rates.xlsx",
        epa_crosswalk="repo_data/plants/epa_eia_crosswalk.csv",
    output:
        powerplants="resources/powerplants/powerplants.csv",
    log:
        "logs/build_powerplants.log",
    resources:
        mem_mb=30000,
        walltime=config_provider("walltime", "build_powerplants", default="00:30:00"),
    script:
        "../scripts/build_powerplants.py"


rule add_electricity:
    params:
        length_factor=config["lines"]["length_factor"],
        renewable=config["renewable"],
        renewable_carriers=config["electricity"]["renewable_carriers"],
        extendable_carriers=config["electricity"]["extendable_carriers"],
        conventional_carriers=config["electricity"]["conventional_carriers"],
        conventional=config["conventional"],
        costs=config["costs"],
        planning_horizons=config["scenario"]["planning_horizons"],
        eia_api=config["api"]["eia"],
    input:
        unpack(dynamic_fuel_price_files),
        **(
            {
                # For GODEEEP future scenarios: pass all horizon-specific profiles
                f"profile_{tech}_{horizon}": PROFILES
                + f"{{interconnect}}/{horizon}/profile_{tech}_s{{simpl}}.nc"
                for tech in config["electricity"]["renewable_carriers"]
                if tech != "hydro"
                for horizon in config["scenario"]["planning_horizons"]
            }
            if godeeep_planning_horizon
            else {
                # For historical or AtLite: pass single profile
                f"profile_{tech}": PROFILES
                + "{interconnect}"
                + f"/profile_{tech}_s{{simpl}}.nc"
                for tech in config["electricity"]["renewable_carriers"]
                if tech != "hydro"
            }
        ),
        **{
            f"conventional_{carrier}_{attr}": fn
            for carrier, d in config.get("conventional", {None: {}}).items()
            if carrier in config["electricity"]["conventional_carriers"]
            for attr, fn in d.items()
            if str(fn).startswith("data/")
        },
        **{
            f"gen_cost_mult_{Path(x).stem}": f"repo_data/locational_multipliers/{Path(x).name}"
            for x in Path("repo_data/locational_multipliers/").glob("*")
        },
        base_network=NETWORKS + "{interconnect}/elec_s{simpl}_dem.nc",
        tech_costs=COSTS + f"costs_{config['scenario']['planning_horizons'][0]}.csv",
        # attach first horizon costs
        all_reeds_shapes="repo_data/geospatial/Reeds_Shapes/rb_and_ba_areas.shp",
        reeds_memberships="repo_data/ReEDS_Constraints/membership.csv",
        regions_onshore=GEOSPATIAL + "{interconnect}/regions_onshore_s{simpl}.geojson",
        regions_offshore=GEOSPATIAL + "{interconnect}/regions_offshore_s{simpl}.geojson",
        reeds_shapes=GEOSPATIAL + "{interconnect}/reeds_shapes.geojson",
        powerplants="resources/powerplants/powerplants.csv",
        plants_breakthrough=DATA + "breakthrough_network/base_grid/plant.csv",
        hydro_breakthrough=DATA + "breakthrough_network/base_grid/hydro.csv",
        bus2sub=BUSMAPS + "{interconnect}/bus2sub.csv",
        busmap_s=BUSMAPS + "{interconnect}/busmap_s{simpl}.csv",
        pudl_fuel_costs=PRICES + "{interconnect}/pudl_fuel_costs.csv",
        specs_egs=(
            PROFILES + "{interconnect}/specs_EGS_s{simpl}.nc"
            if "EGS" in config["electricity"]["extendable_carriers"]["Generator"]
            else []
        ),
        profile_egs=(
            PROFILES + "{interconnect}/profile_EGS_s{simpl}.nc"
            if "EGS" in config["electricity"]["extendable_carriers"]["Generator"]
            else []
        ),
        seismic_exclusion=(
            DATA + "seismic_risk_exclusion/seismic_risk_mask.geojson"
            if "EGS" in config["electricity"]["extendable_carriers"]["Generator"]
            and config.get("renewable", {})
            .get("EGS", {})
            .get("seismic_exclusion", False)
            else []
        ),
    output:
        NETWORKS + "{interconnect}/elec_s{simpl}_l_pp.pkl",
    log:
        LOGS + "{interconnect}/elec_s{simpl}_add_electricity.log",
    benchmark:
        BENCHMARKS + "{interconnect}/elec_s{simpl}_add_electricity"
    threads: 1
    resources:
        mem_mb=lambda wildcards, input, attempt: (input.size // 400000) * attempt * 2,
        walltime=config_provider("walltime", "add_electricity", default="01:00:00"),
    script:
        "../scripts/add_electricity.py"


################# ----------- Rules to Aggregate & Simplify Network ---------- #################
rule aggregate_to_substations:
    params:
        aggregation_strategies=config["clustering"].get("aggregation_strategies", {}),
        topological_boundaries=config_provider(
            "model_topology", "topological_boundaries"
        ),
    input:
        bus2sub=BUSMAPS + "{interconnect}/bus2sub.csv",
        sub=BUSMAPS + "{interconnect}/sub.csv",
        network=NETWORKS + "{interconnect}/elec_base_network.nc",
    output:
        network=NETWORKS + "{interconnect}/elec_b.nc",
        busmap=BUSMAPS + "{interconnect}/busmap_b.csv",
    log:
        "logs/aggregate_to_substations/{interconnect}.log",
    threads: 1
    resources:
        mem_mb=lambda wildcards, input, attempt: (input.size // 150000) * attempt * 1.5,
        walltime=config_provider(
            "walltime", "aggregate_to_substations", default="01:00:00"
        ),
    script:
        "../scripts/aggregate_to_substations.py"


if "EGS" in config["electricity"]["extendable_carriers"]["Generator"]:

    rule aggregate_egs:
        input:
            specs=DATA + "EGS/{interconnect}/specs_EGS.nc",
            profile=DATA + "EGS/{interconnect}/profile_EGS.nc",
            busmap=BUSMAPS + "{interconnect}/busmap_s{simpl}.csv",
        output:
            specs=PROFILES + "{interconnect}/specs_EGS_s{simpl}.nc",
            profile=PROFILES + "{interconnect}/profile_EGS_s{simpl}.nc",
        log:
            LOGS + "{interconnect}/aggregate_egs_s{simpl}.log",
        threads: 1
        resources:
            mem_mb=lambda wildcards, input, attempt: (input.size // 200000)
            * attempt
            * 2,
            walltime=config_provider("walltime", "aggregate_egs", default="01:00:00"),
        script:
            "../scripts/aggregate_egs.py"


rule cluster_resources:
    params:
        aggregation_strategies=config["clustering"].get("aggregation_strategies", {}),
        focus_weights=config_provider("focus_weights", default=False),
        simplify_network=config_provider("clustering", "simplify_network"),
        planning_horizons=config_provider("scenario", "planning_horizons"),
    input:
        network=NETWORKS + "{interconnect}/elec_b.nc",
        regions_onshore=GEOSPATIAL + "{interconnect}/regions_onshore.geojson",
        regions_offshore=GEOSPATIAL + "{interconnect}/regions_offshore.geojson",
    output:
        network=NETWORKS + "{interconnect}/elec_s{simpl}.nc",
        regions_onshore=GEOSPATIAL + "{interconnect}/regions_onshore_s{simpl}.geojson",
        regions_offshore=GEOSPATIAL + "{interconnect}/regions_offshore_s{simpl}.geojson",
        busmap=BUSMAPS + "{interconnect}/busmap_s{simpl}.csv",
    log:
        "logs/cluster_resources/{interconnect}/elec_s{simpl}.log",
    threads: 1
    resources:
        mem_mb=lambda wildcards, input, attempt: (input.size // 150000) * attempt * 1.5,
        walltime=config_provider("walltime", "cluster_resources", default="01:00:00"),
    script:
        "../scripts/cluster_simpl.py"


rule cluster_network:
    params:
        cluster_network=config_provider("clustering", "cluster_network"),
        conventional_carriers=config_provider("electricity", "conventional_carriers"),
        renewable_carriers=config_provider("electricity", "renewable_carriers"),
        aggregation_strategies=config_provider("clustering", "aggregation_strategies"),
        custom_busmap=config_provider("enable", "custom_busmap", default=False),
        focus_weights=config_provider("focus_weights", default=False),
        length_factor=config_provider("lines", "length_factor"),
        costs=config_provider("costs"),
        planning_horizons=config_provider("scenario", "planning_horizons"),
        transmission_network=config_provider("model_topology", "transmission_network"),
        topological_boundaries=config_provider(
            "model_topology", "topological_boundaries"
        ),
        topology_aggregation=config_provider("model_topology", "aggregate"),
        s_max_pu=config_provider("lines", "s_max_pu", default=0.7),
    input:
        network=NETWORKS + "{interconnect}/elec_s{simpl}_l_pp.pkl",
        regions_onshore=GEOSPATIAL + "{interconnect}/regions_onshore_s{simpl}.geojson",
        regions_offshore=GEOSPATIAL + "{interconnect}/regions_offshore_s{simpl}.geojson",
        custom_busmap=(
            DATA + "{interconnect}/custom_busmap_{clusters}.csv"
            if config["enable"].get("custom_busmap", False)
            else []
        ),
        tech_costs=COSTS + f"costs_{config['scenario']['planning_horizons'][0]}.csv",
        itl_reeds_zone="repo_data/ReEDS_Constraints/transmission/transmission_capacity_init_AC_ba_NARIS2024.csv",
        itl_county="repo_data/ReEDS_Constraints/transmission/transmission_capacity_init_AC_county_NARIS2024.csv",
        itl_trans_grp="repo_data/ReEDS_Constraints/transmission/transmission_capacity_init_AC_transgrp_NARIS2024.csv",
        itl_costs_reeds_zone="repo_data/ReEDS_Constraints/transmission/transmission_distance_cost_500kVac_ba.csv",
        itl_costs_county="repo_data/ReEDS_Constraints/transmission/transmission_distance_cost_500kVac_county.csv",
        itl_state="repo_data/ReEDS_Constraints/transmission/transmission_capacity_init_AC_state_NARIS2024.csv",
        itl_costs_state="repo_data/ReEDS_Constraints/transmission/transmission_distance_cost_500kVac_state.csv",
    output:
        network=NETWORKS + "{interconnect}/elec_s{simpl}_c{clusters}.nc",
        regions_onshore=GEOSPATIAL
        + "{interconnect}/regions_onshore_s{simpl}_{clusters}.geojson",
        regions_offshore=GEOSPATIAL
        + "{interconnect}/regions_offshore_s{simpl}_{clusters}.geojson",
        busmap=BUSMAPS + "{interconnect}/busmap_s{simpl}_{clusters}.csv",
        linemap=BUSMAPS + "{interconnect}/linemap_s{simpl}_{clusters}.csv",
    log:
        "logs/cluster_network/{interconnect}/elec_s{simpl}_c{clusters}.log",
    benchmark:
        "benchmarks/cluster_network/{interconnect}/elec_s{simpl}_c{clusters}"
    threads: 1
    resources:
        walltime=config_provider("walltime", "cluster_network", default="01:30:00"),
        mem_mb=lambda wildcards, input, attempt: (input.size // 100000) * attempt * 2,
    script:
        "../scripts/cluster_network.py"


def flowgates_for_extra_components(wildcards):
    if not config["model_topology"]["transmission_network"] == "reeds":
        return []
    if config["model_topology"]["topological_boundaries"] == "county":
        return "repo_data/ReEDS_Constraints/transmission/transmission_capacity_init_AC_county_NARIS2024.csv"
    else:  # bas and states use the same flowgates
        return "repo_data/ReEDS_Constraints/transmission/transmission_capacity_init_AC_ba_NARIS2024.csv"


rule add_extra_components:
    input:
        **{
            f"phs_shp_{hour}": "repo_data/"
            + f"psh/40-100-dam-height-{hour}hr-no-croplands-no-ephemeral-no-highways.gpkg"
            for phs_tech in config["electricity"]["extendable_carriers"]["StorageUnit"]
            if "PHS" in phs_tech
            for hour in phs_tech.split("hr_")
            if hour.isdigit()
        },
        network=NETWORKS + "{interconnect}/elec_s{simpl}_c{clusters}.nc",
        tech_costs=lambda wildcards: expand(
            COSTS + "costs_{year}.csv",
            year=config["scenario"]["planning_horizons"],
        ),
        regions_onshore=GEOSPATIAL
        + "{interconnect}/regions_onshore_s{simpl}_{clusters}.geojson",
        flowgates=flowgates_for_extra_components,
        reeds_memberships="repo_data/ReEDS_Constraints/membership.csv",
        co2_storage=(
            CO2 + "{interconnect}/co2_storage_s{simpl}_{clusters}.csv"
            if config["scenario"]["sector"] == "" and config["co2"]["storage"] is True
            else []
        ),
        county_shapes=DATA + "counties/cb_2020_us_county_500k.shp",
    params:
        retirement=config["electricity"].get("retirement", "technical"),
        demand_response=config["electricity"].get("demand_response", {}),
        trim_network=config_provider("model_topology", "trim", default=False),
        imports=config_provider("electricity", "imports", default={}),
        exports=config_provider("electricity", "exports", default={}),
        weather_year=config_provider("renewable_weather_years"),
        eia_api=config_provider("api", "eia"),
        topological_boundaries=config_provider(
            "model_topology", "topological_boundaries"
        ),
        transmission_network=config_provider("model_topology", "transmission_network"),
        costs=config_provider("costs"),
    output:
        NETWORKS + "{interconnect}/elec_s{simpl}_c{clusters}_ec.nc",
    log:
        "logs/add_extra_components/{interconnect}/elec_s{simpl}_c{clusters}_ec.log",
    threads: 1
    resources:
        mem_mb=lambda wildcards, input, attempt: (input.size // 100000) * attempt * 2,
        walltime=config_provider("walltime", "add_extra_components", default="00:30:00"),
    group:
        "prepare"
    script:
        "../scripts/add_extra_components.py"


rule prepare_network:
    params:
        time_resolution=config_provider("clustering", "temporal", "resolution_elec"),
        links=config_provider("links"),
        lines=config_provider("lines"),
        co2limit=config_provider("electricity", "co2limit"),
        co2limit_enable=config_provider("electricity", "co2limit_enable", default=False),
        gaslimit=config_provider("electricity", "gaslimit"),
        gaslimit_enable=config_provider("electricity", "gaslimit_enable", default=False),
        transmission_network=config_provider("model_topology", "transmission_network"),
        costs=config_provider("costs"),
    input:
        network=(
            config["custom_files"]["files_path"]
            + config["custom_files"]["network_name"]
            if config["custom_files"].get("activate", False)
            else NETWORKS + "{interconnect}/elec_s{simpl}_c{clusters}_ec.nc"
        ),
        tech_costs=(
            config["custom_files"]["files_path"] + "costs_2030.csv"
            if config["custom_files"].get("activate", False)
            else RESOURCES
            + f"costs/costs_{config['scenario']['planning_horizons'][0]}.csv"
        ),
    output:
        NETWORKS + "{interconnect}/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}.nc",
    log:
        "logs/prepare_network/{interconnect}/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}.log",
    threads: 1
    resources:
        walltime=config_provider("walltime", "prepare_network", default="00:30:00"),
        mem_mb=lambda wildcards, input, attempt: (input.size // 100000) * attempt * 6,
    group:
        "prepare"
    script:
        "../scripts/prepare_network.py"
