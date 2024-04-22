################# ----------- Rules to Build Network ---------- #################


rule build_shapes:
    params:
        source_offshore_shapes=config["offshore_shape"],
        offwind_params=config["renewable"]["offwind"],
    input:
        zone=DATA + "breakthrough_network/base_grid/zone.csv",
        nerc_shapes="repo_data/NERC_Regions/NERC_Regions_Subregions.shp",
        reeds_shapes="repo_data/Reeds_Shapes/rb_and_ba_areas.shp",
        onshore_shapes="repo_data/BA_shapes_new/Modified_BE_BA_Shapes.shp",
        offshore_shapes_ca_osw="repo_data/BOEM_CA_OSW_GIS/CA_OSW_BOEM_CallAreas.shp",
        offshore_shapes_eez=DATA + "eez/conus_eez.shp",
        county_shapes=DATA + "counties/cb_2020_us_county_500k.shp",
    output:
        country_shapes=RESOURCES + "{interconnect}/country_shapes.geojson",
        onshore_shapes=RESOURCES + "{interconnect}/onshore_shapes.geojson",
        offshore_shapes=RESOURCES + "{interconnect}/offshore_shapes.geojson",
        state_shapes=RESOURCES + "{interconnect}/state_boundaries.geojson",
        reeds_shapes=RESOURCES + "{interconnect}/reeds_shapes.geojson",
        county_shapes=RESOURCES + "{interconnect}/county_shapes.geojson",
    log:
        "logs/build_shapes/{interconnect}.log",
    threads: 1
    resources:
        mem_mb=2000,
    script:
        "../scripts/build_shapes.py"


rule build_base_network:
    params:
        build_offshore_network=config["offshore_network"],
        links=config["links"],
    input:
        buses=DATA + "breakthrough_network/base_grid/bus.csv",
        lines=DATA + "breakthrough_network/base_grid/branch.csv",
        links=DATA + "breakthrough_network/base_grid/dcline.csv",
        bus2sub=DATA + "breakthrough_network/base_grid/bus2sub.csv",
        sub=DATA + "breakthrough_network/base_grid/sub.csv",
        onshore_shapes=RESOURCES + "{interconnect}/onshore_shapes.geojson",
        offshore_shapes=RESOURCES + "{interconnect}/offshore_shapes.geojson",
        state_shapes=RESOURCES + "{interconnect}/state_boundaries.geojson",
        reeds_shapes=RESOURCES + "{interconnect}/reeds_shapes.geojson",
        reeds_memberships="repo_data/ReEDS_Constraints/membership.csv",
        county_shapes=RESOURCES + "{interconnect}/county_shapes.geojson",
    output:
        bus2sub=DATA + "breakthrough_network/base_grid/{interconnect}/bus2sub.csv",
        sub=DATA + "breakthrough_network/base_grid/{interconnect}/sub.csv",
        bus_gis=RESOURCES + "{interconnect}/bus_gis.csv",
        lines_gis=RESOURCES + "{interconnect}/lines_gis.csv",
        network=RESOURCES + "{interconnect}/elec_base_network.nc",
    log:
        "logs/create_network/{interconnect}.log",
    threads: 1
    resources:
        mem_mb=1000,
    script:
        "../scripts/build_base_network.py"


rule build_bus_regions:
    params:
        aggregation_zone=config["clustering"]["cluster_network"]["aggregation_zones"],
    input:
        country_shapes=RESOURCES + "{interconnect}/country_shapes.geojson",
        state_shapes=RESOURCES + "{interconnect}/state_boundaries.geojson",
        ba_region_shapes=RESOURCES + "{interconnect}/onshore_shapes.geojson",
        reeds_shapes=RESOURCES + "{interconnect}/reeds_shapes.geojson",
        offshore_shapes=RESOURCES + "{interconnect}/offshore_shapes.geojson",
        base_network=RESOURCES + "{interconnect}/elec_base_network.nc",
        bus2sub=DATA + "breakthrough_network/base_grid/{interconnect}/bus2sub.csv",
        sub=DATA + "breakthrough_network/base_grid/{interconnect}/sub.csv",
    output:
        regions_onshore=RESOURCES + "{interconnect}/regions_onshore.geojson",
        regions_offshore=RESOURCES + "{interconnect}/regions_offshore.geojson",
    log:
        "logs/build_bus_regions/{interconnect}.log",
    threads: 1
    resources:
        mem_mb=1000,
    script:
        "../scripts/build_bus_regions.py"


rule build_cost_data:
    input:
        nrel_atb=DATA + "costs/nrel_atb.parquet",
        pypsa_technology_data=RESOURCES + "costs/{year}/pypsa_eur.csv",
    output:
        tech_costs=RESOURCES + "costs_{year}.csv",
    log:
        LOGS + "costs_{year}.log",
    threads: 1
    resources:
        mem_mb=1000,
    script:
        "../scripts/build_cost_data.py"


ATLITE_NPROCESSES = config["atlite"].get("nprocesses", 4)

if config["enable"].get("build_cutout", False):

    rule build_cutout:
        params:
            snapshots=config["snapshots"],
            cutouts=config["atlite"]["cutouts"],
            interconnects=config["atlite"]["interconnects"],
        input:
            regions_onshore=RESOURCES + "{interconnect}/country_shapes.geojson",
            regions_offshore=RESOURCES + "{interconnect}/offshore_shapes.geojson",
        output:
            protected("cutouts/" + CDIR + "{interconnect}_{cutout}.nc"),
        log:
            "logs/" + CDIR + "build_cutout/{interconnect}_{cutout}.log",
        benchmark:
            "benchmarks/" + CDIR + "build_cutout_{interconnect}_{cutout}"
        threads: ATLITE_NPROCESSES
        resources:
            mem_mb=ATLITE_NPROCESSES * 1000,
        conda:
            "envs/environment.yaml"
        script:
            "../scripts/build_cutout.py"


rule build_ship_raster:
    input:
        ship_density=DATA + "shipdensity_global.zip",
        cutouts=expand(
            "cutouts/" + CDIR + "western_{cutout}.nc",
            cutout=[
                config["renewable"][carrier]["cutout"]
                for carrier in config["electricity"]["renewable_carriers"]
            ],
        ),
    output:
        RESOURCES + "{interconnect}/shipdensity_raster.tif",
    log:
        LOGS + "{interconnect}/build_ship_raster.log",
    resources:
        mem_mb=5000,
    benchmark:
        BENCHMARKS + "{interconnect}/build_ship_raster"
    script:
        "../subworkflows/pypsa-eur/scripts/build_ship_raster.py"


rule build_hydro_profiles:
    params:
        hydro=config["renewable"]["hydro"],
        countries=config["countries"],
    input:
        ba_region_shapes=RESOURCES + "{interconnect}/onshore_shapes.geojson",
        # eia_hydro_generation=DATA + "eia_hydro_annual_generation.csv",
        cutout=f"cutouts/"
        + CDIR
        + "{interconnect}_"
        + config["renewable"]["hydro"]["cutout"]
        + ".nc",
    output:
        RESOURCES + "{interconnect}/profile_hydro.nc",
    log:
        LOGS + "{interconnect}/build_hydro_profile.log",
    resources:
        mem_mb=5000,
    conda:
        "envs/environment.yaml"
    script:
        "../scripts/build_hydro_profile.py"


rule build_renewable_profiles:
    params:
        renewable=config["renewable"],
        snapshots=config["snapshots"],
    input:
        base_network=RESOURCES + "{interconnect}/elec_base_network.nc",
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
        ship_density=lambda w: (
            RESOURCES + "{interconnect}/shipdensity_raster.tif"
            if "ship_threshold" in config["renewable"][w.technology].keys()
            else []
        ),
        country_shapes=RESOURCES + "{interconnect}/country_shapes.geojson",
        offshore_shapes=RESOURCES + "{interconnect}/offshore_shapes.geojson",
        cec_onwind="repo_data/CEC_Wind_BaseScreen_epsg3310.tif",
        cec_solar="repo_data/CEC_Solar_BaseScreen_epsg3310.tif",
        boem_osw="repo_data/boem_osw_planning_areas.tif",
        regions=lambda w: (
            RESOURCES + "{interconnect}/regions_onshore.geojson"
            if w.technology in ("onwind", "solar")
            else RESOURCES + "{interconnect}/regions_offshore.geojson"
        ),
        cutout=lambda w: "cutouts/"
        + CDIR
        + "{interconnect}_"
        + config["renewable"][w.technology]["cutout"]
        + ".nc",
    output:
        profile=RESOURCES + "{interconnect}/profile_{technology}.nc",
    log:
        LOGS + "{interconnect}/build_renewable_profile_{technology}.log",
    benchmark:
        BENCHMARKS + "{interconnect}/build_renewable_profiles_{technology}"
    threads: ATLITE_NPROCESSES
    resources:
        mem_mb=ATLITE_NPROCESSES * 5000,
    wildcard_constraints:
        technology="(?!hydro).*",  # Any technology other than hydro
    script:
        "../scripts/build_renewable_profiles.py"


rule build_demand:
    params:
        planning_horizons=config["scenario"]["planning_horizons"],
        demand_params=config["electricity"]["demand"],
        snapshots=config["snapshots"],
        eia_api=config["api"]["eia"],
    input:
        base_network=RESOURCES + "{interconnect}/elec_base_network.nc",
        eia=expand(DATA + "GridEmissions/{file}", file=DATAFILES_GE),
        efs=DATA + "nrel_efs/EFSLoadProfile_Reference_Moderate.csv",
    output:
        demand=RESOURCES + "{interconnect}/demand.csv",
    log:
        LOGS + "{interconnect}/build_demand.log",
    benchmark:
        BENCHMARKS + "{interconnect}/build_demand"
    threads: 2
    resources:
        mem_mb=interconnect_mem,
    script:
        "../scripts/build_demand.py"


def ba_gas_dynamic_fuel_price_files(wildcards):
    files = []
    if wildcards.interconnect in ("usa", "western"):
        files.append(DATA + "costs/caiso_ng_power_prices.csv")
    return files


rule build_fuel_prices:
    params:
        snapshots=config["snapshots"],
        api_eia=config["api"]["eia"],
    input:
        gas_balancing_area=ba_gas_dynamic_fuel_price_files,
    output:
        state_ng_fuel_prices=RESOURCES + "{interconnect}/state_ng_power_prices.csv",
        state_coal_fuel_prices=RESOURCES + "{interconnect}/state_coal_power_prices.csv",
        ba_ng_fuel_prices=RESOURCES + "{interconnect}/ba_ng_power_prices.csv",
    log:
        LOGS + "{interconnect}/build_fuel_prices.log",
    benchmark:
        BENCHMARKS + "{interconnect}/build_fuel_prices"
    threads: 1
    resources:
        mem_mb=800,
    script:
        "../scripts/build_fuel_prices.py"


def dynamic_fuel_price_files(wildcards):
    if config["conventional"]["dynamic_fuel_price"]:
        return {
            "state_ng_fuel_prices": RESOURCES
            + "{interconnect}/state_ng_power_prices.csv",
            "state_coal_fuel_prices": RESOURCES
            + "{interconnect}/state_coal_power_prices.csv",
            "ba_ng_fuel_prices": RESOURCES + "{interconnect}/ba_ng_power_prices.csv",
        }
    else:
        return {}


rule add_electricity:
    params:
        length_factor=config["lines"]["length_factor"],
        countries=config["countries"],
        renewable=config["renewable"],
        electricity=config["electricity"],
        conventional=config["conventional"],
        costs=config["costs"],
        planning_horizons=config["scenario"]["planning_horizons"],
        eia_api=config["api"]["eia"],
    input:
        unpack(dynamic_fuel_price_files),
        **{
            f"profile_{tech}": RESOURCES + "{interconnect}" + f"/profile_{tech}.nc"
            for tech in config["electricity"]["renewable_carriers"]
            if tech != "hydro"
        },
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
        base_network=RESOURCES + "{interconnect}/elec_base_network.nc",
        tech_costs=RESOURCES + f"costs_{config['costs']['year']}.csv",
        regions=RESOURCES + "{interconnect}/regions_onshore.geojson",
        plants_eia="repo_data/plants/plants_merged.csv",
        plants_ads="repo_data/plants/ads_plants_locs.csv",
        plants_breakthrough=DATA + "breakthrough_network/base_grid/plant.csv",
        hydro_breakthrough=DATA + "breakthrough_network/base_grid/hydro.csv",
        wind_breakthrough=DATA + "breakthrough_network/base_grid/wind.csv",
        solar_breakthrough=DATA + "breakthrough_network/base_grid/solar.csv",
        bus2sub=DATA + "breakthrough_network/base_grid/{interconnect}/bus2sub.csv",
        ads_renewables=(
            DATA + "WECC_ADS/processed/"
            if config["network_configuration"] == "ads2032"
            else []
        ),
        ads_2032=(
            DATA + "WECC_ADS/downloads/2032/Public Data/Hourly Profiles in CSV format"
            if config["network_configuration"] == "ads2032"
            else []
        ),
        demand=RESOURCES + "{interconnect}/demand.csv",
        fuel_costs="repo_data/plants/fuelCost22.csv",
    output:
        RESOURCES + "{interconnect}/elec_base_network_l_pp.nc",
    log:
        LOGS + "{interconnect}/add_electricity.log",
    benchmark:
        BENCHMARKS + "{interconnect}/add_electricity"
    threads: 1
    resources:
        mem_mb=80000,
    script:
        "../scripts/add_electricity.py"


################# ----------- Rules to Aggregate & Simplify Network ---------- #################
rule simplify_network:
    params:
        aggregation_strategies=config["clustering"].get("aggregation_strategies", {}),
    input:
        bus2sub=DATA + "breakthrough_network/base_grid/{interconnect}/bus2sub.csv",
        sub=DATA + "breakthrough_network/base_grid/{interconnect}/sub.csv",
        network=RESOURCES + "{interconnect}/elec_base_network_l_pp.nc",
    output:
        network=RESOURCES + "{interconnect}/elec_s.nc",
    log:
        "logs/simplify_network/{interconnect}/elec_s.log",
    threads: 1
    resources:
        mem_mb=interconnect_mem_s,
    script:
        "../scripts/simplify_network.py"


rule cluster_network:
    params:
        cluster_network=config["clustering"]["cluster_network"],
        conventional_carriers=config["electricity"].get("conventional_carriers", []),
        renewable_carriers=config["electricity"]["renewable_carriers"],
        aggregation_strategies=config["clustering"].get("aggregation_strategies", {}),
        custom_busmap=config["enable"].get("custom_busmap", False),
        focus_weights=config.get("focus_weights", None),
        max_hours=config["electricity"]["max_hours"],
        length_factor=config["lines"]["length_factor"],
        costs=config["costs"],
    input:
        network=RESOURCES + "{interconnect}/elec_s.nc",
        regions_onshore=RESOURCES + "{interconnect}/regions_onshore.geojson",
        regions_offshore=RESOURCES + "{interconnect}/regions_offshore.geojson",
        busmap=DATA + "breakthrough_network/base_grid/{interconnect}/bus2sub.csv",
        custom_busmap=(
            DATA + "{interconnect}/custom_busmap_{clusters}.csv"
            if config["enable"].get("custom_busmap", False)
            else []
        ),
        tech_costs=RESOURCES + f"costs_{config['costs']['year']}.csv",
    output:
        network=RESOURCES + "{interconnect}/elec_s_{clusters}.nc",
        regions_onshore=RESOURCES
        + "{interconnect}/regions_onshore_s_{clusters}.geojson",
        regions_offshore=RESOURCES
        + "{interconnect}/regions_offshore_s_{clusters}.geojson",
        busmap=RESOURCES + "{interconnect}/busmap_s_{clusters}.csv",
        linemap=RESOURCES + "{interconnect}/linemap_s_{clusters}.csv",
    log:
        "logs/cluster_network/{interconnect}/elec_s_{clusters}.log",
    benchmark:
        "benchmarks/cluster_network/{interconnect}/elec_s_{clusters}"
    threads: 1
    resources:
        mem_mb=interconnect_mem_c,
    script:
        "../scripts/cluster_network.py"


rule add_extra_components:
    input:
        network=RESOURCES + "{interconnect}/elec_s_{clusters}.nc",
        tech_costs=RESOURCES + f"costs_{config['costs']['year']}.csv",
    params:
        retirement=config["electricity"].get("retirement", "technical"),
    output:
        RESOURCES + "{interconnect}/elec_s_{clusters}_ec.nc",
    log:
        "logs/add_extra_components/{interconnect}/elec_s_{clusters}_ec.log",
    threads: 1
    resources:
        mem_mb=4000,
    group:
        "prepare"
    script:
        "../scripts/add_extra_components.py"


rule prepare_network:
    params:
        time_resolution=config["clustering"]["temporal"]["resolution_elec"],
        adjustments=False,
        links=config["links"],
        lines=config["lines"],
        co2base=config["electricity"]["co2base"],
        co2limit=config["electricity"]["co2limit"],
        co2limit_enable=config["electricity"]["co2limit_enable"],
        gaslimit=config["electricity"].get("gaslimit"),
        gaslimit_enable=config["electricity"].get("gaslimit_enable"),
        max_hours=config["electricity"]["max_hours"],
        costs=config["costs"],
        autarky=config["electricity"]["autarky"],
    input:
        network=RESOURCES + "{interconnect}/elec_s_{clusters}_ec.nc",
        tech_costs=RESOURCES + f"costs_{config['costs']['year']}.csv",
    output:
        RESOURCES + "{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}.nc",
    log:
        solver="logs/prepare_network/{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}.log",
    threads: 1
    resources:
        mem_mb=4000,
    group:
        "prepare"
    log:
        "logs/prepare_network",
    script:
        "../scripts/subworkflows/pypsa-eur/scripts/prepare_network.py"
