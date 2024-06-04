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
        snapshots=config["snapshots"],
        planning_horizons=config["scenario"]["planning_horizons"],
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
        bus2sub=RESOURCES + "{interconnect}/bus2sub.csv",
        sub=RESOURCES + "{interconnect}/sub.csv",
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
        bus2sub=RESOURCES + "{interconnect}/bus2sub.csv",
        sub=RESOURCES + "{interconnect}/sub.csv",
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
        pypsa_technology_data=RESOURCES + "costs/pypsa_eur_{year}.csv",
    output:
        tech_costs=RESOURCES + "costs/costs_{year}.csv",
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


rule build_hydro_profile:
    params:
        hydro=config_provider("renewable", "hydro"),
        snapshots=config_provider("snapshots"),
    input:
        reeds_shapes=RESOURCES + "{interconnect}/reeds_shapes.geojson",
        cutout=lambda w: f"cutouts/"
        + CDIR
        + "{interconnect}_"
        + config_provider("renewable", "hydro", "cutout")(w)
        + ".nc",
    output:
        profile=RESOURCES + "{interconnect}/profile_hydro.nc",
    log:
        LOGS + "{interconnect}/build_hydro_profile.log",
    resources:
        mem_mb=5000,
    conda:
        "../envs/environment.yaml"
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
        ship_density=[],
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


def electricty_study_demand(wildcards):
    profile = config["electricity"]["demand"]["profile"]
    if profile == "eia":
        return DATA + "GridEmissions/EIA_DMD_2018_2024.csv"
    elif profile == "efs":
        return DATA + "nrel_efs/EFSLoadProfile_Reference_Moderate.csv"
    else:
        return ""


def electricty_study_dissagregate(wildcards):
    strategy = config["electricity"]["demand"]["disaggregation"]
    if strategy == "pop":
        return ""
    elif strategy == "cliu":
        return DATA + "industry_load/2014_update_20170910-0116.csv"
    else:
        return ""


def sector_study_demand(wildcards):
    end_use = wildcards.end_use
    profile = config["sector"]["demand"]["profile"][end_use]
    if end_use == "residential":
        if profile == "eulp":
            return [
                DATA + f"eulp/res/{state}.csv"
                for state in INTERCONNECT_2_STATE[wildcards.interconnect]
            ]
        elif profile == "efs":
            return DATA + "nrel_efs/EFSLoadProfile_Reference_Moderate.csv"
        else:
            return ""
    elif end_use == "commercial":
        if profile == "eulp":
            return [
                DATA + f"eulp/com/{state}.csv"
                for state in INTERCONNECT_2_STATE[wildcards.interconnect]
            ]
        elif profile == "efs":
            return DATA + "nrel_efs/EFSLoadProfile_Reference_Moderate.csv"
        else:
            return ""
    elif end_use == "industry":
        if profile == "efs":
            return DATA + "nrel_efs/EFSLoadProfile_Reference_Moderate.csv"
        elif profile == "cliu":
            return [
                DATA + "industry_load/2014_update_20170910-0116.csv",  # cliu data
                DATA + "industry_load/epri_industrial_loads.csv",  # epri data
                DATA + "industry_load/table3_2.xlsx",  # mecs data
                DATA + "industry_load/fips_codes.csv",  # fips data
            ]
        else:
            return ""
    elif end_use == "transport":
        if profile == "efs":
            return DATA + "nrel_efs/EFSLoadProfile_Reference_Moderate.csv"
        else:
            return ""
    else:
        return ""


def sector_study_dissagregate(wildcards):
    end_use = wildcards.end_use
    strategy = config["sector"]["demand"]["disaggregation"][end_use]
    if end_use == "residential":
        if strategy == "pop":
            return ""
    elif end_use == "commercial":
        if strategy == "pop":
            return ""
    elif end_use == "industry":
        if strategy == "pop":
            return ""
        elif strategy == "cliu":
            return DATA + "industry_load/2014_update_20170910-0116.csv"
        else:
            return ""
    elif end_use == "transport":
        return ""
    else:
        return ""


rule build_electrical_demand:
    wildcard_constraints:
        end_use="power",  # added for consistency in build_demand.py
    params:
        demand_params=config["electricity"]["demand"],
        eia_api=config["api"]["eia"],
        profile_year=pd.to_datetime(config["snapshots"]["start"]).year,
    input:
        network=RESOURCES + "{interconnect}/elec_base_network.nc",
        demand_files=electricty_study_demand,
        eia=expand(DATA + "GridEmissions/{file}", file=DATAFILES_GE),
        efs=DATA + "nrel_efs/EFSLoadProfile_Reference_Moderate.csv",
        county_industrial_energy=DATA + "industry_load/2014_update_20170910-0116.csv",
    output:
        elec_demand=RESOURCES + "{interconnect}/{end_use}_electricity_demand.csv",
    log:
        LOGS + "{interconnect}/{end_use}_build_demand.log",
    benchmark:
        BENCHMARKS + "{interconnect}/{end_use}_build_demand"
    threads: 2
    resources:
        mem_mb=interconnect_mem,
    script:
        "../scripts/build_demand.py"


rule build_sector_demand:
    wildcard_constraints:
        end_use="residential|commercial|industry|transport",
    params:
        planning_horizons=config["scenario"]["planning_horizons"],
        profile_year=pd.to_datetime(config["snapshots"]["start"]).year,
        demand_params=config["sector"]["demand"],
        eia_api=config["api"]["eia"],
    input:
        network=RESOURCES + "{interconnect}/elec_base_network.nc",
        demand_files=sector_study_demand,
        county_industrial_energy=DATA + "industry_load/2014_update_20170910-0116.csv",
    output:
        elec_demand=RESOURCES + "{interconnect}/{end_use}_electricity_demand.csv",
        heat_demand=RESOURCES + "{interconnect}/{end_use}_heating_demand.csv",
        cool_demand=RESOURCES + "{interconnect}/{end_use}_cooling_demand.csv",
    log:
        LOGS + "{interconnect}/{end_use}_build_demand.log",
    benchmark:
        BENCHMARKS + "{interconnect}/{end_use}_build_demand"
    threads: 2
    resources:
        mem_mb=interconnect_mem,
    script:
        "../scripts/build_demand.py"


def demand_to_add(wildcards):
    if config["scenario"]["sector"] == "E":
        return RESOURCES + "{interconnect}/power_electricity_demand.csv"
    else:
        return [
            RESOURCES + "{interconnect}/residential_electricity_demand.csv",
            RESOURCES + "{interconnect}/residential_heating_demand.csv",
            RESOURCES + "{interconnect}/residential_cooling_demand.csv",
            RESOURCES + "{interconnect}/commercial_electricity_demand.csv",
            RESOURCES + "{interconnect}/commercial_heating_demand.csv",
            RESOURCES + "{interconnect}/commercial_cooling_demand.csv",
            RESOURCES + "{interconnect}/industry_electricity_demand.csv",
            RESOURCES + "{interconnect}/industry_heating_demand.csv",
            RESOURCES + "{interconnect}/industry_cooling_demand.csv",
            RESOURCES + "{interconnect}/transport_electricity_demand.csv",
        ]


rule add_demand:
    params:
        sectors=config["scenario"]["sector"],
        planning_horizons=config["scenario"]["planning_horizons"],
    input:
        network=RESOURCES + "{interconnect}/elec_base_network.nc",
        demand=demand_to_add,
    output:
        network=RESOURCES + "{interconnect}/elec_base_network_dem.nc",
    log:
        LOGS + "{interconnect}/add_demand.log",
    benchmark:
        BENCHMARKS + "{interconnect}/add_demand"
    resources:
        mem_mb=interconnect_mem,
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
        max_hours=config["electricity"]["max_hours"],
        renewable_carriers=config["electricity"]["renewable_carriers"],
        extendable_carriers=config["electricity"]["extendable_carriers"],
        conventional_carriers=config["electricity"]["conventional_carriers"],
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
        base_network=RESOURCES + "{interconnect}/elec_base_network_dem.nc",
        tech_costs=RESOURCES
        + f"costs/costs_{config['scenario']['planning_horizons'][0]}.csv",
        # attach first horizon costs
        regions=RESOURCES + "{interconnect}/regions_onshore.geojson",
        plants_eia="repo_data/plants/plants_merged.csv",
        plants_breakthrough=DATA + "breakthrough_network/base_grid/plant.csv",
        hydro_breakthrough=DATA + "breakthrough_network/base_grid/hydro.csv",
        bus2sub=RESOURCES + "{interconnect}/bus2sub.csv",
        fuel_costs="repo_data/plants/fuelCost22.csv",
    output:
        RESOURCES + "{interconnect}/elec_base_network_l_pp.nc",
    log:
        LOGS + "{interconnect}/add_electricity.log",
    benchmark:
        BENCHMARKS + "{interconnect}/add_electricity"
    threads: 1
    resources:
        mem_mb=interconnect_mem_a,
    script:
        "../scripts/add_electricity.py"


################# ----------- Rules to Aggregate & Simplify Network ---------- #################
rule simplify_network:
    params:
        aggregation_strategies=config["clustering"].get("aggregation_strategies", {}),
    input:
        bus2sub=RESOURCES + "{interconnect}/bus2sub.csv",
        sub=RESOURCES + "{interconnect}/sub.csv",
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
        cluster_network=config_provider("clustering", "cluster_network"),
        conventional_carriers=config_provider("electricity", "conventional_carriers"),
        renewable_carriers=config_provider("electricity", "renewable_carriers"),
        aggregation_strategies=config_provider("clustering", "aggregation_strategies"),
        custom_busmap=config_provider("enable", "custom_busmap", default=False),
        focus_weights=config_provider("focus_weights", default=False),
        max_hours=config_provider("electricity", "max_hours"),
        length_factor=config_provider("lines", "length_factor"),
        costs=config_provider("costs"),
        planning_horizons=config_provider("scenario", "planning_horizons"),
    input:
        network=RESOURCES + "{interconnect}/elec_s.nc",
        regions_onshore=RESOURCES + "{interconnect}/regions_onshore.geojson",
        regions_offshore=RESOURCES + "{interconnect}/regions_offshore.geojson",
        busmap=RESOURCES + "{interconnect}/bus2sub.csv",
        custom_busmap=(
            DATA + "{interconnect}/custom_busmap_{clusters}.csv"
            if config["enable"].get("custom_busmap", False)
            else []
        ),
        tech_costs=RESOURCES
        + f"costs/costs_{config['scenario']['planning_horizons'][0]}.csv",
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
        tech_costs=lambda wildcards: expand(
            RESOURCES + "costs/costs_{year}.csv",
            year=config["scenario"]["planning_horizons"],
        ),
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
        time_resolution=config_provider("clustering", "temporal", "resolution_elec"),
        adjustments=False,
        links=config_provider("links"),
        lines=config_provider("lines"),
        co2base=config_provider("electricity", "co2base"),
        co2limit=config_provider("electricity", "co2limit"),
        co2limit_enable=config_provider("electricity", "co2limit_enable", default=False),
        gaslimit=config_provider("electricity", "gaslimit"),
        gaslimit_enable=config_provider("electricity", "gaslimit_enable", default=False),
        max_hours=config_provider("electricity", "max_hours"),
        costs=config_provider("costs"),
        autarky=config_provider("electricity", "autarky"),
    input:
        network=(
                    config["custom_files"]["files_path"] + config["custom_files"]["network_name"]
                    if config["custom_files"].get("activate", False)
                    else RESOURCES + "{interconnect}/elec_s_{clusters}_ec.nc"
        ),
        tech_costs=(
                    config["custom_files"]["files_path"] + 'costs_2030.csv'
                    if config["custom_files"].get("activate", False)
                    else RESOURCES + f"costs/costs_{config['scenario']['planning_horizons'][0]}.csv"
        ),
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
        "../scripts/prepare_network.py"
