# Rules to Retrieve Data

breakthrough_datafiles = [
    "bus.csv",
    "sub.csv",
    "bus2sub.csv",
    "branch.csv",
    "dcline.csv",
    "demand.csv",
    "plant.csv",
    "solar.csv",
    "wind.csv",
    "hydro.csv",
    "zone.csv",
]

pypsa_usa_datafiles = [
    "gebco/gebco_2023_tid_USA.nc",
    "gebco/gebco_2023_n55.0_s10.0_w-126.0_e-65.0.tif",
    "copernicus/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_USA_EPSG-4326.tif",
    "eez/conus_eez.shp",
    "natura.tiff",
]


def define_zenodo_databundles():
    return {
        "USATestSystem": "https://zenodo.org/record/4538590/files/USATestSystem.zip",
        "pypsa_usa_data": "https://zenodo.org/records/10995249/files/pypsa_usa_data.zip",
    }


def define_sector_databundles():
    return {
        "pypsa_usa_sec": "https://zenodo.org/records/10637836/files/pypsa_usa_sector_data.zip?download=1"
    }


rule retrieve_zenodo_databundles:
    params:
        define_zenodo_databundles(),
    output:
        expand(
            DATA + "breakthrough_network/base_grid/{file}", file=breakthrough_datafiles
        ),
        expand(DATA + "{file}", file=pypsa_usa_datafiles),
    log:
        "logs/retrieve/retrieve_databundles.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/retrieve_databundles.py"


def define_nrel_databundles():
    return {
        "EFS": "https://data.nrel.gov/system/files/126/EFSLoadProfile_Reference_Moderate.zip"
    }


rule retrieve_nrel_efs_data:
    params:
        define_nrel_databundles(),
    output:
        DATA + "nrel_efs/EFSLoadProfile_Reference_Moderate.csv",
    log:
        "logs/retrieve/retrieve_databundles.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/retrieve_databundles.py"


sector_datafiles = [
    # general
    "counties/cb_2020_us_county_500k.shp",
    # heating sector
    "population/DECENNIALDHC2020.P1-Data.csv",
    "urbanization/DECENNIALDHC2020.H2-Data.csv",
    # natural gas
    "natural_gas/EIA-757.csv",
    "natural_gas/EIA-StatetoStateCapacity_Jan2023.xlsx",
    "natural_gas/pipelines.geojson",
]


rule retrieve_sector_databundle:
    params:
        define_sector_databundles(),
    output:
        expand(DATA + "{file}", file=sector_datafiles),
    log:
        LOGS + "retrieve_sector_databundle.log",
    retries: 2
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/retrieve_databundles.py"


if config["network_configuration"] == "ads2032":

    rule retrieve_WECC_forecast_data:
        output:
            ads_2032=directory(
                DATA
                + "WECC_ADS/downloads/2032/Public Data/Hourly Profiles in CSV format"
            ),
            ads_2030=directory(
                DATA
                + "WECC_ADS/downloads/2030/WECC 2030 ADS PCM 2020-12-16 (V1.5) Public Data/CSV Shape Files"
            ),
            ads_dir=directory(DATA + "WECC_ADS/processed"),
        log:
            "logs/retrieve/retrieve_WECC_forecast_data.log",
        script:
            "../scripts/retrieve_forecast_data.py"


DATAFILES_GE = [
    "EIA_DMD_2018_2024.csv",
    "EIA_GridEmissions_all_2018_2024.csv",
    "GridEmissions_co2_2018_2024.csv",
]


rule retrieve_gridemissions_data:
    output:
        expand(DATA + "GridEmissions/{file}", file=DATAFILES_GE),
    log:
        "logs/retrieve/retrieve_gridemissions_data.log",
    resources:
        mem_mb=5000,
    script:
        "../scripts/retrieve_gridemissions_data.py"


RESSTOCK_FILES = [
    "mobile_home",
    "multi-family_with_2_-_4_units",
    "multi-family_with_5plus_units",
    "single-family_attached",
    "single-family_detached",
]

COMSTOCK_FILES = [
    "fullservicerestaurant",
    "hospital",
    "largehotel",
    "largeoffice",
    "mediumoffice",
    "outpatient",
    "primaryschool",
    "quickservicerestaurant",
    "retailstandalone",
    "retailstripmall",
    "secondaryschool",
    "smallhotel",
    "smalloffice",
    "warehouse",
]

# need seperate rules cause cant access params in output
# https://github.com/snakemake/snakemake/issues/1122


rule retrieve_res_eulp:
    log:
        "logs/retrieve/retrieve_res_eulp/{state}.log",
    params:
        stock="res",
        profiles=RESSTOCK_FILES,
        save_dir=DATA + "eulp/res/",
    output:
        expand(DATA + "eulp/res/{{state}}/{profile}.csv", profile=RESSTOCK_FILES),
    script:
        "../scripts/retrieve_eulp.py"


rule retrieve_com_eulp:
    log:
        "logs/retrieve/retrieve_com_eulp/{state}.log",
    params:
        stock="com",
        profiles=COMSTOCK_FILES,
        save_dir=DATA + "eulp/com/",
    output:
        expand(DATA + "eulp/com/{{state}}/{profile}.csv", profile=COMSTOCK_FILES),
    script:
        "../scripts/retrieve_eulp.py"


rule retrieve_ship_raster:
    input:
        HTTP.remote(
            "https://zenodo.org/record/6953563/files/shipdensity_global.zip",
            keep_local=True,
            static=True,
        ),
    output:
        DATA + "shipdensity_global.zip",
    log:
        LOGS + "retrieve_ship_raster.log",
    resources:
        mem_mb=5000,
    retries: 2
    run:
        move(input[0], output[0])


rule retrieve_cutout:
    input:
        HTTP.remote(
            "zenodo.org/records/10067222/files/{interconnect}_{cutout}.nc", static=True
        ),
    output:
        "cutouts/" + CDIR + "{interconnect}_{cutout}.nc",
    log:
        "logs/" + CDIR + "retrieve_cutout_{interconnect}_{cutout}.log",
    resources:
        mem_mb=5000,
    retries: 2
    run:
        move(input[0], output[0])


rule retrieve_cost_data_eur:
    output:
        pypsa_technology_data=RESOURCES + "costs/{year}/pypsa_eur.csv",
    params:
        pypsa_costs_version=config["costs"].get("version", "v0.6.0"),
    log:
        LOGS + "retrieve_cost_data_eur_{year}.log",
    resources:
        mem_mb=1000,
    script:
        "../scripts/retrieve_cost_data_eur.py"


rule retrieve_cost_data_usa:
    output:
        nrel_atb=DATA + "costs/nrel_atb.parquet",
        # nrel_atb_transport = DATA + "costs/nrel_atb_transport.xlsx",
    params:
        # eia_api_key = config["api"].get("eia", None),
        eia_api_key=None,
    log:
        LOGS + "retrieve_cost_data_usa.log",
    resources:
        mem_mb=1000,
    script:
        "../scripts/retrieve_cost_data_usa.py"


rule retrieve_caiso_data:
    params:
        fuel_year=config["costs"]["ng_fuel_year"],
    input:
        fuel_regions="repo_data/plants/wecc_fuelregions.xlsx",
    output:
        fuel_prices=DATA + "costs/caiso_ng_power_prices.csv",
    log:
        LOGS + "retrieve_caiso_data.log",
    shadow:
        "minimal"
    resources:
        mem_mb=2000,
    script:
        "../scripts/retrieve_caiso_data.py"
