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
        'USATestSystem':"https://zenodo.org/record/4538590/files/USATestSystem.zip",
        'pypsa_usa_data':"https://zenodo.org/records/10480944/files/pypsa_usa_data.zip"
        }

def define_sector_databundles():
    return {
        'pypsa_usa_sec':"https://zenodo.org/records/10637836/files/pypsa_usa_sector_data.zip?download=1"
        }

rule retrieve_zenodo_databundles:
    params:
        define_zenodo_databundles()
    output:
        expand(DATA + "breakthrough_network/base_grid/{file}", file=breakthrough_datafiles),
        expand(DATA + "{file}", file=pypsa_usa_datafiles),
    log:
        "logs/retrieve/retrieve_databundles.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/retrieve_databundles.py"    

def define_nrel_databundles():
    return {
        'EFS':"https://data.nrel.gov/system/files/126/EFSLoadProfile_Reference_Moderate.zip"
        }

# rule retrieve_nrel_efs_data:
#     params:
#         define_nrel_databundles()
#     output:
#         DATA + "nrel_efs/EFSLoadProfile_Reference_Moderate.csv",
#     log:
#         "logs/retrieve/retrieve_databundles.log",
#     conda:
#         "../envs/environment.yaml"
#     script:
#         "../scripts/retrieve_databundles.py"   

sector_datafiles = [
    "counties/cb_2020_us_county_500k.shp",
    "population/DECENNIALDHC2020.P1-Data.csv",
    "urbanization/DECENNIALDHC2020.H2-Data.csv"
]

rule retrieve_sector_databundle:
    params:
        define_sector_databundles()
    output:
        expand(DATA + "{file}", file=sector_datafiles)
    log:
        LOGS + "retrieve_sector_databundle.log",
    retries: 2
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/retrieve_databundles.py"

if config["network_configuration"] == 'ads2032':
    rule retrieve_WECC_forecast_data:
        output:
            ads_2032 = directory(DATA + "WECC_ADS/downloads/2032/Public Data/Hourly Profiles in CSV format"),
            ads_2030 = directory(DATA + "WECC_ADS/downloads/2030/WECC 2030 ADS PCM 2020-12-16 (V1.5) Public Data/CSV Shape Files"),
            ads_dir = directory(DATA + "WECC_ADS/processed"),
        log:
            "logs/retrieve/retrieve_WECC_forecast_data.log",
        script:
            "../scripts/retrieve_forecast_data.py"


if config["enable"].get("download_eia", False):
    rule retrieve_eia_data:
        output:
            expand(DATA + "eia/{file}", file=DATAFILES_DMD),
        log:
            "logs/retrieve/retrieve_historical_load_data.log",
        script:
            "../scripts/retrieve_eia_data.py"

DATAFILES_DMD = ["EIA_DMD_2018_2024.csv"]

rule retrieve_eia_data:
    output:
        expand(DATA + "GridEmissions/{file}", file=DATAFILES_DMD),
    log:
        "logs/retrieve/retrieve_historical_load_data.log",
    resources:
        mem_mb=5000,
    script:
        "../scripts/retrieve_eia_data.py"


rule retrieve_ship_raster:
    input:
        HTTP.remote(
            "https://zenodo.org/record/6953563/files/shipdensity_global.zip",
            keep_local=True,
            static=True,
        ),
    output:
        DATA +"shipdensity_global.zip",
    log:
        LOGS + "retrieve_ship_raster.log",
    resources:
        mem_mb=5000,
    retries: 2
    run:
        move(input[0], output[0])

if config["enable"].get("download_cutout", False):
    rule retrieve_cutout:
        input:
            HTTP.remote(
                'zenodo.org/records/10067222/files/{interconnect}_{cutout}.nc'
                ,static=True),
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
        pypsa_technology_data = RESOURCES + "costs/{year}/pypsa_eur.csv",
    params:
        pypsa_costs_version = config["costs"].get("version", "v0.6.0")
    log:
        LOGS + "retrieve_cost_data_eur_{year}.log",
    resources:
        mem_mb=1000,
    script:
        "../scripts/retrieve_cost_data_eur.py"

rule retrieve_cost_data_usa:
    output:
        nrel_atb = DATA + "costs/nrel_atb.parquet",
        # nrel_atb_transport = DATA + "costs/nrel_atb_transport.xlsx",
        ng_electric_power_price = DATA + "costs/ng_electric_power_price.csv",
        ng_industrial_price = DATA + "costs/ng_industrial_price.csv",
        ng_residential_price = DATA + "costs/ng_commercial_price.csv",
        ng_commercial_price = DATA + "costs/ng_residential_price.csv",
    params:
        eia_api_key = config["costs"].get("eia_aip_key", None),
    log:
        LOGS + "retrieve_cost_data_usa.log",
    resources:
        mem_mb=1000,
    script:
        "../scripts/retrieve_cost_data_usa.py"