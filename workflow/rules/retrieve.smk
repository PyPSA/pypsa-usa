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
    "counties/cb_2020_us_county_500k.shp",
]


def define_zenodo_databundles():
    return {
        "USATestSystem": "https://zenodo.org/record/4538590/files/USATestSystem.zip",
        "pypsa_usa_data": "https://zenodo.org/records/14219029/files/pypsa_usa_data.zip",
    }


def define_sector_databundles():
    return {
        "pypsa_usa_sec": "https://zenodo.org/records/14291626/files/pypsa_usa_sector_data.zip"
    }


rule retrieve_zenodo_databundles:
    params:
        define_zenodo_databundles(),
    output:
        expand(
            DATA + "breakthrough_network/base_grid/{file}", file=breakthrough_datafiles
        ),
        expand(DATA + "{file}", file=pypsa_usa_datafiles),
    resources:
        mem_mb=5000,
    log:
        "logs/retrieve/retrieve_databundles.log",
    script:
        "../scripts/retrieve_databundles.py"


# EGS seismic risk mask (Zenodo 10.5281/zenodo.18793962)
rule retrieve_seismic_risk_mask:
    output:
        DATA + "seismic_risk_exclusion/seismic_risk_mask.geojson",
    resources:
        mem_mb=5000,
        walltime="00:10:00",
    log:
        "logs/retrieve/retrieve_seismic_risk_mask.log",
    script:
        "../scripts/retrieve_seismic_risk_mask.py"


def efs_databundle(wildcards):
    return {
        "EFS": f"https://data.nlr.gov/system/files/126/EFSLoadProfile_{wildcards.efs_case}_{wildcards.efs_speed}.zip"
    }


rule retrieve_nrel_efs_data:
    wildcard_constraints:
        efs_case="Reference|Medium|High",
        efs_speed="Slow|Moderate|Rapid",
    params:
        efs_databundle,
    output:
        DATA + "nrel_efs/EFSLoadProfile_{efs_case}_{efs_speed}.csv",
    resources:
        mem_mb=5000,
    log:
        "logs/retrieve/retrieve_efs_{efs_case}_{efs_speed}.log",
    script:
        "../scripts/retrieve_databundles.py"


rule retrieve_eer_demand_data:
    wildcard_constraints:
        eer_file="demand_EER2025_100by2050|demand_EER2025_Baseline_AEO2023|demand_EER2025_IRAlow",
    params:
        url=lambda wildcards: f"https://zenodo.org/records/18435264/files/{wildcards.eer_file}.h5?download=1",
    output:
        DATA + "eer/{eer_file}.h5",
    resources:
        mem_mb=5000,
    log:
        "logs/retrieve/retrieve_eer_{eer_file}.log",
    script:
        "../scripts/retrieve_eer_data.py"


CPUC_SERVM_URL = "https://files.cpuc.ca.gov/energy/modeling/2026_servm_updates/"


rule retrieve_cpuc_servm_load:
    wildcard_constraints:
        servm_year="2026|2028|2030|2032|2035|2037|2040|2042|2045",
    params:
        url=lambda wildcards: CPUC_SERVM_URL
        + f"HourlyLoad_CA_Regions_V2025E_2224_Mon_{wildcards.servm_year}.csv",
    output:
        DATA + "cpuc/servm/HourlyLoad_CA_Regions_V2025E_2224_Mon_{servm_year}.csv",
    resources:
        mem_mb=5000,
    log:
        "logs/retrieve/retrieve_cpuc_servm_load_{servm_year}.log",
    retries: 2
    script:
        "../scripts/retrieve_cpuc_data.py"


rule retrieve_cpuc_baseline_generators:
    params:
        url=CPUC_SERVM_URL + "BaselineGeneratorList_CAISO.xlsx",
    output:
        DATA + "cpuc/BaselineGeneratorList_CAISO.xlsx",
    resources:
        mem_mb=5000,
    log:
        "logs/retrieve/retrieve_cpuc_baseline_generators.log",
    retries: 2
    script:
        "../scripts/retrieve_cpuc_data.py"


# RESERVED (phase 2): `retrieve_cpuc_thermal_derate` will pull the CPUC SERVM
# unit-specific ambient-temperature derate profiles that back the
# `conventional.ambient_derate` config hook. Until it lands, enabling that key
# raises NotImplementedError in add_electricity.


sector_datafiles = [
    # heating sector
    "population/DECENNIALDHC2020.P1-Data.csv",
    "urbanization/DECENNIALDHC2020.H2-Data.csv",
    # natural gas
    "natural_gas/EIA-757.csv",
    "natural_gas/EIA-StatetoStateCapacity_Jan2023.xlsx",
    "natural_gas/EIA-StatetoStateCapacity_Feb2024.xlsx",
    "natural_gas/pipelines.geojson",
    # industrial demand
    "industry_load/2014_update_20170910-0116.csv",
    "industry_load/epri_industrial_loads.csv",
    "industry_load/fips_codes.csv",
    "industry_load/table3_2.xlsx",
]


rule retrieve_sector_databundle:
    params:
        define_sector_databundles(),
    output:
        expand(DATA + "{file}", file=sector_datafiles),
    log:
        LOGS + "retrieve_sector_databundle.log",
    retries: 2
    script:
        "../scripts/retrieve_databundles.py"


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
        walltime="00:40:00",
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
    retries: 3
    params:
        stock="res",
        profiles=RESSTOCK_FILES,
        save_dir=DATA + "eulp/res",
    output:
        expand(DATA + "eulp/res/{{state}}/{profile}.csv", profile=RESSTOCK_FILES),
        DATA + "eulp/res/{state}.csv",
    script:
        "../scripts/retrieve_eulp.py"


rule retrieve_com_eulp:
    log:
        "logs/retrieve/retrieve_com_eulp/{state}.log",
    retries: 3
    params:
        stock="com",
        profiles=COMSTOCK_FILES,
        save_dir=DATA + "eulp/com",
    output:
        expand(DATA + "eulp/com/{{state}}/{profile}.csv", profile=COMSTOCK_FILES),
        DATA + "eulp/com/{state}.csv",
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


if (
    not config["enable"].get("build_cutout", False)
    and not config["renewable"]["dataset"] == "godeeep"
):

    rule retrieve_cutout:
        input:
            HTTP.remote(
                "zenodo.org/records/14611937/files/usa_{cutout}.nc",
                static=True,
            ),
        output:
            "cutouts/" + CDIR + "usa_{cutout}.nc",
        log:
            "logs/" + CDIR + "retrieve_cutout_usa_{cutout}.log",
        resources:
            walltime="00:50:00",
            mem_mb=5000,
        retries: 2
        run:
            move(input[0], output[0])


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
        walltime="00:10:00",
        mem_mb=2000,
    script:
        "../scripts/retrieve_caiso_data.py"


rule retrieve_pudl:
    output:
        pudl_ferc714=DATA + "pudl/out_ferc714__hourly_estimated_state_demand.parquet",
        census=DATA + "pudl/censusdp1tract.sqlite",
    log:
        LOGS + "retrieve_pudl.log",
    resources:
        walltime="00:30:00",
        mem_mb=5000,
    script:
        "../scripts/retrieve_pudl.py"


rule retrieve_nrel_exclusion_artifact:
    """
    Download a single NREL land-access availability/caps artifact from the
    Zenodo bundle (record nrel_exclusion_v1). Triggered on-demand by
    build_renewable_profiles when avail_*.nc / caps_*.nc are missing locally.
    """
    wildcard_constraints:
        nrel_artifact=r"(avail|caps)_(solar|onwind|offwind|offwind_floating)_(reference|limited|open)(_cec|_boem)?",
    output:
        DATA + "nrel_exclusion/derived/{nrel_artifact}.nc",
    log:
        LOGS + "retrieve/nrel_exclusion_{nrel_artifact}.log",
    resources:
        walltime="00:10:00",
        mem_mb=2000,
    script:
        "../scripts/retrieve_nrel_exclusion.py"


# GODEEEP climate scenarios; the scenario is a directory/record component, never
# part of the CF file name (see godeeep_cf_registry.cf_filename).
GODEEEP_SCENARIOS = (
    "historical",
    "rcp45cooler",
    "rcp45hotter",
    "rcp85cooler",
    "rcp85hotter",
)


rule retrieve_godeeep_cf:
    """
    Place one compressed GODEEEP capacity-factor file where
    build_renewable_profiles expects it. The source (local mirror or Zenodo
    record) is declared per (dataset key, year) in the `godeeep_cf_registry`
    config block; an undeclared dataset/year raises instead of falling back to
    another year, hub height or screening variant.
    """
    wildcard_constraints:
        scenario="|".join(GODEEEP_SCENARIOS),
        cf_file=r"(solar|wind)_gen_cf_\d{4}(_\d+m)?_compressed",
    output:
        DATA + "godeeep/{scenario}/{cf_file}.nc",
    log:
        LOGS + "retrieve/godeeep_cf_{scenario}_{cf_file}.log",
    resources:
        walltime="01:00:00",
        mem_mb=2000,
    script:
        "../scripts/retrieve_godeeep_cf.py"


if "EGS" in config["electricity"]["extendable_carriers"]["Generator"]:

    rule retrieve_egs:
        params:
            dispatch=config["renewable"]["EGS"]["dispatch"],
            subdir=DATA + "EGS/{interconnect}",
        output:
            DATA + "EGS/{interconnect}/specs_EGS.nc",
            DATA + "EGS/{interconnect}/profile_EGS.nc",
        resources:
            walltime="00:30:00",
            mem_mb=5000,
        log:
            LOGS + "retrieve_EGS_{interconnect}.log",
        script:
            "../scripts/retrieve_egs.py"
