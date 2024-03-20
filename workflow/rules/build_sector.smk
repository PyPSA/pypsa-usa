"""Rules for building sector coupling network"""


def sector_input_files(wildcards):
    input_files = {
        "network": RESOURCES + "{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}.nc"
    }
    sectors = wildcards.sector.split("-")
    if "G" in sectors:
        ng_files = {
            "county": DATA + "counties/cb_2020_us_county_500k.shp",
            "pipeline_capacity": DATA
            + "natural_gas/EIA-StatetoStateCapacity_Jan2023.xlsx",
            "pipeline_shape": DATA + "natural_gas/pipelines.geojson",
            "eia_757": DATA + "natural_gas/EIA-757.csv",
        }
        input_files.update(ng_files)

    return input_files


rule add_sectors:
    params:
        electricity=config["electricity"],
        costs=config["costs"],
        plotting=config["plotting"],
        natural_gas=config["sector"].get("natural_gas", None),
        snapshots=config["snapshots"],
        api=config["api"],
    input:
        unpack(sector_input_files),
    output:
        network=RESOURCES
        + "{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}.nc",
    log:
        "logs/add_sectors/{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}.log",
    group:
        "prepare"
    threads: 1
    resources:
        mem_mb=4000,
    script:
        "../scripts/add_sectors.py"


rule build_population_layouts:
    input:
        county_shapes=DATA + "counties/cb_2020_us_county_500k.shp",
        urban_percent=DATA + "urbanization/DECENNIALDHC2020.H2-Data.csv",
        population=DATA + "population/DECENNIALDHC2020.P1-Data.csv",
        cutout="cutouts/"
        + CDIR
        + "{interconnect}_"
        + config["atlite"]["default_cutout"]
        + ".nc",
    output:
        pop_layout_total=RESOURCES + "{interconnect}/pop_layout_total.nc",
        pop_layout_urban=RESOURCES + "{interconnect}/pop_layout_urban.nc",
        pop_layout_rural=RESOURCES + "{interconnect}/pop_layout_rural.nc",
    log:
        LOGS + "{interconnect}/build_population_layouts.log",
    resources:
        mem_mb=20000,
    benchmark:
        BENCHMARKS + "{interconnect}/build_population_layouts"
    threads: 8
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_population_layouts.py"


rule build_heat_demands:
    params:
        snapshots=config["snapshots"],
    input:
        pop_layout=RESOURCES + "{interconnect}/pop_layout_{scope}.nc",
        # regions_onshore=RESOURCES + "regions_onshore_elec_s{simpl}_{clusters}.geojson",
        regions_onshore=RESOURCES
        + "{interconnect}/regions_onshore_s_{clusters}.geojson",
        cutout="cutouts/"
        + CDIR
        + "{interconnect}_"
        + config["atlite"]["default_cutout"]
        + ".nc",
    output:
        # heat_demand=RESOURCES + "heat_demand_{scope}_elec_s{simpl}_{clusters}.nc",
        heat_demand=RESOURCES
        + "{interconnect}/heat_demand_{scope}_elec_s_{clusters}.nc",
    resources:
        mem_mb=20000,
    threads: 8
    log:
        # LOGS + "build_heat_demands_{scope}_{simpl}_{clusters}.loc",
        LOGS + "{interconnect}/build_heat_demands_{scope}_{clusters}.loc",
    benchmark:
        BENCHMARKS + "{interconnect}/build_heat_demands/{scope}_s_{clusters}"
        # BENCHMARKS + "build_heat_demands/{scope}_s{simpl}_{clusters}"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_heat_demand.py"


rule build_temperature_profiles:
    params:
        snapshots=config["snapshots"],
    input:
        pop_layout=RESOURCES + "{interconnect}/pop_layout_{scope}.nc",
        # regions_onshore = RESOURCES + "regions_onshore_elec_s{simpl}_{clusters}.geojson",
        regions_onshore=RESOURCES
        + "{interconnect}/regions_onshore_s_{clusters}.geojson",
        cutout="cutouts/"
        + CDIR
        + "{interconnect}_"
        + config["atlite"]["default_cutout"]
        + ".nc",
    output:
        # temp_soil = RESOURCES + "temp_soil_{scope}_elec_s{simpl}_{clusters}.nc",
        # temp_air = RESOURCES + "temp_air_{scope}_elec_s{simpl}_{clusters}.nc",
        temp_soil=RESOURCES + "{interconnect}/temp_soil_{scope}_elec_s_{clusters}.nc",
        temp_air=RESOURCES + "{interconnect}/temp_air_{scope}_elec_s_{clusters}.nc",
    resources:
        mem_mb=20000,
    threads: 8
    log:
        LOGS + "{interconnect}/build_temperature_profiles_{scope}_{clusters}.log",
        # LOGS + "build_temperature_profiles_{scope}_{simpl}_{clusters}.log",
    benchmark:
        # BENCHMARKS + "build_temperature_profiles/{scope}_s{simpl}_{clusters}"
        BENCHMARKS + "{interconnect}/build_temperature_profiles/{scope}_s_{clusters}"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_temperature_profiles.py"


# rule build_solar_thermal_profiles:
#     params:
#         snapshots=config["snapshots"],
#         solar_thermal=config["solar_thermal"],
#     input:
#         pop_layout=RESOURCES + "pop_layout_{scope}.nc",
#         regions_onshore=RESOURCES + "regions_onshore_elec_s{simpl}_{clusters}.geojson",
#         cutout="cutouts/" + CDIR + config["atlite"]["default_cutout"] + ".nc",
#     output:
#         solar_thermal=RESOURCES + "solar_thermal_{scope}_elec_s{simpl}_{clusters}.nc",
#     resources:
#         mem_mb=20000,
#     threads: 16
#     log:
#         LOGS + "build_solar_thermal_profiles_{scope}_s{simpl}_{clusters}.log",
#     benchmark:
#         BENCHMARKS + "build_solar_thermal_profiles/{scope}_s{simpl}_{clusters}"
#     conda:
#         "../envs/environment.yaml"
#     script:
#         "../scripts/build_solar_thermal_profiles.py"


rule build_simplified_population_layouts:
    input:
        pop_layout_total=RESOURCES + "{interconnect}/pop_layout_total.nc",
        pop_layout_urban=RESOURCES + "{interconnect}/pop_layout_urban.nc",
        pop_layout_rural=RESOURCES + "{interconnect}/pop_layout_rural.nc",
        # regions_onshore=RESOURCES + "regions_onshore_elec_s{simpl}.geojson",
        regions_onshore=RESOURCES + "{interconnect}/regions_onshore.geojson",
        cutout="cutouts/"
        + CDIR
        + "{interconnect}_"
        + config["atlite"]["default_cutout"]
        + ".nc",
    output:
        # clustered_pop_layout=RESOURCES + "pop_layout_elec_s{simpl}.csv",
        clustered_pop_layout=RESOURCES + "{interconnect}/pop_layout_elec_s.csv",
    resources:
        mem_mb=10000,
    log:
        # LOGS + "build_simplified_population_layouts_{simpl}",
        LOGS + "{interconnect}/build_simplified_population_layouts",
    benchmark:
        # BENCHMARKS + "build_simplified_population_layouts/s{simpl}"
        BENCHMARKS + "{interconnect}/build_simplified_population_layouts/s"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_clustered_population_layouts.py"


rule build_clustered_population_layouts:
    input:
        pop_layout_total=RESOURCES + "{interconnect}/pop_layout_total.nc",
        pop_layout_urban=RESOURCES + "{interconnect}/pop_layout_urban.nc",
        pop_layout_rural=RESOURCES + "{interconnect}/pop_layout_rural.nc",
        # regions_onshore=RESOURCES + "regions_onshore_elec_s{simpl}_{clusters}.geojson",
        regions_onshore=RESOURCES
        + "{interconnect}/regions_onshore_s_{clusters}.geojson",
        cutout="cutouts/"
        + CDIR
        + "{interconnect}_"
        + config["atlite"]["default_cutout"]
        + ".nc",
    output:
        # clustered_pop_layout=RESOURCES + "pop_layout_elec_s{simpl}_{clusters}.csv",
        clustered_pop_layout=RESOURCES
        + "{interconnect}/pop_layout_elec_s_{clusters}.csv",
    log:
        # LOGS + "build_clustered_population_layouts_{simpl}_{clusters}.log",
        LOGS + "{interconnect}/build_clustered_population_layouts_{clusters}.log",
    resources:
        mem_mb=10000,
    benchmark:
        # BENCHMARKS + "build_clustered_population_layouts/s{simpl}_{clusters}"
        BENCHMARKS + "{interconnect}/build_clustered_population_layouts/s_{clusters}"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_clustered_population_layouts.py"


rule build_cop_profiles:
    params:
        heat_pump_sink_T=config["sector"]["heating"]["heat_pump_sink_T"],
    input:
        temp_soil_total=RESOURCES
        + "{interconnect}/temp_soil_total_elec_s_{clusters}.nc",
        temp_soil_rural=RESOURCES
        + "{interconnect}/temp_soil_rural_elec_s_{clusters}.nc",
        temp_soil_urban=RESOURCES
        + "{interconnect}/temp_soil_urban_elec_s_{clusters}.nc",
        temp_air_total=RESOURCES + "{interconnect}/temp_air_total_elec_s_{clusters}.nc",
        temp_air_rural=RESOURCES + "{interconnect}/temp_air_rural_elec_s_{clusters}.nc",
        temp_air_urban=RESOURCES + "{interconnect}/temp_air_urban_elec_s_{clusters}.nc",
        # temp_soil_total=RESOURCES + "temp_soil_total_elec_s{simpl}_{clusters}.nc",
        # temp_soil_rural=RESOURCES + "temp_soil_rural_elec_s{simpl}_{clusters}.nc",
        # temp_soil_urban=RESOURCES + "temp_soil_urban_elec_s{simpl}_{clusters}.nc",
        # temp_air_total=RESOURCES + "temp_air_total_elec_s{simpl}_{clusters}.nc",
        # temp_air_rural=RESOURCES + "temp_air_rural_elec_s{simpl}_{clusters}.nc",
        # temp_air_urban=RESOURCES + "temp_air_urban_elec_s{simpl}_{clusters}.nc",
    output:
        cop_soil_total=RESOURCES + "{interconnect}/cop_soil_total_elec_s_{clusters}.nc",
        cop_soil_rural=RESOURCES + "{interconnect}/cop_soil_rural_elec_s_{clusters}.nc",
        cop_soil_urban=RESOURCES + "{interconnect}/cop_soil_urban_elec_s_{clusters}.nc",
        cop_air_total=RESOURCES + "{interconnect}/cop_air_total_elec_s_{clusters}.nc",
        cop_air_rural=RESOURCES + "{interconnect}/cop_air_rural_elec_s_{clusters}.nc",
        cop_air_urban=RESOURCES + "{interconnect}/cop_air_urban_elec_s_{clusters}.nc",
        # cop_soil_total=RESOURCES + "cop_soil_total_elec_s{simpl}_{clusters}.nc",
        # cop_soil_rural=RESOURCES + "cop_soil_rural_elec_s{simpl}_{clusters}.nc",
        # cop_soil_urban=RESOURCES + "cop_soil_urban_elec_s{simpl}_{clusters}.nc",
        # cop_air_total=RESOURCES + "cop_air_total_elec_s{simpl}_{clusters}.nc",
        # cop_air_rural=RESOURCES + "cop_air_rural_elec_s{simpl}_{clusters}.nc",
        # cop_air_urban=RESOURCES + "cop_air_urban_elec_s{simpl}_{clusters}.nc",
    resources:
        mem_mb=20000,
    log:
        LOGS + "{interconnect}/build_cop_profiles_s{clusters}.log",
        # LOGS + "build_cop_profiles_s{simpl}_{clusters}.log",
    benchmark:
        # BENCHMARKS + "build_cop_profiles/s{simpl}_{clusters}"
        BENCHMARKS + "{interconnect}/build_cop_profiles/s_{clusters}"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_cop_profiles.py"
