"""Rules for building sector coupling network"""

rule force_sector:
    input:
        population = DATA + "population/DECENNIALDHC2020.P1-Data.csv",

rule build_population_layouts:
    input:
        county_shapes = DATA + "counties/cb_2020_us_county_500k.shp",
        urban_percent = DATA + "urbanization/DECENNIALDHC2020.H2-Data.csv",
        population = DATA + "population/DECENNIALDHC2020.P1-Data.csv",
        cutout = "cutouts/" + CDIR + "{interconnect}_{cutout}.nc",
    output:
        pop_layout_total = RESOURCES + "pop_layout_total.nc",
        pop_layout_urban = RESOURCES + "pop_layout_urban.nc",
        pop_layout_rural = RESOURCES + "pop_layout_rural.nc",
    log:
        LOGS + "build_population_layouts.log",
    resources:
        mem_mb=20000,
    benchmark:
        BENCHMARKS + "build_population_layouts"
    threads: 8
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_population_layouts.py"

rule build_heat_demands:
    params:
        snapshots=config["snapshots"],
    input:
        pop_layout=RESOURCES + "pop_layout_{scope}.nc",
        # regions_onshore=RESOURCES + "regions_onshore_elec_s{simpl}_{clusters}.geojson",
        regions_onshore=RESOURCES + "{interconnect}/regions_onshore_s_{clusters}.geojson",
        cutout="cutouts/" + CDIR + config["atlite"]["default_cutout"] + ".nc",
    output:
        heat_demand=RESOURCES + "heat_demand_{scope}_elec_s{simpl}_{clusters}.nc",
    resources:
        mem_mb=20000,
    threads: 8
    log:
        LOGS + "build_heat_demands_{scope}_{simpl}_{clusters}.loc",
    benchmark:
        BENCHMARKS + "build_heat_demands/{scope}_s{simpl}_{clusters}"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_heat_demand.py"


# rule build_temperature_profiles:
#     params:
#         snapshots=config["snapshots"],
#     input:
#         pop_layout=RESOURCES + "pop_layout_{scope}.nc",
#         regions_onshore=RESOURCES + "regions_onshore_elec_s{simpl}_{clusters}.geojson",
#         cutout="cutouts/" + CDIR + config["atlite"]["default_cutout"] + ".nc",
#     output:
#         temp_soil=RESOURCES + "temp_soil_{scope}_elec_s{simpl}_{clusters}.nc",
#         temp_air=RESOURCES + "temp_air_{scope}_elec_s{simpl}_{clusters}.nc",
#     resources:
#         mem_mb=20000,
#     threads: 8
#     log:
#         LOGS + "build_temperature_profiles_{scope}_{simpl}_{clusters}.log",
#     benchmark:
#         BENCHMARKS + "build_temperature_profiles/{scope}_s{simpl}_{clusters}"
#     conda:
#         "../envs/environment.yaml"
#     script:
#         "../scripts/build_temperature_profiles.py"

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