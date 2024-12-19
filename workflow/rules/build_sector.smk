"""Rules for building sector coupling network"""


def sector_input_files(wildcards):
    input_files = {
        "network": RESOURCES
        + "{interconnect}/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}.nc",
        "tech_costs": RESOURCES
        + f"costs/sector_costs_{config['scenario']['planning_horizons'][0]}.csv",
    }
    sectors = wildcards.sector.split("-")
    if "G" in sectors:
        ng_files = {
            "county": DATA + "counties/cb_2020_us_county_500k.shp",
            "pipeline_capacity": DATA
            + "natural_gas/EIA-StatetoStateCapacity_Feb2024.xlsx",
            "pipeline_shape": DATA + "natural_gas/pipelines.geojson",
            "eia_757": DATA + "natural_gas/EIA-757.csv",
            "cop_soil_total": RESOURCES
            + "{interconnect}/cop_soil_total_elec_s{simpl}_c{clusters}.nc",
            "cop_soil_rural": RESOURCES
            + "{interconnect}/cop_soil_rural_elec_s{simpl}_c{clusters}.nc",
            "cop_soil_urban": RESOURCES
            + "{interconnect}/cop_soil_urban_elec_s{simpl}_c{clusters}.nc",
            "cop_air_total": RESOURCES
            + "{interconnect}/cop_air_total_elec_s{simpl}_c{clusters}.nc",
            "cop_air_rural": RESOURCES
            + "{interconnect}/cop_air_rural_elec_s{simpl}_c{clusters}.nc",
            "cop_air_urban": RESOURCES
            + "{interconnect}/cop_air_urban_elec_s{simpl}_c{clusters}.nc",
            "clustered_pop_layout": RESOURCES
            + "{interconnect}/pop_layout_elec_s{simpl}_c{clusters}.csv",
            "ev_policy": config["sector"]["transport_sector"]["investment"][
                "ev_policy"
            ],
            "residential_stock": "repo_data/sectors/residential_stock",
            "commercial_stock": "repo_data/sectors/commercial_stock",
            "industrial_stock": "repo_data/sectors/industrial_stock/Table5_6.xlsx",
        }
        input_files.update(ng_files)

    return input_files


rule add_sectors:
    params:
        electricity=config["electricity"],
        costs=config["costs"],
        max_hours=config["electricity"]["max_hours"],
        plotting=config["plotting"],
        snapshots=config["snapshots"],
        api=config["api"],
        sector=config["sector"],
    input:
        unpack(sector_input_files),
    output:
        network=RESOURCES
        + "{interconnect}/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc",
    log:
        "logs/add_sectors/{interconnect}/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.log",
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


rule build_temperature_profiles:
    wildcard_constraints:
        scope="urban|rural|total",
    params:
        snapshots=config["snapshots"],
    input:
        pop_layout=RESOURCES + "{interconnect}/pop_layout_{scope}.nc",
        regions_onshore=RESOURCES
        + "{interconnect}/Geospatial/regions_onshore_s{simpl}_{clusters}.geojson",
        cutout="cutouts/"
        + CDIR
        + "{interconnect}_"
        + config["atlite"]["default_cutout"]
        + ".nc",
    output:
        temp_soil=RESOURCES
        + "{interconnect}/temp_soil_{scope}_elec_s{simpl}_c{clusters}.nc",
        temp_air=RESOURCES
        + "{interconnect}/temp_air_{scope}_elec_s{simpl}_c{clusters}.nc",
    resources:
        mem_mb=20000,
    threads: 8
    log:
        LOGS
        + "{interconnect}/build_temperature_profiles_{scope}_{simpl}_{clusters}.log",
    benchmark:
        (
            BENCHMARKS
            + "{interconnect}/build_temperature_profiles/{scope}_s{simpl}_c{clusters}"
        )
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_temperature_profiles.py"


rule build_simplified_population_layouts:
    input:
        pop_layout_total=RESOURCES + "{interconnect}/pop_layout_total.nc",
        pop_layout_urban=RESOURCES + "{interconnect}/pop_layout_urban.nc",
        pop_layout_rural=RESOURCES + "{interconnect}/pop_layout_rural.nc",
        regions_onshore=RESOURCES
        + "{interconnect}/Geospatial/regions_onshore_s{simpl}.geojson",
        cutout="cutouts/"
        + CDIR
        + "{interconnect}_"
        + config["atlite"]["default_cutout"]
        + ".nc",
    output:
        clustered_pop_layout=RESOURCES + "{interconnect}/pop_layout_elec_s.csv",
    resources:
        mem_mb=50000,
    log:
        LOGS + "{interconnect}/build_simplified_population_layouts",
    benchmark:
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
        regions_onshore=RESOURCES
        + "{interconnect}/Geospatial/regions_onshore_s{simpl}_{clusters}.geojson",
        cutout="cutouts/"
        + CDIR
        + "{interconnect}_"
        + config["atlite"]["default_cutout"]
        + ".nc",
    output:
        clustered_pop_layout=RESOURCES
        + "{interconnect}/pop_layout_elec_s{simpl}_c{clusters}.csv",
    log:
        LOGS
        + "{interconnect}/build_clustered_population_layouts_{simpl}_{clusters}.log",
    resources:
        mem_mb=50000,
    benchmark:
        (
            BENCHMARKS
            + "{interconnect}/build_clustered_population_layouts/s{simpl}_c{clusters}"
        )
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_clustered_population_layouts.py"


rule build_cop_profiles:
    params:
        heat_pump_sink_T=config["sector"]["heating"]["heat_pump_sink_T"],
    input:
        temp_soil_total=RESOURCES
        + "{interconnect}/temp_soil_total_elec_s{simpl}_c{clusters}.nc",
        temp_soil_rural=RESOURCES
        + "{interconnect}/temp_soil_rural_elec_s{simpl}_c{clusters}.nc",
        temp_soil_urban=RESOURCES
        + "{interconnect}/temp_soil_urban_elec_s{simpl}_c{clusters}.nc",
        temp_air_total=RESOURCES
        + "{interconnect}/temp_air_total_elec_s{simpl}_c{clusters}.nc",
        temp_air_rural=RESOURCES
        + "{interconnect}/temp_air_rural_elec_s{simpl}_c{clusters}.nc",
        temp_air_urban=RESOURCES
        + "{interconnect}/temp_air_urban_elec_s{simpl}_c{clusters}.nc",
    output:
        cop_soil_total=RESOURCES
        + "{interconnect}/cop_soil_total_elec_s{simpl}_c{clusters}.nc",
        cop_soil_rural=RESOURCES
        + "{interconnect}/cop_soil_rural_elec_s{simpl}_c{clusters}.nc",
        cop_soil_urban=RESOURCES
        + "{interconnect}/cop_soil_urban_elec_s{simpl}_c{clusters}.nc",
        cop_air_total=RESOURCES
        + "{interconnect}/cop_air_total_elec_s{simpl}_c{clusters}.nc",
        cop_air_rural=RESOURCES
        + "{interconnect}/cop_air_rural_elec_s{simpl}_c{clusters}.nc",
        cop_air_urban=RESOURCES
        + "{interconnect}/cop_air_urban_elec_s{simpl}_c{clusters}.nc",
    resources:
        mem_mb=20000,
    log:
        LOGS + "{interconnect}/build_cop_profiles_s{simpl}_c{clusters}.log",
    benchmark:
        BENCHMARKS + "{interconnect}/build_cop_profiles/s{simpl}_c{clusters}"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_cop_profiles.py"
