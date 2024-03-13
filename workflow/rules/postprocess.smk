"""Rules for post procesing solved networks"""


rule copy_config:
    params:
        RDIR=RDIR,
    output:
        RESULTS + "config.yaml",
    threads: 1
    resources:
        mem_mb=1000,
    benchmark:
        BENCHMARKS + "copy_config"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/subworkflows/pypsa-eur/scripts/copy_config.py"


FIGURES_MAPS = [
    "capacity_map_base",
    "capacity_map_optimized",
    "capacity_map_new",
    "demand_map",
    "emissions_map",
    "renewable_potential_map",
    "lmp_map",
]

rule plot_network_maps:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}.nc",
        regions_onshore=RESOURCES
        + "{interconnect}/regions_onshore_s_{clusters}.geojson",
        regions_offshore=RESOURCES
        + "{interconnect}/regions_offshore_s_{clusters}.geojson",
    params:
        electricity=config["electricity"],
        plotting=config["plotting"],
        retirement=config["electricity"].get("retirement", "technical"),
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/cluster_{clusters}/l{ll}_{opts}_{sector}/%s.pdf"
            % fig
            for fig in FIGURES_MAPS
        },
    log:
        "logs/plot_figures/{interconnect}_{clusters}_l{ll}_{opts}_{sector}.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/plot_network_maps.py"


FIGURES_SINGLE_HTML = [
    "production_area_html",
    "emissions_area_html",
    "emissions_region_html",
    "emissions_accumulated_tech_html",
]

FIGURES_STATS = [
    "costs_bar",
    "production_bar",
    "production_area",
    "emissions_area",
    "emissions_accumulated_tech",
    "capacity_additions_bar",
    "bar_regional_capacity_additions",
    "bar_regional_emissions",
    "global_constraint_shadow_prices",
    "generator_data_panel",
    "curtailment_heatmap",
    "capfac_heatmap",
    "region_lmps",
]

rule plot_statistics:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}.nc",
        regions_onshore=RESOURCES
        + "{interconnect}/regions_onshore_s_{clusters}.geojson",
        regions_offshore=RESOURCES
        + "{interconnect}/regions_offshore_s_{clusters}.geojson",
    params:
        electricity=config["electricity"],
        plotting=config["plotting"],
        retirement=config["electricity"].get("retirement", "technical"),
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/cluster_{clusters}/l{ll}_{opts}_{sector}/%s.pdf"
            % fig
            for fig in FIGURES_STATS
        },
        **{
            fig: RESULTS
            + "{interconnect}/figures/cluster_{clusters}/l{ll}_{opts}_{sector}/html/%s.html"
            % fig
            for fig in FIGURES_SINGLE_HTML
        },
    log:
        "logs/plot_figures/{interconnect}_{clusters}_l{ll}_{opts}_{sector}.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/plot_statistics.py"