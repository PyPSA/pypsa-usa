"""Rules for post procesing solved networks"""


rule plot_network_maps:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}.nc",
        regions_onshore=(
                config["custom_files"]["files_path"] + 'regions_onshore_s_{clusters}.geojson'
                if config["custom_files"].get("activate", False)
                else RESOURCES + "{interconnect}/regions_onshore_s_{clusters}.geojson"
        ),
        regions_offshore=(
                config["custom_files"]["files_path"] + 'regions_offshore_s_{clusters}.geojson'
                if config["custom_files"].get("activate", False)
                else RESOURCES + "{interconnect}/regions_offshore_s_{clusters}.geojson"
        ),
    params:
        electricity=config["electricity"],
        plotting=config["plotting"],
        retirement=config["electricity"].get("retirement", "technical"),
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/cluster_{clusters}/l{ll}_{opts}_{sector}/maps/%s"
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


rule plot_natural_gas:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}.nc",
    params:
        plotting=config["plotting"],
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/cluster_{clusters}/l{ll}_{opts}_{sector}/gas/%s"
            % fig
            for fig in FIGURES_NATURAL_GAS
        },
    log:
        "logs/plot_figures/gas/{interconnect}_{clusters}_l{ll}_{opts}_{sector}.log",
    script:
        "../scripts/plot_natural_gas.py"


rule plot_statistics:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}.nc",
        regions_onshore=(
                config["custom_files"]["files_path"] + 'regions_onshore_s_{clusters}.geojson'
                if config["custom_files"].get("activate", False)
                else RESOURCES + "{interconnect}/regions_onshore_s_{clusters}.geojson"
        ),
        regions_offshore=(
                config["custom_files"]["files_path"] + 'regions_offshore_s_{clusters}.geojson'
                if config["custom_files"].get("activate", False)
                else RESOURCES + "{interconnect}/regions_offshore_s_{clusters}.geojson"
        ),
    params:
        electricity=config["electricity"],
        plotting=config["plotting"],
        retirement=config["electricity"].get("retirement", "technical"),
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/cluster_{clusters}/l{ll}_{opts}_{sector}/emissions/%s"
            % fig
            for fig in FIGURES_EMISSIONS
        },
        **{
            fig: RESULTS
            + "{interconnect}/figures/cluster_{clusters}/l{ll}_{opts}_{sector}/production/%s"
            % fig
            for fig in FIGURES_PRODUCTION
        },
        **{
            fig: RESULTS
            + "{interconnect}/figures/cluster_{clusters}/l{ll}_{opts}_{sector}/system/%s"
            % fig
            for fig in FIGURES_SYSTEM
        },
    log:
        "logs/plot_figures/{interconnect}_{clusters}_l{ll}_{opts}_{sector}.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/plot_statistics.py"
