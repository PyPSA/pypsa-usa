"""Rules for post procesing solved networks"""


# Rule always defined; requested only when EGS + seismic_exclusion in electricity_figures()
rule plot_egs_seismic_availability:
    input:
        specs_egs=DATA + "EGS/{interconnect}/specs_EGS.nc",
        bus2sub=RESOURCES + "{interconnect}/bus2sub.csv",
        seismic_exclusion=DATA + "seismic_risk_exclusion/seismic_risk_mask.geojson",
        regions_onshore=RESOURCES + "{interconnect}/Geospatial/regions_onshore.geojson",
    output:
        plot=RESULTS
        + "{interconnect}/figures/s{simpl}_cluster_{clusters}/l{ll}_{opts}_{sector}/maps/land_use_availability_EGS.png",
    log:
        LOGS
        + "plot_egs_seismic_availability_{interconnect}_s{simpl}_c{clusters}_l{ll}_{opts}_{sector}.log",
    benchmark:
        BENCHMARKS
        +"plot_egs_seismic_availability_{interconnect}_s{simpl}_c{clusters}_l{ll}_{opts}_{sector}"
    threads: 1
    resources:
        mem_mb=4000,
        walltime="00:15:00",
    script:
        "../scripts/plot_egs_seismic_availability.py"


rule plot_network_maps:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc",
        regions_onshore=(
            config["custom_files"]["files_path"]
            + "regions_onshore_s{simpl}_{clusters}.geojson"
            if config["custom_files"].get("activate", False)
            else RESOURCES
            + "{interconnect}/Geospatial/regions_onshore_s{simpl}_{clusters}.geojson"
        ),
        regions_offshore=(
            config["custom_files"]["files_path"]
            + "regions_offshore_s{simpl}_{clusters}.geojson"
            if config["custom_files"].get("activate", False)
            else RESOURCES
            + "{interconnect}/Geospatial/regions_offshore_s{simpl}_{clusters}.geojson"
        ),
    params:
        electricity=config["electricity"],
        plotting=config["plotting"],
        retirement=config["electricity"].get("retirement", "technical"),
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/s{simpl}_cluster_{clusters}/l{ll}_{opts}_{sector}/maps/%s"
            % fig
            for fig in FIGURES_MAPS
        },
    log:
        "logs/plot_figures/{interconnect}_{simpl}_{clusters}_l{ll}_{opts}_{sector}.log",
    threads: 1
    resources:
        mem_mb=14000,
        walltime="00:30:00",
    script:
        "../scripts/plot_network_maps.py"


rule plot_statistics:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc",
        regions_onshore=(
            config["custom_files"]["files_path"]
            + "regions_onshore_s_{clusters}.geojson"
            if config["custom_files"].get("activate", False)
            else RESOURCES
            + "{interconnect}/Geospatial/regions_onshore_s{simpl}_{clusters}.geojson"
        ),
        regions_offshore=(
            config["custom_files"]["files_path"]
            + "regions_offshore_s_{clusters}.geojson"
            if config["custom_files"].get("activate", False)
            else RESOURCES
            + "{interconnect}/Geospatial/regions_offshore_s{simpl}_{clusters}.geojson"
        ),
    params:
        electricity=config["electricity"],
        plotting=config["plotting"],
        retirement=config["electricity"].get("retirement", "technical"),
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/s{simpl}_cluster_{clusters}/l{ll}_{opts}_{sector}/emissions/%s"
            % fig
            for fig in FIGURES_EMISSIONS
        },
        **{
            fig: RESULTS
            + "{interconnect}/figures/s{simpl}_cluster_{clusters}/l{ll}_{opts}_{sector}/production/%s"
            % fig
            for fig in FIGURES_PRODUCTION
        },
        **{
            fig: RESULTS
            + "{interconnect}/figures/s{simpl}_cluster_{clusters}/l{ll}_{opts}_{sector}/system/%s"
            % fig
            for fig in FIGURES_SYSTEM
        },
        statistics_summary=RESULTS
        + "{interconnect}/figures/s{simpl}_cluster_{clusters}/l{ll}_{opts}_{sector}/statistics/statistics.csv",
        statistics_dissaggregated=RESULTS
        + "{interconnect}/figures/s{simpl}_cluster_{clusters}/l{ll}_{opts}_{sector}/statistics/statistics_dissaggregated.csv",
        generators=RESULTS
        + "{interconnect}/figures/s{simpl}_cluster_{clusters}/l{ll}_{opts}_{sector}/statistics/generators.csv",
        storage_units=RESULTS
        + "{interconnect}/figures/s{simpl}_cluster_{clusters}/l{ll}_{opts}_{sector}/statistics/storage_units.csv",
        links=RESULTS
        + "{interconnect}/figures/s{simpl}_cluster_{clusters}/l{ll}_{opts}_{sector}/statistics/links.csv",
        buses=RESULTS
        + "{interconnect}/figures/s{simpl}_cluster_{clusters}/l{ll}_{opts}_{sector}/statistics/buses.csv",
        lines=RESULTS
        + "{interconnect}/figures/s{simpl}_cluster_{clusters}/l{ll}_{opts}_{sector}/statistics/lines.csv",
        stores=RESULTS
        + "{interconnect}/figures/s{simpl}_cluster_{clusters}/l{ll}_{opts}_{sector}/statistics/stores.csv",
        global_constraints=RESULTS
        + "{interconnect}/figures/s{simpl}_cluster_{clusters}/l{ll}_{opts}_{sector}/statistics/global_constraints.csv",
    log:
        "logs/plot_figures/{interconnect}_{simpl}_{clusters}_l{ll}_{opts}_{sector}.log",
    threads: 1
    resources:
        mem_mb=20000,
        walltime="00:30:00",
    script:
        "../scripts/plot_statistics.py"
