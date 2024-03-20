
rule plot_validation_figures:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}_operations.nc",
        historic_first=DATA + "eia/6moFiles/EIA930_BALANCE_2019_Jan_Jun.csv",
        historic_second=DATA + "eia/6moFiles/EIA930_BALANCE_2019_Jul_Dec.csv",
        regions_onshore=RESOURCES
        + "{interconnect}/regions_onshore_s_{clusters}.geojson",
        regions_offshore=RESOURCES
        + "{interconnect}/regions_offshore_s_{clusters}.geojson",
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/cluster_{clusters}/l{ll}_{opts}_{sector}/validation/%s"
            % fig
            for fig in FIGURES_VALIDATE
        },
    log:
        "logs/plot_figures/validation_{interconnect}_{clusters}_l{ll}_{opts}_{sector}.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/plot_validation_production.py"
