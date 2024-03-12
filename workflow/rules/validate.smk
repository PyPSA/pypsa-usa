
rule plot_validation_figures:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}_operations.nc",
        historic_first=DATA + "eia/6moFiles/EIA930_BALANCE_2019_Jan_Jun.csv",
        historic_second=DATA + "eia/6moFiles/EIA930_BALANCE_2019_Jul_Dec.csv",
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/cluster_{clusters}/l{ll}_{opts}_{sector}/%s.pdf"
            % fig
            for fig in FIGURES_VALIDATE
        },
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/plot_validation_production.py"
