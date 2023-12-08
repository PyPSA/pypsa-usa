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

rule plot_figures:
    input:
        network="results/{interconnect}/networks/elec_s_{clusters}_ec_l{ll}_{opts}.nc",
        regions_onshore=RESOURCES + "{interconnect}/regions_onshore_s_{clusters}.geojson",
        regions_offshore=RESOURCES + "{interconnect}/regions_offshore_s_{clusters}.geojson",
    params:
        plotting=config["plotting"],
        retirement = config["electricity"].get("retirement", "technical")
    output:
        **{
            fig: "results/{interconnect}/figures/cluster_{clusters}/l{ll}_{opts}_%s.pdf"
            % fig
            for fig in FIGURES_SINGLE
        },
        **{
            fig: "results/{interconnect}/figures/cluster_{clusters}/l{ll}_{opts}_%s.html"
            % fig
            for fig in FIGURES_SINGLE_HTML
        },
    script:
        "../scripts/plot_figures.py"


