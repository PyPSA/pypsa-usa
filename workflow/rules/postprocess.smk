"""Rules for post procesing solved networks"""

rule plot_figures:
    input:
        network="results/{interconnect}/networks/elec_s_{clusters}_ec_l{ll}_{opts}.nc",
        regions_onshore=RESOURCES + "{interconnect}/regions_onshore_s_{clusters}.geojson",
        regions_offshore=RESOURCES + "{interconnect}/regions_offshore_s_{clusters}.geojson",
    params:
        plotting=config["plotting"],
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