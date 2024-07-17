"""Rules for post procesing solved sector coupled networks"""

FIGURES_SECTOR_EMISSIONS = ["emissions_by_sector", "emissions_by_state"]
FIGURES_SECTOR_PRODUCTION = [
    "load_factor_boxplot",
    "hp_cop",
    "production_time_series",
    "production_total",
]
FIGURES_SECTOR_CAPACITY = [
    "end_use_capacity_per_node_absolute",
    "end_use_capacity_per_node_percentage",
]
FIGURES_SECTOR_LOADS = []
FIGURES_SECTOR_NATURAL_GAS = [
    "natural_gas_demand.html",
    "natural_gas_processing.html",
    "natural_gas_linepack.html",
    "natural_gas_storage.html",
    "natural_gas_domestic_trade.html",
    "natural_gas_international_trade.html",
]


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
            for fig in FIGURES_SECTOR_NATURAL_GAS
        },
    log:
        "logs/plot_figures/gas/{interconnect}_{clusters}_l{ll}_{opts}_{sector}.log",
    script:
        "../scripts/plot_natural_gas.py"


rule plot_sector_emissions:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}.nc",
    params:
        plotting=config["plotting"],
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/cluster_{clusters}/l{ll}_{opts}_{sector}/emissions/%s.png"
            % fig
            for fig in FIGURES_SECTOR_EMISSIONS
        },
    log:
        "logs/plot_figures/{interconnect}_{clusters}_l{ll}_{opts}_{sector}_emissions.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/plot_statistics_sector.py"


rule plot_sector_production:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}.nc",
    params:
        plotting=config["plotting"],
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/cluster_{clusters}/l{ll}_{opts}_{sector}/production/%s.png"
            % fig
            for fig in FIGURES_SECTOR_PRODUCTION
        },
    log:
        "logs/plot_figures/{interconnect}_{clusters}_l{ll}_{opts}_{sector}_production.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/plot_statistics_sector.py"


rule plot_sector_capacity:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}.nc",
    params:
        plotting=config["plotting"],
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/cluster_{clusters}/l{ll}_{opts}_{sector}/capacity/%s.png"
            % fig
            for fig in FIGURES_SECTOR_CAPACITY
        },
    log:
        "logs/plot_figures/{interconnect}_{clusters}_l{ll}_{opts}_{sector}_capacity.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/plot_statistics_sector.py"


rule plot_sector_loads:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}.nc",
    params:
        plotting=config["plotting"],
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/cluster_{clusters}/l{ll}_{opts}_{sector}/loads/%s.png"
            % fig
            for fig in FIGURES_SECTOR_LOADS
        },
    log:
        "logs/plot_figures/{interconnect}_{clusters}_l{ll}_{opts}_{sector}_loads.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/plot_statistics_sector.py"


rule plot_energy_sankey:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}.nc",
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/cluster_{clusters}/l{ll}_{opts}_{sector}/sankey/%s"
            % fig
            for fig in ["usa.pdf"]
        },
    log:
        "logs/plot_figures/sankey/{interconnect}_{clusters}_l{ll}_{opts}_{sector}.log",
    script:
        "../scripts/plot_energy_sankey.py"
