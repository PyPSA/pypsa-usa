"""Rules for post procesing solved sector coupled networks"""

# state and system figures
FIGURES_SECTOR_EMISSIONS = ["emissions_by_sector", "emissions_by_state"]
FIGURES_SECTOR_PRODUCTION = [
    "load_factor_boxplot",
    "hp_cop",
    "production_time_series",
    "production_total",
]
FIGURES_SECTOR_CAPACITY = [
    "end_use_capacity_per_carrier",
    "end_use_capacity_per_node_absolute",
    "end_use_capacity_per_node_percentage",
    "end_use_capacity_state_brownfield",
    "power_capacity_per_carrier",
]
FIGURES_SECTOR_LOADS = [
    # "load_timeseries_residential",
    # "load_timeseries_commercial",
    # "load_timeseries_industrial",
    # "load_timeseries_transport",
    "load_barplot"
]
FIGURES_SECTOR_VALIDATE = [
    "emissions_by_sector_validation",
    "emissions_by_state_validation",
    "generation_by_state_validation",
    "transportation_by_mode_validation",
]
FIGURES_SECTOR_NATURAL_GAS = [
    "natural_gas_demand.html",
    "natural_gas_processing.html",
    "natural_gas_linepack.html",
    "natural_gas_storage.html",
    "natural_gas_domestic_trade.html",
    "natural_gas_international_trade.html",
]

# system figures
FIGURES_SYSTEM_PRODUCTION = ["system_consumption"]
FIGURES_SYSTEM_VALIDATION = [
    # "system_consumption_validation",
    "system_emission_validation_state"
]


rule plot_natural_gas:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc",
    params:
        plotting=config["plotting"],
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/s{simpl}_c{clusters}/l{ll}_{opts}_{sector}/gas/%s"
            % fig
            for fig in FIGURES_SECTOR_NATURAL_GAS
        },
    log:
        "logs/plot_figures/gas/{interconnect}_s{simpl}_c{clusters}_l{ll}_{opts}_{sector}.log",
    script:
        "../scripts/plot_natural_gas.py"


rule plot_sector_emissions:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc",
    params:
        plotting=config["plotting"],
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/s{simpl}_c{clusters}/l{ll}_{opts}_{sector}/{state}/emissions/%s.png"
            % fig
            for fig in FIGURES_SECTOR_EMISSIONS
        },
    log:
        "logs/plot_figures/{interconnect}_s{simpl}_c{clusters}_l{ll}_{opts}_{sector}_{state}_emissions.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/plot_statistics_sector.py"


rule plot_sector_production:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc",
    params:
        plotting=config["plotting"],
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/s{simpl}_c{clusters}/l{ll}_{opts}_{sector}/{state}/production/%s.png"
            % fig
            for fig in FIGURES_SECTOR_PRODUCTION
        },
    log:
        "logs/plot_figures/{interconnect}_s{simpl}_c{clusters}_l{ll}_{opts}_{sector}_{state}_production.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/plot_statistics_sector.py"


rule plot_sector_capacity:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc",
    params:
        plotting=config["plotting"],
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/s{simpl}_c{clusters}/l{ll}_{opts}_{sector}/{state}/capacity/%s.png"
            % fig
            for fig in FIGURES_SECTOR_CAPACITY
        },
    log:
        "logs/plot_figures/{interconnect}_s{simpl}_c{clusters}_l{ll}_{opts}_{sector}_{state}_capacity.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/plot_statistics_sector.py"


rule plot_sector_loads:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc",
    params:
        plotting=config["plotting"],
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/s{simpl}_c{clusters}/l{ll}_{opts}_{sector}/{state}/loads/%s.png"
            % fig
            for fig in FIGURES_SECTOR_LOADS
        },
    log:
        "logs/plot_figures/{interconnect}_s{simpl}_c{clusters}_l{ll}_{opts}_{sector}_{state}_loads.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/plot_statistics_sector.py"


rule plot_sector_validate:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc",
    params:
        plotting=config["plotting"],
        eia_api=config["api"]["eia"],
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/s{simpl}_c{clusters}/l{ll}_{opts}_{sector}/{state}/validate/%s.png"
            % fig
            for fig in FIGURES_SECTOR_VALIDATE
        },
    log:
        "logs/plot_figures/{interconnect}_s{simpl}_c{clusters}_l{ll}_{opts}_{sector}_{state}_validate.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/plot_statistics_sector.py"


rule plot_system_production:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc",
    params:
        plotting=config["plotting"],
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/s{simpl}_c{clusters}/l{ll}_{opts}_{sector}/system/production/%s.png"
            % fig
            for fig in FIGURES_SYSTEM_PRODUCTION
        },
    log:
        "logs/plot_figures/{interconnect}_s{simpl}_c{clusters}_l{ll}_{opts}_{sector}_system_additional_production.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/plot_statistics_sector.py"


rule plot_system_validate:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc",
    params:
        plotting=config["plotting"],
        eia_api=config["api"]["eia"],
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/s{simpl}_c{clusters}/l{ll}_{opts}_{sector}/system/validate/%s.png"
            % fig
            for fig in FIGURES_SYSTEM_VALIDATION
        },
    log:
        "logs/plot_figures/{interconnect}_s{simpl}_c{clusters}_l{ll}_{opts}_{sector}_system_additional_validate.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/plot_statistics_sector.py"
