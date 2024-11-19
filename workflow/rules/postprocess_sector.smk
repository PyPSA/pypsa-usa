"""Rules for post procesing solved sector coupled networks"""

# state and system figures
FIGURES_SECTOR_EMISSIONS = ["emissions_by_sector", "emissions_by_state"]
FIGURES_SECTOR_PRODUCTION = [
    "load_factor_boxplot",
    "../hp_cop",  # same for all sectors
    "production_time_series",
    "production_total",
]
FIGURES_SECTOR_CAPACITY = [
    "end_use_capacity_per_carrier",
    # "end_use_capacity_per_node_absolute",
    # "end_use_capacity_per_node_percentage",
    # "end_use_capacity_state_brownfield",
    # "power_capacity_per_carrier",
]
FIGURES_SECTOR_LOADS = [
    # "load_timeseries_residential",
    # "load_timeseries_commercial",
    # "load_timeseries_industrial",
    # "load_timeseries_transport",
    "load_barplot"
]
FIGURES_SECTOR_VALIDATE = [
    "emissions_by_sector",
    # "emissions_by_state_validation",
    # "generation_by_state_validation",
    # "transportation_by_mode_validation",
]
FIGURES_SECTOR_NATURAL_GAS = [
    "demand",
    "processing",
    "linepack",
    "storage",
    "domestic_trade",
    "international_trade",
    "fuel_price",
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
            + "{interconnect}/figures/s{simpl}_c{clusters}/l{ll}_{opts}_{sector}/system/natural_gas/%s.png"
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
        result="emissions",
        plotting=config["plotting"],
        root_dir=RESULTS
        + "{interconnect}/figures/s{simpl}_c{clusters}/l{ll}_{opts}_{sector}/",
    output:
        expand(
            RESULTS
            + "{{interconnect}}/figures/s{{simpl}}_c{{clusters}}/l{{ll}}_{{opts}}_{{sector}}/system/emissions/{fig}.png",
            fig=FIGURES_SECTOR_EMISSIONS,
        ),
    log:
        "logs/plot_figures/{interconnect}_s{simpl}_c{clusters}_l{ll}_{opts}_{sector}_emissions.log",
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
        result="production",
        plotting=config["plotting"],
        root_dir=RESULTS
        + "{interconnect}/figures/s{simpl}_c{clusters}/l{ll}_{opts}_{sector}/",
    output:
        expand(
            RESULTS
            + "{{interconnect}}/figures/s{{simpl}}_c{{clusters}}/l{{ll}}_{{opts}}_{{sector}}/system/production/{sec}/{fig}.png",
            sec=["res"],
            fig=FIGURES_SECTOR_PRODUCTION,
        ),
    log:
        "logs/plot_figures/{interconnect}_s{simpl}_c{clusters}_l{ll}_{opts}_{sector}_production.log",
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
        result="capacity",
        plotting=config["plotting"],
        root_dir=RESULTS
        + "{interconnect}/figures/s{simpl}_c{clusters}/l{ll}_{opts}_{sector}/",
    output:
        expand(
            RESULTS
            + "{{interconnect}}/figures/s{{simpl}}_c{{clusters}}/l{{ll}}_{{opts}}_{{sector}}/system/capacity/{sec}/{fig}.png",
            sec=["res"],
            fig=FIGURES_SECTOR_CAPACITY,
        ),
    log:
        "logs/plot_figures/{interconnect}_s{simpl}_c{clusters}_l{ll}_{opts}_{sector}_capacity.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/plot_statistics_sector.py"


rule plot_sector_validation:
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc",
    params:
        plotting=config["plotting"],
        eia_api=config["api"]["eia"],
        root_dir=RESULTS
        + "{interconnect}/figures/s{simpl}_c{clusters}/l{ll}_{opts}_{sector}/",
    output:
        expand(
            RESULTS
            + "{{interconnect}}/figures/s{{simpl}}_c{{clusters}}/l{{ll}}_{{opts}}_{{sector}}/system/validation/{fig}.png",
            sec=["res"],
            fig=FIGURES_SECTOR_VALIDATE,
        ),
    log:
        "logs/plot_figures/{interconnect}_s{simpl}_c{clusters}_l{ll}_{opts}_{sector}_validate.log",
    threads: 1
    resources:
        mem_mb=5000,
    script:
        "../scripts/plot_statistics_sector.py"
