rule solve_network_validation:
    params:
        solving=config["solving"],
        foresight=config["foresight"],
    input:
        network=NETWORKS
        + "{interconnect}/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc",
        flowgates="repo_data/ReEDS_Constraints/transmission/transmission_capacity_init_AC_ba_NARIS2024.csv",
        safer_reeds="config/policy_constraints/reeds/prm_annual.csv",
        rps_reeds="config/policy_constraints/reeds/rps_fraction.csv",
        ces_reeds="config/policy_constraints/reeds/ces_fraction.csv",
        interface_limits="config/policy_constraints/transmission_interface_limits.csv",
    output:
        network=RESULTS
        + "{interconnect}/networks/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}_operations.nc",
        config=RESULTS
        + "{interconnect}/configs/config.elec_s{simpl}_c{clusters}_l{ll}_{opts}_{sector}.yaml",
    log:
        solver=normpath(
            LOGS
            + "solve_network/{interconnect}/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}_solver.log"
        ),
        python=LOGS
        + "solve_network/{interconnect}/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}_python.log",
    benchmark:
        (
            BENCHMARKS
            + "solve_network/{interconnect}/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}"
        )
    threads: solver_threads
    resources:
        walltime=config_provider("walltime", "solve_network_validation"),
        mem_mb=lambda wildcards, input, attempt: (input.size // 100000) * 90,
    script:
        "../scripts/solve_network.py"


rule plot_validation_figures:
    params:
        eia_api=config["api"]["eia"],
        snapshots=config["snapshots"],
    input:
        network=RESULTS
        + "{interconnect}/networks/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}_operations.nc",
        demand_ge=DATA + "GridEmissions/EIA_DMD_2018_2024.csv",
        ge_all=DATA + "GridEmissions/EIA_GridEmissions_all_2018_2024.csv",
        ge_co2=DATA + "GridEmissions/GridEmissions_co2_2018_2024.csv",
        regions_onshore=GEOSPATIAL
        + "{interconnect}/regions_onshore_s{simpl}_{clusters}.geojson",
        regions_offshore=GEOSPATIAL
        + "{interconnect}/regions_offshore_s{simpl}_{clusters}.geojson",
        historical_generation="repo_data/annual_generation_state.xls",
    output:
        **{
            fig: RESULTS
            + "{interconnect}/figures/s{simpl}_cluster_{clusters}/l{ll}_{opts}_{sector}/%s"
            % fig
            for fig in FIGURES_VALIDATE
        },
        val_statistics=RESULTS
        + "{interconnect}/figures/s{simpl}_cluster_{clusters}/l{ll}_{opts}_{sector}/statistics.csv",
    log:
        "logs/plot_figures/validation_{interconnect}_{simpl}_{clusters}_l{ll}_{opts}_{sector}.log",
    threads: 1
    resources:
        walltime="00:30:00",
        mem_mb=5000,
    script:
        "../scripts/plot_validation_production.py"


# Compares installed capacity against the CPUC Baseline Generator List,
# aggregated by SERVM benchmark region and technology, per planning horizon.
# Gated by run.benchmark_cpuc (see benchmark_figures in the Snakefile).
# Deliberately network-free: a fleet benchmark (default horizon 2026 = today's
# plants) compares powerplants.csv against the CPUC workbook and must not drag
# in a network build. run.benchmark_cpuc_horizons overrides the scenario
# planning_horizons so the benchmark year is decoupled from the study years.
rule benchmark_cpuc_baseline:
    params:
        planning_horizons=config_provider("scenario", "planning_horizons"),
        benchmark_horizons=config_provider("run", "benchmark_cpuc_horizons", default=[]),
    input:
        powerplants="resources/powerplants/powerplants.csv",
        cpuc_baseline=DATA + "cpuc/BaselineGeneratorList_CAISO.xlsx",
        region_map="repo_data/CPUC/servm_benchmark_regions.csv",
        tech_map="repo_data/CPUC/servm_tech_map.csv",
        out_of_state="repo_data/CPUC/servm_out_of_state_units.csv",
    output:
        comparison=RESULTS + "cpuc_benchmark/cpuc_capacity_benchmark.csv",
        heatmap=RESULTS + "cpuc_benchmark/cpuc_capacity_deviation.pdf",
        composition=RESULTS + "cpuc_benchmark/cpuc_capacity_composition.pdf",
    log:
        LOGS + "benchmark_cpuc_baseline.log",
    threads: 1
    resources:
        walltime="00:20:00",
        mem_mb=5000,
    script:
        "../scripts/benchmark_cpuc_baseline.py"
