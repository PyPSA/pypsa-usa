rule solve_network_validation:
    params:
        solving=config["solving"],
        foresight=config["foresight"],
        planning_horizons=config["scenario"]["planning_horizons"],
        co2_sequestration_potential=config["sector"].get(
            "co2_sequestration_potential", 200
        ),
    input:
        network=RESOURCES
        + "{interconnect}/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc",
        flowgates="repo_data/ReEDS_Constraints/transmission/transmission_capacity_init_AC_ba_NARIS2024.csv",
        safer_reeds="config/policy_constraints/reeds/prm_annual.csv",
        rps_reeds="config/policy_constraints/reeds/rps_fraction.csv",
        ces_reeds="config/policy_constraints/reeds/ces_fraction.csv",
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
    threads: 8
    resources:
        mem_mb=memory,
        walltime=config["solving"].get("walltime", "12:00:00"),
    conda:
        "../envs/environment.yaml"
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
        regions_onshore=RESOURCES
        + "{interconnect}/regions_onshore_s{simpl}_{clusters}.geojson",
        regions_offshore=RESOURCES
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
        mem_mb=5000,
    script:
        "../scripts/plot_validation_production.py"
