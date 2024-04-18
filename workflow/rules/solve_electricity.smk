# Rules to Optimize/Solve Network


rule solve_network:
    params:
        solving=config["solving"],
        foresight=config["foresight"],
        planning_horizons=config["scenario"]["planning_horizons"],
        co2_sequestration_potential=config["sector"].get(
            "co2_sequestration_potential", 200
        ),
    input:
        network=RESOURCES
        + "{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}.nc",
        config=RESULTS + "config.yaml",
        flowgates="repo_data/ReEDS_Constraints/transmission/transmission_capacity_init_AC_ba_NARIS2024.csv",
        safer_reeds="repo_data/ReEDS_Constraints/reserves/prm_annual.csv",
    output:
        network=RESULTS
        + "{interconnect}/networks/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}.nc",
    log:
        solver=normpath(
            LOGS
            + "solve_network/{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}_solver.log"
        ),
        python=LOGS
        + "solve_network/{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}_python.log",
    benchmark:
        (
            BENCHMARKS
            + "solve_network/{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}"
        )
    threads: 8
    resources:
        mem_mb=memory,
        walltime=config["solving"].get("walltime", "12:00:00"),
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_network.py"


rule solve_network_operations:
    params:
        solving=config["solving"],
        foresight=config["foresight"],
        planning_horizons=config["scenario"]["planning_horizons"],
        co2_sequestration_potential=config["sector"].get(
            "co2_sequestration_potential", 200
        ),
    input:
        network=RESOURCES
        + "{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}.nc",
        config=RESULTS + "config.yaml",
        flowgates="repo_data/ReEDS_Constraints/transmission/transmission_capacity_init_AC_ba_NARIS2024.csv",
    output:
        network=RESULTS
        + "{interconnect}/networks/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}_operations.nc",
    log:
        solver=normpath(
            LOGS
            + "solve_network/{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}_solver.log"
        ),
        python=LOGS
        + "solve_network/{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}_python.log",
    benchmark:
        (
            BENCHMARKS
            + "solve_network/{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}_{sector}"
        )
    threads: 8
    resources:
        mem_mb=memory,
        walltime=config["solving"].get("walltime", "12:00:00"),
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_network.py"
