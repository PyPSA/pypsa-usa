# Rules to Optimize/Solve Network


def pop_layout_input(wildcards):
    if wildcards["sector"] != "E":
        return RESOURCES + "{interconnect}/pop_layout_elec_s{simpl}_c{clusters}.csv"
    else:
        return ""


rule solve_network:
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        planning_horizons=config["scenario"]["planning_horizons"],
        co2_sequestration_potential=config["sector"].get(
            "co2_sequestration_potential", 200
        ),
        transmission_network=config_provider("model_topology", "transmission_network"),
    input:
        network=RESOURCES
        + "{interconnect}/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc",
        flowgates="repo_data/ReEDS_Constraints/transmission/transmission_capacity_init_AC_ba_NARIS2024.csv",
        safer_reeds="config/policy_constraints/reeds/prm_annual.csv",
        rps_reeds="config/policy_constraints/reeds/rps_fraction.csv",
        ces_reeds="config/policy_constraints/reeds/ces_fraction.csv",
        pop_layout=pop_layout_input,
    output:
        network=RESULTS
        + "{interconnect}/networks/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc",
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
        mem_mb=memory,
        walltime=config["solving"].get("walltime", "12:00:00"),
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_network.py"
