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
        network=RESOURCES + "{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}.nc",
        config=RESULTS + "config.yaml",
    output:
        network=RESULTS + "{interconnect}/networks/elec_s_{clusters}_ec_l{ll}_{opts}.nc",
    log:
        solver=normpath(
            LOGS + "solve_network/{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}_solver.log"
        ),
        python=LOGS
        + "solve_network/{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}_python.log",
    benchmark:
        BENCHMARKS + "solve_network/{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}"
    threads: 8
    resources:
        mem_mb=memory,
        walltime=config["solving"].get("walltime", "12:00:00"),
    shadow:
        "minimal"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/subworkflows/pypsa-eur/scripts/solve_network.py"
