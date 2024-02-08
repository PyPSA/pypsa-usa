# Rules to Optimize/Solve Network 

rule add_extra_components:
    input:
        network=RESOURCES + "{interconnect}/elec_s_{clusters}.nc",
        tech_costs=DATA + f"costs_{config['costs']['year']}.csv",
    params:
        retirement=config["electricity"].get("retirement", "technical")
    output:
        RESOURCES + "{interconnect}/elec_s_{clusters}_ec.nc",
    log:
        "logs/add_extra_components/{interconnect}/elec_s_{clusters}_ec.log",
    threads: 4
    resources:
        mem=500,
    script:
        "../scripts/add_extra_components.py"

rule prepare_network:
    params:
        links=config["links"],
        lines=config["lines"],
        co2base=config["electricity"]["co2base"],
        co2limit=config["electricity"]["co2limit"],
        gaslimit=config["electricity"].get("gaslimit"),
        max_hours=config["electricity"]["max_hours"],
        costs=config["costs"],
    input:
        network=RESOURCES + "{interconnect}/elec_s_{clusters}_ec.nc",
        tech_costs=DATA + f"costs_{config['costs']['year']}.csv",
    output:
        RESOURCES + "{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}.nc",
    log:
        solver="logs/prepare_network/{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}.log",
    threads: 4
    resources:
        mem=5000,
    log:
        "logs/prepare_network",
    script:
        "../scripts/subworkflows/pypsa-eur/scripts/prepare_network.py" 


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
    threads: 4
    resources:
        mem_mb=memory,
        walltime=config["solving"].get("walltime", "12:00:00"),
    shadow:
        "minimal"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/subworkflows/pypsa-eur/scripts/solve_network.py"

rule solve_network_operations:
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
        network=RESULTS + "{interconnect}/networks/elec_s_{clusters}_ec_l{ll}_{opts}_operations.nc",
    log:
        solver=normpath(
            LOGS + "solve_network/{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}_solver.log"
        ),
        python=LOGS
        + "solve_network/{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}_python.log",
    benchmark:
        BENCHMARKS + "solve_network/{interconnect}/elec_s_{clusters}_ec_l{ll}_{opts}"
    threads: 4
    resources:
        mem_mb=memory,
        walltime=config["solving"].get("walltime", "12:00:00"),
    shadow:
        "minimal"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_network.py"
