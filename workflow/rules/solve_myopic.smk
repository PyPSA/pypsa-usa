
rule add_existing_baseyear:
    params:
        baseyear=config["scenario"]["planning_horizons"][0],
        sector=config["sector"],
        existing_capacities=config["existing_capacities"],
        costs=config["costs"],
    input:
        network=RESULTS
        + "prenetworks/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
        busmap_s=RESOURCES + "busmap_elec_s{simpl}.csv",
        busmap=RESOURCES + "busmap_elec_s{simpl}_{clusters}.csv",
    output:
        RESULTS
        + "prenetworks-brownfield/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
    wildcard_constraints:
        planning_horizons=config["scenario"]["planning_horizons"][0],  #only applies to baseyear
    threads: 1
    resources:
        mem_mb=2000,
    log:
        LOGS
        + "add_existing_baseyear_elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/add_existing_baseyear.py"


rule add_brownfield:
    params:
        H2_retrofit=config["sector"]["H2_retrofit"],
        H2_retrofit_capacity_per_CH4=config["sector"]["H2_retrofit_capacity_per_CH4"],
        threshold_capacity=config["existing_capacities"]["threshold_capacity"],
    input:
        network=RESULTS
        + "prenetworks/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
        network_p=solved_previous_horizon,  #solved network at previous time step
        costs=DATA + "costs_{planning_horizons}.csv",
    output:
        RESULTS
        + "prenetworks-brownfield/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
    threads: 4
    resources:
        mem_mb=10000,
    log:
        LOGS
        + "add_brownfield_elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/add_brownfield.py"


ruleorder: add_existing_baseyear > add_brownfield


rule solve_sector_network_myopic:
    params:
        solving=config["solving"],
        foresight=config["foresight"],
        planning_horizons=config["scenario"]["planning_horizons"],
        co2_sequestration_potential=config["sector"].get(
            "co2_sequestration_potential", 200
        ),
    input:
        network=RESULTS
        + "prenetworks-brownfield/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
        costs=DATA + "costs_{planning_horizons}.csv",
        config=RESULTS + "config.yaml",
    output:
        RESULTS
        + "postnetworks/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
    shadow:
        "shallow"
    log:
        solver=LOGS
        + "elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}_solver.log",
        python=LOGS
        + "elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}_python.log",
    threads: 4
    resources:
        mem_mb=config["solving"]["mem"],
        walltime=config["solving"].get("walltime", "12:00:00"),
    benchmark:
        (
            BENCHMARKS
            + "solve_sector_network/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}"
        )
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_network.py"
