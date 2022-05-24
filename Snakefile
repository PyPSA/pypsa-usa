# Copyright 2021-2022 Martha Frysztacki (KIT)

from os.path import normpath, exists
from shutil import copyfile

from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider

HTTP = HTTPRemoteProvider()


configfile: "config.yaml"


# define subworkflow here
# currently only function imports, no snakemake rules can be re-used due to leap year
subworkflow_dir = config["subworkflow"]


subworkflow pypsaeur:
    workdir:
        subworkflow_dir
    snakefile:
        subworkflow_dir + "Snakefile"


configfile: subworkflow_dir + "config.default.yaml"  #append to existing config
configfile: "config.yaml"  #read config twice in case some keys were be overwritten


wildcard_constraints:
    interconnect="usa|texas|western|eastern",
    simpl="[a-zA-Z0-9]*|all",
    clusters="[0-9]+m?|all",
    ll="(v|c)([0-9\.]+|opt|all)|all",
    opts="[-+a-zA-Z0-9\.]*",


rule test_all:
    input:
        expand(
            "results/networks/texas_s_10_ec_l{ll}_{opts}.nc",
            ll=["vopt"],
            opts=[
                "Co2L0.5-2920SEG",
                "Co2L1.0-2920SEG",
            ],
        ),


rule solve_all:
    input:
        expand(
            "results/networks/usa_s_200_ec_l{ll}_{opts}.nc",
            ll=["vopt"],
            opts=[
                "Co2L0.0-2920SEG",
                "Co2L0.1-2920SEG",
                "Co2L0.2-2920SEG",
                "Co2L0.3-2920SEG",
                "Co2L0.4-2920SEG",
                "Co2L0.5-2920SEG",
                "Co2L0.6-2920SEG",
                "Co2L0.7-2920SEG",
                "Co2L0.8-2920SEG",
                "Co2L0.9-2920SEG",
                "Co2L1.0-2920SEG",
            ],
        ),


if not config["zenodo_repository"]["use"]:

    rule create_network:
        input:
            tech_costs=subworkflow_dir + "data/costs.csv",
        output:
            bus2sub="data/base_grid/{interconnect}/bus2sub.csv",
            sub="data/base_grid/{interconnect}/sub.csv",
            network="networks/{interconnect}.nc",
        log:
            "logs/create_network/{interconnect}.log",
        threads: 4
        resources:
            mem=500,
        script:
            "scripts/create_network_from_powersimdata.py"


else:

    DATAFILES = [
        "bus.csv",
        "sub.csv",
        "bus2sub.csv",
        "branch.csv",
        "dcline.csv",
        "demand.csv",
        "plant.csv",
        "solar.csv",
        "wind.csv",
        "hydro.csv",
    ]

    rule retrieve_data_from_zenodo:
        output:
            expand("data/base_grid/usa/{file}", file=DATAFILES),
        log:
            "logs/retrieve_data_from_zenodo.log",
        script:
            "scripts/retrieve_data_from_zenodo.py"

    rule create_network:
        input:
            buses="data/base_grid/usa/bus.csv",
            lines="data/base_grid/usa/branch.csv",
            links="data/base_grid/usa/dcline.csv",
            plants="data/base_grid/usa/plant.csv",
            wind="data/base_grid/usa/wind.csv",
            solar="data/base_grid/usa/solar.csv",
            hydro="data/base_grid/usa/hydro.csv",
            demand="data/base_grid/usa/demand.csv",
            bus2sub="data/base_grid/usa/bus2sub.csv",
            sub="data/base_grid/usa/sub.csv",
            tech_costs=subworkflow_dir + "data/costs.csv",
        output:
            network="networks/usa.nc",
        log:
            "logs/create_network.log",
        threads: 4
        resources:
            mem=500,
        script:
            "scripts/create_network_from_zenodo.py"


rule simplify_network:
    input:
        bus2sub="data/base_grid/{interconnect}/bus2sub.csv",
        sub="data/base_grid/{interconnect}/sub.csv",
        network="networks/{interconnect}.nc",
    output:
        network="networks/{interconnect}_s.nc",
    log:
        "logs/simplify_network/{interconnect}_s.log",
    threads: 4
    resources:
        mem=500,
    script:
        "scripts/simplify_network.py"


rule cluster_network:
    input:
        "networks/{interconnect}_s.nc",
    output:
        network="networks/{interconnect}_s_{clusters}.nc",
        busmap="resources/busmap_{interconnect}_s_{clusters}.csv",
    log:
        "logs/cluster_network/{interconnect}_s_{clusters}.log",
    threads: 1
    resources:
        mem=500,
    script:
        "scripts/cluster_network.py"


rule add_extra_components:
    input:
        network="networks/{interconnect}_s_{nclusters}.nc",
        tech_costs=subworkflow_dir + "data/costs.csv",
    output:
        "networks/{interconnect}_s_{nclusters}_ec.nc",
    log:
        "logs/add_extra_components/{interconnect}_s_{nclusters}_ec.log",
    threads: 4
    resources:
        mem=500,
    script:
        pypsaeur("scripts/add_extra_components.py")


rule prepare_network:
    input:
        network="networks/{interconnect}_s_{nclusters}_ec.nc",
        tech_costs=subworkflow_dir + "data/costs.csv",
    output:
        "networks/{interconnect}_s_{nclusters}_ec_l{ll}_{opts}.nc",
    log:
        solver="logs/prepare_network/{interconnect}_s_{nclusters}_ec_l{ll}_{opts}.log",
    threads: 4
    resources:
        mem=5000,
    log:
        "logs/prepare_network",
    script:
        pypsaeur("scripts/prepare_network.py")


def memory(w):
    factor = 3.0
    for o in w.opts.split("-"):
        m = re.match(r"^(\d+)h$", o, re.IGNORECASE)
        if m is not None:
            factor /= int(m.group(1))
            break
    for o in w.opts.split("-"):
        m = re.match(r"^(\d+)seg$", o, re.IGNORECASE)
        if m is not None:
            factor *= int(m.group(1)) / 8760
            break
    if w.clusters.endswith("m"):
        return int(factor * (18000 + 180 * int(w.clusters[:-1])))
    elif w.clusters == "all":
        return int(factor * (18000 + 180 * 4000))
    else:
        return int(factor * (10000 + 195 * int(w.clusters)))


rule solve_network:
    input:
        "networks/{interconnect}_s_{clusters}_ec_l{ll}_{opts}.nc",
    output:
        "results/networks/{interconnect}_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc",
    log:
        solver=normpath(
            "logs/solve_network/{interconnect}_s{simpl}_{clusters}_ec_l{ll}_{opts}_solver.log"
        ),
        python="logs/solve_network/{interconnect}_s{simpl}_{clusters}_ec_l{ll}_{opts}_python.log",
        memory="logs/solve_network/{interconnect}_s{simpl}_{clusters}_ec_l{ll}_{opts}_memory.log",
    benchmark:
        "benchmarks/solve_network/{interconnect}_s{simpl}_{clusters}_ec_l{ll}_{opts}"
    threads: 4
    resources:
        mem_mb=memory,
    shadow:
        "minimal"
    script:
        pypsaeur("scripts/solve_network.py")
