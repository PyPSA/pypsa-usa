# Copyright 2021-2022 Martha Frysztacki (KIT)

from os.path import normpath, exists
from shutil import copyfile

from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider
HTTP = HTTPRemoteProvider()

configfile: "config.yaml"

#define subworkflow here
#currently only function imports, no snakemake rules can be re-used due to leap year
subworkflow_dir = config["subworkflow"]
subworkflow pypsaeur:
    workdir: subworkflow_dir
    snakefile: subworkflow_dir + "Snakefile"
    configfile: subworkflow_dir + "/config.yaml"
configfile: subworkflow_dir + "config.default.yaml" #append to existing config
configfile: "config.yaml" #read config twice in case some keys were be overwritten

wildcard_constraints:
    simpl="[a-zA-Z0-9]*|all",
    clusters="[0-9]+m?|all",
    ll="(v|c)([0-9\.]+|opt|all)|all",
    opts="[-+a-zA-Z0-9\.]*"


datafiles = ['bus.csv', 'sub.csv', 'bus2sub.csv', 'branch.csv', 'dcline.csv', 'demand.csv',
             'plant.csv', 'solar.csv', 'wind.csv', 'hydro.csv']


if config['enable'].get('retrieve_data', True):
    rule retrieve_databundle:
        output: expand('data/base_grid/{file}', file=datafiles)
        log: "logs/retrieve_databundle.log"
        script: 'scripts/retrieve_databundle.py'


rule create_network:
    input:
        buses   = "data/base_grid/bus.csv",
        lines   = "data/base_grid/branch.csv",
        links   = "data/base_grid/dcline.csv",
        plants  = "data/base_grid/plant.csv",
        bus2sub = "data/base_grid/bus2sub.csv",
        wind    = "data/base_grid/wind.csv",
        solar   = "data/base_grid/solar.csv",
        hydro   = "data/base_grid/hydro.csv",
        demand  = "data/base_grid/demand.csv",
        tech_costs = subworkflow_dir + "data/costs.csv"
    output: "networks/elec.nc"
    log: "logs/create_network.log"
    threads: 4
    resources: mem=500
    script: "scripts/create_network.py"


rule simplify_network:
    input:
        network = "networks/elec.nc",
        bus2sub = "data/base_grid/bus2sub.csv",
        sub     = "data/base_grid/sub.csv",
    output: "networks/elec_s.nc"
    log: "logs/simplify_network/elec_s.log"
    threads: 4
    resources: mem=500
    script: "scripts/simplify_network.py"


rule cluster_network:
    input: 'networks/elec_s.nc'
    output:
        network = "networks/elec_s_{clusters}.nc",
        busmap  = "resources/busmap_elec_s_{clusters}.csv",
    log: "logs/cluster_network/elec_s_{clusters}.log"
    threads: 1
    resources: mem=500
    script: "scripts/cluster_network.py"


rule add_extra_components:
    input:
        network = "networks/elec_s_{nclusters}.nc",
        tech_costs = subworkflow_dir + "data/costs.csv"
    output: "networks/elec_s_{nclusters}_ec.nc"
    log: "logs/add_extra_components/elec_s_{nclusters}_ec.log"
    threads: 4
    resources: mem=500
    script: pypsaeur("scripts/add_extra_components.py")


rule prepare_network:
    input:
        network= "networks/elec_s_{nclusters}_ec.nc",
        tech_costs = subworkflow_dir + "data/costs.csv"
    output: "networks/elec_s_{nclusters}_ec_l{ll}_{opts}.nc"
    log:
        solver = "logs/prepare_network/elec_s_{nclusters}_ec_l{ll}_{opts}.log"
    threads: 4
    resources: mem=5000
    log: "logs/prepare_network"
    script: pypsaeur("scripts/prepare_network.py")


def memory(w):
    factor = 3.
    for o in w.opts.split('-'):
        m = re.match(r'^(\d+)h$', o, re.IGNORECASE)
        if m is not None:
            factor /= int(m.group(1))
            break
    for o in w.opts.split('-'):
        m = re.match(r'^(\d+)seg$', o, re.IGNORECASE)
        if m is not None:
            factor *= int(m.group(1)) / 8760
            break
    if w.clusters.endswith('m'):
        return int(factor * (18000 + 180 * int(w.clusters[:-1])))
    elif w.clusters == "all":
        return int(factor * (18000 + 180 * 4000))
    else:
        return int(factor * (10000 + 195 * int(w.clusters)))


rule solve_network:
    input: "networks/elec_s_{clusters}_ec_l{ll}_{opts}.nc"
    output: "results/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc"
    log:
        solver=normpath("logs/solve_network/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_solver.log"),
        python="logs/solve_network/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_python.log",
        memory="logs/solve_network/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_memory.log"
    benchmark: "benchmarks/solve_network/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}"
    threads: 4
    resources: mem_mb=memory
    shadow: "minimal"
    script: pypsaeur("scripts/solve_network.py")