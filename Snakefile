
from os.path import normpath, exists
from shutil import copyfile

from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider
HTTP = HTTPRemoteProvider()

configfile: "config.yaml"


wildcard_constraints:
    simpl="[a-zA-Z0-9]*|all",
    clusters="[0-9]+m?|all",
    ll="(v|c)([0-9\.]+|opt|all)|all",
    opts="[-+a-zA-Z0-9\.]*"


datafiles = ['bus.csv', 'sub.csv', 'bus2sub.csv', 'branch.csv', 'dcline.csv', 'demand.csv',
             'plant.csv', 'solar.csv', 'wind.csv', 'costs.csv']


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
        tech_costs = "data/costs.csv"
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


rule add_storage:
    input:
        network= "networks/elec_s_{nclusters}.nc",
        tech_costs= "data/costs.csv"
    output: "networks/elec_s_{nclusters}_ec.nc"
    log: "logs/add_storage/elec_s_{nclusters}_ec.log"
    threads: 4
    resources: mem=500
    script: "scripts/storage.py"


rule add_co2:
    input:
        network= "networks/elec_s_{nclusters}_ec.nc",
    output: "networks/elec_s_{nclusters}_ec_Co2L-{nH}.nc"
    log:
        solver = "logs/add_co2/elec_s_{nclusters}_ec_Co2L-{nH}_solver.log"
    threads: 4
    resources: mem=5000
    log: "logs/add_co2"
    script: "scripts/add_co2.py"
