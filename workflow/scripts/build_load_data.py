'''
Preprocesses Historical and Forecasted Load, Solar, and Wind Data

Written by Kamran Tehranchi, Stanford University.
'''
import pandas as pd, glob, os, logging, pypsa
from _helpers import progress_retrieve, configure_logging

def process_ads_data(file_patterns):
    for year, file_patterns_year in file_patterns.items():
        ads_filelist = glob.glob(os.path.join(snakemake.input[f'ads_{year}'], '*.csv'))
        for profile_type, pattern in file_patterns_year.items():
            modify_ads_files(profile_type, [s for s in ads_filelist if pattern in s], year)

def modify_ads_files(profiletype, paths, year):
    """
    Preprocesses the load, solar, and wind data from the ADS PCM
    :param profiletype: string, either 'load', 'solar', or 'wind'
    :param paths: list of strings, paths to the files to be processed
    :param output_path: string, path to the directory where the processed files will be saved
    :return: None
    """
    df_combined = pd.DataFrame()
    for i in range(len(paths)):
        df = pd.read_csv(paths[i], header=0, index_col=0)
        # df.columns = df.columns.str.split('_').str[1]
        df.columns = df.columns.str.removeprefix(f'{profiletype}_')
        # df.columns = df.columns.str.removeprefix('PV_')
        # df.columns = df.columns.str.removeprefix('WT_')
        # df.columns = df.columns.str.removesuffix('.dat')
        # df.columns = df.columns.str.removesuffix(f'_{year}')
        # df.columns = df.columns.str.removesuffix(f'_[18].dat: {year}')
        df = df.iloc[1:8785, :]
        df_combined = pd.concat([df_combined, df], axis=1)
    df_combined.to_csv(os.path.join("resources/WECC_ADS/processed", f'{profiletype}_{year}.csv'))

    return None

def add_breakthrough_demand_from_file(n, fn_demand):

    """
    Zone power demand is disaggregated to buses proportional to Pd,
    where Pd is the real power demand (MW).
    """

    demand = pd.read_csv(fn_demand, index_col=0)
    # zone_id is int, therefore demand.columns should be int first
    demand.columns = demand.columns.astype(int)
    demand.index = n.snapshots

    intersection = set(demand.columns).intersection(n.buses.zone_id.unique())
    demand = demand[list(intersection)]

    demand_per_bus_pu = (n.buses.set_index("zone_id").Pd / n.buses.groupby("zone_id").sum().Pd)
    demand_per_bus = demand_per_bus_pu.multiply(demand)
    demand_per_bus.columns = n.buses.index

    n.madd( "Load", demand_per_bus.columns, bus=demand_per_bus.columns, p_set=demand_per_bus)
    return n

def add_ads_demand_from_file(n, fn_demand):

    """
    Zone power demand is disaggregated to buses proportional to Pd,
    where Pd is the real power demand (MW).
    """

    demand = pd.read_csv(fn_demand, index_col=0)
    # zone_id is int, therefore demand.columns should be int first
    demand.columns = demand.columns.astype(int)
    demand.index = n.snapshots

    intersection = set(demand.columns).intersection(n.buses.zone_id.unique())
    demand = demand[list(intersection)]

    demand_per_bus_pu = (n.buses.set_index("zone_id").Pd / n.buses.groupby("zone_id").sum().Pd)
    demand_per_bus = demand_per_bus_pu.multiply(demand)
    demand_per_bus.columns = n.buses.index

    n.madd( "Load", demand_per_bus.columns, bus=demand_per_bus.columns, p_set=demand_per_bus)
    return n


if __name__ == "__main__":
    
    logger = logging.getLogger(__name__)
    configure_logging(snakemake)

    network = pypsa.Network(snakemake.input['network'])

    if snakemake.config['load_data']['use_ads']:

        logger.info("Preproccessing ADS data")
        os.makedirs("resources/WECC_ADS/processed/", exist_ok=True)

        file_patterns = {
                2032: {
                    'Load': 'Profile_Load',
                    'Solar': 'Profile_Solar',
                    'Wind': 'Profile_Wind'
                },
                2030: {
                    'Load': 'Profile_Load',
                    'Solar': 'Profile_Solar',
                    'Wind': 'Profile_Wind'
                }
            }

        process_ads_data(file_patterns)

    else: #else standard breakthrough configuration
        # load data
        logger.info("Adding Breakthrough Energy Network Demand data")
        n = add_breakthrough_demand_from_file(n, snakemake.input["demand_breakthrough_2016"])




    # n.set_snapshots(
    #     pd.date_range(freq="h", start="2016-01-01", end="2017-01-01", closed="left")
    # )

    # # attach load costs
    # Nyears = n.snapshot_weightings.generators.sum() / 8784.0
    # costs = load_costs(
    #     snakemake.input.tech_costs,
    #     snakemake.config["costs"],
    #     snakemake.config["electricity"],
    #     Nyears,
    # )

    # # should renaming technologies move to config.yaml?
    # costs = costs.rename(index={"onwind": "wind", "OCGT": "ng"})


