'''
Preprocesses Historical and Forecasted Load, Solar, and Wind Data

Written by Kamran Tehranchi, Stanford University.
'''
import pandas as pd, glob, os, logging, pypsa
from _helpers import progress_retrieve, configure_logging
from add_electricity import load_costs, _add_missing_carriers_from_costs

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
    # import pdb; pdb.set_trace()
    for i in range(len(paths)):
        df = pd.read_csv(paths[i], header=0, index_col=0,low_memory=False)
        df = df.iloc[1:8785, :]
        df_combined = pd.concat([df_combined, df], axis=1)
    df_combined.to_csv(os.path.join("resources/WECC_ADS/processed", f'{profiletype}_{year}.csv'))

    return None

def preprocess_ads_load(df_ads,data_year):
    df_ads['CISO-PGAE'] = df_ads.pop('CIPV') + df_ads.pop('CIPB') + df_ads.pop('SPPC')#hotfix see github issue #15
    df_ads['BPAT'] = df_ads.pop('BPAT') + df_ads.pop('TPWR') + df_ads.pop('SCL')
    df_ads['IPCO'] = df_ads.pop('IPFE') + df_ads.pop('IPMV') + df_ads.pop('IPTV')
    df_ads['PACW'] = df_ads.pop('PAID') + df_ads.pop('PAUT') + df_ads.pop('PAWY')
    df_ads['Arizona'] = df_ads.pop('SRP') + df_ads.pop('AZPS') 
    df_ads.drop(columns=['Unnamed: 44', 'TH_Malin', 'TH_Mead', 'TH_PV'],inplace=True)
    ba_list_map = {'CISC': 'CISO-SCE', 'CISD': 'CISO-SDGE','LDWP': 'LADWP','NWMT': 'MT_west','TIDC': 'TID','VEA': 'CISO-VEA','WAUW': 'WAUW_SPP'}
    df_ads.rename(columns=ba_list_map,inplace=True)
    df_ads['datetime'] = pd.Timestamp(f'{data_year}-01-01')+pd.to_timedelta(df_ads.index, unit='H')
    df_ads.set_index('datetime',inplace=True)
    if len(df_ads.index) > 8761: #remove leap year day
        df_ads= df_ads[~(df_ads.index.date == pd.to_datetime(f'{data_year}-04-29'))]

    # not_in_list = df_ads.loc[:,~df_ads.columns.isin(ba_list)]
    return df_ads


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

    n = pypsa.Network(snakemake.input['network'])

    interconnect = snakemake.wildcards.interconnect
    # interconnect in raw data given with an uppercase first letter
    if interconnect != "usa":
        interconnect = interconnect[0].upper() + interconnect[1:]


    if snakemake.config['load_data']['use_ads']:

        logger.info("Preproccessing ADS data")
        os.makedirs("resources/WECC_ADS/processed/", exist_ok=True)

        file_patterns = {   # Processed file name : Unprocessed file name
                2032: {
                    'load': 'Profile_Load',
                    'solar': 'Profile_Solar',
                    'wind': 'Profile_Wind',
                    'hydro': 'Profile_Hydro',
                    'btm_solar': 'Profile_BTM Solar',
                    'pumped_storage': 'Profile_Pumped Storage',
                    'pump_load': 'Profile_Pumps',
                },
                2030: {
                    'load': 'Data_Load',
                    'solar': 'Data_Solar PV',
                    'wind': 'Data_WT',
                    'hydro': 'Data_Hydro',
                    'btm_solar': 'Data_SolarPV_Rooftop',
                    'pumped_storage': 'Data_PumpStorage',
                    'pump_load': 'Data_Pump',
                }
            }
        process_ads_data(file_patterns)
        n = add_ads_demand_from_file(n, snakemake.input["demand_breakthrough_2016"])
        n.export_to_netcdf(snakemake.output.network)

    else:   # else standard breakthrough configuration
        # load data
        logger.info("Adding Breakthrough Energy Network Demand data")

        n.set_snapshots(
            pd.date_range(freq="h", start="2016-01-01", end="2017-01-01", closed="left")
        )

        # attach load costs
        Nyears = n.snapshot_weightings.generators.sum() / 8784.0
        costs = load_costs(
            snakemake.input.tech_costs,
            snakemake.config["costs"],
            snakemake.config["electricity"],
            Nyears,
        )

        # should renaming technologies move to config.yaml?
        costs = costs.rename(index={"onwind": "wind", "OCGT": "ng"})
        n = add_breakthrough_demand_from_file(n, snakemake.input["demand_breakthrough_2016"])
        n.export_to_netcdf(snakemake.output.network)