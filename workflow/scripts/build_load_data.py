'''
Preprocesses Historical and Forecasted Load, Solar, and Wind Data

Written by Kamran Tehranchi, Stanford University.
'''
import pandas as pd, glob, os, logging, pypsa
from _helpers import progress_retrieve, configure_logging

def add_breakthrough_demand_from_file(n, fn_demand):

    """
    Zone power demand is disaggregated to buses proportional to Pd,
    where Pd is the real power demand (MW).
    """

    demand = pd.read_csv(fn_demand, index_col=0)
    demand.columns = demand.columns.astype(int)
    demand.index = n.snapshots

    intersection = set(demand.columns).intersection(n.buses.zone_id.unique())
    demand = demand[list(intersection)]

    demand_per_bus_pu = (n.buses.set_index("zone_id").Pd / n.buses.groupby("zone_id").sum().Pd)
    demand_per_bus = demand_per_bus_pu.multiply(demand)
    demand_per_bus.columns = n.buses.index

    n.madd( "Load", demand_per_bus.columns, bus=demand_per_bus.columns, p_set=demand_per_bus, carrier='AC')
    return n

def prepare_ads_files(file_patterns):
    for year, file_patterns_year in file_patterns.items():
        ads_filelist = glob.glob(os.path.join(snakemake.input[f'ads_{year}'], '*.csv'))
        for profile_type, pattern in file_patterns_year.items():
            read_ads_files(profile_type, [s for s in ads_filelist if pattern in s], year)

def read_ads_files(profiletype, paths, year):
    df_combined = pd.DataFrame()
    for i in range(len(paths)):
        df = pd.read_csv(paths[i], header=0, index_col=0,low_memory=False)
        df = df.iloc[1:8785, :]
        df_combined = pd.concat([df_combined, df], axis=1)
    df_combined.to_csv(os.path.join("resources/WECC_ADS/processed", f'{profiletype}_{year}.csv'))

def prepare_ads_load_data(ads_load_path, data_year):
    '''
    Modify ads load data to match the balancing authority names in the network. Need to test with all potential ADS years
    '''
    df_ads = pd.read_csv(ads_load_path)
    df_ads.columns = df_ads.columns.str.removeprefix('Load_')
    df_ads.columns = df_ads.columns.str.removesuffix('.dat')
    df_ads.columns = df_ads.columns.str.removesuffix(f'_{data_year}')
    df_ads.columns = df_ads.columns.str.removesuffix(f'_[18].dat: {data_year}')
    df_ads['CISO-PGAE'] = df_ads.pop('CIPV') + df_ads.pop('CIPB') + df_ads.pop('SPPC')#hotfix see github issue #15
    df_ads['BPAT'] = df_ads.pop('BPAT') + df_ads.pop('TPWR') + df_ads.pop('SCL')
    df_ads['IPCO'] = df_ads.pop('IPFE') + df_ads.pop('IPMV') + df_ads.pop('IPTV')
    df_ads['PACW'] = df_ads.pop('PAID') + df_ads.pop('PAUT') + df_ads.pop('PAWY')
    df_ads['Arizona'] = df_ads.pop('SRP') + df_ads.pop('AZPS') 
    df_ads.drop(columns=['Unnamed: 44', 'TH_Malin', 'TH_Mead', 'TH_PV'],inplace=True)
    ba_list_map = {'CISC': 'CISO-SCE', 'CISD': 'CISO-SDGE','VEA': 'CISO-VEA','WAUW': 'WAUW_SPP'}
    df_ads.rename(columns=ba_list_map,inplace=True)
    # df_ads['datetime'] = pd.Timestamp(f'{data_year}-01-01')+pd.to_timedelta(df_ads.index, unit='H')
    df_ads.set_index('Index',inplace=True)
    # if len(df_ads.index) > 8761: #remove leap year day
    #     df_ads= df_ads[~(df_ads.index.date == pd.to_datetime(f'{data_year}-04-29'))]

    # not_in_list = df_ads.loc[:,~df_ads.columns.isin(ba_list)]
    return df_ads

def add_ads_demand(n, demand):
    """
    Zone power demand is disaggregated to buses proportional to Pd,
    where Pd is the real power demand (MW).
    """
    demand.index = n.snapshots 
    # n.buses['ba_load_data'] = n.buses.balancing_area.replace({'CISO-PGAE': 'CISO', 'CISO-SCE': 'CISO', 'CISO-VEA': 'CISO', 'CISO-SDGE': 'CISO'})
    n.buses['ba_load_data'] = n.buses.balancing_area.replace({'': 'missing_ba'})

    intersection = set(demand.columns).intersection(n.buses.ba_load_data.unique())
    demand = demand[list(intersection)]

    demand_per_bus_pu = (n.buses.set_index("ba_load_data").Pd / n.buses.groupby("ba_load_data").sum().Pd)
    demand_per_bus = demand_per_bus_pu.multiply(demand)
    demand_per_bus.fillna(0,inplace=True)
    demand_per_bus.columns = n.buses.index

    n.madd( "Load", demand_per_bus.columns, bus=demand_per_bus.columns, p_set=demand_per_bus, carrier='AC') 
    return n


def add_eia_demand(n, demand):
    """
    Zone power demand is disaggregated to buses proportional to Pd,
    where Pd is the real power demand (MW).
    """
    demand.set_index('timestamp', inplace=True)
    demand.index = n.snapshots #maybe add check to make sure they match?

    demand['Arizona'] = demand.pop('SRP') + demand.pop('AZPS')
    n.buses['ba_load_data'] = n.buses.balancing_area.replace({'CISO-PGAE': 'CISO', 'CISO-SCE': 'CISO', 'CISO-VEA': 'CISO', 'CISO-SDGE': 'CISO'})
    n.buses['ba_load_data'] = n.buses.ba_load_data.replace({'': 'missing_ba'})

    intersection = set(demand.columns).intersection(n.buses.ba_load_data.unique())
    demand = demand[list(intersection)]

    demand_per_bus_pu = (n.buses.set_index("ba_load_data").Pd / n.buses.groupby("ba_load_data").sum().Pd)
    demand_per_bus = demand_per_bus_pu.multiply(demand)
    demand_per_bus.fillna(0,inplace=True)
    demand_per_bus.columns = n.buses.index

    n.madd( "Load", demand_per_bus.columns, bus=demand_per_bus.columns, p_set=demand_per_bus, carrier='AC') 

    return n
    

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input['network'])


    if sum([snakemake.config['load_data']['use_eia'], snakemake.config['load_data']['use_ads']], snakemake.config['load_data']['use_breakthrough']) > 1:
        raise ValueError("Only one of the load_data configs can be set to true")
    elif snakemake.config['load_data']['use_eia']:
        load_year = snakemake.config['load_data']['historical_year']
        logger.info(f'Building Load Data using EIA demand data year {load_year}')
        eia_demand = pd.read_csv(snakemake.input['eia'][load_year%2015])
        n.set_snapshots(pd.date_range(freq="h", start=f"{load_year}-01-01",
                                        end=f"{load_year+1}-01-01",
                                        inclusive="left")
                        )
        n = add_eia_demand(n, eia_demand)
    elif snakemake.config['load_data']['use_ads']:     ###### using ADS Data ######
        load_year = snakemake.config['load_data']['future_year']
        logger.info(f'Building Load Data using EIA Historical Data from year {load_year}')
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
        prepare_ads_files(file_patterns)
        demand = prepare_ads_load_data(f'resources/WECC_ADS/processed/load_{load_year}.csv',load_year)
        n.set_snapshots(pd.date_range(freq="h", start=f"{load_year}-01-01",
                                        end=f"{load_year+1}-01-01",
                                        inclusive="left")
                        )
        n = add_ads_demand(n,demand)
        n.export_to_netcdf(snakemake.output.network)
    elif snakemake.config['load_data']['use_breakthrough']:  # else standard breakthrough configuration
        logger.info("Adding Breakthrough Energy Network Demand data from 2016")
        n.set_snapshots(
            pd.date_range(freq="h", start="2016-01-01", end="2017-01-01", inclusive="left")
        )
        n = add_breakthrough_demand_from_file(n, snakemake.input["demand_breakthrough_2016"])
    else:
        raise ValueError("No load data specified in config.yaml")

        

    # import pdb; pdb.set_trace()
    n.export_to_netcdf(snakemake.output.network)