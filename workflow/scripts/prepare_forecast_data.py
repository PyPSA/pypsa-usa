import pandas as pd
import glob, os
from pathlib import Path
# import urllib.request
import logging
import zipfile

from pathlib import Path
import os

from _helpers import progress_retrieve, configure_logging


logger = logging.getLogger(__name__)


#configs and snakemake inputs
path = '/Users/kamrantehranchi/Library/CloudStorage/OneDrive-Stanford/Kamran_OSW/WECC ADS PCM/ADS 2032/2032_ads_pcm_Public Data/Hourly Profiles in CSV format'
data_year = 2032
planning_horizons = [2025, 2030, 2035, 2040, 2045]

output_path = '/Users/kamrantehranchi/Local_Documents/pypsa-breakthroughenergy-usa/workflow/resources'

ba_list = ['AEC', 'AECI', 'AVA', 'Arizona', 'BANC', 'BPAT', 'CHPD', 'CISO-PGAE', 'CISO-SCE', 'CISO-SDGE', 'CISO-VEA', 'SPP-CSWS', 'Carolina', 'DOPD', 'SPP-EDE', 'EPE', 'ERCO-C', 'ERCO-E', 'ERCO-FW', 'ERCO-N', 'ERCO-NC', 'ERCO-S', 'ERCO-SC', 'ERCO-W', 'Florida', 'GCPD', 'SPP-GRDA', 'GRID', 'IID', 'IPCO', 'ISONE-Connecticut', 'ISONE-Maine', 'ISONE-Massachusetts', 'ISONE-New Hampshire', 'ISONE-Rhode Island', 'ISONE-Vermont', 'SPP-KACY', 'SPP-KCPL', 'LADWP', 'SPP-LES', 'MISO-0001', 'MISO-0027', 'MISO-0035', 'MISO-0004', 'MISO-0006', 'MISO-8910', 'SPP-MPS', 'MT_west', 'NEVP', 'SPP-NPPD', 'NYISO-A', 'NYISO-B', 'NYISO-C', 'NYISO-D', 'NYISO-E', 'NYISO-F', 'NYISO-G', 'NYISO-H', 'NYISO-I', 'NYISO-J', 'NYISO-K', 'SPP-OKGE', 'SPP-OPPD', 'PACE', 'PACW', 'PGE', 'PJM_AE', 'PJM_AEP', 'PJM_AP', 'PJM_ATSI', 'PJM_BGE', 'PJM_ComEd', 'PJM_DAY', 'PJM_DEO&K', 'PJM_DLCO', 'PJM_DP&L', 'PJM_Dominion', 'PJM_EKPC', 'PJM_JCP&L', 'PJM_METED', 'PJM_PECO', 'PJM_PENELEC', 'PJM_PEPCO', 'PJM_PPL', 'PJM_PSEG', 'PJM_RECO', 'PNM', 'PSCO', 'PSEI', 'SPP-SECI', 'SOCO', 'SPP-SPRM', 'SPP-SPS', 'TEPC', 'TID', 'TVA', 'WACM', 'WALC', 'SPP-WAUE_West','SPP-WAUE_2','SPP-WAUE_3','SPP-WAUE_4','SPP-WAUE_5','SPP-WAUE_6','SPP-WAUE_7','SPP-WAUE_8','SPP-WAUE_9', 'SPP-WFEC', 'SPP-WR']

planning_horizons = [2030, 2035, 2040 , 2045]

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

def combine_cec_ads_load(df_cec, df_ads,planning_horizons):
    df_ads

if __name__ == "__main__":
    
    logger = logging.getLogger(__name__)
    rootpath = "../" #remove . for snakemake
    # configure_logging(snakemake)

    adspath = Path(f"{rootpath}/resources/ads_private_data")
    cecpath = Path(f"{rootpath}/resources/cec_forecasts")

    #Use ADS Data
    # if snakemake.config['forecast_data']['use_ads']:
    load_path = Path(f"{adspath}/load_{data_year}.csv")
    df_ads = pd.read_csv(load_path,index_col=0)
    df_ads = preprocess_ads_load(df_ads,data_year)
    df_ads.to_csv(Path(f"{output_path}/ads_private_data/processed/load_{data_year}.csv").mkdir(parents=True, exist_ok=True))

    #Clean up CEC data
    df_cec_pge = pd.read_csv(cecpath/"pge.csv",index_col=0)
    datetime = pd.to_datetime(df_cec_pge.reset_index()[['YEAR','MONTH','DAY','HOUR']],format='%Y-%m-%d %H',yearfirst=True)
    df_cec_pge.index = datetime
    df_cec_pge.drop(columns=['YEAR','MONTH','DAY','HOUR'],inplace=True)


    investment_periods = planning_horizons #snakemake.config["planning_horizons"]
    snapshots = pd.DatetimeIndex([])
    for year in investment_periods:
        period = pd.date_range(
            start="{}-01-01 00:00".format(year),
            freq="1H",
            periods=8760 / float(1),
        )
        snapshots = snapshots.append(period)



    #Use BE Data
    # if not snakemake.config['forecast_data']['use_ads']:
    bepath = Path(f"{rootpath}/data")
    df_2030 = pd.read_csv(bepath/"2030_current_goals/demand.csv",index_col=0)
    df_2030

 pd.DataFrame({'year': [2015, 2016], 'month': [2, 3],'day': [4, 5]})

    #combine datasources into one long dataframe

    #save to csv





