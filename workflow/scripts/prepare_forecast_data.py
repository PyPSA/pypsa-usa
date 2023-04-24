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
output_path = '/Users/kamrantehranchi/Local_Documents/pypsa-breakthroughenergy-usa/workflow/resources'

ba_list = ['AEC', 'AECI', 'AVA', 'Arizona', 'BANC', 'BPAT', 'CHPD', 'CISO-PGAE', 'CISO-SCE', 'CISO-SDGE', 'CISO-VEA', 'SPP-CSWS', 'Carolina', 'DOPD', 'SPP-EDE', 'EPE', 'ERCO-C', 'ERCO-E', 'ERCO-FW', 'ERCO-N', 'ERCO-NC', 'ERCO-S', 'ERCO-SC', 'ERCO-W', 'Florida', 'GCPD', 'SPP-GRDA', 'GRID', 'IID', 'IPCO', 'ISONE-Connecticut', 'ISONE-Maine', 'ISONE-Massachusetts', 'ISONE-New Hampshire', 'ISONE-Rhode Island', 'ISONE-Vermont', 'SPP-KACY', 'SPP-KCPL', 'LADWP', 'SPP-LES', 'MISO-0001', 'MISO-0027', 'MISO-0035', 'MISO-0004', 'MISO-0006', 'MISO-8910', 'SPP-MPS', 'MT_west', 'NEVP', 'SPP-NPPD', 'NYISO-A', 'NYISO-B', 'NYISO-C', 'NYISO-D', 'NYISO-E', 'NYISO-F', 'NYISO-G', 'NYISO-H', 'NYISO-I', 'NYISO-J', 'NYISO-K', 'SPP-OKGE', 'SPP-OPPD', 'PACE', 'PACW', 'PGE', 'PJM_AE', 'PJM_AEP', 'PJM_AP', 'PJM_ATSI', 'PJM_BGE', 'PJM_ComEd', 'PJM_DAY', 'PJM_DEO&K', 'PJM_DLCO', 'PJM_DP&L', 'PJM_Dominion', 'PJM_EKPC', 'PJM_JCP&L', 'PJM_METED', 'PJM_PECO', 'PJM_PENELEC', 'PJM_PEPCO', 'PJM_PPL', 'PJM_PSEG', 'PJM_RECO', 'PNM', 'PSCO', 'PSEI', 'SPP-SECI', 'SOCO', 'SPP-SPRM', 'SPP-SPS', 'TEPC', 'TID', 'TVA', 'WACM', 'WALC', 'SPP-WAUE_West','SPP-WAUE_2','SPP-WAUE_3','SPP-WAUE_4','SPP-WAUE_5','SPP-WAUE_6','SPP-WAUE_7','SPP-WAUE_8','SPP-WAUE_9', 'SPP-WFEC', 'SPP-WR']


if __name__ == "__main__":
    
    logger = logging.getLogger(__name__)
    rootpath = "../" #remove . for snakemake
    # configure_logging(snakemake)

    adspath = Path(f"{rootpath}/resources/ads_private_data")
    cecpath = Path(f"{rootpath}/resources/cec_forecasts")

    #Use ADS Data
    # if snakemake.config['forecast_data']['use_ads']:
    load_path = Path(f"{adspath}/load_{data_year}.csv")
    df = pd.read_csv(load_path,index_col=0)

    df.columns
    df.columns[df.columns.isin(ba_list)]
    df.columns[~df.columns.isin(ba_list)]

    #clean up ADS data

    #clean up CEC data


    #Use BE Data
    # if not snakemake.config['forecast_data']['use_ads']:
    bepath = Path(f"{rootpath}/data")
    df_2030 = pd.read_csv(bepath/"2030_current_goals/demand.csv",index_col=0)
    

    #combine datasources into one long dataframe

    #save to csv





