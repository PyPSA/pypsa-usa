'''
Preprocesses Historical and Forecasted Load, Solar, and Wind Data

Written by Kamran Tehranchi, Stanford University.
'''
import pandas as pd
import glob, os
from pathlib import Path
# import urllib.request
import logging
import zipfile
import requests, zipfile, io
from pathlib import Path
import os

from _helpers import progress_retrieve, configure_logging

logger = logging.getLogger(__name__)

#configs and snakemake inputs
PATH_DATABUNDLE = '/Users/kamrantehranchi/Local_Documents/pypsa-breakthroughenergy-usa/workflow/resources'

def preprocess_ads_data(profiletype, paths, PATH_OUTPUT, year):
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
        df.columns = df.columns.str.removeprefix('PV_')
        df.columns = df.columns.str.removeprefix('WT_')
        
        df.columns = df.columns.str.removesuffix('.dat')
        df.columns = df.columns.str.removesuffix(f'_{year}')
        df.columns = df.columns.str.removesuffix(f'_[18].dat: {year}')
        df = df.iloc[1:8785, :]
        df_combined = pd.concat([df_combined, df], axis=1)
    df_combined.to_csv(os.path.join(PATH_OUTPUT, 'WECC_ADS', f'{profiletype}_{data_year}.csv'))
    

if __name__ == "__main__":
    
    logger = logging.getLogger(__name__)
    rootpath = "../" #remove . for snakemake
    # configure_logging(snakemake)

    #Process ASD PCM data
    logger.info("Preprocessing ADS data")
    PATH_ADS_2032 = 'resources/WECC_ADS/downloads/2032/Public Data/Hourly Profiles in CSV format'
    PATH_ADS_2030 = 'resources/WECC_ADS/downloads/2030/WECC 2030 ADS PCM 2020-12-16 (V1.5) Public Data/CSV Shape Files'
    ads_filelist = []
    for file in glob.glob(os.path.join(rootpath,PATH_ADS_2032,"*.csv")):
        ads_filelist.append(os.path.join(file))
    data_year = 2032
    preprocess_ads_data('Load', [s for s in ads_filelist if "Profile_Load" in s], PATH_DATABUNDLE,data_year)
    preprocess_ads_data('Solar', [s for s in ads_filelist if "Profile_Solar" in s], PATH_DATABUNDLE,data_year)
    preprocess_ads_data('wind', [s for s in ads_filelist if "Profile_Wind" in s], PATH_DATABUNDLE,data_year)

    for file in glob.glob(os.path.join(rootpath,PATH_ADS_2030,"*.csv")):
        ads_filelist.append(os.path.join(rootpath,PATH_ADS_2030,file))
    data_year = 2030
    preprocess_ads_data('Load', [s for s in ads_filelist if "Profile_Load" in s], PATH_DATABUNDLE,data_year)
    preprocess_ads_data('Solar', [s for s in ads_filelist if "Profile_Solar" in s], PATH_DATABUNDLE,data_year)
    preprocess_ads_data('wind', [s for s in ads_filelist if "Profile_Wind" in s], PATH_DATABUNDLE,data_year)


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


