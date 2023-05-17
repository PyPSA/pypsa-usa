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
output_path = '/Users/kamrantehranchi/Local_Documents/pypsa-breakthroughenergy-usa/workflow/resources'

def download_wecc_forecasts(url, rootpath,year):
    save_to_path = Path(f"{rootpath}/{year}")
    if os.path.isfile(save_to_path):
        logger.info(f"Data bundle already downloaded.")
    else:
        logger.info(f"Downloading databundle from '{url}'.")
        r = requests.get(url, stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(save_to_path)

def download_cec_forecasts(urls, path):
    for key, url in urls.items():
        save_path = Path(f"{path}/{key}.xlsx")
        progress_retrieve(url, save_path)
        csv_path = Path(f"{path}/{key}.csv")
        pd.read_excel(save_path, sheet_name='Data', header=0, index_col=0).to_csv(csv_path)
        os.remove(save_path)

def preprocess_ads_data(profiletype, paths, output_path, year):
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
    df_combined.to_csv(os.path.join(output_path, 'ads_private_data', f'{profiletype}_{data_year}.csv'))
    

if __name__ == "__main__":
    
    logger = logging.getLogger(__name__)
    rootpath = "../" #remove . for snakemake
    # configure_logging(snakemake)

    #Download CEC forecasts
    logger.info("Downloading CEC Forecasts")
    cecpath = Path(f"{rootpath}/resources/cec_forecasts")
    cecpath.mkdir(parents=True, exist_ok=True)
    cec_urls = {'pge':'https://efiling.energy.ca.gov/GetDocument.aspx?tn=248357', 'sce':'https://efiling.energy.ca.gov/GetDocument.aspx?tn=248355','sdge':'https://efiling.energy.ca.gov/GetDocument.aspx?tn=248353'}
    # download_cec_forecasts(cec_urls, cecpath)

    logger.info("Downloading WECC ADS Forecasts")
    #Download ADS Data
    adspath = Path(f"{rootpath}/resources/ads_private_data")
    adspath_zips = Path(f"{rootpath}/resources/ads_private_data/downloads")
    adspath.mkdir(parents=True, exist_ok=True)
    wecc_url_2032 = 'https://www.wecc.org/Reliability/2032%20ADS%20PCM%20V2.3.2%20Public%20Data.zip'
    wecc_url_2030 = 'https://www.wecc.org/Reliability/WECC%202030%20ADS%20PCM%202020-12-16%20(V1.5)%20Public%20Data.zip'
    download_wecc_forecasts(wecc_url_2032, adspath_zips, 2032)
    download_wecc_forecasts(wecc_url_2030, adspath_zips, 2030)

    #Process ASD PCM data
    logger.info("Preprocessing ADS data")
    wecc_path_2032 = 'resources/ads_private_data/downloads/2032/Public Data/Hourly Profiles in CSV format'
    wecc_path_2030 = 'resources/ads_private_data/downloads/2030/WECC 2030 ADS PCM 2020-12-16 (V1.5) Public Data/CSV Shape Files'
    ads_filelist = []
    for file in glob.glob(os.path.join(rootpath,wecc_path_2032,"*.csv")):
        ads_filelist.append(os.path.join(file))
    data_year = 2032
    preprocess_ads_data('Load', [s for s in ads_filelist if "Profile_Load" in s], output_path,data_year)
    preprocess_ads_data('Solar', [s for s in ads_filelist if "Profile_Solar" in s], output_path,data_year)
    preprocess_ads_data('wind', [s for s in ads_filelist if "Profile_Wind" in s], output_path,data_year)

    for file in glob.glob(os.path.join(rootpath,wecc_path_2030,"*.csv")):
        ads_filelist.append(os.path.join(rootpath,wecc_path_2030,file))
    data_year = 2030
    preprocess_ads_data('Load', [s for s in ads_filelist if "Profile_Load" in s], output_path,data_year)
    preprocess_ads_data('Solar', [s for s in ads_filelist if "Profile_Solar" in s], output_path,data_year)
    preprocess_ads_data('wind', [s for s in ads_filelist if "Profile_Wind" in s], output_path,data_year)




