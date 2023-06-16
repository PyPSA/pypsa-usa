'''
Download forecast data from external sources (CEC, WECC) and save to resources folder.

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

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    rootpath = "../" #remove . for snakemake
    # configure_logging(snakemake)

    #Download CEC forecasts
    logger.info("Downloading CEC Forecasts")
    PATH_CEC = Path(f"{rootpath}/resources/cec_forecasts")
    PATH_CEC.mkdir(parents=True, exist_ok=True)
    URL_CEC = {'pge':'https://efiling.energy.ca.gov/GetDocument.aspx?tn=248357', 'sce':'https://efiling.energy.ca.gov/GetDocument.aspx?tn=248355','sdge':'https://efiling.energy.ca.gov/GetDocument.aspx?tn=248353'}
    # download_cec_forecasts(cec_urls, cecpath)

    logger.info("Downloading WECC ADS Forecasts")
    #Download ADS Data
    PATH_ADS = Path(f"{rootpath}/resources/WECC_ADS")
    adspath_zips = Path(f"{rootpath}/resources/WECC_ADS/downloads")
    PATH_ADS.mkdir(parents=True, exist_ok=True)
    URL_WECC_2032 = 'https://www.wecc.org/Reliability/2032%20ADS%20PCM%20V2.3.2%20Public%20Data.zip'
    URL_WECC_2030 = 'https://www.wecc.org/Reliability/WECC%202030%20ADS%20PCM%202020-12-16%20(V1.5)%20Public%20Data.zip'
    download_wecc_forecasts(URL_WECC_2032, adspath_zips, 2032)
    download_wecc_forecasts(URL_WECC_2030, adspath_zips, 2030)


