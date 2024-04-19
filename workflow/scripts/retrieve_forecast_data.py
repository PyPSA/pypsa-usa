"""
Download forecast data from external sources (CEC, WECC) and save to resources
folder.

Written by Kamran Tehranchi, Stanford University.
"""

import glob
import io
import logging
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests
from _helpers import configure_logging, progress_retrieve

logger = logging.getLogger(__name__)


# configs and snakemake inputs
def download_wecc_forecasts(url, rootpath, year):
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
        pd.read_excel(save_path, sheet_name="Data", header=0, index_col=0).to_csv(
            csv_path,
        )
        os.remove(save_path)


def prepare_ads_files(file_patterns, path_2032, path_2030):
    for year, file_patterns_year in file_patterns.items():
        if year == 2032:
            path_year = path_2032
        elif year == 2030:
            path_year = path_2030
        else:
            raise ValueError(f"Invalid year {year}, must be 2030 or 2032")
        ads_filelist = glob.glob(os.path.join(path_year, "*.csv"))
        for profile_type, pattern in file_patterns_year.items():
            read_ads_files(
                profile_type,
                [s for s in ads_filelist if pattern in s],
                year,
            )


def read_ads_files(profiletype, paths, year):
    df_combined = pd.DataFrame()
    for i in range(len(paths)):
        df = pd.read_csv(paths[i], header=0, index_col=0, low_memory=False)
        df = df.iloc[1:8785, :]
        df_combined = pd.concat([df_combined, df], axis=1)
    df_combined.to_csv(
        os.path.join("data/WECC_ADS/processed", f"{profiletype}_{year}.csv"),
    )


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    rootpath = "./"

    logger.info("Downloading CEC Forecasts")
    PATH_CEC = Path(f"{rootpath}/data/cec_forecasts")
    PATH_CEC.mkdir(parents=True, exist_ok=True)
    URL_CEC = {
        "pge": "https://efiling.energy.ca.gov/GetDocument.aspx?tn=248357",
        "sce": "https://efiling.energy.ca.gov/GetDocument.aspx?tn=248355",
        "sdge": "https://efiling.energy.ca.gov/GetDocument.aspx?tn=248353",
    }
    # download_cec_forecasts(cec_urls, cecpath)

    logger.info("Downloading WECC ADS Forecasts")
    PATH_ADS = Path(f"{rootpath}/data/WECC_ADS")
    adspath_zips = Path(f"{rootpath}/data/WECC_ADS/downloads")
    PATH_ADS.mkdir(parents=True, exist_ok=True)
    URL_WECC_2032 = (
        "https://www.wecc.org/Reliability/2032%20ADS%20PCM%20V2.3.2%20Public%20Data.zip"
    )
    URL_WECC_2030 = "https://www.wecc.org/Reliability/WECC%202030%20ADS%20PCM%202020-12-16%20(V1.5)%20Public%20Data.zip"
    download_wecc_forecasts(URL_WECC_2032, adspath_zips, 2032)
    download_wecc_forecasts(URL_WECC_2030, adspath_zips, 2030)

    logger.info("Preproccessing ADS 2032 data")
    PATH_2032 = "data/WECC_ADS/downloads/2032/Public Data/Hourly Profiles in CSV format"
    PATH_2030 = "data/WECC_ADS/downloads/2030/WECC 2030 ADS PCM 2020-12-16 (V1.5) Public Data/CSV Shape Files"
    os.makedirs("data/WECC_ADS/processed/", exist_ok=True)
    file_patterns = {  # Processed file name : Unprocessed file name
        2032: {
            "load": "Profile_Load",
            "solar": "Profile_Solar",
            "wind": "Profile_Wind",
            "hydro": "Profile_Hydro",
            "btm_solar": "Profile_BTM Solar",
            "pumped_storage": "Profile_Pump Storage",
            "pump_load": "Profile_Pumps",
        },
        2030: {
            "load": "Data_Load",
            "solar": "Data_Solar PV",
            "wind": "Data_WT",
            "hydro": "Data_Hydro",
            "btm_solar": "Data_SolarPV_Rooftop",
            "pumped_storage": "Data_PumpStorage",
            "pump_load": "Data_Pump",
        },
    }
    prepare_ads_files(file_patterns, PATH_2032, PATH_2030)
