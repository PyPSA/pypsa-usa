'''
Downloads historical BA Data from gridemissions tool, and saves it to a csv file.

github: https://github.com/jdechalendar/gridemissions
site: https://gridemissions.jdechalendar.su.domains/#/code
'''
import requests
import os
import pandas as pd
import gzip
from io import BytesIO
import logging
import re
from pathlib import Path
import warnings


def download_historical_load_data(url, output_path):
    response = requests.get(url)
    if response.status_code == 200:  # Check if the request was successful
        buffer = BytesIO(response.content)
        with gzip.open(buffer, 'rt') as file:
            df = pd.read_csv(file)
    else:
        print("Failed to download the gzipped file.")
    df.to_csv(output_path)
    return df

def prepare_historical_load_data(df, year):
    # pattern = r'EBA\..*-ALL\.D\.H'  # Define the header filter pattern
    pattern = r'EBA\.(.*?)-ALL\.D\.H'
    filtered_columns = [col for col in df.columns if re.match(pattern, col)]
    filtered_df = df[filtered_columns]
    updated_columns = [re.search(pattern, col).group(1) for col in filtered_columns]
    filtered_df.columns = updated_columns
    filtered_df.insert(0,"timestamp", "")
    filtered_df.iloc[:,0] = pd.to_datetime(df['Unnamed: 0'])
    df = filtered_df.set_index('timestamp')
    df = df.loc[f'{year}-01-01':f'{year}-12-31']
    return df

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    pd.options.mode.chained_assignment = None
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # URL of the gzipped CSV file
    url_2018_present = 'https://gridemissions.s3.us-east-2.amazonaws.com/EBA_elec.csv.gz'
    url_2015_2018 = 'https://gridemissions.s3.us-east-2.amazonaws.com/EBA_opt_no_src.csv.gz'

    rootpath = "./"
    PATH_DOWNLOAD = Path(f"{rootpath}/resources/eia")
    PATH_DOWNLOAD_RAW = Path(f"{rootpath}/resources/eia/raw")
    PATH_DOWNLOAD_RAW.mkdir(parents=True, exist_ok=True)
    PATH_DOWNLOAD.mkdir(parents=True, exist_ok=True)
    i = 0
    
    if os.path.isfile(os.path.join(PATH_DOWNLOAD, snakemake.output[len(snakemake.output)-1])):
        logger.info("EIA Data bundle already downloaded.")
    else:
        logger.info("Downloading EIA Data")
        print('Downloading EIA Data')       # Download the gzipped CSV file
        df_2015 = download_historical_load_data(url_2015_2018,  os.path.join(PATH_DOWNLOAD_RAW,  '2015_2018_raw.csv'))
        df_present = download_historical_load_data(url_2018_present, os.path.join(PATH_DOWNLOAD_RAW, '2018_present_raw.csv'))

        for year in range(2015, 2024):
            if year >= 2019:
                df = prepare_historical_load_data(df_present, year)
            elif year == 2018:
                df_1 = prepare_historical_load_data(df_2015, year)
                df_2 = prepare_historical_load_data(df_present, year)
                df = pd.concat([df_1.iloc[:-1, :], df_2])  # one duplicate row removed
            else:
                df = prepare_historical_load_data(df_2015, year)
            df.to_csv(os.path.join(snakemake.output[i]))
            logger.info('saving to ', os.path.join(snakemake.output[i]))
            i += 1

        logger.info("EIA Data bundle downloaded.")