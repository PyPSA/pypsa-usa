'''
Downloads historical BA Data from EIA, proccesses and saves to a csv file.

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

logger = logging.getLogger(__name__)

def download_csvs(urls, folder_path):
    """
    Downloads a set of csv's from a list of urls and saves them in a folder. 
    To be used for downloading EIA data.

    Parameters:
    urls (list): A list of urls to download csv's from.
    folder_path (str): The path of the folder to save the csv's in.

    Returns:
    None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for url in urls:
        response = requests.get(url)
        file_name = url.split('/')[-1]
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'wb') as f:
            f.write(response.content)
            print(f"{file_name} downloaded successfully")


def read_and_concat_EIA_930(folder_path: str, 
                            output_folder_path: str):
    """
    Reads and cleans a set of EIA930 6 month file csvs.

    Parameters:
    folder_path (str): The path of the folder to read the csv's from.
    columns_to_keep (list): A list of column names to keep in the concatenated dataframe.
    output_folder_path (str): The path of the folder to save the concatenated csv.

    Returns:
    None
    """
    columns_to_keep = ['UTC Time at End of Hour', 'Balancing Authority', 'Demand (MW) (Adjusted)']
    dfs = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            year = file_name.split('_')[2]
            if year not in dfs:
                dfs[year] = []
            df = pd.read_csv(file_path, usecols=columns_to_keep, dtype={columns_to_keep[2]: str})
            df.columns = ['region','timestamp','demand_mw']
            df.demand_mw = df.demand_mw.str.replace(',', '')
            df.demand_mw = df.demand_mw.astype(float)
            dfs[year].append(df)

    for year, dfs_list in dfs.items():
        concatenated_df = pd.concat(dfs_list, ignore_index=True)
        concatenated_df = concatenated_df.set_index(['timestamp','region']).unstack(level=1)['demand_mw']
        concatenated_df.dropna(axis=1, how='all', inplace=True)
        concatenated_df.index = pd.to_datetime(concatenated_df.index)
        concatenated_df.sort_values(by=['timestamp'], inplace=True)
        concatenated_df.interpolate(method='linear', axis=0, inplace=True)
        
        output_file = os.path.join(output_folder_path, f'EIA_DMD_{year}.csv')
        concatenated_df.to_csv(output_file)


def prepare_eia_load_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modifies EIA Load Data Files from GridEmissions
    """
    df = df[['period', 'value', 'region']]
    df.columns = ['timestamp', 'value', 'region']
    df = df.drop_duplicates().dropna().set_index(['timestamp','region']).unstack(level=1)
    return df

def download_historical_load_data(url: str, output_path: str) -> pd.DataFrame:
    """
    Downloads historical load data from GridEmissions
    """
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
    """
    Old Function Need to check use.... Pretty sure used in GridEmission workflow. which we will switch back to.
    """
    # pattern = r'EBA\..*-ALL\.D\.H'  # Define the header filter pattern
    pattern = r'EBA\.(.*?)-ALL\.D\.H'
    pattern2 = r'E_\.(.*?)_\.D\.H'

    filtered_columns = [col for col in df.columns if re.match(pattern, col)]
    filtered_df = df[filtered_columns]
    updated_columns = [re.search(pattern, col).group(1) for col in filtered_columns]
    filtered_df.columns = updated_columns
    filtered_df.insert(0,"timestamp", "")
    if 'period' in df.columns:
        filtered_df.iloc[:,0] = pd.to_datetime(df['period'])
    else:
        filtered_df.iloc[:,0] = pd.to_datetime(df['Unnamed: 0'])
    df = filtered_df.set_index('timestamp')
    df = df.loc[f'{year}-01-01':f'{year}-12-31']
    return df

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    warnings.simplefilter(action='ignore', category=FutureWarning)

    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake("retrieve_eia_data")

    # URL of the gzipped CSV file.... Don't use these since Historical GridEmissions is Down
    url_2018_present = 'https://gridemissions.s3.us-east-2.amazonaws.com/EBA_elec.csv.gz'
    url_2015_2018 = 'https://gridemissions.s3.us-east-2.amazonaws.com/EBA_opt_no_src.csv.gz'
    url_2015_present = 'https://gridemissions.s3.us-east-2.amazonaws.com/EBA_raw.csv.gz'


    rootpath = "./"
    PATH_DOWNLOAD = Path(f"{rootpath}/data/eia")
    PATH_DOWNLOAD_CSV = Path(f"{rootpath}/data/eia/6moFiles")

    PATH_DOWNLOAD_CSV.mkdir(parents=True, exist_ok=True)
    PATH_DOWNLOAD.mkdir(parents=True, exist_ok=True)

    urls = [
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2023_Jan_Jun.csv',
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2022_Jul_Dec.csv',
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2022_Jan_Jun.csv',
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2021_Jul_Dec.csv',
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2021_Jan_Jun.csv',
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2020_Jul_Dec.csv',
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2020_Jan_Jun.csv',
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2019_Jul_Dec.csv',
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2019_Jan_Jun.csv',
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2018_Jul_Dec.csv',
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2018_Jan_Jun.csv',
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2017_Jul_Dec.csv',
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2017_Jan_Jun.csv',
    ]

    logger.info("Downloading EIA Data")

    download_csvs(urls, PATH_DOWNLOAD_CSV)
    read_and_concat_EIA_930(PATH_DOWNLOAD_CSV, PATH_DOWNLOAD)

    logger.info("EIA Data bundle downloaded.")