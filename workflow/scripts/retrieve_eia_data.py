"""
**Description**

Historical electrical load data from 2019-2023 are retrieved from the [US Energy Information Agency](https://www.eia.gov/) (EIA) and [GridEmissions](https://gridemissions.jdechalendar.su.domains/#/code). Data is downloaded at hourly temporal resolution and at a spatial resolution of balancing authority region.

**Outputs**

- ``data/GridEmissions/EIA_DMD_2018_2024.csv``
- ``data/eia/EIA_DMD_*.csv``
"""

import glob
import gzip
import logging
import os
import re
import tarfile
import warnings
from io import BytesIO
from pathlib import Path

import pandas as pd
import progressbar
import requests

logger = logging.getLogger(__name__)


def download_csvs(urls, folder_path):
    """
    Downloads a set of csv's from a list of urls and saves them in a folder. To
    be used for downloading EIA data.

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
        file_name = url.split("/")[-1]
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, "wb") as f:
            f.write(response.content)
            print(f"{file_name} downloaded successfully")


def read_and_concat_EIA_930(
    folder_path: str,
    output_folder_path: str,
):
    """
    Reads and cleans a set of EIA930 6 month file csvs.

    Parameters:
    folder_path (str): The path of the folder to read the csv's from.
    columns_to_keep (list): A list of column names to keep in the concatenated dataframe.
    output_folder_path (str): The path of the folder to save the concatenated csv.

    Returns:
    None
    """
    columns_to_keep = [
        "UTC Time at End of Hour",
        "Balancing Authority",
        "Demand (MW) (Adjusted)",
    ]
    dfs = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            year = file_name.split("_")[2]
            if year not in dfs:
                dfs[year] = []
            df = pd.read_csv(
                file_path,
                usecols=columns_to_keep,
                dtype={columns_to_keep[2]: str},
            )
            df.columns = ["region", "timestamp", "demand_mw"]
            df.demand_mw = df.demand_mw.str.replace(",", "")
            df.demand_mw = df.demand_mw.astype(float)
            dfs[year].append(df)

    for year, dfs_list in dfs.items():
        concatenated_df = pd.concat(dfs_list, ignore_index=True)
        concatenated_df = concatenated_df.set_index(["timestamp", "region"]).unstack(
            level=1,
        )["demand_mw"]
        concatenated_df.dropna(axis=1, how="all", inplace=True)
        concatenated_df.index = pd.to_datetime(concatenated_df.index)
        concatenated_df.sort_values(by=["timestamp"], inplace=True)
        concatenated_df.interpolate(method="linear", axis=0, inplace=True)

        output_file = os.path.join(output_folder_path, f"EIA_DMD_{year}.csv")
        concatenated_df.to_csv(output_file)


def download_and_extract(url, extract_path):
    # Get the file name from the URL
    filename = url.split("/")[-1]

    # Start the download
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))

    # Setup progress bar
    with open(filename, "wb") as file, progressbar.ProgressBar(
        max_value=total_size_in_bytes,
    ) as bar:
        for data in response.iter_content(1024):
            file.write(data)
            bar.update(bar.value + len(data))

    # Check if the download was successful
    if total_size_in_bytes != 0 and bar.value != total_size_in_bytes:
        print("ERROR, something went wrong with the download")
        return

    # Extract the .tar.gz file
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=extract_path)

    # Optional: Remove the .tar.gz file after extraction
    os.remove(filename)
    print(f"File extracted to {extract_path}")


def prepare_historical_load_data(PATH_DOWNLOAD: str) -> None:
    """
    Combines and filters EIA Load Data Files from GridEmissions files.

    Returns single dataframe of all demand data.
    """
    file_paths = glob.glob(f"{PATH_DOWNLOAD}/processed/*_elec.csv")
    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dfs.append(df)
    df = pd.concat(dfs)
    df["period"] = pd.to_datetime(df["period"])
    df.sort_values(by="period", inplace=True)

    pattern = (
        r".*_D$"  # Define the header filter pattern to match columns ending with "_D"
    )
    filtered_columns = [col for col in df.columns if re.match(pattern, col)]
    filtered_columns.insert(0, "period")
    filtered_df = df[filtered_columns]
    updated_columns = [
        re.sub(r"^E_|_D$", "", col) for col in filtered_columns
    ]  # Remove 'E_' and '_D' from the column names
    filtered_df.columns = updated_columns

    filtered_df = filtered_df.rename(columns={"period": "timestamp"}).set_index(
        "timestamp",
    )
    return filtered_df


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    warnings.simplefilter(action="ignore", category=FutureWarning)

    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_eia_data")

    # URL of the gzipped CSV file.... Don't use these since Historical GridEmissions is Down
    url_2018_present = (
        "https://gridemissions.s3.us-east-2.amazonaws.com/EBA_elec.csv.gz"
    )
    url_2015_2018 = (
        "https://gridemissions.s3.us-east-2.amazonaws.com/EBA_opt_no_src.csv.gz"
    )
    url_2015_present = "https://gridemissions.s3.us-east-2.amazonaws.com/EBA_raw.csv.gz"

    url_new = "https://gridemissions.s3.us-east-2.amazonaws.com/processed.tar.gz"

    PATH_DOWNLOAD = Path(f"../data/GridEmissions")
    PATH_DOWNLOAD.mkdir(parents=True, exist_ok=True)
    download_and_extract(url_new, PATH_DOWNLOAD)
    df = prepare_historical_load_data(PATH_DOWNLOAD)
    df.to_csv(f"{snakemake.output[0]}")
    logger.info("GridEmissions Demand Data bundle downloaded.")

    # EIA 6 mo file method
    rootpath = "./"
    PATH_DOWNLOAD = Path(f"{rootpath}/data/eia")
    PATH_DOWNLOAD_CSV = Path(f"{rootpath}/data/eia/6moFiles")

    PATH_DOWNLOAD_CSV.mkdir(parents=True, exist_ok=True)
    PATH_DOWNLOAD.mkdir(parents=True, exist_ok=True)

    urls = [
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2023_Jul_Dec.csv',
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2023_Jan_Jun.csv',
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2022_Jul_Dec.csv',
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2022_Jan_Jun.csv',
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2021_Jul_Dec.csv',
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2021_Jan_Jun.csv',
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2020_Jul_Dec.csv',
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2020_Jan_Jun.csv',
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2019_Jul_Dec.csv',
        'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2019_Jan_Jun.csv',
        # 'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2018_Jul_Dec.csv',
        # 'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2018_Jan_Jun.csv',
        # 'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2017_Jul_Dec.csv',
        # 'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2017_Jan_Jun.csv',
    ]
    logger.info("Downloading EIA Data")
    download_csvs(urls, PATH_DOWNLOAD_CSV)
    read_and_concat_EIA_930(PATH_DOWNLOAD_CSV, PATH_DOWNLOAD)
