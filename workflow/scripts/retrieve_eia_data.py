"""
**Description**

Historical electrical load data from 2019-2023 are retrieved from the `US Energy Information Agency <https://www.eia.gov/>`_ (EIA) and `GridEmissions <https://gridemissions.jdechalendar.su.domains/#/code>`_. Data is downloaded at hourly temporal resolution and at a spatial resolution of balancing authority region.

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
    return filtered_df, df


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    warnings.simplefilter(action="ignore", category=FutureWarning)

    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_eia_data")

    grid_emissions_data_url = "https://gridemissions.s3.us-east-2.amazonaws.com/processed.tar.gz"

    PATH_DOWNLOAD = Path(f"data/GridEmissions")
    PATH_DOWNLOAD.mkdir(parents=True, exist_ok=True)
    download_and_extract(grid_emissions_data_url, PATH_DOWNLOAD)
    demand_df, df = prepare_historical_load_data(PATH_DOWNLOAD)
    demand_df.to_csv(f"{snakemake.output[0]}")
    df.to_csv(f"{snakemake.output[1]}")
    logger.info("GridEmissions Demand Data bundle downloaded.")
