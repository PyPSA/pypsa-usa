# By PyPSA-USA Authors
"""
Historical daily natural gas fuel prices are retrieved from CAISO's Open Access
Same-time Information System (OASIS). Data is collected on a daily basis for
each Balancing Area and Fuel Region that had joined the Western Energy
Imbalance Market (WEIM) during the time period designated in the configuration
``fuel_year``.

.. image:: https://img.shields.io/badge/URL-CAISO_OASIS-blue
    :target: http://www.caiso.com/participate/Pages/oasis.aspx
    :alt: CAISO

**Relevant Settings**

.. code-block:: yaml

    fuel_year:

**Inputs**

- ``repo_data/wecc_fuelregions.xlsx``: A list of fuel regions and their corresponding Balancing Authorities.

**Outputs**

- ``data/fuel_prices.csv``: A CSV file containing the daily average fuel prices for each Balancing Authority in the WEIM.
"""

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import seaborn as sns


def download_oasis_report(
    queryname,
    startdatetime,
    enddatetime,
    version,
    node="ALL",
    resultformat="6",
):
    """
    Download a report from CAISO's OASIS, tailored for fuel prices.

    Args:
    - queryname: Name of the query, e.g., 'PRC_FUEL'.
    - startdatetime: Start datetime in 'YYYYMMDDTHH:MM-0000' format.
    - enddatetime: End datetime in 'YYYYMMDDTHH:MM-0000' format.
    - version: Version of the report.
    - node: Specific fuel region ID or 'ALL' for all regions.
    - resultformat: Format of the result ('6' for CSV, '5' for XML).

    Returns:
    - None. Downloads the file to the current directory.
    """
    base_url = "http://oasis.caiso.com/oasisapi/SingleZip"
    params = {
        "queryname": queryname,
        "startdatetime": startdatetime,
        "enddatetime": enddatetime,
        "version": version,
        "fuel_region_id": node,  # Use 'fuel_region_id' instead of 'node' for clarity
        "resultformat": resultformat,
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        filename = f"{queryname}_{startdatetime}_{enddatetime}.{resultformat}.zip"
        filename = filename.replace(":", "_")  # Replace colons with underscores
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f"Report downloaded successfully: {filename}")
    else:
        print(f"Failed to download report. Status Code: {response.status_code}")


def generate_monthly_intervals(year):
    """
    Generate monthly start and end datetime strings for a given year.
    """
    intervals = []
    for month in range(1, 13):
        start_date = datetime(year, month, 1)
        end_date = (start_date + timedelta(days=31)).replace(day=1) - timedelta(days=1)
        intervals.append(
            (
                start_date.strftime("%Y%m%dT%H:%M-0000"),
                end_date.strftime("%Y%m%dT%H:%M-0000"),
            ),
        )
    return intervals


def step_download_oasis_reports(
    queryname,
    version,
    node="ALL",
    resultformat="6",
    year=2019,
):
    """
    Download and combine OASIS reports for each month of a given year into a
    single DataFrame.
    """
    monthly_intervals = generate_monthly_intervals(year)
    file_names = []
    for startdatetime, enddatetime in monthly_intervals:
        download_oasis_report(
            queryname,
            startdatetime,
            enddatetime,
            version,
            node,
            resultformat,
        )
        filename = f"/{queryname}_{startdatetime}_{enddatetime}.{resultformat}.zip"
        file_names.append(filename)
        time.sleep(5)

    return file_names


def combine_reports(file_names, year):
    """
    Combine all reports into a single DataFrame.
    """
    all_data_frames = []
    for file in file_names:
        file = file.replace(":", "_")
        df = pd.read_csv(os.getcwd() + "/" + file, compression="zip")
        all_data_frames.append(df)

    combined_data = pd.concat(all_data_frames, ignore_index=True)
    combined_data.sort_values(by="INTERVALSTARTTIME_GMT", inplace=True)
    return combined_data


def get_files_starting_with(folder_path, prefix):
    """
    Get all file names in a folder that start with a particular string.

    Args:
    - folder_path: Path to the folder.
    - prefix: The string that the file names should start with.

    Returns:
    - A list of file names that start with the specified prefix.
    """
    file_names = []
    for file_name in os.listdir(folder_path):
        if file_name.startswith(prefix):
            file_names.append(file_name)
    return file_names


def merge_fuel_regions_data(combined_data, year):
    """
    Merge the fuel regions with the combined data.
    """
    df = pd.read_excel(snakemake.input.fuel_regions, sheet_name="GPI_Fuel_Region")
    df = df[["Fuel Region", "Balancing Authority"]]
    df["Fuel Region"] = df["Fuel Region"].str.strip(" ")

    combined_data_merged = pd.merge(
        combined_data,
        df,
        left_on="FUEL_REGION_ID",
        right_on="Fuel Region",
        how="left",
    )
    combined_data_merged.drop(
        columns=["Fuel Region", "FUEL_REGION_ID_XML"],
        inplace=True,
    )
    return combined_data_merged


def reduce_select_pricing_nodes(combined_data_merged):
    """
    Reduces data to day of year and Balancing Authority.

    Averages across all pricing nodes for each day of year and Balancing
    Authority
    """
    combined_data_merged["day_of_year"] = pd.to_datetime(
        combined_data_merged["INTERVALSTARTTIME_GMT"],
    ).dt.dayofyear

    avg_doy = (
        combined_data_merged[["day_of_year", "Balancing Authority", "PRC"]]
        .groupby(["day_of_year", "Balancing Authority"])
        .mean()
    )
    return avg_doy


def main(snakemake):

    fuel_year = snakemake.params.fuel_year

    file_names = step_download_oasis_reports(
        queryname="PRC_FUEL",
        version="1",
        node="ALL",
        resultformat="6",
        year=fuel_year,
    )

    combined_data = combine_reports(file_names, fuel_year)

    combined_data_merged = merge_fuel_regions_data(combined_data, year=fuel_year)
    reduced_fuel_price_data = reduce_select_pricing_nodes(combined_data_merged)

    reduced_fuel_price_data.to_csv(snakemake.output.fuel_prices)


if __name__ == "__main__":
    main(snakemake)
