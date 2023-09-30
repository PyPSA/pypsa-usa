"""Retrives data for sector coupling

Geographic boundaries of the United States counties are taken from the 
United States Census Bureau. Note, these follow 2020 boundaries to match 
census numbers 

[![URL](https://img.shields.io/badge/URL-Cartographic_Boundaries-blue)](<https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.2020.html#list-tab-1883739534>)

County level populations are taken from the United States Census Bureau. Filters applied:
 - Geography: All Counties within United States and Puerto Rico
 - Year: 2020
 - Surveys: Decennial Census, Demographic and Housing Characteristics
 
Sheet Name: Decennial Census - P1 | Total Population - 2020: DEC Demographic and Housing Characteristics

[![URL](https://img.shields.io/badge/URL-United_States_Census_Bureau-blue)](<https://data.census.gov/>)

County level urbanization rates are taken from the United States Census Bureau. Filters applied:
 - Geography: All Counties within United States and Puerto Rico
 - Year: 2020
 - Surveys: Decennial Census, Demographic and Housing Characteristics
 
Sheet Name: Decennial Census - H1 | Housing Units - 2020: DEC Demographic and Housing Characteristics

[![URL](https://img.shields.io/badge/URL-United_States_Census_Bureau-blue)](<https://data.census.gov/>)
"""

import logging

logger = logging.getLogger(__name__)

import tarfile
from pathlib import Path

from _helpers import configure_logging, progress_retrieve

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_sector_databundle")
        rootpath = ".."
    else:
        rootpath = "."
    configure_logging(snakemake)

    url = "https://zenodo.org/record/5824485/files/pypsa-eur-sec-data-bundle.tar.gz"

    tarball_fn = Path(f"{rootpath}/sector-bundle.tar.gz")
    to_fn = Path(rootpath) / Path(snakemake.output[0]).parent.parent

    logger.info(f"Downloading databundle from '{url}'.")
    disable_progress = snakemake.config["run"].get("disable_progressbar", False)
    progress_retrieve(url, tarball_fn, disable=disable_progress)

    logger.info("Extracting databundle.")
    tarfile.open(tarball_fn).extractall(to_fn)

    tarball_fn.unlink()

    logger.info(f"Databundle available in '{to_fn}'.")