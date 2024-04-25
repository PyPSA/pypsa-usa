"""
Module to download end use load profiles (eulp) for comstock and restock data.

Notes:
    - Downloaded at state level
    - Multisector 15-min load profiles for a year (ie. lots of data)
    - Locked to 2018 Amy Weather data
    - https://data.openei.org/submissions/4520
"""

import logging
from pathlib import Path
from typing import List, Optional

import constants
import pandas as pd
import requests
from _helpers import configure_logging

logger = logging.getLogger(__name__)


class OediDownload:
    """
    Downlaods Oedi restock or comstock data at a state level.
    """

    root = "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024"

    res_files = [
        "mobile_home",
        "multi-family_with_2_-_4_units",
        "multi-family_with_5plus_units",
        "single-family_attached",
        "single-family_detached",
    ]

    com_files = [
        "fullservicerestaurant",
        "hospital",
        "largehotel",
        "largeoffice",
        "mediumoffice",
        "outpatient",
        "primaryschool",
        "quickservicerestaurant",
        "retailstandalone",
        "retailstripmall",
        "secondaryschool",
        "smallhotel",
        "smalloffice",
        "warehouse",
    ]

    def __init__(self, stock: str) -> None:
        assert stock in ["res", "com"]
        self.stock = stock
        self.release = 2 if self.stock == "res" else 1

    def _get_html_folder(self, state: str, upgrade: Optional[int] = 0) -> str:
        """
        Gets html of folder.
        """

        if self.stock == "res":
            data_folder = f"resstock_amy2018_release_{self.release}"
        elif self.stock == "com":
            data_folder = f"comstock_amy2018_release_{self.release}"

        return f"{self.root}/{data_folder}/timeseries_aggregates/by_state/upgrade={upgrade}/state={state.upper()}"

    def _get_htmls(
        self,
        state: str,
        buildings: str | list[str],
        upgrade: int = 0,
    ) -> list[str]:

        folder = self._get_html_folder(state)

        if isinstance(buildings, str):
            buildings = [buildings]

        upgrade_padded = f"{upgrade:02d}"

        htmls = []
        for building in buildings:
            htmls.append(f"{folder}/up{upgrade_padded}-{state.lower()}-{building}.csv")

        return htmls

    def _request_data(self, url: str, save: str) -> dict[str, dict | str]:

        response = requests.get(url)
        if response.status_code == 200:
            logger.info(f"Writing {save}")
            with open(save, "wb") as f:
                f.write(response.content)
        else:
            raise requests.ConnectionError(f"Status code {response.status_code}")

    def _get_building_from_html(self, html: str, state: int) -> str:
        return html.split(f"-{state.lower()}-")[-1].split(".csv")[0]

    def _create_save_dir(self, directory: str) -> None:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

    def download_data(
        self,
        state: str,
        buildings: Optional[str | list[str]] = None,
        upgrade: int = 0,
        directory: Optional[str] = None,
    ) -> None:
        """
        Public method to interface with.
        """

        if not directory:
            directory = f"{state}"
        else:
            directory = f"{directory}/{state}"

        self._create_save_dir(directory)

        if not buildings:
            if self.stock == "res":
                buildings = self.res_files
            elif self.stock == "com":
                buildings = self.com_files

        htmls = self._get_htmls(state, buildings)

        for html in htmls:
            save_name = self._get_building_from_html(html, state)
            save_path = f"{directory}/{save_name}.csv"
            self._request_data(html, save_path)


if __name__ == "__main__":

    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_res_eulp", state="WA")
    configure_logging(snakemake)

    stock = snakemake.params.stock
    state = snakemake.wildcards.state
    buildings = snakemake.params.profiles
    save_dir = snakemake.params.save_dir

    oedi = OediDownload(stock)

    oedi.download_data(state, buildings, 0, save_dir)
