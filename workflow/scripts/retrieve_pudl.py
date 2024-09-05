"""
Retrieves PUDL data.
"""

import logging
import zipfile
import zlib
from pathlib import Path

import requests
from _helpers import mock_snakemake, progress_retrieve
from tqdm import tqdm

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_pudl")
        rootpath = ".."
    else:
        rootpath = "."

    # Recommended to use the stable version of PUDL documented here: https://catalystcoop-pudl.readthedocs.io/en/latest/data_access.html#stable-builds
    url_pudl = "https://zenodo.org/records/13346011/files/pudl.sqlite.zip?download=1"

    url_census = (
        "https://zenodo.org/records/13346011/files/censusdp1tract.sqlite.zip?download=1"
    )
    save_pudl = snakemake.output.pudl
    save_census = snakemake.output.census

    if not Path(save_census).exists():
        progress_retrieve(url_census, save_census + ".zip")
        with zipfile.ZipFile(save_census + ".zip", "r") as zip_ref:
            zip_ref.extractall(Path(save_census).parent)

    if not Path(save_pudl).exists():
        progress_retrieve(url_pudl, save_pudl + ".zip")
        with zipfile.ZipFile(save_pudl + ".zip", "r") as zip_ref:
            zip_ref.extractall(Path(save_pudl).parent)

    # Get PUDL FERC Form 714 Parquet
    parquet = f"https://zenodo.org/records/11292273/files/out_ferc714__hourly_estimated_state_demand.parquet?download=1"

    save_ferc = snakemake.output.pudl_ferc714

    if not Path(save_ferc).exists():
        logger.info(f"Downloading FERC 714 Demand from '{parquet}'")
        progress_retrieve(parquet, save_ferc)
