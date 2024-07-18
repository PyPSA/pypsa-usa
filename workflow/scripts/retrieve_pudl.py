"""
Retrieves PUDL data.
"""

import logging
import zlib
from pathlib import Path

import requests
from _helpers import mock_snakemake, progress_retrieve
from tqdm import tqdm

logger = logging.getLogger(__name__)


def retrieve_gzip(url: str, save: str):
    """
    Retrieves a gzip file from a URL and saves it to a local file.

    Args:
        url (str): URL of the gzip file to retrieve.
        save (str): Path to save the gzip file to.
    """
    logger.info(f"Downloading Data from '{url}'")
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            with open(save, "wb") as fd:
                for chunk in r.iter_content(chunk_size=128):
                    progress_bar.update(len(chunk))
                    fd.write(d.decompress(chunk))


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_pudl")
        rootpath = ".."
    else:
        rootpath = "."

    # Recommended to use the stable version of PUDL documented here: https://catalystcoop-pudl.readthedocs.io/en/latest/data_access.html#stable-builds
    url_pudl = (
        "http://pudl.catalyst.coop.s3.us-west-2.amazonaws.com/stable/pudl.sqlite.gz"
    )
    url_census = (
        "https://zenodo.org/records/11292273/files/censusdp1tract.sqlite.gz?download=1"
    )
    save_pudl = snakemake.output.pudl
    save_census = snakemake.output.census

    if not Path(save_census).exists():
        retrieve_gzip(url_census, save_census)

    if not Path(save_pudl).exists():
        retrieve_gzip(url_pudl, save_pudl)

    # Get PUDL FERC Form 714 Parquet
    parquet = f"https://zenodo.org/records/11292273/files/out_ferc714__hourly_estimated_state_demand.parquet?download=1"

    save_ferc = snakemake.output.pudl_ferc714

    if not Path(save_ferc).exists():
        logger.info(f"Downloading FERC 714 Demand from '{parquet}'")
        progress_retrieve(parquet, save_ferc)
