"""
Retrieves PUDL data.
"""

import logging
import zlib
from pathlib import Path

import requests
from _helpers import mock_snakemake
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
    url = "http://pudl.catalyst.coop.s3.us-west-2.amazonaws.com/stable/pudl.sqlite.gz"
    save_pudl = snakemake.output.pudl

    if not Path(save_pudl).exists():
        logger.info(f"Downloading PUDL from '{url}'")
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
                with open(save_pudl, "wb") as fd:
                    for chunk in r.iter_content(chunk_size=128):
                        progress_bar.update(len(chunk))
                        fd.write(d.decompress(chunk))
