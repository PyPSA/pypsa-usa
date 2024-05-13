"""
Retrieves PUDL data.
"""

import logging
from pathlib import Path

from _helpers import mock_snakemake, progress_retrieve

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_pudl")
        rootpath = ".."
    else:
        rootpath = "."

    # Recommended to use the stable version of PUDL documented here: https://catalystcoop-pudl.readthedocs.io/en/latest/data_access.html#stable-builds
    url = 'http://pudl.catalyst.coop.s3.us-west-2.amazonaws.com/stable/pudl.sqlite.gz'
    save_pudl = snakemake.output.pudl

    if not Path(save_pudl).exists():
        logger.info(f"Downloading PUDL from '{url}'")
        progress_retrieve(url, save_pudl)
