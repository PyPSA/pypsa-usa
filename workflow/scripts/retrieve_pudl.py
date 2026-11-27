"""Retrieves PUDL data."""

import logging
from pathlib import Path

from _helpers import mock_snakemake, progress_retrieve

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake("retrieve_pudl")
        rootpath = ".."
    else:
        rootpath = "."

    # Get PUDL FERC Form 714 Parquet
    parquet = "https://zenodo.org/records/11292273/files/out_ferc714__hourly_estimated_state_demand.parquet?download=1"

    save_ferc = snakemake.output.pudl_ferc714

    if not Path(save_ferc).exists():
        logger.info(f"Downloading FERC 714 Demand from '{parquet}'")
        progress_retrieve(parquet, save_ferc)
