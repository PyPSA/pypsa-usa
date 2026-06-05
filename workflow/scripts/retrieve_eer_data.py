"""Retrieve EER electricity demand profiles."""

import logging
from pathlib import Path

from _helpers import configure_logging, progress_retrieve

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_eer_demand_data")

    configure_logging(snakemake)

    output = Path(snakemake.output[0])
    output.parent.mkdir(parents=True, exist_ok=True)

    url = snakemake.params.url
    logger.info(f"Downloading EER demand data from '{url}'.")
    progress_retrieve(url, output)
