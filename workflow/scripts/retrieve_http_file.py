"""Download one workflow input from the URL in ``snakemake.params.url``."""

import logging
from pathlib import Path

from _helpers import configure_logging, progress_retrieve

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "retrieve_reeds_vre_file",
            reeds_vre_tech="solar",
            reeds_vre_file="sc_upv_reference.csv",
        )
    configure_logging(snakemake)
    output = Path(snakemake.output[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s.", snakemake.params.url, output)
    progress_retrieve(snakemake.params.url, output)
