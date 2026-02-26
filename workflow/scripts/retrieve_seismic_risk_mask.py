"""
Download the EGS seismic risk mask from Zenodo.

Dataset: Seismic Risk Map for Enhanced Geothermal Systems in the Continental United States
DOI: 10.5281/zenodo.18793962
URL: https://zenodo.org/records/18793962
"""

import logging
from pathlib import Path

from _helpers import configure_logging, progress_retrieve

logger = logging.getLogger(__name__)

ZENODO_RECORD = "18793962"
ZENODO_FILE = "seismic_risk_mask.geojson"
DOWNLOAD_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{ZENODO_FILE}?download=1"


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_seismic_risk_mask")
    configure_logging(snakemake)

    outpath = Path(snakemake.output[0])
    outpath.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading EGS seismic risk mask from Zenodo (record {ZENODO_RECORD}).")
    progress_retrieve(DOWNLOAD_URL, outpath)
    logger.info(f"Saved to {outpath}")
