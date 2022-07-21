# Copyright 2021-2022 Martha Frysztacki (KIT)

"""
.. imagedata:: https://zenodo.org/record/4538590/files/USATestSystem.zip
   :target: https://doi.org/10.5281/zenodo.3530898


"""

import logging
import zipfile

from pathlib import Path

import sys
import os

sys.path.append(os.path.join("workflow", "subworkflows", "pypsa-eur", "scripts"))
from _helpers import progress_retrieve, configure_logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    rootpath = "."
    configure_logging(snakemake)

    url = snakemake.config["zenodo_repository"]["url"]

    # Save locations
    tarball_fn = Path(f"{rootpath}/USATestSystem.zip")
    to_fn = Path(f"{rootpath}/data")

    logger.info(f"Downloading databundle from '{url}'.")
    progress_retrieve(url, tarball_fn)

    logger.info(f"Extracting databundle.")
    with zipfile.ZipFile(tarball_fn, "r") as zip_ref:
        zip_ref.extractall(to_fn)

    logger.info(f"Databundle available in '{to_fn}'.")
