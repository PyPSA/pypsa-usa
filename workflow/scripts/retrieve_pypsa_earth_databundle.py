import logging, zipfile, os
from pathlib import Path

from _helpers import progress_retrieve, configure_logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    rootpath = "."
    configure_logging(snakemake)

    url = snakemake.config["pypsa_earth_repository"]["url"]

    # Save locations
    tarball_fn = Path(f"{rootpath}/data_v0.1.zip.zip")
    to_fn = Path(f"{rootpath}/data/pypsa_earth/")

    if os.path.isfile(tarball_fn):
        logger.info(f"Data bundle already downloaded.")
    else:
        logger.info(f"Downloading databundle from '{url}'.")
        progress_retrieve(url, tarball_fn)
    
    logger.info(f"Extracting databundle.")
    with zipfile.ZipFile(tarball_fn, "r") as zip_ref:
        zip_ref.extractall(to_fn)

    logger.info(f'Databundle available in {to_fn}')
