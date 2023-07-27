import logging, zipfile, os, io, requests
from pathlib import Path
from _helpers import progress_retrieve, configure_logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    rootpath = "."
    configure_logging(snakemake)

    repositories = snakemake.config["zenodo_repositories"]
    for repository in repositories:
        url = repositories[repository]

        # Save locations
        if repository == 'USATestSystem':
            subdir = 'breakthrough_network/'
        else:
            subdir = ''
        tarball_fn = Path(f"{rootpath}/{repository}.zip")
        to_fn = Path(f"{rootpath}/data/{subdir}")

        if os.path.isfile(tarball_fn):
            logger.info(f"{repository} zenodo repository already downloaded.")
        else:
            logger.info(f"Downloading {repository} zenodo repositoy from '{url}'.")
            progress_retrieve(url, tarball_fn)
        
        logger.info(f"Extracting {repository} databundle.")
        with zipfile.ZipFile(tarball_fn, "r") as zip_ref:
            zip_ref.extractall(to_fn)
        logger.info(f'{repository} Databundle available in {to_fn}')