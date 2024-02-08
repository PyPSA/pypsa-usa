import logging, zipfile, os, io, requests
from pathlib import Path
from _helpers import progress_retrieve, configure_logging
import subprocess


logger = logging.getLogger(__name__)

#Note: when adding files to pypsa_usa_data.zip, be sure to zip the folder w/o the root folder included:
# ` cd pypsa_usa_data && zip -r ../pypsa_usa_data.zip . `

def download_repository(url, rootpath, repository):
    # Save locations
    if repository == 'USATestSystem':
        subdir = 'breakthrough_network/'
    elif repository == 'EFS':
        subdir = 'nrel_efs/'
    else:
        subdir = ''
    tarball_fn = Path(f"{rootpath}/{repository}.zip")
    to_fn = Path(f"{rootpath}/data/{subdir}")

    logger.info(f"Downloading {repository} zenodo repository from '{url}'.")
    progress_retrieve(url, tarball_fn)

    logger.info(f"Extracting {repository} databundle.")
    if repository == 'EFS': #deflate64 compression not supported by zipFile, current subprocess command will only work on linux and mac
        cmd = ['unzip', tarball_fn, '-d', to_fn]
        subprocess.run(cmd, check=True)
    else:
        with zipfile.ZipFile(tarball_fn, "r") as zip_ref:
            zip_ref.extractall(to_fn)
    logger.info(f'{repository} Databundle available in {to_fn}')

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        # snakemake = mock_snakemake("retrieve_zenodo_databundles")
        # snakemake = mock_snakemake('retrieve_sector_databundle')
        snakemake = mock_snakemake("retrieve_nrel_efs_data")
        rootpath = ".."
    else:
        rootpath = "."
    configure_logging(snakemake)

    repositories = snakemake.params[0]
    for repository in repositories:
        url = repositories[repository]
        download_repository(url, rootpath, repository)