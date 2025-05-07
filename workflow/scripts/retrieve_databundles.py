import io
import logging
import os
import platform
import subprocess
import zipfile
import zipfile_deflate64
from pathlib import Path

import requests
from _helpers import configure_logging, progress_retrieve

logger = logging.getLogger(__name__)

# Note: when adding files to pypsa_usa_data.zip, be sure to zip the folder w/o the root folder included:
# ` cd pypsa_usa_data && zip -r ../pypsa_usa_data.zip . `


def download_repository(url, rootpath, repository):
    # Save locations
    if repository == "USATestSystem":
        subdir = "breakthrough_network/"
    elif repository == "EFS":
        subdir = "nrel_efs/"
    else:
        subdir = ""
    tarball_fn = Path(f"{rootpath}/{repository}.zip")
    to_fn = Path(f"{rootpath}/data/{subdir}")

    logger.info(f"Downloading {repository} zenodo repository from '{url}'.")
    progress_retrieve(url, tarball_fn)

    logger.info(f"Extracting {repository} databundle.")
    try:
        if repository == "EFS":
            try:
                # First try to use zipfile-deflate64
                with zipfile_deflate64.ZipFile(tarball_fn, "r") as zip_ref:
                    zip_ref.extractall(to_fn)
            except Exception as e:
                logger.warning(f"Failed to extract using zipfile-deflate64: {e}")
                logger.info("Falling back to system commands...")
               
                # Fallback to system commands
                if platform.system() == "Windows":
                   cmd = ["tar", "-xf", tarball_fn, "-C", to_fn]
                else:
                   cmd = ["unzip", tarball_fn, "-d", to_fn]
                subprocess.run(cmd, check=True)
        else:
            with zipfile.ZipFile(tarball_fn, "r") as zip_ref:
                zip_ref.extractall(to_fn)
               
        logger.info(f"{repository} Databundle successfully extracted to {to_fn}")
       
    except Exception as e:
        logger.error(f"Failed to extract {repository} databundle: {e}")
        raise


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        # snakemake = mock_snakemake("retrieve_zenodo_databundles")
        # snakemake = mock_snakemake('retrieve_sector_databundle')
        snakemake = mock_snakemake(
            "retrieve_nrel_efs_data",
            efs_case="Reference",
            efs_speed="Moderate",
        )
        rootpath = ".."
    else:
        rootpath = "."
    configure_logging(snakemake)

    repositories = snakemake.params[0]
    for repository in repositories:
        url = repositories[repository]
        download_repository(url, rootpath, repository)