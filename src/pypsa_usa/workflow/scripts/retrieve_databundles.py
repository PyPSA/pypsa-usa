"""Script retrieves data from various zenodo repositories specified by the snakemake rule. Used by multiple snakemake rules."""

import logging
import platform
import subprocess
import zipfile
from pathlib import Path

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
    # When running from Snakemake, we need to extract to user_workspace/data/
    # When running standalone, we extract to the current directory
    if "snakemake" in globals():
        # Check if we're already in user_workspace directory
        current_dir = Path.cwd()
        if current_dir.name == "user_workspace":
            # We're already in user_workspace, so extract to data/ directly
            to_fn = Path(f"data/{subdir}")
        else:
            # We're in the root directory, so extract to user_workspace/data/
            to_fn = Path(f"user_workspace/data/{subdir}")
    else:
        to_fn = Path(f"{rootpath}/data/{subdir}")

    # Create the target directory if it doesn't exist
    to_fn.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {repository} zenodo repository from '{url}'.")
    progress_retrieve(url, tarball_fn)

    logger.info(f"Extracting {repository} databundle.")
    if (
        repository == "EFS"
    ):  # deflate64 compression not supported by zipFile, current subprocess command will only work on linux and mac
        if platform.system() == "Windows":
            cmd = ["tar", "-xf", tarball_fn, "-C", to_fn]
        else:
            # For EFS, extract directly to the target directory
            # The zip contains just the CSV file, not in a subdirectory
            cmd = ["unzip", tarball_fn, "-d", to_fn]
            subprocess.run(cmd, check=True)
    else:
        # For other repositories, extract to the target directory
        # The zip files contain the data structure we want
        with zipfile.ZipFile(tarball_fn, "r") as zip_ref:
            zip_ref.extractall(to_fn)
    logger.info(f"{repository} Databundle available in {to_fn}")


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
