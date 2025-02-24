import logging
import zipfile
from pathlib import Path

from _helpers import configure_logging, progress_retrieve

logger = logging.getLogger(__name__)

# Note: when adding files to pypsa_usa_data.zip, be sure to zip the folder w/o the root folder included:
# ` cd pypsa_usa_data && zip -r ../pypsa_usa_data.zip . `


def download_egs_repository(interconnect, dispatch, subdir):
    # Save locations
    url = f"https://zenodo.org/records/14221666/files/{interconnect}_7km_{dispatch}.zip"

    tarball_fn = Path(f"{rootpath}/EGS_{interconnect}_{dispatch}.zip")
    to_fn = Path(f"{rootpath}/{subdir}")

    logger.info(f"Downloading EGS zenodo repository from '{url}'.")
    progress_retrieve(url, tarball_fn)

    logger.info(f"Extracting {dispatch} EGS databundle.")

    with zipfile.ZipFile(tarball_fn, "r") as zip_ref:
        zip_ref.extractall(to_fn)

    logger.info(f"{dispatch} EGS Databundle available in {to_fn}")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_egs")
        rootpath = ".."
    else:
        rootpath = "."
    configure_logging(snakemake)

    interconnect = snakemake.config["scenario"]["interconnect"][0]  # snakemake.wildcards.interconnect
    dispatch = snakemake.params.dispatch
    subdir = snakemake.params.subdir
    # interconnect = snakemake.params[1]
    # subdir = snakemake.output[0]

    download_egs_repository(interconnect, dispatch, subdir)
