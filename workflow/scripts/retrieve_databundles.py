"""Script retrieves data from various zenodo repositories specified by the snakemake rule. Used by multiple snakemake rules."""

import logging
import struct
import zipfile
import zlib
from pathlib import Path

import inflate64
from _helpers import configure_logging, progress_retrieve

logger = logging.getLogger(__name__)

#: Zip compression method of Deflate64, which the NREL EFS archives use and
#: Python's zipfile cannot read
ZIP_DEFLATE64 = 9
_LOCAL_FILE_HEADER = struct.Struct("<4s5H3I2H")
_CHUNK_BYTES = 1 << 20

# Note: when adding files to pypsa_usa_data.zip, be sure to zip the folder w/o the root folder included:
# ` cd pypsa_usa_data && zip -r ../pypsa_usa_data.zip . `


def _extract_deflate64_member(raw, info: zipfile.ZipInfo, to_dir: Path) -> None:
    """Write one Deflate64 member of the open archive ``raw`` under ``to_dir``."""
    target = (to_dir / info.filename).resolve()
    if not target.is_relative_to(to_dir.resolve()):
        raise zipfile.BadZipFile(f"{info.filename} would be extracted outside of {to_dir}")
    if info.flag_bits & 0x1:
        raise NotImplementedError(f"{info.filename} is encrypted")
    target.parent.mkdir(parents=True, exist_ok=True)

    # the compressed bytes follow the local header, whose name and extra fields vary in length
    raw.seek(info.header_offset)
    header = _LOCAL_FILE_HEADER.unpack(raw.read(_LOCAL_FILE_HEADER.size))
    if header[0] != b"PK\x03\x04":
        raise zipfile.BadZipFile(f"No local file header for {info.filename}")
    raw.seek(header[-2] + header[-1], 1)

    inflater = inflate64.Inflater()
    crc = 0
    remaining = info.compress_size
    with open(target, "wb") as out:
        while remaining > 0:
            chunk = raw.read(min(_CHUNK_BYTES, remaining))
            if not chunk:
                raise zipfile.BadZipFile(f"{info.filename} is truncated")
            remaining -= len(chunk)
            data = inflater.inflate(chunk)
            crc = zlib.crc32(data, crc)
            out.write(data)
    if crc != info.CRC:
        raise zipfile.BadZipFile(f"Bad CRC-32 for {info.filename}")


def extract_zip(zip_fn: Path | str, to_dir: Path | str) -> None:
    """
    Extract a zip archive, including members compressed with Deflate64.

    Python's zipfile handles every member but the Deflate64 ones, which are
    decompressed with ``inflate64``. No ``unzip`` or ``tar`` on the PATH is
    needed, so this works the same on Linux, macOS and Windows.
    """
    to_dir = Path(to_dir)
    with zipfile.ZipFile(zip_fn, "r") as archive, open(zip_fn, "rb") as raw:
        for info in archive.infolist():
            if info.compress_type == ZIP_DEFLATE64 and not info.is_dir():
                _extract_deflate64_member(raw, info, to_dir)
            else:
                archive.extract(info, to_dir)


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
    extract_zip(tarball_fn, to_fn)
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
