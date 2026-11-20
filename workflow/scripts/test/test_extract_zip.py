"""Zip extraction of ``retrieve_databundles``, including Deflate64 members (#466).

The NREL EFS load profiles are zipped with Deflate64 (compression method 9),
which Python's ``zipfile`` cannot read.
"""

import os
import struct
import sys
import zipfile
import zlib

import inflate64
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from retrieve_databundles import ZIP_DEFLATE64, extract_zip

pytestmark = pytest.mark.fast

CSV = b"State,Hour,LoadMW\n" + b"CA,1,12345.6\n" * 50_000  # larger than the 64 KiB Deflate64 window


def _write_zip(path, members):
    """Write a zip of ``(name, method, payload, crc, size)`` members by hand.

    ``zipfile`` cannot write Deflate64, so the headers are packed here.
    """
    central = b""
    with open(path, "wb") as fh:
        for name, method, payload, crc, size in members:
            offset = fh.tell()
            encoded = name.encode()
            fh.write(
                struct.pack(
                    "<4s5H3I2H", b"PK\x03\x04", 20, 0, method, 0, 0x21, crc, len(payload), size, len(encoded), 0
                )
            )
            fh.write(encoded + payload)
            central += struct.pack(
                "<4s6H3I5H2I",
                b"PK\x01\x02",
                20,
                20,
                0,
                method,
                0,
                0x21,
                crc,
                len(payload),
                size,
                len(encoded),
                0,
                0,
                0,
                0,
                0,
                offset,
            )
            central += encoded
        start = fh.tell()
        fh.write(central)
        fh.write(struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, len(members), len(members), len(central), start, 0))


def _deflate64(data: bytes) -> bytes:
    deflater = inflate64.Deflater()
    return deflater.deflate(data) + deflater.flush()


@pytest.fixture
def efs_like_zip(tmp_path):
    """An archive with one Deflate64 member in a folder and one stored member."""
    readme = b"EFS load profiles\n"
    path = tmp_path / "EFS.zip"
    _write_zip(
        path,
        [
            ("profiles/EFSLoadProfile.csv", ZIP_DEFLATE64, _deflate64(CSV), zlib.crc32(CSV), len(CSV)),
            ("README.txt", zipfile.ZIP_STORED, readme, zlib.crc32(readme), len(readme)),
        ],
    )
    return path


def test_pythons_zipfile_cannot_read_the_archive(efs_like_zip, tmp_path):
    """Guards the premise: if this starts passing, the workaround can go."""
    with zipfile.ZipFile(efs_like_zip) as archive, pytest.raises(NotImplementedError):
        archive.extractall(tmp_path / "plain")


def test_deflate64_and_other_members_are_extracted(efs_like_zip, tmp_path):
    out = tmp_path / "data"

    extract_zip(efs_like_zip, out)

    assert (out / "profiles" / "EFSLoadProfile.csv").read_bytes() == CSV
    assert (out / "README.txt").read_bytes() == b"EFS load profiles\n"


def test_a_corrupt_deflate64_member_is_rejected(tmp_path):
    path = tmp_path / "bad.zip"
    _write_zip(path, [("profile.csv", ZIP_DEFLATE64, _deflate64(CSV), zlib.crc32(CSV) ^ 1, len(CSV))])

    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        extract_zip(path, tmp_path / "data")


def test_a_member_cannot_escape_the_target_folder(tmp_path):
    path = tmp_path / "evil.zip"
    _write_zip(path, [("../escaped.csv", ZIP_DEFLATE64, _deflate64(CSV), zlib.crc32(CSV), len(CSV))])

    with pytest.raises(zipfile.BadZipFile, match="outside"):
        extract_zip(path, tmp_path / "data")
    assert not (tmp_path / "escaped.csv").exists()
