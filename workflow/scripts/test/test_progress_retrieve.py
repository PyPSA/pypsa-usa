"""``_helpers.progress_retrieve`` downloads files of any size."""

import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import progress_retrieve

pytestmark = pytest.mark.fast


@pytest.mark.parametrize("size", [1, 300, 8193, 320_000])
def test_any_file_size_downloads(tmp_path, size):
    """The last block overshoots the size; small files used to fail at 101%."""
    source = tmp_path / "source.bin"
    source.write_bytes(os.urandom(size))
    target = tmp_path / "target.bin"

    progress_retrieve(source.as_uri(), target)

    assert target.read_bytes() == source.read_bytes()
