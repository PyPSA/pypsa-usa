"""master reads the GODEEEP CF files develop retrieved, never Zenodo.

Small-input test: a fake ``data/`` tree in ``tmp_path``. No pypsa, no network.
"""

from __future__ import annotations

from pathlib import Path

from tests.equivalence.build import mirror_godeeep_cf_for_master


def _seed(data: Path) -> Path:
    src = data / "godeeep" / "rcp85cooler" / "wind_gen_cf_2030_125m_compressed.nc"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"cf")
    (data / "godeeep" / "rcp85cooler" / "solar_gen_cf_2030_compressed.nc").write_bytes(b"pv")
    return src


def test_links_every_scenario_file(tmp_path: Path) -> None:
    data = tmp_path / "data"
    src = _seed(data)
    created = mirror_godeeep_cf_for_master(data)
    dst = data / "zenodo" / "rcp85cooler" / src.name
    assert dst in created and len(created) == 2
    assert dst.read_bytes() == b"cf"
    assert dst.stat().st_ino == src.stat().st_ino  # a hard link, not a copy


def test_existing_master_file_is_left_alone(tmp_path: Path) -> None:
    data = tmp_path / "data"
    src = _seed(data)
    dst = data / "zenodo" / "rcp85cooler" / src.name
    dst.parent.mkdir(parents=True)
    dst.write_bytes(b"already")
    assert mirror_godeeep_cf_for_master(data) == [data / "zenodo" / "rcp85cooler" / "solar_gen_cf_2030_compressed.nc"]
    assert dst.read_bytes() == b"already"


def test_no_godeeep_dir_is_a_noop(tmp_path: Path) -> None:
    assert mirror_godeeep_cf_for_master(tmp_path / "data") == []


def test_idempotent(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed(data)
    assert len(mirror_godeeep_cf_for_master(data)) == 2
    assert mirror_godeeep_cf_for_master(data) == []
