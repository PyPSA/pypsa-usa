"""Unit tests for remap_caps_to_cluster (build_renewable_profiles.py).

NREL caps files are rolled up against the NATIONAL substation tessellation;
footprint-scoped runs drop every entry outside the busmap. These tests pin:
  * the default (flag-off) path is byte-identical to the legacy dropna path,
  * the dropped-MW accounting warning reports correct numbers,
  * opt-in reassignment folds unmapped entries onto the nearest in-footprint
    cluster but only within max_km,
  * enabling reassignment against a caps file without x/y coordinates raises
    a clear config error (the published Zenodo caps predate the coordinate-
    preserving rollup in nrel_exclusion/build_nrel_bus_capacities.py).
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from build_renewable_profiles import remap_caps_to_cluster

pytestmark = pytest.mark.fast

# Geometry: entries 100/200 are in-footprint (mapped). Entry 300 sits ~91 km
# east of entry 200 (recoverable at max_km=100, not at 50). Entry 400 sits
# ~1,860 km away (never recoverable at sane max_km).
ENTRY_XY = {
    "100.0": (-120.0, 35.0),
    "200.0": (-119.0, 35.0),
    "300.0": (-118.0, 35.0),
    "400.0": (-100.0, 30.0),
}


def make_caps(with_coords: bool = True) -> xr.Dataset:
    bus = list(ENTRY_XY)
    data_vars = {
        "p_nom_max": ("bus", np.array([10.0, 20.0, 40.0, 80.0], dtype=np.float32)),
        "potential": ("bus", np.array([10.0, 20.0, 40.0, 80.0], dtype=np.float32)),
        "weight": ("bus", np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32)),
        "average_distance": ("bus", np.array([5.0, 10.0, 20.0, 40.0], dtype=np.float32)),
        "avg_cf": ("bus", np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float32)),
    }
    if with_coords:
        data_vars["x"] = ("bus", np.array([xy[0] for xy in ENTRY_XY.values()], dtype=np.float32))
        data_vars["y"] = ("bus", np.array([xy[1] for xy in ENTRY_XY.values()], dtype=np.float32))
    return xr.Dataset(data_vars, coords={"bus": bus})


@pytest.fixture
def busmap() -> pd.Series:
    # Bare-int string index (like busmap_s{simpl}.csv) so the float→int→str
    # normalization fallback is exercised, matching real caps keys "100.0".
    return pd.Series({"100": "p1 0", "200": "p2 0"})


def test_default_matches_legacy_dropna(busmap):
    """Flag off (or absent) must reproduce today's drop-unmapped behavior."""
    legacy = remap_caps_to_cluster(make_caps(), busmap)
    off = remap_caps_to_cluster(make_caps(), busmap, tech="onwind", reassign={"enable": False, "max_km": 100})
    xr.testing.assert_identical(legacy, off)

    assert sorted(legacy.bus.values) == ["p1 0", "p2 0"]
    assert legacy["p_nom_max"].sel(bus="p1 0") == pytest.approx(10.0)
    assert legacy["p_nom_max"].sel(bus="p2 0") == pytest.approx(20.0)


def test_coords_never_reach_output(busmap):
    """Per-entry x/y are consumed by the remap, not aggregated into outputs."""
    out = remap_caps_to_cluster(make_caps(), busmap)
    assert "x" not in out.data_vars and "y" not in out.data_vars
    out_on = remap_caps_to_cluster(make_caps(), busmap, tech="onwind", reassign={"enable": True, "max_km": 100})
    assert "x" not in out_on.data_vars and "y" not in out_on.data_vars


def test_dropped_accounting_warned(busmap, caplog):
    """Unconditional WARNING: dropped count, dropped MW, % of national total."""
    with caplog.at_level(logging.WARNING, logger="build_renewable_profiles"):
        remap_caps_to_cluster(make_caps(), busmap, tech="onwind")
    assert "onwind" in caplog.text
    assert "2/4 entries" in caplog.text
    assert "120.0 MW" in caplog.text  # 40 + 80 unmapped
    assert "80.0%" in caplog.text  # of national 150 MW


def test_reassign_recovers_within_max_km(busmap, caplog):
    """Entry 300 (~91 km from in-footprint entry 200) folds into p2 0; entry
    400 (~1,860 km away) stays dropped.
    """
    with caplog.at_level(logging.WARNING, logger="build_renewable_profiles"):
        out = remap_caps_to_cluster(
            make_caps(),
            busmap,
            tech="onwind",
            reassign={"enable": True, "max_km": 100},
        )

    assert sorted(out.bus.values) == ["p1 0", "p2 0"]
    # p1 0 untouched
    assert out["p_nom_max"].sel(bus="p1 0") == pytest.approx(10.0)
    # p2 0 absorbs entry 300: extensive sums, capacity-weighted intensive means
    assert out["p_nom_max"].sel(bus="p2 0") == pytest.approx(60.0)
    assert out["weight"].sel(bus="p2 0") == pytest.approx(6.0)
    assert out["avg_cf"].sel(bus="p2 0") == pytest.approx((0.4 * 2 + 0.5 * 4) / 6, abs=1e-5)
    assert out["average_distance"].sel(bus="p2 0") == pytest.approx((10 * 2 + 20 * 4) / 6, abs=1e-4)
    # recovery accounting logged
    assert "recovered 1" in caplog.text
    assert "40.0 MW" in caplog.text
    assert "80.0 MW) remain" in caplog.text


def test_reassign_respects_max_km(busmap):
    """max_km below the 300↔200 distance (~91 km) recovers nothing — output
    identical to the flag-off path.
    """
    tight = remap_caps_to_cluster(
        make_caps(),
        busmap,
        tech="onwind",
        reassign={"enable": True, "max_km": 50},
    )
    off = remap_caps_to_cluster(make_caps(), busmap)
    xr.testing.assert_identical(tight, off)


def test_reassign_without_coords_raises(busmap):
    """Published caps artifacts carry no x/y — enabling the flag must fail
    loudly and point at the HPC rollup regeneration.
    """
    with pytest.raises(ValueError, match="per-entry x/y"):
        remap_caps_to_cluster(
            make_caps(with_coords=False),
            busmap,
            tech="onwind",
            reassign={"enable": True, "max_km": 100},
        )


def test_no_coords_flag_off_still_works(busmap):
    """Without coordinates the default path is unaffected."""
    out = remap_caps_to_cluster(make_caps(with_coords=False), busmap, tech="onwind")
    assert sorted(out.bus.values) == ["p1 0", "p2 0"]
    assert out["p_nom_max"].sel(bus="p2 0") == pytest.approx(20.0)
