"""Unit tests for the multi-weather-year GODEEEP fallback.

The NREL land-access path (#745) reads per-cell COMPRESSED GODEEEP records,
but the *historical* compressed records hold only weather year 2012. Every
other historical year falls back to the pre-#745 bus-aggregated archives
(solar 1980-2022 / wind 2001-2022), which are keyed by the unclustered
network's substation IDs and must be remapped through ``busmap_s{simpl}.csv``.

These tests pin:
  * year routing — 2012 stays on the compressed path, other years fall back,
    out-of-range years raise, climate scenarios never fall back,
  * the gating: the fallback is opt-in (`godeeep_allow_unscreened_fallback`,
    default false) and hard-errors when a renewable carrier is extendable or
    when a run would mix screened (2012) and unscreened weather years,
  * the provenance attrs stamped onto the output .nc,
  * aggregated filename / Zenodo record-key construction, including the fact
    that the aggregated wind archive is 100 m regardless of the 125 m
    ``godeeep_wind_height`` used for the compressed records,
  * ``remap_aggregated_profile`` on a synthetic archive + busmap: the
    capacity-weighted cluster mean, the unweighted fallback for zero-weight
    clusters, out-of-footprint substations getting dropped, and schema
    equality with the profile the compressed path emits,
  * the fallback WARNING text (including the hub-height note).

No network access: every test builds its inputs in memory.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from build_renewable_profiles import (
    GODEEEP_AGGREGATED_HISTORICAL_YEARS,
    GODEEEP_AGGREGATED_WIND_HEIGHT,
    check_godeeep_fallback_allowed,
    check_godeeep_weather_year_consistency,
    godeeep_aggregated_filename,
    godeeep_aggregated_record_key,
    godeeep_profile_source,
    map_bus_keys,
    remap_aggregated_profile,
)
from zenodo_downloader import ZenodoScenarioDownloader

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------
# Year routing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("technology", ["solar", "wind"])
def test_2012_stays_on_compressed_path(technology):
    """2012 is the only historical year the compressed records carry."""
    assert godeeep_profile_source("historical", technology, 2012) == "compressed"


@pytest.mark.parametrize(
    ("technology", "year"),
    [("solar", 2019), ("solar", 1980), ("solar", 2022), ("wind", 2019), ("wind", 2001), ("wind", 2022)],
)
def test_other_historical_years_fall_back(technology, year):
    assert godeeep_profile_source("historical", technology, year) == "aggregated"


def test_solar_covers_more_years_than_wind():
    """Solar 1980-2022 (43 files); wind 2001-2022 (22 files)."""
    assert godeeep_profile_source("historical", "solar", 1995) == "aggregated"
    with pytest.raises(ValueError, match="2001-2022"):
        godeeep_profile_source("historical", "wind", 1995)


@pytest.mark.parametrize("technology", ["solar", "wind"])
def test_year_beyond_archive_raises(technology):
    with pytest.raises(ValueError, match="not available"):
        godeeep_profile_source("historical", technology, 2030)


@pytest.mark.parametrize("scenario", ["rcp45hotter", "rcp85cooler"])
@pytest.mark.parametrize("technology", ["solar", "wind"])
def test_climate_scenarios_never_fall_back(scenario, technology):
    """Only the historical archive was ever published bus-aggregated."""
    assert godeeep_profile_source(scenario, technology, 2050) == "compressed"


def test_unknown_technology_raises():
    with pytest.raises(ValueError, match="Unknown GODEEEP technology"):
        godeeep_profile_source("historical", "hydro", 2019)


def test_archive_year_ranges_match_zenodo_manifests():
    """43 solar files (1980-2022), 22 wind files (2001-2022)."""
    assert len(GODEEEP_AGGREGATED_HISTORICAL_YEARS["solar"]) == 43
    assert len(GODEEEP_AGGREGATED_HISTORICAL_YEARS["wind"]) == 22


# --------------------------------------------------------------------------
# Gating: the fallback is opt-in and never drives capacity expansion
# --------------------------------------------------------------------------


def test_fallback_is_refused_by_default():
    """`godeeep_allow_unscreened_fallback` defaults to false; no silent fallback."""
    with pytest.raises(ValueError, match="godeeep_allow_unscreened_fallback"):
        check_godeeep_fallback_allowed(2019, "solar", allow_fallback=False)


def test_default_config_ships_the_gate_closed():
    """workflow/config/ is gitignored; repo_data/config/ is the tracked source."""
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    tracked = os.path.join(root, "repo_data", "config", "config.common.yaml")
    with open(tracked) as fh:
        assert "godeeep_allow_unscreened_fallback: false" in fh.read()

    # If the working copy has been initialised, it must agree.
    working = os.path.join(root, "config", "config.common.yaml")
    if os.path.exists(working):
        with open(working) as fh:
            assert "godeeep_allow_unscreened_fallback: false" in fh.read()


def test_new_config_key_is_documented():
    docs = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "docs",
        "source",
        "configtables",
        "nrel_exclusion.csv",
    )
    with open(docs) as fh:
        assert "godeeep_allow_unscreened_fallback" in fh.read()


def test_opt_in_allows_an_operational_run():
    """Nothing extendable → the fallback is permitted."""
    check_godeeep_fallback_allowed(
        2019,
        "solar",
        allow_fallback=True,
        extendable_generators=["nuclear", "CCGT"],
    )


@pytest.mark.parametrize("carrier", ["solar", "onwind", "offwind", "offwind_floating"])
def test_extendable_renewables_hard_error_even_when_opted_in(carrier):
    """Unscreened profiles must never drive capacity-expansion siting."""
    with pytest.raises(ValueError, match="extendable_carriers"):
        check_godeeep_fallback_allowed(
            2019,
            "solar",
            allow_fallback=True,
            extendable_generators=["CCGT", carrier],
        )


def test_extendable_error_names_the_offending_carriers():
    with pytest.raises(ValueError) as exc:
        check_godeeep_fallback_allowed(
            2019,
            "onwind",
            allow_fallback=True,
            extendable_generators=["onwind", "solar", "battery"],
        )
    assert "'onwind'" in str(exc.value) and "'solar'" in str(exc.value)
    assert "battery" not in str(exc.value)


def test_mixing_screened_and_unscreened_years_hard_errors():
    """2012 is screened, 2019 is not — a run must not blend them."""
    with pytest.raises(ValueError, match="mixes NREL-screened and unscreened"):
        check_godeeep_weather_year_consistency("historical", "solar", [2012, 2019])


@pytest.mark.parametrize(
    ("years", "expected"),
    [([2012], "compressed"), ([2019], "aggregated"), ([2018, 2019, 2020], "aggregated")],
)
def test_consistent_year_sets_pass_and_report_their_source(years, expected):
    assert check_godeeep_weather_year_consistency("historical", "solar", years) == expected


def test_empty_weather_years_raises():
    with pytest.raises(ValueError, match="empty"):
        check_godeeep_weather_year_consistency("historical", "solar", [])


def test_mixing_check_runs_for_every_historical_run():
    """Regression: the mixing gate must not hang off the fallback branch.

    ``year`` is ``renewable_weather_years[0]``, so a [2012, 2019] run routes
    THIS invocation to the screened path. Gating the consistency check on
    ``source == "aggregated"`` would let that mismatch through.
    """
    source = _script_source()
    body = source.split("source = godeeep_profile_source(")[1]
    consistency_at = body.index("check_godeeep_weather_year_consistency(")
    fallback_at = body.index('if source == "aggregated":')
    assert consistency_at < fallback_at, (
        "check_godeeep_weather_year_consistency must run before (and outside) the `source == 'aggregated'` branch."
    )
    assert 'if scenario == "historical":' in body[:consistency_at]


# --------------------------------------------------------------------------
# Filename / record-key construction
# --------------------------------------------------------------------------


def test_solar_filename():
    assert godeeep_aggregated_filename("solar", 2019) == "solar_gen_cf_2019_aggregated.nc"


def test_wind_filename_is_always_100m():
    """The 125 m ``godeeep_wind_height`` must not leak into this filename."""
    assert godeeep_aggregated_filename("wind", 2019) == "wind_gen_cf_2019_100m_aggregated.nc"
    assert GODEEEP_AGGREGATED_WIND_HEIGHT == "_100m"


def test_filename_unknown_technology_raises():
    with pytest.raises(ValueError, match="Unknown GODEEEP technology"):
        godeeep_aggregated_filename("geothermal", 2019)


def test_record_keys_are_registered_in_the_downloader(tmp_path):
    """The routing keys must resolve to the old multi-year Zenodo records."""
    downloader = ZenodoScenarioDownloader(download_dir=tmp_path)
    assert downloader.scenario_records[godeeep_aggregated_record_key("solar")] == 18293999
    assert downloader.scenario_records[godeeep_aggregated_record_key("wind")] == 18331699


def test_record_key_names():
    assert godeeep_aggregated_record_key("solar") == "solar_historical_aggregated"
    assert godeeep_aggregated_record_key("wind") == "wind_100m_historical_aggregated"


# --------------------------------------------------------------------------
# Bus-key normalization
# --------------------------------------------------------------------------


def test_map_bus_keys_tolerates_float_and_int_spellings():
    """Archives write "35827.0"; busmap_s{simpl}.csv is indexed by "35827"."""
    busmap = pd.Series({"35827": "p10 0", "35828": "p10 1"})
    out = map_bus_keys(["35827.0", "35828.0", "99999.0"], busmap)
    assert out["35827.0"] == "p10 0"
    assert out["35828.0"] == "p10 1"
    assert pd.isna(out["99999.0"])


# --------------------------------------------------------------------------
# remap_aggregated_profile
# --------------------------------------------------------------------------

# Substations 100/101 → cluster "p1 0"; 200 → "p1 1"; 300 → "p1 2" (zero caps
# weight); 900 is outside the run footprint and must be dropped.
BUSMAP = pd.Series(
    {"100": "p1 0", "101": "p1 0", "200": "p1 1", "300": "p1 2"},
    name="cluster_bus",
)
ARCHIVE_BUSES = ["100.0", "101.0", "200.0", "300.0", "900.0"]
TIMES = pd.date_range("2019-01-01", periods=4, freq="h")


def _archive(values):
    """Build a synthetic aggregated archive DataArray, dims (time, bus)."""
    return xr.DataArray(
        np.asarray(values, dtype="float32"),
        coords={"time": TIMES, "bus": np.asarray(ARCHIVE_BUSES)},
        dims=("time", "bus"),
        name="profile",
    )


# columns:      100.0 101.0 200.0 300.0 900.0
VALUES = [
    [0.10, 0.20, 0.50, 0.70, 0.99],
    [0.20, 0.40, 0.60, 0.80, 0.99],
    [0.30, 0.60, 0.70, 0.90, 0.99],
    [0.40, 0.80, 0.80, 1.00, 0.99],
]
# 100 gets 3x the caps capacity of 101; 300 has no developable land.
WEIGHTS = pd.Series({"100.0": 300.0, "101.0": 100.0, "200.0": 50.0, "300.0": 0.0})


def test_remap_produces_capacity_weighted_cluster_mean():
    out = remap_aggregated_profile(_archive(VALUES), BUSMAP, weights=WEIGHTS, tech="solar")

    assert out.dims == ("time", "bus")
    assert list(out.bus.values) == ["p1 0", "p1 1", "p1 2"]

    # p1 0 = (300 * sub100 + 100 * sub101) / 400
    expected = (300 * np.array([0.10, 0.20, 0.30, 0.40]) + 100 * np.array([0.20, 0.40, 0.60, 0.80])) / 400
    np.testing.assert_allclose(out.sel(bus="p1 0").values, expected, rtol=1e-6)

    # p1 1 has a single substation — weighting is a no-op.
    np.testing.assert_allclose(out.sel(bus="p1 1").values, [0.50, 0.60, 0.70, 0.80], rtol=1e-6)


def test_zero_weight_cluster_falls_back_to_unweighted_mean():
    out = remap_aggregated_profile(_archive(VALUES), BUSMAP, weights=WEIGHTS, tech="solar")
    # p1 2's only substation has p_nom_max == 0, so the weighted mean is 0/0.
    np.testing.assert_allclose(out.sel(bus="p1 2").values, [0.70, 0.80, 0.90, 1.00], rtol=1e-6)
    assert not np.isnan(out.values).any()


def test_out_of_footprint_substations_are_dropped():
    out = remap_aggregated_profile(_archive(VALUES), BUSMAP, weights=WEIGHTS)
    assert "900.0" not in out.bus.values
    # 0.99 is the sentinel value carried only by the out-of-footprint bus.
    assert not np.isclose(out.values, 0.99).any()


def test_remap_without_weights_is_an_unweighted_mean():
    out = remap_aggregated_profile(_archive(VALUES), BUSMAP, weights=None)
    expected = (np.array([0.10, 0.20, 0.30, 0.40]) + np.array([0.20, 0.40, 0.60, 0.80])) / 2
    np.testing.assert_allclose(out.sel(bus="p1 0").values, expected, rtol=1e-6)


def test_remap_accepts_bare_int_archive_keys():
    """Some archives may spell the substation key without the ``.0``."""
    da = _archive(VALUES).assign_coords(bus=np.asarray(["100", "101", "200", "300", "900"]))
    out = remap_aggregated_profile(da, BUSMAP, weights=WEIGHTS)
    assert list(out.bus.values) == ["p1 0", "p1 1", "p1 2"]


def test_remap_raises_when_nothing_maps():
    da = _archive(VALUES)
    with pytest.raises(RuntimeError, match="none of the"):
        remap_aggregated_profile(da, pd.Series({"55555": "q9 0"}), tech="solar")


def test_remap_rejects_wrong_dims():
    da = _archive(VALUES).rename({"bus": "sub_id"})
    with pytest.raises(ValueError, match=r"\(time, bus\) dims"):
        remap_aggregated_profile(da, BUSMAP)


def test_schema_matches_the_compressed_path_output():
    """The fallback profile must be drop-in for the NREL path's profile.

    The compressed path emits ``weighted_bus_aggregation(...)["profile"]``:
    a float32 DataArray named "profile" over (time, bus) with a datetime64
    time coord and string bus labels. add_electricity then does
    ``ds["profile"].transpose("time", "bus").to_pandas()``.
    """
    compressed_like = xr.DataArray(
        np.zeros((len(TIMES), 3), dtype="float32"),
        coords={"time": TIMES.to_numpy(), "bus": np.asarray(["p1 0", "p1 1", "p1 2"])},
        dims=("time", "bus"),
        name="profile",
    )
    out = remap_aggregated_profile(_archive(VALUES), BUSMAP, weights=WEIGHTS)

    assert out.name == compressed_like.name
    assert out.dims == compressed_like.dims
    assert out.dtype == compressed_like.dtype
    assert out.sizes == compressed_like.sizes
    assert out.time.dtype == compressed_like.time.dtype
    assert out.bus.values.tolist() == compressed_like.bus.values.tolist()
    assert np.issubdtype(out.bus.dtype, np.str_)

    # The merge + downstream consumption both round-trip cleanly.
    merged = xr.merge([out.rename("profile")], compat="override")
    frame = merged["profile"].transpose("time", "bus").to_pandas()
    assert list(frame.columns) == ["p1 0", "p1 1", "p1 2"]
    assert len(frame.index) == len(TIMES)


# --------------------------------------------------------------------------
# Fallback warning
# --------------------------------------------------------------------------


def _fallback_warning(technology, year, wind_height):
    """Reproduce the warning the script emits (kept in sync by assertion)."""
    height_note = ""
    if technology == "wind" and wind_height != GODEEEP_AGGREGATED_WIND_HEIGHT:
        height_note = (
            f" Hub-height inconsistency: godeeep_wind_height is "
            f"'{wind_height}' but the aggregated archive was only "
            f"published at '{GODEEEP_AGGREGATED_WIND_HEIGHT}', so this "
            "year's wind profile is a 100 m profile."
        )
    return (
        f"weather year {year}: using pre-aggregated GODEEEP profiles; "
        "NREL land-access exclusions do not apply to this year's "
        "profile aggregation. These archives were rolled up once, on "
        "the county-based substation tessellation they were published "
        "with — the run's own regions cannot re-cut them, only "
        "re-aggregate them." + height_note
    )


def _script_source():
    script = os.path.join(os.path.dirname(__file__), "..", "build_renewable_profiles.py")
    with open(script) as fh:
        return fh.read()


def test_warning_text_is_present_in_the_script_source():
    """Pin the exact wording the deliverable requires."""
    source = _script_source()
    assert 'f"weather year {year}: using pre-aggregated GODEEEP profiles; "' in source
    assert '"NREL land-access exclusions do not apply to this year\'s "' in source
    assert '"profile aggregation. These archives were rolled up once, on "' in source


def test_warning_names_the_year_and_all_three_caveats():
    msg = _fallback_warning("solar", 2019, "")
    assert "weather year 2019" in msg
    assert "NREL land-access exclusions do not apply" in msg
    assert "county-based substation tessellation" in msg
    assert "Hub-height" not in msg  # solar has no hub height


def test_provenance_attrs_are_stamped_on_the_output():
    """The output .nc must record the land-access treatment and hub height."""
    source = _script_source()
    assert 'profile_provenance["land_access"] = "none (unscreened county-aggregated fallback)"' in source
    assert 'ds.attrs.update({"renewable_dataset": dataset, **profile_provenance})' in source
    # The compressed path stamps the real access scenario, not the fallback text.
    assert 'profile_provenance["land_access"] = access' in source


def test_warning_flags_the_125m_vs_100m_hub_height_mismatch():
    msg = _fallback_warning("wind", 2019, "_125m")
    assert "Hub-height inconsistency" in msg
    assert "_125m" in msg and "_100m" in msg


def test_no_hub_height_note_when_config_already_says_100m():
    msg = _fallback_warning("wind", 2019, "_100m")
    assert "Hub-height" not in msg


def test_remap_logs_footprint_and_zero_weight_counts(caplog):
    with caplog.at_level(logging.INFO, logger="build_renewable_profiles"):
        remap_aggregated_profile(_archive(VALUES), BUSMAP, weights=WEIGHTS, tech="solar")
    text = caplog.text
    assert "4/5 archive substations fall inside the run footprint" in text
    assert "1 cluster buses have zero NREL caps weight" in text
