"""Unit tests for the config-declared GODEEEP CF registry (godeeep_cf_registry.py).

The registry replaces the inline record-key/filename construction in
build_renewable_profiles.py and the hardcoded record table in
zenodo_downloader.py, which together produced issue #803: a config asking for
``godeeep_wind_height: "_100m"`` against Zenodo-only sources fell through
``ZenodoScenarioDownloader.download_scenario_file`` -> ``None`` and only blew up
much later as a ``TypeError`` on a ``None`` path.

These tests pin the invariants that make that impossible:
  * published file names and dataset keys are locked to the strings on disk /
    on Zenodo, so a refactor cannot rename them silently,
  * there is no ``"_100m"`` hub-height default and no ``"_80m"`` dataset,
  * sources are tried in configured order — first hit wins, no fallback after,
  * an unavailable (dataset, year) raises CfNotAvailableError whose message
    names the dataset key, the requested year and the available years,
  * validate_godeeep_cf_config aggregates every problem into one parse-time error.

Everything is built from inline dicts: no filesystem, no network.
"""

import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from godeeep_cf_registry import (
    CfNotAvailableError,
    cf_filename,
    dataset_key,
    godeeep_tech_spec,
    load_sources,
    parse_years,
    resolve_cf,
    resolve_weather_year,
    validate_godeeep_cf_config,
)

pytestmark = pytest.mark.fast

OAK_ROOT = "/oak/stanford/groups/iazevedo/GoDEEEP_Capacity_Factors_compressed"

LOCAL_SOURCE = {
    "kind": "local",
    "root": OAK_ROOT,
    "layout": "{scenario}/{tech_dir}/{filename}",
    "datasets": {
        "solar_historical_compressed": {"years": "1980-2022"},
        "wind_100m_historical_compressed": {"years": "1980-2022"},
        "wind_125m_historical_compressed": {"years": "1980-2022"},
    },
}

ZENODO_SOURCE = {
    "kind": "zenodo",
    "datasets": {
        "solar_historical_compressed": {"record": 20127513, "years": [2012]},
        "wind_125m_historical_compressed": {"record": 20127520, "years": [2012]},
        "solar_rcp85cooler_compressed": {"record": 20127633, "years": [2030, 2040, 2050]},
        "wind_125m_rcp85cooler_compressed": {"record": 20127645, "years": [2030, 2040, 2050]},
    },
}


def make_config(sources, **overrides):
    """A minimal godeeep config; ``overrides`` replace/add top-level keys."""
    config = {
        "renewable": {"dataset": "godeeep", "solar": {}, "onwind": {}},
        "renewable_scenarios": ["historical"],
        "renewable_weather_years": [2012],
        "godeeep_wind_height": "_125m",
        "scenario": {"planning_horizons": [2030]},
        "godeeep_cf_registry": {"copy_local": False, "sources": sources},
    }
    config.update(overrides)
    return config


# --------------------------------------------------------------------------
# parse_years
# --------------------------------------------------------------------------


def test_parse_years_inclusive_range_string():
    years = parse_years("1980-2022")
    assert years[0] == 1980
    assert years[-1] == 2022
    # 1980..2022 inclusive is the 43 files actually present on the Oak mirror.
    assert len(years) == 43


def test_parse_years_explicit_list():
    assert parse_years([2030, 2040, 2050]) == [2030, 2040, 2050]
    assert parse_years([2012]) == [2012]
    assert parse_years([]) == []
    # Deduped and sorted so availability checks are order-independent.
    assert parse_years([2040, 2030, 2040]) == [2030, 2040]


def test_parse_years_accepts_numeric_strings():
    assert parse_years(["2030", 2040]) == [2030, 2040]


@pytest.mark.parametrize(
    "garbage",
    [
        "1980_2022",  # wrong separator
        "1980-2022-2030",  # not a two-endpoint range
        "1980 to 2022",
        "2022-1980",  # reversed endpoints would silently yield an empty set
        "historical",
        "",
        None,
        True,
        {"start": 1980},
        [1980, "twenty-twenty"],
        [None],
    ],
)
def test_parse_years_rejects_garbage(garbage):
    with pytest.raises(ValueError):
        parse_years(garbage)


# --------------------------------------------------------------------------
# dataset_key / cf_filename — locked to the published strings
# --------------------------------------------------------------------------


def test_dataset_key_matches_legacy_record_keys():
    # Exactly the keys build_renewable_profiles.py built inline and
    # zenodo_downloader.py's scenario_records table used.
    assert dataset_key("solar", "", "historical") == "solar_historical_compressed"
    assert dataset_key("wind", "_125m", "historical") == "wind_125m_historical_compressed"
    assert dataset_key("wind", "_100m", "historical") == "wind_100m_historical_compressed"
    assert dataset_key("wind", "_125m", "rcp85cooler") == "wind_125m_rcp85cooler_compressed"
    assert dataset_key("solar", "", "rcp45hotter") == "solar_rcp45hotter_compressed"


def test_cf_filename_locks_published_names():
    assert cf_filename("solar", "", "historical", 2019) == "solar_gen_cf_2019_compressed.nc"
    assert cf_filename("wind", "_125m", "historical", 2019) == "wind_gen_cf_2019_125m_compressed.nc"
    assert cf_filename("wind", "_100m", "historical", 2019) == "wind_gen_cf_2019_100m_compressed.nc"


def test_cf_filename_is_scenario_independent():
    # The scenario lives in the containing record/directory, never in the name —
    # the Zenodo future records publish e.g. solar_gen_cf_2030_compressed.nc.
    assert cf_filename("solar", "", "rcp85cooler", 2030) == "solar_gen_cf_2030_compressed.nc"
    assert cf_filename("wind", "_125m", "rcp45cooler", 2050) == "wind_gen_cf_2050_125m_compressed.nc"
    for scenario in ("historical", "rcp45hotter", "rcp85cooler"):
        assert cf_filename("solar", "", scenario, 2030) == "solar_gen_cf_2030_compressed.nc"


# --------------------------------------------------------------------------
# godeeep_tech_spec — no hub-height default
# --------------------------------------------------------------------------


def test_godeeep_tech_spec_solar_ignores_wind_height():
    spec = godeeep_tech_spec("solar", {"godeeep_wind_height": "_125m"})
    assert (spec.technology, spec.wind_height) == ("solar", "")
    assert spec.tech_dir == "solar"
    # Still tuple-unpackable like the code it replaces.
    technology, wind_height = spec
    assert (technology, wind_height) == ("solar", "")


@pytest.mark.parametrize("tech", ["onwind", "offwind", "offwind_floating"])
def test_godeeep_tech_spec_wind_uses_configured_height(tech):
    spec = godeeep_tech_spec(tech, {"godeeep_wind_height": "_100m"})
    assert (spec.technology, spec.wind_height) == ("wind", "_100m")
    assert spec.tech_dir == "wind_100m"


def test_godeeep_tech_spec_raises_when_wind_height_missing():
    # The old code silently defaulted to "_100m", which Zenodo does not publish.
    with pytest.raises(ValueError) as excinfo:
        godeeep_tech_spec("onwind", {})
    message = str(excinfo.value)
    assert "godeeep_wind_height" in message
    assert "_100m" in message and "_125m" in message


def test_godeeep_tech_spec_rejects_80m():
    with pytest.raises(ValueError) as excinfo:
        godeeep_tech_spec("onwind", {"godeeep_wind_height": "_80m"})
    message = str(excinfo.value)
    assert "_80m" in message
    assert "_100m" in message and "_125m" in message


def test_godeeep_tech_spec_rejects_unknown_technology():
    with pytest.raises(ValueError) as excinfo:
        godeeep_tech_spec("hydro", {"godeeep_wind_height": "_125m"})
    assert "hydro" in str(excinfo.value)


# --------------------------------------------------------------------------
# resolve_weather_year / resolve_scenario
# --------------------------------------------------------------------------


def test_resolve_weather_year_historical_reads_weather_years():
    config = make_config([LOCAL_SOURCE], renewable_weather_years=[2019, 2020])
    assert resolve_weather_year(config, planning_horizon=2030) == 2019


def test_resolve_weather_year_future_reads_planning_horizon():
    config = make_config([ZENODO_SOURCE], renewable_scenarios=["rcp85cooler"])
    assert resolve_weather_year(config, planning_horizon=2040) == 2040


def test_resolve_weather_year_reports_missing_renewable_scenarios():
    config = make_config([LOCAL_SOURCE])
    del config["renewable_scenarios"]
    with pytest.raises(ValueError) as excinfo:
        resolve_weather_year(config, planning_horizon=2019)
    assert "renewable_scenarios" in str(excinfo.value)


# --------------------------------------------------------------------------
# resolve_cf — source precedence, no fallback
# --------------------------------------------------------------------------


def test_resolve_cf_local_source_wins_over_zenodo_for_overlapping_year():
    # 2012 is offered by BOTH sources; the local source is listed first.
    config = make_config([LOCAL_SOURCE, ZENODO_SOURCE], renewable_weather_years=[2012])
    resolution = resolve_cf(config, "onwind")
    assert resolution.kind == "local"
    assert resolution.source_index == 0
    assert resolution.dataset_key == "wind_125m_historical_compressed"
    assert resolution.year == 2012
    assert resolution.filename == "wind_gen_cf_2012_125m_compressed.nc"
    assert resolution.path == f"{OAK_ROOT}/historical/wind_125m/wind_gen_cf_2012_125m_compressed.nc"
    assert resolution.record_id is None


def test_resolve_cf_order_is_config_order_not_kind_priority():
    # Same two sources, zenodo declared first -> zenodo wins for 2012.
    config = make_config([ZENODO_SOURCE, LOCAL_SOURCE], renewable_weather_years=[2012])
    resolution = resolve_cf(config, "onwind")
    assert resolution.kind == "zenodo"
    assert resolution.source_index == 0
    assert resolution.record_id == "20127520"
    assert resolution.path is None
    assert resolution.location == "zenodo:20127520"


def test_resolve_cf_falls_through_to_zenodo_only_when_local_lacks_the_dataset():
    # Future scenario: local mirror holds historical only.
    config = make_config(
        [LOCAL_SOURCE, ZENODO_SOURCE],
        renewable_scenarios=["rcp85cooler"],
    )
    resolution = resolve_cf(config, "solar", planning_horizon=2040)
    assert resolution.kind == "zenodo"
    assert resolution.record_id == "20127633"
    assert resolution.dataset_key == "solar_rcp85cooler_compressed"
    assert resolution.filename == "solar_gen_cf_2040_compressed.nc"


def test_resolve_cf_local_path_uses_configured_layout():
    config = make_config(
        [{**LOCAL_SOURCE, "root": "/mnt/cf", "layout": "{scenario}/{dataset_key}/{year}/{filename}"}],
        renewable_weather_years=[1995],
    )
    resolution = resolve_cf(config, "solar")
    assert resolution.path == ("/mnt/cf/historical/solar_historical_compressed/1995/solar_gen_cf_1995_compressed.nc")


def test_resolve_cf_copy_local_defaults_from_registry_block():
    config = make_config([LOCAL_SOURCE], renewable_weather_years=[2012])
    config["godeeep_cf_registry"]["copy_local"] = True
    assert resolve_cf(config, "solar").copy_local is True


def test_resolve_cf_raises_with_dataset_key_year_and_available_years():
    # 1999 exists on Oak but not on Zenodo; a Zenodo-only registry must refuse it.
    config = make_config([ZENODO_SOURCE], renewable_weather_years=[1999])
    with pytest.raises(CfNotAvailableError) as excinfo:
        resolve_cf(config, "solar")
    message = str(excinfo.value)
    assert "solar_historical_compressed" in message  # dataset key
    assert "1999" in message  # requested year
    assert "2012" in message  # available years
    assert "solar_gen_cf_1999_compressed.nc" in message


def test_resolve_cf_never_substitutes_a_neighbouring_year():
    config = make_config([LOCAL_SOURCE], renewable_weather_years=[2023])
    with pytest.raises(CfNotAvailableError) as excinfo:
        resolve_cf(config, "solar")
    # 2022 is available and adjacent — it must be reported, not silently used.
    assert "1980-2022" in str(excinfo.value)


def test_resolve_cf_reports_unpublished_dataset_with_empty_year_list():
    # Mirrors wind_125m_rcp45hotter: a record id exists in config but Zenodo
    # returns 404, so it is declared with no years.
    source = {
        "kind": "zenodo",
        "datasets": {"wind_125m_rcp45hotter_compressed": {"record": 20127545, "years": []}},
    }
    config = make_config([source], renewable_scenarios=["rcp45hotter"])
    with pytest.raises(CfNotAvailableError) as excinfo:
        resolve_cf(config, "onwind", planning_horizon=2030)
    message = str(excinfo.value)
    assert "wind_125m_rcp45hotter_compressed" in message
    assert "2030" in message
    # Distinguished from "not declared at all" so the operator knows the record
    # id is configured but the record itself publishes nothing.
    assert "no years published" in message


def test_resolve_cf_803_regression_100m_against_zenodo_only_sources():
    """Issue #803: this combination used to yield ``None``, then a TypeError."""
    config = make_config(
        [ZENODO_SOURCE],
        godeeep_wind_height="_100m",
        renewable_weather_years=[2012],
    )
    with pytest.raises(CfNotAvailableError) as excinfo:
        resolve_cf(config, "onwind")
    message = str(excinfo.value)
    assert "wind_100m_historical_compressed" in message
    assert "2012" in message
    assert "dataset key not declared" in message
    # And the same request against the Oak mirror resolves cleanly.
    ok = resolve_cf(
        make_config([LOCAL_SOURCE], godeeep_wind_height="_100m", renewable_weather_years=[2012]),
        "onwind",
    )
    assert ok.path == f"{OAK_ROOT}/historical/wind_100m/wind_gen_cf_2012_100m_compressed.nc"


# --------------------------------------------------------------------------
# load_sources — malformed registries
# --------------------------------------------------------------------------


def test_load_sources_preserves_declared_order():
    sources = load_sources(make_config([LOCAL_SOURCE, ZENODO_SOURCE]))
    assert [source.kind for source in sources] == ["local", "zenodo"]
    assert sources[0].years_for("solar_historical_compressed")[0] == 1980


def test_load_sources_requires_the_registry_block():
    config = make_config([LOCAL_SOURCE])
    del config["godeeep_cf_registry"]
    with pytest.raises(ValueError) as excinfo:
        load_sources(config)
    assert "godeeep_cf_registry" in str(excinfo.value)


@pytest.mark.parametrize(
    "bad_source",
    [
        {"kind": "s3", "datasets": {}},  # unknown kind
        {"kind": "local", "datasets": {}},  # local without a root
        {"kind": "local", "root": "/tmp", "datasets": {"k": "nineteen-eighty"}},  # bad years
        {"kind": "zenodo", "datasets": {"k": {"years": [2012]}}},  # zenodo without a record
        {"kind": "zenodo", "datasets": [1, 2, 3]},  # datasets not a mapping
    ],
)
def test_load_sources_rejects_malformed_sources(bad_source):
    with pytest.raises(ValueError):
        load_sources(make_config([bad_source]))


# --------------------------------------------------------------------------
# validate_godeeep_cf_config — parse-time gate
# --------------------------------------------------------------------------


def test_validate_passes_for_a_resolvable_config():
    config = make_config([LOCAL_SOURCE, ZENODO_SOURCE], renewable_weather_years=[2019])
    resolved = validate_godeeep_cf_config(config)
    assert {resolution.dataset_key for resolution in resolved} == {
        "solar_historical_compressed",
        "wind_125m_historical_compressed",
    }
    assert all(resolution.kind == "local" for resolution in resolved)


def test_validate_fails_loudly_for_1999_against_a_zenodo_only_registry():
    config = make_config([ZENODO_SOURCE], renewable_weather_years=[1999])
    with pytest.raises(CfNotAvailableError) as excinfo:
        validate_godeeep_cf_config(config)
    message = str(excinfo.value)
    assert "1999" in message
    assert "solar_historical_compressed" in message
    assert "wind_125m_historical_compressed" in message
    assert "2012" in message


def test_validate_reports_missing_renewable_scenarios_instead_of_keyerror():
    # build_electricity.smk / build_renewable_profiles.py did a bare
    # config["renewable_scenarios"][0]; the KeyError named nothing useful.
    config = make_config([LOCAL_SOURCE])
    del config["renewable_scenarios"]
    with pytest.raises(CfNotAvailableError) as excinfo:
        validate_godeeep_cf_config(config)
    assert "renewable_scenarios" in str(excinfo.value)


def test_validate_reports_missing_weather_years():
    config = make_config([LOCAL_SOURCE])
    del config["renewable_weather_years"]
    with pytest.raises(CfNotAvailableError) as excinfo:
        validate_godeeep_cf_config(config)
    assert "renewable_weather_years" in str(excinfo.value)


def test_validate_aggregates_every_problem_into_one_error():
    config = make_config(
        [ZENODO_SOURCE],
        godeeep_wind_height="_80m",
        renewable_weather_years=[1999],
    )
    del config["renewable_scenarios"]
    with pytest.raises(CfNotAvailableError) as excinfo:
        validate_godeeep_cf_config(config)
    message = str(excinfo.value)
    assert "_80m" in message
    assert "renewable_scenarios" in message
    assert "problem(s)" in message
    # Numbered bullets: more than one failure survived into the single report.
    assert "(1)" in message and "(2)" in message


def test_validate_803_regression_100m_against_zenodo_only_registry():
    config = make_config(
        [ZENODO_SOURCE],
        godeeep_wind_height="_100m",
        renewable_weather_years=[2012],
    )
    with pytest.raises(CfNotAvailableError) as excinfo:
        validate_godeeep_cf_config(config)
    message = str(excinfo.value)
    assert "wind_100m_historical_compressed" in message
    assert "2012" in message


def test_validate_checks_every_planning_horizon_of_a_future_scenario():
    config = make_config(
        [ZENODO_SOURCE],
        renewable_scenarios=["rcp85cooler"],
        scenario={"planning_horizons": [2030, 2040, 2060]},
    )
    with pytest.raises(CfNotAvailableError) as excinfo:
        validate_godeeep_cf_config(config)
    message = str(excinfo.value)
    assert "2060" in message
    # 2030/2040 resolve, so they must not be reported as problems.
    assert message.count("2060") >= 1
    assert "problem(s)" in message


def test_validate_skips_non_godeeep_datasets():
    config = make_config([ZENODO_SOURCE], renewable_weather_years=[1999])
    config["renewable"] = {"dataset": "atlite"}
    assert validate_godeeep_cf_config(config) == []
