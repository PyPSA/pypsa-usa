"""Tests for EER demand profile loading."""

import numpy as np
import pandas as pd
import pytest
import tables
from _helpers import get_multiindex_snapshots
from build_demand import ReadEer


def _write_eer_fixture(path, model_year=2030):
    hours = ReadEer.HOURS_PER_YEAR
    total_hours = hours * len(ReadEer.WEATHER_YEARS)

    with tables.open_file(path, "w") as h5:
        group = h5.create_group("/", str(model_year))
        h5.create_array(
            group,
            "columns",
            np.array([b"datetime", b"AL", b"CA"]),
        )
        h5.create_carray(
            group,
            "datetime",
            tables.StringAtom(itemsize=30),
            shape=(total_hours,),
        )
        for state, offset in {"AL": 0, "CA": 1000000}.items():
            values = np.arange(offset, offset + total_hours, dtype=np.int32)
            node = h5.create_carray(
                group,
                state,
                tables.Int32Atom(),
                shape=(total_hours,),
            )
            node[:] = values


def test_read_eer_selects_model_and_weather_year_with_cst_shift(tmp_path):
    path = tmp_path / "eer.h5"
    _write_eer_fixture(path, model_year=2030)

    reader = ReadEer(
        str(path),
        planning_horizons=[2030],
        renewable_weather_years=[2010],
    )
    demand = reader.read_demand()

    weather_start = ReadEer.WEATHER_YEARS.index(2010) * ReadEer.HOURS_PER_YEAR
    first_snapshot = pd.Timestamp("2030-01-01 00:00:00")
    sixth_snapshot = pd.Timestamp("2030-01-01 06:00:00")

    assert len(demand) == ReadEer.HOURS_PER_YEAR
    assert demand.index.names == ["snapshot", "sector", "subsector", "fuel"]
    assert "Alabama" in demand.columns
    assert "California" in demand.columns
    assert demand.loc[(first_snapshot, "all", "all", "electricity"), "Alabama"] == (
        weather_start + ReadEer.HOURS_PER_YEAR - ReadEer.CST_TO_UTC_SHIFT
    )
    assert demand.loc[(sixth_snapshot, "all", "all", "electricity"), "Alabama"] == weather_start


def test_read_eer_leap_planning_horizon_matches_network_snapshots(tmp_path):
    """A leap investment period lands on the network's own calendar.

    ``get_snapshots`` drops February 29, so the 2040 period runs through
    December 31. The EER strip has to cover exactly those hours.
    """
    path = tmp_path / "eer.h5"
    _write_eer_fixture(path, model_year=2040)

    reader = ReadEer(
        str(path),
        planning_horizons=[2040],
        renewable_weather_years=[2010],
    )
    demand = reader.read_demand()

    network_sns = get_multiindex_snapshots(
        {"start": "2019-01-01", "end": "2020-01-01", "inclusive": "left"},
        [2040],
    ).get_level_values("timestep")
    demand_sns = pd.DatetimeIndex(demand.index.get_level_values("snapshot"))

    assert len(demand_sns) == ReadEer.HOURS_PER_YEAR
    assert not ((demand_sns.month == 2) & (demand_sns.day == 29)).any()
    assert demand_sns.equals(pd.DatetimeIndex(network_sns))

    # values stay positional: the first UTC hour of March 1 is hour 59*24 of
    # the CST-shifted weather-year strip
    weather_start = ReadEer.WEATHER_YEARS.index(2010) * ReadEer.HOURS_PER_YEAR
    march_first = pd.Timestamp("2040-03-01 06:00:00")
    assert demand.loc[(march_first, "all", "all", "electricity"), "Alabama"] == weather_start + 59 * 24


def test_read_eer_rejects_invalid_weather_year(tmp_path):
    with pytest.raises(ValueError, match="supports weather years"):
        ReadEer(str(tmp_path / "eer.h5"), planning_horizons=[2030], renewable_weather_years=[2014])


def test_read_eer_rejects_multiple_weather_years(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        ReadEer(str(tmp_path / "eer.h5"), planning_horizons=[2030], renewable_weather_years=[2010, 2011])


def test_read_eer_rejects_unsupported_planning_horizon(tmp_path):
    with pytest.raises(ValueError, match="unsupported year"):
        ReadEer(str(tmp_path / "eer.h5"), planning_horizons=[2032], renewable_weather_years=[2010])
