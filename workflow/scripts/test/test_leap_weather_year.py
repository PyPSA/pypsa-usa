"""A leap weather year is laid onto the 365-day model calendar the same way by every input.

Every weather-driven input must read the same weather hour at the same snapshot,
or demand and capacity factors drift a day apart from 1 March on (issue #919).
The rule is :func:`_helpers.weather_year_hours`: by position, so weather-year
February 29 takes the 1 March label and weather-year 31 December is not used.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from _helpers import get_snapshots, select_weather_hours, weather_year_hours
from build_demand import ReadEer
from build_renewable_profiles import get_renewable_snapshots
from nrel_exclusion.aggregate_godeeep_weighted import fix_godeeep_time
from test_build_demand_eer import _write_eer_fixture

pytestmark = pytest.mark.fast

WEATHER_YEAR = 2012
JAN_1 = pd.Timestamp(f"{WEATHER_YEAR}-01-01")
MODEL_SNS = get_snapshots(
    {"start": f"{WEATHER_YEAR}-01-01", "end": f"{WEATHER_YEAR + 1}-01-01", "inclusive": "left"},
)
# Feb 28, Mar 1 and Dec 31 sit on either side of the leap day and at the end of
# the year; the first hours of Jan 1 are left out because the EER CST->UTC roll
# wraps the end of the strip onto them.
PROBES = pd.DatetimeIndex(
    ["2012-01-01 12:00", "2012-02-28 23:00", "2012-03-01 00:00", "2012-07-04 15:00", "2012-12-31 23:00"],
)


def _hour_of_year_axis(time: pd.DatetimeIndex) -> np.ndarray:
    """Value of each stamp = hours since 1 January, on the true (leap) calendar."""
    return ((time - JAN_1) / pd.Timedelta(hours=1)).to_numpy()


def test_weather_year_hours_non_leap_year_is_identity():
    sns = get_snapshots({"start": "2019-01-01", "end": "2020-01-01", "inclusive": "left"})
    assert weather_year_hours(sns).equals(sns)


def test_weather_year_hours_leap_year_is_positional():
    hours = weather_year_hours(MODEL_SNS)

    assert len(hours) == len(MODEL_SNS) == 8760
    assert hours[0] == pd.Timestamp("2012-01-01 00:00")
    assert hours[-1] == pd.Timestamp("2012-12-30 23:00")
    assert ((hours.month == 2) & (hours.day == 29)).sum() == 24
    # consecutive: position i of the model year reads hour i of the weather year
    assert (_hour_of_year_axis(hours) == np.arange(8760)).all()


def test_weather_year_hours_rejects_feb_29():
    with pytest.raises(ValueError, match="February 29"):
        weather_year_hours(pd.date_range("2012-02-29", periods=3, freq="h"))


def test_select_weather_hours_relabels_onto_the_snapshots():
    time = pd.date_range(f"{WEATHER_YEAR}-01-01", f"{WEATHER_YEAR}-12-31 23:00", freq="h")
    ds = xr.Dataset({"v": ("time", _hour_of_year_axis(time))}, coords={"time": time})

    out = select_weather_hours(ds, MODEL_SNS)

    assert pd.DatetimeIndex(out.indexes["time"]).equals(MODEL_SNS)
    assert out["time"].dtype == ds["time"].dtype
    assert (out["v"].to_numpy() == np.arange(8760)).all()


def _cutout_hour(label: pd.Timestamp) -> float:
    """Atlite CF and temperature/COP: select_weather_hours on the full-year cutout."""
    time = pd.date_range(f"{WEATHER_YEAR}-01-01", f"{WEATHER_YEAR}-12-31 23:00", freq="h")
    ds = xr.Dataset({"v": ("time", _hour_of_year_axis(time))}, coords={"time": time})
    return float(select_weather_hours(ds, MODEL_SNS)["v"].sel(time=label))


def _godeeep_hour(label: pd.Timestamp) -> float:
    """GODEEEP CF: fix_godeeep_time on the raw axis, then the renewable-snapshot window."""
    # raw GODEEEP stamps are hour-ending (1 Jan 01:00 ... 1 Jan 00:00 of the
    # next year); fix_godeeep_time moves them back to hour-beginning
    start = pd.date_range(f"{WEATHER_YEAR}-01-01", f"{WEATHER_YEAR}-12-31 23:00", freq="h")
    hour_ending = start + pd.Timedelta(hours=1)
    raw = xr.Dataset({"v": ("Time", _hour_of_year_axis(start))}, coords={"Time": hour_ending.to_numpy()})
    ds = fix_godeeep_time(raw, WEATHER_YEAR)
    ds = ds.sel(time=get_renewable_snapshots({}, WEATHER_YEAR))
    return float(ds["v"].sel(time=label))


def _eer_hour(label: pd.Timestamp, tmp_path) -> float:
    """EER demand: the 8760-row CST strip, rolled to UTC, on the model calendar."""
    model_year = 2030
    path = tmp_path / "eer.h5"
    _write_eer_fixture(path, model_year=model_year)
    demand = ReadEer(str(path), planning_horizons=[model_year], renewable_weather_years=[WEATHER_YEAR]).read_demand()

    weather_start = ReadEer.WEATHER_YEARS.index(WEATHER_YEAR) * ReadEer.HOURS_PER_YEAR
    row = demand.loc[(label.replace(year=model_year), "all", "all", "electricity"), "Alabama"]
    # the fixture holds the strip position; strip position p is local CST hour p
    # of the weather year (with Feb 29), i.e. UTC hour p + 6
    return float(row - weather_start + ReadEer.CST_TO_UTC_SHIFT)


@pytest.mark.parametrize("label", PROBES)
def test_every_input_reads_the_same_weather_hour(label, tmp_path):
    expected = _hour_of_year_axis(weather_year_hours(pd.DatetimeIndex([label])))[0]

    assert _cutout_hour(label) == expected
    assert _godeeep_hour(label) == expected
    assert _eer_hour(label, tmp_path) == expected


def test_leap_day_weather_is_on_march_1():
    """The case of #919: atlite used to read Mar 1 weather at the Mar 1 label, GODEEEP and EER Feb 29 weather."""
    feb_29 = (pd.Timestamp("2012-02-29 00:00") - JAN_1) / pd.Timedelta(hours=1)
    assert _cutout_hour(pd.Timestamp("2012-03-01 00:00")) == feb_29
    assert _godeeep_hour(pd.Timestamp("2012-03-01 00:00")) == feb_29
