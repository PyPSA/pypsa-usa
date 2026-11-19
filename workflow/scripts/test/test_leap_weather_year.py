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
from _helpers import (
    WEATHER_CALENDAR_ATTR,
    check_weather_calendar,
    get_snapshots,
    label_with_snapshots,
    lay_on_model_calendar,
    model_calendar_labels,
    select_weather_hours,
    stamp_weather_calendar,
    weather_year_hours,
)
from atlite.pv.solar_position import SolarPosition
from build_cop_profiles import cop_profile
from build_demand import ReadEer, ReadEia
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


def test_model_calendar_labels_is_the_inverse_of_weather_year_hours():
    weather = pd.date_range(f"{WEATHER_YEAR}-01-01", f"{WEATHER_YEAR}-12-31 23:00", freq="h")

    labels = model_calendar_labels(weather)

    # 31 December of a leap year is read by no snapshot
    assert labels.isna().sum() == 24
    assert labels[labels.isna()].equals(labels[weather.dayofyear == 366])
    assert labels.dropna().equals(MODEL_SNS)
    assert weather_year_hours(labels.dropna()).equals(weather[labels.notna()])


def test_model_calendar_labels_non_leap_year_is_identity():
    time = pd.date_range("2019-01-01", "2019-12-31 23:00", freq="h")
    assert model_calendar_labels(time).equals(time)


def test_lay_on_model_calendar_keeps_the_index_name_and_repeated_stamps():
    # FERC 714 arrives with one row per state and hour
    time = pd.DatetimeIndex(["2012-02-28 05:00", "2012-02-29 05:00", "2012-02-29 05:00", "2012-12-31 05:00"])
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]}, index=time.rename("snapshot"))

    out = lay_on_model_calendar(df)

    assert out.index.name == "snapshot"
    assert out.index.tolist() == [pd.Timestamp("2012-02-28 05:00")] + [pd.Timestamp("2012-03-01 05:00")] * 2
    assert out["v"].tolist() == [1.0, 2.0, 3.0]


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


# balancing areas ReadEia folds into the regions of the GIS shapes
_EIA_FOLDED_BAS = [
    *["SRP", "AZPS", "TEPC"],
    *["CPLE", "CPLW", "DUK", "SC", "SCEG", "YAD"],
    *["FPC", "FPL", "GVL", "JEA", "NSB", "SEC", "TAL", "TEC", "HST", "FMPP"],
    "SWPP",
]


def _eia_hour(label: pd.Timestamp, tmp_path) -> float:
    """EIA historical demand: the real calendar of the weather year, read by ReadEia."""
    time = pd.date_range(f"{WEATHER_YEAR}-01-01", f"{WEATHER_YEAR}-12-31 23:00", freq="h")
    raw = pd.DataFrame({"CISO": _hour_of_year_axis(time)}, index=time.rename("timestamp"))
    raw[_EIA_FOLDED_BAS] = 0.0
    path = tmp_path / "eia_demand.csv"
    raw.to_csv(path)

    demand = ReadEia(str(path)).read_demand()
    return float(demand.loc[(label, "all", "all", "electricity"), "CISO"])


def _hydro_hour(label: pd.Timestamp) -> float:
    """Breakthrough hydro: a leap-year csv on its real calendar, as add_electricity lays it on."""
    time = pd.date_range(f"{WEATHER_YEAR}-01-01", f"{WEATHER_YEAR}-12-31 23:00", freq="h")
    # read_csv(index_col=0) leaves the stamps as strings
    profile = pd.DataFrame({"plant": _hour_of_year_axis(time)}, index=time.astype(str))
    return float(lay_on_model_calendar(profile).at[label, "plant"])


@pytest.mark.parametrize("label", PROBES)
def test_every_input_reads_the_same_weather_hour(label, tmp_path):
    expected = _hour_of_year_axis(weather_year_hours(pd.DatetimeIndex([label])))[0]

    assert _cutout_hour(label) == expected
    assert _godeeep_hour(label) == expected
    assert _eer_hour(label, tmp_path) == expected
    assert _eia_hour(label, tmp_path) == expected
    assert _hydro_hour(label) == expected


def test_historical_demand_carries_no_feb_29_and_ends_on_dec_31(tmp_path):
    """Historical demand on a leap year covers the model snapshots, hour for hour."""
    time = pd.date_range(f"{WEATHER_YEAR}-01-01", f"{WEATHER_YEAR}-12-31 23:00", freq="h")
    raw = pd.DataFrame({"CISO": _hour_of_year_axis(time)}, index=time.rename("timestamp"))
    raw[_EIA_FOLDED_BAS] = 0.0
    path = tmp_path / "eia_demand.csv"
    raw.to_csv(path)

    demand = ReadEia(str(path)).read_demand()

    assert pd.DatetimeIndex(demand.index.get_level_values("snapshot")).equals(MODEL_SNS)
    assert (demand["CISO"].to_numpy() == np.arange(8760)).all()


def test_godeeep_hour_beginning_axis_is_not_shifted():
    """An axis that already holds 1 January 00:00 is hour-beginning and keeps its stamps."""
    start = pd.date_range("2019-01-01", "2019-12-31 23:00", freq="h")
    raw = xr.Dataset({"v": ("Time", np.arange(len(start)))}, coords={"Time": start.to_numpy()})

    fixed = fix_godeeep_time(raw, 2019)

    assert pd.DatetimeIndex(fixed.indexes["time"]).equals(start)


def test_godeeep_hour_ending_axis_moves_back_an_hour():
    start = pd.date_range("2019-01-01", "2019-12-31 23:00", freq="h")
    raw = xr.Dataset(
        {"v": ("Time", np.arange(len(start)))},
        coords={"Time": (start + pd.Timedelta(hours=1)).to_numpy()},
    )

    fixed = fix_godeeep_time(raw, 2019)

    assert pd.DatetimeIndex(fixed.indexes["time"]).equals(start)


def _point_cutout(time: pd.DatetimeIndex) -> xr.Dataset:
    """One grid cell without stored sun position, as an older cutout has."""
    ds = xr.Dataset(
        {"influx_direct": (("time", "y", "x"), np.ones((len(time), 1, 1)))},
        coords={"time": time, "y": [35.0], "x": [-120.0]},
    )
    return ds.assign_coords(lon=ds["x"], lat=ds["y"]).chunk(time=len(time))


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_sun_position_follows_the_weather_hour_not_the_label():
    """Atlite derives the sun position from the time axis, so it is computed before relabelling."""
    time = pd.date_range(f"{WEATHER_YEAR}-01-01", f"{WEATHER_YEAR}-12-31 23:00", freq="h")
    cutout = _point_cutout(time)
    # build_renewable_profiles: select on the weather-year calendar, convert, relabel
    weather = cutout.sel(time=weather_year_hours(MODEL_SNS))
    altitude = label_with_snapshots(SolarPosition(weather)["altitude"], MODEL_SNS)

    true_altitude = SolarPosition(cutout)["altitude"]
    label, weather_hour = pd.Timestamp("2012-07-04 20:00"), pd.Timestamp("2012-07-03 20:00")
    assert float(altitude.sel(time=label)) == float(true_altitude.sel(time=weather_hour))
    # relabelling first pairs the 3 July irradiance with the 4 July sun
    relabelled_first = SolarPosition(select_weather_hours(cutout, MODEL_SNS))["altitude"]
    assert float(relabelled_first.sel(time=label)) == float(true_altitude.sel(time=label))
    assert float(relabelled_first.sel(time=label)) != float(altitude.sel(time=label))


def _profile(time: pd.DatetimeIndex) -> xr.DataArray:
    return xr.DataArray(np.ones((len(time), 1)), coords={"time": time, "bus": ["b1"]}, dims=["time", "bus"])


def test_unstamped_leap_year_profile_is_rejected():
    """A profile built before the positional rule is one day off from 1 March on."""
    # a window without 29 February: the time axis alone cannot show the old rule
    june_2020 = get_snapshots({"start": "2020-06-01", "end": "2020-06-08", "inclusive": "left"})

    with pytest.raises(ValueError, match="rebuild"):
        check_weather_calendar(_profile(june_2020), "profile_solar.nc")


def test_stamped_and_non_leap_profiles_pass():
    june_2020 = get_snapshots({"start": "2020-06-01", "end": "2020-06-08", "inclusive": "left"})
    june_2019 = get_snapshots({"start": "2019-06-01", "end": "2019-06-08", "inclusive": "left"})

    check_weather_calendar(stamp_weather_calendar(_profile(june_2020)), "profile_solar.nc")
    # both rules give the same result for a non-leap year
    check_weather_calendar(_profile(june_2019), "profile_solar.nc")


def test_stamp_survives_from_temperature_to_cop(tmp_path):
    """build_temperature_profiles stamps, build_cop_profiles carries it over, build_heat checks it."""
    june_2020 = get_snapshots({"start": "2020-06-01", "end": "2020-06-08", "inclusive": "left"})
    temp_path, cop_path = tmp_path / "temp_air.nc", tmp_path / "cop_air.nc"

    stamp_weather_calendar(_profile(june_2020) * 10.0).to_netcdf(temp_path)
    cop_profile(xr.open_dataarray(temp_path), sink_t=55.0, source="air").to_netcdf(cop_path)

    cop = xr.open_dataarray(cop_path)
    assert WEATHER_CALENDAR_ATTR in cop.attrs
    check_weather_calendar(cop, str(cop_path))


def test_cop_from_unstamped_temperatures_stays_unstamped():
    """A stale temperature profile must not come out of build_cop_profiles looking current."""
    june_2020 = get_snapshots({"start": "2020-06-01", "end": "2020-06-08", "inclusive": "left"})

    cop = cop_profile(_profile(june_2020) * 10.0, sink_t=55.0, source="air")

    with pytest.raises(ValueError, match="rebuild"):
        check_weather_calendar(cop, "cop_air.nc")


def test_leap_day_weather_is_on_march_1():
    """The case of #919: atlite used to read Mar 1 weather at the Mar 1 label, GODEEEP and EER Feb 29 weather."""
    feb_29 = (pd.Timestamp("2012-02-29 00:00") - JAN_1) / pd.Timedelta(hours=1)
    assert _cutout_hour(pd.Timestamp("2012-03-01 00:00")) == feb_29
    assert _godeeep_hour(pd.Timestamp("2012-03-01 00:00")) == feb_29
