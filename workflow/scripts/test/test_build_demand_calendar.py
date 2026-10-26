"""Activity-driven demand profiles must follow the weather year's weekdays.

Renewable and heat-pump profiles are dated by the weather year and moved to
each planning horizon by month and day, so the weekends that line up with
them are the weather year's. Two profiles ignored this:

  (a) EPRI industrial shapes were read for weekdays only and applied to every
      day, weekends included. Weekend shapes are now retrieved from EPRI and
      used on the weather year's Saturdays and Sundays. The NERC region to
      state mapping also put states in the wrong region (IL and WI in ECAR,
      DE and NM in MAPP, PA in NPCC_NE, nothing in SPP).
  (b) EFS lays every year on one synthetic calendar whose January 1 is a
      Saturday, so its weekends landed on whatever weekday January 1 of the
      label year was. The strip is now rolled by up to three days so its
      weekends fall on the weather year's.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import constants as const
import retrieve_epri_load_shapes as epri
from build_demand import ReadCliu, ReadEfs

pytestmark = pytest.mark.fast

CONUS = sorted(set(const.STATE_2_TIMEZONE) - {"AK", "HI"})


def test_epri_regions_cover_every_contiguous_state_once():
    assigned = [state for states in ReadCliu.EPRI_NERC_2_STATE.values() for state in states]
    assert sorted(assigned) == CONUS
    assert set(ReadCliu.EPRI_NERC_2_STATE) == set(epri.REGIONS)


def _epri_shapes(weekday: float, weekend: float) -> pd.DataFrame:
    """Flat shapes for every region, season and end use, as retrieved."""
    rows = []
    for region in epri.REGIONS:
        for end_use in epri.END_USES.values():
            for season in epri.SEASONS:
                for day_type, level in (("AverageWeekday", weekday), ("AverageWeekend", weekend)):
                    rows.append(
                        {
                            "Region": region,
                            "Season": season,
                            "Day Type": day_type,
                            "Sector": "Industrial",
                            "End Use": end_use,
                            "Scale Multiplier": 6000.0,
                            **{f"HR{h}": level for h in range(1, 25)},
                        },
                    )
    return pd.DataFrame(rows)


def test_epri_weekend_shape_on_weather_year_weekends(tmp_path):
    path = tmp_path / "epri.csv"
    _epri_shapes(weekday=1.0, weekend=0.5).to_csv(path, index=False)

    # 2019: January 5 and 6 are Saturday and Sunday; in the 2014 labels they are not.
    reader = ReadCliu("unused", epri_filepath=str(path), calendar_year=2019)
    profiles = reader.get_demand_profiles(add_lpg=False)["electricity"]

    daily = profiles.xs("California", level="state").resample("D").sum()
    weekday_level = daily.loc["2014-01-03"]  # Thursday 2019-01-03
    assert daily.loc["2014-01-05"] == pytest.approx(0.5 * weekday_level)
    assert daily.loc["2014-01-06"] == pytest.approx(0.5 * weekday_level)
    assert daily.loc["2014-01-07"] == pytest.approx(weekday_level)

    # 52 or 53 weekends: every 2019 Saturday and Sunday, and nothing else, is low.
    low = daily[daily < 0.75 * weekday_level].index
    expected = pd.date_range("2019-01-01", "2019-12-31")
    expected = expected[expected.dayofweek >= 5].map(lambda d: d.replace(year=2014))
    assert list(low) == list(expected)

    # the year still sums to one per state
    assert profiles.xs("California", level="state").sum() == pytest.approx(1.0)


def test_epri_without_weekend_shapes_fails_loudly(tmp_path):
    path = tmp_path / "epri.csv"
    shapes = _epri_shapes(weekday=1.0, weekend=0.5)
    shapes[shapes["Day Type"] == "AverageWeekday"].to_csv(path, index=False)
    reader = ReadCliu("unused", epri_filepath=str(path), calendar_year=2019)
    with pytest.raises(ValueError, match="retrieve_epri_load_shapes"):
        reader.get_demand_profiles()


def test_retrieved_epri_layout_is_what_readcliu_reads(monkeypatch):
    monkeypatch.setattr(epri, "PAUSE_SECONDS", 0)
    monkeypatch.setattr(epri, "fetch_scale", lambda session, region, end_use: 6000.0)
    monkeypatch.setattr(
        epri,
        "fetch_shape",
        lambda session, region, end_use, season, day_type: [0.5 if day_type == "AverageWeekend" else 1.0] * 24,
    )
    shapes = epri.retrieve_load_shapes(session=None)

    assert len(shapes) == len(epri.REGIONS) * len(epri.END_USES) * len(epri.SEASONS) * len(epri.DAY_TYPES)
    formatted = ReadCliu("unused")._format_epri_data(shapes.drop(columns="Scale Multiplier"))
    assert set(formatted.index.get_level_values("day_type")) == {"AverageWeekday", "AverageWeekend"}
    assert set(formatted.columns) == {"electricity", "heat"}


@pytest.mark.parametrize(
    ("year", "shift"),
    [(2022, 0), (2019, -3), (2018, -2), (2021, 1), (2020, 3)],
)
def test_efs_weekday_shift(year, shift):
    # EFS January 1 is a Saturday; 2022 starts on one, 2019 on a Tuesday.
    assert ReadEfs(calendar_year=year).weekday_shift_days() == shift


def test_efs_weekends_moved_onto_weather_year_weekends():
    # EFS weekend: days 1-2, 8-9, ... of the synthetic year. Mark them low.
    hours = np.arange(1, const.HOURS_PER_YEAR + 1)
    day_of_year = (hours - 1) // 24
    load = np.where(day_of_year % 7 < 2, 0.5, 1.0)
    df = pd.DataFrame({"State": "NY", "LocalHourID": hours, "LoadMW": load})

    aligned = ReadEfs(calendar_year=2019)._align_weekdays(df.copy())
    daily = aligned.assign(day=(aligned.LocalHourID - 1) // 24).groupby("day").LoadMW.mean()

    weekend_2019 = pd.date_range("2019-01-01", periods=365)[daily.index].dayofweek >= 5
    wrong = daily.index[(daily == 0.5) != weekend_2019]
    # 365 days are not whole weeks: only a day next to the year wrap may be off
    assert len(wrong) <= 1
    assert all(day >= 365 - 3 for day in wrong)


def test_efs_without_calendar_year_is_unchanged():
    df = pd.DataFrame({"State": ["CA"], "LocalHourID": [1]})
    assert ReadEfs()._align_weekdays(df.copy()).LocalHourID.tolist() == [1]
