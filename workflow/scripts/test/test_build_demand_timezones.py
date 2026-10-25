"""Demand readers must land every source on UTC snapshots.

Snapshots are UTC with no daylight saving: EFS, EER and SERVM roll their
local-standard-time strips forward, and FERC 714 / EIA-930 arrive in UTC.
Two sector-study readers did not:

  (a) EULP (ResStock/ComStock) is stamped in EST for every state. The reader
      computed a per-state EST -> local shift and then discarded it, so all
      residential and commercial loads sat 5 h early against UTC.
  (b) EPRI industrial shapes are in local time and their HR1-HR24 labels were
      used as UTC hours, putting industrial loads 5-8 h early.
  (c) EULP 15-min values are stamped at the end of each interval, but were
      binned left-closed into hours, so every hour ran 15 min early and the
      closing 2019-01-01 00:00 stamp landed in the first hour of 2018.

These tests pin the UTC alignment of both readers, the EULP hourly binning,
and the shared offset table EFS uses.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import constants as const
from build_demand import ReadCliu, ReadEfs, ReadEulp, _state_utc_offset
from eulp import Eulp

pytestmark = pytest.mark.fast

HOURS = const.HOURS_PER_YEAR


def _write_eulp_state(directory, state: str) -> str:
    """One profile-year EULP strip whose value is its EST hour of year."""
    index = pd.date_range("2018-01-01", periods=HOURS, freq="h", name="timestamp")
    values = np.arange(HOURS, dtype=float)
    df = pd.DataFrame(
        {
            "electricity": values,
            "heating": values,
            "space_heating": values,
            "water_heating": values,
            "cooling": values,
        },
        index=index,
    )
    path = os.path.join(directory, f"{state}.csv")
    df.to_csv(path)
    return path


def test_eulp_rolls_est_to_utc_for_every_state(tmp_path):
    files = [_write_eulp_state(tmp_path, state) for state in ("CA", "TX", "NY")]
    demand = ReadEulp(files, "res").read_demand()

    elec = demand.xs("electricity", level="fuel").droplevel(["sector", "subsector"])
    assert len(elec) == HOURS

    # EST midnight is 05:00 UTC in every state, whatever its own timezone.
    est_midnight = pd.Timestamp("2018-01-01 05:00")
    assert (elec.loc[est_midnight] == 0).all()

    # The last five EST hours of the year wrap to the start of the strip.
    assert (elec.loc[pd.Timestamp("2018-01-01 00:00")] == HOURS - ReadEulp.EST_TO_UTC_SHIFT).all()

    # Energy is only moved, never lost.
    assert np.allclose(elec.sum(), np.arange(HOURS).sum())


def test_eulp_interval_end_stamps_binned_into_their_starting_hour(tmp_path):
    # Raw EULP layout: 15-min kWh stamped at interval end, 00:15 ... 2019-01-01 00:00.
    stamps = pd.date_range("2018-01-01 00:15", "2019-01-01 00:00", freq="15min", name="timestamp")
    kwh = np.zeros(len(stamps))
    kwh[:4] = 1000.0  # the four intervals of the first hour: ending 00:15 to 01:00
    kwh[-1] = 4000.0  # the last interval of the year, stamped 2019-01-01 00:00
    raw = pd.DataFrame({"out.electricity.plug_loads.energy_consumption.kwh": kwh}, index=stamps)
    path = tmp_path / "raw.csv"
    raw.to_csv(path)

    elec = Eulp(str(path)).electric  # MWh per hour

    assert len(elec) == HOURS
    assert elec.index[0] == pd.Timestamp("2018-01-01 00:00")
    assert elec.index[-1] == pd.Timestamp("2018-12-31 23:00")
    assert elec.loc["2018-01-01 00:00"] == pytest.approx(4.0)
    assert elec.loc["2018-01-01 01:00"] == pytest.approx(0.0)
    assert elec.loc["2018-12-31 23:00"] == pytest.approx(4.0)
    assert elec.sum() == pytest.approx(kwh.sum() / 1000)


def _epri_frame() -> pd.DataFrame:
    """One NERC region per coast with a single spike at local hour 12 (HR13)."""
    hours = {f"HR{x}": 0.0 for x in range(1, 25)}
    hours["HR13"] = 1.0
    rows = []
    for region in ("WSCC_CNV", "NPCC_NE"):
        for season in ("Peak", "OffPeak"):
            rows.append(
                {
                    "Region": region,
                    "Season": season,
                    "Day Type": "AverageWeekday",
                    "Sector": "Industrial",
                    "End Use": "Lighting",
                    **hours,
                },
            )
    return pd.DataFrame(rows)


def test_epri_typical_day_rotated_to_utc():
    reader = ReadCliu("unused")
    shapes = reader._format_epri_data(_epri_frame())["electricity"]

    def spike_hour(state: str) -> int:
        day = shapes.xs((state, 1), level=["state", "month"])
        return int(day.idxmax())

    assert spike_hour("CA") == 12 + 8  # PST noon is 20:00 UTC
    assert spike_hour("NV") == 12 + 8
    assert spike_hour("MA") == 12 + 5  # EST noon is 17:00 UTC

    # Each typical day still has all 24 hours and the same energy.
    day = shapes.xs(("CA", 1), level=["state", "month"])
    assert sorted(day.index) == list(range(24))
    assert day.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("state", "offset"),
    [("ME", 5), ("IL", 6), ("CO", 7), ("AZ", 7), ("WA", 8)],
)
def test_state_utc_offsets(state, offset):
    assert _state_utc_offset(state) == offset


def test_efs_local_hour_mapped_forward_to_utc():
    df = pd.DataFrame({"State": ["CA", "NY"], "LocalHourID": [1, HOURS]})
    out = ReadEfs._apply_timezones(df)
    assert out.UtcHourID.tolist() == [1 + 8, 5]  # NY's last local hour wraps
