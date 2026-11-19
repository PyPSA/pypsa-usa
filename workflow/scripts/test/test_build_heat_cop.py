"""Unit test for aligning heat pump COP profiles to the snapshots in ``build_heat.reindex_cop``."""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import get_multiindex_snapshots, get_snapshots, weather_year_hours
from build_heat import reindex_cop

pytestmark = pytest.mark.fast


def _cop(time: pd.DatetimeIndex, values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        np.repeat(values[:, None], 2, axis=1),
        coords={"time": time, "name": ["p1", "p2"]},
        dims=["time", "name"],
    )


def test_reindex_cop_from_a_leap_weather_year():
    """A leap weather year arrives on the 365-day calendar and maps onto a non-leap period."""
    sns = get_multiindex_snapshots(
        sns_config={"start": "2030-02-28 00:00", "end": "2030-03-01 23:00", "inclusive": "both"},
        invest_periods=[2030],
    )
    # build_temperature_profiles lays weather year 2020 onto the 365-day
    # calendar by position: the 1 March label holds the Feb 29 weather.
    time = get_snapshots({"start": "2020-02-28 00:00", "end": "2020-03-01 23:00", "inclusive": "both"})
    weather_day = weather_year_hours(time).day.to_numpy(dtype=float)

    cop = reindex_cop(sns, _cop(time, weather_day))

    assert cop.index.equals(sns)
    assert set(np.unique(cop.to_numpy())) == {28.0, 29.0}


def test_reindex_cop_rejects_a_profile_with_feb_29():
    """Dropping Feb 29 here would put the COP a day off from the other weather-driven inputs."""
    sns = get_multiindex_snapshots(
        sns_config={"start": "2030-02-28 00:00", "end": "2030-03-01 23:00", "inclusive": "both"},
        invest_periods=[2030],
    )
    time = pd.date_range("2020-02-28 00:00", "2020-03-01 23:00", freq="h")

    with pytest.raises(ValueError, match="February 29"):
        reindex_cop(sns, _cop(time, time.day.to_numpy(dtype=float)))
