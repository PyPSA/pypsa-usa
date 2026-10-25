"""Unit test for aligning heat pump COP profiles to the snapshots in ``build_heat.reindex_cop``."""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import get_multiindex_snapshots
from build_heat import reindex_cop

pytestmark = pytest.mark.fast


def test_reindex_cop_from_a_leap_year():
    """A COP profile of a leap year carries Feb 29, which has no date in a non-leap investment period."""
    sns = get_multiindex_snapshots(
        sns_config={"start": "2030-02-28 00:00", "end": "2030-03-01 23:00", "inclusive": "both"},
        invest_periods=[2030],
    )
    time = pd.date_range("2020-02-28 00:00", "2020-03-01 23:00", freq="h")
    values = np.repeat(time.day.to_numpy(dtype=float)[:, None], 2, axis=1)  # COP = day of month
    da = xr.DataArray(values, coords={"time": time, "name": ["p1", "p2"]}, dims=["time", "name"])

    cop = reindex_cop(sns, da)

    assert cop.index.equals(sns)
    assert set(np.unique(cop.to_numpy())) == {28.0, 1.0}  # Feb 28 and Mar 1; Feb 29 is dropped
