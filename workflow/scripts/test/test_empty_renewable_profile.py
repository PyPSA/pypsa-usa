"""A renewable technology with no region in the model gets an empty profile (#426)."""

import os
import sys

import pandas as pd
import pytest
import xarray as xr

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from build_renewable_profiles import empty_profile

pytestmark = pytest.mark.fast

SNAPSHOTS = pd.date_range("2019-01-01", periods=24, freq="h")


@pytest.mark.parametrize("offshore", [True, False])
def test_empty_profile_is_skipped_by_add_electricity(tmp_path, offshore):
    """add_electricity leaves out a carrier whose profile has no buses."""
    fn = tmp_path / "profile.nc"
    empty_profile(SNAPSHOTS, offshore).to_netcdf(fn)

    with xr.open_dataset(fn) as ds:
        assert ds.indexes["bus"].empty  # the condition add_electricity skips on
        assert ds["profile"].dims == ("time", "bus")
        assert ds.sizes["time"] == 24
        for name in ("weight", "p_nom_max", "potential", "average_distance"):
            assert ds[name].dims == ("bus",)
        assert ("underwater_fraction" in ds) == offshore
