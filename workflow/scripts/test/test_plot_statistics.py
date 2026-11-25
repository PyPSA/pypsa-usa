"""Helpers of plot_statistics."""

import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from plot_statistics import single_signed_columns

pytestmark = pytest.mark.fast


def test_area_columns_are_clipped_to_one_sign():
    """Slightly negative dispatch from a barrier solution must not break the stacked area chart (#639)."""
    energy_mix = pd.DataFrame(
        {
            "solar": [-1e-3, 5.0],
            "battery_discharger": [1e-4, 2.0],
            "battery_charger": [-3.0, 1e-4],
            "exports": [-1.0, 1e-5],
        },
    )
    clipped = single_signed_columns(energy_mix)

    assert (clipped[["solar", "battery_discharger"]] >= 0).all().all()
    assert (clipped[["battery_charger", "exports"]] <= 0).all().all()
    assert clipped.loc[1, "solar"] == 5.0
    assert clipped.loc[0, "battery_charger"] == -3.0
    assert energy_mix.loc[0, "solar"] == -1e-3  # the input is left as it was
    clipped.plot.area()  # raises if a column mixes signs
