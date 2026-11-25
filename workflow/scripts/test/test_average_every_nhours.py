"""Unit tests for the hourly resampling in `prepare_network`."""

import os
import sys

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from prepare_network import average_every_nhours

pytestmark = pytest.mark.fast


def _two_period_network():
    n = pypsa.Network()
    sns = pd.date_range("2030-01-01", periods=24, freq="h").append(
        pd.date_range("2040-01-01", periods=24, freq="h"),
    )
    n.set_snapshots(pd.MultiIndex.from_arrays([sns.year, sns]))
    n.set_investment_periods([2030, 2040])
    n.investment_period_weightings["years"] = [10.0, 10.0]
    n.investment_period_weightings["objective"] = [8.98, 7.37]
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=pd.Series(range(48), index=n.snapshots, dtype=float))
    return n


def test_investment_period_weightings_survive_resampling():
    n = _two_period_network()
    m = average_every_nhours(n, "3h")
    pd.testing.assert_frame_equal(m.investment_period_weightings, n.investment_period_weightings)


def test_snapshot_weightings_are_summed():
    n = _two_period_network()
    m = average_every_nhours(n, "3h")
    assert len(m.snapshots) == 16
    assert (m.snapshot_weightings.objective == 3.0).all()
