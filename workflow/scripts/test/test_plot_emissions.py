"""Emission charts of plot_statistics, one panel per investment period (#345)."""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import plot_statistics
from _helpers import get_multiindex_snapshots

pytestmark = pytest.mark.fast


@pytest.fixture
def network():
    """Two investment periods of three hours each, with gas and coal carriers."""
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 02:00", "inclusive": "both"},
        invest_periods=[2030, 2040],
    )
    n.set_investment_periods(periods=[2030, 2040])
    n.add("Carrier", ["gas", "coal"], nice_name=["Gas", "Coal"], color=["#aa0000", "#333333"])
    return n


@pytest.fixture
def emissions(network):
    return pd.DataFrame({"Gas": 1.0, "Coal": 2.0}, index=network.snapshots)


def test_running_total_restarts_at_every_period(emissions):
    total = plot_statistics.cumulative_per_period(emissions)
    assert total.loc[2030, "Gas"].tolist() == [1.0, 2.0, 3.0]
    assert total.loc[2040, "Gas"].tolist() == [1.0, 2.0, 3.0]


def test_one_panel_per_investment_period(network, emissions, tmp_path, monkeypatch):
    panels = {}
    monkeypatch.setattr(plt, "close", lambda fig=None: panels.setdefault("axes", fig.axes))

    plot_statistics.plot_area_per_period(network, emissions, "Emissions", "MT", str(tmp_path / "emissions.pdf"))

    assert [ax.get_title() for ax in panels["axes"]] == ["Emissions (2030)", "Emissions (2040)"]
    assert (tmp_path / "emissions.pdf").exists()


def test_hourly_emissions_chart_is_split_by_period(network, emissions, tmp_path, monkeypatch):
    panels = {}
    monkeypatch.setattr(plt, "close", lambda fig=None: panels.setdefault("axes", fig.axes))
    monkeypatch.setattr(plot_statistics, "get_tech_emissions_timeseries", lambda n: emissions * 1e6)

    plot_statistics.plot_hourly_emissions(network, str(tmp_path / "hourly.pdf"), interconnect="western")

    assert len(panels["axes"]) == 2
    assert all(ax.get_ylabel() == "Emissions [MT]" for ax in panels["axes"])
