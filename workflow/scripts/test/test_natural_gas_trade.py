"""Unit tests for the gas trade links in ``build_natural_gas.TradeGasPipelineCapacity``.

The class is built without its EIA-backed ``__init__``; prices are synthetic, so
nothing here needs the API or the sector databundle.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import get_multiindex_snapshots
from build_natural_gas import TradeGasPipelineCapacity

pytestmark = pytest.mark.fast

PIPELINE_COLUMNS = [
    "STATE_NAME_TO",
    "STATE_NAME_FROM",
    "CAPACITY_MW",
    "STATE_TO",
    "STATE_FROM",
    "INTERCONNECT_TO",
    "INTERCONNECT_FROM",
]


def make_trade(data=None):
    trade = TradeGasPipelineCapacity.__new__(TradeGasPipelineCapacity)
    trade.year = 2018
    trade.interconnect = "western"
    trade.domestic = True
    trade.api = None
    trade._data = data
    return trade


def hourly_costs(end, value=3.0):
    return pd.DataFrame({"value": value}, index=pd.date_range("2018-01-01", end, freq="h"))


def test_expand_costs_drops_january_of_next_year(base_network):
    """EIA monthly prices run to January 1st of the next year, which the year swap duplicates."""
    costs = hourly_costs("2019-01-01 00:00", value=1.0)
    costs.iloc[-1] = 99.0

    expanded = make_trade()._expand_costs(base_network, costs)

    assert expanded.index.equals(base_network.snapshots)
    assert (expanded["value"] == 1.0).all()


def test_expand_costs_from_a_leap_year(base_network):
    """Prices of a leap year carry Feb 29, which has no date in a non-leap investment period."""
    base_network.set_snapshots(
        get_multiindex_snapshots(
            sns_config={"start": "2030-02-28 00:00", "end": "2030-03-01 23:00", "inclusive": "both"},
            invest_periods=[2030],
        ),
    )
    index = pd.date_range("2020-01-01", "2020-12-31 23:00", freq="h")
    costs = pd.DataFrame({"value": index.day.astype(float)}, index=index)

    expanded = make_trade()._expand_costs(base_network, costs)

    assert expanded.index.equals(base_network.snapshots)
    assert set(expanded["value"]) == {28.0, 1.0}  # Feb 28 and Mar 1; Feb 29 is dropped


def test_trade_link_costs_follow_link_names(base_network, monkeypatch):
    """Import and export costs are built separately; each must land on its own link."""
    base_network.add("Bus", ["CA", "TX"], suffix=" gas", carrier="gas")
    data = pd.DataFrame(
        [
            ["California", "Arizona", 100.0, "CA", "AZ", "western", "western"],
            ["Nevada", "California", 50.0, "NV", "CA", "western", "western"],
            ["California", "Nevada", 80.0, "CA", "NV", "western", "western"],
        ],
        columns=PIPELINE_COLUMNS,
    )
    trade = make_trade(data)
    monkeypatch.setattr(trade, "_get_domestic_costs", lambda: hourly_costs("2018-12-31 23:00"))

    trade.build_infrastructure(base_network)

    trade_links = base_network.links[base_network.links.carrier == "gas trade"]
    assert set(trade_links.index) == {"AZ CA gas trade", "NV CA gas trade", "CA NV gas trade", "CA AZ gas trade"}
    mc = base_network.links_t.marginal_cost
    for link in ("AZ CA gas trade", "NV CA gas trade"):  # imports pay the price
        assert (mc[link] == 3.0).all()
    for link in ("CA NV gas trade", "CA AZ gas trade"):  # exports earn it, less 0.1
        assert (mc[link] == -2.9).all()
