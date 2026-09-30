"""Unit tests for the EIA gating in ``build_fuel_prices``.

A default power-only run leaves ``conventional: dynamic_fuel_price: enable``
false, and the state-level price tables it produces are never read. These tests
pin that contract: with the feature off, no HTTP request is made even when an
API key is configured; with it on, the EIA fetch is attempted.

Nothing here touches the network — ``eia.FuelCosts`` and ``requests.Session``
are both replaced with fakes that fail the test if they are reached.
"""

import os
import sys
from typing import ClassVar

import pandas as pd
import pytest
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import build_fuel_prices as bfp

pytestmark = pytest.mark.fast

SNAPSHOTS = pd.date_range("2019-01-01", periods=24, freq="h")
FAKE_KEY = "fake-eia-key"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if anything in these tests opens an HTTP connection."""

    def _boom(*args, **kwargs):
        raise AssertionError("unexpected HTTP request")

    monkeypatch.setattr(requests.Session, "get", _boom)
    monkeypatch.setattr(requests, "get", _boom)


class _RecordingFuelCosts:
    """Stand-in for ``eia.FuelCosts`` that records the calls it receives."""

    calls: ClassVar[list[tuple]] = []

    def __init__(self, fuel, year, api_key, industry=None):
        type(self).calls.append((fuel, year, api_key, industry))
        self.fuel = fuel

    def get_data(self, pivot=False):
        # Month-start index, one column, as the real pivoted payload looks.
        index = pd.date_range("2019-01-01", periods=12, freq="MS")
        return pd.DataFrame({"California": 1.0}, index=index)


@pytest.fixture
def fuel_costs(monkeypatch):
    _RecordingFuelCosts.calls = []
    monkeypatch.setattr(bfp.eia, "FuelCosts", _RecordingFuelCosts)
    return _RecordingFuelCosts


def test_disabled_dynamic_fuel_price_skips_fetch(fuel_costs, caplog):
    """enable=false with a key present: empty tables, zero EIA calls."""
    with caplog.at_level("INFO", logger=bfp.logger.name):
        ng, coal = bfp.build_state_power_prices(
            SNAPSHOTS,
            FAKE_KEY,
            {"enable": False, "pudl": True, "wholesale": True},
        )

    assert fuel_costs.calls == []
    assert ng.empty and coal.empty
    assert list(ng.index) == list(SNAPSHOTS)
    assert list(coal.index) == list(SNAPSHOTS)
    assert "dynamic_fuel_price.enable is false" in caplog.text


def test_missing_dynamic_fuel_price_params_skips_fetch(fuel_costs):
    """A rule that passes no params at all must not reach the API either."""
    ng, coal = bfp.build_state_power_prices(SNAPSHOTS, FAKE_KEY, None)

    assert fuel_costs.calls == []
    assert ng.empty and coal.empty


def test_enabled_without_key_skips_fetch_and_warns(fuel_costs, caplog):
    """enable=true but no key: empty tables and a loud warning, no API call."""
    with caplog.at_level("WARNING", logger=bfp.logger.name):
        ng, coal = bfp.build_state_power_prices(SNAPSHOTS, "", {"enable": True})

    assert fuel_costs.calls == []
    assert ng.empty and coal.empty
    assert "no EIA API key" in caplog.text


def test_enabled_with_key_fetches(fuel_costs):
    """enable=true with a key: both the gas and the coal fetch are attempted."""
    ng, coal = bfp.build_state_power_prices(
        SNAPSHOTS,
        FAKE_KEY,
        {"enable": True, "pudl": True, "wholesale": True},
    )

    fuels = [call[0] for call in fuel_costs.calls]
    assert fuels == ["gas", "coal"]
    assert all(call[2] == FAKE_KEY for call in fuel_costs.calls)
    assert all(call[3] == "power" for call in fuel_costs.calls)

    # the fetched frames are made hourly, so they are not the empty fallback
    assert not ng.empty
    assert not coal.empty
