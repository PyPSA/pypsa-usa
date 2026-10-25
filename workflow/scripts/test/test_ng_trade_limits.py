"""
Test the natural gas import and export limits.

``add_ng_import_export_limits`` bounds the flow on each interstate gas trade link
by the EIA annual volume for that flow. The volume must be scaled to the hours
modelled, and each link must get exactly one value even though EIA returns the
year before the data year as well. A ``min`` of 0 must disable the floor rather
than fall back to 1.
"""

import os
import sys

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import opts.sector as sector
from _helpers import get_multiindex_snapshots
from constants import HOURS_PER_YEAR, NG_MWH_2_MMCF

IMPORT_LINK = "WY UT gas trade"
EXPORT_LINK = "UT WY gas trade"

# annual volumes in MMCF, as EIA reports them for the year before and the data year
VOLUMES = {
    "WY-UT": {2017: 1000.0, 2018: 1200.0},
    "UT-WY": {2017: 300.0, 2018: 400.0},
}

# Fixtures


class FakeTrade:
    """Stands in for eia.Trade: domestic flows from ``VOLUMES``, no international ones."""

    volumes = VOLUMES

    def __init__(self, fuel, international, direction, year, api):
        self.international = international

    def get_data(self):
        """Annual volumes in the shape eia.Trade returns: indexed by period, one row per flow and year."""
        rows = (
            []
            if self.international
            else [
                (period, state, value) for state, by_year in self.volumes.items() for period, value in by_year.items()
            ]
        )
        df = pd.DataFrame(rows, columns=["period", "state", "value"]).set_index("period")
        df["series-description"] = ""
        df["units"] = "MMCF"
        return df


@pytest.fixture
def trade_network(monkeypatch):
    """UT gas bus with one import link from WY and one export link to WY, 24 hourly snapshots."""
    monkeypatch.setattr(sector, "Trade", FakeTrade)

    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 23:00", "inclusive": "both"},
        invest_periods=[2030],
    )
    n.set_investment_periods(periods=[2030])

    n.add("Bus", ["UT gas", "WY gas trade"], carrier="gas")
    n.add("Link", IMPORT_LINK, bus0="WY gas trade", bus1="UT gas", carrier="gas trade", p_nom=1e6, marginal_cost=1.0)
    n.add("Link", EXPORT_LINK, bus0="UT gas", bus1="WY gas trade", carrier="gas trade", p_nom=1e6, marginal_cost=1.0)
    # both buses can take up or supply any gas the trade limits push through
    n.add("Generator", ["UT balance", "WY balance"], bus=["UT gas", "WY gas trade"], p_nom=1e6, p_min_pu=-1)
    return n


# Helpers


def build_model(n, imports=None, exports=None):
    config = {
        "api": {"eia": "key"},
        "snapshots": {"start": "2018-01-01"},
        "sector": {
            "natural_gas": {
                "imports": imports or {"min": 0.99, "max": 1.01},
                "exports": exports or {"min": 0.99, "max": 1.01},
            },
        },
    }
    n.optimize.create_model(multi_investment_periods=True)
    sector.add_ng_import_export_limits(n, config)
    return n.model


def rhs(model, name):
    return float(model.constraints[name].rhs.item())


def expected(mmcf, multiplier, hours=24):
    return mmcf * NG_MWH_2_MMCF * multiplier * hours / HOURS_PER_YEAR


# Tests


def test_limits_use_the_data_year_scaled_to_the_modelled_hours(trade_network):
    model = build_model(trade_network)

    assert rhs(model, f"ng_limit_import_min-2030-{IMPORT_LINK}") == pytest.approx(expected(1200.0, 0.99))
    assert rhs(model, f"ng_limit_import_max-2030-{IMPORT_LINK}") == pytest.approx(expected(1200.0, 1.01))
    assert rhs(model, f"ng_limit_export_min-2030-{EXPORT_LINK}") == pytest.approx(expected(400.0, 0.99))
    assert rhs(model, f"ng_limit_export_max-2030-{EXPORT_LINK}") == pytest.approx(expected(400.0, 1.01))


def test_short_run_with_limits_is_feasible(trade_network):
    n = trade_network
    model = build_model(n)
    status, _ = n.optimize.solve_model(solver_name="highs")
    assert status == "ok"
    assert model.objective.value is not None


def test_falls_back_to_the_year_before_when_the_data_year_is_missing(trade_network, monkeypatch):
    monkeypatch.setattr(FakeTrade, "volumes", {"WY-UT": {2017: 1000.0}, "UT-WY": {2017: 300.0}})
    model = build_model(trade_network)

    assert rhs(model, f"ng_limit_import_min-2030-{IMPORT_LINK}") == pytest.approx(expected(1000.0, 0.99))


def test_zero_minimum_disables_the_floor(trade_network):
    model = build_model(trade_network, imports={"min": 0, "max": 1.01}, exports={"min": 0, "max": 1.01})

    assert rhs(model, f"ng_limit_import_min-2030-{IMPORT_LINK}") == 0.0
    assert rhs(model, f"ng_limit_export_min-2030-{EXPORT_LINK}") == 0.0
    assert rhs(model, f"ng_limit_import_max-2030-{IMPORT_LINK}") == pytest.approx(expected(1200.0, 1.01))
