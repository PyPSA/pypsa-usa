"""Regional component capex: the ReEDS county overnight-capex multipliers.

The real ReEDS table is read rather than mocked, so a data refresh that changes its
shape fails here.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import calculate_annuity
from regional_cost import (
    carrier_capital_charge_rate,
    carrier_multiplier,
    county_multiplier_table,
    load_reg_cap_cost_diff,
    overnight_delta_capital_cost,
    regional_capital_cost,
)

REG_CAP_COST_DIFF = Path(__file__).parents[2] / "repo_data" / "costs" / "reg_cap_cost_diff_default.csv"


@pytest.mark.fast
def test_county_multiplier_table_expands_groups():
    table = county_multiplier_table(
        load_reg_cap_cost_diff(str(REG_CAP_COST_DIFF)),
        {"NUCLEAR": ["nuclear", "SMR"], "BATTERY|PVB": ["4hr_battery_storage"]},
    )
    raw = load_reg_cap_cost_diff(str(REG_CAP_COST_DIFF))
    assert table.loc["p01001", "nuclear"] == pytest.approx(1 + raw.loc["p01001", "NUCLEAR"])
    assert (table["nuclear"] == table["SMR"]).all()
    with pytest.raises(ValueError):
        county_multiplier_table(raw, {})


@pytest.mark.fast
def test_regional_costs_use_the_capital_charge_rate():
    costs = pd.DataFrame(
        {"capital_charge_rate": [0.0342, 0.0], "cost_recovery_period_years": [30, 30], "wacc_real": [0.04, 0.04]},
        index=["solar", "CCGT"],
    )
    assert carrier_capital_charge_rate(costs, "solar") == 0.0342
    # load_costs zero-fills a missing rate: fall back to the capital recovery factor
    assert carrier_capital_charge_rate(costs, "CCGT") == pytest.approx(calculate_annuity(30, 0.04))


@pytest.mark.fast
def test_overnight_multiplier_leaves_fom_untouched():
    costs = pd.DataFrame(
        {
            "annualized_capex_fom": {"CCGT": 150_000.0},
            "capex_overnight_per_kw": {"CCGT": 1000.0},
            "cost_recovery_period_years": {"CCGT": 30.0},
            "wacc_real": {"CCGT": 0.05},
        },
    )
    bus_multipliers = pd.DataFrame({"CCGT": {"b1": 1.1, "b2": 0.9}})
    got = regional_capital_cost(costs, "CCGT", ["b1", "b2", "b1"], bus_multipliers)
    annuity = calculate_annuity(30.0, 0.05)
    assert got == pytest.approx(
        [150_000 + annuity * 0.1 * 1e6, 150_000 - annuity * 0.1 * 1e6, 150_000 + annuity * 0.1 * 1e6],
    )
    assert overnight_delta_capital_cost(150_000.0, 1000.0, annuity, 1.0) == pytest.approx(150_000.0)
    # unmapped carrier or disabled feature: uniform national value
    assert carrier_multiplier(bus_multipliers, "OCGT", ["b1"]) == 1.0
    assert regional_capital_cost(costs, "CCGT", ["b1"], None) == pytest.approx(150_000.0)
