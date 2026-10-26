"""ATB slice selection in ``build_cost_data`` and ``_helpers.load_costs``.

The 2025 ATB publishes four model cases (``R&D``, ``R&D + TC``, ``Exp``,
``Exp + TC``), and the expanded cost-driver cases only for a few technologies.
These tests pin the two contracts that keep a partial case from reaching the
model as zero capital cost: ``build_cost_data`` completes every slice from its
R&D counterpart, and ``load_costs`` refuses a slice the cost table does not
carry instead of silently selecting nothing.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import load_costs
from build_cost_data import complete_atb_slice

pytestmark = pytest.mark.fast


def _atb_rows(rows):
    return pd.DataFrame(rows, columns=["pypsa-name", "scenario_atb", "model_case_nrelatb", "capex_per_kw"])


ATB_YEAR = _atb_rows(
    [
        ("solar", "Moderate", "R&D", 1000.0),
        ("solar", "Moderate", "Exp", 1100.0),
        ("CCGT", "Moderate", "R&D", 900.0),
        ("CCGT", "Moderate", "R&D + TC", 890.0),
        ("solar", "Moderate", "R&D + TC", 990.0),
        ("solar", "Moderate", "Exp + TC", 1090.0),
    ],
)


def test_exp_slice_takes_carriers_it_lacks_from_rd():
    sub = complete_atb_slice(ATB_YEAR, "Moderate", "Exp").set_index("pypsa-name")
    assert sub.loc["solar", "capex_per_kw"] == 1100.0  # Exp row kept, not overwritten
    assert sub.loc["CCGT", "capex_per_kw"] == 900.0  # filled from R&D


def test_exp_tc_slice_falls_back_to_rd_tc():
    sub = complete_atb_slice(ATB_YEAR, "Moderate", "Exp + TC").set_index("pypsa-name")
    assert sub.loc["CCGT", "capex_per_kw"] == 890.0


def test_a_case_the_scenario_does_not_publish_is_empty():
    assert complete_atb_slice(ATB_YEAR, "Advanced", "R&D").empty


def test_a_slice_still_missing_a_carrier_raises():
    partial = ATB_YEAR[~((ATB_YEAR["pypsa-name"] == "CCGT") & (ATB_YEAR.model_case_nrelatb == "R&D + TC"))]
    with pytest.raises(ValueError, match="CCGT"):
        complete_atb_slice(partial, "Moderate", "R&D + TC")


@pytest.fixture
def tech_costs(tmp_path):
    rows = [
        ("solar", "capex_per_kw", 1000.0, "Moderate", "R&D"),
        ("solar", "capex_per_kw", 990.0, "Moderate", "R&D + TC"),
        ("solar", "capex_per_kw", 800.0, "Advanced", "R&D + TC"),
        ("CCGT", "capex_per_kw", 900.0, "Moderate", "R&D"),
        ("CCGT", "capex_per_kw", 890.0, "Moderate", "R&D + TC"),
        ("CCGT", "capex_per_kw", 700.0, "Advanced", "R&D + TC"),
        ("EGS", "capex_per_kw", 5000.0, None, None),
    ]
    path = tmp_path / "costs_2030.csv"
    pd.DataFrame(rows, columns=["pypsa-name", "parameter", "value", "atb_scenario", "atb_model_case"]).to_csv(
        path,
        index=False,
    )
    return str(path)


def test_load_costs_defaults_to_rd_tc(tech_costs):
    costs = load_costs(tech_costs, {"atb": {"scenario": "Moderate"}})
    assert costs.at["solar", "capex_per_kw"] == 990.0
    assert costs.at["EGS", "capex_per_kw"] == 5000.0  # scenario-independent rows always apply


def test_load_costs_applies_per_carrier_overrides(tech_costs):
    cfg = {"atb": {"scenario": "Moderate", "model_case": "R&D + TC", "overrides": {"solar": {"scenario": "Advanced"}}}}
    costs = load_costs(tech_costs, cfg)
    assert costs.at["solar", "capex_per_kw"] == 800.0
    assert costs.at["CCGT", "capex_per_kw"] == 890.0


@pytest.mark.parametrize(
    "atb_cfg",
    [
        {"scenario": "Moderate", "model_case": "Market"},  # the 2024 ATB label
        {"scenario": "Moderate", "model_case": "R&D + TC", "overrides": {"CCGT": {"model_case": "Market"}}},
    ],
)
def test_load_costs_rejects_a_slice_the_table_does_not_carry(tech_costs, atb_cfg):
    with pytest.raises(ValueError, match="Market"):
        load_costs(tech_costs, {"atb": atb_cfg})
