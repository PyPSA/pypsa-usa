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
from build_cost_data import complete_atb_slice, implied_itc, offwind_floating_before_first_year

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


# ATB 2025 Moderate offshore wind, 30-year cost recovery (PUDL v2026.10.0), as in issue #938
OFFSHORE_COLS = [
    "pypsa-name",
    "projection_year",
    "scenario_atb",
    "model_case_nrelatb",
    "cost_recovery_period_years",
    "capex_per_kw",
    "wacc_real",
    "fixed_charge_rate",
    "model_tax_credit_case_nrelatb",
]
OFFSHORE = pd.DataFrame(
    [
        ("offwind", 2025, "Moderate", "R&D + TC", 30, 3500.0, 0.0430, 0.0357, "ITC"),
        ("offwind", 2030, "Moderate", "R&D + TC", 30, 3000.0, 0.0312, 0.0543, None),
        ("offwind_floating", 2030, "Moderate", "R&D + TC", 30, 5000.0, 0.0312, 0.0543, None),
        ("offwind", 2025, "Moderate", "R&D", 30, 3500.0, 0.0420, 0.0620, None),
        ("offwind_floating", 2030, "Moderate", "R&D", 30, 5000.0, 0.0400, 0.0610, None),
        # a scenario with no fixed-bottom row in 2025
        ("offwind_floating", 2030, "Advanced", "R&D", 30, 4500.0, 0.0400, 0.0610, None),
    ],
    columns=OFFSHORE_COLS,
)


def test_floating_offshore_takes_the_cost_years_financing():
    """Before 2030 floating offshore keeps its 2030 capex but is financed like fixed-bottom (#938)."""
    floating = offwind_floating_before_first_year(OFFSHORE, 2025).set_index(["scenario_atb", "model_case_nrelatb"])

    with_credit = floating.loc[("Moderate", "R&D + TC")]
    assert with_credit["capex_per_kw"] == 5000.0
    assert with_credit["projection_year"] == 2030
    assert (with_credit["wacc_real"], with_credit["fixed_charge_rate"]) == (0.0430, 0.0357)
    assert with_credit["model_tax_credit_case_nrelatb"] == "ITC"

    without_credit = floating.loc[("Moderate", "R&D")]
    assert (without_credit["wacc_real"], without_credit["fixed_charge_rate"]) == (0.0420, 0.0620)


def test_floating_offshore_without_a_fixed_bottom_row_keeps_its_financing():
    floating = offwind_floating_before_first_year(OFFSHORE, 2025).set_index(["scenario_atb", "model_case_nrelatb"])
    advanced = floating.loc[("Advanced", "R&D")]
    assert (advanced["wacc_real"], advanced["fixed_charge_rate"]) == (0.0400, 0.0610)


def test_floating_and_fixed_offshore_hold_the_same_itc_before_2030():
    floating = offwind_floating_before_first_year(OFFSHORE, 2025)
    year = pd.concat([OFFSHORE[OFFSHORE.projection_year == 2025], floating], ignore_index=True)
    year = year[year.scenario_atb == "Moderate"]
    with_credit = year[year.model_case_nrelatb == "R&D + TC"]
    base = year[year.model_case_nrelatb == "R&D"]

    itc = implied_itc(with_credit, base).groupby(with_credit["pypsa-name"]).first()

    assert itc["offwind"] > 0.3
    assert itc["offwind_floating"] == pytest.approx(itc["offwind"])
