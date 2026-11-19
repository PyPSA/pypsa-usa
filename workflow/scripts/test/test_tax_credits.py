"""Tax credits: the ATB ``+ TC`` cases and the ``costs: itc_modifier`` / ``ptc_modifier`` overrides.

The ITC reaches the model through the capital charge rate that annualizes capex
(``build_cost_data.capital_charge_rate``), batteries borrowing the financing of
their ``FINANCE_PROXY``; ``itc_effective`` records how much of that rate the ATB
credit takes off (``build_cost_data.implied_itc``). The PTC is backed out of the
ATB LCOE (``build_cost_data.levelized_ptc``) and taken off the marginal cost of
each new vintage (``add_extra_components.apply_ptc``). ``load_costs`` replaces
either with the configured value of the build year
(``tax_credits.resolve_tax_credits``).
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import calculate_annuity, get_multiindex_snapshots, load_costs
from add_extra_components import apply_ptc
from build_cost_data import (
    apply_finance_proxies,
    capital_charge_rate,
    implied_itc,
    levelized_ptc,
    project_finance_factor,
)
from tax_credits import (
    PTC_YEARS,
    base_model_case,
    check_tax_credit_years,
    credit_for_year,
    levelize_ptc,
)

pytestmark = pytest.mark.fast


def _lcoe(fcr, capex, fom, cf, vom, ptc):
    return (fcr * capex + fom) * 1e3 / (8760 * cf) + vom - ptc


def _atb_rows(label, ptc):
    return pd.DataFrame(
        {
            "fixed_charge_rate": [0.057],
            "capex_per_kw": [1800.0],
            "opex_fixed_per_kw": [30.0],
            "capacity_factor": [0.4],
            "opex_variable_per_mwh": [np.nan],
            "fuel_cost_per_mwh": [np.nan],
            "levelized_cost_of_energy_per_mwh": [_lcoe(0.057, 1800.0, 30.0, 0.4, 0.0, ptc)],
            "model_tax_credit_case_nrelatb": [label],
        },
    )


def test_ptc_is_recovered_from_the_lcoe():
    assert levelized_ptc(_atb_rows("PTC", 19.5)).iloc[0] == pytest.approx(19.5)


def test_an_expired_ptc_is_zero():
    assert levelized_ptc(_atb_rows("PTC", 0.0)).iloc[0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("label", ["ITC", None])
def test_rows_claiming_no_ptc_get_none(label):
    # an ITC carrier whose LCOE does not close exactly must not read as a PTC
    assert levelized_ptc(_atb_rows(label, 5.0)).iloc[0] == 0.0


def _finance(names, wacc, fcr, label=None):
    return pd.DataFrame(
        {
            "pypsa-name": names,
            "cost_recovery_period_years": 30.0,
            "wacc_real": wacc,
            "fixed_charge_rate": fcr,
            "model_tax_credit_case_nrelatb": label,
        },
    )


def test_implied_itc_reads_the_credit_off_the_finance_factor():
    crf = {w: calculate_annuity(30, w) for w in (0.03, 0.04)}
    # R&D: finance factor 1.05 for both; + TC: a higher WACC, and solar's factor holds an ITC
    base = _finance(["solar", "CCGT"], 0.03, [crf[0.03] * 1.05] * 2)
    tc = _finance(["solar", "CCGT"], 0.04, [crf[0.04] * 0.63, crf[0.04] * 1.05], ["ITC", None])
    itc = implied_itc(tc, base).set_axis(tc["pypsa-name"])
    assert itc["solar"] == pytest.approx(1 - 0.63 / 1.05)
    # the WACC change alone is no credit
    assert itc["CCGT"] == 0.0
    assert (implied_itc(tc, None) == 0.0).all()


def test_implied_itc_ignores_rows_naming_no_itc():
    crf = calculate_annuity(30, 0.03)
    base = _finance(["onwind"], 0.03, [crf * 1.05])
    tc = _finance(["onwind"], 0.03, [crf * 0.9], ["PTC"])
    assert implied_itc(tc, base).iloc[0] == 0.0


def _pivot():
    return pd.DataFrame(
        {
            "pypsa-name": ["geothermal", "4hr_battery_storage", "CCGT"],
            "cost_recovery_period_years": [30.0, 20.0, 30.0],
            "wacc_real": [0.0428, np.nan, 0.0428],
            "fixed_charge_rate": [0.0356, np.nan, np.nan],
            "itc_effective": [0.4, 0.0, 0.0],
        },
    )


def test_batteries_borrow_the_financing_of_their_proxy():
    out = apply_finance_proxies(_pivot()).set_index("pypsa-name")
    finance_factor = 0.0356 / calculate_annuity(30, 0.0428)
    assert out.at["4hr_battery_storage", "wacc_real"] == 0.0428
    assert out.at["4hr_battery_storage", "fixed_charge_rate"] == pytest.approx(
        calculate_annuity(20, 0.0428) * finance_factor,
    )
    # the borrowed finance factor carries the proxy's ITC
    assert out.at["4hr_battery_storage", "itc_effective"] == 0.4
    assert project_finance_factor(out.loc[["4hr_battery_storage"]]).iloc[0] == pytest.approx(finance_factor)


def test_carriers_without_a_fixed_charge_rate_use_their_capital_recovery_factor():
    pivot = apply_finance_proxies(_pivot())
    rate = capital_charge_rate(pivot).set_axis(pivot["pypsa-name"])
    assert rate["geothermal"] == 0.0356
    assert rate["CCGT"] == pytest.approx(calculate_annuity(30, 0.0428))


# --- costs: itc_modifier / ptc_modifier -------------------------------------

CRP, WACC = 30.0, 0.05
# (rate, itc_effective, ptc) per carrier and case. solar's + TC rate holds an ATB ITC
# of 0.25; onwind claims the PTC, and its + TC rate differs by WACC only.
ATB = {
    ("solar", "R&D"): (0.06, 0.0, 0.0),
    ("solar", "R&D + TC"): (0.045, 0.25, 0.0),
    ("onwind", "R&D"): (0.055, 0.0, 0.0),
    ("onwind", "R&D + TC"): (0.05, 0.0, 15.0),
}
CAPEX = {"solar": (1000.0, 20.0), "onwind": (1300.0, 30.0)}


@pytest.fixture
def tech_costs(tmp_path):
    rows = []
    for (carrier, case), (rate, itc, ptc) in ATB.items():
        capex, fom = CAPEX[carrier]
        params = {
            "capital_charge_rate": rate,
            "itc_effective": itc,
            "ptc_per_mwh": ptc,
            "capex_per_kw": capex,
            "opex_fixed_per_kw": fom,
            "annualized_capex_per_mw": rate * capex * 1e3,
            "annualized_capex_fom": (rate * capex + fom) * 1e3,
            "cost_recovery_period_years": CRP,
            "wacc_real": WACC,
        }
        rows += [(carrier, p, v, "Moderate", case) for p, v in params.items()]
    rows += [
        ("EGS", "annualized_capex_per_mw", 400000.0, None, None),
        ("EGS", "annualized_capex_fom", 500000.0, None, None),
    ]
    path = tmp_path / "costs_2030.csv"
    pd.DataFrame(rows, columns=["pypsa-name", "parameter", "value", "atb_scenario", "atb_model_case"]).to_csv(
        path,
        index=False,
    )
    return str(path)


def _cfg(model_case="R&D + TC", **credits):
    return {"atb": {"scenario": "Moderate", "model_case": model_case}, **credits}


def test_without_overrides_the_atb_case_holds(tech_costs):
    costs = load_costs(tech_costs, _cfg(itc_modifier=None, ptc_modifier={}))
    assert costs.at["solar", "capital_charge_rate"] == 0.045
    assert costs.at["onwind", "ptc_per_mwh"] == 15.0


def test_an_itc_replaces_the_atb_itc(tech_costs):
    costs = load_costs(tech_costs, _cfg(itc_modifier={"solar": 0.3}))
    rate = 0.045 / (1 - 0.25) * (1 - 0.3)  # the WACC of the + TC case, its credit swapped
    assert costs.at["solar", "capital_charge_rate"] == pytest.approx(rate)
    assert costs.at["solar", "itc_effective"] == 0.3
    assert costs.at["solar", "annualized_capex_per_mw"] == pytest.approx(rate * 1000.0 * 1e3)
    # FOM is untouched
    assert costs.at["solar", "annualized_capex_fom"] == pytest.approx((rate * 1000.0 + 20.0) * 1e3)


def test_an_itc_on_a_case_without_tax_credits(tech_costs):
    costs = load_costs(tech_costs, _cfg(model_case="R&D", itc_modifier={"solar": 0.3}))
    assert costs.at["solar", "capital_charge_rate"] == pytest.approx(0.06 * 0.7)


def test_a_zero_itc_where_the_atb_grants_none_changes_nothing(tech_costs):
    atb = load_costs(tech_costs, _cfg())
    costs = load_costs(tech_costs, _cfg(itc_modifier={"onwind": 0}))
    pd.testing.assert_series_equal(costs.loc["onwind"], atb.loc["onwind"])


def test_a_positive_itc_drops_the_atb_ptc(tech_costs):
    costs = load_costs(tech_costs, _cfg(itc_modifier={"onwind": 0.3}))
    assert costs.at["onwind", "ptc_per_mwh"] == 0.0
    assert costs.at["onwind", "capital_charge_rate"] == pytest.approx(0.05 * 0.7)


def test_a_positive_ptc_drops_the_atb_itc(tech_costs):
    costs = load_costs(tech_costs, _cfg(ptc_modifier={"solar": 27.5}))
    assert costs.at["solar", "capital_charge_rate"] == pytest.approx(0.045 / (1 - 0.25))
    assert costs.at["solar", "itc_effective"] == 0.0
    assert costs.at["solar", "ptc_per_mwh"] == pytest.approx(levelize_ptc(27.5, WACC, CRP))


def test_both_credits_set_apply_both(tech_costs):
    costs = load_costs(tech_costs, _cfg(itc_modifier={"solar": 0.1}, ptc_modifier={"solar": 10.0}))
    assert costs.at["solar", "capital_charge_rate"] == pytest.approx(0.045 / 0.75 * 0.9)
    assert costs.at["solar", "ptc_per_mwh"] == pytest.approx(levelize_ptc(10.0, WACC, CRP))


def test_per_year_values_apply_to_their_build_year_only(tech_costs):
    cfg = _cfg(itc_modifier={"solar": {2030: 0.2, 2040: 0.0}, "onwind": {2040: 0.3}})
    costs = load_costs(tech_costs, cfg)  # build year from costs_2030.csv
    assert costs.at["solar", "capital_charge_rate"] == pytest.approx(0.045 / 0.75 * 0.8)
    # 2030 unset: the ATB case holds
    assert costs.at["onwind", "capital_charge_rate"] == 0.05
    assert costs.at["onwind", "ptc_per_mwh"] == 15.0
    # an explicit year wins over the file name
    later = load_costs(tech_costs, cfg, year=2040)
    assert later.at["solar", "capital_charge_rate"] == pytest.approx(0.06)
    assert later.at["onwind", "ptc_per_mwh"] == 0.0


def test_an_annualized_cost_given_directly_is_scaled(tech_costs):
    costs = load_costs(tech_costs, _cfg(itc_modifier={"EGS": 0.3}))
    assert costs.at["EGS", "annualized_capex_per_mw"] == pytest.approx(0.7 * 400000.0)
    assert costs.at["EGS", "annualized_capex_fom"] == pytest.approx(500000.0 - 0.3 * 400000.0)
    assert costs.at["EGS", "capital_charge_rate"] == 0.0  # it has none, and gets none


def test_unknown_carriers_are_ignored_with_a_warning(tech_costs, caplog):
    with caplog.at_level(logging.WARNING):
        costs = load_costs(tech_costs, _cfg(itc_modifier={"8hr_PHS": 0.3}))
    assert "8hr_PHS" in caplog.text
    assert costs.at["solar", "capital_charge_rate"] == 0.045


def test_a_table_without_itc_effective_warns(tmp_path, caplog):
    path = tmp_path / "costs_2030.csv"
    pd.DataFrame(
        [("solar", "capital_charge_rate", 0.045), ("solar", "annualized_capex_per_mw", 45000.0)],
        columns=["pypsa-name", "parameter", "value"],
    ).to_csv(path, index=False)
    with caplog.at_level(logging.WARNING):
        costs = load_costs(str(path), {"itc_modifier": {"solar": 0.3}})
    assert "Rebuild the cost tables" in caplog.text
    assert costs.at["solar", "capital_charge_rate"] == pytest.approx(0.045 * 0.7)


@pytest.mark.parametrize(
    "key,spec",
    [("itc_modifier", 1.0), ("itc_modifier", -0.1), ("ptc_modifier", -5.0)],
)
def test_out_of_range_values_raise(key, spec):
    with pytest.raises(ValueError, match=key):
        credit_for_year(spec, 2030, key, "solar")


def test_per_year_values_need_the_build_year():
    with pytest.raises(ValueError, match="build year"):
        credit_for_year({2030: 0.3}, None, "itc_modifier", "solar")
    assert credit_for_year(0.3, None, "itc_modifier", "solar") == 0.3


def test_ptc_is_levelized_like_a_ten_year_credit():
    pv = 27.5 * (1 - (1 + WACC) ** -PTC_YEARS) / WACC
    assert levelize_ptc(27.5, WACC, CRP) == pytest.approx(pv * calculate_annuity(CRP, WACC))
    assert levelize_ptc(27.5, 0.0, 20.0) == pytest.approx(27.5 * PTC_YEARS / 20.0)


def test_base_model_case():
    assert base_model_case("R&D + TC") == "R&D"
    assert base_model_case("Exp + TC") == "Exp"
    assert base_model_case("R&D") == "R&D"


def test_years_that_are_no_investment_period_warn(caplog):
    with caplog.at_level(logging.WARNING):
        check_tax_credit_years({"itc_modifier": {"solar": {2030: 0.3, 2045: 0.0}, "onwind": 0.3}}, [2030, 2040])
    assert "2045" in caplog.text
    assert "onwind" not in caplog.text


@pytest.fixture
def vintage_network():
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 03:00", "inclusive": "both"},
        invest_periods=[2025, 2030],
    )
    n.set_investment_periods(periods=[2025, 2030])
    n.add("Bus", "z1")
    for year in (2025, 2030):
        n.add("Generator", f"wind {year}", bus="z1", carrier="onwind", p_nom_extendable=True, build_year=year)
    n.add("Generator", "wind existing", bus="z1", carrier="onwind", p_nom=100, build_year=2010)
    n.add("Generator", "solar 2025", bus="z1", carrier="solar", p_nom_extendable=True, build_year=2025)
    # a time-varying marginal cost must be credited too
    n.generators_t.marginal_cost["wind 2030"] = 1.0
    return n


def test_each_vintage_claims_the_ptc_of_its_build_year(vintage_network):
    n = vintage_network
    costs_dict = {
        2025: pd.DataFrame({"ptc_per_mwh": [19.5, 0.0]}, index=["onwind", "solar"]),
        2030: pd.DataFrame({"ptc_per_mwh": [0.0, 0.0]}, index=["onwind", "solar"]),
    }
    n.generators_t.marginal_cost["wind 2025"] = 2.0
    apply_ptc(n, costs_dict)

    assert n.generators.at["wind 2025", "marginal_cost"] == pytest.approx(-19.5)
    assert np.allclose(n.generators_t.marginal_cost["wind 2025"], 2.0 - 19.5)
    assert (n.generators_t.marginal_cost["wind 2030"] == 1.0).all()  # credit expired
    assert n.generators.at["wind existing", "marginal_cost"] == 0.0  # existing fleet
    assert n.generators.at["solar 2025", "marginal_cost"] == 0.0  # ITC carrier
