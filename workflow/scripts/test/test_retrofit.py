"""Unit tests for the retrofits of existing gas plants to hydrogen turbines and carbon capture."""

import os
import sys

import numpy as np
import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import calculate_annuity, get_multiindex_snapshots
from add_extra_components import add_co2_emissions, attach_retrofits
from add_sectors import convert_generators_2_links, get_pwr_co2_intensity
from build_cost_data import (
    ATB_VALUE_COLS,
    get_gas_ccs_retrofit_cost,
    process_atb_scenario,
)
from money import deflator
from opts.retrofit import add_retrofit_constraints

pytestmark = pytest.mark.fast

CCS_RETROFIT_COSTS = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "repo_data",
    "costs",
    "reeds_gas_ccs_retrofit_costs.csv",
)
RETROFIT = {target: {"enable": True, "lifetime": 20} for target in ("OCHT", "CCHT", "CCGT-90CCS")}


def test_gas_ccs_retrofit_cost_of_the_cost_year_in_the_model_dollar_year():
    # ReEDS atb22_1p2_moderate in 2030, 2004 USD/kW
    cost = get_gas_ccs_retrofit_cost(CCS_RETROFIT_COSTS, 2030, 2023)
    assert cost == pytest.approx(953.613375 * deflator(2004, 2023))
    assert get_gas_ccs_retrofit_cost(CCS_RETROFIT_COSTS, 2030, 2025) == pytest.approx(cost * deflator(2023, 2025))
    # held at 2050 beyond the ReEDS years
    assert get_gas_ccs_retrofit_cost(CCS_RETROFIT_COSTS, 2060, 2023) == pytest.approx(
        652.0679573 * deflator(2004, 2023)
    )


def test_retrofit_costs_are_set_on_the_technologies_retrofitted_to():
    atb = pd.DataFrame(
        {
            "pypsa-name": ["OCGT", "CCGT", "CCGT-95CCS", "solar"],
            "heat_rate_mmbtu_per_mwh": [9.7, 6.2, 7.0, np.nan],
            "capex_overnight_per_kw": [1000.0, 1200.0, 2300.0, 1000.0],
            "capex_grid_connection_per_kw": [100.0, 100.0, 100.0, 100.0],
            "capex_construction_finance_factor": [100.0, 120.0, 230.0, 50.0],
            "capex_per_kw": [1200.0, 1420.0, 2630.0, 1150.0],
            "opex_variable_per_mwh": [8.0, 2.0, 4.0, 0.0],
            "opex_fixed_per_kw": [26.0, 32.0, 60.0, 20.0],
            "cost_recovery_period_years": [30.0, 30.0, 30.0, 30.0],
            "wacc_real": [0.05, 0.05, 0.05, 0.05],
        },
    )
    for col in ATB_VALUE_COLS:
        if col not in atb:
            atb[col] = np.nan
    aeo = pd.DataFrame({"pypsa-name": ["natural_gas"], "parameter": ["fuel_cost_real_per_mwhth"], "value": [15.0]})

    costs = process_atb_scenario(atb, aeo, None, 1500.0).pivot(index="pypsa-name", columns="parameter", values="value")

    # ReEDS: 33% and 28% of the overnight cost of the gas plant, not of the hydrogen turbine
    assert costs.at["OCHT", "retrofit_capex_per_kw"] == pytest.approx(0.33 * 1000)
    assert costs.at["CCHT", "retrofit_capex_per_kw"] == pytest.approx(0.28 * 1200)
    # ReEDS: carbon capture retrofits capture 90%, with the heat rate and O&M of the new plant capturing 95%
    assert costs.at["CCGT-90CCS", "retrofit_capex_per_kw"] == pytest.approx(1500)
    for attr in ("efficiency", "opex_fixed_per_kw", "opex_variable_per_mwh", "fuel_cost_real_per_mwhth", "wacc_real"):
        assert costs.at["CCGT-90CCS", attr] == pytest.approx(costs.at["CCGT-95CCS", attr], nan_ok=True)
    assert costs[["retrofit_capex_per_kw"]].loc[["OCGT", "CCGT", "CCGT-95CCS", "solar"]].isna().all().all()
    # without the CCS retrofit cost there are no retrofit costs
    no_retrofits = process_atb_scenario(atb, aeo)
    assert "retrofit_capex_per_kw" not in no_retrofits.parameter.values
    assert "CCGT-90CCS" not in no_retrofits["pypsa-name"].values


def retrofit_costs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "efficiency": [0.35, 0.35, 0.55, 0.55, 0.5],
            "opex_fixed_per_kw": [26.0, 26.0, 32.0, 32.0, 60.0],
            "opex_variable_per_mwh": [8.0, 8.0, 2.0, 2.0, 4.0],
            "fuel_cost_real_per_mwhth": [15.0, 68.24, 15.0, 68.24, 15.0],
            "wacc_real": [0.05, 0.05, 0.05, 0.05, 0.05],
            "cost_recovery_period_years": [30.0, 30.0, 30.0, 30.0, 30.0],
            "retrofit_capex_per_kw": [np.nan, 330.0, np.nan, 336.0, 1500.0],
        },
        index=["OCGT", "OCHT", "CCGT", "CCHT", "CCGT-90CCS"],
    )


def two_period_network() -> pypsa.Network:
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 01:00", "inclusive": "both"},
        invest_periods=[2030, 2040],
    )
    n.set_investment_periods(periods=[2030, 2040])
    n.add("Bus", "p1", carrier="AC")
    # the OCGT retires in 2035, the CCGT is extendable for economic retirement
    n.add(
        "Generator",
        ["p1 OCGT existing", "p1 CCGT existing", "p1 CCGT"],
        bus="p1",
        carrier=["OCGT", "CCGT", "CCGT"],
        p_nom=[100.0, 200.0, 0.0],
        p_nom_max=[np.inf, 200.0, np.inf],
        p_nom_extendable=[False, True, True],
        efficiency=[0.3, 0.5, 0.55],
        marginal_cost=[15 / 0.3 + 5, 15 / 0.5 + 3, 15 / 0.55 + 2],
        capital_cost=[90000.0, 32000.0, 100000.0],
        build_year=[1990, 2000, 2030],
        lifetime=[45, 55, 55],
    )
    n.generators["vom_cost"] = [5.0, 3.0, 2.0]
    return n


def test_existing_plants_get_a_retrofit_in_every_period_they_are_active_in():
    n = two_period_network()
    retrofit = RETROFIT | {"CCGT-90CCS": {"enable": True, "lifetime": 40}}
    attach_retrofits(n, {2030: retrofit_costs(), 2040: retrofit_costs()}, retrofit)

    retrofits = n.generators[n.generators.retrofit_from.fillna("") != ""]
    assert sorted(retrofits.index) == [
        "p1 CCGT-90CCS retrofit_2030",
        "p1 CCGT-90CCS retrofit_2040",
        "p1 CCHT retrofit_2030",
        "p1 CCHT retrofit_2040",
        "p1 OCHT retrofit_2030",
    ]
    # only existing plants are retrofitted, not new ones
    assert set(retrofits.retrofit_from) == {"p1 OCGT existing", "p1 CCGT existing"}

    ccs = n.generators.loc["p1 CCGT-90CCS retrofit_2040"]
    efficiency = 0.5 * 0.5 / 0.55
    assert (ccs.carrier, ccs.bus, ccs.build_year, ccs.lifetime) == ("CCGT-90CCS", "p1", 2040, 40)
    assert ccs.efficiency == pytest.approx(efficiency)
    # the plant's fuel, burnt at the retrofit's efficiency, plus its VOM
    assert ccs.marginal_cost == pytest.approx(15 / efficiency + 4)
    # recovered over the ATB cost recovery period, shorter than its lifetime
    assert ccs.capital_cost == pytest.approx((calculate_annuity(30, 0.05) * 1500 + 60) * 1e3)
    # at most the plant's fuel input
    assert ccs.p_nom_max / ccs.efficiency == pytest.approx(200 / 0.5)
    assert ccs.p_nom_extendable

    ocht = n.generators.loc["p1 OCHT retrofit_2030"]
    assert ocht.efficiency == pytest.approx(0.3)
    # hydrogen of the cost data
    assert ocht.marginal_cost == pytest.approx(68.24 / 0.3 + 8)
    assert ocht.capital_cost == pytest.approx((calculate_annuity(20, 0.05) * 330 + 26) * 1e3)


def test_plants_that_can_be_retrofitted_retire_economically():
    n = two_period_network()
    attach_retrofits(n, {2030: retrofit_costs(), 2040: retrofit_costs()}, {"OCHT": RETROFIT["OCHT"]})

    plant = n.generators.loc["p1 OCGT existing"]
    assert plant.p_nom_extendable
    assert (plant.p_nom_min, plant.p_nom_max) == (0, 100)
    assert plant.capital_cost == pytest.approx(26e3)  # its fixed O&M
    # CCGT retrofits are disabled
    assert not n.generators.carrier.isin(["CCHT", "CCGT-90CCS"]).any()


def test_no_retrofits_unless_enabled():
    n = two_period_network()
    disabled = {target: options | {"enable": False} for target, options in RETROFIT.items()}
    attach_retrofits(n, {2030: retrofit_costs(), 2040: retrofit_costs()}, disabled)

    assert "retrofit_from" not in n.generators
    assert not n.generators.loc["p1 OCGT existing", "p_nom_extendable"]


def test_retrofits_keep_their_plant_as_links():
    n = two_period_network()
    n.buses["STATE"] = "UT"
    n.add("Bus", ["UT gas", "UT pwr-co2"])
    attach_retrofits(n, {2030: retrofit_costs(), 2040: retrofit_costs()}, RETROFIT)

    convert_generators_2_links(n, "CCGT-90CCS", " gas", 0.01)

    link = n.links.loc["p1 CCGT-90CCS retrofit_2030"]
    assert (link.retrofit_from, link.retrofit_factor) == ("p1 CCGT existing", 1)
    assert link.p_nom_max == pytest.approx(200 / 0.5)  # rated on the plant's fuel input


@pytest.mark.parametrize("as_link", [False, True])
def test_retrofits_share_the_plant_fuel_input(as_link):
    n = pypsa.Network()
    n.set_snapshots(pd.MultiIndex.from_product([[2030], pd.date_range("2030-01-01", periods=2, freq="h")]))
    n.set_investment_periods([2030])
    n.add("Bus", ["p1", "H2"])
    n.add("Load", "p1", bus="p1", p_set=180)
    # the plant may retire; its retrofit is cheaper to run but less efficient
    n.add(
        "Generator",
        "p1 CCGT existing",
        bus="p1",
        p_nom=200,
        p_nom_max=200,
        p_nom_extendable=True,
        efficiency=0.5,
        marginal_cost=100,
        capital_cost=1,
    )
    if as_link:
        # as in sector models: rated on its fuel input
        n.add("Generator", "H2 supply", bus="H2", p_nom=1e4)
        n.add(
            "Link",
            "p1 CCHT retrofit_2030",
            bus0="H2",
            bus1="p1",
            efficiency=0.4,
            p_nom_extendable=True,
            marginal_cost=10 * 0.4,
            capital_cost=1,
        )
        n.links["retrofit_from"] = "p1 CCGT existing"
        n.links["retrofit_factor"] = 1.0
    else:
        n.add(
            "Generator",
            "p1 CCHT retrofit_2030",
            bus="p1",
            efficiency=0.4,
            p_nom_extendable=True,
            marginal_cost=10,
            capital_cost=1,
            retrofit_from="p1 CCGT existing",
        )

    def extra_functionality(n, sns):
        add_retrofit_constraints(n)

    status, _ = n.optimize(solver_name="highs", multi_investment_periods=True, extra_functionality=extra_functionality)
    assert status == "ok"

    # fuel input: plant / 0.5 + retrofit / 0.4 <= 200 / 0.5, with plant + retrofit = 180
    plant = n.generators.at["p1 CCGT existing", "p_nom_opt"]
    retrofit_input = n.links.at["p1 CCHT retrofit_2030", "p_nom_opt"] if as_link else None
    retrofit = retrofit_input * 0.4 if as_link else n.generators.at["p1 CCHT retrofit_2030", "p_nom_opt"]
    assert plant == pytest.approx(100, abs=1e-3)
    assert retrofit == pytest.approx(80, abs=1e-3)
    assert plant / 0.5 + retrofit / 0.4 == pytest.approx(400, abs=1e-3)


def test_carbon_capture_retrofits_capture_90_percent():
    costs = pd.DataFrame({"co2_emissions": [0.2]}, index=["gas"])
    assert get_pwr_co2_intensity("CCGT-90CCS", costs) == pytest.approx(0.2 * 0.1)

    n = pypsa.Network()
    n.add("Carrier", ["CCGT", "CCGT-90CCS"])
    add_co2_emissions(n, pd.DataFrame({"co2_emissions": [0.2]}, index=["CCGT"]), ["CCGT-90CCS"])
    assert n.carriers.at["CCGT-90CCS", "co2_emissions"] == pytest.approx(0.2 * 0.1)
