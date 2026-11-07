"""Unit tests for biomass power with 95% carbon capture (biomass-95CCS): costs, CO2 and sector models."""

import os
import sys

import numpy as np
import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import get_multiindex_snapshots
from add_extra_components import add_co2_emissions, add_co2_storage
from add_sectors import add_biomass_supply, convert_generators_2_links, get_period_fuel_price
from build_cost_data import ATB_VALUE_COLS, get_beccs_data, process_atb_scenario
from constants import BECCS_CO2_REMOVAL
from money import deflator

pytestmark = pytest.mark.fast


def test_beccs_data_in_the_model_dollar_year():
    # Sargent & Lundy Case 8, 2023 USD
    beccs = get_beccs_data(2023)
    assert beccs.capcost == pytest.approx(12631)
    assert beccs.fom == pytest.approx(261.18)
    assert beccs.vom == pytest.approx(9.65)
    assert beccs.heatrate == pytest.approx(19.965)

    # another dollar year changes the costs only
    later = get_beccs_data(2025)
    factor = deflator(2023, 2025)
    assert factor > 1
    assert later.capcost == pytest.approx(12631 * factor)
    assert later.fom == pytest.approx(261.18 * factor)
    assert later.vom == pytest.approx(9.65 * factor)
    assert later.heatrate == pytest.approx(19.965)


def test_beccs_removal_is_95_percent_of_the_base_co2_rate():
    # 206 lb/MMBtu
    assert BECCS_CO2_REMOVAL / 3.412 == pytest.approx(0.95 * 0.09344, abs=1e-5)


def test_beccs_costs_build_on_the_biomass_plant():
    atb = pd.DataFrame(
        {
            "pypsa-name": ["biomass", "solar"],
            "fuel_cost_per_mwh": [78.0, np.nan],  # USD/MWh_el
            "heat_rate_mmbtu_per_mwh": [13.3, np.nan],
            "capex_overnight_per_kw": [4000.0, 1000.0],
            "capex_grid_connection_per_kw": [100.0, 100.0],
            "capex_construction_finance_factor": [410.0, 50.0],  # USD/kW: 10% of OCC + GCC
            "capex_per_kw": [4510.0, 1150.0],
            "opex_variable_per_mwh": [5.0, 0.0],
            "opex_fixed_per_kw": [120.0, 20.0],
            "cost_recovery_period_years": [30.0, 30.0],
            "wacc_real": [0.05, 0.05],
        },
    )
    for col in ATB_VALUE_COLS:
        if col not in atb:
            atb[col] = np.nan
    aeo = pd.DataFrame({"pypsa-name": ["natural_gas"], "parameter": ["fuel_cost_real_per_mwhth"], "value": [15.0]})
    beccs = pd.Series({"capcost": 7000.0, "fom": 200.0, "vom": 20.0, "heatrate": 15.0})

    costs = process_atb_scenario(atb, aeo, beccs).pivot(index="pypsa-name", columns="parameter", values="value")

    row = costs.loc["biomass-95CCS"]
    # the overnight cost, which holds the grid connection, plus financing at the 10% share of biomass
    assert row.capex_construction_finance_factor == pytest.approx(700, abs=0.01)
    assert row.capex_per_kw == pytest.approx(7000 + 700, abs=0.01)
    assert row.capex_grid_connection_per_kw == 0
    assert row.opex_fixed_per_kw == pytest.approx(200.0)
    assert row.opex_variable_per_mwh == pytest.approx(20.0)
    assert row.efficiency == pytest.approx(3.412 / 15.0, abs=0.01)
    # it burns biomass at the ATB biomass fuel cost per MWh_th
    assert row.fuel_cost_real_per_mwhth == pytest.approx(costs.at["biomass", "fuel_cost_real_per_mwhth"])
    assert row.co2_emissions == pytest.approx(-BECCS_CO2_REMOVAL, abs=0.01)
    assert row.wacc_real == pytest.approx(0.05)
    # the lifetime and cost recovery period of the biomass plant
    for carrier in ["biomass", "biomass-95CCS"]:
        assert costs.at[carrier, "lifetime"] == 45.0
        assert costs.at[carrier, "cost_recovery_period_years"] == 30.0
    # without BECCS data there is no biomass-95CCS
    assert "biomass-95CCS" not in process_atb_scenario(atb, aeo)["pypsa-name"].values


def test_beccs_removes_co2_in_electricity_only_models():
    n = pypsa.Network()
    n.add("Carrier", ["biomass", "biomass-95CCS", "coal", "coal-95CCS"])
    costs = pd.DataFrame(
        {"co2_emissions": [0.0, -BECCS_CO2_REMOVAL, 0.3]},
        index=["biomass", "biomass-95CCS", "coal"],
    )

    add_co2_emissions(n, costs, ["biomass-95CCS", "coal-95CCS"])

    # the removal of the cost data, not 5% of the zero emissions of biomass
    assert n.carriers.at["biomass-95CCS", "co2_emissions"] == pytest.approx(-0.3029, abs=1e-4)
    assert n.carriers.at["coal-95CCS", "co2_emissions"] == pytest.approx(0.3 * 0.05)


def storage_config() -> dict:
    return {
        "plotting": {"tech_colors": {"co2": "#000000"}, "nice_names": {"co2": "CO2"}},
        "dac": {"granularity": "node"},
    }


@pytest.fixture
def storage_csv(tmp_path):
    fn = tmp_path / "co2_storage.csv"
    pd.DataFrame({"node": ["p1"], "potential [MtCO2]": [10], "cost [USD/tCO2]": [5]}).to_csv(fn, index=False)
    return str(fn)


def two_period_network() -> pypsa.Network:
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 01:00", "inclusive": "both"},
        invest_periods=[2030, 2040],
    )
    n.set_investment_periods(periods=[2030, 2040])
    return n


def test_beccs_sends_its_co2_to_storage(storage_csv):
    costs = pd.DataFrame({"co2_emissions": [0.2, 0.3]}, index=["gas", "coal"])

    # electricity only: per MWh of electricity, the atmosphere loses what the storage gains
    n = two_period_network()
    n.add("Bus", "p1", carrier="AC")
    n.add("Carrier", "biomass-95CCS", color="#000000", nice_name="Biomass CCS")
    n.add("Generator", "p1 biomass-95CCS_2030", bus="p1", carrier="biomass-95CCS", efficiency=0.25)
    add_co2_storage(n, storage_config(), storage_csv, costs, sector=False)
    cc = n.links.loc["p1 biomass-95CCS_2030"]
    assert cc.carrier == "biomass-95CCS"
    assert (cc.bus2, cc.bus3) == ("p1 atmosphere", "p1 co2 capture")
    assert cc.efficiency2 == pytest.approx(-BECCS_CO2_REMOVAL / 0.25)
    assert cc.efficiency3 == pytest.approx(BECCS_CO2_REMOVAL / 0.25)

    # sector models: per MWh of biomass, the power sector tracker loses what the storage gains
    n = two_period_network()
    n.add("Bus", ["p1", "UT biomass", "UT pwr-co2"])
    n.add("Carrier", "biomass-95CCS", color="#000000", nice_name="Biomass CCS")
    n.add(
        "Link",
        "p1 biomass-95CCS_2030",
        bus0="UT biomass",
        bus1="p1",
        bus2="UT pwr-co2",
        carrier="biomass-95CCS",
        efficiency=0.25,
        efficiency2=-BECCS_CO2_REMOVAL,
    )
    add_co2_storage(n, storage_config(), storage_csv, costs, sector=True)
    cc = n.links.loc["p1 biomass-95CCS_2030"]
    assert cc.carrier == "biomass-95CCS"
    assert cc.bus4 == "p1 co2 capture"
    assert cc.efficiency2 == pytest.approx(-BECCS_CO2_REMOVAL)
    assert cc.efficiency4 == pytest.approx(BECCS_CO2_REMOVAL)


def test_beccs_draws_state_biomass_in_sector_models():
    n = two_period_network()
    n.add("Bus", ["p1", "UT biomass", "UT pwr-co2"], carrier=["AC", "biomass", "co2"], STATE="UT")
    # as add_sector_foundation builds the state buses
    n.buses["STATE_NAME"] = "Utah"
    n.buses["interconnect"] = "western"
    n.add(
        "Generator",
        "p1 biomass-95CCS_2030",
        bus="p1",
        carrier="biomass-95CCS",
        p_nom_extendable=True,
        efficiency=0.25,
        capital_cost=1000,
        build_year=2030,
        lifetime=30,
    )
    n.generators["vom_cost"] = 20.0
    costs = {
        2030: pd.DataFrame({"fuel_cost_real_per_mwhth": [19.9]}, index=["biomass-95CCS"]),
        2040: pd.DataFrame({"fuel_cost_real_per_mwhth": [21.0]}, index=["biomass-95CCS"]),
    }

    add_biomass_supply(n, get_period_fuel_price(n, "biomass", costs, "biomass-95CCS"))
    convert_generators_2_links(n, "biomass-95CCS", " biomass", -BECCS_CO2_REMOVAL)

    supply = n.links.loc["UT biomass supply"]
    assert (supply.bus0, supply.bus1) == ("UT biomass supply", "UT biomass")
    assert n.links_t.marginal_cost["UT biomass supply"].groupby(level="period").first().tolist() == [19.9, 21.0]
    store = n.stores.loc["UT biomass supply"]
    assert (store.e_min_pu, store.e_max_pu, store.e_initial_per_period) == (-1, 0, True)

    beccs = n.links.loc["p1 biomass-95CCS_2030"]
    assert (beccs.bus0, beccs.bus1, beccs.bus2) == ("UT biomass", "p1", "UT pwr-co2")
    assert beccs.efficiency2 == pytest.approx(-BECCS_CO2_REMOVAL)
    # links are rated on the biomass drawn
    assert beccs.capital_cost == pytest.approx(1000 * 0.25)
    assert beccs.marginal_cost == pytest.approx(20 * 0.25)
    assert "p1 biomass-95CCS_2030" not in n.generators.index
