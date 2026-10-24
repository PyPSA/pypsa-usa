"""Unit tests for plant VOM and new-build parameters.

- ``add_sectors.convert_generators_2_links`` replaces gas/coal/oil generators
  with links that draw from the state fuel bus, which prices the fuel. The link
  must still carry the plant's VOM.
- ``add_extra_components`` gives new units copied from an existing fleet the
  ATB efficiency and VOM, at the fleet's fuel price.
- ``add_electricity.apply_dynamic_pricing`` adds each plant's own VOM.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import add_extra_components as aec
from _helpers import get_multiindex_snapshots
from add_electricity import apply_dynamic_pricing
from add_sectors import convert_generators_2_links
from build_powerplants import filter_outliers_zscore
from solve_network import freeze_prior_periods

pytestmark = pytest.mark.fast

COSTS = pd.DataFrame(
    {
        "opex_variable_per_mwh": {"gas": 2.0, "CCGT-95CCS": 6.0, "solar": 0.0},
        "opex_fixed_per_kw": {"gas": 10.0, "CCGT-95CCS": 30.0, "solar": 20.0},
        "annualized_capex_fom": {"gas": 1e5, "CCGT-95CCS": 2e5, "solar": 5e4},
        "cost_recovery_period_years": {"gas": 30, "CCGT-95CCS": 30, "solar": 30},
        "lifetime": {"gas": 30, "CCGT-95CCS": 30, "solar": 30},
        "marginal_cost": {"gas": 30.0, "CCGT-95CCS": 40.0, "solar": 0.0},
        "efficiency": {"gas": 0.5, "CCGT-95CCS": 0.45, "solar": 0.0},
        "heat_rate_mmbtu_per_mwh": {"gas": 6.824, "CCGT-95CCS": 7.58, "solar": 0.0},
        "co2_emissions": {"gas": 0.2, "CCGT-95CCS": 0.01, "CCGT": 0.2, "solar": 0.0},
    },
)


# ---------------------------------------------------------------------------
# convert_generators_2_links
# ---------------------------------------------------------------------------


@pytest.fixture
def sector_network(base_network):
    """base_network with its gas plants as CCGTs, plus state fuel and co2 buses."""
    n = base_network
    n.buses["STATE"] = n.buses.reeds_state
    for state in ("CA", "TX"):
        n.add("Bus", f"{state} gas", carrier="gas")
        n.add("Bus", f"{state} pwr-co2", carrier="co2")
    n.generators.loc[["gas1", "gas2"], "carrier"] = "CCGT"
    n.generators.loc[["gas1", "gas2"], "efficiency"] = 0.5
    n.generators.loc[["gas1", "gas2"], "vom_cost"] = [3.0, 4.0]
    return n


def test_link_carries_vom_per_unit_of_fuel(sector_network):
    n = sector_network
    convert_generators_2_links(n, "CCGT", " gas", co2_intensity=0.2)

    links = n.links.loc[["gas1", "gas2"]]
    assert links.bus0.tolist() == ["CA gas", "TX gas"]
    # 3 and 4 $/MWh_el at 50 % efficiency is 1.5 and 2 $ per MWh of gas burnt
    assert links.marginal_cost.tolist() == pytest.approx([1.5, 2.0])


def test_time_varying_fuel_cost_is_not_copied(sector_network):
    n = sector_network
    # fuel + VOM per MWh_el, as dynamic fuel pricing writes it
    n.generators_t.marginal_cost["gas1"] = 25.0

    convert_generators_2_links(n, "CCGT", " gas")

    assert "gas1" not in n.links_t.marginal_cost.columns
    assert n.links.at["gas1", "marginal_cost"] == pytest.approx(1.5)


def test_plant_without_vom_cost_gets_zero_and_a_warning(sector_network, caplog):
    n = sector_network
    n.generators.loc["gas2", "vom_cost"] = np.nan

    convert_generators_2_links(n, "CCGT", " gas")

    assert n.links.at["gas2", "marginal_cost"] == 0
    assert "No VOM for 1 CCGT generators" in caplog.text


# ---------------------------------------------------------------------------
# new builds in add_extra_components
# ---------------------------------------------------------------------------


@pytest.fixture
def fleet_network(base_network):
    """base_network with plant efficiency and VOM on its existing gas fleet."""
    n = base_network
    n.generators["land_region"] = ""
    n.generators.loc[["gas1", "gas2"], "efficiency"] = 0.4
    n.generators.loc[["gas1", "gas2"], "vom_cost"] = [3.0, 4.0]
    return n


def _new_build_mc(fleet_mc, fleet_vom, fleet_eff=0.4, atb_eff=0.5, atb_vom=2.0):
    """Fleet fuel price per MWh_th, burnt at the ATB efficiency, plus ATB VOM."""
    return (fleet_mc - fleet_vom) * fleet_eff / atb_eff + atb_vom


def test_retirement_split_new_unit_uses_atb(fleet_network):
    n = fleet_network
    n.generators_t.marginal_cost["gas2"] = 20.0  # dynamic fuel price

    aec.split_retirement_gens(n, COSTS, ["gas"], economic=True)

    existing = n.generators.loc["gas2 existing"]
    assert (existing.efficiency, existing.vom_cost, existing.marginal_cost) == (0.4, 4.0, 18)
    assert (n.generators_t.marginal_cost["gas2 existing"] == 20.0).all()

    new = n.generators.loc["gas2"]
    assert (new.efficiency, new.vom_cost) == (0.5, 2.0)
    assert new.marginal_cost == pytest.approx(_new_build_mc(18, 4.0))
    assert n.generators_t.marginal_cost["gas2"].to_numpy() == pytest.approx(_new_build_mc(20.0, 4.0))


def test_technical_retirement_splits_fixed_existing_from_new_unit(fleet_network):
    n = fleet_network

    aec.split_retirement_gens(n, COSTS, ["gas"], economic=False)

    existing = n.generators.loc["gas2 existing"]
    assert (existing.p_nom, existing.p_nom_extendable, existing.efficiency) == (200, False, 0.4)
    new = n.generators.loc["gas2"]
    assert (new.p_nom, new.p_nom_extendable, new.efficiency, new.vom_cost) == (0, True, 0.5, 2.0)


@pytest.mark.parametrize(("retirement", "retirable"), [("economic", True), ("technical", False)])
def test_myopic_freeze_retires_existing_only_with_economic_retirement(fleet_network, retirement, retirable):
    n = fleet_network
    aec.split_retirement_gens(n, COSTS, ["gas"], economic=retirement == "economic")
    n.config = {"electricity": {"renewable_carriers": ["onwind", "solar"], "retirement": retirement}}
    n.generators["p_nom_opt"] = n.generators.p_nom

    freeze_prior_periods(n, 2030)

    existing = n.generators.loc["gas2 existing"]
    assert existing.p_nom_extendable == retirable
    assert existing.p_nom == 200
    assert not n.generators.at["gas2", "p_nom_extendable"]  # prior new build is frozen either way


def test_multihorizon_copies_use_atb_only_without_existing_capacity(fleet_network):
    n = fleet_network
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 23:00", "inclusive": "both"},
        invest_periods=[2030, 2040],
    )
    n.set_investment_periods(periods=[2030, 2040])
    gens = n.generators.loc[["gas2"]]

    for year in (2030, 2040):
        aec.attach_multihorizon_existing_generators(n, COSTS, gens, year)

    # 2030 carries the fleet's 200 MW, so keeps its parameters
    first = n.generators.loc["gas2 2030"]
    assert (first.p_nom, first.efficiency, first.vom_cost, first.marginal_cost) == (200, 0.4, 4.0, 18)
    later = n.generators.loc["gas2 2040"]
    assert (later.p_nom, later.efficiency, later.vom_cost) == (0, 0.5, 2.0)
    assert later.marginal_cost == pytest.approx(_new_build_mc(18, 4.0))


def test_new_parameters_are_idempotent(fleet_network):
    """Copying an already converted new unit again leaves its marginal cost."""
    n = fleet_network
    aec.split_retirement_gens(n, COSTS, ["gas"], economic=True)
    converted = n.generators.loc[["gas2"]].copy()

    n.add("Generator", "gas2 copy", bus="z3", carrier="gas", p_nom=0, marginal_cost=converted.marginal_cost.iloc[0])
    aec.apply_new_build_parameters(n, converted, pd.Index(["gas2 copy"]), COSTS)

    assert n.generators.at["gas2 copy", "marginal_cost"] == pytest.approx(converted.marginal_cost.iloc[0])


def test_renewable_copies_keep_fleet_values(fleet_network):
    n = fleet_network
    solar_mc = n.generators.at["solar1", "marginal_cost"]

    aec.split_retirement_gens(n, COSTS, ["solar"], economic=False)

    assert n.generators.at["solar1", "efficiency"] == 1
    assert n.generators.at["solar1", "marginal_cost"] == solar_mc


def test_new_carrier_generators_get_vom(fleet_network, monkeypatch):
    n = fleet_network
    monkeypatch.setattr(aec, "snakemake", SimpleNamespace(config={"costs": {}}), raising=False)
    n.add("Carrier", "CCGT")  # CO2 intensity is looked up on the base technology

    aec.attach_multihorizon_new_generators(n, COSTS, ["CCGT-95CCS"], 2030)

    new = n.generators[n.generators.carrier == "CCGT-95CCS"]
    assert len(new) == len(n.buses)
    assert (new.vom_cost == 6.0).all()


# ---------------------------------------------------------------------------
# dynamic fuel prices in add_electricity
# ---------------------------------------------------------------------------


def test_dynamic_pricing_adds_each_plants_own_vom(fleet_network):
    n = fleet_network
    n.generators.loc["gas2", "vom_cost"] = np.nan  # falls back to the passed VOM
    hours = n.snapshots.get_level_values(1)
    fuel = pd.DataFrame({"CA": 8.0, "TX": 10.0}, index=hours)  # $/MWh_th

    apply_dynamic_pricing(n, "gas", "reeds_state", fuel, vom=2.5)

    mc = n.generators_t.marginal_cost
    assert mc["gas1"].to_numpy() == pytest.approx(8.0 / 0.4 + 3.0)
    assert mc["gas2"].to_numpy() == pytest.approx(10.0 / 0.4 + 2.5)


def test_dynamic_pricing_matches_state_names_and_eia_balancing_areas(fleet_network):
    """The wholesale price tables use state names and EIA balancing authorities."""
    n = fleet_network
    n.generators.loc[["gas1", "gas2"], "ba_eia"] = ["CISO", np.nan]
    hours = n.snapshots.get_level_values(1)

    # coarse to fine, as add_electricity applies them
    apply_dynamic_pricing(n, "gas", "state", pd.DataFrame({"California": 8.0, "Texas": 10.0}, index=hours))
    apply_dynamic_pricing(n, "gas", "balancing_area", pd.DataFrame({"CISO": 6.0}, index=hours))

    mc = n.generators_t.marginal_cost
    assert mc["gas1"].to_numpy() == pytest.approx(6.0 / 0.4 + 3.0)  # balancing area overrides state
    assert mc["gas2"].to_numpy() == pytest.approx(10.0 / 0.4 + 4.0)  # no balancing area, state price


# ---------------------------------------------------------------------------
# PUDL fuel cost / heat rate outlier filter
# ---------------------------------------------------------------------------


def test_zscore_filter_keeps_generators_without_spread():
    """One month, or a constant series, cannot be scored; a real outlier is still dropped."""
    data = pd.DataFrame(
        {
            "generator_name": ["single"] + ["flat"] * 3 + ["spread"] * 12,
            "fuel_cost_per_mwh": [20.0] + [15.0] * 3 + [10.0] * 11 + [100.0],
        },
    )

    kept = filter_outliers_zscore(data, "fuel_cost_per_mwh")

    assert kept.groupby("generator_name").size().to_dict() == {"flat": 3, "single": 1, "spread": 11}
    assert kept.fuel_cost_per_mwh.max() == 20.0
