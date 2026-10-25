"""Water heater tank costs and the CO2 of existing fuel-fired water heaters."""

import os
import sys

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from _helpers import get_multiindex_snapshots
from build_cost_data import gallon_cost_factor
from build_heat import add_service_water_store
from build_stock_data import add_service_brownfield

pytestmark = pytest.mark.fast

GAS_WH = "Residential Gas-Fired Storage Water Heaters"
ELEC_WH = "Residential Electric-Resistance Storage Water Heaters"
OIL_WH = "Residential Oil-Fired Storage Water Heaters"
CO2 = {"gas": 0.18, "oil": 0.26}


def test_gallon_cost_is_per_mwh_held_across_the_test_temperature_rise():
    factor = gallon_cost_factor(pd.Series([GAS_WH, "Commercial Gas-Fired Storage Water Heaters"]))
    # a 40 gal residential tank heated 58 -> 125 F holds 6.55 kWh
    assert 40 / factor[0] * 1e3 == pytest.approx(6.55, abs=0.01)
    # commercial units are tested 70 -> 140 F
    assert factor[0] / factor[1] == pytest.approx(70 / 67)


@pytest.fixture
def costs():
    techs = pd.DataFrame(
        {"capital_cost": [400.0, 300.0, 500.0], "efficiency": [0.6, 0.9, 0.55], "lifetime": [15.0, 15.0, 15.0]},
        index=[GAS_WH, ELEC_WH, OIL_WH],
    )
    fuels = pd.DataFrame({"co2_emissions": list(CO2.values())}, index=list(CO2))
    return pd.concat([techs, fuels])


@pytest.fixture
def network():
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 02:00", "inclusive": "both"},
        invest_periods=[2030],
    )
    n.set_investment_periods(periods=[2030])
    for bus, carrier in [
        ("CA gas", "gas"),
        ("CA oil", "oil"),
        ("CA res-co2", "co2"),
        ("p1 res-total-elec", "res-total-elec"),
        ("p1 res-total-water-heat", "res-total-water-heat"),
    ]:
        n.add("Bus", bus, carrier=carrier, STATE="CA", x=0.0, y=0.0)
    n.add(
        "Load",
        "p1 res-total-water-heat",
        bus="p1 res-total-water-heat",
        carrier="res-total-water-heat",
        p_set=pd.Series(10.0, index=n.snapshots),
    )
    return n


def test_new_tank_cost_is_on_the_fuel_side(network, costs):
    add_service_water_store(network, "res", "total", "gas", costs, simple_storage=True, n_hours=4)
    discharger = network.links.loc["p1 res-total-water-gas-heater-discharger"]
    assert discharger.efficiency == 0.6
    # one MWh of stored fuel heats 0.6 MWh of water
    assert discharger.capital_cost == pytest.approx(400.0 * 0.6 / 4)


def test_existing_fuel_water_heaters_book_co2(network, costs):
    ratios = pd.DataFrame({"electricity": [40.0], "gas": [50.0], "lpg": [10.0]}, index=["CA"])
    add_service_brownfield(network, "res", "water_heating", 1.0, ratios, costs, simple_storage=True)

    links = network.links
    assert not links.empty
    for fuel in ("gas", "oil"):
        heaters = links[links.carrier == f"res-total-water-{fuel}"]
        assert not heaters.empty
        assert (heaters.bus0 == f"CA {fuel}").all()
        assert (heaters.bus2 == "CA res-co2").all()
        assert (heaters.efficiency2 == CO2[fuel]).all()
    electric = links[links.carrier == "res-total-water-elec"]
    assert not electric.empty
    assert (electric.bus2 == "").all()
