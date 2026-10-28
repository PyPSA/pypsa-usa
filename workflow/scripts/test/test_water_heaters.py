"""Storage water heaters: the heater charges a tank sized by the energy to power ratio of the unit."""

import os
import sys

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from _helpers import get_multiindex_snapshots
from build_cost_data import size_water_heaters
from build_heat import add_service_water_store
from build_stock_data import add_service_brownfield
from opts.sector import add_water_heater_tank_constraints

pytestmark = pytest.mark.fast

GAS_WH = "Residential Gas-Fired Storage Water Heaters"
ELEC_WH = "Residential Electric-Resistance Storage Water Heaters"
OIL_WH = "Residential Oil-Fired Storage Water Heaters"
CO2 = {"gas": 0.18, "oil": 0.26}
RATIO = {GAS_WH: 0.5, ELEC_WH: 2.0, OIL_WH: 0.2}
STANDING_LOSS = 0.01


def test_water_heaters_are_priced_per_mw_of_input():
    rows = [
        (GAS_WH, "investment", 1216.0, "USD/unit"),
        (GAS_WH, "tank volume", 40.0, "gal"),
        (GAS_WH, "input capacity", 11.723, "kW"),
        ("Commercial Gas-Fired Storage Water Heaters", "investment", 6830.0, "USD/unit"),
        ("Commercial Gas-Fired Storage Water Heaters", "tank volume", 40.0, "gal"),
        ("Commercial Gas-Fired Storage Water Heaters", "input capacity", 11.723, "kW"),
        ("Residential Gas-Fired Furnaces", "investment", 1.0, "USD/MW"),
    ]
    df = pd.DataFrame(rows, columns=["technology", "parameter", "value", "unit"])
    df["source"], df["further_description"] = "", ""

    df = size_water_heaters(df).set_index(["technology", "parameter"])

    assert df.at[(GAS_WH, "investment"), "value"] == pytest.approx(1216.0 / 11.723e-3)
    assert df.at[(GAS_WH, "investment"), "unit"] == "USD/MW"
    # a 40 gal residential tank heated 58 -> 125 F holds 6.55 kWh
    assert df.at[(GAS_WH, "energy to power ratio"), "value"] == pytest.approx(6.55 / 11.723, abs=1e-3)
    # commercial units are tested 70 -> 140 F
    commercial = df.at[("Commercial Gas-Fired Storage Water Heaters", "energy to power ratio"), "value"]
    assert commercial / df.at[(GAS_WH, "energy to power ratio"), "value"] == pytest.approx(70 / 67)
    assert df.at[("Residential Gas-Fired Furnaces", "investment"), "value"] == 1.0


@pytest.fixture
def costs():
    techs = pd.DataFrame(
        {
            "capital_cost": [400.0, 300.0, 500.0],
            "efficiency": [0.6, 0.9, 0.55],
            "lifetime": [15.0, 15.0, 15.0],
            "energy to power ratio": list(RATIO.values()),
        },
        index=list(RATIO),
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
        p_set=pd.Series([10.0, 0.0, 20.0], index=n.snapshots),
    )
    return n


def add_water_heaters(n, costs):
    for fuel in ("gas", "elec", "oil"):
        add_service_water_store(n, "res", "total", fuel, costs, standing_loss=STANDING_LOSS)


def test_new_heater_charges_the_tank(network, costs):
    add_water_heaters(network, costs)

    charger = network.links.loc["p1 res-total-water-gas-heater-charger"]
    assert charger.bus1 == "p1 res-total-water-gas-heater"
    assert charger.efficiency == 0.6
    assert charger.capital_cost == 400.0
    assert charger.energy_to_power_ratio == 0.5
    assert (charger.bus2, charger.efficiency2) == ("CA res-co2", CO2["gas"])

    discharger = network.links.loc["p1 res-total-water-gas-heater-discharger"]
    assert discharger.efficiency == 1.0
    assert discharger.p_nom_extendable
    # it retires with the new heater, after the existing ones
    assert (discharger.build_year, discharger.lifetime) == (charger.build_year, charger.lifetime)

    tank = network.stores.loc["p1 res-total-water-gas-heater"]
    assert tank.bus == charger.bus1
    assert (tank.capital_cost, tank.standing_loss) == (0.0, STANDING_LOSS)


def test_existing_heaters_charge_their_own_tanks(network, costs):
    add_water_heaters(network, costs)
    ratios = pd.DataFrame({"electricity": [40.0], "gas": [50.0], "lpg": [10.0]}, index=["CA"])
    add_service_brownfield(network, "res", "water_heating", 1.0, ratios, costs, standing_loss=STANDING_LOSS)

    links = network.links[network.links.index.str.contains("existing")]
    stores = network.stores.loc[links.index.str.removesuffix("-charger")]
    assert not links.empty
    assert (stores.bus == links.bus1.to_numpy()).all()
    assert (stores.standing_loss == STANDING_LOSS).all()
    for fuel, tech in (("gas", GAS_WH), ("elec", ELEC_WH), ("oil", OIL_WH)):
        heaters = links[links.carrier == f"res-total-water-{fuel}"]
        assert not heaters.empty
        assert (heaters.bus1 == f"p1 res-total-water-{fuel}-heater").all()
        tanks = network.stores.loc[heaters.index.str.removesuffix("-charger")]
        assert tanks.e_nom.to_numpy() == pytest.approx(heaters.p_nom.to_numpy() * RATIO[tech])
        if fuel == "elec":
            assert (heaters.bus2 == "").all()
        else:
            assert (heaters.bus0 == f"CA {fuel}").all()
            assert (heaters.bus2 == "CA res-co2").all()
            assert (heaters.efficiency2 == CO2[fuel]).all()


def test_tank_is_sized_to_the_heater(network, costs):
    for fuel in ("gas", "elec"):
        add_service_water_store(network, "res", "total", fuel, costs, standing_loss=STANDING_LOSS)
    for bus in ("p1 res-total-elec", "CA gas"):
        network.add("Generator", bus, bus=bus, p_nom=100.0, build_year=2030, lifetime=20)
    # fuel is cheap in the first hour only, so the tanks are worth filling ahead
    network.generators_t.marginal_cost = pd.DataFrame(
        {bus: [1.0, 50.0, 50.0] for bus in network.generators.index},
        index=network.snapshots,
    )
    network.add("Store", "CA res-co2", bus="CA res-co2", e_nom_extendable=True)
    # both heaters are built, so the two tanks of different ratios are each tied to their own heater
    chargers = ["p1 res-total-water-gas-heater-charger", "p1 res-total-water-elec-heater-charger"]
    network.links.loc[chargers, "p_nom_min"] = 5.0

    status, condition = network.optimize(
        multi_investment_periods=True,
        solver_name="highs",
        extra_functionality=lambda n, sns: add_water_heater_tank_constraints(n),
    )
    assert (status, condition) == ("ok", "optimal")

    for fuel, tech in (("gas", GAS_WH), ("elec", ELEC_WH)):
        heater = network.links.at[f"p1 res-total-water-{fuel}-heater-charger", "p_nom_opt"]
        tank = network.stores.at[f"p1 res-total-water-{fuel}-heater", "e_nom_opt"]
        assert heater >= 5.0
        assert tank == pytest.approx(heater * RATIO[tech])
        # the nominal cost of the free draw settles it at the peak draw
        draw = f"p1 res-total-water-{fuel}-heater-discharger"
        assert network.links.at[draw, "p_nom_opt"] == pytest.approx(network.links_t.p0[draw].max())
