"""Tests for the gas end uses that are not space or water heating."""

import os
import sys

import numpy as np
import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from build_heat import add_service_gas_other

pytestmark = pytest.mark.fast

CO2_INTENSITY = 0.18
LOAD = [1.0, 3.0, 2.0]
GAS_OTHER_BUS = "p1 0-res-gas-other"


@pytest.fixture
def gas_other_network():
    """One state with a gas supply, a CO2 sink and a residential gas-other load on its own bus."""
    n = pypsa.Network()
    n.set_snapshots(range(len(LOAD)))
    n.add("Bus", "CA gas", carrier="gas", STATE="CA")
    n.add("Bus", "CA res-co2", carrier="co2", STATE="CA")
    n.add("Bus", GAS_OTHER_BUS, carrier="res-gas-other", STATE="CA")
    n.add("Generator", "CA gas supply", bus="CA gas", carrier="gas", p_nom=10.0, marginal_cost=1.0)
    n.add("Store", "CA res-co2", bus="CA res-co2", carrier="co2", e_nom=1e3)
    n.add("Load", GAS_OTHER_BUS, bus=GAS_OTHER_BUS, carrier="res-gas-other", p_set=LOAD)
    costs = pd.DataFrame({"co2_emissions": [CO2_INTENSITY]}, index=["gas"])
    return n, costs


def test_gas_other_load_draws_state_gas_and_books_co2(gas_other_network):
    n, costs = gas_other_network
    add_service_gas_other(n, "res", costs)

    link = n.links.loc[f"{GAS_OTHER_BUS}-supply"]
    assert link.bus0 == "CA gas"
    assert link.bus1 == GAS_OTHER_BUS
    assert link.bus2 == "CA res-co2"
    assert link.efficiency == 1.0
    assert link.efficiency2 == CO2_INTENSITY
    assert link.p_nom == max(LOAD)
    assert not link.p_nom_extendable

    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"
    np.testing.assert_allclose(n.generators_t.p["CA gas supply"], LOAD)
    np.testing.assert_allclose(n.stores_t.e["CA res-co2"], np.cumsum(LOAD) * CO2_INTENSITY)


def test_no_gas_other_load_adds_nothing(gas_other_network):
    n, costs = gas_other_network
    add_service_gas_other(n, "com", costs)
    assert n.links.empty
