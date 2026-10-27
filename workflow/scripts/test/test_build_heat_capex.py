"""Capital costs of output-rated heating and cooling technologies on the link input."""

import os
import sys

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from _helpers import get_multiindex_snapshots
from build_heat import add_air_cons, add_service_heat_pumps, capex_per_input

pytestmark = pytest.mark.fast

AC = "Residential Central Air Conditioner"
ASHP = "Residential Air-Source Heat Pump"
COST_PER_MW_COOLING = 500.0
COP = 4.0


@pytest.fixture
def costs():
    return pd.DataFrame(
        {"capital_cost": [COST_PER_MW_COOLING] * 2, "efficiency": [COP] * 2, "lifetime": [20.0] * 2},
        index=[AC, ASHP],
    )


def _network(load_carrier: str) -> pypsa.Network:
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 02:00", "inclusive": "both"},
        invest_periods=[2030],
    )
    n.set_investment_periods(periods=[2030])
    n.add("Bus", "p1 res-total-elec", carrier="res-total-elec")
    n.add("Bus", f"p1 {load_carrier}", carrier=load_carrier)
    n.add("Load", f"p1 {load_carrier}", bus=f"p1 {load_carrier}", carrier=load_carrier, p_set=8.0)
    return n


def test_capex_per_input_scales_by_efficiency(costs):
    assert capex_per_input(costs, AC) == COST_PER_MW_COOLING * COP


def test_air_conditioner_pays_rated_cost_per_mw_of_cooling(costs):
    n = _network("res-total-cool")

    add_air_cons(n, "res", "total", costs)

    link = n.links.loc["p1 res-total-elec res-total-air-con"]
    assert link.bus0 == "p1 res-total-elec"
    assert link.efficiency == COP
    # one MW of electric input delivers COP MW of cooling at the rated cost per MW of cooling
    assert link.capital_cost / link.efficiency == pytest.approx(COST_PER_MW_COOLING)


def test_heat_pump_pays_rated_cost_per_mw_of_capacity(costs):
    n = _network("res-total-heat")

    add_service_heat_pumps(n, "res", "total", "heat", "air", costs)

    link = n.links.loc["p1 res-total-ashp"]
    assert link.bus0 == "p1 res-total-elec"
    assert link.efficiency == COP
    # no discount on top of the rated cost
    assert link.capital_cost / link.efficiency == pytest.approx(COST_PER_MW_COOLING)
