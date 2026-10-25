"""Capital costs of output-rated heating and cooling technologies on the link input."""

import os
import sys

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from _helpers import get_multiindex_snapshots
from build_heat import add_air_cons, capex_per_input

pytestmark = pytest.mark.fast

AC = "Residential Central Air Conditioner"
COST_PER_MW_COOLING = 500.0
COP = 4.0


@pytest.fixture
def costs():
    return pd.DataFrame(
        {"capital_cost": [COST_PER_MW_COOLING], "efficiency": [COP], "lifetime": [20.0]},
        index=[AC],
    )


def test_capex_per_input_scales_by_efficiency(costs):
    assert capex_per_input(costs, AC) == COST_PER_MW_COOLING * COP


def test_air_conditioner_pays_rated_cost_per_mw_of_cooling(costs):
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 02:00", "inclusive": "both"},
        invest_periods=[2030],
    )
    n.set_investment_periods(periods=[2030])
    n.add("Bus", "p1 res-total-elec", carrier="res-total-elec")
    n.add("Bus", "p1 res-total-cool", carrier="res-total-cool")
    n.add("Load", "p1 res-total-cool", bus="p1 res-total-cool", carrier="res-total-cool", p_set=8.0)

    add_air_cons(n, "res", "total", costs)

    link = n.links.loc["p1 res-total-elec res-total-air-con"]
    assert link.bus0 == "p1 res-total-elec"
    assert link.efficiency == COP
    # one MW of electric input delivers COP MW of cooling at the rated cost per MW of cooling
    assert link.capital_cost / link.efficiency == pytest.approx(COST_PER_MW_COOLING)
