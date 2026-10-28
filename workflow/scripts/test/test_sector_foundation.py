"""
Test the state level fuel supply of ``add_sector_foundation``.

The supply store of a fuel can deliver ``e_nom`` (1e9 MWh) per investment
period: it starts every period full, rather than carrying over what earlier
periods drew from it.
"""

import os
import sys

import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from _helpers import get_multiindex_snapshots
from add_sectors import add_sector_foundation

PERIODS = [2030, 2040]
DEMAND = 0.6e9  # MWh_th per year, more than half the supply limit


@pytest.fixture
def coal_network():
    """A state coal bus over two one year periods, drawing 0.6e9 MWh_th each year."""
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 01:00", "inclusive": "both"},
        invest_periods=PERIODS,
    )
    n.set_investment_periods(periods=PERIODS)
    for col in n.snapshot_weightings.columns:  # each period stands for one year
        n.snapshot_weightings[col] = 8760 / 2

    n.add("Bus", "p1", carrier="AC", reeds_state="UT", STATE="UT")
    add_sector_foundation(n, "coal")
    # a cost is needed only so create_model can build an objective
    n.stores.loc["UT coal", "marginal_cost"] = 1.0
    n.add("Load", "UT coal", bus="UT coal", p_set=DEMAND / 8760)
    return n


def test_supply_limit_applies_to_each_period(coal_network):
    n = coal_network
    status, _ = n.optimize(solver_name="glpk", multi_investment_periods=True)

    assert status == "ok"
    drawn = n.stores_t.e["UT coal"].groupby(level="period").last()
    assert drawn.tolist() == pytest.approx([-DEMAND, -DEMAND])
