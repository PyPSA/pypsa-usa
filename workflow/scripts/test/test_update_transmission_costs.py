"""Tests for update_transmission_costs in add_electricity."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from add_electricity import update_transmission_costs

HVDC_OVERHEAD = 40.0  # annualized $/MW/km/yr
HVDC_INVERTER_PAIR = 9000.0  # annualized $/MW/yr
LENGTH_KM = 300.0
CORRIDOR_COST = LENGTH_KM * HVDC_OVERHEAD + HVDC_INVERTER_PAIR


@pytest.fixture
def costs():
    return pd.DataFrame(
        {
            "annualized_capex_per_mw_km": [30.0, HVDC_OVERHEAD, 100.0, float("nan")],
            "annualized_capex_per_mw": [float("nan"), float("nan"), float("nan"), HVDC_INVERTER_PAIR],
        },
        index=["HVAC overhead", "HVDC overhead", "HVDC submarine", "HVDC inverter pair"],
    )


@pytest.fixture
def dc_network(base_network):
    """base_network plus a `_fwd`/`_rev` DC link pair and a one-way DC link."""
    n = base_network
    n.add(
        "Link",
        ["dc1_fwd", "dc1_rev", "dc2"],
        bus0=["z1", "z2", "z1"],
        bus1=["z2", "z1", "z3"],
        p_nom=1000.0,
        length=LENGTH_KM,
        underwater_fraction=0.0,
        carrier="DC",
    )
    return n


@pytest.mark.fast
def test_dc_link_pair_shares_the_corridor_cost(dc_network, costs):
    update_transmission_costs(dc_network, costs)
    capital_cost = dc_network.links.capital_cost

    assert capital_cost["dc1_fwd"] == pytest.approx(CORRIDOR_COST / 2)
    assert capital_cost["dc1_rev"] == pytest.approx(CORRIDOR_COST / 2)
    # a link without a counterpart carries the full cost
    assert capital_cost["dc2"] == pytest.approx(CORRIDOR_COST)
