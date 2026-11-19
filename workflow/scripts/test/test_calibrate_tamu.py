"""Tests for calibrate_tamu_transmission_capacity in cluster_network."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pypsa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster_network import calibrate_tamu_transmission_capacity

HVAC_OVERHEAD_COST = 50.0  # annualized $/MW/km/yr


@pytest.fixture
def clustering():
    """Three state buses; the network has an A-B line but no A-C line."""
    n = pypsa.Network()
    n.add(
        "Bus",
        ["a", "b", "c"],
        x=[-100.0, -99.0, -100.0],
        y=[40.0, 40.0, 41.0],
        v_nom=230.0,
        reeds_state=["A", "B", "C"],
        interconnect="western",
    )
    n.add(
        "Line",
        "a-b",
        bus0="a",
        bus1="b",
        s_nom=500.0,
        length=85.0,
        r=1.0,
        x=10.0,
        b=1e-4,
        g=0.0,
    )
    return SimpleNamespace(n=n)


@pytest.fixture
def reeds_capacity_file(tmp_path):
    fn = tmp_path / "reeds_capacity.csv"
    pd.DataFrame(
        {
            "r": ["A", "A"],
            "rr": ["B", "C"],
            "MW_f0": [600.0, 1500.0],
            "MW_r0": [600.0, 1200.0],
        },
    ).to_csv(fn, index=False)
    return fn


@pytest.mark.fast
def test_added_reeds_line_capital_cost_is_per_mw(clustering, reeds_capacity_file):
    costs = pd.DataFrame(
        {"annualized_capex_per_mw_km": [HVAC_OVERHEAD_COST]},
        index=["HVAC overhead"],
    )
    calibrate_tamu_transmission_capacity(
        clustering,
        reeds_capacity_file,
        topological_boundaries="state",
        s_max_pu=0.7,
        length_factor=1.25,
        costs=costs,
        build_year=2025,
        lifetime=60,
    )

    line = clustering.n.lines.loc["REEDS_A_C"]
    assert line.s_nom == pytest.approx(1500.0 / 0.7)
    # capital_cost is per MW of capacity: it must not scale with s_nom
    assert line.capital_cost == pytest.approx(HVAC_OVERHEAD_COST * line.length)
