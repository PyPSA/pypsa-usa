"""Unit tests for the EV charging limits in ``build_transportation.constrain_charing_rates``."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from build_transportation import constrain_charing_rates

pytestmark = pytest.mark.fast


@pytest.fixture
def road_network(base_network):
    """Light-duty EV and LPG links on z1 and z2, both feeding the vehicle load bus."""
    n = base_network
    nodes = pd.Index(["z1", "z2"])
    n.add("Bus", nodes, suffix=" trn-veh-lgt", carrier="trn-veh-lgt")
    demand = pd.DataFrame(
        {"z1 trn-veh-lgt": np.linspace(0, 10, len(n.snapshots)), "z2 trn-veh-lgt": 5.0},
        index=n.snapshots,
    )
    n.add("Load", nodes + " trn-veh-lgt", bus=nodes + " trn-veh-lgt", p_set=demand)
    for fuel in ("elec", "lpg"):
        n.add(
            "Link",
            nodes,
            suffix=f" trn-{fuel}-veh-lgt",
            bus0=nodes,
            bus1=nodes + " trn-veh-lgt",
            carrier=f"trn-{fuel}-veh-lgt",
        )
    return n


def test_must_run_evs_follow_their_load_profile(road_network):
    """Static frames are indexed by 'name' under pypsa 1.x, not by the component name."""
    constrain_charing_rates(road_network, must_run_evs=True)

    p_max_pu = road_network.links_t.p_max_pu
    p_min_pu = road_network.links_t.p_min_pu
    assert set(p_max_pu.columns) == {"z1 trn-elec-veh-lgt", "z2 trn-elec-veh-lgt"}
    assert set(p_min_pu.columns) == {"z1 trn-elec-veh-lgt", "z2 trn-elec-veh-lgt"}
    # rising demand on z1 is normalised to 0..1; the flat profile on z2 stays unconstrained
    assert p_max_pu["z1 trn-elec-veh-lgt"].iloc[[0, -1]].tolist() == [0.01, 1.0]
    assert (p_max_pu["z2 trn-elec-veh-lgt"] == 1.0).all()


def test_lpg_vehicles_are_capped_when_evs_are_not_must_run(road_network):
    constrain_charing_rates(road_network, must_run_evs=False)

    assert set(road_network.links_t.p_max_pu.columns) == {
        "z1 trn-elec-veh-lgt",
        "z2 trn-elec-veh-lgt",
        "z1 trn-lpg-veh-lgt",
        "z2 trn-lpg-veh-lgt",
    }
    assert road_network.links_t.p_min_pu.empty
