"""Line and DC link lengths in build_base_network."""

import os
import sys

import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from build_base_network import assign_line_length, assign_link_length_and_efficiency

pytestmark = pytest.mark.fast

# One degree of longitude on the equator, on a sphere of mean Earth radius (6371 km)
ONE_DEGREE_KM = 111.195


@pytest.fixture
def network():
    n = pypsa.Network()
    n.add("Bus", ["a", "b", "c"], x=[0.0, 1.0, 10.0], y=[0.0, 0.0, 0.0])
    n.add("Line", ["ab", "ac"], bus0="a", bus1=["b", "c"])
    n.add("Link", "ac_dc", bus0="a", bus1="c", carrier="DC")
    return n


def test_line_length_is_the_great_circle_distance(network):
    assign_line_length(network)
    assert network.lines.length.to_dict() == pytest.approx({"ab": ONE_DEGREE_KM, "ac": 10 * ONE_DEGREE_KM}, rel=1e-5)


def test_link_length_and_loss_follow_the_distance(network):
    assign_link_length_and_efficiency(network, length_factor=1.25)
    length = 10 * ONE_DEGREE_KM
    assert network.links.at["ac_dc", "length"] == pytest.approx(length, rel=1e-5)
    # 2 % inverter loss and 3 % per 1000 km of routed line
    assert network.links.at["ac_dc", "efficiency"] == pytest.approx(1 - 0.02 - 0.03 * length * 1.25 / 1000, rel=1e-6)
