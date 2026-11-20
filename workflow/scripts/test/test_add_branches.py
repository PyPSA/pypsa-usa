"""Unit conversion of the TAMU/Breakthrough branch file in build_base_network."""

import os
import sys

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from build_base_network import add_branches_from_file

pytestmark = pytest.mark.fast


@pytest.fixture
def network(tmp_path):
    """Two 230 kV buses joined by a line and a transformer with the same per-unit impedance."""
    n = pypsa.Network()
    n.add("Bus", ["1", "2"], v_nom=230.0)
    branches = pd.DataFrame(
        {
            "from_bus_id": ["1", "1"],
            "to_bus_id": ["2", "2"],
            "r": [0.002, 0.002],
            "x": [0.05, 0.05],
            "b": [0.4, 0.4],
            "rateA": [250.0, 250.0],
            "branch_device_type": ["Line", "Transformer"],
            "interconnect": ["Western", "Western"],
        },
        index=pd.Index(["l", "t"], name="branch_id"),
    )
    fn = tmp_path / "branch.csv"
    branches.to_csv(fn)
    return add_branches_from_file(n, str(fn))


def test_line_impedance_is_in_ohm(network):
    line = network.lines.loc["l"]
    assert line.x == pytest.approx(0.05 * 230**2 / 100)
    assert line.r == pytest.approx(0.002 * 230**2 / 100)
    assert line.b == pytest.approx(0.4 / 230**2)


def test_transformer_impedance_is_per_unit_of_its_rating(network):
    """Pypsa reads transformer r, x and b in per unit of s_nom, not of the 100 MVA system base (#742)."""
    trafo = network.transformers.loc["t"]
    assert trafo.s_nom == 250.0
    assert trafo.x == pytest.approx(0.05 * 250 / 100)
    assert trafo.r == pytest.approx(0.002 * 250 / 100)
    assert trafo.b == pytest.approx(0.4 * 100 / 250)
    assert trafo.type == ""
    assert trafo.model == "pi"


def test_line_and_transformer_agree_in_power_flow(network):
    """The same per-unit impedance must give the same reactance once pypsa converts both."""
    n = network
    n.lines["type"] = ""  # the placeholder type is replaced later in the script
    n.calculate_dependent_values()
    assert n.transformers.at["t", "x_pu"] == pytest.approx(n.lines.at["l", "x_pu"])
    assert n.transformers.at["t", "r_pu"] == pytest.approx(n.lines.at["l", "r_pu"])
