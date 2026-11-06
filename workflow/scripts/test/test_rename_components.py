"""
Test ``rename_components`` of ``add_extra_components``.

It hands renames to pypsa's ``rename_component_names`` wherever that works: always for
buses, and for every other component once pypsa includes PyPSA/PyPSA#2058 (after
1.4.0). With older pypsa versions it renames the component and its time series itself.
"""

import os
import sys

import pypsa
import pytest
from packaging.version import Version

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import add_extra_components
from add_extra_components import pypsa_renames_non_bus_components, rename_components

pytestmark = pytest.mark.fast


@pytest.fixture
def network(base_network):
    """The base network, whose lines, link and generators trip up older pypsa versions, with a time series on its link."""
    base_network.links_t.p_max_pu["link1"] = 0.9
    return base_network


def pypsa_can_rename(monkeypatch, can_rename: bool) -> None:
    monkeypatch.setattr(add_extra_components, "pypsa_renames_non_bus_components", lambda: can_rename)


# with whichever pypsa is installed


def test_generators_are_renamed_with_their_time_series(network):
    n = network
    rename_components(n, "Generator", {"wind1": "wind1 renamed"})

    assert "wind1" not in n.generators.index
    assert n.generators.at["wind1 renamed", "bus"] == "z1"
    assert "wind1" not in n.generators_t.p_max_pu.columns
    assert (n.generators_t.p_max_pu["wind1 renamed"] == 0.8).all()
    assert len(n.generators) == 6  # the others are left alone


def test_links_are_renamed_with_their_time_series(network):
    n = network
    rename_components(n, "Link", {"link1": "link1 renamed"})

    assert n.links.index.tolist() == ["link1 renamed"]
    assert n.links.loc["link1 renamed", ["bus0", "bus1"]].tolist() == ["z1", "z3"]
    assert n.links_t.p_max_pu.columns.tolist() == ["link1 renamed"]


@pytest.mark.skipif(
    Version(pypsa.__version__) > Version("1.4.0"),
    reason="only releases up to 1.4.0 are known to lack PyPSA/PyPSA#2058",
)
def test_pypsa_releases_without_the_fix_are_detected():
    assert not pypsa_renames_non_bus_components()


# with each pypsa behaviour forced


def test_older_pypsa_is_not_asked_to_rename_generators(network, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("rename_component_names cannot rename generators before PyPSA/PyPSA#2058")

    pypsa_can_rename(monkeypatch, False)
    monkeypatch.setattr(pypsa.Network, "rename_component_names", fail)
    n = network
    rename_components(n, "Generator", {"wind1": "wind1 renamed"})

    assert "wind1 renamed" in n.generators.index
    assert "wind1 renamed" in n.generators_t.p_max_pu.columns


def test_fixed_pypsa_renames_generators_itself(network, monkeypatch):
    calls = []
    pypsa_can_rename(monkeypatch, True)
    monkeypatch.setattr(
        pypsa.Network,
        "rename_component_names",
        lambda self, component, **names: calls.append((component, names)),
    )
    rename_components(network, "Generator", {"wind1": "wind1 renamed"})

    assert calls == [("Generator", {"wind1": "wind1 renamed"})]


def test_buses_are_renamed_by_pypsa_along_with_the_references_to_them(network, monkeypatch):
    pypsa_can_rename(monkeypatch, False)
    n = network
    rename_components(n, "Bus", {"z1": "z1 renamed"})

    assert "z1" not in n.buses.index
    assert n.generators.at["wind1", "bus"] == "z1 renamed"
    assert n.lines.at["line1", "bus0"] == "z1 renamed"
    assert n.links.at["link1", "bus0"] == "z1 renamed"
