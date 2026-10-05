"""Unit tests for cluster_simpl helpers."""

import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pypsa
from cluster_simpl import (
    build_county_busmap,
    resolve_simpl_mode,
)


def test_resolve_simpl_mode_identity():
    assert resolve_simpl_mode("") == "identity"


def test_resolve_simpl_mode_county():
    assert resolve_simpl_mode("county") == "county"


def test_resolve_simpl_mode_kmeans_digits():
    assert resolve_simpl_mode("50") == "kmeans"


def test_resolve_simpl_mode_kmeans_large_digits():
    assert resolve_simpl_mode("2000") == "kmeans"


def test_resolve_simpl_mode_unknown_raises():
    with pytest.raises(ValueError, match="Unknown simpl wildcard"):
        resolve_simpl_mode("foo")


def test_resolve_simpl_mode_unknown_lists_sentinels():
    """Error message must list the recognized values so users can self-correct."""
    with pytest.raises(ValueError) as exc:
        resolve_simpl_mode("bar")
    msg = str(exc.value)
    assert '""' in msg
    assert '"county"' in msg
    assert "digits" in msg or "integer" in msg.lower()


@pytest.fixture
def substation_network():
    """Tiny 4-bus substation-level network with the columns cluster_simpl expects.

    Mirrors what aggregate_to_substations produces under topological_boundaries='county':
    every bus has reeds_zone, county (FIPS), Pd, load_weight, LAF_state. No loads
    or generators
    (cluster_simpl runs before add_electricity).
    """
    n = pypsa.Network()
    n.add("Bus", "s1", x=-122.0, y=37.0, carrier="AC")
    n.add("Bus", "s2", x=-122.1, y=37.1, carrier="AC")
    n.add("Bus", "s3", x=-118.0, y=34.0, carrier="AC")
    n.add("Bus", "s4", x=-118.1, y=34.1, carrier="AC")

    n.buses["country"] = ["06001", "06001", "06037", "06037"]
    n.buses["county"] = ["06001", "06001", "06037", "06037"]
    n.buses["reeds_zone"] = ["p9", "p9", "p10", "p10"]
    n.buses["reeds_state"] = ["CA", "CA", "CA", "CA"]
    n.buses["interconnect"] = "western"
    n.buses["Pd"] = [100.0, 200.0, 150.0, 250.0]
    n.buses["load_weight"] = [100.0, 200.0, 150.0, 250.0]
    n.buses["LAF_state"] = [0.25, 0.25, 0.25, 0.25]
    n.buses["substation_lv"] = True

    n.add("Line", "l1", bus0="s1", bus1="s2", x=0.01, r=0.001, s_nom=500)
    n.add("Line", "l2", bus0="s3", bus1="s4", x=0.01, r=0.001, s_nom=500)
    n.add("Line", "l3", bus0="s2", bus1="s3", x=0.05, r=0.005, s_nom=300)

    n.add("Carrier", "AC", co2_emissions=0)
    return n


def test_build_county_busmap_happy_path(substation_network):
    busmap = build_county_busmap(substation_network)
    assert list(busmap.index) == ["s1", "s2", "s3", "s4"]
    assert busmap.tolist() == ["p9_06001", "p9_06001", "p10_06037", "p10_06037"]


def test_build_county_busmap_unique_cluster_count(substation_network):
    busmap = build_county_busmap(substation_network)
    assert busmap.nunique() == 2


def test_build_county_busmap_missing_county_column_raises(substation_network):
    substation_network.buses = substation_network.buses.drop(columns=["county"])
    with pytest.raises(ValueError, match="county"):
        build_county_busmap(substation_network)


def test_build_county_busmap_missing_county_error_mentions_topological_boundaries(substation_network):
    """Error must steer users toward fixing model_topology.topological_boundaries."""
    substation_network.buses = substation_network.buses.drop(columns=["county"])
    with pytest.raises(ValueError, match="topological_boundaries"):
        build_county_busmap(substation_network)


def test_build_county_busmap_nan_county_raises(substation_network):
    substation_network.buses.loc["s2", "county"] = None
    with pytest.raises(ValueError, match="county"):
        build_county_busmap(substation_network)
