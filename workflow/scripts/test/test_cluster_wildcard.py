"""Suffix semantics of the ``{clusters}`` wildcard (cluster_network.parse_clusters_wildcard)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster_network import clustering_for_n_clusters, is_noop_clustering, parse_clusters_wildcard

ALL = {"onwind", "solar", "CCGT", "coal", "nuclear"}
CONV = {"CCGT", "coal", "nuclear"}


def _parse(wc, exclude=()):
    return parse_clusters_wildcard(wc, ALL, CONV, ALL - set(exclude), n_buses=500)


@pytest.mark.fast
@pytest.mark.parametrize("wc", ["33", "33m"])
def test_plain_and_m_aggregate_conventional_only(wc):
    n, agg, keep = _parse(wc)
    assert n == 33
    assert agg == CONV
    assert keep == {"onwind", "solar"}


@pytest.mark.fast
def test_s_aggregates_every_carrier():
    n, agg, keep = _parse("33s")
    assert (n, agg, keep) == (33, ALL, set())


@pytest.mark.fast
def test_a_aggregates_nothing():
    n, agg, keep = _parse("33a")
    assert (n, agg, keep) == (33, set(), ALL)


@pytest.mark.fast
def test_c_aggregates_all_but_conventional():
    n, agg, keep = _parse("33c")
    assert n == 33
    assert agg == {"onwind", "solar"}
    assert keep == CONV


@pytest.mark.fast
def test_all_keeps_every_bus():
    n, agg, keep = _parse("all")
    assert (n, agg, keep) == (500, ALL, set())


@pytest.mark.fast
def test_exclude_carriers_are_never_aggregated():
    _, agg, keep = _parse("33s", exclude={"nuclear"})
    assert "nuclear" not in agg and "nuclear" in keep


@pytest.mark.fast
def test_unknown_suffix_rejected():
    with pytest.raises(ValueError):
        _parse("33x")


@pytest.mark.fast
@pytest.mark.parametrize(
    ("wc", "n_clusters", "aggregate", "transport", "noop"),
    [
        ("all", 33, CONV, False, True),  # all keeps the network as it is
        ("33a", 33, set(), False, True),  # nothing to merge
        ("33", 33, CONV, False, False),  # same buses, but conventional generators are merged
        ("33s", 33, ALL, False, False),
        ("20", 33, CONV, False, False),  # fewer clusters than buses
        ("33a", 33, set(), True, False),  # the transport model always rebuilds the network
    ],
)
def test_noop_clustering_only_when_nothing_is_merged(wc, n_clusters, aggregate, transport, noop):
    assert is_noop_clustering(wc, n_clusters, 33, aggregate, transport) is noop


@pytest.mark.fast
def test_cluster_count_equal_to_bus_count_still_merges_generators():
    """An identity busmap keeps buses and lines and merges the generators of a carrier per bus."""
    import pypsa

    n = pypsa.Network()
    n.add("Bus", ["a", "b"], x=[0, 1], y=[0, 0], country="z", Pd=0.0, load_weight=1.0, LAF_state=0.5)
    n.add("Line", "ab", bus0="a", bus1="b", x=0.1, s_nom=100, length=10)
    n.add("Generator", ["ccgt1", "ccgt2"], bus="a", carrier="CCGT", p_nom=[100, 200], marginal_cost=[30, 40])
    n.add("Generator", "wind", bus="b", carrier="onwind", p_nom=50)

    clustering = clustering_for_n_clusters(n, 2, n.buses.index.to_series(), aggregate_carriers={"CCGT"})

    assert list(clustering.n.buses.index) == ["a", "b"]
    assert len(clustering.n.lines) == 1
    gens = clustering.n.generators
    assert (gens.carrier == "CCGT").sum() == 1
    assert gens.loc[gens.carrier == "CCGT", "p_nom"].sum() == pytest.approx(300)
    assert (gens.carrier == "onwind").sum() == 1
