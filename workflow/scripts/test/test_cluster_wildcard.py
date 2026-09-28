"""Suffix semantics of the ``{clusters}`` wildcard (cluster_network.parse_clusters_wildcard)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster_network import parse_clusters_wildcard  # noqa: E402

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
