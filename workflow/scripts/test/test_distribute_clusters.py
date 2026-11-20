"""Allocation of clusters to regions in ``cluster_network`` (issues #421, #629)."""

import itertools
import os
import sys

import numpy as np
import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cluster_network import allocate_clusters, distribute_clusters

pytestmark = pytest.mark.fast


def _objective(allocation, target):
    return float(((np.asarray(allocation) - np.asarray(target)) ** 2).sum())


def _best_by_enumeration(target, n_clusters, n_nodes):
    """Lowest objective over every feasible whole-number allocation."""
    ranges = [range(1, upper + 1) for upper in n_nodes]
    feasible = (combo for combo in itertools.product(*ranges) if sum(combo) == n_clusters)
    return min(_objective(combo, target) for combo in feasible)


def test_allocation_follows_the_targets():
    target = pd.Series({"CA": 6.0, "TX": 3.0, "NY": 1.0})
    n_nodes = pd.Series({"CA": 50, "TX": 50, "NY": 50})

    allocation = allocate_clusters(target, 10, n_nodes)

    assert allocation.to_dict() == {"CA": 6, "TX": 3, "NY": 1}


def test_every_region_gets_a_cluster_and_none_more_than_its_buses():
    # NY's share rounds to zero and CA has too few buses for its share
    target = pd.Series({"CA": 9.6, "TX": 0.3, "NY": 0.1})
    n_nodes = pd.Series({"CA": 4, "TX": 50, "NY": 50})

    allocation = allocate_clusters(target, 10, n_nodes)

    assert allocation.sum() == 10
    assert allocation["CA"] == 4
    assert (allocation >= 1).all()


@pytest.mark.parametrize("seed", range(25))
def test_allocation_is_optimal(seed):
    """The greedy allocation reaches the optimum found by enumerating every allocation."""
    rng = np.random.default_rng(seed)
    regions = [f"r{i}" for i in range(rng.integers(2, 6))]
    n_nodes = pd.Series(rng.integers(1, 7, len(regions)), index=regions)
    n_clusters = int(rng.integers(len(regions), n_nodes.sum() + 1))
    shares = rng.dirichlet(np.ones(len(regions)))
    target = pd.Series(shares * n_clusters, index=regions)

    allocation = allocate_clusters(target, n_clusters, n_nodes)

    assert allocation.sum() == n_clusters
    assert ((allocation >= 1) & (allocation <= n_nodes)).all()
    assert _objective(allocation, target) == pytest.approx(_best_by_enumeration(target, n_clusters, n_nodes))


def test_an_impossible_cluster_count_is_rejected():
    target = pd.Series({"CA": 1.0, "TX": 1.0})
    n_nodes = pd.Series({"CA": 2, "TX": 2})
    with pytest.raises(ValueError, match="2 <= n_clusters <= 4"):
        allocate_clusters(target, 5, n_nodes)
    with pytest.raises(ValueError, match="2 <= n_clusters <= 4"):
        allocate_clusters(target, 1, n_nodes)


def test_clusters_are_distributed_by_load_weight_without_a_solver():
    """Runs with no solver installed: this used to need Gurobi or Ipopt through pyomo."""
    n = pypsa.Network()
    n.add("Bus", [f"ca{i}" for i in range(6)], country="CA")
    n.add("Bus", [f"tx{i}" for i in range(6)], country="TX")
    n.buses["load_weight"] = [3.0] * 6 + [1.0] * 6  # CA carries three quarters
    n.add(
        "Line",
        [f"ca_l{i}" for i in range(5)],
        bus0=[f"ca{i}" for i in range(5)],
        bus1=[f"ca{i + 1}" for i in range(5)],
        x=0.1,
    )
    n.add(
        "Line",
        [f"tx_l{i}" for i in range(5)],
        bus0=[f"tx{i}" for i in range(5)],
        bus1=[f"tx{i + 1}" for i in range(5)],
        x=0.1,
    )
    n.determine_network_topology()

    clusters = distribute_clusters(n, 8, weighting_strategy="population")

    assert clusters.groupby(level="country").sum().to_dict() == {"CA": 6, "TX": 2}
