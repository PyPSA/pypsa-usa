"""
Test the bidirectional link constraints.

Every extendable '_fwd' / '_rev' link pair must expand by the same amount,
including the vintaged pairs ('_fwd_<horizon>' / '_rev_<horizon>') that
cluster_network.add_itls creates for each future planning horizon.
"""

import os
import sys

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from opts.bidirectional_link import add_bidirectional_link_constraints

# Fixtures


@pytest.fixture
def itl_network():
    """Two zones joined by one ITL with a base pair, a 2040 vintage pair and a lone fwd link."""
    n = pypsa.Network()
    n.set_snapshots(pd.RangeIndex(2))
    n.add("Bus", ["p1", "p2"])
    n.add("Load", "load", bus="p2", p_set=100)
    n.add("Generator", "gen", bus="p1", p_nom=1000, marginal_cost=10)

    links = {
        "p1||p2_fwd": ("p1", "p2", 300.0),
        "p1||p2_rev": ("p2", "p1", 200.0),
        "p1||p2_fwd_2040": ("p1", "p2", 0.0),
        "p1||p2_rev_2040": ("p2", "p1", 0.0),
        "p1||p3_fwd": ("p1", "p2", 50.0),  # no matching _rev
        "p1||p2_exp": ("p1", "p2", 0.0),  # not a directional name
    }
    for name, (bus0, bus1, p_nom) in links.items():
        n.add("Link", name, bus0=bus0, bus1=bus1, p_nom=p_nom, p_nom_min=p_nom, p_nom_extendable=True, carrier="AC")
    return n


def bidirectional_constraints(n):
    return sorted(c for c in n.model.constraints if c.startswith("bidirectional_link_"))


# Tests


def test_base_and_vintaged_pairs_are_constrained(itl_network):
    """The base pair and the vintaged pair each get their own constraint; nothing else does."""
    n = itl_network
    n.optimize.create_model()
    add_bidirectional_link_constraints(n)

    assert bidirectional_constraints(n) == ["bidirectional_link_p1||p2", "bidirectional_link_p1||p2_2040"]


@pytest.mark.parametrize(
    "name,fwd,rev",
    [
        ("bidirectional_link_p1||p2", "p1||p2_fwd", "p1||p2_rev"),
        ("bidirectional_link_p1||p2_2040", "p1||p2_fwd_2040", "p1||p2_rev_2040"),
    ],
)
def test_constraint_ties_expansion_within_a_vintage(itl_network, name, fwd, rev):
    """fwd.p_nom_opt - rev.p_nom_opt == fwd.p_nom - rev.p_nom, on the links of that vintage only."""
    n = itl_network
    n.optimize.create_model()
    add_bidirectional_link_constraints(n)

    con = n.model.constraints[name]
    labels = n.model.variables["Link-p_nom"].labels.to_series()
    coeffs = dict(zip(con.vars.values.ravel(), con.coeffs.values.ravel()))

    assert coeffs == {labels[fwd]: 1.0, labels[rev]: -1.0}
    assert float(con.rhs) == n.links.p_nom[fwd] - n.links.p_nom[rev]


def test_non_extendable_vintage_is_skipped(itl_network):
    """A vintage pair that prepare_network left non-extendable has no variable to constrain."""
    n = itl_network
    n.links.loc[["p1||p2_fwd_2040", "p1||p2_rev_2040"], "p_nom_extendable"] = False
    n.optimize.create_model()
    add_bidirectional_link_constraints(n)

    assert bidirectional_constraints(n) == ["bidirectional_link_p1||p2"]
