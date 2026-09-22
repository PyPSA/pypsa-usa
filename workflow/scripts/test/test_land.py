"""
Test the land use constraints functionality.

This module contains tests for the land use constraints in PyPSA-USA.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# Fixtures


@pytest.fixture
def land_use_network(base_network):
    """
    Adapt base network for land use constraint testing.

    Extends the base network with specific parameters relevant to land use constraints.
    """
    n = base_network.copy()

    # Add another generator in region_a to test constraints with multiple generators in the same region
    n.add(
        "Generator",
        "wind3",
        bus="z1",
        p_nom=0,
        p_nom_extendable=True,
        carrier="onwind",
        capital_cost=1050,
        marginal_cost=0.11,
        p_max_pu=pd.Series(0.85, index=n.snapshots),
        land_region="region_a",
        p_nom_max=300,
    )

    return n


@pytest.fixture
def myopic_land_network(base_network):
    """
    A single-horizon network shaped like the second leg of a myopic solve.

    ``solve_network.freeze_prior_periods`` turns every prior-horizon asset into a
    non-extendable generator carrying the capacity it was built with. Here a
    prior horizon built 200 MW of ``onwind`` in ``region_a`` whose developable
    potential is 300 MW, so only 100 MW of headroom is left for the extendable
    ``wind1`` sharing that land region.
    """
    n = base_network.copy()

    # one extendable onwind generator in region_a, potential 300 MW
    n.generators.loc["wind1", "p_nom_max"] = 300
    # make wind cheap enough that the optimizer wants all the land it can get
    n.generators.loc["wind1", "capital_cost"] = 1

    # the frozen prior-period build: non-extendable, p_nom > 0, same land region.
    # p_max_pu is scalar so it does not override PyPSA's activity mask.
    n.add(
        "Generator",
        "wind1 2030",
        bus="z1",
        p_nom=200,
        p_nom_extendable=False,
        carrier="onwind",
        capital_cost=1050,
        marginal_cost=0.11,
        p_max_pu=0.8,
        land_region="region_a",
        p_nom_max=200,
        build_year=2030,
        lifetime=30,
    )

    return n


# Helpers


def _legacy_rhs(n):
    """Reproduce the pre-fix right-hand side: max ``p_nom_max`` per group."""
    generators = n.generators.query("p_nom_extendable & land_region != '' ")
    maximum = generators.groupby(["carrier", "land_region"])["p_nom_max"].max()
    return maximum[np.isfinite(maximum)]


def _rhs(n):
    """The right-hand side of the built land-use constraint, as a Series."""
    return n.model.constraints["land_use_constraint"].rhs.to_pandas()


def _lhs_terms(n, carrier, land_region):
    """{generator name: coefficient} of one group's left-hand side."""
    con = n.model.constraints["land_use_constraint"]
    var_labels = np.atleast_1d(con.vars.to_pandas().loc[(carrier, land_region)])
    coeffs = np.atleast_1d(con.coeffs.to_pandas().loc[(carrier, land_region)])

    labels = n.model.variables["Generator-p_nom"].labels.to_pandas()
    label_to_name = pd.Series(labels.index.to_numpy(), index=labels.to_numpy())

    return {label_to_name[label]: float(coeff) for label, coeff in zip(var_labels, coeffs) if label != -1}


# Tests


def test_land_use_rhs_matches_legacy_without_fixed_capacity(land_use_network):
    """Perfect foresight with no fixed land-region capacity is untouched by the fix.

    Every generator carrying a ``land_region`` here is extendable, so the
    right-hand side must still be the group's ``max`` of ``p_nom_max`` and the
    left-hand side must still be the plain sum of the group's ``p_nom``
    variables.
    """
    from opts.land import add_land_use_constraints

    n = land_use_network
    n.optimize.create_model(multi_investment_periods=True)
    add_land_use_constraints(n)

    legacy = _legacy_rhs(n)
    rhs = _rhs(n)

    assert set(rhs.index) == set(legacy.index)
    pd.testing.assert_series_equal(
        rhs.reindex(legacy.index).astype(float),
        legacy.astype(float),
        check_names=False,
    )
    # region_a onwind: wind1 (500 MW potential) and wind3 (300 MW) share the land
    assert rhs.loc[("onwind", "region_a")] == 500
    assert _lhs_terms(n, "onwind", "region_a") == {"wind1": 1.0, "wind3": 1.0}


def test_land_use_ignores_generators_without_land_region(land_use_network):
    """``gas1`` is non-extendable with no ``land_region``; it must not appear anywhere."""
    from opts.land import add_land_use_constraints

    n = land_use_network
    assert not n.generators.loc["gas1", "p_nom_extendable"]
    assert n.generators.loc["gas1", "land_region"] in ("", None) or pd.isna(n.generators.loc["gas1", "land_region"])

    n.optimize.create_model(multi_investment_periods=True)
    add_land_use_constraints(n)

    assert "gas" not in {carrier for carrier, _ in _rhs(n).index}


def test_frozen_prior_build_reduces_land_headroom(myopic_land_network):
    """A frozen prior-horizon build consumes its land: 300 MW potential - 200 MW built = 100 MW."""
    from opts.land import add_land_use_constraints

    n = myopic_land_network
    n.optimize.create_model(multi_investment_periods=True)
    add_land_use_constraints(n)

    assert _rhs(n).loc[("onwind", "region_a")] == pytest.approx(100.0)
    # the frozen generator has no p_nom variable, so only wind1 is on the LHS
    assert _lhs_terms(n, "onwind", "region_a") == {"wind1": 1.0}
    # other land regions are unaffected
    assert _rhs(n).loc[("onwind", "region_b")] == pytest.approx(400.0)


def test_frozen_prior_build_binds_in_solve(myopic_land_network):
    """Solved with HiGHS, the extendable onwind cannot exceed the remaining 100 MW."""
    from opts.land import add_land_use_constraints

    n = myopic_land_network

    # control: without the land-use constraint the cheap wind builds well past 100 MW
    control = n.copy()
    control.optimize(solver_name="highs", multi_investment_periods=True)
    assert control.generators.loc["wind1", "p_nom_opt"] > 100.0

    def extra_functionality(network, snapshots):
        add_land_use_constraints(network)

    status, _ = n.optimize(
        solver_name="highs",
        multi_investment_periods=True,
        extra_functionality=extra_functionality,
    )
    assert status == "ok"
    assert n.generators.loc["wind1", "p_nom_opt"] <= 100.0 + 1e-6
    # the land limit is what stops it, so it builds exactly the remaining headroom
    assert n.generators.loc["wind1", "p_nom_opt"] == pytest.approx(100.0, abs=1e-4)


def test_inactive_frozen_capacity_does_not_reduce_land_headroom(myopic_land_network):
    """Capacity that is retired, or not yet built, releases (or has not taken) its land."""
    from opts.land import add_land_use_constraints

    n = myopic_land_network

    # the frozen build retired before the solved horizon (2000 + 10 < 2030)
    n.generators.loc["wind1 2030", "build_year"] = 2000
    n.generators.loc["wind1 2030", "lifetime"] = 10

    # and a frozen build that only arrives in a later horizon
    n.add(
        "Generator",
        "wind1 2040",
        bus="z1",
        p_nom=150,
        p_nom_extendable=False,
        carrier="onwind",
        marginal_cost=0.11,
        p_max_pu=0.8,
        land_region="region_a",
        p_nom_max=150,
        build_year=2040,
        lifetime=30,
    )

    n.optimize.create_model(multi_investment_periods=True)
    add_land_use_constraints(n)

    assert _rhs(n).loc[("onwind", "region_a")] == pytest.approx(300.0)


def test_extendable_existing_vintage_is_not_double_counted(myopic_land_network):
    """An economically retirable "existing" vintage stays on the LHS and is not subtracted.

    ``add_extra_components`` leaves the split-out existing generator extendable
    when economic retirement is enabled, capped at its own capacity. It must be
    charged to the land exactly once — through the left-hand side.
    """
    from opts.land import add_land_use_constraints

    n = myopic_land_network
    n.generators.loc["wind1 2030", "p_nom_extendable"] = True  # economic retirement enabled

    n.optimize.create_model(multi_investment_periods=True)
    add_land_use_constraints(n)

    # nothing is subtracted; the group potential is still max(300, 200)
    assert _rhs(n).loc[("onwind", "region_a")] == pytest.approx(300.0)
    assert _lhs_terms(n, "onwind", "region_a") == {"wind1": 1.0, "wind1 2030": 1.0}


def test_oversubscribed_land_is_clipped_to_zero(myopic_land_network, caplog):
    """Existing capacity beyond the potential leaves no headroom, not a negative bound."""
    from opts.land import add_land_use_constraints

    n = myopic_land_network
    n.generators.loc["wind1 2030", "p_nom"] = 400  # more than the 300 MW potential

    n.optimize.create_model(multi_investment_periods=True)
    with caplog.at_level("WARNING"):
        add_land_use_constraints(n)

    assert _rhs(n).loc[("onwind", "region_a")] == pytest.approx(0.0)
    assert "exceeds the developable potential" in caplog.text
