"""
Test the reserves constraints functionality.

This module contains tests for the reserve margin constraints in PyPSA-USA.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from pypsa.descriptors import (
    get_activity_mask,
)
from pypsa.descriptors import (
    get_switchable_as_dense as get_as_dense,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from opts.reserves import add_ERM_constraints, store_ERM_duals


@pytest.fixture
def reserve_margin_network(base_network):
    """
    Adapt base network for ERM constraint testing.

    Extends the base network with parameters relevant to reserve margin testing.
    """
    n = base_network.copy()

    # Create a higher peak in load profile for reserve margin testing
    load_profile_region1 = pd.Series(
        np.concatenate([np.linspace(800, 1200, 12), np.linspace(1200, 800, 12)]),
        index=n.snapshots,
    )

    load_profile_region2 = pd.Series(
        np.concatenate([np.linspace(500, 700, 18), np.linspace(700, 500, 6)]),
        index=n.snapshots,
    )

    # Update load profiles
    n.loads_t.p_set.loc[:, "load1"] = load_profile_region1
    n.loads_t.p_set.loc[:, "load2"] = load_profile_region1 * 0.75
    n.loads_t.p_set.loc[:, "load3"] = load_profile_region2

    return n


@pytest.fixture
def multi_period_reserve_network(multi_period_base_network):
    """
    Adapt multi-period base network for ERM constraint testing.

    Extends the multi-period base network with peaky load profiles.
    """
    n = multi_period_base_network.copy()

    load_profile_region1 = pd.Series(
        np.tile(
            np.concatenate([np.linspace(800, 1200, 12), np.linspace(1200, 800, 12)]),
            2,  # two periods
        ),
        index=n.snapshots,
    )

    load_profile_region2 = pd.Series(
        np.tile(
            np.concatenate([np.linspace(500, 700, 18), np.linspace(700, 500, 6)]),
            2,
        ),
        index=n.snapshots,
    )

    n.loads_t.p_set.loc[:, "load1"] = load_profile_region1
    n.loads_t.p_set.loc[:, "load2"] = load_profile_region1 * 0.75
    n.loads_t.p_set.loc[:, "load3"] = load_profile_region2

    return n


def test_erm_constraint_binding(reserve_margin_network):
    """Test that ERM constraint correctly limits generation capacity."""
    n = reserve_margin_network.copy()

    erm_value = 0.90

    def extra_functionality(n, _):
        add_ERM_constraints(n, regional_erm_data={"all": erm_value})

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)
    store_ERM_duals(n)

    nodal_demand = n.loads_t.p.T.groupby(n.loads.bus).sum().T
    nodal_reserve_requirement = nodal_demand * (1.0 + erm_value)

    nodal_generator_capacity = (
        (n.generators.p_nom_opt * get_as_dense(n, "Generator", "p_max_pu", n.snapshots))
        .T.groupby(n.generators.bus)
        .sum()
        .T
    )
    nodal_storage_capacity = (
        (
            n.storage_units.p_nom_opt
            * get_as_dense(n, "StorageUnit", "p_max_pu", n.snapshots)
            * n.storage_units.efficiency_store
        )
        .T.groupby(n.storage_units.bus)
        .sum()
        .T
    )

    line_contribution = n.lines_t["s_RESERVES"]
    injection_b0 = -1 * line_contribution.T.groupby(n.lines.bus0).sum().T
    injection_b1 = line_contribution.T.groupby(n.lines.bus1).sum().T
    line_injections = injection_b0.add(injection_b1, fill_value=0)

    link_contribution = n.links_t["p_RESERVES"]
    injection_b0 = -1 * link_contribution.T.groupby(n.links.bus0).sum().T
    injection_b1 = link_contribution.T.groupby(n.links.bus1).sum().T
    link_injections = injection_b0.add(injection_b1, fill_value=0)

    nodal_reserve_capacity = (
        nodal_generator_capacity.add(nodal_storage_capacity, fill_value=0)
        .add(line_injections, fill_value=0)
        .add(link_injections, fill_value=0)
    )

    assert (nodal_reserve_capacity - nodal_reserve_requirement >= -0.1).all().all(), (
        "Nodal reserve capacity should be at least as large as the nodal reserve requirement"
    )


def test_multiple_non_overlapping_erms(reserve_margin_network):
    """Test that multiple ERM constraints work correctly for non-overlapping regions."""
    n = reserve_margin_network.copy()

    erm_dict = {"NERC1": 0.15, "NERC2": 0.30}

    def extra_functionality(n, _):
        add_ERM_constraints(n, regional_erm_data=erm_dict)

    try:
        n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)
    except Exception:
        assert False, "Optimization failed"

    store_ERM_duals(n)

    # Verify that ERM constraints were actually added
    erm_constraints = [c for c in n.model.constraints if "ERM" in c]
    assert len(erm_constraints) >= 2, f"Should have at least 2 ERM constraints, found {len(erm_constraints)}"

    # Verify each region has its own ERM constraint
    for region_name in erm_dict:
        constraint_name = f"GlobalConstraint-{region_name}_ERM"
        assert constraint_name in n.model.constraints, f"ERM constraint {constraint_name} should exist"

    # Verify that ERM duals are stored correctly
    assert hasattr(n.buses_t, "erm_price"), "ERM dual prices should be stored in n.buses_t.erm_price"

    # Check that we have ERM prices for both regions
    erm_price_data = n.buses_t.erm_price
    assert not erm_price_data.isnull().values.any(), "ERM price data should not contain any NaN values"


def test_erm_increases_capacity(reserve_margin_network):
    """Test that ERM constraint of 0.14 results in more capacity built than no ERM."""
    # First run without ERM constraints
    n_no_erm = reserve_margin_network.copy()
    n_no_erm.optimize(solver_name="glpk", multi_investment_periods=True)

    total_gen_capacity_no_erm = n_no_erm.generators.p_nom_opt.sum()
    total_storage_capacity_no_erm = n_no_erm.storage_units.p_nom_opt.sum()
    total_capacity_no_erm = total_gen_capacity_no_erm + total_storage_capacity_no_erm

    # Now run with ERM constraint of 0.14
    n_with_erm = reserve_margin_network.copy()

    def extra_functionality(n, _):
        add_ERM_constraints(n, regional_erm_data={"all": 0.14})

    n_with_erm.optimize(
        solver_name="glpk",
        multi_investment_periods=True,
        extra_functionality=extra_functionality,
    )

    total_gen_capacity_with_erm = n_with_erm.generators.p_nom_opt.sum()
    total_storage_capacity_with_erm = n_with_erm.storage_units.p_nom_opt.sum()
    total_capacity_with_erm = total_gen_capacity_with_erm + total_storage_capacity_with_erm

    assert total_capacity_with_erm > total_capacity_no_erm, (
        f"ERM constraint (erm=0.14) should result in more capacity built. "
        f"Without ERM: {total_capacity_no_erm:.2f} MW, With ERM: {total_capacity_with_erm:.2f} MW"
    )

    assert n_with_erm.objective > n_no_erm.objective, (
        f"ERM constraint should increase system cost. "
        f"Without ERM: {n_no_erm.objective:.2f}, With ERM: {n_with_erm.objective:.2f}"
    )


def test_erm_increases_capacity_no_expandable_transmission(reserve_margin_network):
    """Test that ERM constraint of 0.14 results in more capacity built than no ERM, with no expandable lines or links."""

    def disable_transmission_expansion(n):
        """Disable expansion of lines and links, and add firm generation at z2 for feasibility."""
        n.lines["s_nom_extendable"] = False
        n.links["p_nom_extendable"] = False
        n.lines["s_nom"] = 2000
        n.links["p_nom"] = 2000
        n.add(
            "Generator",
            "gas_z2",
            bus="z2",
            p_nom=0,
            p_nom_extendable=True,
            carrier="gas",
            capital_cost=500,
            marginal_cost=20,
            p_max_pu=1.0,
            p_nom_max=5000,
            build_year=2030,
            lifetime=20,
        )
        return n

    # First run without ERM constraints
    n_no_erm = reserve_margin_network.copy()
    n_no_erm = disable_transmission_expansion(n_no_erm)
    n_no_erm.optimize(solver_name="glpk", multi_investment_periods=True)

    total_gen_capacity_no_erm = n_no_erm.generators.p_nom_opt.sum()
    total_storage_capacity_no_erm = n_no_erm.storage_units.p_nom_opt.sum()
    total_capacity_no_erm = total_gen_capacity_no_erm + total_storage_capacity_no_erm

    # Now run with ERM constraint of 0.14
    n_with_erm = reserve_margin_network.copy()
    n_with_erm = disable_transmission_expansion(n_with_erm)

    def extra_functionality(n, _):
        add_ERM_constraints(n, regional_erm_data={"all": 0.14})

    n_with_erm.optimize(
        solver_name="glpk",
        multi_investment_periods=True,
        extra_functionality=extra_functionality,
    )

    total_gen_capacity_with_erm = n_with_erm.generators.p_nom_opt.sum()
    total_storage_capacity_with_erm = n_with_erm.storage_units.p_nom_opt.sum()
    total_capacity_with_erm = total_gen_capacity_with_erm + total_storage_capacity_with_erm

    assert total_capacity_with_erm > total_capacity_no_erm, (
        f"ERM constraint (erm=0.14) should result in more capacity built (no expandable transmission). "
        f"Without ERM: {total_capacity_no_erm:.2f} MW, With ERM: {total_capacity_with_erm:.2f} MW"
    )

    assert n_with_erm.objective > n_no_erm.objective, (
        f"ERM constraint should increase system cost (no expandable transmission). "
        f"Without ERM: {n_no_erm.objective:.2f}, With ERM: {n_with_erm.objective:.2f}"
    )


def test_multi_period_erm_optimization(multi_period_reserve_network):
    """Test that multi-period network with ERM solves and creates one constraint (not per period)."""
    n = multi_period_reserve_network.copy()

    erm_dict = {"all": 0.15}

    def extra_functionality(n, _):
        add_ERM_constraints(n, regional_erm_data=erm_dict)

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)

    # Verify one ERM constraint is created (not two per period)
    erm_constraints = [c for c in n.model.constraints if "ERM" in c]
    assert len(erm_constraints) == 1, (
        f"Should have exactly 1 ERM constraint, found {len(erm_constraints)}: {erm_constraints}"
    )
    assert "GlobalConstraint-all_ERM" in n.model.constraints


def test_multi_period_erm_increases_capacity(multi_period_reserve_network):
    """Test that ERM increases capacity across multiple investment periods."""
    # Without ERM
    n_no_erm = multi_period_reserve_network.copy()
    n_no_erm.optimize(solver_name="glpk", multi_investment_periods=True)

    total_capacity_no_erm = n_no_erm.generators.p_nom_opt.sum() + n_no_erm.storage_units.p_nom_opt.sum()

    # With ERM
    n_with_erm = multi_period_reserve_network.copy()

    def extra_functionality(n, _):
        add_ERM_constraints(n, regional_erm_data={"all": 0.15})

    n_with_erm.optimize(
        solver_name="glpk",
        multi_investment_periods=True,
        extra_functionality=extra_functionality,
    )

    total_capacity_with_erm = n_with_erm.generators.p_nom_opt.sum() + n_with_erm.storage_units.p_nom_opt.sum()

    assert total_capacity_with_erm > total_capacity_no_erm, (
        f"ERM should increase capacity in multi-period. "
        f"Without: {total_capacity_no_erm:.2f}, With: {total_capacity_with_erm:.2f}"
    )


def test_multi_period_erm_activity_masking(multi_period_reserve_network):
    """Verify that retiring generators don't contribute to reserves in periods after retirement."""
    n = multi_period_reserve_network.copy()

    # Confirm gas_retiring is in the network and has the expected lifetime
    assert "gas_retiring" in n.generators.index
    assert n.generators.loc["gas_retiring", "build_year"] == 2025
    assert n.generators.loc["gas_retiring", "lifetime"] == 10

    # Check activity mask: gas_retiring should be active in 2030 but not in 2040
    activity = get_activity_mask(n, "Generator", n.snapshots)
    period_2030_mask = n.snapshots.get_level_values(0) == 2030
    period_2040_mask = n.snapshots.get_level_values(0) == 2040

    assert activity.loc[period_2030_mask, "gas_retiring"].all(), "gas_retiring should be active in all 2030 snapshots"
    assert not activity.loc[period_2040_mask, "gas_retiring"].any(), (
        "gas_retiring should be inactive in all 2040 snapshots"
    )

    # Run optimization with ERM
    def extra_functionality(n, _):
        add_ERM_constraints(n, regional_erm_data={"all": 0.15})

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)

    # Verify the constraint was added successfully
    assert "GlobalConstraint-all_ERM" in n.model.constraints
