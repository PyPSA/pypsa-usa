"""
Test the reserves constraints functionality.

This module contains tests for the reserve margin constraints in PyPSA-USA.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from opts.reserves import add_ERM_constraints, add_PRM_constraints  # noqa: E402


@pytest.fixture
def reserve_margin_data():
    """Create test data for reserve margin constraints."""
    return pd.DataFrame(
        [
            {
                "name": "test_region",
                "region": "US",
                "prm": 0.15,  # 15% reserve margin
                "planning_horizon": 2030,
            },
        ],
    )


def test_erm_constraint_creation(reserve_margin_network, reserve_margin_data):
    """Test that energy reserve margin constraints are correctly created."""
    n = reserve_margin_network.copy()

    # Prepare the optimization (creates the model)
    n.optimize(solver_name="glpk")

    # Add energy reserve margin constraints with direct data input
    add_ERM_constraints(n, regional_prm_data=reserve_margin_data)

    # Check that we have ERM constraints
    erm_constraints = [c for c in n.model.constraints if "ERM_hr" in c]
    assert len(erm_constraints) > 0, "Should have at least one ERM constraint"


def test_prm_constraint_creation(reserve_margin_network, reserve_margin_data):
    """Test that planning reserve margin constraints are correctly created."""
    n = reserve_margin_network.copy()

    # Prepare the optimization (creates the model)
    n.optimize(solver_name="glpk")

    # Add planning reserve margin constraints with direct data input
    add_PRM_constraints(n, regional_prm_data=reserve_margin_data)

    # Check that we have PRM constraints
    prm_constraints = [c for c in n.model.constraints if "_PRM" in c]
    assert len(prm_constraints) > 0, "Should have at least one PRM constraint"


def test_erm_constraint_binding(reserve_margin_network, reserve_margin_data):
    """Test that ERM constraint correctly limits generation capacity."""
    n = reserve_margin_network.copy()

    # Set a high ERM requirement (50%)
    test_data = reserve_margin_data.copy()
    test_data.loc[0, "prm"] = 0.5  # 50% reserve margin

    # Run optimization with ERM constraints
    def add_constraints(n):
        add_ERM_constraints(n, regional_prm_data=test_data)

    n.optimize(solver_name="glpk", extra_functionality=add_constraints)

    # Get total energy demand
    total_energy_demand = n.loads_t.p.sum().sum()

    # Get total energy generation
    total_energy_generation = n.generators_t.p.sum().sum() + n.storage_units_t.p.sum().sum()

    # Check that energy generation meets demand plus reserves
    assert (
        total_energy_generation >= total_energy_demand * 1.5
    ), "Total generation should be at least 150% of demand due to 50% ERM"


def test_prm_constraint_binding(reserve_margin_network, reserve_margin_data):
    """Test that PRM constraint correctly limits generation capacity."""
    n = reserve_margin_network.copy()

    # Set a high PRM requirement (30%)
    test_data = reserve_margin_data.copy()
    test_data.loc[0, "prm"] = 0.3  # 30% reserve margin

    # Run optimization with PRM constraints
    def add_constraints(n):
        add_PRM_constraints(n, regional_prm_data=test_data)

    n.lopf(pyomo=False, solver_name="glpk", formulation="kirchhoff", extra_functionality=add_constraints)

    # Calculate total peak demand
    peak_demand = n.loads_t.p.sum(axis=1).max()

    # Calculate total firm capacity (excluding variable renewables)
    firm_generators = n.generators[~n.generators.carrier.isin(["onwind", "solar"])]
    total_firm_capacity = firm_generators.p_nom_opt.sum()

    # Check that firm capacity meets peak demand plus reserves
    assert (
        total_firm_capacity >= peak_demand * 1.3
    ), "Firm capacity should be at least 130% of peak demand due to 30% PRM"


def test_both_constraints_together(reserve_margin_network, reserve_margin_data):
    """Test that both ERM and PRM constraints can be applied together."""
    n = reserve_margin_network.copy()

    # Set both ERM and PRM requirements
    test_data = pd.DataFrame(
        [
            {
                "name": "test_region_erm",
                "region": "US",
                "prm": 0.2,  # 20% energy reserve margin
                "planning_horizon": "2030",
            },
            {
                "name": "test_region_prm",
                "region": "US",
                "prm": 0.15,  # 15% planning reserve margin
                "planning_horizon": "2030",
            },
        ],
    )

    def add_both_constraints(n):
        add_ERM_constraints(n, regional_prm_data=test_data)
        add_PRM_constraints(n, regional_prm_data=test_data)

    # Run optimization with both constraints
    n.lopf(pyomo=False, solver_name="glpk", formulation="kirchhoff", extra_functionality=add_both_constraints)

    # Calculate total energy demand
    total_energy_demand = n.loads_t.p.sum().sum()

    # Calculate total energy generation
    total_energy_generation = n.generators_t.p.sum().sum() + n.storage_units_t.p.sum().sum()

    # Calculate total peak demand
    peak_demand = n.loads_t.p.sum(axis=1).max()

    # Calculate total firm capacity (excluding variable renewables)
    firm_generators = n.generators[~n.generators.carrier.isin(["onwind", "solar"])]
    total_firm_capacity = firm_generators.p_nom_opt.sum()

    # Check that both constraints are satisfied
    assert (
        total_energy_generation >= total_energy_demand * 1.2
    ), "Total generation should be at least 120% of demand due to 20% ERM"

    assert (
        total_firm_capacity >= peak_demand * 1.15
    ), "Firm capacity should be at least 115% of peak demand due to 15% PRM"


def test_constraints_with_storage(reserve_margin_network, reserve_margin_data):
    """Test how reserve margin constraints interact with storage."""
    n = reserve_margin_network.copy()

    # Ensure we have storage in our test network
    assert n.storage_units.p_nom_extendable.any(), "Test network should have extendable storage"

    # Set moderate reserve requirements
    test_data = pd.DataFrame(
        [
            {
                "name": "test_region",
                "region": "US",
                "prm": 0.1,  # 10% reserve margin
                "planning_horizon": "2030",
            },
        ],
    )

    def add_both_constraints(n):
        add_ERM_constraints(n, regional_prm_data=test_data)
        add_PRM_constraints(n, regional_prm_data=test_data)

    # Run optimization with both constraints
    n.lopf(pyomo=False, solver_name="glpk", formulation="kirchhoff", extra_functionality=add_both_constraints)

    # Check that storage was built
    total_storage_capacity = n.storage_units.p_nom_opt.sum()
    assert total_storage_capacity > 0, "Storage should be built to help meet reserve requirements"

    # Check that storage is being used
    storage_discharge = n.storage_units_t.p.sum().sum()
    assert storage_discharge > 0, "Storage should be discharged to help meet demand"
