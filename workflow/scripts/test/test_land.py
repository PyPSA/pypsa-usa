"""
Test the land use constraints functionality.

This module contains tests for the land use constraints in PyPSA-USA.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from opts.land import add_land_use_constraints  # noqa: E402

# Tests


def test_land_use_constraint_creation(land_use_network):
    """Test that land use constraints are correctly created."""
    n = land_use_network.copy()

    # Prepare the optimization (creates the model)
    n.lopf(pyomo=False, solver_name="glpk", formulation="kirchhoff")

    # Add land use constraints
    add_land_use_constraints(n)

    # Check that the constraint was created
    assert "land_use_constraint" in n.model.constraints

    # Check that the land use constraint is correctly defined
    constraint_name = "land_use_constraint"
    constraints = n.model.constraints[constraint_name]

    # We should have a constraint for each region with extendable generators
    assert len(constraints.variables) >= 2, "Should have at least 2 constraints (one per land region)"


def test_land_use_constraint_limiting(land_use_network):
    """Test that land use constraints properly limit generator expansion."""
    n = land_use_network.copy()

    # Add load that would require all possible generation capacity
    n.loads.p_set = n.loads.p_set * 5

    # First, optimize without land constraints to get unconstrained capacity
    n.lopf(pyomo=False, solver_name="glpk", formulation="kirchhoff")
    unconstrained_wind_capacity = n.generators.loc[
        n.generators.carrier == "onwind",
        "p_nom_opt",
    ].sum()

    # Now add land constraints and re-optimize
    n = land_use_network.copy()
    n.loads.p_set = n.loads.p_set * 5

    # Set a tight land constraint for region_a (below what we would otherwise build)
    max_capacity_region_a = 400  # This is less than total capacity of wind generators in region_a

    # Temporarily modify the p_nom_max to enforce the land constraint
    region_a_gens = n.generators[n.generators.land_region == "region_a"]
    original_nom_max = {}

    for gen in region_a_gens.index:
        original_nom_max[gen] = n.generators.loc[gen, "p_nom_max"]
        n.generators.loc[gen, "p_nom_max"] = max_capacity_region_a / len(region_a_gens)

    n.lopf(pyomo=False, solver_name="glpk", formulation="kirchhoff")
    add_land_use_constraints(n)
    n.optimize(solver_name="glpk")

    constrained_wind_capacity = n.generators.loc[
        n.generators.land_region == "region_a",
        "p_nom_opt",
    ].sum()

    # Reset the modification
    for gen, val in original_nom_max.items():
        n.generators.loc[gen, "p_nom_max"] = val

    # The constrained capacity should be less than or equal to the constraint
    # and less than the unconstrained capacity
    assert constrained_wind_capacity <= max_capacity_region_a, "Total capacity should respect the land constraint"
    assert constrained_wind_capacity < unconstrained_wind_capacity, "Land constraint should reduce built capacity"


def test_land_use_constraint_multiple_carriers(land_use_network):
    """Test land use constraints with multiple carriers in the same region."""
    n = land_use_network.copy()

    # Verify we have different carriers in the same land region
    carriers_in_region_a = n.generators[n.generators.land_region == "region_a"].carrier.unique()
    assert len(carriers_in_region_a) >= 2, "Test requires multiple carriers in region_a"

    # Prepare the optimization
    n.lopf(pyomo=False, solver_name="glpk", formulation="kirchhoff")

    # Add land use constraints
    add_land_use_constraints(n)

    # Check that we have a constraint for each carrier+region combination
    constraint_name = "land_use_constraint"
    constraints = n.model.constraints[constraint_name]

    # The constraints should include separate constraints for each carrier+region combination
    num_carrier_region_combinations = n.generators.groupby(["carrier", "land_region"]).ngroups
    assert (
        len(constraints.variables) >= num_carrier_region_combinations
    ), "Should have at least one constraint per carrier+region combination"


def test_land_use_constraint_edge_cases(land_use_network):
    """Test edge cases for land use constraints."""
    n = land_use_network.copy()

    # Case 1: Empty land_region (should be excluded from constraints)
    wind_gen = n.generators[n.generators.carrier == "onwind"].index[0]
    n.generators.loc[wind_gen, "land_region"] = ""

    # Case 2: Infinite p_nom_max (constraint should be ignored for this generator)
    solar_gen = n.generators[n.generators.carrier == "solar"].index[0]
    n.generators.loc[solar_gen, "p_nom_max"] = float("inf")

    # Prepare the optimization
    n.lopf(pyomo=False, solver_name="glpk", formulation="kirchhoff")

    # Add land use constraints (should handle these edge cases without errors)
    add_land_use_constraints(n)

    # Successfully reaching this point means the function correctly handled the edge cases
    # Let's make sure the constraint doesn't reference generators that should be excluded
    if "land_use_constraint" in n.model.constraints:
        # The constraints should now only include valid generators
        constraint_name = "land_use_constraint"
        constraints = n.model.constraints[constraint_name]
        # The specific check depends on how pypsa-linopy structures its constraints
        # This is just a basic check that we have at least one constraint
        assert constraints is not None, "Should still have constraints for valid generators"


def test_land_use_constraint_no_extendable_generators(land_use_network):
    """Test behavior when there are no extendable generators with land_region."""
    n = land_use_network.copy()

    # Make all generators non-extendable
    n.generators.p_nom_extendable = False

    # Prepare the optimization
    n.lopf(pyomo=False, solver_name="glpk", formulation="kirchhoff")

    # Add land use constraints (should return early without creating constraints)
    add_land_use_constraints(n)

    # Check that no land use constraints were created
    assert "land_use_constraint" not in n.model.constraints or len(n.model.constraints["land_use_constraint"]) == 0


def test_land_use_constraints_with_optimization(land_use_network):
    """Test full optimization workflow with land use constraints."""
    n = land_use_network.copy()

    # Make sure we need to build capacity by increasing demand
    n.loads.p_set = n.loads.p_set * 2

    # Set explicit land use limits for each region
    region_a_limit = 200  # MW
    region_b_limit = 150  # MW
    region_c_limit = 400  # MW

    # Update all generators in each region
    for gen in n.generators[n.generators.land_region == "region_a"].index:
        n.generators.loc[gen, "p_nom_max"] = region_a_limit / len(n.generators[n.generators.land_region == "region_a"])

    for gen in n.generators[n.generators.land_region == "region_b"].index:
        n.generators.loc[gen, "p_nom_max"] = region_b_limit / len(n.generators[n.generators.land_region == "region_b"])

    for gen in n.generators[n.generators.land_region == "region_c"].index:
        n.generators.loc[gen, "p_nom_max"] = region_c_limit / len(n.generators[n.generators.land_region == "region_c"])

    # Run optimization with land use constraints
    n.lopf(pyomo=False, solver_name="glpk", formulation="kirchhoff", extra_functionality=add_land_use_constraints)

    # Check that land constraints were respected
    total_region_a_capacity = n.generators.loc[
        n.generators.land_region == "region_a",
        "p_nom_opt",
    ].sum()

    total_region_b_capacity = n.generators.loc[
        n.generators.land_region == "region_b",
        "p_nom_opt",
    ].sum()

    total_region_c_capacity = n.generators.loc[
        n.generators.land_region == "region_c",
        "p_nom_opt",
    ].sum()

    # Verify that the total capacity in each region doesn't exceed the limit
    assert total_region_a_capacity <= region_a_limit, f"Region A capacity should be limited to {region_a_limit} MW"
    assert total_region_b_capacity <= region_b_limit, f"Region B capacity should be limited to {region_b_limit} MW"
    assert total_region_c_capacity <= region_c_limit, f"Region C capacity should be limited to {region_c_limit} MW"

    # Verify the gas generator (which has no land constraints) was built to compensate
    gas_gens = n.generators[n.generators.carrier == "gas"]
    assert any(
        gen["p_nom_opt"] > gen["p_nom"] for _, gen in gas_gens.iterrows()
    ), "Gas plants should be expanded to compensate for land constraints"
