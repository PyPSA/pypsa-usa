"""
Test the reserves constraints functionality.

This module contains tests for the reserve margin constraints in PyPSA-USA.
"""
import os
import sys

import pandas as pd
import pytest
from pypsa.descriptors import (
    get_switchable_as_dense as get_as_dense,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from opts.reserves import add_ERM_constraints, store_erm_data  # noqa: E402


@pytest.fixture
def reserve_margin_data():
    """Create test data for reserve margin constraints."""
    return pd.DataFrame(
        [
            {
                "name": "test_region",
                "region": "all",
                "prm": 0.15,  # 15% reserve margin
                "planning_horizon": 2030,
            },
        ],
    )


def test_erm_constraint_binding(reserve_margin_network, reserve_margin_data):
    """Test that ERM constraint correctly limits generation capacity."""
    n = reserve_margin_network.copy()

    # Set a high ERM requirement (1100%)
    test_data = reserve_margin_data.copy()
    test_data.loc[0, "prm"] = 10

    # Run optimization with ERM constraints
    def extra_functionality(n, _):
        add_ERM_constraints(n, regional_prm_data=test_data)

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)
    store_erm_data(n)

    nodal_demand = n.loads_t.p.groupby(n.loads.bus, axis=1).sum()
    peak_demand_hour = nodal_demand.sum(axis=1).idxmax()
    nodal_reserve_requirement = nodal_demand * (1.0 + test_data.loc[0, "prm"])
    req_node_reserve_peak = nodal_reserve_requirement.loc[peak_demand_hour]

    nodal_generator_capacity = (
        (n.generators.p_nom_opt * get_as_dense(n, "Generator", "p_max_pu", n.snapshots))
        .groupby(n.generators.bus, axis=1)
        .sum()
    )
    nodal_storage_capacity = (
        (
            n.storage_units.p_nom_opt
            * get_as_dense(n, "StorageUnit", "p_max_pu", n.snapshots)
            * n.storage_units.efficiency_store
        )
        .groupby(n.storage_units.bus, axis=1)
        .sum()
    )

    line_contribution = n.lines_t.s_reserves
    injection_b0 = -1 * line_contribution.groupby(n.lines.bus0, axis=1).sum()
    injection_b1 = line_contribution.groupby(n.lines.bus1, axis=1).sum()
    line_injections = injection_b0.add(injection_b1, fill_value=0)
    nodal_reserve_capacity = nodal_generator_capacity.add(nodal_storage_capacity, fill_value=0).add(
        line_injections,
        fill_value=0,
    )

    nodal_reserve_capacity_peak = nodal_reserve_capacity.loc[peak_demand_hour]

    assert all(
        nodal_reserve_capacity_peak >= req_node_reserve_peak,
    ), "Nodal reserve capacity should be at least as large as the nodal reserve requirement"
