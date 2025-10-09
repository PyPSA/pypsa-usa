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
    get_switchable_as_dense as get_as_dense,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from opts.reserves import add_ERM_constraints, store_ERM_duals


@pytest.fixture
def erm_config():
    """Create a config dictionary with ERM settings."""
    return {
        "electricity": {
            "SAFE_regional_reservemargins": os.path.join(os.path.dirname(__file__), "fixtures/test_prm.csv"),
        },
    }


@pytest.fixture
def erm_multi_region_config():
    """Create a config dictionary with ERM settings for multiple regions and different reserve margins."""
    return {
        "electricity": {
            "SAFE_regional_reservemargins": os.path.join(os.path.dirname(__file__), "fixtures/multi_region_prm.csv"),
        },
    }


@pytest.fixture
def reserve_margin_network(base_network):
    """
    Adapt base network for ERM and PRM constraint testing.

    Extends the base network with parameters relevant to reserve margin testing.
    """
    n = base_network.copy()

    # Create a higher peak in load profile for reserve margin testing
    # Make load profile peaky with peak at hour 11 for region 1 and hour 18 for region 2
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


def test_erm_constraint_binding(reserve_margin_network, erm_config):
    """Test that ERM constraint correctly limits generation capacity."""
    n = reserve_margin_network.copy()

    # Read the PRM data from the CSV file
    test_data = pd.read_csv(erm_config["electricity"]["SAFE_regional_reservemargins"])

    # Set a high ERM requirement (1100%)
    test_data.loc[0, "prm"] = 10

    # Run optimization with ERM constraints
    def extra_functionality(n, _):
        add_ERM_constraints(n, regional_prm_data=test_data)

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)
    store_ERM_duals(n)

    nodal_demand = n.loads_t.p.groupby(n.loads.bus, axis=1).sum()
    nodal_reserve_requirement = nodal_demand * (1.0 + test_data.loc[0, "prm"])

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

    link_contribution = n.links_t.p_reserves
    injection_b0 = -1 * link_contribution.groupby(n.links.bus0, axis=1).sum()
    injection_b1 = link_contribution.groupby(n.links.bus1, axis=1).sum()
    link_injections = injection_b0.add(injection_b1, fill_value=0)

    nodal_reserve_capacity = (
        nodal_generator_capacity.add(nodal_storage_capacity, fill_value=0)
        .add(line_injections, fill_value=0)
        .add(link_injections, fill_value=0)
    )

    assert all(
        nodal_reserve_capacity - nodal_reserve_requirement >= -0.1,
    ), "Nodal reserve capacity should be at least as large as the nodal reserve requirement"
