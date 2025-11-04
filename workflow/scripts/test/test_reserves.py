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
from opts._helpers import get_region_buses
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
def erm_non_overlapping_config():
    """Create a config dictionary with ERM settings for non-overlapping regions."""
    return {
        "electricity": {
            "SAFE_regional_reservemargins": os.path.join(os.path.dirname(__file__), "fixtures/non_overlapping_erm.csv"),
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

    nodal_demand = n.loads_t.p.T.groupby(n.loads.bus).sum().T
    nodal_reserve_requirement = nodal_demand * (1.0 + test_data.loc[0, "prm"])

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

    # breakpoint()
    assert (nodal_reserve_capacity - nodal_reserve_requirement >= -0.1).all().all(), (
        "Nodal reserve capacity should be at least as large as the nodal reserve requirement"
    )


def test_multiple_non_overlapping_erms(reserve_margin_network, erm_non_overlapping_config):
    """Test that multiple ERM constraints work correctly for non-overlapping regions."""
    n = reserve_margin_network.copy()

    # Read the PRM data from the CSV file
    test_data = pd.read_csv(erm_non_overlapping_config["electricity"]["SAFE_regional_reservemargins"])

    # Run optimization with multiple ERM constraints
    def extra_functionality(n, _):
        add_ERM_constraints(n, regional_prm_data=test_data)

    try:
        n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)
    except Exception:
        assert False, "Optimization failed"

    store_ERM_duals(n)

    # Verify that ERM constraints were actually added (regardless of optimization success)
    erm_constraints = [c for c in n.model.constraints if "ERM" in c]
    assert len(erm_constraints) >= 2, f"Should have at least 2 ERM constraints, found {len(erm_constraints)}"

    # Check that ERM constraints are satisfied for each region separately
    for _, erm in test_data.iterrows():
        # Get buses for this specific region
        region_list = [region_.strip() for region_ in erm.region.split(",")]
        region_buses = get_region_buses(n, region_list)

        if region_buses.empty:
            continue

        # Calculate demand for this region
        regional_demand = n.loads_t.p.groupby(n.loads.bus, axis=1).sum()
        regional_demand = regional_demand.reindex(columns=region_buses.index, fill_value=0)
        nodal_reserve_requirement = regional_demand * (1.0 + erm.prm)

        # Calculate capacity for this region
        region_gens = n.generators[n.generators.bus.isin(region_buses.index)]
        region_storage = n.storage_units[n.storage_units.bus.isin(region_buses.index)]

        # Generator capacity
        if not region_gens.empty:
            nodal_generator_capacity = (
                (n.generators.p_nom_opt * get_as_dense(n, "Generator", "p_max_pu", n.snapshots))
                .groupby(n.generators.bus, axis=1)
                .sum()
                .reindex(columns=region_buses.index, fill_value=0)
            )
        else:
            nodal_generator_capacity = pd.DataFrame(0, index=n.snapshots, columns=region_buses.index)

        # Storage capacity
        if not region_storage.empty:
            nodal_storage_capacity = (
                (
                    n.storage_units.p_nom_opt
                    * get_as_dense(n, "StorageUnit", "p_max_pu", n.snapshots)
                    * n.storage_units.efficiency_store
                )
                .groupby(n.storage_units.bus, axis=1)
                .sum()
                .reindex(columns=region_buses.index, fill_value=0)
            )
        else:
            nodal_storage_capacity = pd.DataFrame(0, index=n.snapshots, columns=region_buses.index)

        # Line contributions (only for lines within the region)
        region_lines = n.lines[(n.lines.bus0.isin(region_buses.index)) & (n.lines.bus1.isin(region_buses.index))]
        if not region_lines.empty and hasattr(n, "lines_t") and "s_reserves" in n.lines_t:
            line_contribution = n.lines_t.s_reserves[region_lines.index]
            injection_b0 = -1 * line_contribution.groupby(n.lines.loc[region_lines.index, "bus0"], axis=1).sum()
            injection_b1 = line_contribution.groupby(n.lines.loc[region_lines.index, "bus1"], axis=1).sum()
            line_injections = injection_b0.add(injection_b1, fill_value=0).reindex(
                columns=region_buses.index,
                fill_value=0,
            )
        else:
            line_injections = pd.DataFrame(0, index=n.snapshots, columns=region_buses.index)

        # Link contributions (only for links within the region)
        region_links = n.links[(n.links.bus0.isin(region_buses.index)) & (n.links.bus1.isin(region_buses.index))]
        if not region_links.empty and hasattr(n, "links_t") and "p_reserves" in n.links_t:
            link_contribution = n.links_t.p_reserves[region_links.index]
            injection_b0 = -1 * link_contribution.groupby(n.links.loc[region_links.index, "bus0"], axis=1).sum()
            injection_b1 = link_contribution.groupby(n.links.loc[region_links.index, "bus1"], axis=1).sum()
            link_injections = injection_b0.add(injection_b1, fill_value=0).reindex(
                columns=region_buses.index,
                fill_value=0,
            )
        else:
            link_injections = pd.DataFrame(0, index=n.snapshots, columns=region_buses.index)

        # Total reserve capacity for this region
        nodal_reserve_capacity = (
            nodal_generator_capacity.add(nodal_storage_capacity, fill_value=0)
            .add(line_injections, fill_value=0)
            .add(link_injections, fill_value=0)
        )

        # Check that reserve capacity meets requirement for this region
        assert (nodal_reserve_capacity - nodal_reserve_requirement >= -0.1).all().all(), (
            f"Regional reserve capacity for {erm.region} should be at least as large as the regional reserve requirement"
        )

    # Verify that ERM duals are stored correctly
    assert hasattr(n.buses_t, "erm_price"), "ERM dual prices should be stored in n.buses_t.erm_price"

    # Check that we have ERM prices for both regions
    erm_price_data = n.buses_t.erm_price
    assert not erm_price_data.isnull().values.any(), "ERM price data should not contain any NaN values"
