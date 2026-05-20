"""
Test the content.py policy constraints functionality.

This module contains tests for the policy constraints in PyPSA-USA,
including Technology Capacity Targets (TCT), Renewable Portfolio Standards (RPS),
and Regional CO2 Limits.
"""

import logging
import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from opts._helpers import get_region_buses
from prepare_network import average_every_nhours
from summary import get_node_emissions_timeseries

logger = logging.getLogger(__name__)

# Fixtures


@pytest.fixture
def policy_network(base_network):
    """
    Adapt base network for policy constraint testing (RPS, TCT, CO2 limits).

    Extends the base network with parameters needed for policy constraint testing.
    """
    n = base_network.copy()

    # For policy constraints, we want clearer isolation of regions
    # Add a nuclear generator (non-renewable but clean)
    n.add(
        "Generator",
        "nuclear1",
        bus="z3",
        p_nom=300,
        p_nom_extendable=True,
        carrier="nuclear",
        capital_cost=2500,
        marginal_cost=5,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
        p_nom_max=1500,
    )

    # Add a coal generator (high CO2 emissions)
    n.add(
        "Generator",
        "coal1",
        bus="z1",
        p_nom=500,
        p_nom_extendable=True,
        carrier="coal",
        capital_cost=2000,
        marginal_cost=30,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
        p_nom_max=1000,
    )

    # Add CO2 emissions to carriers
    n.carriers.loc["coal", "co2_emissions"] = 0.8  # tonnes CO2 per MWh
    n.carriers.loc["nuclear", "co2_emissions"] = 0.0
    n.carriers.loc["gas", "co2_emissions"] = 0.4  # tonnes CO2 per MWh

    # Add nice_name to carriers for emissions calculation
    n.carriers["nice_name"] = n.carriers.index

    return n


@pytest.fixture
def clustered_policy_network(policy_network):
    """Create a time-clustered version of the policy network."""
    return average_every_nhours(policy_network, "3h")


@pytest.fixture
def co2_config():
    """Create a config dictionary for regional CO2 limit constraints."""
    return {
        "electricity": {
            "regional_Co2_limits": os.path.join(os.path.dirname(__file__), "fixtures/regional_co2_limits.csv"),
        },
        "scenario": {
            "planning_horizons": ["2030"],
        },
    }


@pytest.fixture
def rps_config():
    """Create a config dictionary for RPS constraints."""
    # Create config dictionary
    config = {
        "electricity": {
            "portfolio_standards": os.path.join(os.path.dirname(__file__), "fixtures/portfolio_standards.csv"),
        },
    }

    # Create a mock snakemake object
    class MockSnakemake:
        def __init__(self):
            self.input = type(
                "obj",
                (object,),
                {
                    "rps_reeds": os.path.join(os.path.dirname(__file__), "fixtures/rps_reeds.csv"),
                    "ces_reeds": os.path.join(os.path.dirname(__file__), "fixtures/ces_reeds.csv"),
                },
            )
            self.params = type(
                "obj",
                (object,),
                {
                    "planning_horizons": [2030],
                },
            )

    snakemake = MockSnakemake()

    return config, snakemake


@pytest.fixture
def tct_config():
    """Create a config dictionary for TCT constraints."""
    return {
        "electricity": {
            "technology_capacity_targets": os.path.join(
                os.path.dirname(__file__),
                "fixtures/technology_capacity_targets.csv",
            ),
        },
    }


def test_add_regional_co2limit(policy_network, co2_config):
    """Test that regional CO2 limits are correctly added to the network."""
    from opts.policy import add_regional_co2limit

    n = policy_network
    config = co2_config

    # Add regional CO2 limits
    def extra_functionality(n, _):
        add_regional_co2limit(n, config)

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)

    # Check that constraints were added
    assert any("co2_limit" in c for c in n.model.constraints), "No CO2 limit constraints were added"

    # Get emissions data
    emissions = get_node_emissions_timeseries(n)
    # Check that emissions are within limits for each region
    # Get the regional CO2 limits from config file
    co2_limits = pd.read_csv(config["electricity"]["regional_Co2_limits"])
    epsilon = 1e-2  # Small numerical tolerance
    for _, row in co2_limits.iterrows():
        limit = row["limit"]
        region_list = [region.strip() for region in row.regions.split(",")]
        region_buses = get_region_buses(n, region_list)
        constraint_emissions = emissions.loc[:, region_buses.index].sum().sum()
        assert constraint_emissions <= limit + epsilon, f"Emissions in region {row.name} exceed limit of {limit}"


def test_add_regional_co2limit_clustered(clustered_policy_network, co2_config):
    """Test that regional CO2 limits are correctly added to a time-clustered network."""
    from opts.policy import add_regional_co2limit

    n = clustered_policy_network
    config = co2_config

    # Add regional CO2 limits
    def extra_functionality(n, _):
        add_regional_co2limit(n, config)

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)

    # Check that constraints were added
    assert any("co2_limit" in c for c in n.model.constraints), "No CO2 limit constraints were added"

    # Get emissions data
    emissions = get_node_emissions_timeseries(n)

    # Get the regional CO2 limits from config file
    co2_limits = pd.read_csv(config["electricity"]["regional_Co2_limits"])
    epsilon = 1e-2  # Small numerical tolerance
    for _, row in co2_limits.iterrows():
        limit = row["limit"]
        region_list = [region.strip() for region in row.regions.split(",")]
        region_buses = get_region_buses(n, region_list)
        constraint_emissions = emissions.loc[:, region_buses.index].sum().sum()
        assert constraint_emissions <= limit + epsilon, f"Emissions in region {row.name} exceed limit of {limit}"


def test_add_rps_constraints(policy_network, rps_config):
    """Test that RPS constraints are correctly added to the network."""
    from opts.policy import add_RPS_constraints

    n = policy_network
    config, snakemake = rps_config

    # Add RPS constraints
    def extra_functionality(n, _):
        add_RPS_constraints(n, config, snakemake=snakemake)

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)

    # Check that constraints were added
    assert any("rps_limit" in c for c in n.model.constraints), "No RPS limit constraints were added"

    # Get the portfolio standards from config file
    portfolio_standards = pd.read_csv(config["electricity"]["portfolio_standards"])

    # Check that renewable generation meets RPS requirements
    for _, row in portfolio_standards.iterrows():
        region_list = [region.strip() for region in row.region.split(",")]
        region_buses = get_region_buses(n, region_list)

        if region_buses.empty:
            continue

        carriers = [carrier.strip() for carrier in row.carrier.split(",")]
        region_gens = n.generators[n.generators.bus.isin(region_buses.index)]
        region_gens_eligible = region_gens[region_gens.carrier.isin(carriers)]

        if region_gens_eligible.empty:
            continue

        # Calculate total generation from eligible sources
        eligible_generation = n.generators_t.p[region_gens_eligible.index].sum().sum()

        # Calculate total demand in the region
        region_demand = n.loads_t.p_set.loc[:, n.loads.bus.isin(region_buses.index)].sum().sum()
        logger.info(
            f"RPS Check: Region: {row.region}, Carriers: {carriers}, "
            f"Eligible Generation: {eligible_generation:.2f} MW, "
            f"Total Demand: {region_demand:.2f} MW, "
            f"Required %: {row.pct * 100:.1f}%, "
            f"Actual %: {(eligible_generation / region_demand) * 100:.1f}%",
        )
        # Check if RPS requirement is met with small epsilon for rounding errors
        epsilon = 1e-3
        assert eligible_generation >= (row.pct * region_demand) - epsilon, (
            f"RPS requirement of {row.pct * 100}% not met for region {row.region}"
        )


def test_add_rps_constraints_clustered(clustered_policy_network, rps_config):
    """Test that RPS constraints are correctly added to a time-clustered network."""
    from opts.policy import add_RPS_constraints

    n = clustered_policy_network
    config, snakemake = rps_config

    # Add RPS constraints
    def extra_functionality(n, _):
        add_RPS_constraints(n, config, snakemake=snakemake)

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)

    # Check that constraints were added
    assert any("rps_limit" in c for c in n.model.constraints), "No RPS limit constraints were added"

    # Get the portfolio standards from config file
    portfolio_standards = pd.read_csv(config["electricity"]["portfolio_standards"])

    # Check that renewable generation meets RPS requirements
    for _, row in portfolio_standards.iterrows():
        region_list = [region.strip() for region in row.region.split(",")]
        region_buses = get_region_buses(n, region_list)

        if region_buses.empty:
            continue

        carriers = [carrier.strip() for carrier in row.carrier.split(",")]
        region_gens = n.generators[n.generators.bus.isin(region_buses.index)]
        region_gens_eligible = region_gens[region_gens.carrier.isin(carriers)]

        if region_gens_eligible.empty:
            continue

        # Calculate total generation from eligible sources
        eligible_generation = n.generators_t.p[region_gens_eligible.index].sum().sum()

        # Calculate total demand in the region
        region_demand = n.loads_t.p_set.loc[:, n.loads.bus.isin(region_buses.index)].sum().sum()
        logger.info(
            f"RPS Check (Clustered): Region: {row.region}, Carriers: {carriers}, "
            f"Eligible Generation: {eligible_generation:.2f} MW, "
            f"Total Demand: {region_demand:.2f} MW, "
            f"Required %: {row.pct * 100:.1f}%, "
            f"Actual %: {(eligible_generation / region_demand) * 100:.1f}%",
        )
        # Check if RPS requirement is met with small epsilon for rounding errors
        epsilon = 1e-3
        assert eligible_generation >= (row.pct * region_demand) - epsilon, (
            f"RPS requirement of {row.pct * 100}% not met for region {row.region} in clustered network"
        )


def test_add_technology_capacity_target_constraints(policy_network, tct_config):
    """Test that technology capacity target constraints are correctly added to the network."""
    from opts.policy import add_technology_capacity_target_constraints

    n = policy_network
    config = tct_config

    # Add TCT constraints
    def extra_functionality(n, _):
        add_technology_capacity_target_constraints(n, config)

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)

    # Check that constraints were added
    assert any("min" in c for c in n.model.constraints), "No TCT minimum constraints were added"
    assert any("max" in c for c in n.model.constraints), "No TCT maximum constraints were added"

    # Get the technology capacity targets from config file
    tct_data = pd.read_csv(config["electricity"]["technology_capacity_targets"])

    # Check that capacity targets are met
    for _, target in tct_data.iterrows():
        region_list = [region.strip() for region in target.region.split(",")]
        region_buses = get_region_buses(n, region_list)

        if region_buses.empty:
            continue

        carriers = [carrier.strip() for carrier in target.carrier.split(",")]

        # Get total capacity (existing + new) for the target technology
        total_capacity = 0

        # Check generators
        gens = n.generators[(n.generators.bus.isin(region_buses.index)) & (n.generators.carrier.isin(carriers))]
        total_capacity += gens.p_nom_opt.sum()

        # Check storage units
        storage = n.storage_units[
            (n.storage_units.bus.isin(region_buses.index)) & (n.storage_units.carrier.isin(carriers))
        ]
        total_capacity += storage.p_nom_opt.sum()

        # Check links
        links = n.links[(n.links.bus0.isin(region_buses.index)) & (n.links.carrier.isin(carriers))]
        total_capacity += links.p_nom_opt.sum()

        # Get min and max targets, handling NaN values
        min_target = float(target["min"]) if not pd.isna(target["min"]) else None
        max_target = float(target["max"]) if not pd.isna(target["max"]) else None

        logger.info(
            f"TCT Check: Region: {target.region}, Carrier: {carriers}, "
            f"Total Capacity: {total_capacity:.2f} MW, "
            f"Min Target: {min_target}, "
            f"Max Target: {max_target}",
        )

        # Check minimum capacity if specified
        if min_target is not None:
            assert total_capacity >= min_target, (
                f"Minimum capacity target of {min_target} MW not met for {target['carrier']} in {target['region']}"
            )

        # Check maximum capacity if specified
        if max_target is not None:
            assert total_capacity <= max_target, (
                f"Maximum capacity target of {max_target} MW exceeded for {target['carrier']} in {target['region']}"
            )


def test_apply_forced_retirements_zeroes_nonextendable(policy_network, tmp_path):
    """apply_forced_retirements should zero non-extendable generators matching max=0 targets.

    Simulates the brownfield scenario where prepare_brownfield() has locked in
    previous-period capacity as non-extendable. Verifies that:
    - Non-extendable CCGT in the target region (CA/z1) is zeroed.
    - Non-extendable CCGT outside the target region (TX/z2) is untouched.
    - Extendable generators are not modified (the TCT constraint handles those).
    """
    from opts.policy import apply_forced_retirements

    n = policy_network.copy()

    # Add non-extendable CCGT generators simulating brownfield-locked capacity
    n.add(
        "Generator",
        "ccgt_ca",
        bus="z1",
        carrier="CCGT",
        p_nom=500.0,
        p_nom_min=0.0,
        p_nom_max=500.0,
        p_nom_extendable=False,
        marginal_cost=40,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
    )
    n.add(
        "Generator",
        "ccgt_tx",
        bus="z2",
        carrier="CCGT",
        p_nom=300.0,
        p_nom_min=0.0,
        p_nom_max=300.0,
        p_nom_extendable=False,
        marginal_cost=40,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
    )
    # Extendable CCGT — should NOT be touched by apply_forced_retirements
    n.add(
        "Generator",
        "ccgt_ca_ext",
        bus="z1",
        carrier="CCGT",
        p_nom=0.0,
        p_nom_min=0.0,
        p_nom_max=1000.0,
        p_nom_extendable=True,
        marginal_cost=40,
        capital_cost=600,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
    )

    # Write a TCT CSV forcing CCGT retirement in CA (reeds_state) by 2030
    tct_csv = tmp_path / "tct_forced.csv"
    tct_csv.write_text(
        "name,planning_horizon,region,carrier,min,max\nccgt_retire,2030,CA,CCGT,,0\n",
    )

    config = {"electricity": {"technology_capacity_targets": str(tct_csv)}}

    apply_forced_retirements(n, planning_horizon=2030, config=config)

    # CA non-extendable CCGT should be zeroed
    assert n.generators.loc["ccgt_ca", "p_nom"] == 0.0, "ccgt_ca p_nom should be 0"
    assert n.generators.loc["ccgt_ca", "p_nom_max"] == 0.0, "ccgt_ca p_nom_max should be 0"

    # TX non-extendable CCGT should be untouched
    assert n.generators.loc["ccgt_tx", "p_nom"] == 300.0, "ccgt_tx p_nom should be unchanged"

    # Extendable CCGT should be untouched (handled by TCT constraint, not this function)
    assert n.generators.loc["ccgt_ca_ext", "p_nom_max"] == 1000.0, "extendable ccgt should be unchanged"


def test_apply_forced_retirements_future_horizon_skipped(policy_network, tmp_path):
    """Forced retirement rows with planning_horizon > current should be skipped."""
    from opts.policy import apply_forced_retirements

    n = policy_network.copy()
    n.add(
        "Generator",
        "ccgt_ca",
        bus="z1",
        carrier="CCGT",
        p_nom=500.0,
        p_nom_min=0.0,
        p_nom_max=500.0,
        p_nom_extendable=False,
        marginal_cost=40,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
    )

    # Retirement target is 2050 — should not apply during the 2030 solve
    tct_csv = tmp_path / "tct_future.csv"
    tct_csv.write_text(
        "name,planning_horizon,region,carrier,min,max\nccgt_retire,2050,CA,CCGT,,0\n",
    )

    config = {"electricity": {"technology_capacity_targets": str(tct_csv)}}

    apply_forced_retirements(n, planning_horizon=2030, config=config)

    assert n.generators.loc["ccgt_ca", "p_nom"] == 500.0, "2050 target should not retire capacity in 2030"


def test_apply_forced_retirements_all_region(policy_network, tmp_path):
    """planning_horizon='all' with region='all' should zero matching gens in every region."""
    from opts.policy import apply_forced_retirements

    n = policy_network.copy()
    n.add(
        "Generator",
        "coal_ca",
        bus="z1",
        carrier="coal",
        p_nom=400.0,
        p_nom_min=0.0,
        p_nom_max=400.0,
        p_nom_extendable=False,
        marginal_cost=30,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
    )
    n.add(
        "Generator",
        "coal_tx",
        bus="z2",
        carrier="coal",
        p_nom=250.0,
        p_nom_min=0.0,
        p_nom_max=250.0,
        p_nom_extendable=False,
        marginal_cost=30,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
    )

    tct_csv = tmp_path / "tct_all.csv"
    tct_csv.write_text(
        "name,planning_horizon,region,carrier,min,max\ncoal_retire,all,all,coal,,0\n",
    )

    config = {"electricity": {"technology_capacity_targets": str(tct_csv)}}

    apply_forced_retirements(n, planning_horizon=2030, config=config)

    assert n.generators.loc["coal_ca", "p_nom"] == 0.0
    assert n.generators.loc["coal_tx", "p_nom"] == 0.0
