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
def rps_config_with_btm():
    """Create a config dict for RPS constraints that includes BTM solar credit data."""
    config = {
        "electricity": {
            "portfolio_standards": os.path.join(os.path.dirname(__file__), "fixtures/portfolio_standards.csv"),
        },
    }

    class MockSnakemakeWithBTM:
        def __init__(self):
            self.input = type(
                "obj",
                (object,),
                {
                    "rps_reeds": os.path.join(os.path.dirname(__file__), "fixtures/rps_reeds.csv"),
                    "ces_reeds": os.path.join(os.path.dirname(__file__), "fixtures/ces_reeds.csv"),
                    "small_scale_solar": os.path.join(os.path.dirname(__file__), "fixtures/small_scale_solar.csv"),
                },
            )
            self.params = type(
                "obj",
                (object,),
                {
                    "planning_horizons": [2030],
                },
            )

    snakemake = MockSnakemakeWithBTM()
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
        add_RPS_constraints(n, config, sector=False, snakemake=snakemake)

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
        add_RPS_constraints(n, config, sector=False, snakemake=snakemake)

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


def test_rooftop_solar_counts_toward_rps(policy_network, rps_config):
    """Rooftop solar (carrier 'solar-rooftop') must be credited toward the RPS.

    This is a regression test for the bug where 'solar-rooftop' was absent from
    RPS_CARRIERS, causing rooftop solar generators to be silently ignored when the
    optimizer evaluates RPS compliance.

    The test adds a fixed-output rooftop solar generator to the CA bus and sets an
    RPS percentage that can only be satisfied if rooftop solar counts.  If
    'solar-rooftop' is not in RPS_CARRIERS the constraint would be infeasible or
    the model would need to build significantly more utility-scale capacity.
    """
    from opts.policy import RPS_CARRIERS, add_RPS_constraints

    # Verify the fix is in place before running the optimisation
    assert "solar-rooftop" in RPS_CARRIERS, (
        "'solar-rooftop' is missing from RPS_CARRIERS — rooftop solar will not count toward RPS"
    )

    n = policy_network.copy()
    config, snakemake = rps_config

    solar_profile = pd.Series(0.5, index=n.snapshots)

    # Add a rooftop solar generator on the CA bus (z1)
    n.add(
        "Generator",
        "rooftop_solar_z1",
        bus="z1",
        p_nom=200,
        p_nom_extendable=False,
        carrier="solar-rooftop",
        capital_cost=0,
        marginal_cost=0,
        p_max_pu=solar_profile,
    )
    n.add("Carrier", "solar-rooftop", co2_emissions=0)

    def extra_functionality(n, _):
        add_RPS_constraints(n, config, sector=False, snakemake=snakemake)

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)

    assert any("rps_limit" in c for c in n.model.constraints), "No RPS limit constraints were added"

    # Verify that rooftop solar generation is counted on the LHS of the constraint
    region_buses = get_region_buses(n, ["CA"])
    rooftop_gens = n.generators[
        (n.generators.bus.isin(region_buses.index)) & (n.generators.carrier == "solar-rooftop")
    ]
    assert not rooftop_gens.empty, "Rooftop solar generator missing from CA region"

    rooftop_gen = n.generators_t.p[rooftop_gens.index].sum().sum()
    assert rooftop_gen > 0, "Rooftop solar generator produced no energy — check p_max_pu"

    # Confirm that total eligible generation (including rooftop) meets the RPS target
    eligible_carriers = ["solar", "solar-rooftop", "onwind"]
    eligible_gens = n.generators[
        (n.generators.bus.isin(region_buses.index)) & (n.generators.carrier.isin(eligible_carriers))
    ]
    eligible_gen = n.generators_t.p[eligible_gens.index].sum().sum()
    region_demand = n.loads_t.p_set.loc[:, n.loads.bus.isin(region_buses.index)].sum().sum()

    rps_pct = 0.9  # matches fixtures/portfolio_standards.csv for CA
    epsilon = 1e-3
    assert eligible_gen >= rps_pct * region_demand - epsilon, (
        f"RPS not met even with rooftop solar: {eligible_gen:.1f} MWh < {rps_pct * region_demand:.1f} MWh"
    )


def test_btm_solar_credit_reduces_rps_rhs(policy_network, rps_config, rps_config_with_btm):
    """BTM solar credit should reduce the required utility-scale renewable generation.

    The RPS constraint RHS is adjusted from ``pct * net_load`` to
    ``pct * net_load - (1 - pct) * rooftop_gen``.  This means when small-scale
    (behind-the-meter) solar data is provided, the optimizer needs to build
    *less* utility-scale renewable capacity to satisfy the same statutory target.
    The test confirms this by comparing solutions with and without the BTM data.

    Test network (from fixtures/small_scale_solar.csv):
      CA demand  = 300 MW × 24 h = 7 200 MWh,  btm_solar = 720 MWh
      CA pct     = 0.90  (from fixtures/portfolio_standards.csv)
      Without BTM:  rhs = 0.90 × 7200          = 6 480 MWh
      With BTM:     rhs = 0.90 × 7200 - 0.10 × 720 = 6 408 MWh  (72 MWh less)
    """
    from opts.policy import add_RPS_constraints

    # --- Solve WITHOUT BTM credit ---
    n_no_btm = policy_network.copy()
    config_no_btm, snakemake_no_btm = rps_config

    def extra_no_btm(n, _):
        add_RPS_constraints(n, config_no_btm, sector=False, snakemake=snakemake_no_btm)

    n_no_btm.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_no_btm)

    # --- Solve WITH BTM credit ---
    n_with_btm = policy_network.copy()
    config_with_btm, snakemake_with_btm = rps_config_with_btm

    def extra_with_btm(n, _):
        add_RPS_constraints(n, config_with_btm, sector=False, snakemake=snakemake_with_btm)

    n_with_btm.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_with_btm)

    # Measure CA utility-scale eligible generation in each solution
    region_buses_ca = get_region_buses(n_no_btm, ["CA"])
    eligible_carriers = ["solar", "onwind"]

    def _eligible_gen(n):
        gens = n.generators[
            (n.generators.bus.isin(region_buses_ca.index)) & (n.generators.carrier.isin(eligible_carriers))
        ]
        return n.generators_t.p[gens.index].sum().sum()

    gen_no_btm = _eligible_gen(n_no_btm)
    gen_with_btm = _eligible_gen(n_with_btm)

    logger.info(
        f"BTM credit test: CA eligible gen without BTM = {gen_no_btm:.1f} MWh, "
        f"with BTM = {gen_with_btm:.1f} MWh  (reduction = {gen_no_btm - gen_with_btm:.1f} MWh)"
    )

    assert gen_with_btm <= gen_no_btm + 1e-3, (
        f"BTM credit should reduce or not increase required renewable generation: "
        f"with_btm={gen_with_btm:.1f} > no_btm={gen_no_btm:.1f}"
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
