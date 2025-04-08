"""
Test the content.py policy constraints functionality.

This module contains tests for the policy constraints in PyPSA-USA,
including Technology Capacity Targets (TCT) and Renewable Portfolio Standards (RPS).
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from opts.content import (  # noqa: E402
    add_RPS_constraints,
    add_technology_capacity_target_constraints,
)

# Fixtures


@pytest.fixture
def rps_test_network():
    """Create a network for testing RPS constraints."""
    # Create a simple network for testing
    n = pypsa.Network()

    # Add snapshots and investment periods
    n.set_snapshots(pd.date_range("2030-01-01 00:00", "2030-01-01 23:00", freq="H"))
    n.set_investment_periods(["2030"])

    # Add buses with reeds_state attribute for RPS constraints
    n.add("Bus", "bus1", x=0, y=0, country="US", reeds_state="CA", reeds_zone="CA_Z1")
    n.add("Bus", "bus2", x=1, y=1, country="US", reeds_state="CA", reeds_zone="CA_Z2")
    n.add("Bus", "bus3", x=2, y=2, country="US", reeds_state="TX", reeds_zone="TX_Z1")

    # Add renewable generators (eligible for RPS)
    n.add(
        "Generator",
        "solar_ca1",
        bus="bus1",
        p_nom=0,
        p_nom_extendable=True,
        carrier="solar",
        capital_cost=800,
        marginal_cost=0.05,
        p_max_pu=pd.Series(
            np.concatenate([np.zeros(8), np.linspace(0, 1, 8), np.linspace(1, 0, 8)]),
            index=n.snapshots,
        ),
    )

    n.add(
        "Generator",
        "wind_ca",
        bus="bus2",
        p_nom=0,
        p_nom_extendable=True,
        carrier="onwind",
        capital_cost=1000,
        marginal_cost=0.1,
        p_max_pu=pd.Series(0.8, index=n.snapshots),
    )

    n.add(
        "Generator",
        "solar_tx",
        bus="bus3",
        p_nom=0,
        p_nom_extendable=True,
        carrier="solar",
        capital_cost=750,
        marginal_cost=0.04,
        p_max_pu=pd.Series(
            np.concatenate([np.zeros(7), np.linspace(0, 1, 9), np.linspace(1, 0, 8)]),
            index=n.snapshots,
        ),
    )

    # Add non-renewable generators
    n.add(
        "Generator",
        "gas_ca",
        bus="bus1",
        p_nom=200,
        p_nom_extendable=True,
        carrier="gas",
        capital_cost=500,
        marginal_cost=20,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
    )

    n.add(
        "Generator",
        "gas_tx",
        bus="bus3",
        p_nom=200,
        p_nom_extendable=True,
        carrier="gas",
        capital_cost=450,
        marginal_cost=18,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
    )

    # Add loads
    n.add(
        "Load",
        "load_ca1",
        bus="bus1",
        p_set=pd.Series(300, index=n.snapshots),
    )

    n.add(
        "Load",
        "load_ca2",
        bus="bus2",
        p_set=pd.Series(200, index=n.snapshots),
    )

    n.add(
        "Load",
        "load_tx",
        bus="bus3",
        p_set=pd.Series(300, index=n.snapshots),
    )

    return n


@pytest.fixture
def tct_test_network():
    """Create a network for testing Technology Capacity Target constraints."""
    # Create a simple network for testing
    n = pypsa.Network()

    # Add snapshots and investment periods
    n.set_snapshots(pd.date_range("2030-01-01 00:00", "2030-01-01 23:00", freq="H"))
    n.set_investment_periods(["2030"])

    # Add buses
    n.add("Bus", "bus1", x=0, y=0, country="US", region="west")
    n.add("Bus", "bus2", x=1, y=1, country="US", region="west")
    n.add("Bus", "bus3", x=2, y=2, country="US", region="east")

    # Add technology types that would be targeted by TCT constraints
    n.add(
        "Generator",
        "wind_west1",
        bus="bus1",
        p_nom=0,
        p_nom_extendable=True,
        carrier="onwind",
        capital_cost=1000,
        marginal_cost=0.1,
        p_max_pu=pd.Series(0.8, index=n.snapshots),
    )

    n.add(
        "Generator",
        "wind_west2",
        bus="bus2",
        p_nom=0,
        p_nom_extendable=True,
        carrier="onwind",
        capital_cost=1100,
        marginal_cost=0.12,
        p_max_pu=pd.Series(0.75, index=n.snapshots),
    )

    n.add(
        "Generator",
        "wind_east",
        bus="bus3",
        p_nom=0,
        p_nom_extendable=True,
        carrier="onwind",
        capital_cost=950,
        marginal_cost=0.09,
        p_max_pu=pd.Series(0.85, index=n.snapshots),
    )

    # Add storage that could be targeted by TCT
    n.add(
        "StorageUnit",
        "battery_west",
        bus="bus1",
        p_nom=0,
        p_nom_extendable=True,
        carrier="battery",
        capital_cost=300,
        marginal_cost=0.01,
        efficiency_store=0.95,
        efficiency_dispatch=0.95,
        standing_loss=0.01,
        max_hours=6,
    )

    # Add other generation technologies
    n.add(
        "Generator",
        "gas_west",
        bus="bus2",
        p_nom=100,
        p_nom_extendable=True,
        carrier="gas",
        capital_cost=500,
        marginal_cost=30,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
    )

    n.add(
        "Generator",
        "gas_east",
        bus="bus3",
        p_nom=100,
        p_nom_extendable=True,
        carrier="gas",
        capital_cost=450,
        marginal_cost=28,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
    )

    # Add loads
    n.add(
        "Load",
        "load_west1",
        bus="bus1",
        p_set=pd.Series(200, index=n.snapshots),
    )

    n.add(
        "Load",
        "load_west2",
        bus="bus2",
        p_set=pd.Series(150, index=n.snapshots),
    )

    n.add(
        "Load",
        "load_east",
        bus="bus3",
        p_set=pd.Series(250, index=n.snapshots),
    )

    return n


@pytest.fixture
def rps_config():
    """Create a config dictionary and temporary files for RPS constraints."""
    # Create temporary portfolio standards file
    portfolio_standards_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    pd.DataFrame(
        [
            {
                "name": "CA_RPS",
                "region": "CA",
                "carrier": "solar, onwind",
                "planning_horizon": "2030",
                "pct": 0.5,
            },
            {
                "name": "TX_RPS",
                "region": "TX",
                "carrier": "solar, onwind",
                "planning_horizon": "2030",
                "pct": 0.3,
            },
        ],
    ).to_csv(portfolio_standards_file.name, index=False)

    # Create temporary RPS REEDS file
    rps_reeds_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    pd.DataFrame(
        [
            {"st": "CA", "t": "2030", "rps_all": 0.6},
            {"st": "TX", "t": "2030", "rps_all": 0.4},
        ],
    ).to_csv(rps_reeds_file.name, index=False)

    # Create temporary CES REEDS file
    ces_reeds_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    pd.DataFrame(
        [
            {"st": "CA", "t": "2030", "rps_all": 0.7},
            {"st": "TX", "t": "2030", "rps_all": 0.5},
        ],
    ).to_csv(ces_reeds_file.name, index=False)

    # Create config dictionary
    config = {
        "electricity": {
            "portfolio_standards": portfolio_standards_file.name,
        },
    }

    # Create a mock snakemake object
    class MockSnakemake:
        def __init__(self):
            self.input = type(
                "obj",
                (object,),
                {
                    "rps_reeds": rps_reeds_file.name,
                    "ces_reeds": ces_reeds_file.name,
                },
            )
            self.params = type(
                "obj",
                (object,),
                {
                    "planning_horizons": ["2030"],
                },
            )

    snakemake = MockSnakemake()

    yield config, snakemake

    # Clean up temporary files
    os.unlink(portfolio_standards_file.name)
    os.unlink(rps_reeds_file.name)
    os.unlink(ces_reeds_file.name)


@pytest.fixture
def tct_config():
    """Create a config dictionary and temporary files for TCT constraints."""
    # Create temporary technology capacity targets file
    tct_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    pd.DataFrame(
        [
            {
                "name": "wind_west_min",
                "region": "west",
                "carrier": "onwind",
                "planning_horizon": "2030",
                "min": 300,
                "max": float("nan"),
            },
            {
                "name": "wind_west_max",
                "region": "west",
                "carrier": "onwind",
                "planning_horizon": "2030",
                "min": float("nan"),
                "max": 500,
            },
            {
                "name": "battery_west_min",
                "region": "west",
                "carrier": "battery",
                "planning_horizon": "2030",
                "min": 100,
                "max": float("nan"),
            },
        ],
    ).to_csv(tct_file.name, index=False)

    # Create config dictionary
    config = {
        "electricity": {
            "technology_capacity_targets": tct_file.name,
        },
    }

    yield config

    # Clean up temporary file
    os.unlink(tct_file.name)


# Tests for RPS constraints


def test_rps_constraint_creation(rps_test_network, rps_config):
    """Test that RPS constraints are correctly created."""
    n = rps_test_network.copy()
    config, snakemake = rps_config

    # Prepare the optimization
    n.lopf(pyomo=False, solver_name="glpk", formulation="kirchhoff")

    # Add RPS constraints
    add_RPS_constraints(n, config, sector=False, snakemake=snakemake)

    # Check that RPS constraints were created
    rps_constraints = [c for c in n.model.constraints if "_rps_limit" in c]
    assert len(rps_constraints) > 0, "RPS constraints should be created"

    # Check that we have a constraint for each region in the portfolio standards
    ca_rps_constraints = [c for c in rps_constraints if "CA_RPS" in c]
    tx_rps_constraints = [c for c in rps_constraints if "TX_RPS" in c]

    assert len(ca_rps_constraints) > 0, "CA should have RPS constraints"
    assert len(tx_rps_constraints) > 0, "TX should have RPS constraints"


def test_rps_constraint_effect(rps_test_network, rps_config):
    """Test that RPS constraints influence the generation mix."""
    n = rps_test_network.copy()
    config, snakemake = rps_config

    # First, run optimization without RPS to get baseline
    n.lopf(pyomo=False, solver_name="glpk", formulation="kirchhoff")

    # Store baseline generation results for each carrier
    no_rps_renewable_gen_ca = n.generators.loc[
        (n.generators.carrier.isin(["solar", "onwind"])) & (n.generators.bus.isin(["bus1", "bus2"])),
        "p_nom_opt",
    ].sum()

    no_rps_total_gen_ca = n.generators.loc[
        n.generators.bus.isin(["bus1", "bus2"]),
        "p_nom_opt",
    ].sum()

    no_rps_renewable_fraction_ca = no_rps_renewable_gen_ca / no_rps_total_gen_ca if no_rps_total_gen_ca > 0 else 0

    # Now run with RPS constraints
    n = rps_test_network.copy()

    # Set a high cost for gas to exaggerate RPS effect
    n.generators.loc[n.generators.carrier == "gas", "marginal_cost"] = 50

    # Prepare optimization with RPS constraints
    def add_rps_extra(n):
        add_RPS_constraints(n, config, sector=False, snakemake=snakemake)

    n.lopf(pyomo=False, solver_name="glpk", formulation="kirchhoff", extra_functionality=add_rps_extra)

    # Analyze results with RPS
    with_rps_renewable_gen_ca = n.generators.loc[
        (n.generators.carrier.isin(["solar", "onwind"])) & (n.generators.bus.isin(["bus1", "bus2"])),
        "p_nom_opt",
    ].sum()

    with_rps_total_gen_ca = n.generators.loc[
        n.generators.bus.isin(["bus1", "bus2"]),
        "p_nom_opt",
    ].sum()

    with_rps_renewable_fraction_ca = (
        with_rps_renewable_gen_ca / with_rps_total_gen_ca if with_rps_total_gen_ca > 0 else 0
    )

    # The RPS should generally increase the renewable percentage
    # CA has a 50% RPS requirement in our test
    assert with_rps_renewable_fraction_ca >= 0.5, "CA should have at least 50% renewable generation with RPS"

    # Check if RPS actually influenced the results (it may not if baseline was already compliant)
    if no_rps_renewable_fraction_ca < 0.5:
        assert (
            with_rps_renewable_fraction_ca > no_rps_renewable_fraction_ca
        ), "RPS should increase renewable fraction if baseline was below target"


def test_rps_constraint_multi_region(rps_test_network, rps_config):
    """Test RPS constraints applied to multiple regions with different targets."""
    n = rps_test_network.copy()
    config, snakemake = rps_config

    # Make gas much more attractive in TX to create a natural low-renewable situation
    n.generators.loc[n.generators.bus == "bus3", "marginal_cost"] = 5

    # Make the optimization with RPS
    def add_rps_extra(n):
        add_RPS_constraints(n, config, sector=False, snakemake=snakemake)

    n.lopf(pyomo=False, solver_name="glpk", formulation="kirchhoff", extra_functionality=add_rps_extra)

    # Check California (50% RPS)
    ca_renewable_gen = n.generators.loc[
        (n.generators.carrier.isin(["solar", "onwind"])) & (n.generators.bus.isin(["bus1", "bus2"])),
        "p_nom_opt",
    ].sum()

    ca_total_gen = n.generators.loc[
        n.generators.bus.isin(["bus1", "bus2"]),
        "p_nom_opt",
    ].sum()

    ca_renewable_fraction = ca_renewable_gen / ca_total_gen if ca_total_gen > 0 else 0

    # Check Texas (30% RPS)
    tx_renewable_gen = n.generators.loc[
        (n.generators.carrier.isin(["solar", "onwind"])) & (n.generators.bus == "bus3"),
        "p_nom_opt",
    ].sum()

    tx_total_gen = n.generators.loc[
        n.generators.bus == "bus3",
        "p_nom_opt",
    ].sum()

    tx_renewable_fraction = tx_renewable_gen / tx_total_gen if tx_total_gen > 0 else 0

    # Each region should meet its RPS target
    assert ca_renewable_fraction >= 0.5, "CA should meet its 50% RPS target"
    assert tx_renewable_fraction >= 0.3, "TX should meet its 30% RPS target"


# Tests for Technology Capacity Target constraints


def test_tct_constraint_creation(tct_test_network, tct_config):
    """Test that Technology Capacity Target constraints are correctly created."""
    n = tct_test_network.copy()

    # Prepare the optimization
    n.lopf(pyomo=False, solver_name="glpk", formulation="kirchhoff")

    # Add TCT constraints
    add_technology_capacity_target_constraints(n, tct_config)

    # Check that TCT constraints were created
    min_constraints = [c for c in n.model.constraints if "_min" in c]
    max_constraints = [c for c in n.model.constraints if "_max" in c]

    assert len(min_constraints) > 0, "Minimum TCT constraints should be created"
    assert len(max_constraints) > 0, "Maximum TCT constraints should be created"

    # Check that we have a constraint for each TCT in the config
    wind_west_min = [c for c in min_constraints if "wind_west_min" in c]
    wind_west_max = [c for c in max_constraints if "wind_west_max" in c]
    battery_west_min = [c for c in min_constraints if "battery_west_min" in c]

    assert len(wind_west_min) > 0, "wind_west_min constraint should be created"
    assert len(wind_west_max) > 0, "wind_west_max constraint should be created"
    assert len(battery_west_min) > 0, "battery_west_min constraint should be created"


def test_tct_minimum_constraint(tct_test_network, tct_config):
    """Test that minimum Technology Capacity Target constraints are enforced."""
    n = tct_test_network.copy()

    # Make wind less attractive to ensure it wouldn't be built without constraint
    n.generators.loc[n.generators.carrier == "onwind", "capital_cost"] = 2000

    # Run optimization with TCT constraints
    def add_tct_extra(n):
        add_technology_capacity_target_constraints(n, tct_config)

    n.lopf(pyomo=False, solver_name="glpk", formulation="kirchhoff", extra_functionality=add_tct_extra)

    # Check that the minimum wind capacity constraint in the west region is met
    west_wind_capacity = n.generators.loc[
        (n.generators.carrier == "onwind") & (n.generators.bus.isin(["bus1", "bus2"])),
        "p_nom_opt",
    ].sum()

    assert west_wind_capacity >= 300, "West region should have at least 300 MW of wind capacity"

    # Check that the minimum battery capacity constraint in the west region is met
    west_battery_capacity = n.storage_units.loc[
        (n.storage_units.carrier == "battery") & (n.storage_units.bus.isin(["bus1", "bus2"])),
        "p_nom_opt",
    ].sum()

    assert west_battery_capacity >= 100, "West region should have at least 100 MW of battery capacity"


def test_tct_maximum_constraint(tct_test_network, tct_config):
    """Test that maximum Technology Capacity Target constraints are enforced."""
    n = tct_test_network.copy()

    # Make wind very attractive to ensure it would be built to maximum without constraint
    n.generators.loc[n.generators.carrier == "onwind", "capital_cost"] = 100
    n.generators.loc[n.generators.carrier == "onwind", "marginal_cost"] = 0.01

    # Make gas expensive to ensure wind would be preferred
    n.generators.loc[n.generators.carrier == "gas", "marginal_cost"] = 50

    # Increase load to require more generation
    n.loads.p_set = n.loads.p_set * 3

    # Run optimization with TCT constraints
    def add_tct_extra(n):
        add_technology_capacity_target_constraints(n, tct_config)

    n.lopf(pyomo=False, solver_name="glpk", formulation="kirchhoff", extra_functionality=add_tct_extra)

    # Check that the maximum wind capacity constraint in the west region is respected
    west_wind_capacity = n.generators.loc[
        (n.generators.carrier == "onwind") & (n.generators.bus.isin(["bus1", "bus2"])),
        "p_nom_opt",
    ].sum()

    assert west_wind_capacity <= 500, "West region should have at most 500 MW of wind capacity"

    # Wind in the east region should not be constrained
    east_wind_capacity = n.generators.loc[
        (n.generators.carrier == "onwind") & (n.generators.bus == "bus3"),
        "p_nom_opt",
    ].sum()

    # East wind should be built more than west wind if truly unconstrained
    assert east_wind_capacity > 0, "East region should build wind capacity"


def test_tct_existing_value(tct_test_network, tct_config):
    """Test 'existing' value handling in Technology Capacity Target constraints."""
    n = tct_test_network.copy()

    # Set non-zero existing capacity for wind
    n.generators.loc["wind_west1", "p_nom"] = 50
    n.generators.loc["wind_west1", "p_nom_extendable"] = False

    # Create a modified TCT file with "existing" values
    tct_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    pd.DataFrame(
        [
            {
                "name": "wind_west_existing_min",
                "region": "west",
                "carrier": "onwind",
                "planning_horizon": "2030",
                "min": "existing",
                "max": float("nan"),
            },
            {
                "name": "wind_west_existing_max",
                "region": "west",
                "carrier": "onwind",
                "planning_horizon": "2030",
                "min": float("nan"),
                "max": "existing",
            },
        ],
    ).to_csv(tct_file.name, index=False)

    # Update config with new file
    modified_config = {
        "electricity": {
            "technology_capacity_targets": tct_file.name,
        },
    }

    # Prepare optimization
    n.lopf(pyomo=False, solver_name="glpk", formulation="kirchhoff")

    # Add TCT constraints with modified config
    add_technology_capacity_target_constraints(n, modified_config)

    # Clean up
    os.unlink(tct_file.name)

    # Check that the constraints were created (the function should handle "existing" values)
    min_constraints = [c for c in n.model.constraints if "wind_west_existing_min" in c]
    max_constraints = [c for c in n.model.constraints if "wind_west_existing_max" in c]

    assert len(min_constraints) > 0, "Min constraint with 'existing' value should be created"
    assert len(max_constraints) > 0, "Max constraint with 'existing' value should be created"
