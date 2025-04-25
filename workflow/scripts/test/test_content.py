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


# Tests
