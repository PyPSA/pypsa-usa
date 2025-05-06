"""
Test the content.py policy constraints functionality.

This module contains tests for the policy constraints in PyPSA-USA,
including Technology Capacity Targets (TCT) and Renewable Portfolio Standards (RPS).
"""

import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Fixtures


@pytest.fixture
def policy_network(base_network):
    """
    Adapt base network for policy constraint testing (RPS, TCT).

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
        capital_cost=3000,
        marginal_cost=5,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
        p_nom_max=1500,
    )

    return n


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
                    "planning_horizons": ["2030"],
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
