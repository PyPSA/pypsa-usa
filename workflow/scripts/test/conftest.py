"""
Common fixtures for PyPSA-USA tests.

This module contains shared fixtures used across multiple test files.
These fixtures provide reusable test data and network configurations.
"""
import os
import sys

import numpy as np
import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from _helpers import get_multiindex_snapshots  # noqa: E402


@pytest.fixture
def base_network():
    """
    Create a basic test network with common components.

    This is a base network that can be extended or modified by other fixtures.
    It includes:
    - Basic network structure with 3 buses
    - Common generation technologies (wind, solar, gas)
    - Basic load profiles
    - Standard time representation
    """
    # Create a simple network for testing
    n = pypsa.Network()

    # Add snapshots and investment periods
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 23:00", "inclusive": "both"},
        invest_periods=[2030],
    )
    n.set_investment_periods(periods=[2030])
    # Add buses with comprehensive attributes for different test cases
    n.add(
        "Bus",
        "z1",
        x=0,
        y=0,
        carrier="AC",
    )

    n.add(
        "Bus",
        "z2",
        x=1,
        y=1,
        carrier="AC",
    )

    n.add(
        "Bus",
        "z3",
        x=2,
        y=2,
        carrier="AC",
    )

    n.buses.loc[n.buses.index, "country"] = "US"
    n.buses.loc[n.buses.index, "interconnect"] = "west"
    n.buses.loc["z1", "region"] = "west"
    n.buses.loc["z2", "region"] = "east"
    n.buses.loc["z3", "region"] = "east"
    n.buses.loc["z1", "nerc_reg"] = "NERC1"
    n.buses.loc["z2", "nerc_reg"] = "NERC2"
    n.buses.loc["z3", "nerc_reg"] = "NERC2"
    n.buses.loc["z1", "reeds_state"] = "CA"
    n.buses.loc["z2", "reeds_state"] = "TX"
    n.buses.loc["z3", "reeds_state"] = "TX"
    n.buses.loc["z1", "reeds_zone"] = "CA_Z1"
    n.buses.loc["z2", "reeds_zone"] = "TX_Z1"
    n.buses.loc["z3", "reeds_zone"] = "TX_Z1"

    # Add versatile generators for different test scenarios
    # Wind generators
    n.add(
        "Generator",
        "wind1",
        bus="z1",
        p_nom=0,
        p_nom_extendable=True,
        carrier="onwind",
        capital_cost=1000,
        marginal_cost=0.1,
        p_max_pu=pd.Series(0.8, index=n.snapshots),
        p_nom_max=500,
        build_year=2030,
        lifetime=20,
    )

    n.add(
        "Generator",
        "wind2",
        bus="z2",
        p_nom=0,
        p_nom_extendable=True,
        carrier="onwind",
        capital_cost=1100,
        marginal_cost=0.12,
        p_max_pu=pd.Series(0.75, index=n.snapshots),
        p_nom_max=400,
        build_year=2030,
        lifetime=20,
    )

    n.generators.loc["wind1", "land_region"] = "region_a"
    n.generators.loc["wind2", "land_region"] = "region_b"

    # Solar generators
    solar_profile = pd.Series(
        np.concatenate([np.zeros(8), np.linspace(0, 1, 8), np.linspace(1, 0, 8)]),
        index=n.snapshots,
    )

    n.add(
        "Generator",
        "solar1",
        bus="z1",
        p_nom=0,
        p_nom_extendable=True,
        carrier="solar",
        capital_cost=800,
        marginal_cost=0.05,
        p_max_pu=solar_profile,
        p_nom_max=1000,
        build_year=2030,
        lifetime=20,
    )

    n.add(
        "Generator",
        "solar2",
        bus="z3",
        p_nom=0,
        p_nom_extendable=True,
        carrier="solar",
        capital_cost=750,
        marginal_cost=0.04,
        p_max_pu=solar_profile,
        p_nom_max=1500,
        build_year=2030,
        lifetime=20,
    )

    n.generators.loc["solar1", "land_region"] = "region_a"
    n.generators.loc["solar2", "land_region"] = "region_b"

    # Gas generators (non-renewable)
    n.add(
        "Generator",
        "gas1",
        bus="z1",
        p_nom=200,
        p_nom_extendable=False,
        carrier="gas",
        capital_cost=500,
        marginal_cost=20,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
        p_nom_max=5000,
        p_nom_min=200,
        build_year=2030,
        lifetime=20,
    )

    n.add(
        "Generator",
        "gas2",
        bus="z3",
        p_nom=200,
        p_nom_extendable=True,
        carrier="gas",
        capital_cost=450,
        marginal_cost=18,
        p_max_pu=pd.Series(1.0, index=n.snapshots),
        p_nom_max=10000,
        p_nom_min=200,
        build_year=2030,
        lifetime=20,
    )

    # Add storage
    n.add(
        "StorageUnit",
        "battery1",
        bus="z1",
        p_nom=0,
        p_nom_extendable=True,
        carrier="battery",
        capital_cost=300,
        marginal_cost=2,
        efficiency_store=0.85,
        efficiency_dispatch=0.85,
        standing_loss=0.01,
        max_hours=4,
        build_year=2030,
        lifetime=20,
    )

    # Add load
    n.add(
        "Load",
        "load1",
        bus="z1",
        p_set=pd.Series(300, index=n.snapshots),
    )

    n.add(
        "Load",
        "load2",
        bus="z2",
        carrier="AC",
        p_set=pd.Series(200, index=n.snapshots),
    )

    n.add(
        "Load",
        "load3",
        bus="z3",
        carrier="AC",
        p_set=pd.Series(300, index=n.snapshots),
    )

    # Add lines for power transfer between regions
    n.add(
        "Line",
        "line1",
        bus0="z1",
        bus1="z2",
        carrier="AC",
        x=0.1,
        r=0.01,
        s_nom=500,
        s_nom_min=500,
        capital_cost=300,
        s_nom_extendable=True,
    )

    n.add(
        "Line",
        "line2",
        bus0="z2",
        bus1="z3",
        carrier="AC",
        x=0.2,
        r=0.02,
        s_nom=300,
        s_nom_min=300,
        s_nom_extendable=True,
        capital_cost=300,
    )

    # Add links
    n.add(
        "Link",
        "link1",
        bus0="z1",
        bus1="z3",
        carrier="AC",
        p_nom=100,
        p_nom_min=100,
        p_nom_extendable=True,
        capital_cost=250,
    )

    # Define Carriers
    n.add(
        "Carrier",
        "nuclear",
        co2_emissions=0,
    )
    n.add(
        "Carrier",
        "onwind",
        co2_emissions=0,
    )
    n.add(
        "Carrier",
        "solar",
        co2_emissions=0,
    )
    n.add(
        "Carrier",
        "gas",
        co2_emissions=10,
    )
    n.add(
        "Carrier",
        "battery",
        co2_emissions=0,
    )
    n.add(
        "Carrier",
        "AC",
        co2_emissions=0,
    )
    return n
