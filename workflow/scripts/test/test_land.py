"""
Test the land use constraints functionality.

This module contains tests for the land use constraints in PyPSA-USA.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# Fixtures


@pytest.fixture
def land_use_network(base_network):
    """
    Adapt base network for land use constraint testing.

    Extends the base network with specific parameters relevant to land use constraints.
    """
    n = base_network.copy()

    # Add another generator in region_a to test constraints with multiple generators in the same region
    n.add(
        "Generator",
        "wind3",
        bus="z1",
        p_nom=0,
        p_nom_extendable=True,
        carrier="onwind",
        capital_cost=1050,
        marginal_cost=0.11,
        p_max_pu=pd.Series(0.85, index=n.snapshots),
        land_region="region_a",
        p_nom_max=300,
    )

    return n


# Tests
