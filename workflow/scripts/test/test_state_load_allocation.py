"""``build_base_network.state_load_allocation_factors``: state shares on the national bus set."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from build_base_network import state_load_allocation_factors

pytestmark = pytest.mark.fast


@pytest.fixture
def memberships(tmp_path):
    fn = tmp_path / "membership.csv"
    pd.DataFrame({"ba": ["p1", "p2", "p48", "p60"], "st": ["WA", "OR", "TX", "TX"]}).to_csv(fn, index=False)
    return str(fn)


def buses(rows):
    """rows: bus_id -> (lon, lat, reeds_zone, county, full_state, load_weight)."""
    columns = ["lon", "lat", "reeds_zone", "county", "full_state", "load_weight"]
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    frame.index.name = "bus_id"
    return frame


def test_split_state_keeps_its_share(memberships):
    """El Paso's bus carries its share of Texas, not all of it."""
    gdf = buses(
        {
            1: (-106.4, 31.8, "p48", "48141", "Texas", 3.0),
            2: (-97.0, 32.8, "p60", "48113", "Texas", 60.0),
            3: (-95.4, 29.8, "p60", "48201", "Texas", 37.0),
        },
    )
    laf = pd.Series(state_load_allocation_factors(gdf, memberships), index=gdf.index)
    assert laf.tolist() == pytest.approx([0.03, 0.6, 0.37])


def test_normalised_over_the_reeds_state(memberships):
    """A bus whose point falls in Oregon but whose county is in Washington counts for Washington."""
    gdf = buses(
        {
            1: (-122.7, 45.6, "p1", "53011", "Oregon", 1.0),  # Clark County WA, point across the river
            2: (-122.6, 45.7, "p1", "53011", "Washington", 3.0),
            3: (-122.7, 45.5, "p2", "41051", "Oregon", 4.0),
        },
    )
    laf = pd.Series(state_load_allocation_factors(gdf, memberships), index=gdf.index)
    assert laf.tolist() == pytest.approx([0.25, 0.75, 1.0])


def test_duplicate_rows_count_once(memberships):
    gdf = buses(
        {
            1: (-122.6, 45.7, "p1", "53011", "Washington", 1.0),
            2: (-122.5, 47.6, "p1", "53033", "Washington", 3.0),
        },
    )
    gdf = pd.concat([gdf, gdf.loc[[1]]])
    laf = state_load_allocation_factors(gdf, memberships)
    assert laf.tolist() == pytest.approx([0.25, 0.75, 0.25])


def test_bus_outside_the_interconnects_shapes_takes_its_geographic_state(memberships):
    """ReEDS shapes cover only the modelled interconnect; ERCOT buses still count for Texas."""
    gdf = buses(
        {
            1: (-106.4, 31.8, "p48", "48141", "Texas", 3.0),
            2: (-97.0, 32.8, None, None, "Texas", 60.0),
            3: (-95.4, 29.8, None, None, "Texas", 37.0),
            4: (-98.0, 38.5, None, None, "Kansas", 50.0),
        },
    )
    laf = pd.Series(state_load_allocation_factors(gdf, memberships), index=gdf.index)
    assert laf.tolist() == pytest.approx([0.03, 0.6, 0.37, 1.0])


def test_network_state_overrides_the_national_key(memberships):
    """A bus the network files under another state (shape gap, nearest neighbour) counts for that state."""
    gdf = buses(
        {
            1: (-122.7, 45.6, None, None, "Oregon", 1.0),  # in a ReEDS shape gap on the border
            2: (-122.6, 45.7, "p1", "53011", "Washington", 3.0),
            3: (-122.7, 45.5, "p2", "41051", "Oregon", 4.0),
        },
    )
    network_state = pd.Series({"1": "WA", "2": "WA", "3": "OR"})
    laf = state_load_allocation_factors(gdf, memberships, network_state)
    assert laf.tolist() == pytest.approx([0.25, 0.75, 1.0])
    assert laf.groupby(network_state.to_numpy()).sum().tolist() == pytest.approx([1.0, 1.0])
