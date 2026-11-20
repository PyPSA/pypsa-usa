# BY PyPSA-USA Authors
"""``build_base_network.map_bus_to_region`` returns one row per bus.

Regression for the eastern/usa ``build_base_network`` crash: the BA and ReEDS
shapes overlap in places, a left spatial join returns a bus once per polygon
it sits in, and the duplicated ``bus_id`` rows then broke the population
load-weight reindex ("cannot reindex on an axis with duplicate labels") and
double-counted the affected buses in the ``LAF_state`` denominators.
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).parent.parent))

from build_base_network import map_bus_to_region
from build_bus_population import assign_load_weight

pytestmark = pytest.mark.fast


@pytest.fixture
def buses():
    """Three buses: inside zone A only, inside the A/B overlap, outside both."""
    return gpd.GeoDataFrame(
        {"sub_id": [1, 2, 3], "Pd": [10.0, 20.0, 30.0]},
        index=pd.Index(["only_a", "overlap", "outside"], name="bus_id"),
        geometry=[Point(0.5, 0.5), Point(1.5, 0.5), Point(9.0, 9.0)],
        crs=4326,
    )


@pytest.fixture
def overlapping_zones():
    """Zones A (x 0-2) and B (x 1-3) share the strip x 1-2."""
    return gpd.GeoDataFrame(
        {"county": ["pA", "pB"]},
        geometry=[box(0, 0, 2, 1), box(1, 0, 3, 1)],
        crs=4326,
    )


def test_bus_in_overlapping_shapes_is_not_duplicated(buses, overlapping_zones):
    mapped = map_bus_to_region(buses, overlapping_zones, ["county"])
    assert list(mapped.index) == ["only_a", "overlap", "outside"]
    assert mapped.index.name == "bus_id"


def test_first_match_is_kept_and_unmatched_bus_stays_missing(buses, overlapping_zones):
    mapped = map_bus_to_region(buses, overlapping_zones, ["county"])
    assert mapped.loc["only_a", "county"] == "pA"
    assert mapped.loc["overlap", "county"] == "pA"
    assert pd.isna(mapped.loc["outside", "county"])


def test_caller_names_list_is_not_mutated(buses, overlapping_zones):
    names = ["county"]
    map_bus_to_region(buses, overlapping_zones, names)
    assert names == ["county"]


def test_population_weight_runs_on_buses_mapped_through_overlapping_shapes(buses, overlapping_zones):
    """The step that crashed: weighting buses after a join onto overlapping shapes."""
    mapped = map_bus_to_region(buses, overlapping_zones, ["county"])
    weight = assign_load_weight(mapped, pd.Series({"pA": 1000.0, "pB": 400.0}))
    assert list(weight.index) == ["only_a", "overlap", "outside"]
    # pA's population splits evenly over its two substations; pB gets no bus
    assert weight["only_a"] == pytest.approx(500.0)
    assert weight["overlap"] == pytest.approx(500.0)
    assert weight["outside"] == 0.0
