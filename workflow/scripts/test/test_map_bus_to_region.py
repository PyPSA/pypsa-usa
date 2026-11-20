# BY PyPSA-USA Authors
"""``build_base_network.map_bus_to_region`` returns one row per bus.

The balancing-area and ReEDS shapes overlap along a few borders (NYISO/PJM/ISONE),
so a left spatial join puts a bus there once per polygon it falls in. Duplicate
``bus_id`` rows made ``assign_load_weight`` fail on reindexing and would count a
bus twice in its state's ``LAF_state`` denominator.
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


def _buses():
    """Bus 1 lies where the two shapes overlap, bus 2 only in the second, bus 3 in neither."""
    return gpd.GeoDataFrame(
        {"sub_id": [10, 20, 30]},
        index=pd.Index([1, 2, 3], name="bus_id"),
        geometry=[Point(1.5, 0.5), Point(2.5, 0.5), Point(9.0, 9.0)],
        crs=4326,
    )


def _overlapping_shapes(column):
    return gpd.GeoDataFrame(
        {column: ["A", "B"]},
        geometry=[box(0, 0, 2, 1), box(1, 0, 3, 1)],
        crs=4326,
    )


@pytest.mark.fast
def test_bus_in_two_polygons_keeps_its_first_match():
    out = map_bus_to_region(_buses(), _overlapping_shapes("balancing_area"), ["balancing_area"])
    assert out.index.tolist() == [1, 2, 3]
    assert out.loc[1, "balancing_area"] == "A"
    assert out.loc[2, "balancing_area"] == "B"
    assert pd.isna(out.loc[3, "balancing_area"])


@pytest.mark.fast
def test_successive_overlapping_joins_still_give_one_row_per_bus():
    """Two overlapping layers in a row used to multiply the rows (2 x 2 for bus 1)."""
    out = map_bus_to_region(_buses(), _overlapping_shapes("balancing_area"), ["balancing_area"])
    out = map_bus_to_region(out, _overlapping_shapes("reeds_zone"), ["reeds_zone"])
    assert out.index.is_unique
    assert out.loc[1, ["balancing_area", "reeds_zone"]].tolist() == ["A", "A"]


@pytest.mark.fast
def test_load_weight_works_after_overlapping_joins():
    out = map_bus_to_region(_buses(), _overlapping_shapes("county"), ["county"])
    weight = assign_load_weight(out, pd.Series({"A": 100.0, "B": 50.0}))
    assert weight.index.is_unique
    assert weight.loc[[1, 2]].tolist() == [100.0, 50.0]
