"""Unit test for the county -> grid cell population mapping in ``build_population_layouts``."""

import os
import sys

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from build_population_layouts import population_per_cell

pytestmark = pytest.mark.fast


def test_population_per_cell_with_geoid_index():
    """Counties are indexed by GEOID strings, as read from the census shapefile.

    atlite indexes the county shapes by integer position; under pandas 3 that
    raised ``KeyError`` on the GEOID index.
    """
    counties = gpd.GeoDataFrame(
        {"population": [100.0, 40.0], "rural_population": [10.0, 40.0]},
        geometry=[box(0, 0, 2, 1), box(2, 0, 3, 1)],
        index=["06001", "06003"],
        crs="EPSG:4326",
    )
    grid_cells = gpd.GeoSeries([box(x, 0, x + 1, 1) for x in range(3)], crs="EPSG:4326")

    pops = population_per_cell(counties, grid_cells, {"total": "population", "rural": "rural_population"})

    # the first county is split evenly over two cells, the second fills the third
    np.testing.assert_allclose(pops["total"].to_numpy(), [50.0, 50.0, 40.0])
    np.testing.assert_allclose(pops["rural"].to_numpy(), [5.0, 5.0, 40.0])
