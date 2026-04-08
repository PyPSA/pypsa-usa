"""
Test the coords/layoutmatrix/iloc logic in build_renewable_profiles (average distance loop).

Run from workflow dir (no conda needed):
  python scripts/test/test_build_renewable_profiles_coords.py

With conda env pypsa-usa:
  pytest scripts/test/test_build_renewable_profiles_coords.py -v
"""

import os
import sys

import numpy as np
import pandas as pd

# Add workflow/scripts to path for pypsa.geo.haversine when available
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from pypsa.geo import haversine
except ImportError:
    # Minimal haversine mock: (bus_coords_1d, coords_df) -> array of distances
    def haversine(bus_coords, coords_df):
        bus = np.array([bus_coords["x"], bus_coords["y"]])
        pts = coords_df[["x", "y"]].values.T
        return np.linalg.norm(pts - bus[:, None], axis=0)


def test_coords_iloc_loop():
    """Simulate the atlite average-distance loop: coords from (y,x), row from layoutmatrix, co = coords.iloc[idx]."""
    np.random.seed(42)
    n_y, n_x = 10, 12
    n_spatial = n_y * n_x
    n_buses = 3
    bus_names = [f"bus_{i}" for i in range(n_buses)]

    # Synthetic (y, x) like availability.coords -> coords in stack order
    y_vals = np.linspace(30, 40, n_y)
    x_vals = np.linspace(-120, -110, n_x)
    yy, xx = np.meshgrid(y_vals, x_vals, indexing="ij")
    coords = pd.DataFrame({"x": xx.ravel(), "y": yy.ravel()})
    assert len(coords) == n_spatial

    # Synthetic layoutmatrix (bus, spatial) - plain array, one row per bus
    layoutmatrix = np.random.rand(n_buses, n_spatial).astype(np.float64)

    # Synthetic bus_coords (one point per bus)
    bus_coords = pd.DataFrame(
        {"x": np.random.rand(n_buses) * 5 - 115, "y": np.random.rand(n_buses) * 5 + 35},
        index=bus_names,
    )

    average_distance = []
    centre_of_mass = []
    for i, bus in enumerate(bus_names):
        row = layoutmatrix[i].flatten()
        nz_b = row != 0
        row = row[nz_b]
        idx = np.where(nz_b)[0]
        co = coords.iloc[idx]
        distances = haversine(bus_coords.loc[bus], co)
        average_distance.append((distances * (row / row.sum())).sum())
        centre_of_mass.append(co.values.T @ (row / row.sum()))

    assert len(average_distance) == n_buses
    assert len(centre_of_mass) == n_buses
    for cm in centre_of_mass:
        assert cm.shape == (2,)  # x, y
    print("test_coords_iloc_loop passed: coords.iloc[idx] and haversine/centre_of_mass OK")


def test_coords_length_matches_spatial():
    """Ensure coords built from product (y,x) has same length as layoutmatrix spatial dim."""
    n_y, n_x = 15, 20
    n_spatial = n_y * n_x
    y_vals = np.linspace(25, 45, n_y)
    x_vals = np.linspace(-125, -100, n_x)
    yy, xx = np.meshgrid(y_vals, x_vals, indexing="ij")
    coords = pd.DataFrame({"x": xx.ravel(), "y": yy.ravel()})
    assert len(coords) == n_spatial
    # Simulate one bus row
    row = np.random.rand(n_spatial)
    nz_b = row > 0.3
    idx = np.where(nz_b)[0]
    co = coords.iloc[idx]
    assert len(co) == idx.size
    print("test_coords_length_matches_spatial passed")


def test_coords_from_product_when_layout_and_availability_differ():
    """
    When layout and availability have different grid sizes (e.g. onwind/solar vs offshore),
    product = layout * availability has the broadcast grid. Coords must be built from product,
    not from availability alone, or idx can be out-of-bounds.
    """
    np.random.seed(42)
    # "availability" grid (e.g. smaller for some tech)
    n_y_a, n_x_a = 6, 8
    n_spatial_a = n_y_a * n_x_a
    # "layout" / product grid (e.g. full cutout - larger)
    n_y_p, n_x_p = 10, 12
    n_spatial_p = n_y_p * n_x_p
    n_buses = 2
    bus_names = [f"bus_{i}" for i in range(n_buses)]

    # Wrong: coords from "availability" grid only -> len 48
    y_a = np.linspace(30, 35, n_y_a)
    x_a = np.linspace(-120, -115, n_x_a)
    yy_a, xx_a = np.meshgrid(y_a, x_a, indexing="ij")
    coords_wrong = pd.DataFrame({"x": xx_a.ravel(), "y": yy_a.ravel()})
    assert len(coords_wrong) == n_spatial_a

    # Right: coords from "product" grid -> len 120
    y_p = np.linspace(30, 40, n_y_p)
    x_p = np.linspace(-120, -110, n_x_p)
    yy_p, xx_p = np.meshgrid(y_p, x_p, indexing="ij")
    coords = pd.DataFrame({"x": xx_p.ravel(), "y": yy_p.ravel()})
    assert len(coords) == n_spatial_p

    # layoutmatrix has product's spatial size (120)
    layoutmatrix = np.random.rand(n_buses, n_spatial_p).astype(np.float64)
    bus_coords = pd.DataFrame(
        {"x": [-115, -112], "y": [35, 37]},
        index=bus_names,
    )

    for i, bus in enumerate(bus_names):
        row = layoutmatrix[i].flatten()
        nz_b = row != 0
        row = row[nz_b]
        idx = np.where(nz_b)[0]
        # Would fail: co = coords_wrong.iloc[idx]  # IndexError: indices out-of-bounds (idx up to 119, len 48)
        co = coords.iloc[idx]
        _ = haversine(bus_coords.loc[bus], co)
    print("test_coords_from_product_when_layout_and_availability_differ passed")


if __name__ == "__main__":
    test_coords_iloc_loop()
    test_coords_length_matches_spatial()
    test_coords_from_product_when_layout_and_availability_differ()
    print("All tests passed.")
