"""
Aggregate raw GODEEEP capacity factors to PyPSA buses, weighted by NREL land
availability.

For each bus b and time t:
    cf_bus[b, t] = sum_{cells c in b}(avail[c] * cf[c, t]) / sum_{c in b}(avail[c])

Buses with zero total availability (fully excluded) are dropped. The output also
carries `bus_avail` (the per-bus mean availability over contributing cells) and
`n_cells` (count of contributing cells) for diagnostics.

This is the runtime-side counterpart to `build_nrel_availability.py`:
availability is computed once per (tech, access); this script pulls it together
with a raw GODEEEP file and bus shapes at runtime.
"""

import argparse
import calendar
import hashlib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


def _shapes_cache_key(shape_paths: list[str]) -> str:
    """Stable short hash of bus-shape inputs (path + size + mtime)."""
    parts = []
    for p in sorted(shape_paths):
        st = Path(p).stat()
        parts.append(f"{p}|{st.st_size}|{int(st.st_mtime)}")
    return hashlib.md5("\n".join(parts).encode()).hexdigest()[:12]


def get_cell_to_bus_mapping(
    xlong: np.ndarray,
    xlat: np.ndarray,
    shape_paths: list[str],
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Return cell→bus mapping, cached on disk by bus-shape contents.

    `shape_paths` is one or more geojson files whose features get unioned into
    a single bus set. Mapping depends only on these shapes and the (fixed)
    GODEEEP grid, so results are cached by `cache_dir` when provided.
    """
    if cache_dir is None:
        return build_cell_to_bus_mapping(xlong, xlat, shape_paths)

    cache_path = Path(cache_dir) / f"cell_to_bus_{_shapes_cache_key(shape_paths)}.parquet"
    if cache_path.exists():
        print(f"[mapping] cache HIT: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"[mapping] cache MISS, building → {cache_path}")
    mapping = build_cell_to_bus_mapping(xlong, xlat, shape_paths)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_parquet(cache_path)
    return mapping


def build_cell_to_bus_mapping(
    xlong: np.ndarray,
    xlat: np.ndarray,
    shape_paths: list[str],
) -> pd.DataFrame:
    """Return a DataFrame with columns [NS, EW, name] mapping cells to buses.

    Mirrors aggregate_godeeep.py: spatial join cell centers to bus polygons,
    with a nearest-neighbor fallback for buses that receive no cells.
    """
    ny, nx = xlong.shape
    coords = pd.DataFrame(
        {
            "x": xlong.flatten(),
            "y": xlat.flatten(),
            "NS": np.repeat(np.arange(ny), nx),
            "EW": np.tile(np.arange(nx), ny),
        }
    )
    points = gpd.GeoDataFrame(
        coords,
        geometry=gpd.points_from_xy(coords["x"], coords["y"]),
        crs="EPSG:4326",
    )

    shapes = [gpd.read_file(p).to_crs(4326) for p in shape_paths]
    gdf_shapes = pd.concat(shapes, axis=0, ignore_index=True)[["name", "geometry"]]

    joined = gpd.sjoin(gdf_shapes, points, how="left", predicate="intersects")
    joined = joined.drop_duplicates(subset=["name", "NS", "EW"], keep="first")

    # Nearest-neighbor fill for buses with no intersecting cells
    missing = joined["NS"].isna()
    if missing.any():
        centroids = (
            joined.loc[missing, "geometry"]
            .to_crs("EPSG:3857")
            .centroid.to_crs(joined.crs)
        )
        valid = points.dropna(subset=["NS", "EW"]).to_crs("EPSG:3857")
        tree = cKDTree(np.c_[valid.geometry.x, valid.geometry.y])
        centroids_proj = centroids.to_crs("EPSG:3857")
        _, idx = tree.query(np.c_[centroids_proj.x, centroids_proj.y])
        nearest = points.iloc[idx]
        for col in ("NS", "EW", "x", "y"):
            joined.loc[missing, col] = nearest[col].values

    joined["NS"] = joined["NS"].astype(int)
    joined["EW"] = joined["EW"].astype(int)
    return joined[["name", "NS", "EW"]].reset_index(drop=True)


def weighted_bus_aggregation(
    cf: xr.DataArray,
    avail: xr.DataArray,
    mapping: pd.DataFrame,
    chunk_t: int = 500,
) -> xr.Dataset:
    """Compute availability-weighted bus profile + diagnostics.

    cf:       (time, south_north, west_east) float; read in chunks over time to
              bound memory (important for uint8-encoded files that decode to
              ~4 GB float32 if materialized at once).
    avail:    (south_north, west_east) float in [0, 1], NaN where outside raster
    mapping:  DataFrame with columns [name, NS, EW]
    """
    av_arr = avail.values.astype(np.float64)
    av_arr = np.where(np.isnan(av_arr), 0.0, av_arr)

    buses = sorted(mapping["name"].unique())
    bus_idx = {b: i for i, b in enumerate(buses)}
    B = len(buses)
    T = cf.sizes["time"]

    # Precompute vectors aligned to mapping rows
    row_bus = mapping["name"].map(bus_idx).to_numpy(dtype=np.int64)
    row_ns = mapping["NS"].to_numpy(dtype=np.int64)
    row_ew = mapping["EW"].to_numpy(dtype=np.int64)
    row_w = av_arr[row_ns, row_ew]                     # weight per (cell, bus) row
    keep = row_w > 0
    row_bus_k = row_bus[keep]
    row_ns_k = row_ns[keep]
    row_ew_k = row_ew[keep]
    row_w_k = row_w[keep]

    # Per-bus diagnostics (cheap, before streaming CF)
    avail_sum = np.zeros(B, dtype=np.float64)
    n_cells = np.zeros(B, dtype=np.int32)
    np.add.at(avail_sum, row_bus, row_w)
    np.add.at(n_cells, row_bus, 1)
    den = np.zeros(B, dtype=np.float64)
    np.add.at(den, row_bus_k, row_w_k)

    num = np.zeros((T, B), dtype=np.float64)           # ~8760*B*8 bytes ≈ 333 MB @ B=4751

    for t0 in range(0, T, chunk_t):
        t1 = min(t0 + chunk_t, T)
        chunk = cf.isel(time=slice(t0, t1)).values     # (t1-t0, NS, EW), forces decode of one slab
        # Gather CF at the (NS, EW) of each kept mapping row -> (t1-t0, n_kept)
        cf_rows = chunk[:, row_ns_k, row_ew_k]
        cf_rows *= row_w_k                             # broadcast weights over time
        # Scatter-add weighted CF into bus columns
        for ti in range(t1 - t0):
            np.add.at(num[t0 + ti], row_bus_k, cf_rows[ti])
        del chunk, cf_rows

    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(den > 0, num / den, np.nan)
    mean_avail = np.where(n_cells > 0, avail_sum / n_cells, np.nan)

    ds = xr.Dataset(
        {
            "profile": (("time", "bus"), profile.astype(np.float32)),
            "bus_avail": (("bus",), mean_avail.astype(np.float32)),
            "n_cells": (("bus",), n_cells),
        },
        coords={"time": cf["time"].values, "bus": buses},
    )
    return ds


def fix_godeeep_time(ds: xr.Dataset, year: int) -> xr.Dataset:
    """Replicate the time-correction logic from aggregate_godeeep.py."""
    missing_time = pd.Timestamp(f"{year}-01-01T00:00:00")
    time_index = pd.DatetimeIndex(pd.to_datetime(ds["Time"].values, unit="ns"))
    if missing_time.value not in ds["Time"].values:
        time_index = time_index - pd.Timedelta(hours=1)

    if calendar.isleap(year) and pd.Timestamp(f"{year}-02-29") in time_index:
        leap_day = pd.Timestamp(f"{year}-02-29")
        time_index = pd.DatetimeIndex(
            [t + pd.Timedelta(days=1) if t >= leap_day else t for t in time_index]
        )
    return ds.assign_coords(Time=time_index).rename({"Time": "time"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--godeeep", required=True, help="Raw GODEEEP NetCDF.")
    ap.add_argument("--avail", required=True, help="avail[NS, EW] NetCDF.")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--onshore-shapes", required=True)
    ap.add_argument("--offshore-shapes", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--mapping-cache-dir",
        default=None,
        help="If set, cache cell→bus mapping here (~14 min first run, seconds thereafter).",
    )
    args = ap.parse_args()

    print(f"[agg] loading godeeep: {args.godeeep}")
    ds = xr.open_dataset(args.godeeep)
    ds = fix_godeeep_time(ds, args.year)
    ds = ds.rename({"XLONG": "x", "XLAT": "y"})
    cf = ds["capacity_factor"]
    print(f"[agg] cf shape: {cf.shape}")

    print(f"[agg] loading avail: {args.avail}")
    avail = xr.open_dataarray(args.avail)
    assert avail.shape == cf.shape[1:], (
        f"avail shape {avail.shape} must match cf spatial shape {cf.shape[1:]}"
    )

    print("[agg] building cell->bus mapping…")
    shape_paths = [args.onshore_shapes]
    if args.offshore_shapes:
        shape_paths.append(args.offshore_shapes)
    mapping = get_cell_to_bus_mapping(
        ds["x"].values, ds["y"].values,
        shape_paths,
        cache_dir=args.mapping_cache_dir,
    )
    print(f"[agg] {mapping['name'].nunique()} buses, {len(mapping)} cell-bus rows")

    print("[agg] computing weighted aggregation…")
    out = weighted_bus_aggregation(cf, avail, mapping)

    n_total = out.sizes["bus"]
    n_valid = int((~np.isnan(out["profile"].isel(time=0))).sum())
    print(
        f"[agg] buses: {n_valid}/{n_total} have nonzero availability; "
        f"mean bus_avail={float(out['bus_avail'].mean(skipna=True)):.3f}  "
        f"profile mean={float(out['profile'].mean(skipna=True)):.3f}"
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(args.output)
    print(f"[agg] wrote {args.output}")


if __name__ == "__main__":
    main()
