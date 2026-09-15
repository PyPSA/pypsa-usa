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

Two weightings are available for the simplify-early DAG, where the target bus
is a ``s{simpl}`` cluster rather than a substation:

* ``weighted_bus_aggregation`` — one availability-weighted mean over every cell
  of the cluster (develop's behaviour before 2026-09-15).
* ``capacity_weighted_bus_aggregation`` — two stages collapsed into a single
  scatter-add: availability-weighted within each substation, then
  **installable-capacity** (NREL caps ``p_nom_max``) weighted across the
  substations of a cluster, by re-weighting each (cell, substation) row with
  ``a_c * p_nom_max_sub / A_sub``.

The two coincide only when MW of NREL caps per unit availability is uniform
across the substations of a cluster.

Neither reproduces `master`, and ``capacity`` deliberately does not try to.
Master reaches the cluster profile through `simplify_network`, where pypsa
0.30.2's ``clustering/spatial.py:aggregateoneport`` weights ``p_max_pu`` by
**existing** ``p_nom`` and (via ``normed_or_uniform``) falls back to a plain
arithmetic mean when the group's ``p_nom`` sums to zero. On western, 12 of 20
onwind clusters carry no existing capacity at all, so master's cluster CF there
is the unweighted mean of its substations' CFs: a 1 MW polygon counts as much as
a 10 GW one. Develop weights by installable capacity instead, because that is
what an extendable resource generator's profile will actually be applied to.
Logged as hot-fix HF-25 (``tests/equivalence/hotfixes.yaml``) and deltas-ledger
row DL-19.
"""

import argparse
import calendar
import hashlib
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


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
        },
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
        centroids = joined.loc[missing, "geometry"].to_crs("EPSG:3857").centroid.to_crs(joined.crs)
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
    bus_count = len(buses)
    time_count = cf.sizes["time"]

    # Precompute vectors aligned to mapping rows
    row_bus = mapping["name"].map(bus_idx).to_numpy(dtype=np.int64)
    row_ns = mapping["NS"].to_numpy(dtype=np.int64)
    row_ew = mapping["EW"].to_numpy(dtype=np.int64)
    row_w = av_arr[row_ns, row_ew]  # weight per (cell, bus) row
    keep = row_w > 0
    row_bus_k = row_bus[keep]
    row_ns_k = row_ns[keep]
    row_ew_k = row_ew[keep]
    row_w_k = row_w[keep]

    # Per-bus diagnostics (cheap, before streaming CF)
    avail_sum = np.zeros(bus_count, dtype=np.float64)
    n_cells = np.zeros(bus_count, dtype=np.int32)
    np.add.at(avail_sum, row_bus, row_w)
    np.add.at(n_cells, row_bus, 1)
    den = np.zeros(bus_count, dtype=np.float64)
    np.add.at(den, row_bus_k, row_w_k)

    num = np.zeros((time_count, bus_count), dtype=np.float64)  # ~8760*bus_count*8 bytes ≈ 333 MB @ B=4751

    for t0 in range(0, time_count, chunk_t):
        t1 = min(t0 + chunk_t, time_count)
        chunk = cf.isel(time=slice(t0, t1)).values  # (t1-t0, NS, EW), forces decode of one slab
        # Gather CF at the (NS, EW) of each kept mapping row -> (t1-t0, n_kept)
        cf_rows = chunk[:, row_ns_k, row_ew_k]
        cf_rows *= row_w_k  # broadcast weights over time
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


def _as_series(obj) -> pd.Series:
    """Coerce a Series / DataArray / 1-var Dataset to a string-indexed Series."""
    if isinstance(obj, xr.Dataset):
        if len(obj.data_vars) != 1:
            raise ValueError(
                f"expected a single-variable Dataset, got {list(obj.data_vars)}",
            )
        obj = obj[next(iter(obj.data_vars))]
    if isinstance(obj, xr.DataArray):
        obj = obj.to_series()
    if not isinstance(obj, pd.Series):
        obj = pd.Series(obj)
    obj = obj.copy()
    obj.index = obj.index.astype(str)
    return obj[~obj.index.duplicated(keep="first")]


def _map_with_int_fallback(keys: pd.Index, lookup: pd.Series) -> pd.Series:
    """Map unique string `keys` through `lookup`, retrying on a normalised key.

    NREL caps and `regions_onshore.geojson` carry substation IDs with a ``.0``
    suffix ("35827.0") while `busmap_s{simpl}.csv` indexes bare ints
    ("35827"). Same normalisation as ``remap_caps_to_cluster``.
    """
    keys = pd.Index(keys.astype(str))
    out = pd.Series(keys.map(lookup).to_numpy(), index=keys)
    missing = np.flatnonzero(out.isna().to_numpy())
    if missing.size:
        try:
            norm = pd.Index(keys[missing]).astype(float).astype(int).astype(str)
            out.iloc[missing] = norm.map(lookup).to_numpy()
        except (TypeError, ValueError):
            pass
    return out


def capacity_weighted_bus_aggregation(
    cf: xr.DataArray,
    avail: xr.DataArray,
    mapping_sub: pd.DataFrame,
    busmap: pd.Series,
    caps_pnom: pd.Series,
    chunk_t: int = 500,
) -> xr.Dataset:
    """Availability-weighted per substation, then ``p_nom_max``-weighted per cluster.

    Develop's simplify-early DAG builds the ``s{simpl}`` cluster profile
    directly, so the two stages are written as one::

        cf_sub[s, t]     = sum_{c in s} a_c cf_c[t] / A_s ,  A_s = sum_{c in s} a_c
        cf_cluster[k, t] = sum_{s in k} P_s cf_sub[s, t] / sum_{s in k} P_s

    with ``P_s`` the substation's NREL-caps ``p_nom_max`` — **installable**, not
    existing, capacity. Substituting the first line into the second collapses
    both stages into one scatter-add over the (cell, substation) rows, with the
    re-weighting::

        w_c = a_c * P_{sub(c)} / A_{sub(c)}
        cf_cluster[k, t] = sum_{c in k} w_c cf_c[t] / sum_{c in k} w_c

    because ``sum_{c in k} w_c == sum_{s in k} P_s``. Memory and passes over
    `cf` are therefore identical to ``weighted_bus_aggregation``.

    **This is not `master`'s weighting, by decision.** Master's cluster CF is
    produced by `simplify_network` → pypsa 0.30.2 ``aggregateoneport``, which
    weights ``p_max_pu`` by **existing** ``p_nom`` and degenerates to a plain
    arithmetic mean over the cluster's substations when their ``p_nom`` sums to
    zero (12 of 20 western onwind clusters). Reconstructing master's ``elec_s20``
    ``p_max_pu`` from master's own nodal outputs confirms it: existing-``p_nom``
    weighting with the uniform fallback reproduces every cluster to |Δ| 0.00000,
    while ``p_nom_max`` weighting is off by mean |Δ| 0.0966 (onwind) and 0.0132
    (solar). Develop keeps the ``p_nom_max`` weighting on purpose — it is the
    physically right weight for an extendable resource generator, whose buildable
    MW is what the profile gets applied to. Hot-fix HF-25, deltas ledger DL-19.

    Each (cell, substation) row is assigned to the cluster of its
    **substation** (via `busmap`), not to the cluster of the cell's own
    polygon, so the substation-level nearest-neighbour fallback and the
    cross-polygon leakage it causes survive the rollup instead of being
    re-decided at cluster resolution.

    Substations whose cells sum to zero availability have no stage-1 profile
    (``A_s = 0``). Their MW is **retained**: it is re-weighted onto the
    cluster's plain availability-weighted CF, which is algebraically the same
    as adding ``P_bad / A_cluster`` to every row weight in that cluster.
    `master` instead drops such substations from the profile file entirely and
    loses their NREL capacity from ``p_nom_max`` (hot-fix HF-24).
    Count and MW are logged at WARNING.

    cf:         (time, south_north, west_east); read in `chunk_t` slabs so an
                uint8-encoded file never decodes in full.
    avail:      (south_north, west_east) in [0, 1], NaN outside the raster.
    mapping_sub: cell→**substation** mapping, columns [name, NS, EW], from
                `get_cell_to_bus_mapping(..., shape_paths=[regions_onshore.geojson])`
                on the NODAL regions.
    busmap:     substation → cluster-bus Series (``busmap_s{simpl}.csv``).
    caps_pnom:  substation-keyed NREL ``p_nom_max``, BEFORE
                ``remap_caps_to_cluster``.

    Returns the same schema as ``weighted_bus_aggregation``: ``profile``
    (time, bus), ``bus_avail`` (bus), ``n_cells`` (bus), with `bus` the
    cluster-bus IDs.
    """
    av_arr = avail.values.astype(np.float64)
    av_arr = np.where(np.isnan(av_arr), 0.0, av_arr)

    busmap = _as_series(busmap).astype(str)
    caps_pnom = _as_series(caps_pnom).astype(float)

    row_sub_name = mapping_sub["name"].astype(str).to_numpy()
    row_ns = mapping_sub["NS"].to_numpy(dtype=np.int64)
    row_ew = mapping_sub["EW"].to_numpy(dtype=np.int64)
    row_a = av_arr[row_ns, row_ew]

    subs = pd.Index(pd.unique(row_sub_name))
    row_sub = pd.Series(np.arange(len(subs)), index=subs).reindex(row_sub_name).to_numpy(dtype=np.int64)

    sub_cluster = _map_with_int_fallback(subs, busmap)
    sub_pnom = _map_with_int_fallback(subs, caps_pnom).astype(float).fillna(0.0).to_numpy()

    unmapped_sub = sub_cluster.isna().to_numpy()
    if unmapped_sub.any():
        logger.warning(
            f"capacity CF weighting: {int(unmapped_sub.sum())}/{len(subs)} substations in the "
            f"nodal regions have no busmap entry ({float(sub_pnom[unmapped_sub].sum()):.1f} MW "
            "p_nom_max); their cells are excluded from the cluster profiles.",
        )

    clusters = sorted(sub_cluster.dropna().unique())
    if not clusters:
        raise RuntimeError(
            "capacity CF weighting: no substation in the nodal regions maps through the busmap — "
            "check bus-ID formatting in regions_onshore.geojson vs busmap_s{simpl}.csv.",
        )
    cluster_pos = {c: i for i, c in enumerate(clusters)}
    bus_count = len(clusters)
    sub_cl = np.array(
        [cluster_pos[c] if isinstance(c, str) else -1 for c in sub_cluster.to_numpy()],
        dtype=np.int64,
    )

    # Keep only rows whose substation resolves to a cluster.
    row_cl = sub_cl[row_sub]
    valid = row_cl >= 0
    row_cl = row_cl[valid]
    row_sub = row_sub[valid]
    row_ns = row_ns[valid]
    row_ew = row_ew[valid]
    row_a = row_a[valid]

    # Stage-1 denominator: availability per substation.
    avail_per_sub = np.zeros(len(subs), dtype=np.float64)
    np.add.at(avail_per_sub, row_sub, row_a)

    # Master drops substations with no availability at all from the profile
    # file, losing their NREL capacity too (HF-24). Retain their MW here by
    # folding it onto the cluster's availability-weighted CF.
    bad_sub = (avail_per_sub <= 0) & (sub_pnom > 0) & (sub_cl >= 0)
    density = np.where(avail_per_sub > 0, sub_pnom / np.where(avail_per_sub > 0, avail_per_sub, 1.0), 0.0)

    avail_per_cluster = np.zeros(bus_count, dtype=np.float64)
    np.add.at(avail_per_cluster, row_cl, row_a)
    pnom_bad_per_cluster = np.zeros(bus_count, dtype=np.float64)
    np.add.at(pnom_bad_per_cluster, sub_cl[bad_sub], sub_pnom[bad_sub])
    # No availability anywhere in the cluster → nothing to fall back onto; the
    # cluster profile stays NaN, exactly as weighted_bus_aggregation leaves it.
    fallback = np.where(
        avail_per_cluster > 0,
        pnom_bad_per_cluster / np.where(avail_per_cluster > 0, avail_per_cluster, 1.0),
        0.0,
    )
    if bad_sub.any():
        stranded = float(pnom_bad_per_cluster[avail_per_cluster <= 0].sum())
        logger.warning(
            f"capacity CF weighting: {int(bad_sub.sum())} substations have zero land "
            f"availability but {float(sub_pnom[bad_sub].sum()):.1f} MW of NREL p_nom_max. "
            "master drops them from the profile file entirely (HF-24); their capacity is "
            "retained here on the cluster's availability-weighted CF"
            + (f" ({stranded:.1f} MW sit in clusters with no availability at all)." if stranded > 0 else "."),
        )

    row_w = row_a * (density[row_sub] + fallback[row_cl])

    keep = row_w > 0
    row_cl_k = row_cl[keep]
    row_ns_k = row_ns[keep]
    row_ew_k = row_ew[keep]
    row_w_k = row_w[keep]

    time_count = cf.sizes["time"]

    # Per-cluster diagnostics (cheap, before streaming CF)
    n_cells = np.zeros(bus_count, dtype=np.int32)
    np.add.at(n_cells, row_cl, 1)
    den = np.zeros(bus_count, dtype=np.float64)
    np.add.at(den, row_cl_k, row_w_k)

    num = np.zeros((time_count, bus_count), dtype=np.float64)

    for t0 in range(0, time_count, chunk_t):
        t1 = min(t0 + chunk_t, time_count)
        chunk = cf.isel(time=slice(t0, t1)).values  # (t1-t0, NS, EW), forces decode of one slab
        cf_rows = chunk[:, row_ns_k, row_ew_k]
        cf_rows *= row_w_k
        for ti in range(t1 - t0):
            np.add.at(num[t0 + ti], row_cl_k, cf_rows[ti])
        del chunk, cf_rows

    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(den > 0, num / den, np.nan)
    mean_avail = np.where(n_cells > 0, avail_per_cluster / np.maximum(n_cells, 1), np.nan)

    return xr.Dataset(
        {
            "profile": (("time", "bus"), profile.astype(np.float32)),
            "bus_avail": (("bus",), mean_avail.astype(np.float32)),
            "n_cells": (("bus",), n_cells),
        },
        coords={"time": cf["time"].values, "bus": clusters},
    )


def fix_godeeep_time(ds: xr.Dataset, year: int) -> xr.Dataset:
    """Replicate the time-correction logic from aggregate_godeeep.py."""
    missing_time = pd.Timestamp(f"{year}-01-01T00:00:00")
    time_index = pd.DatetimeIndex(pd.to_datetime(ds["Time"].values, unit="ns"))
    if missing_time.value not in ds["Time"].values:
        time_index = time_index - pd.Timedelta(hours=1)

    if calendar.isleap(year) and pd.Timestamp(f"{year}-02-29") in time_index:
        leap_day = pd.Timestamp(f"{year}-02-29")
        time_index = pd.DatetimeIndex(
            [t + pd.Timedelta(days=1) if t >= leap_day else t for t in time_index],
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
    assert avail.shape == cf.shape[1:], f"avail shape {avail.shape} must match cf spatial shape {cf.shape[1:]}"

    print("[agg] building cell->bus mapping…")
    shape_paths = [args.onshore_shapes]
    if args.offshore_shapes:
        shape_paths.append(args.offshore_shapes)
    mapping = get_cell_to_bus_mapping(
        ds["x"].values,
        ds["y"].values,
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
        f"profile mean={float(out['profile'].mean(skipna=True)):.3f}",
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(args.output)
    print(f"[agg] wrote {args.output}")


if __name__ == "__main__":
    main()
