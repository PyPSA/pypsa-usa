"""
Roll up NREL supply-curve CSV to per-bus capacity variables.

Produces the fields that `build_renewable_profiles.py` currently pulls from the
Zenodo-preprocessed ERA5/Atlite file:
    p_nom_max       - max installable capacity per bus (MW)
    potential       - capacity potential per bus (MW)  [= p_nom_max for now]
    weight          - layout weight per bus (MW), used for aggregation
    average_distance- capacity-weighted mean haversine distance from bus center
                      to supply-curve sites in that bus (km)
    x, y            - per-bus entry coordinates (lon/lat): capacity-weighted
                      site centroid, bus-polygon centroid when no capacity.
                      Consumed by the opt-in `nrel_caps_reassign` recovery in
                      build_renewable_profiles.py (footprint-scoped runs).

Supply-curve points are spatial-joined to bus polygons. Points outside any bus
(e.g. offshore sites for onshore-only buses) are dropped. Each row in the CSV
represents a developable site (lat/lon, area_developable_sq_km, capacity_ac_mw,
capacity_factor_ac or ncf_2035).

Optional CEC BaseScreen and BOEM OSW filters drop supply-curve points that
fall in excluded areas before the per-bus rollup:
 * CEC (onwind/solar): drop a California-internal point whose CEC raster value
   is 0. Non-CA points are always kept.
 * BOEM (offwind/offwind_floating): drop any point whose BOEM raster value is
   0, including points outside the raster's extent. This mirrors the policy
   reality that federal offshore wind only happens inside BOEM planning areas.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from shapely.geometry import LineString

SUPPLY_CURVES = {
    ("solar", "reference"): "solar/solar_reference_access_2035_moderate_supply_curve.csv",
    ("solar", "limited"): "solar/solar_limited_access_2035_moderate_supply_curve.csv",
    ("solar", "open"): "solar/solar_open_access_2035_moderate_supply_curve.csv",
    ("onwind", "reference"): "onwind/lbw_reference_access_2035_moderate_115hh_170rd_supply_curve.csv",
    ("onwind", "limited"): "onwind/lbw_limited_access_2035_moderate_115hh_170rd_supply_curve.csv",
    ("onwind", "open"): "onwind/lbw_open_access_2035_moderate_115hh_170rd_supply_curve.csv",
    ("offwind", "reference"): "offwind/reference_supply-curve_post_proc.csv",
    ("offwind", "limited"): "offwind/limited_supply-curve_post_proc.csv",
    ("offwind", "open"): "offwind/open_supply-curve_post_proc.csv",
    # Same CSV for both offshore variants; rollup_supply_curve filters by the
    # `technology` column to separate fixed-bottom from floating sites.
    ("offwind_floating", "reference"): "offwind/reference_supply-curve_post_proc.csv",
    ("offwind_floating", "limited"): "offwind/limited_supply-curve_post_proc.csv",
    ("offwind_floating", "open"): "offwind/open_supply-curve_post_proc.csv",
}

# CSV column used for each tech's native capacity factor, and a scale to
# convert to a 0-1 fraction (offwind's ncf_2035 is encoded in percent).
CF_COL = {
    "solar": "capacity_factor_ac",
    "onwind": "capacity_factor_ac",
    "offwind": "ncf_2035",
    "offwind_floating": "ncf_2035",
}
CF_SCALE = {
    "solar": 1.0,
    "onwind": 1.0,
    "offwind": 0.01,
    "offwind_floating": 0.01,
}

# Offshore CSV row-filter: maps pypsa-usa tech → value of the CSV's
# `technology` column. Non-offshore techs are absent from this map.
OFFSHORE_TECH_FILTER = {
    "offwind": "fixed",
    "offwind_floating": "floating",
}


def haversine_km(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371.0
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * earth_radius_km * np.arcsin(np.sqrt(a))


def _sample_raster(path: str | Path, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Sample a raster at (lon, lat) points. Reprojects points into the
    raster's CRS first. Points outside the raster extent return nodata (which
    rasterio.sample yields as the raster's declared nodata value).
    """
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    )
    with rasterio.open(path) as src:
        pts_proj = pts.to_crs(src.crs)
        coords = [(g.x, g.y) for g in pts_proj.geometry]
        vals = np.array([v[0] for v in src.sample(coords)], dtype=np.float32)
    return vals


def filter_supply_curve_points(
    df: pd.DataFrame,
    tech: str,
    apply_cec: bool,
    apply_boem: bool,
    state_shapes_path: str | Path | None,
    repo_data_dir: str | Path | None,
) -> pd.DataFrame:
    """Drop rows whose (latitude, longitude) fall into CEC-excluded (inside
    California) or outside-BOEM-planning areas.
    """
    if not apply_cec and not apply_boem:
        return df.reset_index(drop=True)

    keep = np.ones(len(df), dtype=bool)
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()

    if apply_cec and tech in ("onwind", "solar"):
        if state_shapes_path is None or repo_data_dir is None:
            raise ValueError(
                "apply_cec_basescreen requires --state-shapes and --repo-data-dir",
            )
        states = gpd.read_file(state_shapes_path).to_crs(4326)
        ca_matches = states.loc[states["STUSPS"] == "CA", "geometry"]
        if ca_matches.empty:
            raise ValueError(f"No CA row found in {state_shapes_path}")
        ca_geom = ca_matches.iloc[0]

        pts = gpd.GeoSeries(
            gpd.points_from_xy(lon, lat),
            crs="EPSG:4326",
        )
        in_ca = pts.within(ca_geom).to_numpy()

        cec_name = "CEC_Wind_BaseScreen_epsg3310.tif" if tech == "onwind" else "CEC_Solar_BaseScreen_epsg3310.tif"
        cec_path = Path(repo_data_dir) / "geospatial" / "CEC_GIS" / cec_name
        if in_ca.any():
            cec_vals = _sample_raster(cec_path, lon[in_ca], lat[in_ca])
            ca_keep = cec_vals > 0.5
            idx_in_ca = np.where(in_ca)[0]
            keep[idx_in_ca[~ca_keep]] = False
            n_drop = int((~ca_keep).sum())
            print(f"[caps] CEC filter ({tech}): dropped {n_drop}/{in_ca.sum()} California-internal sites")
        else:
            print(f"[caps] CEC filter ({tech}): no California-internal sites, skipping")

    if apply_boem and tech.startswith("offwind"):
        if repo_data_dir is None:
            raise ValueError("apply_boem_osw requires --repo-data-dir")
        boem_path = Path(repo_data_dir) / "geospatial" / "boem_osw_planning_areas.tif"
        boem_vals = _sample_raster(boem_path, lon, lat)
        boem_keep = boem_vals > 0.5
        n_drop = int((~boem_keep).sum())
        keep = keep & boem_keep
        print(f"[caps] BOEM filter ({tech}): dropped {n_drop}/{len(df)} sites outside planning areas")

    out = df[keep].reset_index(drop=True)
    print(f"[caps] point filter: {len(out)}/{len(df)} supply-curve sites retained")
    return out


def rollup_supply_curve(
    csv_path: str,
    tech: str,
    onshore_shapes: str,
    offshore_shapes: str | None,
    offshore_eez_shape: str | None = None,
    apply_cec: bool = False,
    apply_boem: bool = False,
    state_shapes_path: str | Path | None = None,
    repo_data_dir: str | Path | None = None,
) -> xr.Dataset:
    df = pd.read_csv(csv_path)
    cf_col = CF_COL[tech]
    required = {"latitude", "longitude", "capacity_ac_mw", cf_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"supply curve {csv_path} missing columns: {missing}")

    if tech in OFFSHORE_TECH_FILTER:
        target = OFFSHORE_TECH_FILTER[tech]
        if "technology" not in df.columns:
            raise ValueError(
                f"supply curve {csv_path} missing `technology` column needed to split fixed vs floating offshore sites",
            )
        n_before = len(df)
        df = df[df["technology"] == target].reset_index(drop=True)
        print(f"[caps] offshore filter: kept {len(df)}/{n_before} rows with technology='{target}'")

    df = filter_supply_curve_points(
        df,
        tech,
        apply_cec=apply_cec,
        apply_boem=apply_boem,
        state_shapes_path=state_shapes_path,
        repo_data_dir=repo_data_dir,
    )

    sub = (
        df[["latitude", "longitude", "capacity_ac_mw", cf_col]]
        .rename(
            columns={cf_col: "cf"},
        )
        .copy()
    )
    sub["cf"] = sub["cf"] * CF_SCALE[tech]
    pts = gpd.GeoDataFrame(
        sub,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    shapes = [gpd.read_file(onshore_shapes).to_crs(4326)]
    if offshore_shapes:
        shapes.insert(0, gpd.read_file(offshore_shapes).to_crs(4326))
    gdf_shapes = pd.concat(shapes, axis=0, ignore_index=True)[["name", "geometry"]].dissolve(by="name", as_index=False)

    # Left-join points to buses; drop points that fall outside every bus
    joined = gpd.sjoin(pts, gdf_shapes[["name", "geometry"]], how="left", predicate="intersects")
    total = len(joined)
    joined = joined.dropna(subset=["name"])
    dropped = total - len(joined)
    print(f"[caps] {len(joined)}/{total} supply-curve sites fell inside a bus ({dropped} outside)")

    # Empty short-circuit: no surviving sites (e.g. all WECC-external fixed
    # offshore dropped by BOEM). Write a minimal empty caps Dataset so
    # downstream code can .sel() a zero-row result without crashing.
    if joined.empty:
        print("[caps] no surviving sites — writing empty caps Dataset")

        def empty():
            return np.array([], dtype=np.float32)

        data_vars = {
            "p_nom_max": (("bus",), empty()),
            "potential": (("bus",), empty()),
            "weight": (("bus",), empty()),
            "average_distance": (("bus",), empty()),
            "avg_cf": (("bus",), empty()),
            "x": (("bus",), empty()),
            "y": (("bus",), empty()),
        }
        if offshore_eez_shape is not None:
            data_vars["underwater_fraction"] = (("bus",), empty())
        return xr.Dataset(
            data_vars,
            coords={"bus": np.array([], dtype=object)},
            attrs={
                "tech": tech,
                "source_csv": csv_path,
                "n_sites_used": 0,
                "n_sites_dropped": int(dropped),
                "apply_cec_basescreen": int(bool(apply_cec)),
                "apply_boem_osw": int(bool(apply_boem)),
                "offshore_eez_shape": str(offshore_eez_shape) if offshore_eez_shape else "",
            },
        )

    # Bus center coords for average_distance calc
    bus_centroids = gdf_shapes.set_index("name").to_crs("EPSG:3857").geometry.centroid.to_crs(4326)
    bus_lat = bus_centroids.y
    bus_lon = bus_centroids.x

    joined["bus_lat"] = joined["name"].map(bus_lat)
    joined["bus_lon"] = joined["name"].map(bus_lon)
    joined["dist_km"] = haversine_km(
        joined["latitude"].values,
        joined["longitude"].values,
        joined["bus_lat"].values,
        joined["bus_lon"].values,
    )

    # Weighted aggregation. Also computes cap-weighted site centroid so we can
    # derive underwater_fraction post-groupby when an EEZ polygon is provided.
    def agg(group: pd.DataFrame) -> pd.Series:
        cap = group["capacity_ac_mw"].to_numpy()
        cf = group["cf"].to_numpy()
        d = group["dist_km"].to_numpy()
        site_lat = group["latitude"].to_numpy()
        site_lon = group["longitude"].to_numpy()
        cap_sum = cap.sum()
        if cap_sum <= 0:
            return pd.Series(
                {
                    "p_nom_max": 0.0,
                    "weight": 0.0,
                    "avg_cf": np.nan,
                    "average_distance": np.nan,
                    "site_lat": np.nan,
                    "site_lon": np.nan,
                },
            )
        # Weight for dispatch layout: capacity scaled by CF (energy-weighted)
        # Mirrors atlite's `layout = capacity_factor * area * capacity_per_sqkm`.
        weight = float((cap * cf).sum())
        return pd.Series(
            {
                "p_nom_max": float(cap_sum),
                "weight": weight,
                "avg_cf": float((cap * cf).sum() / cap_sum),
                "average_distance": float((cap * d).sum() / cap_sum),
                "site_lat": float((cap * site_lat).sum() / cap_sum),
                "site_lon": float((cap * site_lon).sum() / cap_sum),
            },
        )

    rollup = joined.groupby("name").apply(agg, include_groups=False).reset_index()
    print(
        f"[caps] rolled up to {len(rollup)} buses; "
        f"total p_nom_max = {rollup['p_nom_max'].sum():.1f} MW; "
        f"mean avg_cf = {rollup['avg_cf'].mean():.3f}",
    )

    # Per-entry coordinates (lon/lat): capacity-weighted site centroid, falling
    # back to the bus-polygon centroid when a bus has zero surviving capacity.
    # Downstream, footprint-scoped runs use these to reassign out-of-footprint
    # entries to the nearest in-footprint bus (config key `nrel_caps_reassign`
    # read by build_renewable_profiles.remap_caps_to_cluster). Caps files
    # written before this change lack x/y, and enabling the reassignment flag
    # against them raises a config error pointing back here.
    site_x = rollup["site_lon"].to_numpy(dtype=np.float64)
    site_y = rollup["site_lat"].to_numpy(dtype=np.float64)
    fallback_x = rollup["name"].map(bus_lon).to_numpy(dtype=np.float64)
    fallback_y = rollup["name"].map(bus_lat).to_numpy(dtype=np.float64)
    entry_x = np.where(np.isfinite(site_x), site_x, fallback_x)
    entry_y = np.where(np.isfinite(site_y), site_y, fallback_y)

    data_vars = {
        "p_nom_max": (("bus",), rollup["p_nom_max"].values.astype(np.float32)),
        "potential": (("bus",), rollup["p_nom_max"].values.astype(np.float32)),
        "weight": (("bus",), rollup["weight"].values.astype(np.float32)),
        "average_distance": (("bus",), rollup["average_distance"].values.astype(np.float32)),
        "avg_cf": (("bus",), rollup["avg_cf"].values.astype(np.float32)),
        "x": (("bus",), entry_x.astype(np.float32)),
        "y": (("bus",), entry_y.astype(np.float32)),
    }

    # underwater_fraction: fraction of the hypothetical DC line from the
    # capacity-weighted site centroid to the bus centroid that runs over water.
    # Used downstream by add_electricity to blend HVDC submarine vs overhead
    # costs. Only meaningful for offshore techs (needs the EEZ polygon).
    if offshore_eez_shape is not None:
        eez = gpd.read_file(offshore_eez_shape).to_crs(4326).unary_union
        uf = np.zeros(len(rollup), dtype=np.float32)
        for i, row in rollup.iterrows():
            name = row["name"]
            if not np.isfinite(row["site_lat"]) or not np.isfinite(row["site_lon"]):
                uf[i] = np.nan
                continue
            site = (row["site_lon"], row["site_lat"])
            bus_pt = (float(bus_lon.loc[name]), float(bus_lat.loc[name]))
            line = LineString([site, bus_pt])
            if line.length == 0:
                uf[i] = 1.0  # site coincides with bus centroid — presume offshore
                continue
            uf[i] = line.intersection(eez).length / line.length
        data_vars["underwater_fraction"] = (("bus",), uf)
        print(
            f"[caps] underwater_fraction: mean={np.nanmean(uf):.3f}  min={np.nanmin(uf):.3f}  max={np.nanmax(uf):.3f}",
        )

    ds = xr.Dataset(
        data_vars,
        coords={"bus": rollup["name"].values},
        attrs={
            "tech": tech,
            "source_csv": csv_path,
            "n_sites_used": len(joined),
            "n_sites_dropped": int(dropped),
            "apply_cec_basescreen": int(bool(apply_cec)),
            "apply_boem_osw": int(bool(apply_boem)),
            "offshore_eez_shape": str(offshore_eez_shape) if offshore_eez_shape else "",
        },
    )
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tech",
        required=True,
        choices=["solar", "onwind", "offwind", "offwind_floating"],
    )
    ap.add_argument("--access", required=True, choices=["reference", "limited", "open"])
    ap.add_argument("--onshore-shapes", required=True)
    ap.add_argument(
        "--offshore-shapes",
        default=None,
        help="Bus offshore polygons (regions_offshore.geojson).",
    )
    ap.add_argument(
        "--offshore-eez-shape",
        default=None,
        help="EEZ / offshore-mask polygon (offshore_shapes.geojson). "
        "When provided, underwater_fraction is computed per bus. "
        "Only meaningful for offshore techs.",
    )
    ap.add_argument(
        "--exclusion-dir",
        default="/home/groups/iazevedo/asia/pypsa-usa/workflow/data/nrel_exclusion",
    )
    ap.add_argument(
        "--repo-data-dir",
        default="/home/groups/iazevedo/asia/pypsa-usa/workflow/repo_data",
    )
    ap.add_argument(
        "--state-shapes",
        default="/home/groups/iazevedo/asia/pypsa-usa/workflow/repo_data/geospatial/us_states_cb_2020_5m/cb_2020_us_state_5m.shp",
    )
    ap.add_argument("--apply-cec-basescreen", action="store_true")
    ap.add_argument("--apply-boem-osw", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    csv_path = str(Path(args.exclusion_dir) / SUPPLY_CURVES[(args.tech, args.access)])
    print(f"[caps] csv: {csv_path}")

    ds = rollup_supply_curve(
        csv_path=csv_path,
        tech=args.tech,
        onshore_shapes=args.onshore_shapes,
        offshore_shapes=args.offshore_shapes,
        offshore_eez_shape=args.offshore_eez_shape,
        apply_cec=args.apply_cec_basescreen,
        apply_boem=args.apply_boem_osw,
        state_shapes_path=args.state_shapes,
        repo_data_dir=args.repo_data_dir,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(args.output)
    print(f"[caps] wrote {args.output}")


if __name__ == "__main__":
    main()
