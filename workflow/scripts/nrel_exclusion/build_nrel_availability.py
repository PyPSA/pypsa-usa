"""
Compute per-GODEEEP-cell NREL land availability.

For a given technology and access scenario (reference | limited | open), project
the NREL exclusion raster onto the GODEEEP (south_north, west_east) grid by
fractional zonal mean over each cell polygon. Output is a small NetCDF with
`avail[south_north, west_east]` in [0, 1].

The GODEEEP grid is Lambert Conformal (curvilinear in lat/lon). We reconstruct
each cell footprint by averaging four neighboring centers to get corners, then
reproject to EPSG:5070 (matches NREL rasters) and run exactextract.

Optional CEC BaseScreen (onwind/solar, CA-only) and BOEM OSW planning-area
(offwind/offwind_floating) filters can be multiplied into the NREL raster
before aggregation. Semantics:
 * CEC: binary 1=allowed / 0=excluded in California; outside CA the mask is
   forced to 1 via a state-polygon rasterization (CEC's own nodata footprint
   bleeds into NV/OR/AZ and can't be used as a proxy for CA).
 * BOEM: binary 1=inside planning area / 0=outside; the raw raster covers
   coastal US only, so cells outside its extent are treated as excluded
   (correct policy behavior: no BOEM planning area → no offshore wind).
"""

import argparse
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from exactextract import exact_extract
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from shapely.geometry import Polygon

RASTER_PATHS = {
    ("solar", "reference"): "solar/solar_reference_access_percent_included.tif",
    ("solar", "limited"): "solar/solar_limited_access_percent_included.tif",
    ("solar", "open"): "solar/solar_open_access_percent_included.tif",
    ("onwind", "reference"): "onwind/lbw_reference_115hh_170rd.tif",
    ("onwind", "limited"): "onwind/lbw_limited_115hh_170rd.tif",
    ("onwind", "open"): "onwind/lbw_open_115hh_170rd.tif",
    ("offwind", "reference"): "offwind/reference_composite_lzw.tif",
    ("offwind", "limited"): "offwind/limited_composite_lzw.tif",
    ("offwind", "open"): "offwind/open_composite_lzw.tif",
    # offwind_floating shares the BOEM composite raster with offwind — the
    # fixed/floating split lives in the supply-curve CSV's `technology` column,
    # not in the availability raster itself.
    ("offwind_floating", "reference"): "offwind/reference_composite_lzw.tif",
    ("offwind_floating", "limited"): "offwind/limited_composite_lzw.tif",
    ("offwind_floating", "open"): "offwind/open_composite_lzw.tif",
}

# The onwind raster declares no nodata in its header, so its 255 sentinel has
# to be masked out by value. Solar and offwind carry a header nodata that is
# read from the file itself.
ONWIND_RASTER_NODATA = 255

# Sentinel used for nodata in the composite GeoTIFF we hand to exactextract.
# Chosen to be well outside the value range of any real raster (0-100 for solar,
# 0-1/0-255 for wind), so exact_extract cleanly masks it out.
COMPOSITE_NODATA = -9999.0


def build_cell_corners(xlong: np.ndarray, xlat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return corner arrays of shape (ny+1, nx+1) in lon/lat.

    Corner (i, j) lies at the meet of cells (i-1, j-1), (i-1, j), (i, j-1),
    (i, j). Interior corners are the mean of the four surrounding centers.
    Edges/corners are extrapolated linearly from the adjacent row/column.
    """
    ny, nx = xlong.shape
    lon_corners = np.full((ny + 1, nx + 1), np.nan, dtype=np.float64)
    lat_corners = np.full((ny + 1, nx + 1), np.nan, dtype=np.float64)

    # Interior corners: mean of 4 neighbors
    lon_corners[1:-1, 1:-1] = 0.25 * (xlong[:-1, :-1] + xlong[:-1, 1:] + xlong[1:, :-1] + xlong[1:, 1:])
    lat_corners[1:-1, 1:-1] = 0.25 * (xlat[:-1, :-1] + xlat[:-1, 1:] + xlat[1:, :-1] + xlat[1:, 1:])

    # Top/bottom edges: extrapolate from first/last two interior rows
    lon_corners[0, 1:-1] = 2 * lon_corners[1, 1:-1] - lon_corners[2, 1:-1]
    lat_corners[0, 1:-1] = 2 * lat_corners[1, 1:-1] - lat_corners[2, 1:-1]
    lon_corners[-1, 1:-1] = 2 * lon_corners[-2, 1:-1] - lon_corners[-3, 1:-1]
    lat_corners[-1, 1:-1] = 2 * lat_corners[-2, 1:-1] - lat_corners[-3, 1:-1]

    # Left/right edges: extrapolate from first/last two interior cols
    lon_corners[1:-1, 0] = 2 * lon_corners[1:-1, 1] - lon_corners[1:-1, 2]
    lat_corners[1:-1, 0] = 2 * lat_corners[1:-1, 1] - lat_corners[1:-1, 2]
    lon_corners[1:-1, -1] = 2 * lon_corners[1:-1, -2] - lon_corners[1:-1, -3]
    lat_corners[1:-1, -1] = 2 * lat_corners[1:-1, -2] - lat_corners[1:-1, -3]

    # Four outer corners: extrapolate along diagonal
    lon_corners[0, 0] = 2 * lon_corners[1, 1] - lon_corners[2, 2]
    lat_corners[0, 0] = 2 * lat_corners[1, 1] - lat_corners[2, 2]
    lon_corners[0, -1] = 2 * lon_corners[1, -2] - lon_corners[2, -3]
    lat_corners[0, -1] = 2 * lat_corners[1, -2] - lat_corners[2, -3]
    lon_corners[-1, 0] = 2 * lon_corners[-2, 1] - lon_corners[-3, 2]
    lat_corners[-1, 0] = 2 * lat_corners[-2, 1] - lat_corners[-3, 2]
    lon_corners[-1, -1] = 2 * lon_corners[-2, -2] - lon_corners[-3, -3]
    lat_corners[-1, -1] = 2 * lat_corners[-2, -2] - lat_corners[-3, -3]

    return lon_corners, lat_corners


def build_cell_polygons(godeeep_path: str) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of GODEEEP cell polygons indexed by (NS, EW)."""
    with xr.open_dataset(godeeep_path) as ds:
        xlong = ds["XLONG"].values.astype(np.float64)
        xlat = ds["XLAT"].values.astype(np.float64)

    ny, nx = xlong.shape
    lon_c, lat_c = build_cell_corners(xlong, xlat)

    polys = np.empty(ny * nx, dtype=object)
    ns_idx = np.empty(ny * nx, dtype=np.int32)
    ew_idx = np.empty(ny * nx, dtype=np.int32)

    k = 0
    for i in range(ny):
        for j in range(nx):
            ring = [
                (lon_c[i, j], lat_c[i, j]),
                (lon_c[i, j + 1], lat_c[i, j + 1]),
                (lon_c[i + 1, j + 1], lat_c[i + 1, j + 1]),
                (lon_c[i + 1, j], lat_c[i + 1, j]),
            ]
            polys[k] = Polygon(ring)
            ns_idx[k] = i
            ew_idx[k] = j
            k += 1

    return gpd.GeoDataFrame(
        {"NS": ns_idx, "EW": ew_idx},
        geometry=list(polys),
        crs="EPSG:4326",
    )


def _warp_mask_to_grid(
    src_path: str | Path,
    dst_transform,
    dst_shape: tuple[int, int],
    dst_crs,
    fill: float,
) -> np.ndarray:
    """Reproject a binary mask raster onto the NREL grid.

    Source pixels are treated as real data (no nodata masking) — the CEC/BOEM
    rasters encode "not-developable / out-of-coverage" as 0 and we want that 0
    to propagate as a real zero. Destination pixels outside the source extent
    are filled with `fill` (1 for CEC → "no restriction outside California",
    0 for BOEM → "outside a planning area").
    """
    # Read raw and pass as numpy so rasterio doesn't inherit the source file's
    # nodata metadata. CEC/BOEM both use nodata=0, but for us 0 IS real data
    # (= "excluded"). Pass an out-of-band sentinel for src_nodata so no pixel
    # is masked — every source 0 and 1 gets reprojected onto the dst grid.
    oob_sentinel = -9999.0
    dst = np.full(dst_shape, fill, dtype=np.float32)
    with rasterio.open(src_path) as src:
        src_arr = src.read(1).astype(np.float32)
        reproject(
            source=src_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            destination=dst,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=oob_sentinel,
            dst_nodata=fill,
            init_dest_nodata=False,
        )
    return dst


def build_composite_raster(
    tech: str,
    access: str,
    exclusion_dir: str | Path,
    repo_data_dir: str | Path | None,
    state_shapes_path: str | Path | None,
    apply_cec: bool,
    apply_boem: bool,
) -> tuple[np.ndarray, "rasterio.Affine", object]:
    """Load NREL raster and optionally multiply in CEC + BOEM masks.

    Returns (composite_array, transform, crs). Composite pixels that are nodata
    in the NREL source are set to COMPOSITE_NODATA.
    """
    nrel_path = Path(exclusion_dir) / RASTER_PATHS[(tech, access)]
    print(f"[avail] raster   : {nrel_path}")

    with rasterio.open(nrel_path) as src:
        nrel = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        shape = src.shape
        src_nodata = src.nodata

    # Normalize NREL nodata → NaN so multiplication preserves missingness.
    if src_nodata is not None:
        nrel = np.where(np.isclose(nrel, src_nodata), np.nan, nrel)
    if tech == "onwind":
        nrel = np.where(nrel == ONWIND_RASTER_NODATA, np.nan, nrel)

    composite = nrel

    if apply_cec and tech in ("onwind", "solar"):
        if repo_data_dir is None or state_shapes_path is None:
            raise ValueError(
                "apply_cec_basescreen requires --repo-data-dir and --state-shapes",
            )
        cec_name = "CEC_Wind_BaseScreen_epsg3310.tif" if tech == "onwind" else "CEC_Solar_BaseScreen_epsg3310.tif"
        cec_path = Path(repo_data_dir) / "geospatial" / "CEC_GIS" / cec_name
        print(f"[avail] CEC mask : {cec_path}")

        cec_on_grid = _warp_mask_to_grid(
            cec_path,
            transform,
            shape,
            crs,
            fill=1.0,
        )

        states = gpd.read_file(state_shapes_path).to_crs(crs)
        ca_matches = states.loc[states["STUSPS"] == "CA", "geometry"]
        if ca_matches.empty:
            raise ValueError(f"No CA row found in {state_shapes_path}")
        ca_geom = ca_matches.iloc[0]
        in_ca = rasterize(
            [(ca_geom, 1)],
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
        ).astype(bool)

        cec_eff = np.where(in_ca, cec_on_grid, 1.0)
        composite = composite * cec_eff
        n_ca = int(in_ca.sum())
        n_ca_excluded = int((in_ca & (cec_on_grid < 0.5)).sum())
        print(f"[avail] CEC: {n_ca:,} CA pixels; {n_ca_excluded:,} excluded")

    if apply_boem and tech.startswith("offwind"):
        if repo_data_dir is None:
            raise ValueError("apply_boem_osw requires --repo-data-dir")
        boem_path = Path(repo_data_dir) / "geospatial" / "boem_osw_planning_areas.tif"
        print(f"[avail] BOEM mask: {boem_path}")

        boem_on_grid = _warp_mask_to_grid(
            boem_path,
            transform,
            shape,
            crs,
            fill=0.0,
        )
        composite = composite * boem_on_grid
        n_in = int((boem_on_grid > 0.5).sum())
        print(f"[avail] BOEM: {n_in:,} pixels inside planning areas")

    # Write NaN → sentinel so exactextract treats them as nodata.
    composite = np.where(np.isnan(composite), COMPOSITE_NODATA, composite)
    return composite, transform, crs


def compute_availability(
    tech: str,
    access: str,
    godeeep_path: str,
    exclusion_dir: str,
    repo_data_dir: str | Path | None = None,
    state_shapes_path: str | Path | None = None,
    apply_cec: bool = False,
    apply_boem: bool = False,
    subset: slice | None = None,
) -> xr.DataArray:
    """Return avail[south_north, west_east] in [0, 1]."""
    print(f"[avail] tech={tech} access={access} cec={apply_cec} boem={apply_boem}")
    print(f"[avail] godeeep  : {godeeep_path}")

    composite, transform, crs = build_composite_raster(
        tech=tech,
        access=access,
        exclusion_dir=exclusion_dir,
        repo_data_dir=repo_data_dir,
        state_shapes_path=state_shapes_path,
        apply_cec=apply_cec,
        apply_boem=apply_boem,
    )
    shape = composite.shape

    cells = build_cell_polygons(godeeep_path)
    print(f"[avail] cells    : {len(cells)} polygons in {cells.crs}")
    cells_proj = cells.to_crs(crs)
    print(f"[avail] reprojected cells to {crs}")

    if subset is not None:
        cells_proj = cells_proj.iloc[subset].copy()
        print(f"[avail] SUBSET   : {len(cells_proj)} cells")

    # exactextract needs a raster source on disk; stash the composite in a
    # temp GeoTIFF and hand it the path. exactextract picks up the nodata
    # sentinel from the file header and excludes those pixels from the mean.
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": COMPOSITE_NODATA,
        "width": shape[1],
        "height": shape[0],
        "count": 1,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }

    print("[avail] running exact_extract on composite raster…")
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=True) as tmp:
        tmp_path = tmp.name
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(composite.astype(np.float32), 1)
        results = exact_extract(
            tmp_path,
            cells_proj,
            ["mean"],
            output="pandas",
            include_geom=False,
        )

    cells_proj = cells_proj.reset_index(drop=True)
    cells_proj["avail"] = results["mean"].values

    ny, nx = xr.open_dataset(godeeep_path)["XLONG"].shape
    avail = np.full((ny, nx), np.nan, dtype=np.float32)
    avail[cells_proj["NS"].values, cells_proj["EW"].values] = cells_proj["avail"].values

    da = xr.DataArray(
        avail,
        dims=("south_north", "west_east"),
        coords={
            "south_north": np.arange(ny),
            "west_east": np.arange(nx),
        },
        name="avail",
        attrs={
            "tech": tech,
            "access": access,
            "apply_cec_basescreen": int(bool(apply_cec)),
            "apply_boem_osw": int(bool(apply_boem)),
            "description": "Mean fractional land availability from NREL exclusion "
            "raster, optionally modulated by CEC BaseScreen (CA-only) "
            "and BOEM OSW planning areas, aggregated per GODEEEP grid "
            "cell via exact_extract area-weighted mean.",
        },
    )
    return da


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tech",
        required=True,
        choices=["solar", "onwind", "offwind", "offwind_floating"],
    )
    ap.add_argument("--access", required=True, choices=["reference", "limited", "open"])
    ap.add_argument("--godeeep", required=True, help="Path to a raw GODEEEP .nc file (any year).")
    ap.add_argument(
        "--exclusion-dir",
        default="/home/groups/iazevedo/asia/pypsa-usa/workflow/data/nrel_exclusion",
    )
    ap.add_argument(
        "--repo-data-dir",
        default="/home/groups/iazevedo/asia/pypsa-usa/workflow/repo_data",
        help="Root of repo_data; used to locate CEC and BOEM rasters.",
    )
    ap.add_argument(
        "--state-shapes",
        default="/home/groups/iazevedo/asia/pypsa-usa/workflow/repo_data/geospatial/us_states_cb_2020_5m/cb_2020_us_state_5m.shp",
        help="Path to Census Bureau state shapefile (used for CA masking).",
    )
    ap.add_argument(
        "--apply-cec-basescreen",
        action="store_true",
        help="Multiply in CEC Wind/Solar BaseScreen (CA-only).",
    )
    ap.add_argument(
        "--apply-boem-osw",
        action="store_true",
        help="Multiply in BOEM OSW planning areas (offwind only).",
    )
    ap.add_argument("--output", required=True, help="Output NetCDF path.")
    ap.add_argument(
        "--subset-n",
        type=int,
        default=0,
        help="If > 0, only process N cells starting at --subset-start (quick validation).",
    )
    ap.add_argument(
        "--subset-start",
        type=int,
        default=0,
        help="Flat index offset for --subset-n.",
    )
    args = ap.parse_args()

    subset = slice(args.subset_start, args.subset_start + args.subset_n) if args.subset_n > 0 else None

    da = compute_availability(
        tech=args.tech,
        access=args.access,
        godeeep_path=args.godeeep,
        exclusion_dir=args.exclusion_dir,
        repo_data_dir=args.repo_data_dir,
        state_shapes_path=args.state_shapes,
        apply_cec=args.apply_cec_basescreen,
        apply_boem=args.apply_boem_osw,
        subset=subset,
    )

    valid = da.values[~np.isnan(da.values)]
    if valid.size == 0:
        print("[avail] stats: n_valid=0 (all NaN — subset likely outside raster extent)")
    else:
        print(
            f"[avail] stats: n_valid={valid.size}  "
            f"min={valid.min():.3f}  max={valid.max():.3f}  mean={valid.mean():.3f}",
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(args.output)
    print(f"[avail] wrote {args.output}")


if __name__ == "__main__":
    main()
