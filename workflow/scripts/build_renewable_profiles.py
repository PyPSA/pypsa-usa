# BY PyPSA-USA Authors
"""
Calculates for each network substation the installable capacity (based on land-
use) and the available generation time series (based on weather data).
"""

import functools
import logging
import time

import atlite
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from _helpers import configure_logging, get_snapshots
from dask.distributed import Client
from pypsa.geo import haversine
from shapely.geometry import LineString
from zenodo_downloader import ZenodoScenarioDownloader

logger = logging.getLogger(__name__)


def _reassign_unmapped_caps(
    df: pd.DataFrame,
    max_km: float,
    tech: str,
) -> pd.DataFrame:
    """Assign each unmapped caps entry to the cluster of the geographically
    nearest MAPPED (in-footprint) entry, but only within ``max_km``.

    Requires per-entry ``x``/``y`` (lon/lat) columns in ``df``. Caps files
    generated before build_nrel_bus_capacities.py wrote coordinates lack
    them — that is a hard config error, not a silent fallback.
    """
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(
            f"nrel_caps_reassign.enable is true for '{tech}', but the NREL caps "
            "file carries no per-entry x/y coordinates, so unmapped entries "
            "cannot be reassigned to the nearest in-footprint bus. The published "
            "Zenodo caps artifacts predate the coordinate-preserving rollup. "
            "Fix: regenerate the caps files on HPC "
            "(workflow/scripts/nrel_exclusion/build_nrel_artifacts.sh, which "
            "runs build_nrel_bus_capacities.py — it now writes per-bus x/y), "
            "or set nrel_caps_reassign.enable: false to keep today's "
            "drop-unmapped behavior.",
        )

    unmapped = df["cluster"].isna()
    anchors = df.loc[~unmapped & df["x"].notna() & df["y"].notna()]
    candidates = df.loc[unmapped & df["x"].notna() & df["y"].notna()]
    if anchors.empty or candidates.empty:
        return df

    anchor_xy = anchors[["x", "y"]].to_numpy(dtype=float)
    n_recovered = 0
    mw_recovered = 0.0
    # Chunk the candidate side so the haversine distance matrix stays small
    # (national caps: roughly 17k unmapped by 2k mapped entries).
    chunk = 2000
    idx = candidates.index.to_numpy()
    for start in range(0, len(idx), chunk):
        rows = idx[start : start + chunk]
        cand_xy = df.loc[rows, ["x", "y"]].to_numpy(dtype=float)
        dist = haversine(cand_xy, anchor_xy)  # km, shape (len(rows), n_anchors)
        nearest = dist.argmin(axis=1)
        nearest_km = dist[np.arange(len(rows)), nearest]
        ok = nearest_km <= max_km
        take = rows[ok]
        df.loc[take, "cluster"] = anchors["cluster"].to_numpy()[nearest[ok]]
        n_recovered += int(ok.sum())
        if "p_nom_max" in df.columns:
            mw_recovered += float(df.loc[take, "p_nom_max"].sum())

    still = df["cluster"].isna()
    mw_still = float(df.loc[still, "p_nom_max"].sum()) if "p_nom_max" in df.columns else float("nan")
    logger.warning(
        f"NREL caps reassignment ({tech}): recovered {n_recovered} out-of-footprint "
        f"entries ({mw_recovered:.1f} MW p_nom_max) onto nearest in-footprint buses "
        f"within {max_km:.0f} km; {int(still.sum())} entries ({mw_still:.1f} MW) "
        f"remain beyond max_km and stay dropped.",
    )
    return df


def remap_caps_to_cluster(
    caps_ds: xr.Dataset,
    busmap: pd.Series,
    tech: str = "",
    reassign: dict | None = None,
) -> xr.Dataset:
    """Remap NREL bus-capacity dataset from substation keys to cluster bus keys.

    NREL caps files are keyed by substation ID (e.g. "0.0", "35000.0"); after
    the simplify-early refactor, the network and regions use simpl-cluster IDs
    (e.g. "p10 0"). Aggregate per cluster_bus:
      * weight, p_nom_max, potential, underwater_fraction_area → sum (extensive)
      * average_distance, avg_cf, underwater_fraction → capacity-weighted mean

    Caps files are rolled up against the NATIONAL substation tessellation; in
    footprint-scoped runs most entries have no busmap match and are dropped.
    Dropped totals are always logged loudly. When ``reassign['enable']`` is
    true, unmapped entries within ``reassign['max_km']`` of an in-footprint
    entry are folded into that entry's cluster instead of being dropped
    (requires per-entry x/y in the caps file).
    """
    extensive = {"p_nom_max", "potential", "weight"}
    intensive = {"average_distance", "avg_cf", "underwater_fraction"}
    coords_only = {"x", "y"}  # per-entry lon/lat — consumed here, never output

    df = caps_ds.to_dataframe().reset_index()
    df["bus"] = df["bus"].astype(str)
    # Try direct match first; fall back to float→int→str normalization since
    # NREL keys carry a ".0" suffix while busmap indices are bare ints.
    df["cluster"] = df["bus"].map(busmap)
    missing = df["cluster"].isna()
    if missing.any():
        try:
            norm = df.loc[missing, "bus"].astype(float).astype(int).astype(str)
            df.loc[missing, "cluster"] = norm.map(busmap).values
        except (TypeError, ValueError):
            pass

    # Loud, unconditional accounting of what the footprint scoping drops.
    unmapped = df["cluster"].isna()
    if unmapped.any():
        national_mw = float(df["p_nom_max"].sum()) if "p_nom_max" in df.columns else float("nan")
        dropped_mw = float(df.loc[unmapped, "p_nom_max"].sum()) if "p_nom_max" in df.columns else float("nan")
        pct = 100.0 * dropped_mw / national_mw if national_mw > 0 else float("nan")
        logger.warning(
            f"NREL caps remap ({tech}): {int(unmapped.sum())}/{len(df)} entries have no "
            f"busmap match (out of the run footprint) — dropping {dropped_mw:.1f} MW "
            f"p_nom_max, {pct:.1f}% of the national total. Border regions straddling "
            "the footprint edge lose their out-of-footprint capacity entirely; see "
            "nrel_caps_reassign in config.common.yaml for opt-in recovery.",
        )
        reassign = reassign or {}
        if reassign.get("enable", False):
            df = _reassign_unmapped_caps(
                df,
                max_km=float(reassign.get("max_km", 100.0)),
                tech=tech,
            )

    df = df.dropna(subset=["cluster"])
    if df.empty:
        raise RuntimeError(
            "NREL caps remap produced 0 rows — busmap and caps bus IDs share no overlap.",
        )

    weights = df["weight"] if "weight" in df.columns else None
    out_cols: dict[str, pd.Series] = {}
    for col in df.columns:
        if col in ("bus", "cluster") or col in coords_only:
            continue
        if col in extensive:
            out_cols[col] = df.groupby("cluster")[col].sum()
        elif col in intensive and weights is not None:
            wsum = (df[col] * weights).groupby(df["cluster"]).sum()
            wtotal = weights.groupby(df["cluster"]).sum()
            out_cols[col] = (wsum / wtotal).where(wtotal > 0, 0.0)
        else:  # unknown column → sum (extensive default, since caps are capacities)
            out_cols[col] = df.groupby("cluster")[col].sum()

    out = xr.Dataset(
        {
            name: xr.DataArray(
                series.values.astype("float32"),
                coords={"bus": series.index.values},
                dims="bus",
            )
            for name, series in out_cols.items()
        },
    )
    return out


# Get renewable snapshots for a given year using month/day from config
def get_renewable_snapshots(config, year):
    ren_sns_config = config.get("renewable_snapshots", {})

    if "start_month" in ren_sns_config:
        start_month = ren_sns_config.get("start_month", 1)
        start_day = ren_sns_config.get("start_day", 1)
        end_month = ren_sns_config.get("end_month", 12)
        end_day = ren_sns_config.get("end_day", 31)
        end_inclusive = ren_sns_config.get("end_inclusive", False)

        start_dt = pd.Timestamp(year=year, month=start_month, day=start_day)
        end_dt = pd.Timestamp(year=year, month=end_month, day=end_day)
        # For hourly snapshots, users usually mean "include the whole end day".
        # Convert this to next-day exclusive to avoid truncating at 00:00.
        if end_inclusive:
            end_dt = end_dt + pd.Timedelta(days=1)

        snapshots_config = {
            "start": start_dt.strftime("%Y-%m-%d %H:%M"),
            "end": end_dt.strftime("%Y-%m-%d %H:%M"),
            "inclusive": "left",
        }
        logger.info(
            f"Using renewable snapshots for year {year}: "
            f"{snapshots_config['start']} to {snapshots_config['end']} "
            f"({'inclusive' if end_inclusive else 'exclusive'} end)",
        )
        return get_snapshots(snapshots_config)
    else:
        # Old format fallback
        snapshots_config = {
            "start": f"{year}-01-01",
            "end": f"{year + 1}-01-01",
            "inclusive": "left",
        }
        logger.info(f"Using renewable snapshots for full year {year}")
        return get_snapshots(snapshots_config)


def plot_data(data):
    x = data.coords["x"].values  # Longitude
    y = data.coords["y"].values  # Latitude
    values = data.values

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(x, y, values, shading="auto", cmap="viridis")
    fig.colorbar(
        im,
        ax=ax,
        label="Value",
    )  # Add a colorbar to represent the value scale

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return fig, ax


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_renewable_profiles",
            technology="solar",
            interconnect="western",
        )
    configure_logging(snakemake)

    nprocesses = int(snakemake.threads)
    noprogress = snakemake.config["run"].get("disable_progressbar", True)
    noprogress = noprogress or not snakemake.config["atlite"]["show_progress"]
    params = snakemake.params.renewable[snakemake.wildcards.technology]
    resource = params["resource"]  # pv panel params / wind turbine params
    correction_factor = params.get("correction_factor", 1.0)
    capacity_per_sqkm = params["capacity_per_sqkm"]
    p_nom_max_meth = params.get("potential", "conservative")

    if isinstance(params.get("corine", {}), list):
        params["corine"] = {"grid_codes": params["corine"]}

    if correction_factor != 1.0:
        logger.info(f"correction_factor is set as {correction_factor}")

    if nprocesses > 1:
        client = Client(n_workers=nprocesses, threads_per_worker=1)
    else:
        client = None

    sns = get_snapshots(snakemake.params.snapshots)

    regions = gpd.read_file(snakemake.input.regions)
    assert "x" in regions.columns and "y" in regions.columns, (
        f"List of regions in {snakemake.input.regions} is empty, please disable the corresponding renewable technology"
    )
    # do not pull up, set_index does not work if geo dataframe is empty
    regions = regions.set_index("name").rename_axis("bus")
    buses = regions.index

    #### start editing to separate out different datasets
    if snakemake.params.renewable.get("dataset", False) == "atlite":
        ### start here
        logger.info("Loading atlite renewable dataset...")

        logger.info(f'using cutout "{snakemake.input.cutout}"')
        cutout = atlite.Cutout(snakemake.input.cutout[0]).sel(
            time=sns,
        )  # Patch fix with [0] move expand in snakemake rule to add_elec for multiple cutouts

        res = params.get("excluder_resolution", 100)
        excluder = atlite.ExclusionContainer(crs=5070, res=res)

        if params["natura"]:
            excluder.add_raster(
                snakemake.input.natura,
                nodata=0,
                allow_no_overlap=True,
            )

        corine = params.get("corine", {})
        if "grid_codes" in corine:
            codes = corine["grid_codes"]
            excluder.add_raster(
                snakemake.input.corine,
                codes=codes,
                invert=True,
                # crs=4326
            )
        if corine.get("distance", 0.0) > 0.0:
            codes = corine["distance_grid_codes"]
            buffer = corine["distance"]
            excluder.add_raster(
                snakemake.input.corine,
                codes=codes,
                buffer=buffer,
                # crs=4326,
            )

        if params.get("cec", 0):
            excluder.add_raster(
                snakemake.input[f"cec_{snakemake.wildcards.technology}"],
                nodata=0,
                allow_no_overlap=True,
            )

        if params.get("boem_screen", 0):
            excluder.add_raster(
                snakemake.input["boem_osw"],
                invert=True,
                nodata=0,
                allow_no_overlap=True,
            )

        if params.get("max_depth"):
            # lambda not supported for atlite + multiprocessing
            # use named function np.greater with partially frozen argument instead
            # and exclude areas where: -max_depth > grid cell depth
            func = functools.partial(np.greater, -params["max_depth"])
            excluder.add_raster(
                snakemake.input.gebco,
                codes=func,
                nodata=-1000,
                # crs=4326,
            )

        if params.get("min_depth"):
            # lambda not supported for atlite + multiprocessing
            # use named function np.greater with partially frozen argument instead
            # and exclude areas where: -min_depth < grid cell depth
            func = functools.partial(np.less, -params["min_depth"])
            excluder.add_raster(
                snakemake.input.gebco,
                codes=func,
                nodata=-1000,
                # crs=4326,
            )

        if "min_shore_distance" in params:
            buffer = params["min_shore_distance"]
            excluder.add_geometry(snakemake.input.country_shapes, buffer=buffer)

        if "max_shore_distance" in params:
            buffer = params["max_shore_distance"]
            excluder.add_geometry(
                snakemake.input.country_shapes,
                buffer=buffer,
                invert=True,
            )

        logger.info("Calculate landuse availability...")
        start = time.time()

        kwargs = dict(nprocesses=nprocesses, disable_progressbar=noprogress)
        availability = cutout.availabilitymatrix(regions, excluder, **kwargs)

        duration = time.time() - start
        logger.info(f"Completed landuse availability calculation ({duration:2.2f}s)")

        area = cutout.grid.to_crs("EPSG: 5070").area / 1e6
        area = xr.DataArray(
            area.values.reshape(cutout.shape),
            [cutout.coords["y"], cutout.coords["x"]],
        )

        potential = capacity_per_sqkm * availability.sum("bus") * area
        func = getattr(cutout, resource.pop("method"))
        if client is not None:
            resource["dask_kwargs"] = {"scheduler": client}
        capacity_factor = correction_factor * func(capacity_factor=True, **resource)
        layout = capacity_factor * area * capacity_per_sqkm
        profile, capacities = func(
            matrix=availability.stack(spatial=["y", "x"]),
            layout=layout,
            index=buses,
            per_unit=True,
            return_capacity=True,
            **resource,
        )

        logger.info(f"Calculating maximal capacity per bus (method '{p_nom_max_meth}')")
        if p_nom_max_meth == "simple":  ## right now the capacities loaded in are "conservative"
            p_nom_max = capacity_per_sqkm * availability @ area
        elif p_nom_max_meth == "conservative":
            max_cap_factor = capacity_factor.where(availability != 0).max(["x", "y"])
            p_nom_max = capacities / max_cap_factor
        else:
            raise AssertionError(
                f'Config key `potential` should be one of "simple" (default) or "conservative", not "{p_nom_max_meth}"',
            )

        logger.info("Calculate average distances.")
        layoutmatrix = (layout * availability).stack(spatial=["y", "x"])

        coords = cutout.grid[["x", "y"]]
        bus_coords = regions[["x", "y"]]

        average_distance = []
        centre_of_mass = []
        for i, bus in enumerate(buses):
            row = layoutmatrix.isel(bus=i).values.ravel()
            nz_b = row != 0
            row = row[nz_b]
            co = coords[nz_b]
            distances = haversine(bus_coords.loc[bus], co)
            average_distance.append((distances * (row / row.sum())).sum())
            centre_of_mass.append(co.values.T @ (row / row.sum()))

        average_distance = xr.DataArray(average_distance, [buses])
        centre_of_mass = xr.DataArray(centre_of_mass, [buses, ("spatial", ["x", "y"])])

    if snakemake.params.renewable.get("dataset", False) == "godeeep":
        logger.info("Loading godeeep renewable data...")
        scenario = snakemake.config["renewable_scenarios"][0]
        tech = snakemake.wildcards.technology
        access = snakemake.config.get("renewable_land_access")
        if not access:
            raise ValueError(
                "renewable_land_access must be set (e.g. 'reference', 'limited', "
                "'open') when renewable.dataset == 'godeeep'.",
            )

        # Determine year based on scenario type
        if scenario == "historical":
            year = snakemake.config["renewable_weather_years"][0]
            logger.info(f"Using historical year: {year} (from renewable_weather_years)")
        else:
            year = snakemake.params.planning_horizon
            logger.info(f"Using future scenario year: {year} (from planning_horizon wildcard)")

        # Get snapshots with appropriate year
        renewable_sns = get_renewable_snapshots(snakemake.config, year)
        downloader = ZenodoScenarioDownloader()

        # Technology configurations for filename construction
        if tech in ["onwind", "offwind", "offwind_floating"]:
            technology = "wind"
            wind_height = snakemake.config.get("godeeep_wind_height", "_100m")
        elif tech == "solar":
            technology = "solar"
            wind_height = ""
        else:
            raise ValueError("Invalid technology type. Choose 'onwind', 'offwind', 'offwind_floating' or 'solar'.")

        # ===== NREL access-scenario path =====
        # Download per-cell compressed CF and apply availability-weighted
        # aggregation onto bus polygons at runtime. Capacity variables come
        # from the NREL supply-curve rollup (caps file).
        from nrel_exclusion.aggregate_godeeep_weighted import (
            fix_godeeep_time,
            get_cell_to_bus_mapping,
            weighted_bus_aggregation,
        )

        logger.info(f"NREL access scenario: {access}")
        cf_filename = f"{technology}_gen_cf_{year}{wind_height}_compressed.nc"
        # Compressed-CF records on Zenodo are split by (tech, scenario), not by
        # year-window — one record holds 2030/2040/2050 for the same scenario.
        cf_record_key = f"{technology}{wind_height}_{scenario}_compressed"
        cf_filepath = downloader.download_scenario_file(cf_record_key, scenario, cf_filename)

        ds_cf = xr.open_dataset(cf_filepath)
        ds_cf = fix_godeeep_time(ds_cf, year)
        ds_cf = ds_cf.rename({"XLONG": "x", "XLAT": "y"})

        avail = xr.open_dataarray(snakemake.input.nrel_avail)
        caps_ds = xr.open_dataset(snakemake.input.nrel_caps)

        # NREL caps are keyed by substation ID; remap to simpl-cluster bus IDs
        # so they intersect with the profile/region bus space.
        busmap_s = pd.read_csv(snakemake.input.busmap_s, index_col=0, dtype=str).iloc[:, 0]
        busmap_s.index = busmap_s.index.astype(str)
        # Opt-in recovery of out-of-footprint caps entries (default off);
        # reaches the script the same way renewable_land_access does.
        reassign_cfg = snakemake.config.get("nrel_caps_reassign") or {}
        caps_ds = remap_caps_to_cluster(caps_ds, busmap_s, tech=tech, reassign=reassign_cfg)
        logger.info(f"Remapped NREL caps to {caps_ds.sizes['bus']} cluster buses.")

        mapping = get_cell_to_bus_mapping(
            ds_cf["x"].values,
            ds_cf["y"].values,
            [snakemake.input.regions],
            cache_dir=snakemake.params.mapping_cache_dir,
        )
        logger.info(f"Cell→bus mapping: {mapping['name'].nunique()} buses, {len(mapping)} cell rows")

        agg = weighted_bus_aggregation(ds_cf["capacity_factor"], avail, mapping)
        profile = agg["profile"].sel(time=renewable_sns)

        capacities = caps_ds["weight"]
        p_nom_max = caps_ds["p_nom_max"]
        potential = caps_ds["potential"]
        average_distance = caps_ds["average_distance"]

        region_buses = buses.values.astype(profile.bus.dtype)
        common_buses = sorted(
            set(profile.bus.values) & set(capacities.bus.values) & set(region_buses),
        )
        # Empty intersection with non-empty inputs means the three bus-ID
        # spaces are formatted differently (e.g. "35827.0" vs "35827") —
        # writing an empty profile would only crash later in add_electricity.
        if not common_buses and profile.sizes["bus"] > 0 and capacities.sizes["bus"] > 0:
            raise RuntimeError(
                f"godeeep bus IDs share no overlap for {tech}: "
                f"profile buses e.g. {[str(b) for b in profile.bus.values[:3]]}, "
                f"caps buses e.g. {[str(b) for b in capacities.bus.values[:3]]}, "
                f"region buses e.g. {[str(b) for b in region_buses[:3]]}. "
                "Check bus-ID formatting in regions_s{simpl}.geojson (cluster_simpl) "
                "and busmap_s{simpl}.csv.",
            )
        profile = profile.sel(bus=common_buses)
        capacities = capacities.sel(bus=common_buses)
        p_nom_max = p_nom_max.sel(bus=common_buses)
        potential = potential.sel(bus=common_buses)
        average_distance = average_distance.sel(bus=common_buses)

        logger.info(f"Profile: {profile.shape}  Capacities: {capacities.shape}")

    # ds of renewable data to be outputted
    ds = xr.merge(
        [
            profile.rename("profile"),
            capacities.rename("weight"),
            p_nom_max.rename("p_nom_max"),
            potential.rename("potential"),
            average_distance.rename("average_distance"),
        ],
        compat="override",
    )

    # Adding 'underwater_fraction' for offshore wind only
    if snakemake.wildcards.technology.startswith("offwind"):
        if snakemake.params.renewable.get("dataset", False) == "atlite":
            logger.info("Calculate underwater fraction of connections.")
            offshore_shape = gpd.read_file(snakemake.input["offshore_shapes"]).unary_union
            underwater_fraction = []
            for bus in buses:
                p = centre_of_mass.sel(bus=bus).data
                line = LineString([p, regions.loc[bus, ["x", "y"]]])
                frac = line.intersection(offshore_shape).length / line.length
                underwater_fraction.append(frac)
            ds["underwater_fraction"] = xr.DataArray(underwater_fraction, [buses])
        elif snakemake.params.renewable.get("dataset", False) == "godeeep":
            # underwater_fraction is baked into the NREL caps file by
            # build_nrel_bus_capacities.py.
            ds["underwater_fraction"] = caps_ds["underwater_fraction"].reindex(
                bus=common_buses,
                fill_value=1.0,
            )

    # select only buses with some capacity and minimal capacity factor
    ds = ds.sel(
        bus=(
            (ds["profile"].mean("time") > params.get("min_p_max_pu", 0.0))
            & (ds["p_nom_max"] > params.get("min_p_nom_max", 0.0))
        ),
    )

    if "clip_p_max_pu" in params:
        min_p_max_pu = params["clip_p_max_pu"]
        ds["profile"] = ds["profile"].where(ds["profile"] >= min_p_max_pu, 0)

    # Apply correction factor only at the very end when writing to disk
    if correction_factor != 1.0:
        logger.info(f"Applying correction factor {correction_factor} to profile...")
        ds["profile"] = ds["profile"] * correction_factor

    ds.to_netcdf(snakemake.output.profile)
    if client is not None:
        client.shutdown()
