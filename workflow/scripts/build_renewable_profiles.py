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
from scipy import sparse
from shapely.geometry import LineString
from zenodo_downloader import ZenodoScenarioDownloader

logger = logging.getLogger(__name__)


# Get renewable snapshots for a given year using month/day from config
def get_renewable_snapshots(config, year):
    """Build hourly renewable snapshots for a given weather year."""
    ren_sns_config = config.get("renewable_snapshots", {})

    if "start_month" in ren_sns_config:
        start_month = ren_sns_config.get("start_month", 1)
        start_day = ren_sns_config.get("start_day", 1)
        end_month = ren_sns_config.get("end_month", 12)
        end_day = ren_sns_config.get("end_day", 31)
        end_inclusive = ren_sns_config.get("end_inclusive", False)

        start = pd.Timestamp(
            year=year,
            month=start_month,
            day=start_day,
        )
        end = pd.Timestamp(
            year=year,
            month=end_month,
            day=end_day,
        )

        # An inclusive end date represents the complete calendar day.
        if end_inclusive:
            end += pd.Timedelta(days=1)

        snapshots_config = {
            "start": str(start),
            "end": str(end),
            "inclusive": "left",
        }
    else:
        # Old format fallback
        snapshots_config = {
            "start": f"{year}-01-01",
            "end": f"{year + 1}-01-01",
            "inclusive": "left",
        }

    renewable_snapshots = get_snapshots(snapshots_config)

    logger.info(
        "Using %s renewable snapshots for year %s: %s to %s",
        len(renewable_snapshots),
        year,
        renewable_snapshots[0],
        renewable_snapshots[-1],
    )

    return renewable_snapshots


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


def aggregate_godeeep_grid(filepath, regions, chunk_size=168):
    """
    Aggregate a gridded GODEEEP capacity-factor dataset by bus region.

    Grid cells are assigned to regions using their centre coordinates. The
    capacity factor of a bus is the equally weighted mean of its assigned
    cells. The time dimension is processed in chunks to limit memory use.
    """
    with xr.open_dataset(filepath, decode_times=False) as dataset:
        required = {"capacity_factor", "XLONG", "XLAT"}
        missing = required.difference(dataset.variables)

        if missing:
            raise ValueError(
                f"Missing required variables in {filepath}: {sorted(missing)}",
            )

        capacity_factor = dataset["capacity_factor"]

        expected_dims = ("Time", "south_north", "west_east")
        if capacity_factor.dims != expected_dims:
            raise ValueError(
                f"Unexpected GODEEEP capacity-factor dimensions: {capacity_factor.dims}; expected {expected_dims}",
            )

        longitude = dataset["XLONG"].values.ravel()
        latitude = dataset["XLAT"].values.ravel()

        valid_coordinates = np.isfinite(longitude) & np.isfinite(latitude)
        cell_indices = np.flatnonzero(valid_coordinates)

        grid_points = gpd.GeoDataFrame(
            {"cell": cell_indices},
            geometry=gpd.points_from_xy(
                longitude[valid_coordinates],
                latitude[valid_coordinates],
            ),
            crs="EPSG:4326",
        )

        region_geometry = regions[["geometry"]].copy()

        if region_geometry.crs is None:
            raise ValueError("The renewable regions do not define a CRS")

        region_geometry = region_geometry.to_crs(grid_points.crs)
        region_geometry = region_geometry.reset_index()

        assignments = gpd.sjoin(
            grid_points,
            region_geometry,
            how="inner",
            predicate="within",
        )

        # Boundary points may not be matched by "within".
        unmatched = grid_points.loc[~grid_points.index.isin(assignments.index)]

        if not unmatched.empty:
            boundary_assignments = gpd.sjoin(
                unmatched,
                region_geometry,
                how="inner",
                predicate="intersects",
            )
            assignments = pd.concat(
                [assignments, boundary_assignments],
                ignore_index=True,
            )

        if assignments.empty:
            raise ValueError(
                "No GODEEEP grid-cell centres intersect the renewable regions",
            )

        assignments = assignments.drop_duplicates(subset="cell", keep="first")

        region_bus_order = regions.index.astype(str)
        bus_lookup = pd.Series(
            np.arange(len(region_bus_order)),
            index=region_bus_order,
        )

        assignments["bus"] = assignments["bus"].astype(str)
        assignments = assignments[assignments["bus"].isin(bus_lookup.index)].copy()

        bus_rows = bus_lookup.loc[assignments["bus"]].to_numpy()

        cells_per_bus = np.bincount(
            bus_rows,
            minlength=len(region_bus_order),
        )

        buses_with_cells = cells_per_bus > 0

        if not buses_with_cells.all():
            logger.warning(
                "%d of %d bus regions contain no GODEEEP grid-cell centre",
                (~buses_with_cells).sum(),
                len(region_bus_order),
            )

        selected_buses = region_bus_order[buses_with_cells]
        selected_bus_lookup = pd.Series(
            np.arange(len(selected_buses)),
            index=selected_buses,
        )

        assignments = assignments[assignments["bus"].isin(selected_buses)].copy()

        matrix_rows = selected_bus_lookup.loc[assignments["bus"]].to_numpy()
        matrix_columns = assignments["cell"].to_numpy()

        selected_counts = np.bincount(
            matrix_rows,
            minlength=len(selected_buses),
        )

        weights = 1.0 / selected_counts[matrix_rows]

        aggregation_matrix = sparse.csr_matrix(
            (
                weights,
                (matrix_rows, matrix_columns),
            ),
            shape=(
                len(selected_buses),
                capacity_factor.sizes["south_north"] * capacity_factor.sizes["west_east"],
            ),
        )

        n_time = capacity_factor.sizes["Time"]
        aggregated = np.empty(
            (n_time, len(selected_buses)),
            dtype=np.float32,
        )

        logger.info(
            "Aggregating %d GODEEEP grid cells into %d bus regions",
            assignments["cell"].nunique(),
            len(selected_buses),
        )

        for start in range(0, n_time, chunk_size):
            stop = min(start + chunk_size, n_time)

            values = capacity_factor.isel(Time=slice(start, stop)).values.reshape(stop - start, -1)

            finite = np.isfinite(values)
            clean_values = np.where(finite, values, 0.0)

            numerator = aggregation_matrix @ clean_values.T
            denominator = aggregation_matrix @ finite.astype(np.float32).T

            chunk = np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.nan, dtype=np.float32),
                where=denominator > 0,
            )

            aggregated[start:stop] = chunk.T

            logger.info(
                "Aggregated GODEEEP hours %d-%d of %d",
                start + 1,
                stop,
                n_time,
            )

        return xr.DataArray(
            aggregated,
            dims=("time", "bus"),
            coords={
                "time": np.arange(n_time),
                "bus": selected_buses,
            },
            name="profile",
        )


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

    # Do not move this before the check above.
    regions = regions.set_index("name").rename_axis("bus")

    if regions.index.has_duplicates:
        logger.warning(
            "Dissolving %d region geometries into %d unique buses",
            len(regions),
            regions.index.nunique(),
        )
        regions = regions.reset_index().dissolve(
            by="bus",
            as_index=True,
            aggfunc="first",
        )

    regions.index = regions.index.astype(str)
    regions.index.name = "bus"
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
        for bus in buses:
            row = layoutmatrix.sel(bus=bus).data
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
            wind_height = "_100m"  ## for now only 100m wind data is available, add functionality for more heights
            start = ((year - 1980) // 20) * 20 + 1980
            end = start + 19
        elif tech == "solar":
            technology = "solar"
            wind_height = ""
            start = ((year - 2020) // 40) * 40 + 2020
            end = start + 39
        else:
            raise ValueError("Invalid technology type. Choose 'onwind', 'offwind', 'offwind_floating' or 'solar'.")

        year_range = "" if scenario == "historical" else f"_{start}_{end}"
        scenario_final = technology + wind_height + f"_{scenario}" + year_range
        aggregated_filename = f"{technology}_gen_cf_{year}{wind_height}_aggregated.nc"

        try:
            filepath = downloader.download_scenario_file(
                scenario_final,
                scenario,
                aggregated_filename,
            )
        except FileNotFoundError:
            gridded_filename = f"{technology}_gen_cf_{year}{wind_height}.nc"

            logger.warning(
                "Aggregated GODEEEP file %s is unavailable; falling back to gridded file %s",
                aggregated_filename,
                gridded_filename,
            )

            filepath = downloader.download_scenario_file(
                scenario_final,
                scenario,
                gridded_filename,
            )
            profile = aggregate_godeeep_grid(filepath, regions)
        else:
            profile = xr.open_dataarray(filepath).load()

        if "Time" in profile.dims and "time" not in profile.dims:
            profile = profile.rename({"Time": "time"})

        if "time" not in profile.dims:
            raise ValueError(
                f"The GODEEEP profile must contain a 'time' or 'Time' dimension. Found dimensions: {profile.dims}",
            )

        if profile.sizes["time"] != len(renewable_sns):
            raise ValueError(
                "Renewable profile length does not match the requested snapshots: "
                f"{profile.sizes['time']} != {len(renewable_sns)}",
            )

        if scenario == "historical":
            profile = profile.sel(time=renewable_sns)
        else:
            # Future files use end-of-interval timestamps. Relabel them with the
            # corresponding start-of-interval model snapshots.
            profile = profile.assign_coords(time=renewable_sns)

        if "bus" not in profile.dims:
            raise ValueError(
                "The downloaded GODEEEP profile is not aggregated by bus. "
                f"Expected a 'bus' dimension, found: {profile.dims}. "
                "A gridded file cannot be used as an aggregated profile.",
            )

        logger.info("Loading preprocessed data from Zenodo...")

        # Extract variables from the preprocessed ERA5/Atlite dataset
        logger.info(f"Pulling preprocessed data for {tech}")

        preprocessed = xr.open_dataset(
            downloader.download_scenario_file(
                "capacities",
                scenario,
                f"profile_{tech}.nc",
            ),
        )

        capacities = preprocessed["weight"]
        p_nom_max = preprocessed["p_nom_max"]
        potential = preprocessed["potential"]
        average_distance = preprocessed["average_distance"]

        region_buses = pd.Index(buses).astype(str)
        godeeep_buses = pd.Index(profile.bus.values).astype(str)
        atlite_buses = pd.Index(capacities.bus.values).astype(str)

        profile = profile.assign_coords(bus=godeeep_buses)
        capacities = capacities.assign_coords(bus=atlite_buses)
        p_nom_max = p_nom_max.assign_coords(
            bus=pd.Index(p_nom_max.bus.values).astype(str),
        )
        average_distance = average_distance.assign_coords(
            bus=pd.Index(average_distance.bus.values).astype(str),
        )

        common_buses = godeeep_buses[godeeep_buses.isin(atlite_buses) & godeeep_buses.isin(region_buses)].tolist()

        if not common_buses:
            raise ValueError(
                "No common buses were found between GODEEEP profiles, preprocessed capacities and renewable regions",
            )

        logger.info(
            "Using %d common buses: %d GODEEEP, %d preprocessed, %d regions",
            len(common_buses),
            len(godeeep_buses),
            len(atlite_buses),
            len(region_buses),
        )

        # Reassign coordinates and filter to common buses
        regions_xy = regions.loc[common_buses]

        regions_x = xr.DataArray(
            regions_xy["x"].to_numpy(dtype=float),
            dims="bus",
            coords={"bus": common_buses},
        )
        regions_y = xr.DataArray(
            regions_xy["y"].to_numpy(dtype=float),
            dims="bus",
            coords={"bus": common_buses},
        )

        potential = potential.sel(
            x=regions_x,
            y=regions_y,
            method="nearest",
        )

        # The vectorized x/y selection introduces a bus dimension.
        # Remove the selected auxiliary coordinates before merging.
        potential = potential.drop_vars(
            ["x", "y"],
            errors="ignore",
        )

        # Filter godeeep profile to only common buses
        logger.info(f"Before filtering Profile shape: {profile.shape}")
        profile = profile.sel(bus=common_buses)

        if profile.sizes["bus"] != len(common_buses):
            raise ValueError(
                "Unexpected bus count after filtering the GODEEEP profile: "
                f"{profile.sizes['bus']} != {len(common_buses)}",
            )

        logger.info("Final data shapes:")
        logger.info(f"Profile: {profile.shape}")
        logger.info(f"Capacities: {capacities.shape}")
        logger.info(f"P_nom_max: {p_nom_max.shape}")
        logger.info(f"Average_distance: {average_distance.shape}")

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
            ds["underwater_fraction"] = preprocessed["underwater_fraction"].sel(bus=common_buses)

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
