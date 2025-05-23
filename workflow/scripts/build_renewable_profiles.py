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
import xarray as xr
from _helpers import configure_logging, get_snapshots
from dask.distributed import Client
from pypsa.geo import haversine
from shapely.geometry import LineString

logger = logging.getLogger(__name__)


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
            interconnect="texas",
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
    logger.info(f'using cutout "{snakemake.input.cutout}"')
    cutout = atlite.Cutout(snakemake.input.cutout[0]).sel(
        time=sns,
    )  # Patch fix with [0] move expand in snakemake rule to add_elec for multiple cutouts

    regions = gpd.read_file(snakemake.input.regions)

    assert "x" in regions.columns and "y" in regions.columns, (
        f"List of regions in {snakemake.input.regions} is empty, please disable the corresponding renewable technology"
    )
    # do not pull up, set_index does not work if geo dataframe is empty
    regions = regions.set_index("name").rename_axis("bus")
    buses = regions.index

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

    # fig, ax = plt.subplots()
    # excluder.plot_shape_availability(regions, ax=ax)
    fig, ax = plot_data(availability.sum("bus"))
    ax.set_title(f"Availability of {snakemake.wildcards.technology} Technology")
    plt.savefig(snakemake.output.availability)
    plt.close(fig)

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
    if p_nom_max_meth == "simple":
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

    ds = xr.merge(
        [
            (correction_factor * profile).rename("profile"),
            capacities.rename("weight"),
            p_nom_max.rename("p_nom_max"),
            potential.rename("potential"),
            average_distance.rename("average_distance"),
        ],
    )
    if snakemake.wildcards.technology.startswith("offwind"):
        logger.info("Calculate underwater fraction of connections.")
        offshore_shape = gpd.read_file(snakemake.input["offshore_shapes"]).unary_union
        underwater_fraction = []
        for bus in buses:
            p = centre_of_mass.sel(bus=bus).data
            line = LineString([p, regions.loc[bus, ["x", "y"]]])
            frac = line.intersection(offshore_shape).length / line.length
            underwater_fraction.append(frac)

        ds["underwater_fraction"] = xr.DataArray(underwater_fraction, [buses])

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

    ds.to_netcdf(snakemake.output.profile)
    if client is not None:
        client.shutdown()
