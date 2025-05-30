# By PyPSA-USA Authors
"""Creates Voronoi shapes for each bus representing both onshore and offshore regions."""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from _helpers import REGION_COLS, configure_logging
from scipy.spatial import Voronoi
from shapely.geometry import Polygon


def voronoi_partition_pts(points, outline):
    """
    Compute the polygons of a voronoi partition of `points` within the
    polygon `outline`. Taken from
    https://github.com/FRESNA/vresutils/blob/master/vresutils/graph.py.

    Attributes
    ----------
    points : Nx2 - ndarray[dtype=float]
    outline : Polygon

    Returns
    -------
    polygons : N - ndarray[dtype=Polygon|MultiPolygon]
    """
    points = np.asarray(points)

    if len(points) == 1:
        polygons = [outline]
    else:
        xmin, ymin = np.amin(points, axis=0)
        xmax, ymax = np.amax(points, axis=0)
        xspan = xmax - xmin
        yspan = ymax - ymin
        buffer = 70.0
        # to avoid any network positions outside all Voronoi cells, append
        # the corners of a rectangle framing these points
        vor = Voronoi(
            np.vstack(
                (
                    points,
                    [
                        [xmin - buffer * xspan, ymin - buffer * yspan],
                        [xmin - buffer * xspan, ymax + buffer * yspan],
                        [xmax + buffer * xspan, ymin - buffer * yspan],
                        [xmax + buffer * xspan, ymax + buffer * yspan],
                    ],
                ),
            ),
        )

        polygons = []
        for i in range(len(points)):
            poly = Polygon(vor.vertices[vor.regions[vor.point_region[i]]])

            if not poly.is_valid:
                poly = poly.buffer(0)

            poly = poly.intersection(outline)

            polygons.append(poly)

    return np.array(polygons, dtype=object)


def main(snakemake):
    # Params
    topological_boundaries = snakemake.params.topological_boundaries

    logger.info(
        "Building bus regions for %s Interconnect",
        snakemake.wildcards.interconnect,
    )
    logger.info("Built for aggregation with %s zones", topological_boundaries)

    n = pypsa.Network(snakemake.input.base_network)

    # Pulling data for bus2sub map, to ensure bus regions are only built for substations
    bus2sub = pd.read_csv(snakemake.input.bus2sub, index_col=0, dtype={"sub_id": str})
    bus2sub.index = bus2sub.index.astype(str)
    bus2sub = bus2sub.reset_index().drop_duplicates(subset="sub_id").set_index("sub_id")

    gpd_counties = gpd.read_file(snakemake.input.county_shapes).set_index("GEOID")
    gpd_reeds = gpd.read_file(snakemake.input.reeds_shapes).set_index("name")

    match topological_boundaries:
        case "county":
            agg_region_shapes = gpd_counties.geometry
        case "reeds_zone":
            agg_region_shapes = gpd_reeds.geometry
        case _:
            raise ValueError(
                "Valid values for `model_topology: zonal_aggregation:` are `reeds_zone` or `county`",
            )

    gpd_offshore_shapes = gpd.read_file(snakemake.input.offshore_shapes)
    offshore_shapes = gpd_offshore_shapes.reindex(columns=REGION_COLS).set_index(
        "name",
    )["geometry"]

    all_locs = bus2sub[["x", "y"]]
    onshore_buses = n.buses[~n.buses.substation_off]
    bus2sub = pd.merge(
        bus2sub.reset_index(),
        n.buses[["reeds_zone", "reeds_ba"]],
        left_on="Bus",
        right_on=n.buses.index,
    ).set_index("sub_id")
    bus2sub_onshore = bus2sub[bus2sub.Bus.isin(onshore_buses.index)]

    logger.info("Building Onshore Regions")
    onshore_regions = []
    for region in bus2sub_onshore[f"{topological_boundaries}"].unique():
        if region == "p06069":
            pass
        region_shape = agg_region_shapes.loc[f"{region}"]  # current shape
        region_subs = bus2sub_onshore[f"{topological_boundaries}"][
            bus2sub_onshore[f"{topological_boundaries}"] == region
        ]  # series of substations in the current BA
        region_locs = all_locs.loc[region_subs.index]  # locations of substations in the current BA
        if region_locs.empty:
            continue  # skip empty BA's which are not in the bus dataframe. ex. portions of eastern texas BA when using the WECC interconnect

        if region == "MISO-0001":
            region_shape = gpd.GeoDataFrame(geometry=region_shape).dissolve().iloc[0].geometry

        onshore_regions.append(
            gpd.GeoDataFrame(
                {
                    "name": region_locs.index,
                    "x": region_locs["x"],
                    "y": region_locs["y"],
                    "geometry": voronoi_partition_pts(region_locs.values, region_shape),
                    "country": region,
                },
            ),
        )

    onshore_regions_concat = pd.concat(onshore_regions, ignore_index=True)
    onshore_regions_concat = onshore_regions_concat[
        ~onshore_regions_concat.geometry.is_empty
    ]  # removing few buses which don't have geometry
    onshore_regions_concat.set_crs(epsg=4326, inplace=True)
    onshore_regions_concat.to_file(snakemake.output.regions_onshore)
    combined_onshore = onshore_regions_concat.geometry.union_all()

    ### Defining Offshore Regions ###
    logger.info("Building Offshore Regions")
    offshore_regions = []
    buffered = combined_onshore.buffer(0.9)
    for i in range(len(offshore_shapes)):
        offshore_shape = offshore_shapes.iloc[i]
        # Trim shape to be within certain distance from onshore_regions
        offshore_shape = offshore_shape.intersection(buffered)
        shape_name = offshore_shapes.index[i]
        offshore_buses = bus2sub_onshore[["x", "y"]]
        if offshore_buses.empty:
            continue
        offshore_regions_c = gpd.GeoDataFrame(
            {
                "name": offshore_buses.index,
                "x": offshore_buses["x"],
                "y": offshore_buses["y"],
                "geometry": voronoi_partition_pts(
                    offshore_buses.values,
                    offshore_shape,
                ),
                "country": shape_name,
            },
        )
        # remove extremely small regions
        offshore_regions_c = offshore_regions_c.loc[offshore_regions_c.area > 1e-2]
        offshore_regions.append(offshore_regions_c)
    # Exporting
    if offshore_regions:
        (pd.concat(offshore_regions, ignore_index=True).set_crs(epsg=4326).to_file(snakemake.output.regions_offshore))
    else:
        offshore_shapes.to_frame().to_file(snakemake.output.regions_offshore)

    if onshore_regions_concat[onshore_regions_concat.geometry.is_empty].shape[0] > 0:
        raise ValueError("Onshore Buses are missing geometry.")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_bus_regions", interconnect="western")
    configure_logging(snakemake)
    main(snakemake)
