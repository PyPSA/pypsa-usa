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
from sklearn.neighbors import BallTree


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

    gpd_reeds = gpd.read_file(snakemake.input.reeds_shapes).set_index("name") #reeds BA shapes
    gpd_counties = gpd.read_file(snakemake.input.county_shapes).set_index("GEOID") #county shapes for the entire US
    agg_region_shapes = gpd_counties.geometry

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
    for region in bus2sub_onshore["county"].unique():
        if region == "p06069":
            pass
        region_shape = agg_region_shapes.loc[f"{region}"]  # current shape
        region_subs = bus2sub_onshore["county"][
            bus2sub_onshore["county"] == region
        ]  # series of substations in the current county
        region_locs = all_locs.loc[region_subs.index]  # locations of substations in the current county
        if region_locs.empty:
            continue  # skip empty counties which are not in the bus dataframe. ex. portions of eastern texas counties when using the WECC interconnect

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

    # Identify empty counties WITHIN the interconnect's BA shapes total footprint (using reeds BA shapes for a cleaner shape)
    combined_bus_regions = gpd_reeds.geometry.union_all()

    # Filter all counties to only those whose centroid is within the interconnect's total footprint
    counties_in_interconnect = {
        c for c in gpd_counties.index
        if gpd_counties.loc[c, "geometry"].centroid.within(combined_bus_regions)
    }

    # Find which of those counties don't have buses
    counties_with_buses = set(onshore_regions_concat["country"].unique())
    empty_counties = counties_in_interconnect - counties_with_buses

    logger.info(
        f"Interconnect footprint contains {len(counties_in_interconnect)} counties, "
        f"{len(counties_with_buses)} have buses, {len(empty_counties)} are empty."
    )

    if empty_counties:
        logger.info(f"Adding {len(empty_counties)} empty counties as separate regions, assigned to nearest bus.")

        # get substation locations for nearest neighbor search
        sub_locs = onshore_regions_concat[["name", "x", "y", "country"]].copy()
        sub_locs = sub_locs.drop_duplicates(subset="name")
        tree = BallTree(sub_locs[["x", "y"]].values, leaf_size=2)

        # build list of empty county entries
        empty_region_rows = []
        for county_id in empty_counties:
            county_geom = gpd_counties.loc[county_id, "geometry"]
            centroid = county_geom.centroid

            # find nearest substation
            _, idx = tree.query([[centroid.x, centroid.y]], k=1)
            nearest_sub = sub_locs.iloc[idx[0][0]]

            # create entry with nearest bus's sub_id as name, but county's own geometry
            empty_region_rows.append({
                "name": nearest_sub["name"],  # assign empty county to nearest bus
                "x": centroid.x,
                "y": centroid.y,
                "geometry": county_geom,  # keep county's own geometry
                "country": county_id,  # county FIPS
            })

        # create GeoDataFrame and append to regions
        empty_regions = gpd.GeoDataFrame(empty_region_rows, crs=onshore_regions_concat.crs)
        onshore_regions_concat = pd.concat(
            [onshore_regions_concat, empty_regions],
            ignore_index=True,
        )

        logger.info(f"Added {len(empty_counties)} empty counties assigned to nearest buses.")
    
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
