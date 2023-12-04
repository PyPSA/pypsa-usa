# By PyPSA-USA Authors

"""

**Relevant Settings**

.. code:: yaml

    interconnect:
    offshore_shape:
    aggregation_zones:
    countries:


**Inputs**

- ``resources/country_shapes.geojson``: confer :ref:`shapes`
- ``resources/offshore_shapes.geojson``: confer :ref:`shapes`
- ``networks/base.nc``: confer :ref:`base`

**Outputs**

- ``resources/regions_onshore.geojson``:

    # .. image:: ../img/regions_onshore.png
    #     :scale: 33 %

- ``resources/regions_offshore.geojson``:

    # .. image:: ../img/regions_offshore.png
    #     :scale: 33 %

**Description**

Creates Voronoi shapes for each bus representing both onshore and offshore regions.

"""

import logging
import pypsa
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from scipy.spatial import Voronoi
from functools import reduce
from _helpers import configure_logging, REGION_COLS
from simplify_network import aggregate_to_substations, simplify_network_to_voltage_level

def voronoi_partition_pts(points, outline):
    """
    Compute the polygons of a voronoi partition of `points` within the
    polygon `outline`. Taken from
    https://github.com/FRESNA/vresutils/blob/master/vresutils/graph.py
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

        # to avoid any network positions outside all Voronoi cells, append
        # the corners of a rectangle framing these points
        vor = Voronoi(np.vstack((points,
                                 [[xmin-3.*xspan, ymin-3.*yspan],
                                  [xmin-3.*xspan, ymax+3.*yspan],
                                  [xmax+3.*xspan, ymin-3.*yspan],
                                  [xmax+3.*xspan, ymax+3.*yspan]])))

        polygons = []
        for i in range(len(points)):
            poly = Polygon(vor.vertices[vor.regions[vor.point_region[i]]])

            if not poly.is_valid:
                poly = poly.buffer(0)

            poly = poly.intersection(outline)

            polygons.append(poly)


    return np.array(polygons, dtype=object)


def main(snakemake):
    #Configurations
    countries = snakemake.config['countries']
    voltage_level = snakemake.config["electricity"]["voltage_simplified"]
    aggregation_zones = snakemake.config['clustering']['cluster_network']['aggregation_zones']

    logger.info("Building bus regions for %s Interconnect", snakemake.wildcards.interconnect)
    logger.info("Built for aggregation with %s zones", aggregation_zones)

    n = pypsa.Network(snakemake.input.base_network)

    #Pulling data for bus2sub map, to ensure bus regions are only built for substations
    bus2sub = pd.read_csv(snakemake.input.bus2sub, index_col=0, dtype={"sub_id": str})
    bus2sub.index = bus2sub.index.astype(str)
    bus2sub = bus2sub.reset_index().drop_duplicates(subset='sub_id').set_index('sub_id')

    gpd_countries = gpd.read_file(snakemake.input.country_shapes).set_index('name')
    gpd_ba_shapes = gpd.read_file(snakemake.input.ba_region_shapes)
    ba_region_shapes = gpd_ba_shapes.set_index('name')['geometry']

    gpd_offshore_shapes = gpd.read_file(snakemake.input.offshore_shapes)
    offshore_shapes = gpd_offshore_shapes.reindex(columns=REGION_COLS).set_index('name')['geometry']

    onshore_regions = []
    offshore_regions = []

    all_locs = bus2sub[["x", "y"]] # all locations of substations in the bus2sub dataframe
    onshore_buses = n.buses[~n.buses.substation_off]
    bus2sub_onshore = bus2sub[bus2sub.Bus.isin(onshore_buses.index)]
    bus2sub_offshore = bus2sub[~bus2sub.Bus.isin(onshore_buses.index)]

    logger.info("Building Onshore Regions")
    for ba in ba_region_shapes.index:
        ba_shape = ba_region_shapes[ba] # current shape
        ba_subs = bus2sub_onshore.balancing_area[bus2sub_onshore.balancing_area == ba] # series of substations in the current BA
        ba_locs = all_locs.loc[ba_subs.index] # locations of substations in the current BA
        if ba_locs.empty: continue # skip empty BA's which are not in the bus dataframe. ex. portions of eastern texas BA when using the WECC interconnect

        onshore_regions.append(gpd.GeoDataFrame({
                'name': ba_locs.index,
                'x': ba_locs['x'],
                'y': ba_locs['y'],
                'geometry': voronoi_partition_pts(ba_locs.values, ba_shape),
                'country': ba,
            }))

    ### Defining Offshore Regions ###
    logger.info("Building Offshore Regions")
    for i in range(len(offshore_shapes)):
        offshore_shape = offshore_shapes.iloc[i]
        shape_name = offshore_shapes.index[i]
        offshore_buses = bus2sub_offshore[["x", "y"]]
        if offshore_buses.empty: continue
        offshore_regions_c = gpd.GeoDataFrame({
            'name': offshore_buses.index,
            'x': offshore_buses['x'],
            'y': offshore_buses['y'],
            'geometry': voronoi_partition_pts(offshore_buses.values, offshore_shape),
            'country': shape_name,})
        offshore_regions_c = offshore_regions_c.loc[offshore_regions_c.area > 1e-2] # remove extremely small regions
        offshore_regions.append(offshore_regions_c)

    onshore_regions_concat = pd.concat(onshore_regions, ignore_index=True)
    onshore_regions_concat.to_file(snakemake.output.regions_onshore)
    if offshore_regions:
        pd.concat(offshore_regions, ignore_index=True).to_file(snakemake.output.regions_offshore)
    else:
        offshore_shapes.to_frame().to_file(snakemake.output.regions_offshore)

    if onshore_regions_concat[onshore_regions_concat.geometry.is_empty].shape[0] > 0:
        ValueError(f"Onshore regions are missing geometry.")

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_bus_regions', interconnect="western")
    configure_logging(snakemake)
    main(snakemake)