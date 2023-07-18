# SPDX-FileCopyrightText: : 2017-2022 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
#
# Edits for PyPSA-USA by Kamran Tehranchi

"""

Relevant Settings
-----------------

.. code:: yaml

    countries:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`toplevel_cf`

Inputs
------

- ``resources/country_shapes.geojson``: confer :ref:`shapes`
- ``resources/offshore_shapes.geojson``: confer :ref:`shapes`
- ``networks/base.nc``: confer :ref:`base`

Outputs
-------

- ``resources/regions_onshore.geojson``:

    .. image:: ../img/regions_onshore.png
        :scale: 33 %

- ``resources/regions_offshore.geojson``:

    .. image:: ../img/regions_offshore.png
        :scale: 33 %

Description
-----------

Creates Voronoi shapes for each bus representing both onshore and offshore regions.

"""

import logging
from helper_functs import abbrev_to_us_state
import pypsa
import os, sys
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from scipy.spatial import Voronoi
import pdb

import log
logger = log.setup_custom_logger('root')
logger.debug('main message')


from _helpers import configure_logging, REGION_COLS
from simplify_network import aggregate_to_substations

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


def assign_bus_ba(n, ba_region_shapes, offshore_shapes):
    bus_df = n.buses
    bus_df["geometry"] = gpd.points_from_xy(bus_df["x"], bus_df["y"])

    combined_shapes = gpd.GeoDataFrame(pd.concat([ba_region_shapes, offshore_shapes],ignore_index=True))
    
    ba_points = gpd.tools.sjoin(gpd.GeoDataFrame(bus_df["geometry"],crs= 4326), combined_shapes, how='left',predicate='within')
    ba_points = ba_points.rename(columns={'name':'balancing_area'})
    bus_df_final = pd.merge(bus_df, ba_points['balancing_area'], left_index=True, right_index=True,how='left').drop(columns=['geometry'])
    n.buses = bus_df_final
    return n

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_bus_regions')
    configure_logging(snakemake)

    countries = snakemake.config['countries']
    balancing_areas = snakemake.config['BA_names']

    n = pypsa.Network(snakemake.input.base_network)
    import pdb; pdb.set_trace()

    #need to aggregate to substation to ensure building bus regions for only substation level nodes
    n.generators['weight'] = 0 #temporary to enable clustering
    busmap_to_sub = pd.read_csv(
        snakemake.input.bus2sub, index_col=0, dtype={"sub_id": str}
    )
    busmap_to_sub.index = busmap_to_sub.index.astype(str)
    substations = pd.read_csv(snakemake.input.sub, index_col=0)
    substations.index = substations.index.astype(str)
    n = aggregate_to_substations(n, substations, busmap_to_sub.sub_id)

    gpd_countries = gpd.read_file(snakemake.input.country_shapes).set_index('name')
    gpd_ba_shapes = gpd.read_file(snakemake.input.ba_region_shapes)
    ba_region_shapes = gpd_ba_shapes.set_index('name')['geometry']

    gpd_offshore_shapes = gpd.read_file(snakemake.input.offshore_shapes)
    offshore_shapes = gpd_offshore_shapes.reindex(columns=REGION_COLS).set_index('name')['geometry']


    onshore_regions = []
    offshore_regions = []

    if snakemake.params.balancing_authorities["use"]==False: #Ignore BA shapes, use only USA Outline
        logger.info("Building bus regions for %s", len(countries))
        for country in countries:
            c_b = n.buses.country == country
            country_geometry = gpd_countries['geometry']
            ba_shape = country_geometry[country]
            ba_locs = n.buses.loc[c_b & n.buses.substation_lv, ["x", "y"]]
            onshore_regions.append(gpd.GeoDataFrame({
                    'name': ba_locs.index,
                    'x': ba_locs['x'],
                    'y': ba_locs['y'],
                    'geometry': voronoi_partition_pts(ba_locs.values, ba_shape),
                    'country': country
                }))

            if country not in offshore_shapes.index: continue
            offshore_shape = offshore_shapes[country]
            offshore_locs = n.buses.loc[c_b & n.buses.substation_off, ["x", "y"]]
            offshore_regions_c = gpd.GeoDataFrame({
                    'name': offshore_locs.index,
                    'x': offshore_locs['x'],
                    'y': offshore_locs['y'],
                    'geometry': voronoi_partition_pts(offshore_locs.values, offshore_shape),
                    'country': country
                })
            offshore_regions_c = offshore_regions_c.loc[offshore_regions_c.area > 1e-2]
            offshore_regions.append(offshore_regions_c)

    else: # Use Balancing Authority shapes
        logger.info("Building bus regions for Balancing Authorities in %s interconnect/region", snakemake.wildcards.interconnect)
        n = assign_bus_ba(n,gpd_ba_shapes, gpd_offshore_shapes)

        for ba in balancing_areas:
            if ba not in ba_region_shapes.index: continue #filter only ba's in interconnection 
            # print('defining bus regions for ', ba)
            ba_shape = ba_region_shapes[ba]
            all_locs = n.buses.loc[n.buses.substation_lv, ["x", "y"]] 

            # ba_locs contains the bus name and locations for all buses in the BA for ba_shape.
            ba_buses = n.buses.balancing_area[n.buses.balancing_area == ba]
            ba_locs = all_locs.loc[ba_buses.index]
            if ba_locs.empty: continue #skip empty BA's which are not in the bus dataframe. ex. eastern texas BA when using the WECC interconnect

            onshore_regions.append(gpd.GeoDataFrame({
                    'name': ba_locs.index,
                    'x': ba_locs['x'],
                    'y': ba_locs['y'],
                    'geometry': voronoi_partition_pts(ba_locs.values, ba_shape),
                    'country': ba,
                }))
            
            n.buses.loc[ba_locs.index, 'country'] = ba #adds abbreviation to the bus dataframe under the country column
            n.buses.loc['37584', 'country'] = 'CISO-SDGE'   #hot fix for imperial beach substation being offshore

        ### Defining Offshore Regions ###
        for i in range(len(offshore_shapes)):
            offshore_shape = offshore_shapes[i]
            shape_name = offshore_shapes.index[i]
            bus_locs = n.buses.loc[n.buses.substation_off, ["x", "y"]] #substation off all true?
            bus_points = gpd.points_from_xy(x=bus_locs.x, y=bus_locs.y)
            offshore_busses = bus_locs[[offshore_shape.buffer(0.2).contains(bus_points[i]) for i in range(len(bus_points))]]  #filter for OSW busses within shape
            offshore_regions_c = gpd.GeoDataFrame({
                'name': offshore_busses.index,
                'x': offshore_busses['x'],
                'y': offshore_busses['y'],
                'geometry': voronoi_partition_pts(offshore_busses.values, offshore_shape),
                'country': shape_name,})
            offshore_regions_c = offshore_regions_c.loc[offshore_regions_c.area > 1e-2]
            offshore_regions.append(offshore_regions_c)

            n.buses.loc[offshore_busses.index, 'country'] = shape_name #adds offshore shape name to the bus dataframe under the country column
            
        ### Remove Extra OSW Busses and Branches ###
        #Removes remaining nodes in network left with country = US (these are offshore busses that are not in the offshore shape or onshore shapes)
        pdb.set_trace()
        #To-do- add filter that checks if the buses being removed are over water. Currently this works for WECC since I have cleaned up the GEOJSON files
        n.mremove("Line", n.lines.loc[n.lines.bus1.isin(n.buses.loc[n.buses.country=='US'].index)].index) 
        n.mremove("Load", n.loads.loc[n.loads.bus.isin(n.buses.loc[n.buses.country=='US'].index)].index)
        n.mremove("Generator", n.generators.loc[n.generators.bus.isin(n.buses.loc[n.buses.country=='US'].index)].index)
        n.mremove("Bus",  n.buses.loc[n.buses.country=='US'].index)


    n.export_to_netcdf(snakemake.output.network)

    pd.concat(onshore_regions, ignore_index=True).to_file(snakemake.output.regions_onshore)
    if offshore_regions:
        pd.concat(offshore_regions, ignore_index=True).to_file(snakemake.output.regions_offshore)
    else:
        offshore_shapes.to_frame().to_file(snakemake.output.regions_offshore)

#TODO: #14 Move all network modifications to build_base_network.