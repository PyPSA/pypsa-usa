# SPDX-FileCopyrightText: : 2017-2022 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
#
# Edits for PyPSA-USA by Kamran Tehranchi (Stanford)

"""
Creates Voronoi shapes for each bus representing both onshore and offshore regions.

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


Workflow:

Build shapes ->  build_bus_regions -> simplify_network  -> cluster_network

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

sys.path.append(os.path.join(os.getcwd(), "subworkflows", "pypsa-eur", "scripts"))
from _helpers import configure_logging, REGION_COLS


logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_bus_regions')
    configure_logging(snakemake)

    countries = snakemake.config['countries']
    states = snakemake.config['states']

    
    n = pypsa.Network(snakemake.input.base_network)

    country_shapes = gpd.read_file(snakemake.input.country_shapes).set_index('name')['geometry']
    state_shapes = gpd.read_file(snakemake.input.state_shapes).set_index('name')['geometry']

    offshore_shapes = gpd.read_file(snakemake.input.offshore_shapes)
    offshore_shapes = offshore_shapes.reindex(columns=REGION_COLS).set_index('name')['geometry']

    onshore_regions = []
    offshore_regions = []

    if snakemake.params.use_state_shapes==False: #Use country shapes to define voronoi regions vs state shapes.
        logger.info("Building bus regions for %s", countries)
        for country in countries:
            c_b = n.buses.country == country

            onshore_shape = country_shapes[country]
            onshore_locs = n.buses.loc[c_b & n.buses.substation_lv, ["x", "y"]]
            onshore_regions.append(gpd.GeoDataFrame({
                    'name': onshore_locs.index,
                    'x': onshore_locs['x'],
                    'y': onshore_locs['y'],
                    'geometry': voronoi_partition_pts(onshore_locs.values, onshore_shape),
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
    else: # when using Custom State/Balancing Authority shapes
        logger.info("Building bus regions for states in %s", snakemake.wildcards.interconnect)
        for state in states:
            # c_b = n.buses.country == 'US' #dont really need this line?
            if snakemake.params.balancing_authorities["use"]:
                if state not in state_shapes.index: continue #need for interconnection wildcard
                onshore_shape = state_shapes[state]
            else:
                if abbrev_to_us_state[state] not in state_shapes.index: continue
                onshore_shape = state_shapes[abbrev_to_us_state[state]]
            onshore_locs = n.buses.loc[n.buses.substation_lv, ["x", "y"]] 
            bus_points = gpd.points_from_xy(x=onshore_locs.x, y=onshore_locs.y)
            state_locs = pd.DataFrame(columns=['x','y'])
            state_locs = pd.concat([state_locs,onshore_locs[[onshore_shape.contains(bus_points[i]) for i in range(len(bus_points)) ]]])

            if state_locs.empty: continue #skip empty BA's which are not in the bus dataframe. ex. eastern texas BA when using the WECC interconnect
            if state== 'SPP-WAUE': continue #revisit this later

            onshore_regions.append(gpd.GeoDataFrame({
                    'name': state_locs.index,
                    'x': state_locs['x'],
                    'y': state_locs['y'],
                    'geometry': voronoi_partition_pts(state_locs.values, onshore_shape),
                    'country': state,
                }))
            n.buses.loc[state_locs.index, 'country'] = state #adds state abbreviation to the bus dataframe under the country column

        ### Defining Offshore Regions ###

        for i in range(len(offshore_shapes)):
            offshore_shape = offshore_shapes[i]
            shape_name = offshore_shapes.index[i]
            bus_locs = n.buses.loc[n.buses.substation_off, ["x", "y"]] #substation off isn't doing anything here, all substations are "offshore"
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
        # n.mremove("Bus",  n.buses.loc[n.buses.country=='US'].index) #remove extra OSW busses from the network, OSW buses need to be tied to a BOEM Call area region. This causes error, need to address later.

        # offshore_shape = offshore_shapes['US']
        # offshore_locs = n.buses.loc[n.buses.country == 'US'].loc[n.buses.substation_off, ["x", "y"]]
        # pdb.set_trace()

        # bus_points = gpd.points_from_xy(x=offshore_locs.x, y=offshore_locs.y)
        # offshore_busses = offshore_locs[[offshore_shape.buffer(200000).contains(bus_points[i]) for i in range(len(bus_points))]]     #checks if offshore bus is within the offshore shape

        # offshore_regions_c = gpd.GeoDataFrame({
        #         'name': offshore_locs.index,
        #         'x': offshore_locs['x'],
        #         'y': offshore_locs['y'],
        #         'geometry': voronoi_partition_pts(offshore_locs.values, offshore_shape),
        #         'country': 'US'
        #     })
        # offshore_regions_c = offshore_regions_c.loc[offshore_regions_c.area > 1e-2]
        # offshore_regions.append(offshore_regions_c)


    n.export_to_netcdf(snakemake.output.network)

    # pdb.set_trace()
    # n.buses.to_csv('/Users/kamrantehranchi/Library/CloudStorage/OneDrive-Stanford/Kamran_OSW/PyPSA_Models/pypsa-breakthroughenergy-usa/workflow/resources/western/busmap_s.csv')

    pd.concat(onshore_regions, ignore_index=True).to_file(snakemake.output.regions_onshore)
    if offshore_regions:
        pd.concat(offshore_regions, ignore_index=True).to_file(snakemake.output.regions_offshore)
    else:
        offshore_shapes.to_frame().to_file(snakemake.output.regions_offshore)