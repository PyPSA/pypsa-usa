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
    
    logger.info("Building bus regions for %s", snakemake.wildcards.interconnect)
    logger.info("Built for aggregation with %s zones", aggregation_zones)

    logger.info("Building bus regions for %s", snakemake.wildcards.interconnect)
    logger.info("Built for aggregation with %s zones", aggregation_zones)

    n = pypsa.Network(snakemake.input.base_network)

    #Pulling data for bus2sub map, to ensure bus regions are only built for substations
    bus2sub = pd.read_csv(snakemake.input.bus2sub, index_col=0, dtype={"sub_id": str})
    bus2sub.index = bus2sub.index.astype(str)
    bus2sub = bus2sub.loc[n.buses.index]
    substations = pd.read_csv(snakemake.input.sub, index_col=0)
    substations.index = substations.index.astype(str)

    # bus2sub['balancing_area'] = bus2sub.index.map(n.buses.balancing_area)
    # bus2sub['x'] = bus2sub.sub_id.map(substations.lon)
    # bus2sub['y'] = bus2sub.sub_id.map(substations.lat)
    bus2sub = bus2sub.reset_index().set_index('sub_id').drop_duplicates()

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

    logger.info("Building Onshore Regions")
    for ba in ba_region_shapes.index:
        print(ba)
        ba_shape = ba_region_shapes[ba] # current shape
        ba_subs = bus2sub_onshore.balancing_area[bus2sub_onshore.balancing_area == ba] # series of substations in the current BA
        ba_locs = all_locs.loc[ba_subs.index] # locations of substations in the current BA
        if ba_locs.empty: continue # skip empty BA's which are not in the bus dataframe. ex. portions of eastern texas BA when using the WECC interconnect
        if ba =="GRID":
            import pdb; pdb.set_trace()
            #issue im running into is that the ba_locs has some buses without gps coordinates.

        if ba =="MISO-0001":
            ba_shape = gpd.GeoDataFrame(geometry = ba_shape).dissolve().iloc[0].geometry

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
        import pdb; pdb.set_trace()
        offshore_shape = offshore_shapes.iloc[i]
        shape_name = offshore_shapes.index[i]
        bus_locs = bus2sub[["x", "y"]]
        bus_points = gpd.points_from_xy(x=bus_locs.x, y=bus_locs.y)
        offshore_busses = bus_locs[[offshore_shape.buffer(0.2).contains(bus_points[i]) for i in range(len(bus_points))]]  #filter for OSW busses within shape
        if offshore_busses.empty: continue
        offshore_regions_c = gpd.GeoDataFrame({
            'name': offshore_busses.index,
            'x': offshore_busses['x'],
            'y': offshore_busses['y'],
            'geometry': voronoi_partition_pts(offshore_busses.values, offshore_shape),
            'country': shape_name,})
        offshore_regions_c = offshore_regions_c.loc[offshore_regions_c.area > 1e-2]
        offshore_regions.append(offshore_regions_c)


    pd.concat(onshore_regions, ignore_index=True).to_file(snakemake.output.regions_onshore)

    if offshore_regions:
        pd.concat(offshore_regions, ignore_index=True).to_file(snakemake.output.regions_offshore)
    else:
        offshore_shapes.to_frame().to_file(snakemake.output.regions_offshore)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_bus_regions', interconnect="usa")
    configure_logging(snakemake)
    main(snakemake)


# balancing_areas = ['AEC', 'AECI', 'AVA', 'Arizona', 'BANC', 'BPAT', 'CHPD', 'CISO-PGAE', 'CISO-SCE', 'CISO-SDGE', 'CISO-VEA', 'SPP-CSWS', 'Carolina', 'DOPD', 'SPP-EDE', 'EPE', 'ERCO-C', 'ERCO-E', 'ERCO-FW', 'ERCO-N', 'ERCO-NC', 'ERCO-S', 'ERCO-SC', 'ERCO-W', 'Florida', 'GCPD', 'SPP-GRDA', 'GRID', 'IID', 'IPCO', 'ISONE-Connecticut', 'ISONE-Maine', 'ISONE-Massachusetts', 'ISONE-New Hampshire', 'ISONE-Rhode Island', 'ISONE-Vermont', 'SPP-KACY', 'SPP-KCPL', 'LDWP', 'SPP-LES', 'MISO-0001', 'MISO-0027', 'MISO-0035', 'MISO-0004', 'MISO-0006', 'MISO-8910', 'SPP-MPS', 'NWMT', 'NEVP', 'SPP-NPPD', 'NYISO-A', 'NYISO-B', 'NYISO-C', 'NYISO-D', 'NYISO-E', 'NYISO-F', 'NYISO-G', 'NYISO-H', 'NYISO-I', 'NYISO-J', 'NYISO-K', 'SPP-OKGE', 'SPP-OPPD', 'PACE', 'PACW', 'PGE', 'PJM_AE', 'PJM_AEP', 'PJM_AP', 'PJM_ATSI', 'PJM_BGE', 'PJM_ComEd', 'PJM_DAY', 'PJM_DEO&K', 'PJM_DLCO', 'PJM_DP&L', 'PJM_Dominion', 'PJM_EKPC', 'PJM_JCP&L', 'PJM_METED', 'PJM_PECO', 'PJM_PENELEC', 'PJM_PEPCO', 'PJM_PPL', 'PJM_PSEG', 'PJM_RECO', 'PNM', 'PSCO', 'PSEI', 'SPP-SECI', 'SOCO', 'SPP-SPRM', 'SPP-SPS', 'TEPC', 'TIDC', 'TVA', 'WACM', 'WALC', 'WAUW','SPP-WAUE_2','SPP-WAUE_3','SPP-WAUE_4','SPP-WAUE_5','SPP-WAUE_6','SPP-WAUE_7','SPP-WAUE_8','SPP-WAUE_9', 'SPP-WFEC', 'SPP-WR']
