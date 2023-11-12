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
from _helpers import setup_custom_logger
logger = setup_custom_logger('root')
logger.debug('main message')

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


def assign_bus_ba(n: pypsa.Network, ba_region_shapes: gpd.GeoDataFrame, offshore_shapes: gpd.GeoDataFrame) -> pypsa.Network:
    """Assigns Balancing Area to each bus in the network"""
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
        snakemake = mock_snakemake('build_bus_regions', interconnect="western")
    configure_logging(snakemake)

    #Configurations
    countries = snakemake.config['countries']
    voltage_level = snakemake.config["electricity"]["voltage_simplified"]
    aggregation_zones = snakemake.config['clustering']['cluster_network']['aggregation_zones']
    
    logger.info("Building bus regions for %s", snakemake.wildcards.interconnect)
    logger.info("Built for aggregation with %s zones", aggregation_zones)

    logger.info("Building bus regions for %s", snakemake.wildcards.interconnect)
    logger.info("Built for aggregation with %s zones", aggregation_zones)

    n_base = pypsa.Network(snakemake.input.base_network)

    #Aggregating to substation to ensure building bus regions for only substation level nodes
    n_base.generators['weight'] = 0 #temporary to enable clustering
    busmap_to_sub = pd.read_csv(
        snakemake.input.bus2sub, index_col=0, dtype={"sub_id": str}
    )
    busmap_to_sub.index = busmap_to_sub.index.astype(str)
    substations = pd.read_csv(snakemake.input.sub, index_col=0)
    substations.index = substations.index.astype(str)

    n_base, trafo_map = simplify_network_to_voltage_level(n_base, voltage_level)

    #new busmap definition
    busmap_to_sub = n_base.buses.sub_id.astype(int).astype(str).to_frame()

    busmaps = [trafo_map, busmap_to_sub.sub_id]
    busmaps = reduce(lambda x, y: x.map(y), busmaps[1:], busmaps[0])

    n = aggregate_to_substations(n_base, substations, busmap_to_sub.sub_id, aggregation_zones)

    gpd_countries = gpd.read_file(snakemake.input.country_shapes).set_index('name')
    gpd_ba_shapes = gpd.read_file(snakemake.input.ba_region_shapes)
    ba_region_shapes = gpd_ba_shapes.set_index('name')['geometry']

    gpd_offshore_shapes = gpd.read_file(snakemake.input.offshore_shapes)
    offshore_shapes = gpd_offshore_shapes.reindex(columns=REGION_COLS).set_index('name')['geometry']

    onshore_regions = []
    offshore_regions = []


    for ba in ba_region_shapes.index:
        ba_shape = ba_region_shapes[ba]
        all_locs = n.buses.loc[n.buses.substation_lv, ["x", "y"]] 

        # ba_locs contains the bus name and locations for all buses in the BA for ba_shape.
        ba_buses = n.buses.balancing_area[n.buses.balancing_area == ba]
        ba_locs = all_locs.loc[ba_buses.index]
        if ba_locs.empty: continue # skip empty BA's which are not in the bus dataframe. ex. eastern texas BA when using the WECC interconnect

        onshore_regions.append(gpd.GeoDataFrame({
                'name': ba_locs.index,
                'x': ba_locs['x'],
                'y': ba_locs['y'],
                'geometry': voronoi_partition_pts(ba_locs.values, ba_shape),
                'country': ba,
            }))

    ### Defining Offshore Regions ###
    for i in range(len(offshore_shapes)):
        # import pdb; pdb.set_trace()
        offshore_shape = offshore_shapes.iloc[i]
        shape_name = offshore_shapes.index[i]
        bus_locs = n.buses.loc[n.buses.substation_off, ["x", "y"]] #substation off all true?
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


    # balancing_areas = ['AEC', 'AECI', 'AVA', 'Arizona', 'BANC', 'BPAT', 'CHPD', 'CISO-PGAE', 'CISO-SCE', 'CISO-SDGE', 'CISO-VEA', 'SPP-CSWS', 'Carolina', 'DOPD', 'SPP-EDE', 'EPE', 'ERCO-C', 'ERCO-E', 'ERCO-FW', 'ERCO-N', 'ERCO-NC', 'ERCO-S', 'ERCO-SC', 'ERCO-W', 'Florida', 'GCPD', 'SPP-GRDA', 'GRID', 'IID', 'IPCO', 'ISONE-Connecticut', 'ISONE-Maine', 'ISONE-Massachusetts', 'ISONE-New Hampshire', 'ISONE-Rhode Island', 'ISONE-Vermont', 'SPP-KACY', 'SPP-KCPL', 'LDWP', 'SPP-LES', 'MISO-0001', 'MISO-0027', 'MISO-0035', 'MISO-0004', 'MISO-0006', 'MISO-8910', 'SPP-MPS', 'NWMT', 'NEVP', 'SPP-NPPD', 'NYISO-A', 'NYISO-B', 'NYISO-C', 'NYISO-D', 'NYISO-E', 'NYISO-F', 'NYISO-G', 'NYISO-H', 'NYISO-I', 'NYISO-J', 'NYISO-K', 'SPP-OKGE', 'SPP-OPPD', 'PACE', 'PACW', 'PGE', 'PJM_AE', 'PJM_AEP', 'PJM_AP', 'PJM_ATSI', 'PJM_BGE', 'PJM_ComEd', 'PJM_DAY', 'PJM_DEO&K', 'PJM_DLCO', 'PJM_DP&L', 'PJM_Dominion', 'PJM_EKPC', 'PJM_JCP&L', 'PJM_METED', 'PJM_PECO', 'PJM_PENELEC', 'PJM_PEPCO', 'PJM_PPL', 'PJM_PSEG', 'PJM_RECO', 'PNM', 'PSCO', 'PSEI', 'SPP-SECI', 'SOCO', 'SPP-SPRM', 'SPP-SPS', 'TEPC', 'TIDC', 'TVA', 'WACM', 'WALC', 'WAUW','SPP-WAUE_2','SPP-WAUE_3','SPP-WAUE_4','SPP-WAUE_5','SPP-WAUE_6','SPP-WAUE_7','SPP-WAUE_8','SPP-WAUE_9', 'SPP-WFEC', 'SPP-WR']
