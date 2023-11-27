# BY PyPSA-USA Authors

"""

**Relevant Settings**

.. code:: yaml

    interconnect:
    countries:


**Inputs**

- ``breakthrough_network/base_grid/zone.csv``: confer :ref:`base`
- ``repo_data/BA_shapes_new/Modified_BE_BA_Shapes.shp``: confer :ref:`base`
- ``repo_data/BOEM_CA_OSW_GIS/CA_OSW_BOEM_CallAreas.shp``: confer :ref:`base`

**Outputs**

- ``resources/country_shapes.geojson``:

    # .. image:: ../img/regions_onshore.png
    #     :scale: 33 %

- ``resources/onshore_shapes.geojson``:

    # .. image:: ../img/regions_offshore.png
    #     :scale: 33 %

- ``resources/offshore_shapes.geojson``:

    # .. image:: ../img/regions_offshore.png
    #     :scale: 33 %

- ``resources/state_boundaries.geojson``:

    # .. image:: ../img/regions_offshore.png
    #     :scale: 33 %

**Description**

The `build_shapes` rule builds the GIS shape files for the balancing authorities and offshore regions. The regions are only built for the {interconnect} wildcard. Because balancing authorities often overlap- we modify the GIS dataset developed by  [Breakthrough Energy Sciences](https://breakthrough-energy.github.io/docs/).

"""

import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import pandas as pd
import logging
from _helpers import mock_snakemake, configure_logging
from typing import List
from constants import *
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon

def filter_small_polygons_gpd(geo_series: gpd.GeoSeries, min_area: float) -> gpd.GeoSeries:
    """
    Filters out polygons within each MultiPolygon in a GeoSeries that are smaller than a specified area.

    Parameters:
    geo_series (gpd.GeoSeries): A GeoSeries containing MultiPolygon geometries.
    min_area (float): The minimum area threshold.

    Returns:
    gpd.GeoSeries: A GeoSeries with MultiPolygons filtered based on the area criterion.
    """
    # Explode the MultiPolygons into individual Polygons
    original_crs = geo_series.crs
    exploded = geo_series.to_crs(MEASUREMENT_CRS).explode(index_parts=True).reset_index(drop=True)

    # Filter based on area
    filtered = exploded[exploded.area >= min_area]

    # Aggregate back into MultiPolygons
    # Group by the original index and create a MultiPolygon from the remaining geometries
    aggregated = filtered.groupby(filtered.index).agg(lambda x: MultiPolygon(x.tolist()) if len(x) > 1 else x.iloc[0])
    return aggregated

def load_na_shapes(state_shape: str = "admin_1_states_provinces") -> gpd.GeoDataFrame:
    """Creates geodataframe of north america"""
    shpfilename = shpreader.natural_earth(
        resolution="10m", category="cultural", name=state_shape
    )
    reader = shpreader.Reader(shpfilename)
    gdf_states = reader.records()
    data = []
    for r in gdf_states:
        attr = r.attributes
        if attr["iso_a2"] in ["US", "CA", "MX"]: # include US and Canada & Mexico
            data.append([attr["name"], attr['iso_a2'], attr["latitude"], attr["longitude"], r.geometry])
    gdf_states = gpd.GeoDataFrame(data, columns=["name", "country", "x", "y", "geometry"]).set_crs(4326)
    return gdf_states

def filter_shapes(data: gpd.GeoDataFrame, zones: pd.DataFrame, interconnect: str = "western", add_regions: List = None) -> gpd.GeoDataFrame:
    """Filters breakthrough energy zone data by interconnect region"""
    
    if interconnect not in ("western", "texas", "eastern", "usa"):
        logger.warning(f"Interconnector of {interconnect} is not valid")
        
    regions = zones.state
    if add_regions:
        if not isinstance(add_regions, list):
            add_regions = list(add_regions)
        regions = pd.concat([regions, pd.Series(add_regions)])
    return data.query("name in @regions.values")

def load_ba_shape(ba_file: str) -> gpd.GeoDataFrame: 
    """Loads balancing authority into a geodataframe"""
    gdf = gpd.read_file(ba_file)
    gdf = gdf.rename(columns={"BA": "name"})
    return gdf.to_crs(4326)

def combine_offshore_shapes(source: str, shape: gpd.GeoDataFrame, interconnect: gpd.GeoDataFrame, buffer: int = 200000) -> gpd.GeoDataFrame:
    """Conbines offshore shapes"""
    if source == "ca_osw":
        offshore = _dissolve_boem(shape)
    elif source == "eez":
        offshore = _dissolve_eez(shape, interconnect, buffer)
    else:
        logger.error(f"source {source} is invalid offshore data source")
        offshore = None
    return offshore
    
def _dissolve_boem(shape: gpd.GeoDataFrame):
    """Dissolves offshore shapes from boem"""
    # crs = ccrs.Mollweide()
    shape_split = shape.dissolve().explode(index_parts=False) 
    # overlap = shape.to_crs(crs).buffer(0).intersects(shape_combine.to_crs(crs)) #maybe typo since overlap not used? 
    shape_split.rename(columns={"Lease_Name": "name"}, inplace=True)
    shape_split.name = ['Morro_Bay','Humboldt']
    return shape_split

def _dissolve_eez(shape: gpd.GeoDataFrame, interconnect: gpd.GeoDataFrame, buffer: int = 1000):
    """Dissolves offshore shapes from eez then filters plolygons that are not near the interconnect shape"""
    shape = filter_small_polygons_gpd(shape, 1e9) 
    shape_split = gpd.GeoDataFrame(geometry = shape.explode(index_parts=False).geometry).set_crs(MEASUREMENT_CRS)
    buffered_interconnect = interconnect.to_crs(MEASUREMENT_CRS).buffer(1e4)
    union_buffered_interconnect = buffered_interconnect.unary_union    
    filtered_shapes = shape_split[shape_split.intersects(union_buffered_interconnect)]
    shape_split = filtered_shapes.to_crs(GPS_CRS)
    return shape_split

def trim_states_to_interconnect(gdf_states: gpd.GeoDataFrame, gdf_nerc: gpd.GeoDataFrame, interconnect: str):
    """Trims states to only include portions of states in NERC Interconnect"""
    if interconnect == "western":
        gdf_nerc_f = gdf_nerc[gdf_nerc.OBJECTID.isin([3,8,9])]
        gdf_states = gpd.overlay(gdf_states, gdf_nerc_f.to_crs(GPS_CRS), how='difference')
        texas_geometry  = gdf_states.loc[gdf_states.name == 'Texas', 'geometry']
        texas_geometry = filter_small_polygons_gpd(texas_geometry, 1e9)
        gdf_states.loc[gdf_states.name == 'Texas', 'geometry'] = texas_geometry.geometry
    return gdf_states

def main(snakemake):
    interconnect = snakemake.wildcards.interconnect
    breakthrough_zones = pd.read_csv(snakemake.input.zone)
    logger.info("Building GIS Shapes for %s Interconnect", interconnect)

    if interconnect != "usa":
        breakthrough_zones= breakthrough_zones[breakthrough_zones['interconnect'].str.contains(interconnect, na=False, case=False)]

    # get North America (na) states and territories
    gdf_na = load_na_shapes()
    gdf_na = gdf_na.query("name not in ['Alaska', 'Hawaii']")

    # Load NERC Shapes
    gdf_nerc = gpd.read_file(snakemake.input.nerc_shapes)

    # apply interconnect wildcard 
    if interconnect == "western": #filter states in interconnect
        gdf_states = filter_shapes(
            data=gdf_na,
            zones=breakthrough_zones,
            interconnect=interconnect, 
            add_regions=['Baja California', 'British Columbia', 'Alberta']
        )
    elif interconnect == "texas":
        gdf_states = filter_shapes(
            data=gdf_na, 
            zones=breakthrough_zones,
            interconnect=interconnect
        )
    elif interconnect == "eastern":
        gdf_states = filter_shapes(
            data=gdf_na, 
            zones=breakthrough_zones,
            interconnect=interconnect, 
            add_regions=[
                "Saskatchewan",
                "Manitoba",
                "Ontario", 
                "Quebec", 
                "New Brunswick", 
                "Nova Scotia"
            ]
        )
    else: # Entire US + MX + CA
        gdf_states = filter_shapes(
            data=gdf_na, 
            zones=breakthrough_zones,
            interconnect=interconnect, 
            add_regions=[
                "Baja California",
                "British Columbia",
                "Alberta",
                "Saskatchewan",
                "Manitoba",
                "Ontario", 
                "Quebec", 
                "New Brunswick", 
                "Nova Scotia"
            ]
        )

    #Trim gdf_states to only include portions of texas in NERC Interconnect
    gdf_states = trim_states_to_interconnect(gdf_states, gdf_nerc, interconnect)

    # save interconnection regions 
    interconnect_regions = gpd.GeoDataFrame([[gdf_states.unary_union, "NERC_Interconnect"]], columns=["geometry", "name"])
    interconnect_regions = interconnect_regions.set_crs(GPS_CRS)
    interconnect_regions.to_file(snakemake.output.country_shapes)

    # save state shapes 
    state_boundaries = gdf_states[["name", "country", "geometry"]].set_crs(GPS_CRS)
    state_boundaries.to_file(snakemake.output.state_shapes)

    # Load balancing authority shapes
    gdf_ba = load_ba_shape(snakemake.input.onshore_shapes)

    # Only include balancing authorities which have intersection with interconnection filtered states
    ba_states_intersect =  gdf_ba['geometry'].apply(
        lambda shp: shp.intersects(interconnect_regions.dissolve().iloc[0]['geometry']))
    ba_states = gdf_ba[ba_states_intersect]

    gdf_ba_states = ba_states.copy()
    gdf_ba_states.rename(columns={"name_1": "name"})
    gdf_ba_states.to_file(snakemake.output.onshore_shapes)

    # load offshore shapes
    offshore_config = snakemake.params.source_offshore_shapes['use']
    if offshore_config == "ca_osw":
        logger.info("Using CA OSW shapes")
        offshore = gpd.read_file(snakemake.input.offshore_shapes_ca_osw)
    elif offshore_config == "eez":
        logger.info("Using EEZ shapes")
        offshore = gpd.read_file(snakemake.input.offshore_shapes_eez)
    else:
        logger.error(f"source {source} is invalid offshore data source")
        offshore = None

    #filter buffer from shore
    buffer_distance = 1000 # buffer distance for offshore shapes from shore.
    buffered_states = state_boundaries.to_crs(MEASUREMENT_CRS).buffer(buffer_distance)
    offshore = offshore.to_crs(MEASUREMENT_CRS).difference(buffered_states.unary_union)

    offshore = combine_offshore_shapes(
        source=offshore_config,
        shape=offshore, 
        interconnect=gdf_states, 
        buffer=buffer_distance
    )

    offshore_c = offshore.set_crs(GPS_CRS)
    offshore_c.to_file(snakemake.output.offshore_shapes)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_shapes', interconnect='eastern')
    configure_logging(snakemake)
    main(snakemake)

# CURRENT IMPLEMENTATION
#COUNTRY SHAPE = UNION OF STATES THAT ARE CONTAINED IN NERC INTERCONNECT √
#STATE SHAPE = STATES IN NERC INTERCONNECT √
#ONSHORE SHAPE = BA IN STATE SHAPES
#OFFSHORE SHAPE = OFFSHORE SHAPES NEAR STATE SHAPES

# TODO
#COUNTRY SHAPE = NERC INTERCONNECT SHAPES
#STATE SHAPE = PORTIONS OF STATES IN NERC INTERCONNECT
#ONSHORE SHAPE = BA IN STATE SHAPES
#OFFSHORE SHAPE = OFFSHORE SHAPES NEAR BA's