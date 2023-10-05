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

The `build_shapes` rule builds the GIS shape files for the balancing authorities and offshore regions. The regions are only built for the {interconnect} wildcard. Because balancing authorities often overlap- we modify the GIS dataset developed by  [Breakthrough Energy Sciences](https://breakthrough-energy.github.io/docs/). The offshore regions are built from the BOEM and weather.gov datasets.

"""

import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import pandas as pd
import logging
from _helpers import mock_snakemake, configure_logging
from typing import List
import matplotlib.pyplot as plt

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
    if source == "weather.gov":
        offshore = _combine_weather_gov_shape(shape, interconnect, buffer)
    elif source == "ca_osw":
        offshore = _dissolve_boem(shape)
    elif source == "eez":
        offshore = _dissolve_eez(shape, interconnect, buffer)
    else:
        logger.error(f"source {source} is invalid offshore data source")
        offshore = None
    return offshore
        
def _combine_weather_gov_shape(shape: gpd.GeoDataFrame, interconnect: gpd.GeoDataFrame, buffer: int = 1000):
    """Combines offshore shapes from weather.gov"""
    crs = ccrs.Mollweide()
    shape = shape.rename(columns={"NAME": "name", "LAT": "y", "LON": "x"})
    gdf_regions_union = interconnect.to_crs(crs).buffer(buffer)
    gdf_regions_union = gpd.GeoSeries(gdf_regions_union[0], shape.index)
    overlap = shape.to_crs(crs).buffer(0).intersects(gdf_regions_union[0])
    offshore = shape[overlap].unary_union
    return gpd.GeoDataFrame([[offshore, "US"]], columns=["geometry", "name"])
    
def _dissolve_boem(shape: gpd.GeoDataFrame):
    """Dissolves offshore shapes from boem"""
    # crs = ccrs.Mollweide()
    shape_split = shape.dissolve().explode(index_parts=False) 
    # overlap = shape.to_crs(crs).buffer(0).intersects(shape_combine.to_crs(crs)) #maybe typo since overlap not used? 
    shape_split.rename(columns={"Lease_Name": "name"}, inplace=True)
    shape_split.name = ['Morro_Bay','Humboldt']
    return shape_split

def _dissolve_eez(shape: gpd.GeoDataFrame, interconnect: gpd.GeoDataFrame, buffer: int = 1000):
    """Dissolves offshore shapes of eez. Creates a buffer around the interconnect and subtracts the interconnect from the eez- since no wind will be built very close to shore.... TBD getting better shape files using research already done on this topic."""
    crs = ccrs.Mollweide()
    # shape_split = shape.dissolve().explode(index_parts=False)         
    shape_split = shape.explode(index_parts=False) 
    shape_split = gpd.GeoSeries.to_crs(shape_split, crs= 4326)

    # shape_split.rename(columns={"GEONAME": "name"}, inplace=True)
    # buffered_interconnect = interconnect.to_crs(crs).buffer(buffer)
    # shape_split = shape_split.to_crs(crs).difference(buffered_interconnect.unary_union)
    # shape_split = gpd.GeoSeries.to_crs(shape_split, crs= 4326)
    return shape_split

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_shapes', interconnect='texas')
    configure_logging(snakemake)

    interconnect = snakemake.wildcards.interconnect
    breakthrough_zones = pd.read_csv(snakemake.input.zone)
    breakthrough_zones= breakthrough_zones[breakthrough_zones['interconnect'].str.contains(interconnect, na=False, case=False)]

    # get usa states and territories
    gdf_na = load_na_shapes()
    gdf_na = gdf_na.query("name not in ['Alaska', 'Hawaii']")
    
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

    # save interconnection regions 
    interconnect_regions = gpd.GeoDataFrame([[gdf_states.unary_union, "NERC_Interconnect"]], columns=["geometry", "name"])
    interconnect_regions = interconnect_regions.set_crs(4326)
    interconnect_regions.to_file(snakemake.output.country_shapes)

    # save state shapes 
    state_boundaries = gdf_states[["name", "country", "geometry"]].set_crs(4326)
    state_boundaries.to_file(snakemake.output.state_shapes)

    # Load balancing authority shapes
    gdf_ba = load_ba_shape(snakemake.input.onshore_shapes)

    # Only include balancing authorities which have intersection with interconnection filtered states
    ba_states_intersect =  gdf_ba['geometry'].apply(
        lambda shp: shp.intersects(interconnect_regions.dissolve().iloc[0]['geometry']))
    ba_states = gdf_ba[ba_states_intersect]
    
    gdf_ba_states = ba_states.copy() # setting with copy
    gdf_ba_states.rename(columns={"name_1": "name"})
    gdf_ba_states.to_file(snakemake.output.onshore_shapes)

    # load offshore shapes
    offshore_config = snakemake.params.source_offshore_shapes
    if offshore_config == "ca_osw":
        offshore = gpd.read_file(snakemake.input.offshore_shapes_ca_osw)
    elif offshore_config == "eez":
        offshore = gpd.read_file(snakemake.input.offshore_shapes_eez)

    #filter buffer from shore
    buffer_distance = 40000 # buffer distance for offshore shapes from shore.
    crs = ccrs.Mollweide()
    buffered_na = gdf_na.to_crs(crs).buffer(buffer_distance)
    offshore = offshore.to_crs(crs).difference(buffered_na.unary_union)

    offshore = combine_offshore_shapes(
        source=offshore_config,
        shape=offshore, 
        interconnect=interconnect_regions, 
        buffer=buffer_distance
    )

    offshore_c = offshore.set_crs(4326)
    offshore_c.to_file(snakemake.output.offshore_shapes)