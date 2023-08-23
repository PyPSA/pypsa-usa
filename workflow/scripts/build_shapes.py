import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import pandas as pd
import os, sys
import pdb
'''
Description: 
This script builds the shapes for the interconnect identified in the config file. 
The shapes are built using the Balancing Authority Shapes and the offshore shapes specified in the config file. 
Geojson files are saved to the resources folder.

'''

sys.path.append(os.path.join(os.getcwd(),'workflow'))

interconnect = snakemake.wildcards.interconnect
breakthrough_zones = pd.read_csv(snakemake.input.zone)
breakthrough_zones= breakthrough_zones[breakthrough_zones['interconnect'].str.contains(interconnect, na=False, case=False)]

####### Load state shapes #######
shapename = snakemake.params.source_states_shapes
shpfilename = shpreader.natural_earth(
    resolution="10m", category="cultural", name=shapename
)
reader = shpreader.Reader(shpfilename)
gdf_states = reader.records()
data = []
for r in gdf_states:
    attr = r.attributes
    if attr["iso_a2"] in ["US", "CA", "MX"]: # include US and Canada & Mexico
        data.append([attr["name"], attr['iso_a2'], attr["latitude"], attr["longitude"], r.geometry])
gdf_states = gpd.GeoDataFrame(data, columns=["name", "country", "x", "y", "geometry"]).set_crs(4326)

#filter US states and territories
gdf_states = gpd.GeoDataFrame(data, columns=["name", "country", "x", "y", "geometry"]).set_crs(4326)
gdf_states = gdf_states.query("name not in ['Alaska', 'Hawaii']")
if interconnect == 'western': #filter states in interconnect
    regions = breakthrough_zones.state
    regions = pd.concat([
        regions,
        pd.Series(['Baja California', 'British Columbia', 'Alberta'])
        ])
    gdf_states = gdf_states.query("name in @regions.values")
elif interconnect == "texas":
    gdf_states = gdf_states.query("name in @breakthrough_zones.state")
elif interconnect == "eastern":
    regions = breakthrough_zones.state
    regions = pd.concat([
        regions,
        pd.Series(['Seskatchewan','Manitoba',
                   'Ontario', 'Quebec', 
                   'New Brunswick', 'Nova Scotia'])
        ])
    gdf_states = gdf_states.query("name in @regions.values")
else: #Entire US + MX + CA
    regions = breakthrough_zones.state
    regions = pd.concat([
        regions,
        pd.Series(['Baja California', 'British Columbia', 
                   'Alberta', 'Seskatchewan', 'Manitoba', 
                   'Ontario', 'Quebec', 'New Brunswick', 'Nova Scotia'])
        ])
    gdf_states = gdf_states.query("name in @regions.values")

gdf_regions_union = gpd.GeoDataFrame([[gdf_states.unary_union, "NERC_Interconnect"]], columns=["geometry", "name"])
gdf_regions_union = gdf_regions_union.set_crs(4326)
gdf_regions_union.to_file(snakemake.output.country_shapes)

####### Load balancing authority shapes #######
gdf_ba = gpd.read_file(snakemake.params.balancing_areas["path"])
gdf_ba.rename(columns={"BA": "name"}, inplace=True)
gdf_ba.to_crs(4326,inplace=True)


#Only include balancing authorities which have intersection with interconnection filtered states
ba_states_intersect =  gdf_ba['geometry'].apply(lambda shp: shp.intersects(gdf_regions_union.dissolve().iloc[0]['geometry']))
ba_states = gdf_ba[ba_states_intersect]
ba_states.rename(columns={"name_1": "name"}, inplace=True)
gdf_states= ba_states


gdf_states.to_file(snakemake.output.onshore_shapes)


####### Load offshore shapes #######
offshore_config = snakemake.params.source_offshore_shapes
offshore_path = offshore_config['offshore_path'][offshore_config['use']]
offshore = gpd.read_file(offshore_path)

crs = ccrs.Mollweide()
if 'weather.gov' in offshore_path:
    offshore.rename(columns={"NAME": "name", "LAT": "y", "LON": "x"}, inplace=True)
    # Find adjacent offshore areas
    gdf_regions_union = gdf_regions_union.to_crs(crs).buffer(snakemake.params.buffer_distance)
    gdf_regions_union = gpd.GeoSeries(gdf_regions_union[0], offshore.index)
    overlap_b = offshore.to_crs(crs).buffer(0).intersects(gdf_regions_union[0])
    offshore_c = offshore[overlap_b].unary_union
    offshore_c = gpd.GeoDataFrame([[offshore_c, "US"]], columns=["geometry", "name"])
elif 'BOEM' in offshore_path:
    offshore_c = offshore.dissolve().explode(index_parts=False) #combine adjacent offshore areas
    overlap_b = offshore.to_crs(crs).buffer(0).intersects(offshore_c)
    offshore_c.rename(columns={"Lease_Name": "name"}, inplace=True)
    offshore_c.name = ['Morro_Bay','Humboldt']

offshore_c = offshore_c.set_crs(4326)
offshore_c.to_file(snakemake.output.offshore_shapes)