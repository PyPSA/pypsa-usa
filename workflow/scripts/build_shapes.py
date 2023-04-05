import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import pandas as pd, os, sys

import pdb
'''
Description: 
This script builds the shapes for the interconnect identified in the config file. 
The shapes are built using the states or Balancing Authority Shapes and the offshore shapes specified in the config file. 
Geojson files are saved to the resources folder.

Build shapes ->  build_bus_regions -> simplify_network  -> cluster_network

*** Future work:
- Remove the use of "States" as shapes, and directly filter balancing authorities based on their interconnection


'''
sys.path.append(os.path.join(os.getcwd(),'workflow'))
states_in_interconnect = pd.read_csv(snakemake.input.zone)
states_in_interconnect= states_in_interconnect[states_in_interconnect['interconnect'].str.contains(snakemake.wildcards.interconnect, na=False, case=False)]

shapename = snakemake.params.source_states_shapes
shpfilename = shpreader.natural_earth(
    resolution="110m", category="cultural", name=shapename
)
reader = shpreader.Reader(shpfilename)
states = reader.records()
data = []
for r in states:
    attr = r.attributes
    data.append([attr["name"], attr["latitude"], attr["longitude"], r.geometry])

states = gpd.GeoDataFrame(data, columns=["name", "x", "y", "geometry"]).set_crs(4326)
states = states.query("name not in ['Alaska', 'Hawaii']")
if snakemake.wildcards.interconnect != "usa": #filter states in interconnect
        states = states.query("name in @states_in_interconnect.state")
    
countries = gpd.GeoDataFrame([[states.unary_union, "US"]], columns=["geometry", "name"])
countries = countries.set_crs(4326)
countries.to_file(snakemake.output.country_shapes)

if snakemake.params.balancing_authorities["use"]:
    ba = gpd.read_file(snakemake.params.balancing_authorities["path"])
    ba.rename(columns={"BA": "name"}, inplace=True)
    ba.to_crs(4326,inplace=True)

    #Only include balancing authorities which have intersection with interconnection filtered states
    ba_states_intersect =  ba['geometry'].apply(lambda shp: shp.intersects(countries.dissolve().iloc[0]['geometry']))
    ba_states = ba[ba_states_intersect]
    ba_states.rename(columns={"name_1": "name"}, inplace=True)
    # ba_states.drop(columns=['name_2'],inplace=True)
    states= ba_states
    # states = gpd.GeoDataFrame(pd.concat([states,ba],ignore_index=True),crs=4326)

states.to_file(snakemake.output.state_shapes)

offshore_config = snakemake.params.source_offshore_shapes
offshore_path = offshore_config['offshore_path'][offshore_config['use']]
offshore = gpd.read_file(offshore_path)

# offshore = gpd.read_file(snakemake.params.source_offshore_shapes)
crs = ccrs.Mollweide()
if 'weather.gov' in offshore_path:
# if 'weather.gov' in snakemake.params.source_offshore_shapes:
    offshore.rename(columns={"NAME": "name", "LAT": "y", "LON": "x"}, inplace=True)
    # Find adjacent offshore areas
    countries = countries.to_crs(crs).buffer(snakemake.params.buffer_distance)
    countries = gpd.GeoSeries(countries[0], offshore.index)
    overlap_b = offshore.to_crs(crs).buffer(0).intersects(countries[0])
    offshore_c = offshore[overlap_b].unary_union
    offshore_c = gpd.GeoDataFrame([[offshore_c, "US"]], columns=["geometry", "name"])
elif 'BOEM' in offshore_path:
    offshore_c = offshore.dissolve().explode(index_parts=False) #combine adjacent offshore areas
    overlap_b = offshore.to_crs(crs).buffer(0).intersects(offshore_c)
    offshore_c.rename(columns={"Lease_Name": "name"}, inplace=True)
    offshore_c.name = ['Morro_Bay','Humboldt']

offshore_c = offshore_c.set_crs(4326)
offshore_c.to_file(snakemake.output.offshore_shapes)