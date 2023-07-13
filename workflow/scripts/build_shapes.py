import geopandas as gpd, cartopy.crs as ccrs, cartopy.io.shapereader as shpreader, pandas as pd, os, sys
'''
Description: 
This script builds the shapes for the interconnect identified in the config file. 
The shapes are built using the Balancing Authority Shapes and the offshore shapes specified in the config file. 
Geojson files are saved to the resources folder.

'''

sys.path.append(os.path.join(os.getcwd(),'workflow'))

breakthrough_zones = pd.read_csv(snakemake.input.zone)
breakthrough_zones= breakthrough_zones[breakthrough_zones['interconnect'].str.contains(snakemake.wildcards.interconnect, na=False, case=False)]

####### Load state shapes #######
shapename = snakemake.params.source_states_shapes
shpfilename = shpreader.natural_earth(
    resolution="110m", category="cultural", name=shapename
)
reader = shpreader.Reader(shpfilename)
gdf_states = reader.records()
data = []
for r in gdf_states:
    attr = r.attributes
    data.append([attr["name"], attr["latitude"], attr["longitude"], r.geometry])

gdf_states = gpd.GeoDataFrame(data, columns=["name", "x", "y", "geometry"]).set_crs(4326)
gdf_states = gdf_states.query("name not in ['Alaska', 'Hawaii']")
if snakemake.wildcards.interconnect != "usa": #filter states in interconnect
        gdf_states = gdf_states.query("name in @breakthrough_zones.state")

gdf_interconnection_states = gpd.GeoDataFrame([[gdf_states.unary_union, "US"]], columns=["geometry", "name"])
gdf_interconnection_states = gdf_interconnection_states.set_crs(4326)
gdf_interconnection_states.to_file(snakemake.output.country_shapes)


####### Load balancing authority shapes #######
if snakemake.params.balancing_authorities["use"]:
    gdf_ba = gpd.read_file(snakemake.params.balancing_authorities["path"])
    gdf_ba.rename(columns={"BA": "name"}, inplace=True)
    gdf_ba.to_crs(4326,inplace=True)

    #Only include balancing authorities which have intersection with interconnection filtered states
    ba_states_intersect =  gdf_ba['geometry'].apply(lambda shp: shp.intersects(gdf_interconnection_states.dissolve().iloc[0]['geometry']))
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
    gdf_interconnection_states = gdf_interconnection_states.to_crs(crs).buffer(snakemake.params.buffer_distance)
    gdf_interconnection_states = gpd.GeoSeries(gdf_interconnection_states[0], offshore.index)
    overlap_b = offshore.to_crs(crs).buffer(0).intersects(gdf_interconnection_states[0])
    offshore_c = offshore[overlap_b].unary_union
    offshore_c = gpd.GeoDataFrame([[offshore_c, "US"]], columns=["geometry", "name"])
elif 'BOEM' in offshore_path:
    offshore_c = offshore.dissolve().explode(index_parts=False) #combine adjacent offshore areas
    overlap_b = offshore.to_crs(crs).buffer(0).intersects(offshore_c)
    offshore_c.rename(columns={"Lease_Name": "name"}, inplace=True)
    offshore_c.name = ['Morro_Bay','Humboldt']

offshore_c = offshore_c.set_crs(4326)
offshore_c.to_file(snakemake.output.offshore_shapes)