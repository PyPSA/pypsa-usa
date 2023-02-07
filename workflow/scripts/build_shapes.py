import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import pandas as pd, os, sys

'''
Description: 
This script builds the shapes for the interconnect identified in the config file. The shapes are built using the states in the interconnect and the offshore shapes specified in the config file. geojson files are saved to the resources folder.

Kamran Modifications:
- Added a new input parameter to the config file: source_offshore_shapes to accomodate for NREL Offshore shapes and BOEM offshore shapes.
- Fixed issue to allow filter by interconnection.

Build shapes ->  build_bus_regions -> simplify_network  -> cluster_network
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
     

states.to_file(snakemake.output.state_shapes)

countries = gpd.GeoDataFrame([[states.unary_union, "US"]], columns=["geometry", "name"])
countries = countries.set_crs(4326)
countries.to_file(snakemake.output.country_shapes)

offshore = gpd.read_file(snakemake.params.source_offshore_shapes)

if 'weather.gov' in snakemake.params.source_offshore_shapes :
    offshore.rename(columns={"NAME": "name", "LAT": "y", "LON": "x"}, inplace=True)

# Find adjacent offshore areas
crs = ccrs.Mollweide()
countries = countries.to_crs(crs).buffer(snakemake.params.buffer_distance)
countries = gpd.GeoSeries(countries[0], offshore.index)
overlap_b = offshore.to_crs(crs).buffer(0).intersects(countries[0])
offshore = offshore[overlap_b].unary_union
offshore = gpd.GeoDataFrame([[offshore, "US"]], columns=["geometry", "name"])
offshore = offshore.set_crs(4326)
offshore.to_file(snakemake.output.offshore_shapes)

# ################################### Testing ########################################
# import geopandas as gpd
# import cartopy.crs as ccrs
# import cartopy.io.shapereader as shpreader
# import pandas as pd, os, sys

# interconnect= 'western' #givin in config
# offshore_config = 'ca'
# country_shapes="resources/{interconnect}/country_shapes.geojson"
# state_shapes="resources/{interconnect}/state_shapes.geojson"
# offshore_shapes="resources/{interconnect}/offshore_shapes.geojson"
# sys.path.append(os.path.join(os.getcwd(),'workflow'))


# offshore_ca_shape='/Users/kamrantehranchi/Library/CloudStorage/OneDrive-Stanford/Kamran_OSW/PyPSA_Models/pypsa-breakthroughenergy-usa/workflow/data/BOEM_CA_OSW_GIS/CA_OSW_BOEM_CallAreas.shp'

# offshore_nrel_shape='/Users/kamrantehranchi/Library/CloudStorage/OneDrive-Stanford/Kamran_OSW/PyPSA_Models/pypsa-breakthroughenergy-usa/workflow/data/Offshore_Wind_Speed_90m/Offshore_Wind_Speed_90m.shp'

# zone="/Users/kamrantehranchi/Library/CloudStorage/OneDrive-Stanford/Kamran_OSW/PyPSA_Models/pypsa-breakthroughenergy-usa/workflow/data/base_grid/zone.csv"
# states_in_interconnect = pd.read_csv(zone)
# states_in_interconnect= states_in_interconnect[states_in_interconnect['interconnect'].str.contains(interconnect, na=False, case=False)]


# shapename = "admin_1_states_provinces" #given in config
# shpfilename = shpreader.natural_earth(
#     resolution="110m", category="cultural", name=shapename)
# reader = shpreader.Reader(shpfilename)
# states = reader.records()
# data = []
# for r in states:
#     attr = r.attributes
#     data.append([attr["name"], attr["latitude"], attr["longitude"], r.geometry])

# states = gpd.GeoDataFrame(data, columns=["name", "x", "y", "geometry"]).set_crs(4326)
# states = states.query("name not in ['Alaska', 'Hawaii']")

# if interconnect != "usa": #filter states in interconnect
#     states = states.query("name in @states_in_interconnect.state") 

# # states.to_file(state_shapes)

# countries = gpd.GeoDataFrame([[states.unary_union, "US"]], columns=["geometry", "name"])
# countries = countries.set_crs(4326)
# # countries.to_file(country_shapes)

# if offshore_config == 'weathergov':
#     offshore = gpd.read_file("https://www.weather.gov/source/gis/Shapefiles/WSOM/oz22mr22.zip")
#     offshore.rename(columns={"NAME": "name", "LAT": "y", "LON": "x"}, inplace=True)
# elif offshore_config == 'ca':
#     offshore = gpd.read_file(offshore_ca_shape)
# elif offshore_config == 'nrel':
#     offshore = gpd.read_file(offshore_nrel_shape)
# else:
#     raise ValueError("Unknown offshore_config: {}".format(offshore_config))

# # Find adjacent offshore areas
# crs = ccrs.Mollweide()
# countries = countries.to_crs(crs).buffer(200000)

# countries = gpd.GeoSeries(countries[0], offshore.index)
# overlap_b = offshore.to_crs(crs).buffer(0).intersects(countries[0])
# offshore = offshore[overlap_b].unary_union
# offshore = gpd.GeoDataFrame([[offshore, "US"]], columns=["geometry", "name"])
# offshore = offshore.set_crs(4326)

# offshore.to_file('/Users/kamrantehranchi/Library/CloudStorage/OneDrive-Stanford/Kamran_OSW/PyPSA_Models/offshore_shapes.geojson')


# '''
# USA interconnection + nrel and weathergov, works
# western interconnect + weathergov, works

# CA offshore has its coordinates in another projection, so it doesnt work

# '''
