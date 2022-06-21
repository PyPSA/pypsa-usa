import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader


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
if snakemake.wildcards.interconnect != "usa":
    states = states[
        states.name.str.contains(snakemake.wildcards.interconnect, case=False)
    ]
states.to_file(snakemake.output.state_shapes)

countries = gpd.GeoDataFrame([[states.unary_union, "US"]], columns=["geometry", "name"])
countries = countries.set_crs(4326)
countries.to_file(snakemake.output.country_shapes)

offshore = gpd.read_file(snakemake.params.source_offshore_shapes)
offshore.rename(columns={"NAME": "name", "LAT": "y", "LON": "x"}, inplace=True)

# Find adjacent offshore areas
crs = ccrs.Mollweide()
countries = countries.to_crs(crs).buffer(2000)
countries = gpd.GeoSeries(countries[0], offshore.index)
overlap_b = offshore.to_crs(crs).buffer(0).intersects(countries[0])
offshore = offshore[overlap_b].unary_union
offshore = gpd.GeoDataFrame([[offshore, "US"]], columns=["geometry", "name"])
offshore = offshore.set_crs(4326)
offshore.to_file(snakemake.output.offshore_shapes)
