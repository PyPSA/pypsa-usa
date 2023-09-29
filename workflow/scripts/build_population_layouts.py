"""
Build mapping between cutout grid cells and population (total, urban, rural).
"""

import logging

logger = logging.getLogger(__name__)

from _helpers import mock_snakemake, configure_logging
import atlite
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

STATES_TO_REMOVE = [
    "Hawaii", 
    "Alaska", 
    "Commonwealth of the Northern Mariana Islands", 
    "United States Virgin Islands", 
    "Guam", 
    "Puerto Rico", 
    "American Samoa"
]

def load_urban_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Loads data to get urban and rural values at a GEOID level
    
    Extracts data from the following source: 
        https://data.census.gov/
        FILTERS: Decennial Census - Universe: Housing units - 2020: DEC Demographic and Housing Characteristics
        
    Note
    ----
    When reading in the data, be sure to skip the first row:
    > pd.read_csv("./UrbanArea.csv", skiprows=1)
        
    """
    df = df.set_index("Geography")
    df.index = df.index.str[-5:] # extract GEOID value 
    df.index.name = "GEOID"
    df = df.rename(columns={x:x.strip() for x in df.columns})
    df = df.rename(columns={"!!Total:":"total", "!!Total:!!Urban":"urban", "!!Total:!!Rural":"rural"})
    df["URBAN"] = (df["urban"] / df["total"]).round(2) # ratios 
    df["RURAL"] = (df["rural"] / df["total"]).round(2)
    df = df[["Geographic Area Name", "URBAN", "RURAL"]]
    return df

def load_population(df: pd.DataFrame) -> pd.DataFrame:
    """Loads population data at a GEOID level
    
    Extracts data from the following source: 
        https://data.census.gov/
        FILTERS: Decennial Census - Universe: Total population - 2020: DEC Demographic and Housing Characteristics
        
    Note
    ----
    When reading in the data, be sure to skip the first row:
    > pd.read_csv("./population.csv", skiprows=1)
        
    """
    df = df.set_index("Geography")
    df.index = df.index.str[-5:]
    df.index.name = "GEOID"
    df = df.rename(columns={x:x.strip() for x in df.columns})
    df = df.rename(columns={"!!Total":"Population"})
    df = df[["Geographic Area Name", "Population"]]
    return df
    

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake("build_population_layouts", interconnect="western", cutout="era5_2019")
    configure_logging(snakemake)

    cutout = atlite.Cutout(snakemake.input.cutout)

    grid_cells = cutout.grid.geometry

    # retrive county level population data 
    counties = gpd.read_file(snakemake.input.county_shapes).set_index("GEOID")
    counties = counties[~(counties.STATE_NAME.isin(STATES_TO_REMOVE))]

    # Indicator matrix counties -> grid cells
    I = atlite.cutout.compute_indicatormatrix(counties.geometry, grid_cells)

    # Indicator matrix grid_cells -> counties; inprinciple Iinv*I is identity
    # but imprecisions mean not perfect
    Iinv = cutout.indicatormatrix(counties.geometry)

    # extract urban fraction in each county 
    urban_fraction = pd.read_csv(snakemake.input.urban_percent, skiprows=1)
    urban_fraction = load_urban_ratio(urban_fraction)

    # extract population in each county 
    pop = pd.read_csv(snakemake.input.population, skiprows=1)
    pop = load_population(pop)
    pop["STATE"] = pop["Geographic Area Name"].map(lambda x: x.split(",")[1].strip())
    pop = pop[~(pop.STATE.isin(STATES_TO_REMOVE))]

    # population in each grid cell
    cell_pop = pd.Series(I.dot(pop["Population"]))

    # in km^2
    cell_areas = grid_cells.to_crs(3035).area / 1e6

    # pop per km^2
    density_cells = cell_pop / cell_areas

    # rural or urban population in grid cell
    pop_rural = pd.Series(0.0, density_cells.index)
    pop_urban = pd.Series(0.0, density_cells.index)
    
    # calcualte rural and urban population per cell 
    
    
    # save total, rural, urban population 
    pops = {"total": cell_pop}
    pops["rural"] = pop_rural
    pops["urban"] = pop_urban

    for key, pop in pops.items():
        ycoords = ("y", cutout.coords["y"].data)
        xcoords = ("x", cutout.coords["x"].data)
        values = pop.values.reshape(cutout.shape)
        layout = xr.DataArray(values, [ycoords, xcoords])

        layout.to_netcdf(snakemake.output[f"pop_layout_{key}"])

    # Below is akin to the PyPSA-Eur implementation of rural/urbal areas. They 
    # build up cells based on population density to hit a generic urbanization rate 
    # for a country. As we have urban rates at a county level, for the time 
    # being we will just use that
    """
    for geoid in counties.index:
        logger.debug(
            f"The urbanization rate for county {geoid} is {round(urban_fraction.loc[geoid]*100)}%"
        )

        # get cells within the county (geoid)
        indicator_geoid = pop.county.apply(lambda x: 1.0 if x == geoid else 0.0)
        indicator_cells_geoid = pd.Series(Iinv.T.dot(indicator_geoid))

        # get population and density withing the county (geoid)
        density_cells_geoid = indicator_cells_geoid * density_cells
        pop_cells_geoid = indicator_cells_geoid * pop_cells

        # correct for imprecision of Iinv*I
        pop_geoid = pop.loc[pop.county == geoid, "pop"].sum()
        pop_cells_geoid *= pop_geoid / pop_cells_geoid.sum()

        # The first low density grid cells to reach rural fraction are rural
        asc_density_i = density_cells_geoid.sort_values().index
        asc_density_cumsum = pop_cells_geoid[asc_density_i].cumsum() / pop_cells_geoid.sum()
        rural_fraction_ct = 1 - urban_fraction[geoid]
        pop_geoid_rural_b = asc_density_cumsum < rural_fraction_ct
        pop_geoid_urban_b = ~pop_geoid_rural_b

        pop_geoid_rural_b[indicator_cells_geoid == 0.0] = False
        pop_geoid_urban_b[indicator_cells_geoid == 0.0] = False

        pop_rural += pop_geoid_rural_b.where(pop_geoid_rural_b, 0.0)
        pop_urban += pop_geoid_rural_b.where(pop_geoid_urban_b, 0.0)
        
    pop_cells = {"total": pop_cells}
    pop_cells["rural"] = pop_rural
    pop_cells["urban"] = pop_urban

    for key, pop in pop_cells.items():
        ycoords = ("y", cutout.coords["y"].data)
        xcoords = ("x", cutout.coords["x"].data)
        values = pop.values.reshape(cutout.shape)
        layout = xr.DataArray(values, [ycoords, xcoords])

        layout.to_netcdf(snakemake.output[f"pop_layout_{key}"])
    """


