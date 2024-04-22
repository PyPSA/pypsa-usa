"""
Builds mapping between cutout grid cells and population (total, urban, rural).

**Relevant Settings**

.. code:: yaml

    scope:

**Inputs**

- ``data/counties/cb_2020_us_county_500k.shp``: County shapes in the USA
- ``data/population/DECENNIALDHC2020.P1-Data.csv``: Population per county in the USA,

    .. image:: _static/pop_layout/population.png
        :scale: 33 %

- ``data/urbanization/DECENNIALDHC2020.H2-Data.csv``: Urbanization rate per county in the USA

    .. image:: _static/pop_layout/urban.png
        :scale: 33 %

- ``cutouts/" + CDIR + "{interconnect}_{cutout}.nc``: : confer :ref:cutout

**Outputs**

- ``resources/pop_layout_total.nc``: total population by grid cell

    .. image:: _static/pop_layout/pop_layout_total.png
        :scale: 80 %

- ``resources/pop_layout_urban.nc``: urban population by grid cell

    .. image:: _static/pop_layout/pop_layout_urban.png
        :scale: 80 %

- ``resources/pop_layout_rural.nc``: rural population by grid cell

    .. image:: _static/pop_layout/pop_layout_rural.png
        :scale: 80 %
"""

import logging

logger = logging.getLogger(__name__)

from pathlib import Path

import atlite
import constants
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from _helpers import configure_logging, mock_snakemake


def load_urban_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loads data to get urban and rural values at a GEOID level.

    Extracts data from the following source:
        https://data.census.gov/
        FILTERS: Decennial Census - Universe: Housing units - 2020: DEC Demographic and Housing Characteristics

    Note
    ----
    When reading in the data, be sure to skip the first row:
    > pd.read_csv("./UrbanArea.csv", skiprows=1)
    """
    df = df.set_index("Geography")
    df.index = df.index.str[-5:]  # extract GEOID value
    df.index.name = "GEOID"
    df = df.rename(columns={x: x.strip() for x in df.columns})
    df = df.rename(
        columns={
            "!!Total:": "total",
            "!!Total:!!Urban": "urban",
            "!!Total:!!Rural": "rural",
        },
    )
    df["URBAN"] = (df["urban"] / df["total"]).round(2)  # ratios
    df["RURAL"] = (df["rural"] / df["total"]).round(2)
    df = df[["Geographic Area Name", "URBAN", "RURAL"]]
    return df


def load_population(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loads population data at a GEOID level.

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
    df = df.rename(columns={x: x.strip() for x in df.columns})
    df = df.rename(columns={"!!Total": "population"})
    df = df[["Geographic Area Name", "population"]]
    return df


def plot_county_data(
    gdf: gpd.GeoDataFrame,
    col: str,
    title: str = None,
    description: str = None,
    save: str = None,
):
    """
    Plots heat map of geodataframe.

    Adapted from
    https://www.relataly.com/visualize-covid-19-data-on-a-geographic-heat-maps/291/
    """

    # for legend range
    vmin = gdf[col].min()
    vmax = gdf[col].max()
    cmap = "viridis"

    # Create figure and axes for Matplotlib
    fig, ax = plt.subplots(1, figsize=(20, 8))
    ax.axis("off")
    gdf.plot(column=col, ax=ax, edgecolor="0.8", linewidth=1, cmap=cmap)

    ax.set_title(title, fontdict={"fontsize": "25", "fontweight": "3"})
    ax.annotate(
        description,
        xy=(0.1, 0.08),
        xycoords="figure fraction",
        horizontalalignment="left",
        verticalalignment="bottom",
        fontsize=10,
    )

    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm._A = []  # Empty array for the data range
    cbaxes = fig.add_axes([0.15, 0.25, 0.01, 0.4])
    cbar = fig.colorbar(sm, cax=cbaxes)

    if save:
        fig.savefig(save)


def plot_grid_data(da: xr.DataArray, title: str = None, save: str = None):
    """
    Plots gridded population layout.
    """

    fig, ax = plt.subplots(figsize=(5, 4))
    da.plot(ax=ax, cbar_kwargs={"label": "population"})
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    if title:
        ax.set_title(title)

    if save:
        fig.savefig(save)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_population_layouts",
            interconnect="western",
            cutout="era5_2019",
        )
        # for plotting
        save_path = Path("..", "..", "docs", "source", "_static", "pop_layout")
    else:
        save_path = None

    configure_logging(snakemake)

    cutout = atlite.Cutout(snakemake.input.cutout)

    grid_cells = cutout.grid.geometry

    # retrive county level population data
    counties = gpd.read_file(snakemake.input.county_shapes).set_index("GEOID")
    counties = counties[~(counties.STATE_NAME.isin(constants.STATES_TO_REMOVE))]

    # extract urban fraction in each county
    urban_fraction = pd.read_csv(snakemake.input.urban_percent, skiprows=1)
    urban_fraction = load_urban_ratio(urban_fraction)

    # extract population in each county
    pop = pd.read_csv(snakemake.input.population, skiprows=1)
    pop = load_population(pop)

    # merge population and urban fraction data
    pop = pop.join(urban_fraction, rsuffix="_pop")
    pop = pop.drop(columns=["Geographic Area Name_pop"])
    pop["STATE"] = pop["Geographic Area Name"].map(lambda x: x.split(",")[1].strip())
    pop = pop[~(pop.STATE.isin(constants.STATES_TO_REMOVE))]

    # calcualte urban and rural populations
    pop["urban_population"] = pop["population"] * pop["URBAN"]
    pop["rural_population"] = pop["population"] * pop["RURAL"]

    # merge population data with county information
    counties = counties.join(pop)

    # Indicator matrix counties -> grid cells
    I = atlite.cutout.compute_indicatormatrix(counties.geometry, grid_cells)

    # population in each grid cell
    cell_pop = pd.Series(I.dot(counties["population"]))
    cell_rural_pop = pd.Series(I.dot(counties["rural_population"]))
    cell_urban_pop = pd.Series(I.dot(counties["urban_population"]))

    # save total, rural, urban population
    pops = {
        "total": cell_pop,
        "rural": cell_rural_pop,
        "urban": cell_urban_pop,
    }

    for key, pop in pops.items():
        ycoords = ("y", cutout.coords["y"].data)
        xcoords = ("x", cutout.coords["x"].data)
        values = pop.values.reshape(cutout.shape)
        layout = xr.DataArray(values, [ycoords, xcoords])
        layout.to_netcdf(snakemake.output[f"pop_layout_{key}"])

        # plot data
        if save_path:
            save = Path(save_path, f"pop_layout_{key}.png")
            title = f"{key} population by grid cell"
            plot_grid_data(layout, title, save)

    # plot data
    if save_path:

        plotting_data = counties.copy()
        plotting_data["density_person_per_km2"] = (
            plotting_data.population / counties.ALAND * 1000000
        )
        columns = {
            "Geographic Area Name": "name",
            "NAMELSAD": "county",
            "STATE_NAME": "state",
            "population": "population",
            "ALAND": "land_area_m2",
            "density_person_per_km2": "density_person_per_km2",
            "URBAN": "urban_area",
            "geometry": "geometry",
        }
        plotting_data = plotting_data[columns.keys()]
        plotting_data = plotting_data.rename(columns=columns)

        title = "Population by County"
        column = "population"
        description = "Source: https://data.census.gov/ \nDecennial Census - Universe: Total population - 2020: DEC Demographic and Housing Characteristics"
        save = Path(save_path, "population.png")
        plot_county_data(plotting_data, column, title, description, str(save))

        title = "Population Density by County (person/km2)"
        column = "density_person_per_km2"
        description = "Source: https://data.census.gov/ \nDecennial Census - Universe: Total population - 2020: DEC Demographic and Housing Characteristics"
        save = Path(save_path, "population_density.png")
        save = Path(save_path, "density.png")
        plot_county_data(plotting_data, column, title, description, str(save))

        title = "Urban Density by County"
        column = "urban_area"
        save = Path(save_path, "urban.png")
        description = "Source: https://data.census.gov/ \nDecennial Census - Universe: Housing units - 2020: DEC Demographic and Housing Characteristics"
        plot_county_data(plotting_data, column, title, description, str(save))

    # Below is akin to the PyPSA-Eur implementation of rural/urbal areas. They
    # build up cells based on population density to hit a generic urbanization rate
    # for a country. As we have urban rates at a county level, for the time
    # being we will just use that

    """
    # in km^2
    cell_areas = grid_cells.to_crs(3035).area / 1e6

    # pop per km^2
    density_cells = cell_pop / cell_areas

    # Indicator matrix grid_cells -> counties; inprinciple Iinv*I is identity
    # but imprecisions mean not perfect
    Iinv = cutout.indicatormatrix(counties.geometry)

    # calcualte rural and urban population per cell
    cell_rural_pop = pd.Series(0.0, density_cells.index)
    cell_urban_pop = pd.Series(0.0, density_cells.index)

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
