# BY PyPSA-USA Authors
"""
**Description**

The `build_shapes` rule builds the GIS shape files for the balancing authorities and offshore regions. The regions are only built for the {interconnect} wildcard. Because balancing authorities often overlap- we modify the GIS dataset developed by  [Breakthrough Energy Sciences](https://breakthrough-energy.github.io/docs/).

**Relevant Settings**

.. code:: yaml

    interconnect:

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
"""


import logging
from typing import List

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from _helpers import configure_logging, mock_snakemake
from constants import *
from shapely.geometry import MultiPolygon


def filter_small_polygons_gpd(
    geo_series: gpd.GeoSeries,
    min_area: float,
) -> gpd.GeoSeries:
    """
    Filters out polygons within each MultiPolygon in a GeoSeries that are
    smaller than a specified area.

    Parameters:
    geo_series (gpd.GeoSeries): A GeoSeries containing MultiPolygon geometries.
    min_area (float): The minimum area threshold.

    Returns:
    gpd.GeoSeries: A GeoSeries with MultiPolygons filtered based on the area criterion.
    """
    # Explode the MultiPolygons into individual Polygons
    original_crs = geo_series.crs
    exploded = (
        geo_series.to_crs(MEASUREMENT_CRS)
        .explode(index_parts=True)
        .reset_index(drop=True)
    )

    # Filter based on area
    filtered = exploded[exploded.area >= min_area]

    # Aggregate back into MultiPolygons
    # Group by the original index and create a MultiPolygon from the remaining geometries
    aggregated = filtered.groupby(filtered.index).agg(
        lambda x: MultiPolygon(x.tolist()) if len(x) > 1 else x.iloc[0],
    )
    aggregated.set_crs(MEASUREMENT_CRS, inplace=True)
    return aggregated.to_crs(original_crs)


def load_na_shapes(
    state_shape: str = "admin_1_states_provinces_lakes",
    countries: list = ["US"],
) -> gpd.GeoDataFrame:
    """
    Creates geodataframe of north america.
    """
    shpfilename = shpreader.natural_earth(
        resolution="10m",
        category="cultural",
        name=state_shape,
    )
    reader = shpreader.Reader(shpfilename)
    gdf_states = reader.records()
    data = []
    for r in gdf_states:
        attr = r.attributes
        if attr["iso_a2"] in countries:  # include US and Canada & Mexico
            data.append(
                [
                    attr["name"],
                    attr["iso_a2"],
                    attr["latitude"],
                    attr["longitude"],
                    r.geometry,
                ],
            )
    gdf_states = gpd.GeoDataFrame(
        data,
        columns=["name", "country", "x", "y", "geometry"],
    ).set_crs(4326)
    return gdf_states


def filter_shapes(
    data: gpd.GeoDataFrame,
    zones: pd.DataFrame,
    interconnect: str = "western",
    add_regions: list = None,
) -> gpd.GeoDataFrame:
    """
    Filters breakthrough energy zone data by interconnect region.
    """

    if interconnect not in ("western", "texas", "eastern", "usa"):
        logger.warning(f"Interconnector of {interconnect} is not valid")

    regions = zones.state
    if add_regions:
        if not isinstance(add_regions, list):
            add_regions = list(add_regions)
        regions = pd.concat([regions, pd.Series(add_regions)])
    return data.query("name in @regions.values")


def load_ba_shape(ba_file: str) -> gpd.GeoDataFrame:
    """
    Loads balancing authority into a geodataframe.
    """
    gdf = gpd.read_file(ba_file)
    gdf = gdf.rename(columns={"BA": "name"})
    return gdf.to_crs(4326)


def load_reeds_shape(reeds_shapes: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(reeds_shapes)
    gdf = gdf.rename(columns={"rb": "name", "BA_Code": "reeds_ba"})
    return gdf.to_crs(4326)


def load_counties_shape(shp_file: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp_file)
    return gdf.to_crs(4326)


def combine_offshore_shapes(
    source: str,
    shape: gpd.GeoDataFrame,
    interconnect: gpd.GeoDataFrame,
    buffer,
) -> gpd.GeoDataFrame:
    """
    Conbines offshore shapes.
    """
    if source == "ca_osw":
        offshore = _dissolve_boem(shape)
    elif source == "eez":
        offshore = _dissolve_eez(shape, interconnect, buffer)
    else:
        logger.error(f"source {source} is invalid offshore data source")
        offshore = None
    return offshore


def _dissolve_boem(shape: gpd.GeoDataFrame):
    """
    Dissolves offshore shapes from boem.
    """
    shape_split = shape.dissolve().explode(index_parts=False)
    shape_split.rename(columns={"Lease_Name": "name"}, inplace=True)
    shape_split.name = ["Morro_Bay", "Humboldt"]
    return shape_split


def _dissolve_eez(
    shape: gpd.GeoDataFrame,
    interconnect: gpd.GeoDataFrame,
    max_buffer: int,
):
    """
    Dissolves offshore shapes from eez then filters plolygons that are not near
    the interconnect shape.
    """
    shape = filter_small_polygons_gpd(shape, 1e9)
    shape_split = gpd.GeoDataFrame(
        geometry=shape.explode(index_parts=False).geometry,
    ).set_crs(MEASUREMENT_CRS)
    buffered_interconnect = interconnect.to_crs(MEASUREMENT_CRS).buffer(max_buffer)
    union_buffered_interconnect = buffered_interconnect.unary_union
    filtered_shapes = shape_split[shape_split.intersects(union_buffered_interconnect)]
    shape_split = filtered_shapes.to_crs(GPS_CRS)
    return shape_split


def trim_states_to_interconnect(
    gdf_states: gpd.GeoDataFrame,
    gdf_nerc: gpd.GeoDataFrame,
    interconnect: str,
):
    """
    Trims states to only include portions of states in NERC Interconnect.
    """
    if interconnect == "western":
        gdf_nerc_f = gdf_nerc[gdf_nerc.OBJECTID.isin([3, 8, 9])]
        gdf_states = gpd.overlay(
            gdf_states,
            gdf_nerc_f.to_crs(GPS_CRS),
            how="difference",
        )
        texas_geometry = gdf_states.loc[gdf_states.name == "Texas", "geometry"]
        texas_geometry = filter_small_polygons_gpd(texas_geometry, 1e8)
        gdf_states.loc[gdf_states.name == "Texas", "geometry"] = (
            texas_geometry.geometry.values
        )
    elif interconnect == "eastern":
        gdf_nerc_f = gdf_nerc[gdf_nerc.OBJECTID.isin([1, 3, 6, 7])]
        gdf_states = gpd.overlay(
            gdf_states,
            gdf_nerc_f.to_crs(GPS_CRS),
            how="difference",
        )
    return gdf_states


def trim_shape_to_interconnect(
    gdf: gpd.GeoDataFrame,
    interconnect_regions: gpd.GeoDataFrame,
    interconnect: str,
    exclusion_dict: dict = None,
):
    """
    Trim Shapes to only portions inside NERC/State Region to ensure renewables
    not built outside shape region.
    """
    shape_intersect = gdf["geometry"].apply(
        lambda shp: shp.intersects(interconnect_regions.dissolve().iloc[0]["geometry"]),
    )
    shape_state_intersection = gdf[shape_intersect]

    if exclusion_dict is not None and interconnect in exclusion_dict.keys():
        shape_state_intersection = shape_state_intersection[
            ~(
                shape_state_intersection.name.str.contains(
                    "|".join(exclusion_dict[interconnect]),
                )
            )
        ]
    return shape_state_intersection


def main(snakemake):
    interconnect = snakemake.wildcards.interconnect
    breakthrough_zones = pd.read_csv(snakemake.input.zone)
    logger.info("Building GIS Shapes for %s Interconnect", interconnect)

    if interconnect != "usa":
        breakthrough_zones = breakthrough_zones[
            breakthrough_zones["interconnect"].str.contains(
                interconnect,
                na=False,
                case=False,
            )
        ]

    # get North America (na) states and territories
    gdf_na = load_na_shapes()
    gdf_na = gdf_na.query("name not in ['Alaska', 'Hawaii']")

    # Build State Shapes filtered by interconnect
    if interconnect == "western":  # filter states that have any portion in interconnect
        gdf_states = filter_shapes(
            data=gdf_na,
            zones=breakthrough_zones,
            interconnect=interconnect,
        )
    elif interconnect == "texas":
        gdf_states = filter_shapes(
            data=gdf_na,
            zones=breakthrough_zones,
            interconnect=interconnect,
        )
    elif interconnect == "eastern":
        gdf_states = filter_shapes(
            data=gdf_na,
            zones=breakthrough_zones,
            interconnect=interconnect,
        )
    else:
        raise NotImplementedError

    # Trim gdf_states to only include portions of texas in NERC Interconnect
    gdf_nerc = gpd.read_file(snakemake.input.nerc_shapes)
    gdf_states = trim_states_to_interconnect(gdf_states, gdf_nerc, interconnect)

    # Save NERC Interconnection shapes
    interconnect_regions = gpd.GeoDataFrame(
        [[gdf_states.unary_union, "NERC_Interconnect"]],
        columns=["geometry", "name"],
    )
    interconnect_regions = interconnect_regions.set_crs(GPS_CRS)
    interconnect_regions.to_file(snakemake.output.country_shapes)

    # Save State shapes
    state_boundaries = gdf_states[["name", "country", "geometry"]].set_crs(GPS_CRS)
    state_boundaries.to_file(snakemake.output.state_shapes)

    # Load & Trim balancing authority shapes
    gdf_ba = load_ba_shape(snakemake.input.onshore_shapes)
    ba_exclusion = {
        "western": ["MISO", "SPP"],
        "texas": ["MISO", "SPP", "EPE"],
        "eastern": ["PNM", "EPE", "PSCO", "WACM", "ERCO", "NWMT"],
    }
    ba_states = trim_shape_to_interconnect(
        gdf_ba,
        interconnect_regions,
        interconnect,
        ba_exclusion,
    )

    # Save BA shapes
    gdf_ba_states = ba_states.copy()
    gdf_ba_states.rename(columns={"name_1": "name"})
    gdf_ba_states.to_file(snakemake.output.onshore_shapes)

    # Load, Trim, Save REeDs Shapes
    gdf_reeds = load_reeds_shape(snakemake.input.reeds_shapes)
    reeds_exclusion = {
        "western": [
            "p36",
            "p35",
            "p38",
            "p32",
            "p39",
            "p52",
            "p49",
            "p48",
            "p47",
            "p61",
            "p19",
        ],
        "eastern": ["p19", "p34", "p32", "p31", "p18"],
        "texas": ["p31", "p59", "p50", "p85", "p58", "p57", "p66", "p47", "p48"],
    }

    gdf_reeds = trim_shape_to_interconnect(
        gdf_reeds,
        interconnect_regions,
        interconnect,
        reeds_exclusion,
    )
    gdf_reeds.to_file(snakemake.output.reeds_shapes)

    # read county shapes
    # takes ~10min to trim shap to interconnect, so skipping
    gdf_counties = load_counties_shape(snakemake.input.county_shapes)
    gdf_counties.to_file(snakemake.output.county_shapes)

    # Load and build offshore shapes
    offshore_config = snakemake.params.source_offshore_shapes["use"]
    if offshore_config == "ca_osw":
        logger.info("Building Offshore GIS shapes with CA OSW shapes")
        offshore = gpd.read_file(snakemake.input.offshore_shapes_ca_osw)
    elif offshore_config == "eez":
        logger.info("Building Offshore GIS shapes with Exclusive Economic Zones shapes")
        offshore = gpd.read_file(snakemake.input.offshore_shapes_eez)
    else:
        logger.error(f"source {offshore_config} is invalid offshore data source")
        offshore = None

    # Filter buffer from shore for Offshore Shapes
    buffer_distance_min = snakemake.params.offwind_params["min_shore_distance"]
    buffer_distance_max = snakemake.params.offwind_params["max_shore_distance"]

    buffered_na = gdf_na.to_crs(MEASUREMENT_CRS).buffer(buffer_distance_min)
    offshore = offshore.to_crs(MEASUREMENT_CRS).difference(buffered_na.unary_union)
    buffered_states = state_boundaries.to_crs(MEASUREMENT_CRS).buffer(
        buffer_distance_min,
    )
    offshore = offshore.to_crs(MEASUREMENT_CRS).difference(buffered_states.unary_union)
    buffer_states_max = state_boundaries.to_crs(MEASUREMENT_CRS).buffer(
        buffer_distance_max,
    )
    offshore = offshore.to_crs(MEASUREMENT_CRS).intersection(
        buffer_states_max.unary_union,
    )

    offshore = offshore[~offshore.is_empty]  # remove empty polygons
    if offshore.empty:
        raise AssertionError("Offshore wind shape is empty")
    offshore = combine_offshore_shapes(
        source=offshore_config,
        shape=offshore,
        interconnect=gdf_states,
        buffer=buffer_distance_max,
    )
    offshore = offshore.set_crs(GPS_CRS).to_file(snakemake.output.offshore_shapes)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_shapes", interconnect="eastern")
    configure_logging(snakemake)
    main(snakemake)
