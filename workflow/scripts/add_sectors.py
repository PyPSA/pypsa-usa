"""
Generic module to add a new energy network.

Reads in the sector wildcard and will call corresponding scripts. In the
future, it would be good to integrate this logic into snakemake
"""

import logging

import geopandas as gpd
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)
import sys

import constants
from _helpers import configure_logging
from build_natural_gas import build_natural_gas
from shapely.geometry import Point


def assign_bus_2_state(
    n: pypsa.Network,
    shp: str,
    states_2_include: list[str] = None,
    state_2_state_name: dict[str, str] = None,
) -> None:
    """
    Adds a state column to the network buses dataframe.

    The shapefile must be the counties shapefile
    """

    buses = n.buses[["x", "y"]].copy()
    buses["geometry"] = buses.apply(lambda x: Point(x.x, x.y), axis=1)
    buses = gpd.GeoDataFrame(buses, crs="EPSG:4269")

    states = gpd.read_file(shp).dissolve("STUSPS")["geometry"]
    states = gpd.GeoDataFrame(states)
    if states_2_include:
        states = states[states.index.isin(states_2_include)]

    # project to avoid CRS warning from geopandas
    buses_projected = buses.to_crs("EPSG:3857")
    states_projected = states.to_crs("EPSG:3857")
    gdf = gpd.sjoin_nearest(buses_projected, states_projected, how="left")

    n.buses["STATE"] = n.buses.index.map(gdf.index_right)

    if state_2_state_name:
        n.buses["STATE_NAME"] = n.buses.STATE.map(state_2_state_name)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_sectors",
            interconnect="western",
            # simpl="",
            clusters="40",
            ll="v1.25",
            opts="Co2L1.25",
            sector="E-G",
        )
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)

    sectors = snakemake.wildcards.sector.split("-")

    # exit if only electricity network
    if all(s == "E" for s in sectors):
        n.export_to_netcdf(snakemake.output.network)
        sys.exit()

    # map states to each clustered bus

    if snakemake.wildcards.interconnect == "usa":
        states_2_map = [
            x
            for x, y in constants.STATES_INTERCONNECT_MAPPER.items()
            if y in ("western", "eastern", "texas")
        ]
    else:
        states_2_map = [
            x
            for x, y in constants.STATES_INTERCONNECT_MAPPER.items()
            if y == snakemake.wildcards.interconnect
        ]

    code_2_state = {v: k for k, v in constants.STATE_2_CODE.items()}
    assign_bus_2_state(n, snakemake.input.county, states_2_map, code_2_state)

    if "G" in sectors:
        build_natural_gas(
            n=n,
            year=pd.to_datetime(snakemake.params.snapshots["start"]).year,
            api=snakemake.params.api["eia"],
            interconnect=snakemake.wildcards.interconnect,
            county_path=snakemake.input.county,
            pipelines_path=snakemake.input.pipeline_capacity,
            pipeline_shape_path=snakemake.input.pipeline_shape,
        )

    n.export_to_netcdf(snakemake.output.network)
