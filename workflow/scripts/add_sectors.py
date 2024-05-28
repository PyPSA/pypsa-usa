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
from typing import Optional

from _helpers import configure_logging, get_snapshots
from build_natural_gas import (
    StateGeometry,
    build_natural_gas,
    convert_generators_2_links,
)
from constants import STATE_2_CODE, STATES_INTERCONNECT_MAPPER
from shapely.geometry import Point

CODE_2_STATE = {v: k for k, v in STATE_2_CODE.items()}


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


def add_sector_foundation(
    n: pypsa.Network,
    carrier: str,
    center_points: Optional[pd.DataFrame] = pd.DataFrame(),
) -> None:
    """
    Adds carrier and state level bus for the energy carrier.
    """

    if carrier == "gas":
        carrier_kwargs = {"color": "#d35050", "nice_name": "Natural Gas"}
    elif carrier == "coal":
        carrier_kwargs = {"color": "#d35050", "nice_name": "Coal"}
    elif carrier == "lpg":
        carrier_kwargs = {"color": "#d35050", "nice_name": "Liquid Petroleum Gas"}
    else:
        carrier_kwargs = {}

    # make primary energy carriers

    if carrier not in n.carriers.index:
        n.add("Carrier", carrier, **carrier_kwargs)

    # make state level primary energy carrier buses

    states = n.buses.STATE.dropna().unique()

    zero_center_points = pd.DataFrame(
        index=states,
        columns=["x", "y"],
        dtype=float,
    ).fillna(0)
    zero_center_points.index.name = "STATE"

    if not center_points.empty:
        points = center_points.loc[states].copy()
        points = (
            pd.concat([points, zero_center_points])
            .reset_index(names=["STATE"])
            .drop_duplicates(keep="first", subset="STATE")
            .set_index("STATE")
        )
    else:
        points = zero_center_points.copy()

    points["name"] = points.index.map(CODE_2_STATE)
    points["interconnect"] = points.index.map(STATES_INTERCONNECT_MAPPER)

    buses_to_create = [f"{x} {carrier}" for x in points.index]
    existing = n.buses[n.buses.index.isin(buses_to_create)].STATE.dropna().unique()

    points = points[~points.index.isin(existing)]

    n.madd(
        "Bus",
        names=points.index,
        suffix=f" {carrier}",
        x=points.x,
        y=points.y,
        carrier=carrier,
        unit="MWh_th",
        interconnect=points.interconnect,
        country=points.index,  # for consistency
        STATE=points.index,
        STATE_NAME=points.name,
    )


def split_loads_by_carrier(n: pypsa.Network):
    """
    Splits loads by carrier.

    At this point, all loads (ie. com-elec, com-heat, com-cool) will be
    nested under the elec bus. This function will create a new bus-load
    pair for each energy carrier
    """

    for bus in n.buses.index.unique():
        df = n.loads[n.loads.bus == bus][["bus", "carrier"]]

        n.madd(
            "Bus",
            df.index,
            v_nom=1,
            x=n.buses.at[bus, "x"],
            y=n.buses.at[bus, "y"],
            carrier=df.carrier,
            country=n.buses.at[bus, "country"],
            interconnect=n.buses.at[bus, "interconnect"],
            STATE=n.buses.at[bus, "STATE"],
            STATE_NAME=n.buses.at[bus, "STATE_NAME"],
        )

    n.loads["bus"] = n.loads.index


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_sectors",
            interconnect="texas",
            # simpl="",
            clusters="20",
            ll="v1.0",
            opts="500SEG",
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
            for x, y in STATES_INTERCONNECT_MAPPER.items()
            if y in ("western", "eastern", "texas")
        ]
    else:
        states_2_map = [
            x
            for x, y in STATES_INTERCONNECT_MAPPER.items()
            if y == snakemake.wildcards.interconnect
        ]

    assign_bus_2_state(n, snakemake.input.county, states_2_map, CODE_2_STATE)

    sns = get_snapshots(snakemake.params.snapshots)

    ###
    # Sector addition starts here
    ###

    split_loads_by_carrier(n)

    # build this first! As it will build primary energy buses for the state
    build_natural_gas(
        n=n,
        year=sns[0].year,
        api=snakemake.params.api["eia"],
        interconnect=snakemake.wildcards.interconnect,
        county_path=snakemake.input.county,
        pipelines_path=snakemake.input.pipeline_capacity,
        pipeline_shape_path=snakemake.input.pipeline_shape,
    )

    center_points = StateGeometry(snakemake.input.county).state_center_points.set_index(
        "STATE",
    )
    for carrier in ("oil", "coal"):
        add_sector_foundation(n, carrier, center_points)
        convert_generators_2_links(n, carrier, f" {carrier}")

    n.export_to_netcdf(snakemake.output.network)
