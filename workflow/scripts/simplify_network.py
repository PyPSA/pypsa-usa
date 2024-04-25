# BY PyPSA-USA Authors
"""
Aggregates network to substations and simplifies to a single voltage level.
"""


import logging
import os
from functools import reduce

import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging, export_network_for_gis_mapping
from pypsa.clustering.spatial import get_clustering_from_busmap

logger = logging.getLogger(__name__)


def convert_to_per_unit(df):
    # Constants
    sqrt_3 = 3**0.5

    # Calculating base values per component
    # df['base_current'] = df['s_nom'] / (df['v_nom'] * sqrt_3)
    df["base_impedance"] = df["v_nom"] ** 2 / df["s_nom"]
    df["base_susceptance"] = 1 / df["base_impedance"]

    # Converting to per-unit values
    df["resistance_pu"] = df["r"] / df["base_impedance"]
    df["reactance_pu"] = df["x"] / df["base_impedance"]
    df["susceptance_pu"] = df["b"] / df["base_susceptance"]

    # Dropping intermediate columns (optional)
    df.drop(["base_impedance", "base_susceptance"], axis=1, inplace=True)

    return df


def convert_to_voltage_level(n, new_voltage):
    """
    Converts network.lines parameters to a given voltage.

    Parameters:
    n (pypsa.Network): Network
    new_voltage (float): New voltage level
    """
    df = convert_to_per_unit(n.lines.copy())

    # Constants
    sqrt_3 = 3**0.5

    df["new_base_impedance"] = new_voltage**2 / df["s_nom"]

    # Convert per-unit values back to actual values using the new base impedance
    df["r"] = df["resistance_pu"] * df["new_base_impedance"]
    df["x"] = df["reactance_pu"] * df["new_base_impedance"]
    df["b"] = df["susceptance_pu"] / df["new_base_impedance"]

    df.v_nom = new_voltage

    # Dropping intermediate column
    df.drop(
        ["new_base_impedance", "resistance_pu", "reactance_pu", "susceptance_pu"],
        axis=1,
        inplace=True,
    )

    # df.r = df.r.fillna(0) #how to deal with existing components that have zero power capacity s_nom
    # df.x = df.x.fillna(0)
    # df.b = df.b.fillna(0)

    # Update network lines
    (linetype,) = n.lines.loc[n.lines.v_nom == voltage_level, "type"].unique()
    df.type = linetype  # Do I even need to set line types? Can drop.

    n.buses["v_nom"] = voltage_level
    n.lines = df
    return n


def remove_transformers(n):
    trafo_map = pd.Series(n.transformers.bus1.values, index=n.transformers.bus0.values)
    trafo_map = trafo_map[~trafo_map.index.duplicated(keep="first")]
    several_trafo_b = trafo_map.isin(trafo_map.index)
    trafo_map.loc[several_trafo_b] = trafo_map.loc[several_trafo_b].map(trafo_map)
    missing_buses_i = n.buses.index.difference(trafo_map.index)
    missing = pd.Series(missing_buses_i, missing_buses_i)
    trafo_map = pd.concat([trafo_map, missing])

    for c in n.one_port_components | n.branch_components:
        df = n.df(c)
        for col in df.columns:
            if col.startswith("bus"):
                df[col] = df[col].map(trafo_map)

    n.mremove("Transformer", n.transformers.index)
    n.mremove("Bus", n.buses.index.difference(trafo_map))
    return n, trafo_map


def aggregate_to_substations(
    network: pypsa.Network,
    substations,
    busmap,
    aggregation_zones: str,
    aggregation_strategies=dict(),
):
    """
    Aggregate network to substations.

    First step in clusterings, if use_ba_zones is True, then the network
    retains balancing Authority zones in clustering.
    """

    logger.info("Aggregating buses to substation level...")

    line_strategies = aggregation_strategies.get("lines", dict())
    generator_strategies = aggregation_strategies.get("generators", dict())
    one_port_strategies = aggregation_strategies.get("one_ports", dict())

    clustering = get_clustering_from_busmap(
        network,
        busmap,
        aggregate_generators_weighted=True,
        aggregate_one_ports=["Load", "StorageUnit"],
        line_length_factor=1.0,
        bus_strategies={
            "type": "max",
            "Pd": "sum",
        },
        generator_strategies=generator_strategies,
    )

    substations = network.buses[
        [
            "sub_id",
            "interconnect",
            "state",
            "country",
            "balancing_area",
            "reeds_zone",
            "reeds_ba",
            "x",
            "y",
        ]
    ]
    substations = substations.drop_duplicates(subset=["sub_id"])
    substations.sub_id = substations.sub_id.astype(int).astype(str)
    substations.index = substations.sub_id

    if aggregation_zones == "balancing_area":
        zone = substations.balancing_area
    elif aggregation_zones == "country":
        zone = substations.country
    elif aggregation_zones == "state":
        zone = substations.state
    elif aggregation_zones == "reeds_zone":
        zone = substations.reeds_zone
    else:
        ValueError("zonal_aggregation must be either balancing_area, country or state")

    network_s = clustering.network

    network_s.buses["interconnect"] = substations.interconnect
    network_s.buses["x"] = substations.x
    network_s.buses["y"] = substations.y
    network_s.buses["substation_lv"] = True
    network_s.buses["country"] = (
        zone  # country field used bc pypsa-eur aggregates based on country boundary
    )
    network_s.lines["type"] = np.nan

    if aggregation_zones != "reeds_zone":
        cols2drop = [
            "balancing_area",
            "state",
            "substation_off",
            "sub_id",
            "reeds_zone",
            "reeds_ba",
            "nerc_reg",
            "trans_reg",
            "reeds_state",
        ]
    else:
        cols2drop = ["balancing_area", "substation_off", "sub_id", "state"]

    network_s.buses.drop(
        columns=cols2drop,
        inplace=True,
    )

    return network_s


def assign_line_lengths(n, line_length_factor):
    """
    Assign line lengths to network.

    Uses haversine function to calculate line lengths.
    """
    logger.info("Assigning line lengths using haversine function...")

    n.lines.length = pypsa.geo.haversine_pts(
        n.buses.loc[n.lines.bus0][["x", "y"]],
        n.buses.loc[n.lines.bus1][["x", "y"]],
    )
    n.lines.length *= line_length_factor

    n.links.length = pypsa.geo.haversine_pts(
        n.buses.loc[n.links.bus0][["x", "y"]],
        n.buses.loc[n.links.bus1][["x", "y"]],
    )
    n.links.length *= line_length_factor

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("simplify_network", interconnect="eastern")
    configure_logging(snakemake)
    params = snakemake.params

    voltage_level = snakemake.config["electricity"]["voltage_simplified"]
    aggregation_zones = snakemake.config["clustering"]["cluster_network"][
        "aggregation_zones"
    ]

    n = pypsa.Network(snakemake.input.network)

    n.generators.drop(
        columns=["ba_eia", "ba_ads"],
        inplace=True,
    )  # temp added these columns and need to drop for workflow

    n = convert_to_voltage_level(n, voltage_level)
    n, trafo_map = remove_transformers(n)

    substations = pd.read_csv(snakemake.input.sub, index_col=0)
    substations.index = substations.index.astype(str)

    # new busmap definition
    busmap_to_sub = n.buses.sub_id.astype(int).astype(str).to_frame()

    busmaps = [trafo_map, busmap_to_sub.sub_id]
    busmaps = reduce(lambda x, y: x.map(y), busmaps[1:], busmaps[0])

    # Haversine is a poor approximation... use 1.25 as an approximiation, but should be replaced with actual line lengths.
    # TODO: WHEN WE REPLACE NETWORK WITH NEW NETWORK WE SHOULD CALACULATE LINE LENGTHS BASED ON THE actual GIS line files.
    n = assign_line_lengths(n, 1.25)
    n.links["underwater_fraction"] = 0  # TODO: CALULATE UNDERWATER FRACTIONS.

    n = aggregate_to_substations(
        n,
        substations,
        busmap_to_sub.sub_id,
        aggregation_zones,
        params.aggregation_strategies,
    )

    n.export_to_netcdf(snakemake.output[0])

    output_path = os.path.dirname(snakemake.output[0]) + "_simplified_"
    export_network_for_gis_mapping(n, output_path)
