# BY PyPSA-USA Authors
"""Aggregates the bare topology network to substation level and normalizes to one voltage.

First stage of the topology-aggregation pipeline. Consumes ``elec_base_network.nc``
(buses, lines, transformers only — no generators, loads, or storage attached) and
reduces it to one bus per substation via:

1. Converting line parameters to a single voltage base (230 kV).
2. Removing transformers, collapsing voltage layers into a single bus per
   substation.
3. Aggregating buses by ``sub_id``.

The downstream ``cluster_simpl`` rule may then optionally apply k-means
clustering to ``{simpl}`` clusters before any per-bus heavy data (renewable
profiles, demand) is built.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging
from pypsa.clustering.spatial import get_clustering_from_busmap

logger = logging.getLogger(__name__)


def convert_to_per_unit(df):
    df["base_impedance"] = df["v_nom"] ** 2 / df["s_nom"]
    df["base_susceptance"] = 1 / df["base_impedance"]
    df["resistance_pu"] = df["r"] / df["base_impedance"]
    df["reactance_pu"] = df["x"] / df["base_impedance"]
    df["susceptance_pu"] = df["b"] / df["base_susceptance"]
    df = df.drop(["base_impedance", "base_susceptance"], axis=1)
    return df


def convert_to_voltage_level(n, new_voltage):
    """Convert network.lines parameters to a given voltage."""
    df = convert_to_per_unit(n.lines.copy())
    df["new_base_impedance"] = new_voltage**2 / df["s_nom"]
    df["r"] = df["resistance_pu"] * df["new_base_impedance"]
    df["x"] = df["reactance_pu"] * df["new_base_impedance"]
    df["b"] = df["susceptance_pu"] / df["new_base_impedance"]
    df.v_nom = new_voltage
    df = df.drop(
        ["new_base_impedance", "resistance_pu", "reactance_pu", "susceptance_pu"],
        axis=1,
    )
    df.type = "Al/St 240/40 2-bundle 220.0"
    n.buses["v_nom"] = new_voltage
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
    topological_boundaries: str,
    aggregation_strategies=dict(),
):
    logger.info("Aggregating buses to substation level...")

    generator_strategies = aggregation_strategies.get("generators", dict())

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
            "county",
            "balancing_area",
            "reeds_zone",
            "reeds_ba",
            "reeds_state",
            "x",
            "y",
        ]
    ]
    substations = substations.drop_duplicates(subset=["sub_id"])
    substations.sub_id = substations.sub_id.astype(int).astype(str)
    substations.index = substations.sub_id

    match topological_boundaries:
        case "county":
            zone = substations.county
        case "reeds_zone":
            zone = substations.reeds_zone
        case "state":
            zone = substations.reeds_state
        case _:
            raise ValueError(
                "zonal_aggregation must be either balancing_area, country, or state",
            )

    network_s = clustering.network

    network_s.buses["interconnect"] = substations.interconnect
    network_s.buses["x"] = substations.x
    network_s.buses["y"] = substations.y
    network_s.buses["substation_lv"] = True
    network_s.buses["country"] = zone  # `country` field drives pypsa aggregation grouping

    network_s.lines["type"] = np.nan

    if topological_boundaries == "reeds_zone" or topological_boundaries == "county":
        cols2drop = [
            "balancing_area",
            "substation_off",
            "sub_id",
            "state",
        ]
    elif topological_boundaries == "state":
        cols2drop = [
            "balancing_area",
            "substation_off",
            "sub_id",
            "county",
            "reeds_zone",
            "reeds_ba",
            "nerc_reg",
            "trans_reg",
            "trans_grp",
            "state",
        ]
    else:
        cols2drop = [
            "balancing_area",
            "state",
            "substation_off",
            "sub_id",
            "reeds_zone",
            "reeds_ba",
            "nerc_reg",
            "trans_reg",
            "trans_grp",
            "reeds_state",
        ]

    cols2drop = [col for col in cols2drop if col in network_s.buses.columns]
    network_s.buses = network_s.buses.drop(columns=cols2drop)
    return network_s, clustering.busmap


def assign_line_lengths(n, line_length_factor):
    """Assign line lengths to network using haversine."""
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

        snakemake = mock_snakemake(
            "aggregate_to_substations",
            interconnect="texas",
        )
    configure_logging(snakemake)
    params = snakemake.params

    topological_boundaries = snakemake.params.topological_boundaries

    n = pypsa.Network(snakemake.input.network)

    n = convert_to_voltage_level(n, 230)
    n, trafo_map = remove_transformers(n)

    substations = pd.read_csv(snakemake.input.sub, index_col=0)
    substations.index = substations.index.astype(str)

    busmap_to_sub = n.buses.sub_id.astype(int).astype(str).to_frame()

    n = assign_line_lengths(n, 1.25)
    n.links["underwater_fraction"] = 0
    n.buses.drop(columns=["substation_off"], inplace=True)

    n, busmap = aggregate_to_substations(
        n,
        substations,
        busmap_to_sub.sub_id,
        topological_boundaries,
        params.aggregation_strategies,
    )

    if topological_boundaries in ["reeds_zone", "state"] and "county" in n.buses.columns:
        n.buses = n.buses.drop(columns=["county"])

    n.export_to_netcdf(snakemake.output.network)
    busmap.to_csv(snakemake.output.busmap, header=["sub_id"])
