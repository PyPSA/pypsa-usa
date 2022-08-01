# Copyright 2021-2022 Martha Frysztacki (KIT)

import pypsa
import logging
import pandas as pd
import numpy as np
from functools import reduce
from pypsa.networkclustering import busmap_by_kmeans, get_clustering_from_busmap


def simplify_network_to_voltage_level(n, voltage_level):
    logger.info("Mapping all network lines onto a single layer")

    n.buses["v_nom"] = voltage_level

    (linetype,) = n.lines.loc[n.lines.v_nom == voltage_level, "type"].unique()
    lines_v_nom_b = n.lines.v_nom != voltage_level
    n.lines.loc[lines_v_nom_b, "num_parallel"] *= (
        n.lines.loc[lines_v_nom_b, "v_nom"] / voltage_level
    ) ** 2
    n.lines.loc[lines_v_nom_b, "v_nom"] = voltage_level
    n.lines.loc[lines_v_nom_b, "type"] = linetype
    n.lines.loc[lines_v_nom_b, "s_nom"] = (
        np.sqrt(3)
        * n.lines["type"].map(n.line_types.i_nom)
        * n.lines.bus0.map(n.buses.v_nom)
        * n.lines.num_parallel
    )

    # Replace transformers by lines
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


def aggregate_to_substations(network, substations, busmap):

    logger.info("Aggregating buses to substation level...")

    clustering = get_clustering_from_busmap(
        network,
        busmap,
        aggregate_generators_weighted=True,
        aggregate_one_ports=["Load", "StorageUnit"],
        line_length_factor=1.0,
        bus_strategies={"type": np.max},
        generator_strategies={
            "marginal_cost": np.mean,
            "p_nom_min": np.sum,
            "p_min_pu": np.mean,
            "p_max_pu": np.mean,
            "ramp_limit_up": np.max,
            "ramp_limit_down": np.max,
        },
    )

    network = clustering.network

    network.buses["interconnect"] = substations.interconnect
    network.buses["x"] = substations.lon
    network.buses["y"] = substations.lat
    network.buses["substation_lv"] = True
    network.buses["substation_off"] = True
    network.buses["country"] = "US"

    network.lines["type"] = np.nan
    network.lines["underwaterfraction"] = 0

    return network


def assign_line_lengths(n, line_length_factor, busmap_to_sub=None, substations=None):

    if (busmap_to_sub is not None) and (substations is not None):
        busmap_to_sub["x"] = busmap_to_sub.sub_id.map(substations["lon"])
        busmap_to_sub["y"] = busmap_to_sub.sub_id.map(substations["lat"])
        n.buses[["x", "y"]] = busmap_to_sub[["x", "y"]]

    logger.info("Assigning line lengths using haversine function...")

    n.lines.length = pypsa.geo.haversine_pts(
        n.buses.loc[n.lines.bus0][["x", "y"]], n.buses.loc[n.lines.bus1][["x", "y"]]
    )
    n.lines.length *= line_length_factor

    n.links.length = pypsa.geo.haversine_pts(
        n.buses.loc[n.links.bus0][["x", "y"]], n.buses.loc[n.links.bus1][["x", "y"]]
    )
    n.links.length *= line_length_factor

    return n


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    voltage_level = snakemake.config["electricity"]["voltage_simplified"]

    n = pypsa.Network(snakemake.input.network)
    n, trafo_map = simplify_network_to_voltage_level(n, voltage_level)

    busmap_to_sub = pd.read_csv(
        snakemake.input.bus2sub, index_col=0, dtype={"sub_id": str}
    )
    busmap_to_sub.index = busmap_to_sub.index.astype(str)
    substations = pd.read_csv(snakemake.input.sub, index_col=0)
    substations.index = substations.index.astype(str)

    busmaps = [trafo_map, busmap_to_sub.sub_id]
    busmaps = reduce(lambda x, y: x.map(y), busmaps[1:], busmaps[0])

    # assign line lengths based on sub_id,
    # otherwise divide by zero error in networkclustering
    n = assign_line_lengths(n, 1.25, busmap_to_sub, substations)

    n = aggregate_to_substations(n, substations, busmap_to_sub.sub_id)

    n.export_to_netcdf(snakemake.output[0])
