# BY PyPSA-USA Authors
"""
Aggregates network to substations and simplifies to a single voltage level.
"""


import logging
from functools import reduce

import dill as pickle
import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging, update_p_nom_max
from cluster_network import cluster_regions, clustering_for_n_clusters
from pypsa.clustering.spatial import get_clustering_from_busmap

logger = logging.getLogger(__name__)


def convert_to_per_unit(df):
    # Calculating base values per component
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

    # Update network lines
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
    """
    Aggregate network to substations.

    First step in clusterings, if use_ba_zones is True, then the network
    retains balancing Authority zones in clustering.
    """

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
        case _:
            raise ValueError(
                "zonal_aggregation must be either balancing_area, country, or state",
            )

    network_s = clustering.network

    network_s.buses["interconnect"] = substations.interconnect
    network_s.buses["x"] = substations.x
    network_s.buses["y"] = substations.y
    network_s.buses["substation_lv"] = True
    network_s.buses["country"] = zone  # country field used bc pypsa algo aggregates based on country field

    network_s.lines["type"] = np.nan

    if topological_boundaries != "reeds_zone" and topological_boundaries != "county":
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
    else:
        cols2drop = ["balancing_area", "substation_off", "sub_id", "state"]

    network_s.buses.drop(
        columns=cols2drop,
        inplace=True,
    )
    return network_s, clustering.busmap


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

        snakemake = mock_snakemake(
            "simplify_network",
            interconnect="texas",
            simpl="50",
        )
    configure_logging(snakemake)
    params = snakemake.params
    solver_name = snakemake.config["solving"]["solver"]["name"]

    topological_boundaries = snakemake.params.topological_boundaries

    # n = pypsa.Network(snakemake.input.network)
    n = pickle.load(open(snakemake.input.network, "rb"))

    n.generators.drop(
        columns=["ba_eia", "ba_ads"],
        inplace=True,
    )  # temp added these columns and need to drop for workflow

    n = convert_to_voltage_level(n, 230)
    n, trafo_map = remove_transformers(n)

    substations = pd.read_csv(snakemake.input.sub, index_col=0)
    substations.index = substations.index.astype(str)

    # new busmap definition
    busmap_to_sub = n.buses.sub_id.astype(int).astype(str).to_frame()

    busmaps = [trafo_map, busmap_to_sub.sub_id]
    busmaps = reduce(lambda x, y: x.map(y), busmaps[1:], busmaps[0])

    # TODO: WHEN WE REPLACE NETWORK WITH NEW NETWORK WE SHOULD CALACULATE LINE LENGTHS BASED ON THE actual GIS line files.
    n = assign_line_lengths(n, 1.25)
    n.links["underwater_fraction"] = 0  # TODO: CALULATE UNDERWATER FRACTIONS.

    n, busmap = aggregate_to_substations(
        n,
        substations,
        busmap_to_sub.sub_id,
        topological_boundaries,
        params.aggregation_strategies,
    )

    if topological_boundaries == "reeds_zone":
        n.buses.drop(columns=["county"], inplace=True)

    if snakemake.wildcards.simpl:
        n.set_investment_periods(periods=snakemake.params.planning_horizons)

        n.loads_t.p = n.loads_t.p.iloc[:, 0:0]
        n.loads_t.q = n.loads_t.q.iloc[:, 0:0]
        attr = [
            "p",
            "q",
            "state_of_charge",
            "mu_state_of_charge_set",
            "mu_energy_balance",
            "mu_lower",
            "mu_upper",
            "spill",
            "p_dispatch",
            "p_store",
        ]
        for attr in attr:
            n.storage_units_t[attr] = n.storage_units_t[attr].iloc[:, 0:0]

        clustering = clustering_for_n_clusters(
            n,
            int(snakemake.wildcards.simpl),
            focus_weights=params.focus_weights,
            solver_name=solver_name,
            algorithm=params.simplify_network["algorithm"],
            feature=params.simplify_network["feature"],
            aggregation_strategies=params.aggregation_strategies,
        )
        n = clustering.network

        cluster_regions((clustering.busmap,), snakemake.input, snakemake.output)
    else:
        for which in ("regions_onshore", "regions_offshore"):  # pass through regions
            regions = gpd.read_file(getattr(snakemake.input, which))
            regions.to_file(getattr(snakemake.output, which))

    update_p_nom_max(n)

    n.export_to_netcdf(snakemake.output[0])
