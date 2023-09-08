# Copyright 2021-2022 Martha Frysztacki (KIT)

import pypsa
import pandas as pd
import numpy as np
from functools import reduce
from pypsa.networkclustering import busmap_by_kmeans, get_clustering_from_busmap
from _helpers import export_network_for_gis_mapping, configure_logging
import logging
import os

logger = logging.getLogger(__name__)

# logger.debug('submodule message')


def simplify_network_to_voltage_level(n, voltage_level):
    '''
    Simplify network to a single voltage level. Network Line Characteristics (s_nom, num_parallel, type) are mapped to the new voltage level.
    '''
    logger.info("Mapping all network lines onto a single layer")

    n.buses["v_nom"] = voltage_level
    # import pdb; pdb.set_trace()
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


def aggregate_to_substations(network, substations, busmap, aggregation_zones=False):
    '''
    Aggregate network to substations. First step in clusterings, if use_ba_zones is True, then the network retains balancing Authority zones in clustering.'''

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
    # sub_index = network.buses.country.index.map(busmap.to_dict())
    # countries = network.buses.country.values
    # countries_dict = dict(zip(sub_index, countries))
    # substations['ba'] = substations.index.map(countries_dict)
    substations = network.buses[['sub_id', 'interconnect', 'state',
                                 'country','balancing_area','x','y']]
    substations = substations.drop_duplicates(subset=['sub_id'])
    substations.sub_id  = substations.sub_id.astype(int).astype(str)
    substations.index = substations.sub_id

    if aggregation_zones == 'balancing_area': 
        zone = substations.balancing_area 
    elif aggregation_zones == 'country':
        zone = substations.country
    elif aggregation_zones == 'state':
        zone = substations.state
    else:
        ValueError('zonal_aggregation must be either balancing_area, country or state')

    network_s = clustering.network

    network_s.buses["interconnect"] = substations.interconnect
    network_s.buses["x"] = substations.x
    network_s.buses["y"] = substations.y
    network_s.buses["substation_lv"] = True
    network_s.buses["substation_off"] = True
    network_s.buses["country"] = zone
    network_s.buses["state"] = substations.state
    network_s.buses["balancing_area"] = substations.balancing_area
    network_s.lines["type"] = np.nan
    return network_s


def assign_line_lengths(n, line_length_factor):
    '''
    Assign line lengths to network. Uses haversine function to calculate line lengths.'''
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
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake("simplify_network", interconnect='western')
    configure_logging(snakemake)

    voltage_level = snakemake.config["electricity"]["voltage_simplified"]
    aggregation_zones = snakemake.config['clustering']['cluster_network']['aggregation_zones']

    n = pypsa.Network(snakemake.input.network)
    n, trafo_map = simplify_network_to_voltage_level(n, voltage_level)

    substations = pd.read_csv(snakemake.input.sub, index_col=0)
    substations.index = substations.index.astype(str)

    #new busmap definition
    busmap_to_sub = n.buses.sub_id.astype(int).astype(str).to_frame()


    busmaps = [trafo_map, busmap_to_sub.sub_id]
    busmaps = reduce(lambda x, y: x.map(y), busmaps[1:], busmaps[0])

    # Haversine is a poor approximation... use 1.25 as an approximiation, but should be replaced with actual line lengths.
    # TODO: WHEN WE REPLACE NETWORK WITH NEW NETWORK WE SHOULD CALACULATE LINE LENGTHS BASED ON THE actual GIS line files.
    n = assign_line_lengths(n, 1.25) 
    n.links["underwater_fraction"] = 0 #TODO: CALULATE UNDERWATER FRACTIONS.


    n = aggregate_to_substations(n, substations, busmap_to_sub.sub_id, aggregation_zones)

    import pdb; pdb.set_trace()

    n.export_to_netcdf(snakemake.output[0])

    output_path = os.path.dirname(snakemake.output[0]) + 'simplified_'
    export_network_for_gis_mapping(n, output_path)
