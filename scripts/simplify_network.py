# Copyright 2021-2022 Martha Frysztacki (KIT)

import pypsa
import logging
import pandas as pd
import numpy as np
from pypsa.networkclustering import busmap_by_kmeans, get_clustering_from_busmap

def aggregate_to_substations(network, substations, busmap):

    logger.info("Aggregating buses to substation level...")
    
    clustering = get_clustering_from_busmap(
        network, busmap,
        bus_strategies={'type':np.max},
    )

    network = clustering.network

    substations.index = substations.index.astype(str)
    
    network.buses['interconnect'] = substations.interconnect
    network.buses['x'] = substations.lon
    network.buses['y'] = substations.lat

    return network

def assign_line_lengths(n, line_length_factor):

    logger.info("Assigning line lengths using haversine function...")

    n.lines.length = pypsa.geo.haversine_pts(n.buses.loc[n.lines.bus0][['x','y']],
                                             n.buses.loc[n.lines.bus1][['x','y']])
    n.lines.length *= line_length_factor

    n.links.length = pypsa.geo.haversine_pts(n.buses.loc[n.links.bus0][['x','y']],
                                             n.buses.loc[n.links.bus1][['x','y']])
    n.links.length *= line_length_factor

    return n


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    
    n = pypsa.Network(snakemake.input.network)
    
    busmap_to_sub = pd.read_csv(snakemake.input.bus2sub, index_col=0)
    substations = pd.read_csv(snakemake.input.sub, index_col=0)

    busmap_to_sub = busmap_to_sub.sub_id.astype(str)
    busmap_to_sub.index = busmap_to_sub.index.astype(str)

    n = aggregate_to_substations(n, substations, busmap_to_sub)
    n = assign_line_lengths(n, 1.25)

    n.export_to_netcdf(snakemake.output[0])
