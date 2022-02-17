# Copyright 2021-2022 Martha Frysztacki (KIT)

import pypsa
import numpy as np
import logging
import pandas as pd

from pypsa.networkclustering import busmap_by_kmeans, get_clustering_from_busmap


def cluster_network(n, n_clusters, algorithm='kmeans'):

    if algorithm == 'kmeans':
        logger.info('creating clustered network using kmeans...')
        bus_weightings = pd.Series(index=n.buses.index, data=1) #no weighting atm
        busmap = busmap_by_kmeans(n, bus_weightings, n_clusters, buses_i=None)

    clustering = get_clustering_from_busmap(
        n, busmap, aggregate_generators_weighted=True,
        aggregate_one_ports=["Load", "StorageUnit"],
        line_length_factor=1.25,
        generator_strategies={'marginal_cost': np.mean, 'p_nom_min': np.sum}
    )

    return clustering.network, clustering.busmap
    
    

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    
    n = pypsa.Network(snakemake.input[0])

    n, busmap = cluster_network(n, int(snakemake.wildcards.clusters), 'kmeans')

    busmap.to_csv(snakemake.output['busmap'])
    n.export_to_netcdf(snakemake.output['network'])
