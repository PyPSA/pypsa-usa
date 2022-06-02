# Copyright 2021-2022 Martha Frysztacki (KIT)

import pypsa
import numpy as np
import logging
import pandas as pd

from pypsa.networkclustering import busmap_by_kmeans, get_clustering_from_busmap

def add_custom_line_type(n):
    n.line_types.loc['Rail'] = pd.Series(
        [60, 0.0683, 0.335, 15, 1.01],
        index=['f_nom','r_per_length','x_per_length','c_per_length','i_nom']
    )


def cluster_network(n, n_clusters, algorithm='kmeans'):

    if algorithm == 'kmeans':
        logger.info('creating clustered network using kmeans...')
        bus_weightings = pd.Series(index=n.buses.index, data=1) #no weighting atm
        busmap = busmap_by_kmeans(n, bus_weightings, n_clusters, buses_i=None)

    clustering = get_clustering_from_busmap(
        n, busmap, aggregate_generators_weighted=True,
        aggregate_one_ports=["Load", "StorageUnit"],
        line_length_factor=1.25,
        generator_strategies={'marginal_cost': np.mean, 
                              'committable': np.any, 
                              'p_nom_min': np.sum, 
                              'p_min_pu': np.mean, 
                              'p_max_pu': np.mean, 
                              "ramp_limit_up": np.max, 
                              "ramp_limit_down": np.max
        }
    )

    return clustering.network, clustering.busmap


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    
    n = pypsa.Network(snakemake.input[0])

    n, busmap = cluster_network(n, int(snakemake.wildcards.clusters), 'kmeans')

    busmap.to_csv(snakemake.output['busmap'])

    #hotfixes for later scripts
    n.buses["country"] = "USA"
    add_custom_line_type(n)

    n.export_to_netcdf(snakemake.output['network'])
