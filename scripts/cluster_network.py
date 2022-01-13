# Copyright 2021-2022 Martha Frysztacki (KIT)

import pypsa
import logging
import pandas as pd

from pypsa.networkclustering import busmap_by_kmeans, get_clustering_from_busmap


def cluster_network(n, n_clusters, algorithm='kmeans'):

    if algorithm == 'kmeans':
        logger.info('creating clustered network using kmeans...')
        bus_weightings = pd.Series(index=n.buses.index, data=1) #no weighting atm
        busmap = busmap_by_kmeans(n, bus_weightings, n_clusters, buses_i=None)

    clustering = get_clustering_from_busmap(n, busmap, with_time=True, line_length_factor=1.25,
                                            aggregate_generators_weighted=False, aggregate_one_ports={},
                                            aggregate_generators_carriers=None,
                                            scale_link_capital_costs=True,
                                            bus_strategies={},
                                            one_port_strategies=dict(),
                                            generator_strategies=dict())

    return clustering.network, clustering.busmap
    
    

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    
    n = pypsa.Network(snakemake.input[0])
    
    n, busmap = cluster_network(n, int(snakemake.wildcards.nclusters), 'kmeans')

    busmap.to_csv(snakemake.output['busmap'])
    n.export_to_netcdf(snakemake.output['network'])
