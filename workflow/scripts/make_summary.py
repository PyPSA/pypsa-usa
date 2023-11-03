"""Export summary statistics of the network."""

import sys
import os
import pypsa
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from _helpers import mock_snakemake

def get_stats(n):
    revenue = n.statistics.revenue()
    revenue.index = revenue.index.map(lambda x: '_'.join(x[:2]))
    revenue = revenue.reset_index()
    revenue['statistic'] = ['revenue' for i in range(len(revenue))]

    opex = n.statistics.opex()
    opex.index = opex.index.map(lambda x: '_'.join(x[:2]))
    opex = opex.reset_index()
    opex['statistic'] = ['opex' for i in range(len(opex))]

    capacity = n.statistics.expanded_capacity()
    capacity.index = capacity.index.map(lambda x: '_'.join(x[:2]))
    capacity = capacity.reset_index()
    capacity['statistic'] = ['ExpCapacity' for i in range(len(capacity))]

    df = pd.concat([revenue, opex, capacity])
    df.rename(columns={'index':'type',0:'value'},inplace=True)
    df['value'] = df['value'].astype(float)
    df[['statistic','type','value']].to_csv(snakemake.output[0], index=False)
    return df


if __name__ == "__main__":
    os.chdir(os.getcwd())
    logger = logging.getLogger(__name__)

    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake("make_summary")

    n = pypsa.Network(snakemake.input.network)
    n_clusters = snakemake.wildcards.clusters

    stats = get_stats(n)

