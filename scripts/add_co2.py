import logging

import pypsa
import numpy as np
import pandas as pd

import sys
sys.path.append(snakemake.config['subworkflow'] + "scripts/")

from prepare_network import average_every_nhours, add_co2limit, add_emission_prices
from solve_network import prepare_network, solve_network

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('prepare_network', network='elec', simpl='',
                                  clusters='208', ll='v1.25', opts='24H')

    n = pypsa.Network(snakemake.input[0])

    Nyears = n.snapshot_weightings.objective.sum() / 8784.

    nH = snakemake.wildcards.nH
    average_every_nhours(n, nH)
    cf = float(snakemake.wildcards.emission)
    add_co2limit(n, snakemake.config['electricity']['co2limit']*cf, Nyears)

    config = snakemake.config
    solve_opts = snakemake.config['solving']['options']

    opts = f'Co2L-{nH}'.split('-')

    tmpdir = solve_opts.get('tmpdir')
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    n = prepare_network(n, solve_opts)  #loadshedding is added here
    n = solve_network(n, config, opts, solver_dir=tmpdir,
                      solver_logfile=snakemake.log.solver)

    n.export_to_netcdf(snakemake.output[0])
