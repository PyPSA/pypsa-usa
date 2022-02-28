import logging

import re
import pypsa
import numpy as np
import pandas as pd

import sys
sys.path.append(snakemake.config['subworkflow'] + "scripts/")

from prepare_network import (average_every_nhours, apply_time_segmentation, add_co2limit,
                             set_line_s_max_pu)
from solve_network import prepare_network, solve_network

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('prepare_network', network='elec', simpl='',
                                  clusters='208', ll='v1.25', opts='24H')

    opts = snakemake.wildcards.opts.split('-')

    n = pypsa.Network(snakemake.input[0])
    Nyears = n.snapshot_weightings.objective.sum() / 8784.

    set_line_s_max_pu(n, snakemake.config['lines']['s_max_pu'])

    for o in opts:
        m = re.match(r'^\d+h$', o, re.IGNORECASE)
        if m is not None:
            n = average_every_nhours(n, m.group(0))
            break

    for o in opts:
        m = re.match(r'^\d+seg$', o, re.IGNORECASE)
        if m is not None:
            solver_name = snakemake.config["solving"]["solver"]["name"]
            n = apply_time_segmentation(n, m.group(0)[:-3], solver_name)
            break

    for o in opts:
        if "Co2L" in o:
            m = re.findall("[0-9]*\.?[0-9]+$", o)
            if len(m) > 0:
                co2limit = float(m[0]) * snakemake.config['electricity']['co2base']
                add_co2limit(n, co2limit, Nyears)
            else:
                add_co2limit(n, snakemake.config['electricity']['co2limit'], Nyears)
            break

    # solving process starts here (put that into own script later):
    config = snakemake.config
    solve_opts = snakemake.config['solving']['options']

    tmpdir = solve_opts.get('tmpdir')
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    n = prepare_network(n, solve_opts)  #loadshedding is added here
    n = solve_network(n, config, opts, solver_dir=tmpdir,
                      solver_logfile=snakemake.log.solver)

    n.export_to_netcdf(snakemake.output[0])
