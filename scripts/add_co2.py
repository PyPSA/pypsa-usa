import logging
from _helpers import configure_logging

import pypsa
import numpy as np
import pandas as pd

from create_network import load_costs

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def add_co2limit(n, Nyears=1.):
    annual_emissions = snakemake.config['electricity']['co2limit']

    n.add("GlobalConstraint", "CO2Limit",
          carrier_attribute="co2_emissions", sense="<=",
          constant=annual_emissions)


def add_emission_prices(n, emission_prices=None, exclude_co2=False):
    if emission_prices is None:
        emission_prices = snakemake.config['costs']['emission_prices']
    if exclude_co2: emission_prices.pop('co2')
    ep = (pd.Series(emission_prices).rename(lambda x: x+'_emissions') *
          n.carriers.filter(like='_emissions')).sum(axis=1)
    gen_ep = n.generators.carrier.map(ep) / n.generators.efficiency
    n.generators['marginal_cost'] += gen_ep
    su_ep = n.storage_units.carrier.map(ep) / n.storage_units.efficiency_dispatch
    n.storage_units['marginal_cost'] += su_ep


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('prepare_network', network='elec', simpl='',
                                  clusters='208', ll='v1.25', opts='24H')
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input[0])
    Nyears = n.snapshot_weightings.sum() / 8784.

    add_co2limit(n, Nyears)
    add_emission_prices(n)
    npd = pd.date_range(freq='h', start="2016-01-01", end="2017-01-01", closed='left')

    # In[18]:

    solver_options = snakemake.config['solving']['solver'].copy()
    n.lopf(npd, pyomo=False, solver_name='gurobi', solver_options=solver_options, solver_logfile=None,
           formulation='kirchhoff', keep_files=True, extra_functionality=None, multi_investment_periods=False)

    n.export_to_netcdf(snakemake.output[0])