import logging
from _helpers import configure_logging

import pypsa
import numpy as np
import pandas as pd

from create_network import load_costs

from pypsa.linopf import network_lopf, ilopf, get_var, linexpr, define_constraints

idx = pd.IndexSlice

logger = logging.getLogger(__name__)

def average_every_nhours(n, offset):
    logger.info(f"Resampling the network to {offset}")
    m = n.copy(with_time=False)

    snapshot_weightings = n.snapshot_weightings.resample(offset).sum()
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name+"_t")
        for k, df in c.pnl.items():
            if not df.empty:
                pnl[k] = df.resample(offset).mean()

    return m

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


def prepare_network(n, solve_opts):
    if 'clip_p_max_pu' in solve_opts:
        for df in (n.generators_t.p_max_pu, n.storage_units_t.inflow):
            df.where(df>solve_opts['clip_p_max_pu'], other=0., inplace=True)

    if solve_opts.get('brownfield'):
        logger.info("setting p_nom_min to p_nom for extendable carriers.")
        ext_gens_i = n.generators[n.generators.p_nom_extendable].index
        n.generators.loc[ext_gens_i, 'p_nom_min'] = n.generators.loc[ext_gens_i, 'p_nom']
        n.generators.loc[ext_gens_i, 'p_nom_max'] = (n.generators.loc[ext_gens_i, ['p_nom_max', 'p_nom_min']]
                                                     .apply(lambda b: b['p_nom_min'] if b['p_nom_min'] > b['p_nom_max'] else b['p_nom_max'], axis=1))

    if solve_opts.get('load_shedding'):
        n.add("Carrier", "Load")
        n.madd("Generator", n.buses.index, " load",
               bus=n.buses.index,
               carrier='load',
               sign=1e-3, # Adjust sign to measure p and p_nom in kW instead of MW
               marginal_cost=1e2, # Eur/kWh
               # intersect between macroeconomic and surveybased
               # willingness to pay
               # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
               p_nom=1e9 # kW
               )

    if solve_opts.get('noisy_costs'):
        for t in n.iterate_components(n.one_port_components):
            #if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if 'marginal_cost' in t.df:
                t.df['marginal_cost'] += (1e-2 + 2e-3 *
                                          (np.random.random(len(t.df)) - 0.5))

        for t in n.iterate_components(['Line', 'Link']):
            t.df['capital_cost'] += (1e-1 +
                2e-2*(np.random.random(len(t.df)) - 0.5)) * t.df['length']

    if solve_opts.get('nhours'):
        nhours = solve_opts['nhours']
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760./nhours

    return n


def add_battery_constraints(n):
    nodes = n.buses.index[n.buses.carrier == "battery"]
    if nodes.empty or ('Link', 'p_nom') not in n.variables.index:
        return
    link_p_nom = get_var(n, "Link", "p_nom")
    lhs = linexpr((1,link_p_nom[nodes + " charger"]),
                  (-n.links.loc[nodes + " discharger", "efficiency"].values,
                   link_p_nom[nodes + " discharger"].values))
    define_constraints(n, lhs, "=", 0, 'Link', 'charger_ratio')


def extra_functionality(n, snapshots):
    opts = n.opts
    config = n.config
    add_battery_constraints(n)


def solve_network(n, config, solver_log=None, opts='', **kwargs):
    solver_options = config['solving']['solver'].copy()
    solver_name = solver_options.pop('name')
    cf_solving = config['solving']['options']
    track_iterations = cf_solving.get('track_iterations', False)
    min_iterations = cf_solving.get('min_iterations', 4)
    max_iterations = cf_solving.get('max_iterations', 6)

    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    if cf_solving.get('skip_iterations', False):
        network_lopf(n, solver_name=solver_name, solver_options=solver_options,
                     extra_functionality=extra_functionality, **kwargs)
    else:
        ilopf(n, solver_name=solver_name, solver_options=solver_options,
              track_iterations=track_iterations,
              min_iterations=min_iterations,
              max_iterations=max_iterations,
              extra_functionality=extra_functionality, **kwargs)
    return n


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('prepare_network', network='elec', simpl='',
                                  clusters='208', ll='v1.25', opts='24H')
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input[0])

    Nyears = n.snapshot_weightings.sum() / 8784.

    nH = snakemake.wildcards.nH
    average_every_nhours(n, nH)
    add_co2limit(n, Nyears)
    add_emission_prices(n)

    config = snakemake.config
    solve_opts = snakemake.config['solving']['options']

    opts = f'Co2L-{nH}'.split('-')

    tmpdir = solve_opts.get('tmpdir')
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    n = prepare_network(n, solve_opts=solve_opts)  #loadshedding is added here
    n = solve_network(n, config, solver_dir=tmpdir,
                      solver_log=snakemake.log.solver, opts=opts)

    n.export_to_netcdf(snakemake.output[0])
