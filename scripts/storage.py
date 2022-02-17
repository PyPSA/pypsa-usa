import logging
from _helpers import configure_logging, update_p_nom_max

import pypsa
import pandas as pd
import numpy as np

from create_network import (load_costs)

idx = pd.IndexSlice

logger = logging.getLogger(__name__)

def _add_missing_carriers_from_costs(n, costs, carriers):
    missing_carriers = pd.Index(carriers).difference(n.carriers.index)
    if missing_carriers.empty: return

    emissions_cols = costs.columns.to_series()\
                           .loc[lambda s: s.str.endswith('_emissions')].values
    suptechs = missing_carriers.str.split('-').str[0]
    emissions = costs.loc[suptechs, emissions_cols].fillna(0.)
    emissions.index = missing_carriers
    n.import_components_from_dataframe(emissions, 'Carrier')

def attach_storageunits(n, costs):
    elec_opts = snakemake.config['electricity']
    carriers = elec_opts['extendable_carriers']['StorageUnit']
    max_hours = elec_opts['max_hours']

    _add_missing_carriers_from_costs(n, costs, carriers)

    buses_i = n.buses.index

    lookup_store = {"H2": "electrolysis", "battery": "battery inverter"}
    lookup_dispatch = {"H2": "fuel cell", "battery": "battery inverter"}

    for carrier in carriers:
        n.madd("StorageUnit", buses_i, ' ' + carrier,
               bus=buses_i,
               carrier=carrier,
               p_nom_extendable=True,
               capital_cost=costs.at[carrier, 'capital_cost'],
               marginal_cost=costs.at[carrier, 'marginal_cost'],
               efficiency_store=costs.at[lookup_store[carrier], 'efficiency'],
               efficiency_dispatch=costs.at[lookup_dispatch[carrier], 'efficiency'],
               max_hours=max_hours[carrier],
               cyclic_state_of_charge=True)



def attach_stores(n, costs):
    elec_opts = snakemake.config['electricity']
    carriers = elec_opts['extendable_carriers']['Store']

    _add_missing_carriers_from_costs(n, costs, carriers)

    buses_i = n.buses.index
    bus_sub_dict = {k: n.buses[k].values for k in ['x', 'y']}

    if 'H2' in carriers:
        h2_buses_i = n.madd("Bus", buses_i + " H2", carrier="H2", **bus_sub_dict)

        n.madd("Store", h2_buses_i,
               bus=h2_buses_i,
               carrier='H2',
               e_nom_extendable=True,
               e_cyclic=True,
               capital_cost=costs.at["hydrogen storage", "capital_cost"])

        n.madd("Link", h2_buses_i + " Electrolysis",
               bus0=buses_i,
               bus1=h2_buses_i,
               carrier='H2 electrolysis',
               p_nom_extendable=True,
               efficiency=costs.at["electrolysis", "efficiency"],
               capital_cost=costs.at["electrolysis", "capital_cost"],
               marginal_cost=costs.at["electrolysis", "marginal_cost"])

        n.madd("Link", h2_buses_i + " Fuel Cell",
               bus0=h2_buses_i,
               bus1=buses_i,
               carrier='H2 fuel cell',
               p_nom_extendable=True,
               efficiency=costs.at["fuel cell", "efficiency"],
               #NB: fixed cost is per MWel
               capital_cost=costs.at["fuel cell", "capital_cost"] * costs.at["fuel cell", "efficiency"],
               marginal_cost=costs.at["fuel cell", "marginal_cost"])

    if 'battery' in carriers:
        b_buses_i = n.madd("Bus", buses_i + " battery", carrier="battery", **bus_sub_dict)

        n.madd("Store", b_buses_i,
               bus=b_buses_i,
               carrier='battery',
               e_cyclic=True,
               e_nom_extendable=True,
               capital_cost=costs.at['battery storage', 'capital_cost'],
               marginal_cost=costs.at["battery", "marginal_cost"])

        n.madd("Link", b_buses_i + " charger",
               bus0=buses_i,
               bus1=b_buses_i,
               carrier='battery charger',
               efficiency=costs.at['battery inverter', 'efficiency'],
               capital_cost=costs.at['battery inverter', 'capital_cost'],
               p_nom_extendable=True,
               marginal_cost=costs.at["battery inverter", "marginal_cost"])

        n.madd("Link", b_buses_i + " discharger",
               bus0=b_buses_i,
               bus1=buses_i,
               carrier='battery discharger',
               efficiency=costs.at['battery inverter','efficiency'],
               p_nom_extendable=True,
               marginal_cost=costs.at["battery inverter", "marginal_cost"])

def attach_hydrogen_pipelines(n, costs):
    elec_opts = snakemake.config['electricity']
    ext_carriers = elec_opts['extendable_carriers']
    as_stores = ext_carriers.get('Store', [])

    if 'H2 pipeline' not in ext_carriers.get('Link',[]): return

    assert 'H2' in as_stores, ("Attaching hydrogen pipelines requires hydrogen "
            "storage to be modelled as Store-Link-Bus combination. See "
            "`config.yaml` at `electricity: extendable_carriers: Store:`.")

    # determine bus pairs
    attrs = ["bus0","bus1","length"]
    candidates = pd.concat([n.lines[attrs], n.links.query('carrier=="DC"')[attrs]])\
                    .reset_index(drop=True)

    # remove bus pair duplicates regardless of order of bus0 and bus1
    h2_links = candidates[~pd.DataFrame(np.sort(candidates[['bus0', 'bus1']])).duplicated()]
    h2_links.index = h2_links.apply(lambda c: f"H2 pipeline {c.bus0}-{c.bus1}", axis=1)

    # add pipelines
    n.madd("Link",
           h2_links.index,
           bus0=h2_links.bus0.values + " H2",
           bus1=h2_links.bus1.values + " H2",
           p_min_pu=-1,
           p_nom_extendable=True,
           length=h2_links.length.values,
           capital_cost=costs.at['H2 pipeline','capital_cost']*h2_links.length,
           efficiency=costs.at['H2 pipeline','efficiency'],
           carrier="H2 pipeline")

def add_nice_carrier_names(n, config=None):
    if config is None: config = snakemake.config
    carrier_i = n.carriers.index
    nice_names = (pd.Series(config['plotting']['nice_names'])
                  .reindex(carrier_i).fillna(carrier_i.to_series().str.title()))
    n.carriers['nice_name'] = nice_names
    colors = pd.Series(config['plotting']['tech_colors']).reindex(carrier_i)
    if colors.isna().any():
        missing_i = list(colors.index[colors.isna()])
        logger.warning(f'tech_colors for carriers {missing_i} not defined '
                       'in config.')
    n.carriers['color'] = colors

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('add_extra_components', network='elec',
                                  simpl='', clusters=5)
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)
    Nyears = n.snapshot_weightings.sum() / 8784

    costs = load_costs(Nyears, tech_costs=snakemake.input.tech_costs,
                       config=snakemake.config['costs'],
                       elec_config=snakemake.config['electricity'])

    attach_storageunits(n, costs)
    attach_stores(n, costs)
    attach_hydrogen_pipelines(n, costs)

    # add_nice_carrier_names(n, config=snakemake.config)

    n.export_to_netcdf(snakemake.output[0])