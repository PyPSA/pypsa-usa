# SPDX-FileCopyrightText: : 2017-2022 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# coding: utf-8
"""
Adds extra extendable components to the clustered and simplified network.

**Relevant Settings**

.. code:: yaml

    costs:
        year:
        version:
        dicountrate:
        emission_prices:

    electricity:
        max_hours:
        marginal_cost:
        capital_cost:
        extendable_carriers:
            StorageUnit:
            Store:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at :ref:`costs_cf`,
    :ref:`electricity_cf`

**Inputs**

- ``resources/costs.csv``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.

**Outputs**

- ``networks/elec_s{simpl}_{clusters}_ec.nc``:


**Description**

The rule :mod:`add_extra_components` attaches additional extendable components to the clustered and simplified network. These can be configured in the ``config.yaml`` at ``electricity: extendable_carriers:``. It processes ``networks/elec_s{simpl}_{clusters}.nc`` to build ``networks/elec_s{simpl}_{clusters}_ec.nc``, which in contrast to the former (depending on the configuration) contain with **zero** initial capacity

- ``StorageUnits`` of carrier 'H2' and/or 'battery'. If this option is chosen, every bus is given an extendable ``StorageUnit`` of the corresponding carrier. The energy and power capacities are linked through a parameter that specifies the energy capacity as maximum hours at full dispatch power and is configured in ``electricity: max_hours:``. This linkage leads to one investment variable per storage unit. The default ``max_hours`` lead to long-term hydrogen and short-term battery storage units.

- ``Stores`` of carrier 'H2' and/or 'battery' in combination with ``Links``. If this option is chosen, the script adds extra buses with corresponding carrier where energy ``Stores`` are attached and which are connected to the corresponding power buses via two links, one each for charging and discharging. This leads to three investment variables for the energy capacity, charging and discharging capacity of the storage unit.
"""
import logging
from _helpers import configure_logging

import pypsa
import pandas as pd
import numpy as np
from typing import List
import geopandas as gpd

from add_electricity import (load_costs, add_nice_carrier_names, calculate_annuity, 
                             _add_missing_carriers_from_costs)

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def attach_storageunits(n, costs, elec_opts):
    carriers = elec_opts['extendable_carriers']['StorageUnit']
    max_hours = elec_opts['max_hours']

    _add_missing_carriers_from_costs(n, costs, carriers)

    buses_i = n.buses.index

    lookup_store = {"H2": "electrolysis", "battery": "battery inverter"}
    lookup_dispatch = {"H2": "fuel cell", "battery": "battery inverter"}

    for carrier in carriers:
        roundtrip_correction = 0.5 if carrier == "battery" else 1
        
        n.madd("StorageUnit", buses_i, ' ' + carrier,
               bus=buses_i,
               carrier=carrier,
               p_nom_extendable=True,
               capital_cost=costs.at[carrier, 'capital_cost'],
               marginal_cost=costs.at[carrier, 'marginal_cost'],
               efficiency_store=costs.at[lookup_store[carrier], 'efficiency']**roundtrip_correction,
               efficiency_dispatch=costs.at[lookup_dispatch[carrier], 'efficiency']**roundtrip_correction,
               max_hours=max_hours[carrier],
               cyclic_state_of_charge=True
        )


def attach_stores(n, costs, elec_opts):
    carriers = elec_opts['extendable_carriers']['Store']

    _add_missing_carriers_from_costs(n, costs, carriers)

    buses_i = n.buses.index
    bus_sub_dict = {k: n.buses[k].values for k in ['x', 'y', 'country']}

    if 'H2' in carriers:
        h2_buses_i = n.madd("Bus", buses_i + " H2", carrier="H2", **bus_sub_dict)

        n.madd("Store", h2_buses_i,
               bus=h2_buses_i,
               carrier='H2',
               e_nom_extendable=True,
               e_cyclic=True,
               capital_cost=costs.at["hydrogen storage underground", "capital_cost"])

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
               # the efficiencies are "round trip efficiencies"
               efficiency=costs.at['battery inverter', 'efficiency']**0.5,
               capital_cost=costs.at['battery inverter', 'capital_cost'],
               p_nom_extendable=True,
               marginal_cost=costs.at["battery inverter", "marginal_cost"])

        n.madd("Link", b_buses_i + " discharger",
               bus0=b_buses_i,
               bus1=buses_i,
               carrier='battery discharger',
               efficiency=costs.at['battery inverter','efficiency']**0.5,
               p_nom_extendable=True,
               marginal_cost=costs.at["battery inverter", "marginal_cost"])


def attach_hydrogen_pipelines(n, costs, elec_opts):
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

def add_economic_retirement(n: pypsa.Network, costs: pd.DataFrame, gens: List[str] = None): 
    """Adds dummy generators to account for economic retirement 
    
    Specifically this function does the following: 
    1. Creates duplicate generators for any that are tagged as extendable. For
    example, an extendable "CCGT" generator will be split into "CCGT" and "CCGT new" 
    2. Capital costs of existing extendable generators are replaced with fixed costs 
    3. p_nom_max of existing extendable generators are set to p_nom
    4. p_nom_min of existing and new generators is set to zero 
    
    Arguments:
    n: pypsa.Network, 
    costs: pd.DataFrame, 
    gens: List[str]
        List of generators to apply economic retirment to. If none provided, it is 
        applied to all extendable generators
    """
    
    # only assign dummy generators to extendable generators
    extend = n.generators[n.generators["p_nom_extendable"] == True]
    if gens: 
        extend = extend[extend["carrier"].isin(gens)]
    if extend.empty:
        return 
    
    # divide by 100 b/c FOM expressed as percentage of CAPEX
    n.generators["capital_cost"] = n.generators.apply(
        lambda row: row["capital_cost"] if not row.name in (extend.index) else row["capital_cost"] * costs.at[row["carrier"], "FOM"] / 100, axis=1
    )

    n.generators["p_nom_max"] = np.where(
        n.generators["p_nom_extendable"],
        n.generators["p_nom"],
        n.generators["p_nom_max"]
    )

    n.generators["p_nom_min"] = np.where(
        n.generators["p_nom_extendable"],
        0,
        n.generators["p_nom_min"]
    )
    
    n.madd(
        "Generator",
        extend.index,
        suffix=" new",
        carrier=extend.carrier,
        bus=extend.bus,
        p_nom_min=0,
        p_nom=0,
        p_nom_max=extend.p_nom_max,
        p_nom_extendable=True,
        ramp_limit_up=extend.ramp_limit_up,
        ramp_limit_down=extend.ramp_limit_down,
        efficiency=extend.efficiency,
        marginal_cost=extend.marginal_cost,
        capital_cost=extend.capital_cost,
        lifetime=extend.lifetime,
        p_min_pu = extend.p_min_pu,
        p_max_pu = extend.p_max_pu,
    )
    
    # time dependent factors added after as not all generators are time dependent 
    marginal_cost_t = n.generators_t["marginal_cost"][[x for x in extend.index if x in n.generators_t.marginal_cost.columns]]
    marginal_cost_t = marginal_cost_t.rename(columns={x:f"{x} new" for x in marginal_cost_t.columns})
    n.generators_t["marginal_cost"] = n.generators_t["marginal_cost"].join(marginal_cost_t)
    
    p_max_pu_t = n.generators_t["p_max_pu"][[x for x in extend.index if x in n.generators_t["p_max_pu"].columns]]
    p_max_pu_t = p_max_pu_t.rename(columns={x:f"{x} new" for x in p_max_pu_t.columns})
    n.generators_t["p_max_pu"] = n.generators_t["p_max_pu"].join(p_max_pu_t)
    
# def add_geothermal(n: pypsa.Network, gens: List[str] = None): 
    
#     n_prep_gen = n.generators.sort_values(by=['carrier', 'bus'], ascending=True)
#     lst_bus = n_prep_gen['bus'].unique()

#     n_prep_template = n_prep_gen.head(1)
#     n_prep_template = n_prep_template.drop(columns=['carrier', 'bus'])
#     n_prep_template['key'] = '1'

#     n_prep_geo = pd.DataFrame({'bus': lst_bus})
#     n_prep_geo['carrier'] = 'geothermal'
#     n_prep_geo['key'] = '1'


#     n_prep_geo = n_prep_geo.merge(n_prep_template, left_on='key', right_on='key')

#     n_prep_geo['p_nom_min'] = 0
#     n_prep_geo['p_nom'] = 0
#     n_prep_geo['efficiency'] = 0.9
#     n_prep_geo['marginal_cost'] = 443000
#     n_prep_geo['capital_cost'] = 14100000
#     n_prep_geo['p_nom_max'] = 50000
#     n_prep_geo['weight'] = 1
#     n_prep_geo['control'] = 'PQ'
#     n_prep_geo['p_max_pu'] = 1
#     n_prep_geo['p_nom_opt'] = np.nan

#     # n_prep_geo['index'] = n_prep_geo['bus'] + n_prep_geo['new'] + 
#     n_prep_geo['Generator'] = n_prep_geo['bus'] + ' ' + n_prep_geo['carrier']
#     n_prep_geo = n_prep_geo.set_index('Generator')

#     n.madd(
#         "Generator",
#         n_prep_geo.index,
#         suffix=" new",
#         carrier= n_prep_geo.carrier, 
#         bus=n_prep_geo.bus,
#         p_nom_min=0,
#         p_nom=0,
#         p_nom_max=n_prep_geo.p_nom_max,
#         p_nom_extendable=True,
#         ramp_limit_up=n_prep_geo.ramp_limit_up,
#         ramp_limit_down=n_prep_geo.ramp_limit_down,
#         efficiency=n_prep_geo.efficiency,
#         marginal_cost=n_prep_geo.marginal_cost,
#         capital_cost=n_prep_geo.capital_cost,
#         lifetime=n_prep_geo.lifetime,
#         p_min_pu = n_prep_geo.p_min_pu,
#         p_max_pu = n_prep_geo.p_max_pu,
#     )

def add_egs(n: pypsa.Network, regions_gdp, geo_egs_sc, cost_reduction): 
    
    region_onshore = gpd.read_file(regions_gdp).rename(columns={'name':'bus'})
    geo_sc = gpd.read_file(geo_egs_sc).to_crs(4326).rename(columns={'name':'county'})

    geo_sc["fom_pct"] = geo_sc['fom_kw'] / geo_sc['capex_kw']


    geo_sc["capital_cost"] = (
            (
                calculate_annuity(30, 0.055)
                + geo_sc["fom_pct"]
            )
            * geo_sc["capex_kw"] * 1e3
            * n.snapshot_weightings.objective.sum() / 8760.0
        )

    geo_sc["investment_annualized"] = (
        calculate_annuity(30, 0.055)
        * geo_sc["capex_kw"] * 1e3
        * n.snapshot_weightings.objective.sum() / 8760.0
    )




    region_onshore_geo = gpd.sjoin(region_onshore, geo_sc, how="left").reset_index(drop=True)
    region_onshore_geo_grp = region_onshore_geo.sort_values(by=['bus', 'capex_kw'], ascending=True).groupby(["bus"])

    region_onshore_geo_grp_c1 = region_onshore_geo_grp.apply(lambda t: t.iloc[0]).reset_index(drop=True)
    # region_onshore_geo_grp_c2 = region_onshore_geo_grp.apply(lambda t: t.iloc[1]).reset_index(drop=True)
    # region_onshore_geo_grp_c3 = region_onshore_geo_grp.apply(lambda t: t.iloc[2]).reset_index(drop=True)
    # region_onshore_geo_grp_c4 = region_onshore_geo_grp.apply(lambda t: t.iloc[3]).reset_index(drop=True)
    # region_onshore_geo_grp_c5 = region_onshore_geo_grp.apply(lambda t: t.iloc[4]).reset_index(drop=True)
    # region_onshore_geo_grp_c6 = region_onshore_geo_grp.apply(lambda t: t.iloc[5]).reset_index(drop=True)

    # region_onshore_geo_grp_c1['tech'], region_onshore_geo_grp_c1['class'] = ['geothermal', '1']
    # region_onshore_geo_grp_c2['tech'], region_onshore_geo_grp_c2['class'] = ['geothermal', '2']
    # region_onshore_geo_grp_c3['tech'], region_onshore_geo_grp_c3['class'] = ['geothermal', '3']
    # region_onshore_geo_grp_c4['tech'], region_onshore_geo_grp_c4['class'] = ['geothermal', '4']
    # region_onshore_geo_grp_c5['tech'], region_onshore_geo_grp_c5['class'] = ['geothermal', '5']
    # region_onshore_geo_grp_c6['tech'], region_onshore_geo_grp_c6['class'] = ['geothermal', '6']

    region_onshore_geo_grp_c1['carrier'] = 'geothermal'
    # region_onshore_geo_grp_c2['carrier'] = region_onshore_geo_grp_c2[['tech', 'class']].agg('_'.join, axis=1)
    # region_onshore_geo_grp_c3['carrier'] = region_onshore_geo_grp_c3[['tech', 'class']].agg('_'.join, axis=1)
    # region_onshore_geo_grp_c4['carrier'] = region_onshore_geo_grp_c4[['tech', 'class']].agg('_'.join, axis=1)
    # region_onshore_geo_grp_c5['carrier'] = region_onshore_geo_grp_c5[['tech', 'class']].agg('_'.join, axis=1)
    # region_onshore_geo_grp_c6['carrier'] = region_onshore_geo_grp_c6[['tech', 'class']].agg('_'.join, axis=1)

    # region_onshore_geo_grp_all_class = pd.concat([region_onshore_geo_grp_c1,
    #         region_onshore_geo_grp_c2,
    #         region_onshore_geo_grp_c3,
    #         region_onshore_geo_grp_c4,
    #         region_onshore_geo_grp_c5,
    #         region_onshore_geo_grp_c6])
    
    region_onshore_geo_grp_all_class = region_onshore_geo_grp_c1[['bus', 'carrier', 'capital_cost', 'capex_kw', 'fom_kw', 'potential_mw']].rename(columns={'potential_mw':"p_nom_max"}).sort_values(by=['bus', 'carrier'], ascending=True)
    region_onshore_geo_grp_all_class['capital_cost'] = region_onshore_geo_grp_all_class['capital_cost'] * (1-cost_reduction)
    region_onshore_geo_grp_all_class['marginal_cost'] = 0
    region_onshore_geo_grp_all_class['Generator'] = region_onshore_geo_grp_all_class['bus'] + ' ' + region_onshore_geo_grp_all_class['carrier']
    region_onshore_geo_grp_all_class = region_onshore_geo_grp_all_class.set_index('Generator')

    region_onshore_geo_grp_all_class['p_nom_min'] = 0
    region_onshore_geo_grp_all_class['p_nom'] = 0
    region_onshore_geo_grp_all_class['efficiency'] = 1
    region_onshore_geo_grp_all_class['weight'] = 1
    region_onshore_geo_grp_all_class['control'] = 'PQ'
    region_onshore_geo_grp_all_class['p_min_pu'] = 0
    region_onshore_geo_grp_all_class['p_max_pu'] = 0.9
    region_onshore_geo_grp_all_class['p_nom_opt'] = np.nan

    n.madd(
        "Generator",
        region_onshore_geo_grp_all_class.index,
        suffix=" new",
        carrier= region_onshore_geo_grp_all_class.carrier, 
        bus=region_onshore_geo_grp_all_class.bus,
        p_nom_min=0,
        p_nom=0,
        p_nom_max=region_onshore_geo_grp_all_class.p_nom_max,
        p_nom_extendable=True,
        ramp_limit_up=0.15,
        ramp_limit_down=0.15,
        efficiency=region_onshore_geo_grp_all_class.efficiency,
        marginal_cost=region_onshore_geo_grp_all_class.marginal_cost,
        capital_cost=region_onshore_geo_grp_all_class.capital_cost,
        lifetime=30,
        p_min_pu = region_onshore_geo_grp_all_class.p_min_pu,
        p_max_pu = region_onshore_geo_grp_all_class.p_max_pu,
    )



if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake("add_extra_components", interconnect="western", clusters=30)
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)
    elec_config = snakemake.config['electricity']

    Nyears = n.snapshot_weightings.objective.sum() / 8760.
    costs = load_costs(snakemake.input.tech_costs, snakemake.config['costs'], elec_config['max_hours'], Nyears)
    
    n.buses['location'] = n.buses.index 

    attach_storageunits(n, costs, elec_config)
    attach_stores(n, costs, elec_config)
    attach_hydrogen_pipelines(n, costs, elec_config)

    add_nice_carrier_names(n, snakemake.config)

    if snakemake.params.retirement == "economic":
        economic_retirement_gens = elec_config['extendable_carriers']['Generator']
        # economic_retirement_gens = elec_config.get("conventional_carriers", None)
        add_economic_retirement(n, costs, economic_retirement_gens)
        if snakemake.params.egs:
            regions_gdp = snakemake.input.regions
            geo_egs_sc = snakemake.input.geo_egs_sc
            cost_reduction = snakemake.params.egs_reduction
            add_egs(n, regions_gdp, geo_egs_sc, cost_reduction)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
