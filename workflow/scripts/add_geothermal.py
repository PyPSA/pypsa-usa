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
import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging
from add_electricity import (
    _add_missing_carriers_from_costs,
    add_nice_carrier_names,
    calculate_annuity,
    load_costs,
)

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def clean_egs_sc(geo_egs_sc, capex_cap = 1e5): 
    geo_sc = gpd.read_file(geo_egs_sc).to_crs(4326).rename(columns={'name':'county'})
    
    geo_sc['capex_kw'] =(geo_sc['capex_kw']/5000).round()*5000
    geo_sc['fom_kw'] =(geo_sc['fom_kw']/500).round()*500
    geo_sc = geo_sc.loc[geo_sc['capex_kw'] <= capex_cap]
    
    return geo_sc

def join_pypsa_cluser(regions_gpd, geo_egs_sc): 
    geo_sc = clean_egs_sc(geo_egs_sc)

    region_onshore = gpd.read_file(regions_gpd).rename(columns={'name':'bus'})
    region_onshore_geo = gpd.sjoin(region_onshore, geo_sc, how="left").reset_index(drop=True)

    region_onshore_geo_grp = region_onshore_geo.groupby(['name', 'capex_kw', 'fom_kw'])['potential_mw'].agg('sum').reset_index()
    region_onshore_geo_grp['class'] = region_onshore_geo_grp.groupby(['name']).cumcount()+1
    region_onshore_geo_grp['class'] = "c" + region_onshore_geo_grp['class'].astype(str)
    region_onshore_geo_grp['tech'] = 'egs'
    region_onshore_geo_grp['carrier'] = region_onshore_geo_grp[['tech', 'class']].agg('_'.join, axis=1)

    region_onshore_geo_grp = region_onshore_geo_grp[['name', 'carrier', 'capex_kw', 'fom_kw', 'potential_mw']].rename(columns={'name':'bus', 'capex_kw':"capital_cost", 'fom_kw':"marginal_cost", 'potential_mw':"p_nom_max"}).sort_values(by=['bus', 'carrier'], ascending=True)
    region_onshore_geo_grp['capital_cost'] = region_onshore_geo_grp['capital_cost'] * 1e3
    region_onshore_geo_grp['marginal_cost'] = region_onshore_geo_grp['marginal_cost'] * 1e3
    region_onshore_geo_grp["fom_pct"] = region_onshore_geo_grp['marginal_cost'] / region_onshore_geo_grp['capital_cost']

    return region_onshore_geo_grp

def add_egs(n: pypsa.Network, regions_gpd, geo_sc, cost_reduction): 
    egs_class = join_pypsa_cluser(regions_gpd, geo_sc)

    egs_class["capital_cost"] = (
            (
                calculate_annuity(30, 0.055)
                + egs_class["fom_pct"]
            )
            * egs_class["capital_cost"]
            * n.snapshot_weightings.objective.sum() / 8760.0
        ) * (1-cost_reduction)

    egs_class['Generator'] = egs_class['bus'] + ' ' + egs_class['carrier']
    egs_class = egs_class.set_index('Generator')
    egs_class['p_nom_min'] = 0
    egs_class['p_nom'] = 0
    egs_class['efficiency'] = 0.9
    egs_class['weight'] = 1
    egs_class['control'] = 'PQ'
    egs_class['p_min_pu'] = 0
    egs_class['p_max_pu'] = 0.9
    egs_class['p_nom_opt'] = np.nan
  
    n.madd(
    "Generator",
    egs_class.index,
    suffix=" new",
    carrier= egs_class.carrier, 
    bus=egs_class.bus,
    p_nom_min=0,
    p_nom=0,
    p_nom_max=egs_class.p_nom_max,
    p_nom_extendable=True,
    ramp_limit_up=0.15,
    ramp_limit_down=0.15,
    efficiency=egs_class.efficiency,
    marginal_cost=0,
    capital_cost=egs_class.capital_cost,
    lifetime=30,
    p_min_pu = egs_class.p_min_pu,
    p_max_pu = egs_class.p_max_pu,
    )

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_extra_components",
            interconnect="texas",
            clusters=40,
        )
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)
    regions_gpd = snakemake.input.regions_onshore
    geo_egs_sc = snakemake.input.geo_egs_sc
    cost_reduction = snakemake.params.cost_reduction

    add_egs(n, regions_gpd, geo_egs_sc, cost_reduction)