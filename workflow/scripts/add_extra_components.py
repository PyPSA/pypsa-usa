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
from typing import List

import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging
from add_electricity import (
    _add_missing_carriers_from_costs,
    add_nice_carrier_names,
    load_costs,
)

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def attach_storageunits(n, costs, elec_opts):
    carriers = elec_opts["extendable_carriers"]["StorageUnit"]

    _add_missing_carriers_from_costs(n, costs, carriers)

    buses_i = n.buses.index

    lookup_store = {"H2": "electrolysis", "battery": "battery inverter"}
    lookup_dispatch = {"H2": "fuel cell", "battery": "battery inverter"}

    for carrier in carriers:
        max_hours = int(carrier.split("hr_")[0])
        roundtrip_correction = 0.5 if "battery" in carrier else 1

        n.madd(
            "StorageUnit",
            buses_i,
            " " + carrier,
            bus=buses_i,
            carrier=carrier,
            p_nom_extendable=True,
            capital_cost=costs.at[carrier, "capital_cost"],
            marginal_cost=costs.at[carrier, "marginal_cost"],
            efficiency_store=costs.at[carrier, "efficiency"] ** roundtrip_correction,
            efficiency_dispatch=costs.at[carrier, "efficiency"] ** roundtrip_correction,
            max_hours=max_hours,
            cyclic_state_of_charge=True,
        )


def attach_stores(n, costs, elec_opts):
    carriers = elec_opts["extendable_carriers"]["Store"]

    _add_missing_carriers_from_costs(n, costs, carriers)

    buses_i = n.buses.index
    bus_sub_dict = {k: n.buses[k].values for k in ["x", "y", "country"]}

    if "H2" in carriers:
        h2_buses_i = n.madd("Bus", buses_i + " H2", carrier="H2", **bus_sub_dict)

        n.madd(
            "Store",
            h2_buses_i,
            bus=h2_buses_i,
            carrier="H2",
            e_nom_extendable=True,
            e_cyclic=True,
            capital_cost=costs.at["hydrogen storage underground", "capital_cost"],
        )

        n.madd(
            "Link",
            h2_buses_i + " Electrolysis",
            bus0=buses_i,
            bus1=h2_buses_i,
            carrier="H2 electrolysis",
            p_nom_extendable=True,
            efficiency=costs.at["electrolysis", "efficiency"],
            capital_cost=costs.at["electrolysis", "capital_cost"],
            marginal_cost=costs.at["electrolysis", "marginal_cost"],
        )

        n.madd(
            "Link",
            h2_buses_i + " Fuel Cell",
            bus0=h2_buses_i,
            bus1=buses_i,
            carrier="H2 fuel cell",
            p_nom_extendable=True,
            efficiency=costs.at["fuel cell", "efficiency"],
            # NB: fixed cost is per MWel
            capital_cost=costs.at["fuel cell", "capital_cost"]
            * costs.at["fuel cell", "efficiency"],
            marginal_cost=costs.at["fuel cell", "marginal_cost"],
        )


def attach_hydrogen_pipelines(n, costs, elec_opts):
    ext_carriers = elec_opts["extendable_carriers"]
    as_stores = ext_carriers.get("Store", [])

    if "H2 pipeline" not in ext_carriers.get("Link", []):
        return

    assert "H2" in as_stores, (
        "Attaching hydrogen pipelines requires hydrogen "
        "storage to be modelled as Store-Link-Bus combination. See "
        "`config.yaml` at `electricity: extendable_carriers: Store:`."
    )

    # determine bus pairs
    attrs = ["bus0", "bus1", "length"]
    candidates = pd.concat(
        [n.lines[attrs], n.links.query('carrier=="DC"')[attrs]],
    ).reset_index(drop=True)

    # remove bus pair duplicates regardless of order of bus0 and bus1
    h2_links = candidates[
        ~pd.DataFrame(np.sort(candidates[["bus0", "bus1"]])).duplicated()
    ]
    h2_links.index = h2_links.apply(lambda c: f"H2 pipeline {c.bus0}-{c.bus1}", axis=1)

    # add pipelines
    n.madd(
        "Link",
        h2_links.index,
        bus0=h2_links.bus0.values + " H2",
        bus1=h2_links.bus1.values + " H2",
        p_min_pu=-1,
        p_nom_extendable=True,
        length=h2_links.length.values,
        capital_cost=costs.at["H2 pipeline", "capital_cost"] * h2_links.length,
        efficiency=costs.at["H2 pipeline", "efficiency"],
        carrier="H2 pipeline",
    )


def add_economic_retirement(
    n: pypsa.Network,
    costs: pd.DataFrame,
    gens: list[str] = None,
):
    """
    Adds dummy generators to account for economic retirement.

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
        lambda row: (
            row["capital_cost"]
            if not row.name in (extend.index)
            else row["capital_cost"] * costs.at[row["carrier"], "FOM"] / 100
        ),
        axis=1,
    )

    n.generators["p_nom_max"] = np.where(
        n.generators["p_nom_extendable"] & n.generators.carrier.isin(gens),
        n.generators["p_nom"],
        n.generators["p_nom_max"],
    )

    n.generators["p_nom_min"] = np.where(
        n.generators["p_nom_extendable"] & n.generators.carrier.isin(gens),
        0,
        n.generators["p_nom_min"],
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
        p_min_pu=extend.p_min_pu,
        p_max_pu=extend.p_max_pu,
    )

    # time dependent factors added after as not all generators are time dependent
    marginal_cost_t = n.generators_t["marginal_cost"][
        [x for x in extend.index if x in n.generators_t.marginal_cost.columns]
    ]
    marginal_cost_t = marginal_cost_t.rename(
        columns={x: f"{x} new" for x in marginal_cost_t.columns},
    )
    n.generators_t["marginal_cost"] = n.generators_t["marginal_cost"].join(
        marginal_cost_t,
    )

    p_max_pu_t = n.generators_t["p_max_pu"][
        [x for x in extend.index if x in n.generators_t["p_max_pu"].columns]
    ]
    p_max_pu_t = p_max_pu_t.rename(columns={x: f"{x} new" for x in p_max_pu_t.columns})
    n.generators_t["p_max_pu"] = n.generators_t["p_max_pu"].join(p_max_pu_t)


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
    elec_config = snakemake.config["electricity"]

    Nyears = n.snapshot_weightings.objective.sum() / 8760.0
    costs = load_costs(
        snakemake.input.tech_costs,
        snakemake.config["costs"],
        elec_config["max_hours"],
        Nyears,
    )

    n.buses["location"] = n.buses.index

    attach_storageunits(n, costs, elec_config)
    attach_stores(n, costs, elec_config)
    attach_hydrogen_pipelines(n, costs, elec_config)

    add_nice_carrier_names(n, snakemake.config)

    if snakemake.params.retirement == "economic":
        economic_retirement_gens = elec_config.get("conventional_carriers", None)
        add_economic_retirement(n, costs, economic_retirement_gens)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
