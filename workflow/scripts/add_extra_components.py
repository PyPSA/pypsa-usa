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

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging
from add_electricity import (
    add_co2_emissions,
    add_missing_carriers,
    calculate_annuity,
    load_costs,
)

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def add_nice_carrier_names(n, config):
    carrier_i = n.carriers.index
    nice_names = (
        pd.Series(config["plotting"]["nice_names"])
        .reindex(carrier_i)
        .fillna(carrier_i.to_series().str.title())
    )
    n.carriers["nice_name"] = nice_names
    colors = pd.Series(config["plotting"]["tech_colors"]).reindex(carrier_i)
    if colors.isna().any():
        missing_i = list(colors.index[colors.isna()])
        logger.warning(f"tech_colors for carriers {missing_i} not defined in config.")
    n.carriers["color"] = colors


def attach_storageunits(n, costs, elec_opts, investment_year):
    carriers = elec_opts["extendable_carriers"]["StorageUnit"]
    carriers = [k for k in carriers if "battery_storage" in k]

    buses_i = n.buses.index

    lookup_store = {"H2": "electrolysis", "battery": "battery inverter"}
    lookup_dispatch = {"H2": "fuel cell", "battery": "battery inverter"}

    add_missing_carriers(n, carriers)
    add_co2_emissions(n, costs, carriers)
    for carrier in carriers:
        max_hours = int(carrier.split("hr_")[0])
        roundtrip_correction = 0.5 if "battery" in carrier else 1

        n.madd(
            "StorageUnit",
            buses_i,
            suffix=f" {carrier}_{investment_year}",
            bus=buses_i,
            carrier=carrier,
            p_nom_extendable=True,
            capital_cost=costs.at[carrier, "capital_cost"],
            marginal_cost=costs.at[carrier, "marginal_cost"],
            efficiency_store=costs.at[carrier, "efficiency"] ** roundtrip_correction,
            efficiency_dispatch=costs.at[carrier, "efficiency"] ** roundtrip_correction,
            max_hours=max_hours,
            cyclic_state_of_charge=True,
            build_year=investment_year,
            lifetime=costs.at[carrier, "lifetime"],
        )


def attach_phs_storageunits(n: pypsa.Network, elec_opts):
    carriers = elec_opts["extendable_carriers"]["StorageUnit"]
    carriers = [k for k in carriers if "PHS" in k]

    for carrier in carriers:
        max_hours = int(carrier.split("hr_")[0])

        psh_resources = (
            gpd.read_file(snakemake.input[f"phs_shp_{max_hours}"])
            .to_crs(4326)
            .rename(
                columns={
                    "System Installed Capacity (Megawatts)": "potential_mw",
                    "System Energy Storage Capacity (Gigawatt hours)": "potential_gwh",
                    "System Cost (2020 US Dollars per Installed Kilowatt)": "cost_kw",
                    "Longitude": "longitude",
                    "Latitude": "latitude",
                },
            )
        )[
            [
                "longitude",
                "latitude",
                "potential_gwh",
                "potential_mw",
                "cost_kw",
                "geometry",
            ]
        ]

        # Round CAPEX to $500 interval
        psh_resources["cost_kw_round"] = (psh_resources["cost_kw"] / 500).round() * 500

        # Join SC to PyPSA cluster
        region_onshore = gpd.read_file(snakemake.input.regions_onshore)
        region_onshore_psh = gpd.sjoin(
            region_onshore,
            psh_resources,
            how="inner",
        ).reset_index(drop=True)
        region_onshore_psh_grp = (
            region_onshore_psh.groupby(["name", "cost_kw_round"])["potential_mw"]
            .agg("sum")
            .reset_index()
        )

        region_onshore_psh_grp["class"] = (
            region_onshore_psh_grp.groupby(["name"]).cumcount() + 1
        )
        region_onshore_psh_grp["class"] = "c" + region_onshore_psh_grp["class"].astype(
            str,
        )
        region_onshore_psh_grp["tech"] = carrier
        region_onshore_psh_grp["carrier"] = region_onshore_psh_grp[
            ["tech", "class"]
        ].agg("_".join, axis=1)
        region_onshore_psh_grp["Generator"] = (
            region_onshore_psh_grp["name"] + " " + region_onshore_psh_grp["carrier"]
        )
        region_onshore_psh_grp = region_onshore_psh_grp.set_index("Generator")

        # Updated annualize capital cost based on real location
        psh_lifetime = 100  # years
        psh_discount_rate = 0.055  # per unit
        psh_fom = 0.885  # %/year
        psh_vom = 0.54  # $/MWh_e

        region_onshore_psh_grp["capital_cost"] = (
            (calculate_annuity(psh_lifetime, psh_discount_rate) + psh_fom / 100)
            * region_onshore_psh_grp["cost_kw_round"]
            * 1e3
            * n.snapshot_weightings.objective.sum()
            / 8760.0
        )

        region_onshore_psh_grp["marginal_cost"] = psh_vom

        # Set RT efficiency = 0.8
        efficiency_store = 0.894427191  # 0.894427191^2 = 0.8
        efficiency_dispatch = 0.894427191  # 0.894427191^2 = 0.8

        n.madd(
            "StorageUnit",
            region_onshore_psh_grp.index,
            bus=region_onshore_psh_grp.name,
            carrier=region_onshore_psh_grp.tech,
            p_nom_max=region_onshore_psh_grp.potential_mw,
            p_nom_extendable=True,
            capital_cost=region_onshore_psh_grp.capital_cost,
            marginal_cost=region_onshore_psh_grp.marginal_cost,
            efficiency_store=efficiency_store,
            efficiency_dispatch=efficiency_dispatch,
            max_hours=max_hours,
            cyclic_state_of_charge=True,
        )


def attach_stores(n, costs, elec_opts, investment_year):
    carriers = elec_opts["extendable_carriers"]["Store"]

    add_missing_carriers(n, carriers)
    add_co2_emissions(n, costs, carriers)

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
            build_year=investment_year,
            lifetime=costs.at["hydrogen storage underground", "lifetime"],
            suffix=f" {investment_year}",
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
            build_year=investment_year,
            lifetime=costs.at["electrolysis", "lifetime"],
            suffix=str(investment_year),
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
            build_year=investment_year,
            lifetime=costs.at["fuel cell", "lifetime"],
            suffix=str(investment_year),
        )


def attach_hydrogen_pipelines(n, costs, elec_opts, investment_year):
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
        build_year=investment_year,
        lifetime=costs.at["H2 pipeline", "lifetime"],
        suffix=f" {investment_year}",
    )


def add_economic_retirement(
    n: pypsa.Network,
    costs: pd.DataFrame,
    gens: list[str] = None,
):
    """
    Seperates extendable conventional generators into existing and new
    generators to support economic retirement.

    Specifically this function does the following:
    1. Creates duplicate generators for any that are tagged as extendable. For
    example, an extendable "CCGT" generator will be split into "CCGT existing" and "CCGT"
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
    retirement_mask = n.generators["p_nom_extendable"] & (
        n.generators["carrier"].isin(gens) if gens else True
    )
    retirement_gens = n.generators[retirement_mask]
    if retirement_gens.empty:
        return

    # divide by 100 b/c FOM expressed as percentage of CAPEX
    n.generators["capital_cost"] = n.generators.apply(
        lambda row: (
            row["capital_cost"]
            if not row.name in (retirement_gens.index)
            else row["capital_cost"] * costs.at[row["carrier"], "FOM"] / 100
        ),
        axis=1,
    )

    # rename retiring generators to include "existing" suffix
    n.generators.index = n.generators.apply(
        lambda row: (
            row.name
            if not row.name in (retirement_gens.index)
            else row.name + " existing"
        ),
        axis=1,
    )

    n.generators["p_nom_max"] = np.where(
        retirement_mask,
        n.generators["p_nom"],
        n.generators["p_nom_max"],
    )

    n.generators["p_nom_min"] = np.where(
        retirement_mask,
        0,
        n.generators["p_nom_min"],
    )

    n.madd(
        "Generator",
        retirement_gens.index,
        carrier=retirement_gens.carrier,
        bus=retirement_gens.bus,
        p_nom_min=0,
        p_nom=0,
        p_nom_max=retirement_gens.p_nom_max,
        p_nom_extendable=True,
        ramp_limit_up=retirement_gens.ramp_limit_up,
        ramp_limit_down=retirement_gens.ramp_limit_down,
        efficiency=retirement_gens.efficiency,
        marginal_cost=retirement_gens.marginal_cost,
        capital_cost=retirement_gens.capital_cost,
        build_year=n.investment_periods[0],
        lifetime=retirement_gens.lifetime,
        p_min_pu=retirement_gens.p_min_pu,
        p_max_pu=retirement_gens.p_max_pu,
    )

    # time dependent factors added after as not all generators are time dependent
    marginal_cost_t = n.generators_t["marginal_cost"][
        [x for x in retirement_gens.index if x in n.generators_t.marginal_cost.columns]
    ]
    marginal_cost_t = marginal_cost_t.rename(
        columns={x: f"{x} existing" for x in marginal_cost_t.columns},
    )
    n.generators_t["marginal_cost"] = n.generators_t["marginal_cost"].join(
        marginal_cost_t,
    )

    p_max_pu_t = n.generators_t["p_max_pu"][
        [x for x in retirement_gens.index if x in n.generators_t["p_max_pu"].columns]
    ]
    p_max_pu_t = p_max_pu_t.rename(
        columns={x: f"{x} existing" for x in p_max_pu_t.columns},
    )
    n.generators_t["p_max_pu"] = n.generators_t["p_max_pu"].join(p_max_pu_t)


def attach_multihorizon_generators(
    n: pypsa.Network,
    costs: dict,
    gens: pd.DataFrame,
    investment_year: int,
):
    """
    Adds multiple investment options for a given set of extendable carriers.

    Specifically this function does the following:
    1. ....

    Arguments:
    n: pypsa.Network,
    costs_dict: dict,
        Dict of costs for each investment period
    carriers: List[str]
        List of carriers to add multiple investment options for
    """
    if gens.empty or len(n.investment_periods) == 1:
        return

    n.madd(
        "Generator",
        gens.index,
        suffix=f" {investment_year}",
        carrier=gens.carrier,
        bus=gens.bus,
        p_nom_min=0 if investment_year != n.investment_periods[0] else gens.p_nom_min,
        p_nom=0 if investment_year != n.investment_periods[0] else gens.p_nom,
        p_nom_max=gens.p_nom_max,
        p_nom_extendable=True,
        ramp_limit_up=gens.ramp_limit_up,
        ramp_limit_down=gens.ramp_limit_down,
        efficiency=gens.efficiency,
        marginal_cost=gens.marginal_cost,
        p_min_pu=gens.p_min_pu,
        p_max_pu=gens.p_max_pu,
        capital_cost=gens.carrier.map(costs.capital_cost),
        build_year=investment_year,
        lifetime=gens.carrier.map(costs.lifetime),
    )

    # time dependent factors added after as not all generators are time dependent
    marginal_cost_t = n.generators_t["marginal_cost"][
        [x for x in gens.index if x in n.generators_t.marginal_cost.columns]
    ]
    marginal_cost_t = marginal_cost_t.rename(
        columns={x: f"{x} {investment_year}" for x in marginal_cost_t.columns},
    )
    n.generators_t["marginal_cost"] = n.generators_t["marginal_cost"].join(
        marginal_cost_t,
    )

    p_max_pu_t = n.generators_t["p_max_pu"][
        [x for x in gens.index if x in n.generators_t["p_max_pu"].columns]
    ]
    p_max_pu_t = p_max_pu_t.rename(
        columns={x: f"{x} {investment_year}" for x in p_max_pu_t.columns},
    )
    n.generators_t["p_max_pu"] = n.generators_t["p_max_pu"].join(p_max_pu_t)


def attach_newCarrier_generators(n, costs, carriers, investment_year):
    """
    Adds new carriers to the network.

    Specifically this function does the following:
    1. Adds new carriers to the network
    2. Adds generators for the new carriers

    Arguments:
    n: pypsa.Network,
    costs: pd.DataFrame,
    carriers: List[str]
        List of carriers to add to the network
    investment_year: int
        Year of investment
    """
    if not carriers:
        return

    add_missing_carriers(n, carriers)
    add_co2_emissions(n, costs, carriers)

    buses_i = n.buses.index
    for carrier in carriers:
        n.madd(
            "Generator",
            buses_i,
            suffix=f" {carrier}_{investment_year}",
            bus=buses_i,
            carrier=carrier,
            p_nom_extendable=True,
            capital_cost=costs.at[carrier, "capital_cost"],
            marginal_cost=costs.at[carrier, "marginal_cost"],
            efficiency=costs.at[carrier, "efficiency"],
            build_year=investment_year,
            lifetime=costs.at[carrier, "lifetime"],
        )


def apply_itc(n, itc_modifier):
    """
    Applies investment tax credit to all extendable components in the network.

    Arguments:
    n: pypsa.Network,
    itc_modifier: dict,
        Dict of ITC modifiers for each carrier
    """
    for carrier in itc_modifier.keys():
        carrier_mask = n.generators["carrier"] == carrier
        n.generators.loc[carrier_mask, "capital_cost"] *= 1 - itc_modifier[carrier]

        carrier_mask = n.storage_units["carrier"] == carrier
        n.storage_units.loc[carrier_mask, "capital_cost"] *= 1 - itc_modifier[carrier]


def apply_ptc(n, ptc_modifier):
    """
    Applies production tax credit to all extendable components in the network.

    Arguments:
    n: pypsa.Network,
    ptc_modifier: dict,
        Dict of PTC modifiers for each carrier
    """
    for carrier in ptc_modifier.keys():
        carrier_mask = n.generators["carrier"] == carrier
        mc = n.get_switchable_as_dense("Generator", "marginal_cost").loc[
            :,
            carrier_mask,
        ]
        n.generators_t.marginal_cost.loc[:, carrier_mask] = mc - ptc_modifier[carrier]
        n.generators.loc[carrier_mask, "marginal_cost"] -= ptc_modifier[carrier]


def apply_max_annual_growth_rate(n, max_growth):
    """
    Applies maximum annual growth rate to all extendable components in the
    network.

    Arguments:
    n: pypsa.Network,
    max_annual_growth_rate: dict,
        Dict of maximum annual growth rate for each carrier
    """
    max_annual_growth_rate = max_growth["rate"]
    growth_base = max_growth["base"]
    years = n.investment_period_weightings.index.diff()
    years = years.dropna().values.mean()
    for carrier in max_annual_growth_rate.keys():
        ann_growth_rate = max_annual_growth_rate[carrier]
        growth_factor = ann_growth_rate**years
        p_nom = n.generators.p_nom.loc[n.generators.carrier == carrier].sum()
        n.carriers.loc[carrier, "max_growth"] = growth_base.get(carrier) or p_nom
        n.carriers.loc[carrier, "max_relative_growth"] = (
            max_annual_growth_rate[carrier] ** years
        )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_extra_components",
            interconnect="texas",
            clusters=10,
        )
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)
    elec_config = snakemake.config["electricity"]

    Nyears = n.snapshot_weightings.loc[n.investment_periods[0]].objective.sum() / 8760.0

    costs_dict = {
        n.investment_periods[i]: load_costs(
            snakemake.input.tech_costs[i],
            snakemake.config["costs"],
            elec_config["max_hours"],
            Nyears,
        )
        for i in range(len(n.investment_periods))
    }

    if snakemake.params.retirement == "economic":
        economic_retirement_gens = elec_config.get("conventional_carriers", None)
        add_economic_retirement(
            n,
            costs_dict[n.investment_periods[0]],
            economic_retirement_gens,
        )

    new_carriers = list(
        set(elec_config["extendable_carriers"].get("Generator", []))
        - set(elec_config["conventional_carriers"])
        - set(elec_config["renewable_carriers"]),
    )

    gens = n.generators[
        n.generators["p_nom_extendable"]
        & n.generators["carrier"].isin(elec_config["extendable_carriers"]["Generator"])
        & ~n.generators.index.str.contains("existing")
    ]

    if any("PHS" in s for s in elec_config["extendable_carriers"]["StorageUnit"]):
        attach_phs_storageunits(n, elec_config)

    for investment_year in n.investment_periods:
        costs = costs_dict[investment_year]
        attach_storageunits(n, costs, elec_config, investment_year)
        attach_stores(n, costs, elec_config, investment_year)
        attach_hydrogen_pipelines(n, costs, elec_config, investment_year)
        attach_multihorizon_generators(n, costs, gens, investment_year)
        attach_newCarrier_generators(n, costs, new_carriers, investment_year)

    if not gens.empty and not len(n.investment_periods) == 1:
        n.mremove(
            "Generator",
            gens.index,
        )  # Remove duplicate generators from first investment period

    apply_itc(n, snakemake.config["costs"]["itc_modifier"])
    apply_ptc(n, snakemake.config["costs"]["ptc_modifier"])
    apply_max_annual_growth_rate(n, snakemake.config["costs"]["max_growth"])
    add_nice_carrier_names(n, snakemake.config)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
