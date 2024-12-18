# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Solves optimal operation and capacity for a network with the option to
iteratively optimize while updating line reactances.

This script is used for optimizing the electrical network as well as the
sector coupled network.

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.

The optimization is based on the :func:`network.optimize` function.
Additionally, some extra constraints specified in :mod:`solve_network` are added.

.. note::

    The rules ``solve_elec_networks`` and ``solve_sector_networks`` run
    the workflow for all scenarios in the configuration file (``scenario:``)
    based on the rule :mod:`solve_network`.
"""
import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
import yaml
from _helpers import (
    configure_logging,
    is_transport_model,
    update_config_from_wildcards,
    update_config_with_sector_opts,
)
from constants import NG_MWH_2_MMCF
from eia import Trade
from pypsa.descriptors import get_switchable_as_dense as get_as_dense

logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)


def add_land_use_constraint_perfect(n):
    """
    Add global constraints for tech capacity limit.
    """
    logger.info("Add land-use constraint for perfect foresight")

    def compress_series(s):
        def process_group(group):
            if group.nunique() == 1:
                return pd.Series(group.iloc[0], index=[None])
            else:
                return group

        return s.groupby(level=[0, 1]).apply(process_group)

    def new_index_name(t):
        # Convert all elements to string and filter out None values
        parts = [str(x) for x in t if x is not None]
        # Join with space, but use a dash for the last item if not None
        return " ".join(parts[:2]) + (f"-{parts[-1]}" if len(parts) > 2 else "")

    def check_p_min_p_max(p_nom_max):
        p_nom_min = n.generators[ext_i].groupby(grouper).sum().p_nom_min
        p_nom_min = p_nom_min.reindex(p_nom_max.index)
        check = p_nom_min.groupby(level=[0, 1]).sum() > p_nom_max.groupby(level=[0, 1]).min()
        if check.sum():
            logger.warning(
                f"summed p_min_pu values at node larger than technical potential {check[check].index}",
            )

    grouper = [n.generators.carrier, n.generators.bus]
    ext_i = n.generators.p_nom_extendable & ~n.generators.index.str.contains("existing")
    # get technical limit per node
    p_nom_max = n.generators[ext_i].groupby(grouper).sum().p_nom_max / len(
        n.investment_periods,
    )

    # drop carriers without tech limit
    p_nom_max = p_nom_max[~p_nom_max.isin([np.inf, np.nan])]
    # carrier
    carriers = p_nom_max.index.get_level_values(0).unique()
    gen_i = n.generators[(n.generators.carrier.isin(carriers)) & (ext_i)].index
    n.generators.loc[gen_i, "p_nom_min"] = 0
    # check minimum capacities
    check_p_min_p_max(p_nom_max)

    df = p_nom_max.reset_index()
    df["name"] = df.apply(lambda row: f"nom_max_{row['carrier']}", axis=1)

    for name in df.name.unique():
        df_carrier = df[df.name == name]
        bus = df_carrier.bus
        n.buses.loc[bus, name] = df_carrier.p_nom_max.values
    return n


def add_co2_sequestration_limit(n, limit=200):
    """
    Add a global constraint on the amount of Mt CO2 that can be sequestered.
    """
    n.carriers.loc["co2 stored", "co2_absorptions"] = -1
    n.carriers.co2_absorptions = n.carriers.co2_absorptions.fillna(0)

    limit = limit * 1e6
    for o in opts:
        if "seq" not in o:
            continue
        limit = float(o[o.find("seq") + 3 :]) * 1e6
        break

    n.add(
        "GlobalConstraint",
        "co2_sequestration_limit",
        sense="<=",
        constant=limit,
        type="primary_energy",
        carrier_attribute="co2_absorptions",
    )


def prepare_network(
    n,
    solve_opts=None,
    config=None,
    foresight=None,
    planning_horizons=None,
    co2_sequestration_potential=None,
):
    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,
            n.storage_units_t.inflow,
        ):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    load_shedding = solve_opts.get("load_shedding")
    if load_shedding:
        # intersect between macroeconomic and surveybased willingness to pay
        # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
        # TODO: retrieve color and nice name from config
        n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
        buses_i = n.buses.query("carrier == 'AC'").index
        if not np.isscalar(load_shedding):
            # TODO: do not scale via sign attribute (use Eur/MWh instead of Eur/kWh)
            load_shedding = 1e2  # Eur/kWh

        n.madd(
            "Generator",
            buses_i,
            " load",
            bus=buses_i,
            carrier="load",
            sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=load_shedding,  # Eur/kWh
            p_nom=1e9,  # kW
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components():
            if "marginal_cost" in t.df:
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (np.random.random(len(t.df)) - 0.5)

        for t in n.iterate_components(["Line", "Link"]):
            t.df["capital_cost"] += (1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    if foresight == "perfect":
        n = add_land_use_constraint_perfect(n)

    if n.stores.carrier.eq("co2 stored").any():
        limit = co2_sequestration_potential
        add_co2_sequestration_limit(n, limit=limit)

    return n


def add_CCL_constraints(n, config):
    """
    Add CCL (country & carrier limit) constraint to the network.

    Add minimum or maximum levels of generator nominal capacity per carrier for individual countries. Each constraint can be designated for a specified planning horizon in multi-period models. Opts and path for agg_p_nom_minmax.csv must be defined in config.yaml. Default file is available at config/policy_constraints/agg_p_nom_minmax.csv.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-CCL-24H]
    electricity:
        agg_p_nom_limits: config/policy_constraints/agg_p_nom_minmax.csv
    """
    agg_p_nom_minmax = pd.read_csv(
        config["electricity"]["agg_p_nom_limits"],
        index_col=[1, 2],
    )
    agg_p_nom_minmax = agg_p_nom_minmax[
        agg_p_nom_minmax.planning_horizon == int(snakemake.params.planning_horizons[0])  # need to make multi-horizon
    ].drop(columns="planning_horizon")

    logger.info("Adding generation capacity constraints per carrier and country")
    p_nom = n.model["Generator-p_nom"]

    gens = n.generators.query("p_nom_extendable").rename_axis(index="Generator-ext")
    grouper = pd.concat([gens.bus.map(n.buses.country), gens.carrier], axis=1)
    lhs = p_nom.groupby(grouper).sum().rename(bus="country")

    minimum = xr.DataArray(agg_p_nom_minmax["min"].dropna()).rename(dim_0="group")
    index = minimum.indexes["group"].intersection(lhs.indexes["group"])
    if not index.empty:
        n.model.add_constraints(
            lhs.sel(group=index) >= minimum.loc[index],
            name="agg_p_nom_min",
        )

    maximum = xr.DataArray(agg_p_nom_minmax["max"].dropna()).rename(dim_0="group")
    index = maximum.indexes["group"].intersection(lhs.indexes["group"])
    if not index.empty:
        n.model.add_constraints(
            lhs.sel(group=index) <= maximum.loc[index],
            name="agg_p_nom_max",
        )


def add_RPS_constraints(n, config):
    """
    Add Renewable Portfolio Standards constraint to the network.

    Add percent levels of generator production (MWh) per carrier or groups of carriers for individual countries. Each constraint can be designated for a specified planning horizon in multi-period models. Opts and path for portfolio_standards.csv must be defined in config.yaml. Default file is available at config/policy_constraints/portfolio_standards.csv.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-RPS-24H]
    electricity:
        portfolio_standards: config/policy_constraints/portfolio_standards.csv
    """
    portfolio_standards = pd.read_csv(
        config["electricity"]["portfolio_standards"],
    )
    rps_carriers = [
        "onwind",
        "offwind",
        "offwind_floating",
        "solar",
        "hydro",
        "geothermal",
        "biomass",
        "EGS",
    ]
    ces_carriers = [
        "onwind",
        "offwind",
        "offwind_floating",
        "solar",
        "hydro",
        "geothermal",
        "EGS",
        "biomass",
        "nuclear",
    ]
    state_memberships = n.buses.groupby("reeds_state")["reeds_zone"].apply(lambda x: ", ".join(x)).to_dict()

    rps_reeds = pd.read_csv(
        snakemake.input.rps_reeds,
    )
    rps_reeds["region"] = rps_reeds["st"].map(state_memberships)
    rps_reeds.dropna(subset="region", inplace=True)
    rps_reeds["carrier"] = [", ".join(rps_carriers)] * len(rps_reeds)
    rps_reeds.rename(
        columns={"t": "planning_horizon", "rps_all": "pct", "st": "name"},
        inplace=True,
    )
    rps_reeds.drop(columns=["rps_solar", "rps_wind"], inplace=True)

    ces_reeds = pd.read_csv(
        snakemake.input.ces_reeds,
    ).melt(id_vars="st", var_name="planning_horizon", value_name="pct")
    ces_reeds["region"] = ces_reeds["st"].map(state_memberships)
    ces_reeds.dropna(subset="region", inplace=True)
    ces_reeds["carrier"] = [", ".join(ces_carriers)] * len(ces_reeds)
    ces_reeds.rename(columns={"st": "name"}, inplace=True)

    portfolio_standards = pd.concat([portfolio_standards, rps_reeds, ces_reeds])
    portfolio_standards = portfolio_standards[portfolio_standards.pct > 0.0]
    portfolio_standards = portfolio_standards[
        portfolio_standards.planning_horizon.isin(snakemake.params.planning_horizons)
    ]
    portfolio_standards.set_index("name", inplace=True)

    for idx, pct_lim in portfolio_standards.iterrows():
        region_list = [region_.strip() for region_ in pct_lim.region.split(",")]
        region_buses = n.buses[
            (
                n.buses.country.isin(region_list)
                | n.buses.reeds_state.isin(region_list)
                | n.buses.interconnect.str.lower().isin(region_list)
                | (1 if "all" in region_list else 0)
            )
        ]

        if region_buses.empty:
            continue

        carriers = [carrier_.strip() for carrier_ in pct_lim.carrier.split(",")]

        # generators
        region_gens = n.generators[n.generators.bus.isin(region_buses.index)]
        region_gens_eligible = region_gens[region_gens.carrier.isin(carriers)]

        if not region_gens.empty:
            p_eligible = n.model["Generator-p"].sel(
                period=pct_lim.planning_horizon,
                Generator=region_gens_eligible.index,
            )
            lhs = p_eligible.sum()

            region_demand = (
                n.loads_t.p_set.loc[
                    pct_lim.planning_horizon,
                    n.loads.bus.isin(region_buses.index),
                ]
                .sum()
                .sum()
            )

            rhs = pct_lim.pct * region_demand

            n.model.add_constraints(
                lhs >= rhs,
                name=f"GlobalConstraint-{pct_lim.name}_{pct_lim.planning_horizon}_rps_limit",
            )
            logger.info(
                f"Adding RPS {pct_lim.name}_{pct_lim.planning_horizon} for {pct_lim.planning_horizon}.",
            )


def add_EQ_constraints(n, o, scaling=1e-1):
    """
    Add equity constraints to the network.

    Currently this is only implemented for the electricity sector only.

    Opts must be specified in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    o : str

    Example
    -------
    scenario:
        opts: [Co2L-EQ0.7-24H]

    Require each country or node to on average produce a minimal share
    of its total electricity consumption itself. Example: EQ0.7c demands each country
    to produce on average at least 70% of its consumption; EQ0.7 demands
    each node to produce on average at least 70% of its consumption.
    """
    # TODO: Generalize to cover myopic and other sectors?
    float_regex = r"[0-9]*\.?[0-9]+"
    level = float(re.findall(float_regex, o)[0])
    if o[-1] == "c":
        ggrouper = n.generators.bus.map(n.buses.country)
        lgrouper = n.loads.bus.map(n.buses.country)
        sgrouper = n.storage_units.bus.map(n.buses.country)
    else:
        ggrouper = n.generators.bus
        lgrouper = n.loads.bus
        sgrouper = n.storage_units.bus
    load = n.snapshot_weightings.generators @ n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    inflow = n.snapshot_weightings.stores @ n.storage_units_t.inflow.groupby(sgrouper, axis=1).sum()
    inflow = inflow.reindex(load.index).fillna(0.0)
    rhs = scaling * (level * load - inflow)
    p = n.model["Generator-p"]
    lhs_gen = (p * (n.snapshot_weightings.generators * scaling)).groupby(ggrouper.to_xarray()).sum().sum("snapshot")
    # TODO: double check that this is really needed, why do have to subtract the spillage
    if not n.storage_units_t.inflow.empty:
        spillage = n.model["StorageUnit-spill"]
        lhs_spill = (
            (spillage * (-n.snapshot_weightings.stores * scaling)).groupby(sgrouper.to_xarray()).sum().sum("snapshot")
        )
        lhs = lhs_gen + lhs_spill
    else:
        lhs = lhs_gen
    n.model.add_constraints(lhs >= rhs, name="equity_min")


def add_BAU_constraints(n, config):
    """
    Add a per-carrier minimal overall capacity.

    BAU_mincapacities and opts must be adjusted in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-BAU-24H]
    electricity:
        BAU_mincapacities:
            solar: 0
            onwind: 0
            OCGT: 100000
            offwind-ac: 0
            offwind-dc: 0
    Which sets minimum expansion across all nodes e.g. in Europe to 100GW.
    OCGT bus 1 + OCGT bus 2 + ... > 100000
    """
    mincaps = pd.Series(config["electricity"]["BAU_mincapacities"])
    p_nom = n.model["Generator-p_nom"]
    ext_i = n.generators.query("p_nom_extendable")
    ext_carrier_i = xr.DataArray(ext_i.carrier.rename_axis("Generator-ext"))
    lhs = p_nom.groupby(ext_carrier_i).sum()
    index = mincaps.index.intersection(lhs.indexes["carrier"])
    rhs = mincaps[index].rename_axis("carrier")
    n.model.add_constraints(lhs >= rhs, name="bau_mincaps")


def add_interface_limits(n, sns, config):
    """
    Adds interface transmission limits to constrain inter-regional transfer
    capacities based on user-defined inter-regional transfer capacity limits.
    """
    logger.info("Adding Interface Transmission Limits.")
    transport_model = is_transport_model(snakemake.params.transmission_network)
    limits = pd.read_csv(snakemake.input.flowgates)
    user_limits = pd.read_csv(
        config["electricity"]["transmission_interface_limits"],
    ).rename(
        columns={
            "region_1": "r",
            "region_2": "rr",
            "flow_12": "MW_f0",
            "flow_21": "MW_r0",
        },
    )

    limits = pd.concat([limits, user_limits])

    for idx, interface in limits.iterrows():
        regions_list_r = [region.strip() for region in interface.r.split(",")]
        regions_list_rr = [region.strip() for region in interface.rr.split(",")]

        zone0_buses = n.buses[n.buses.country.isin(regions_list_r)]
        zone1_buses = n.buses[n.buses.country.isin(regions_list_rr)]
        if zone0_buses.empty | zone1_buses.empty:
            continue

        logger.info(f"Adding Interface Transmission Limit for {interface.interface}")

        interface_lines_b0 = n.lines[n.lines.bus0.isin(zone0_buses.index) & n.lines.bus1.isin(zone1_buses.index)]
        interface_lines_b1 = n.lines[n.lines.bus0.isin(zone1_buses.index) & n.lines.bus1.isin(zone0_buses.index)]
        interface_links_b0 = n.links[n.links.bus0.isin(zone0_buses.index) & n.links.bus1.isin(zone1_buses.index)]
        interface_links_b1 = n.links[n.links.bus0.isin(zone1_buses.index) & n.links.bus1.isin(zone0_buses.index)]

        if not n.lines.empty:
            line_flows = n.model["Line-s"].loc[:, interface_lines_b1.index].sum(
                dims="Line",
            ) - n.model["Line-s"].loc[
                :,
                interface_lines_b0.index,
            ].sum(
                dims="Line",
            )
        else:
            line_flows = 0.0
        lhs = line_flows

        if (
            not (pd.concat([interface_links_b0, interface_links_b1]).empty)
            and ("RESOLVE" in interface.interface or transport_model)
            # Apply link constraints if RESOLVE constraint or if zonal model. ITLs should usually only apply to AC lines if DC PF is used.
        ):
            link_flows = n.model["Link-p"].loc[:, interface_links_b1.index].sum(
                dims="Link",
            ) - n.model["Link-p"].loc[
                :,
                interface_links_b0.index,
            ].sum(
                dims="Link",
            )
            lhs += link_flows

        rhs_pos = interface.MW_f0 * -1
        n.model.add_constraints(lhs >= rhs_pos, name=f"ITL_{interface.interface}_pos")

        rhs_neg = interface.MW_r0
        n.model.add_constraints(lhs <= rhs_neg, name=f"ITL_{interface.interface}_neg")


def add_regional_co2limit(n, sns, config):
    """
    Adding regional regional CO2 Limits Specified in the config.yaml.
    """
    from pypsa.descriptors import get_switchable_as_dense as get_as_dense

    regional_co2_lims = pd.read_csv(
        config["electricity"]["regional_Co2_limits"],
        index_col=[0],
    )
    logger.info("Adding regional Co2 Limits.")
    regional_co2_lims = regional_co2_lims[regional_co2_lims.planning_horizon.isin(snakemake.params.planning_horizons)]
    weightings = n.snapshot_weightings.loc[n.snapshots]

    # if n._multi_invest:
    #     period_weighting = n.investment_period_weightings.years[sns.unique("period")]
    #     weightings = weightings.mul(period_weighting, level=0, axis=0)

    for idx, emmission_lim in regional_co2_lims.iterrows():
        region_list = [region.strip() for region in emmission_lim.regions.split(",")]
        region_buses = n.buses[
            (
                n.buses.country.isin(region_list)
                | n.buses.reeds_state.isin(region_list)
                | n.buses.interconnect.str.lower().isin(region_list)
                | (1 if "all" in region_list else 0)
            )
        ]

        emissions = n.carriers.co2_emissions.fillna(0)[lambda ds: ds != 0]
        region_gens = n.generators[n.generators.bus.isin(region_buses.index)]
        region_gens_em = region_gens.query("carrier in @emissions.index")

        if region_buses.empty or region_gens_em.empty:
            continue

        region_co2lim = emmission_lim.limit
        planning_horizon = emmission_lim.planning_horizon

        efficiency = get_as_dense(
            n,
            "Generator",
            "efficiency",
            inds=region_gens_em.index,
        )  # mw_elect/mw_th
        em_pu = region_gens_em.carrier.map(emissions) / efficiency  # tonnes_co2/mw_electrical
        em_pu = em_pu.multiply(weightings.generators, axis=0).loc[planning_horizon].fillna(0)

        # Emitting Gens
        p_em = n.model["Generator-p"].loc[:, region_gens_em.index].sel(period=planning_horizon)
        lhs = (p_em * em_pu).sum()
        rhs = region_co2lim

        # if EF_imports > 0.0:
        #     region_storage = n.storage_units[n.storage_units.bus.isin(region_buses.index)]
        #     EF_imports = emmission_lim.import_emissions_factor  # MT CO₂e/MWh_elec
        #     # All Gens
        #     p = (
        #         n.model["Generator-p"]
        #         .loc[:, region_gens.index]
        #         .sel(period=planning_horizon)
        #         .mul(weightings.generators.loc[planning_horizon])
        #     )
        #     imports_gen_weightings = pd.DataFrame(columns=region_gens.index, index=n.snapshots, data=1)
        #     weighted_imports_p = (
        #         (imports_gen_weightings * EF_imports).multiply(weightings.generators, axis=0).loc[planning_horizon]
        #     )
        #     lhs -= (p * weighted_imports_p).sum()

        #     if not region_storage.empty:
        #         p_store_discharge = (
        #             n.model["StorageUnit-p_dispatch"].loc[:, region_storage.index].sel(period=planning_horizon)
        #         )
        #         imports_storage_weightings = pd.DataFrame(columns=region_storage.index, index=n.snapshots, data=1)
        #         weighted_imports_p = (
        #             (imports_storage_weightings * EF_imports)
        #             .multiply(weightings.generators, axis=0)
        #             .loc[planning_horizon]
        #         )
        #         lhs -= (p_store_discharge * weighted_imports_p).sum()

        #     region_demand = (
        #         n.loads_t.p_set.loc[
        #             planning_horizon,
        #             n.loads.bus.isin(region_buses.index),
        #         ]
        #         .sum()
        #         .sum()
        #     )

        #     rhs -= region_demand * EF_imports

        n.model.add_constraints(
            lhs <= rhs,
            name=f"GlobalConstraint-{emmission_lim.name}_{planning_horizon}co2_limit",
        )

        logger.info(
            f"Adding regional Co2 Limit for {emmission_lim.name} in {planning_horizon}",
        )


def add_SAFE_constraints(n, config):
    """
    Add a capacity reserve margin of a certain fraction above the peak demand.
    Renewable generators and storage do not contribute. Ignores network.

    Parameters
    ----------
        n : pypsa.Network
        config : dict

    Example
    -------
    config.yaml requires to specify opts:

    scenario:
        opts: [Co2L-SAFE-24H]
    electricity:
        SAFE_reservemargin: 0.1
    Which sets a reserve margin of 10% above the peak demand.
    """
    peakdemand = n.loads_t.p_set.sum(axis=1).max()
    margin = 1.0 + config["electricity"]["SAFE_reservemargin"]
    reserve_margin = peakdemand * margin
    ext_gens_i = n.generators.query(
        "carrier in @conventional_carriers & p_nom_extendable",
    ).index
    p_nom = n.model["Generator-p_nom"].loc[ext_gens_i]
    lhs = p_nom.sum()
    exist_conv_caps = n.generators.query(
        "~p_nom_extendable & carrier in @conventional_carriers",
    ).p_nom.sum()
    rhs = reserve_margin - exist_conv_caps
    n.model.add_constraints(lhs >= rhs, name="safe_mintotalcap")


def add_SAFER_constraints(n, config):
    """
    Add a capacity reserve margin of a certain fraction above the peak demand
    for regions defined in configuration file. Renewable generators and storage
    do not contribute towards PRM.

    Parameters
    ----------
        n : pypsa.Network
        config : dict
    """
    regional_prm = pd.read_csv(
        config["electricity"]["SAFE_regional_reservemargins"],
        index_col=[0],
    )

    reeds_prm = pd.read_csv(
        snakemake.input.safer_reeds,
        index_col=[0],
    )
    NERC_memberships = n.buses.groupby("nerc_reg")["reeds_zone"].apply(lambda x: ", ".join(x)).to_dict()
    reeds_prm["region"] = reeds_prm.index.map(NERC_memberships)
    reeds_prm.dropna(subset="region", inplace=True)
    reeds_prm.drop(
        columns=["none", "ramp2025_20by50", "ramp2025_25by50", "ramp2025_30by50"],
        inplace=True,
    )
    reeds_prm.rename(columns={"static": "prm", "t": "planning_horizon"}, inplace=True)

    regional_prm = pd.concat([regional_prm, reeds_prm])
    regional_prm = regional_prm[regional_prm.planning_horizon.isin(snakemake.params.planning_horizons)]

    for idx, prm in regional_prm.iterrows():
        region_list = [region_.strip() for region_ in prm.region.split(",")]
        region_buses = n.buses[
            (
                n.buses.country.isin(region_list)
                | n.buses.reeds_state.isin(region_list)
                | n.buses.interconnect.str.lower().isin(region_list)
                | n.buses.nerc_reg.isin(region_list)
                | (1 if "all" in region_list else 0)
            )
        ]

        if region_buses.empty:
            continue

        peakdemand = (
            n.loads_t.p_set.loc[
                prm.planning_horizon,
                n.loads.bus.isin(region_buses.index),
            ]
            .sum(axis=1)
            .max()
        )
        margin = 1.0 + prm.prm
        planning_reserve = peakdemand * margin

        region_gens = n.generators[n.generators.bus.isin(region_buses.index)]
        ext_gens_i = region_gens.query(
            "carrier in @conventional_carriers & p_nom_extendable",
        ).index
        p_nom = n.model["Generator-p_nom"].loc[ext_gens_i]
        lhs = p_nom.sum()
        exist_conv_caps = region_gens.query(
            "~p_nom_extendable & carrier in @conventional_carriers",
        ).p_nom.sum()
        rhs = planning_reserve - exist_conv_caps
        n.model.add_constraints(
            lhs >= rhs,
            name=f"GlobalConstraint-{prm.name}_{prm.planning_horizon}_PRM",
        )


def add_operational_reserve_margin(n, sns, config):
    """
    Build reserve margin constraints based on the formulation given in
    https://genxproject.github.io/GenX/dev/core/#Reserves.

    Parameters
    ----------
        n : pypsa.Network
        sns: pd.DatetimeIndex
        config : dict

    Example:
    --------
    config.yaml requires to specify operational_reserve:
    operational_reserve: # like https://genxproject.github.io/GenX/dev/core/#Reserves
        activate: true
        epsilon_load: 0.02 # percentage of load at each snapshot
        epsilon_vres: 0.02 # percentage of VRES at each snapshot
        contingency: 400000 # MW
    """
    reserve_config = config["electricity"]["operational_reserve"]
    EPSILON_LOAD = reserve_config["epsilon_load"]
    EPSILON_VRES = reserve_config["epsilon_vres"]
    CONTINGENCY = reserve_config["contingency"]

    # Reserve Variables
    n.model.add_variables(
        0,
        np.inf,
        coords=[sns, n.generators.index],
        name="Generator-r",
    )
    reserve = n.model["Generator-r"]
    summed_reserve = reserve.sum("Generator")

    # Share of extendable renewable capacities
    ext_i = n.generators.query("p_nom_extendable").index
    vres_i = n.generators_t.p_max_pu.columns
    if not ext_i.empty and not vres_i.empty:
        capacity_factor = n.generators_t.p_max_pu[vres_i.intersection(ext_i)]
        p_nom_vres = n.model["Generator-p_nom"].loc[vres_i.intersection(ext_i)].rename({"Generator-ext": "Generator"})
        lhs = summed_reserve + (p_nom_vres * (-EPSILON_VRES * capacity_factor)).sum(
            "Generator",
        )
    else:  # if no extendable VRES
        lhs = summed_reserve

    # Total demand per t
    demand = get_as_dense(n, "Load", "p_set").sum(axis=1)

    # VRES potential of non extendable generators
    capacity_factor = n.generators_t.p_max_pu[vres_i.difference(ext_i)]
    renewable_capacity = n.generators.p_nom[vres_i.difference(ext_i)]
    potential = (capacity_factor * renewable_capacity).sum(axis=1)

    # Right-hand-side
    rhs = EPSILON_LOAD * demand + EPSILON_VRES * potential + CONTINGENCY

    n.model.add_constraints(lhs >= rhs, name="reserve_margin")

    # additional constraint that capacity is not exceeded
    gen_i = n.generators.index
    ext_i = n.generators.query("p_nom_extendable").index
    fix_i = n.generators.query("not p_nom_extendable").index

    dispatch = n.model["Generator-p"]
    reserve = n.model["Generator-r"]

    capacity_fixed = n.generators.p_nom[fix_i]

    p_max_pu = get_as_dense(n, "Generator", "p_max_pu")

    if not ext_i.empty:
        capacity_variable = n.model["Generator-p_nom"].rename(
            {"Generator-ext": "Generator"},
        )
        lhs = dispatch + reserve - capacity_variable * p_max_pu[ext_i]
    else:
        lhs = dispatch + reserve

    rhs = (p_max_pu[fix_i] * capacity_fixed).reindex(columns=gen_i, fill_value=0)

    n.model.add_constraints(lhs <= rhs, name="Generator-p-reserve-upper")


def add_battery_constraints(n):
    """
    Add constraint ensuring that charger = discharger, i.e.
    1 * charger_size - efficiency * discharger_size = 0
    """
    if not n.links.p_nom_extendable.any():
        return

    discharger_bool = n.links.index.str.contains("battery discharger")
    charger_bool = n.links.index.str.contains("battery charger")

    dischargers_ext = n.links[discharger_bool].query("p_nom_extendable").index
    chargers_ext = n.links[charger_bool].query("p_nom_extendable").index

    eff = n.links.efficiency[dischargers_ext].values
    lhs = n.model["Link-p_nom"].loc[chargers_ext] - n.model["Link-p_nom"].loc[dischargers_ext] * eff

    n.model.add_constraints(lhs == 0, name="Link-charger_ratio")


def add_pipe_retrofit_constraint(n):
    """
    Add constraint for retrofitting existing CH4 pipelines to H2 pipelines.
    """
    gas_pipes_i = n.links.query("carrier == 'gas pipeline' and p_nom_extendable").index
    h2_retrofitted_i = n.links.query(
        "carrier == 'H2 pipeline retrofitted' and p_nom_extendable",
    ).index

    if h2_retrofitted_i.empty or gas_pipes_i.empty:
        return

    p_nom = n.model["Link-p_nom"]

    CH4_per_H2 = 1 / n.config["sector"]["H2_retrofit_capacity_per_CH4"]
    lhs = p_nom.loc[gas_pipes_i] + CH4_per_H2 * p_nom.loc[h2_retrofitted_i]
    rhs = n.links.p_nom[gas_pipes_i].rename_axis("Link-ext")

    n.model.add_constraints(lhs == rhs, name="Link-pipe_retrofit")


def add_sector_co2_constraints(n, config):
    """
    Adds sector co2 constraints.

    Parameters
    ----------
        n : pypsa.Network
        config : dict
    """

    def apply_total_state_limit(n, year, state, value):

        sns = n.snapshots
        snapshot = sns[sns.get_level_values("period") == year][-1]

        stores = n.stores[
            (n.stores.index.str.startswith(state))
            & ((n.stores.index.str.endswith("-co2")) | (n.stores.index.str.endswith("-ch4")))
        ].index

        lhs = n.model["Store-e"].loc[snapshot, stores].sum()

        rhs = value  # value in T CO2

        n.model.add_constraints(lhs <= rhs, name=f"co2_limit-{year}-{state}")

        logger.info(
            f"Adding {state} co2 Limit in {year} of {rhs* 1e-6} MMT CO2",
        )

    def apply_sector_state_limit(n, year, state, sector, value):

        sns = n.snapshots
        snapshot = sns[sns.get_level_values("period") == year][-1]

        stores = n.stores[
            (n.stores.index.str.startswith(state))
            & ((n.stores.index.str.endswith(f"{sector}-co2")) | (n.stores.index.str.endswith(f"{sector}-ch4")))
        ].index

        lhs = n.model["Store-e"].loc[snapshot, stores].sum()

        rhs = value  # value in T CO2

        n.model.add_constraints(lhs <= rhs, name=f"co2_limit-{year}-{state}-{sector}")

        logger.info(
            f"Adding {state} co2 Limit for {sector} in {year} of {rhs* 1e-6} MMT CO2",
        )

    def apply_total_national_limit(n, year, value):

        sns = n.snapshots
        snapshot = sns[sns.get_level_values("period") == year][-1]

        stores = n.stores[((n.stores.index.str.endswith("-co2")) | (n.stores.index.str.endswith("-ch4")))].index

        lhs = n.model["Store-e"].loc[snapshot, stores].sum()

        rhs = value  # value in T CO2

        n.model.add_constraints(lhs <= rhs, name=f"co2_limit-{year}")

        logger.info(
            f"Adding national co2 Limit in {year} of {rhs* 1e-6} MMT CO2",
        )

    def apply_sector_national_limit(n, year, sector, value):

        sns = n.snapshots
        snapshot = sns[sns.get_level_values("period") == year][-1]

        stores = n.stores[
            (n.stores.index.str.endswith(f"{sector}-co2")) | (n.stores.index.str.endswith(f"{sector}-ch4"))
        ].index

        lhs = n.model["Store-e"].loc[snapshot, stores].sum()

        rhs = value  # value in T CO2

        n.model.add_constraints(lhs <= rhs, name=f"co2_limit-{year}-{sector}")

        logger.info(
            f"Adding national co2 Limit for {sector} sector in {year} of {rhs* 1e-6} MMT CO2",
        )

    try:
        f = config["sector"]["co2"]["policy"]
    except KeyError:
        logger.error("No co2 policy constraint file found")
        return

    df = pd.read_csv(f)

    if df.empty:
        logger.warning("No co2 policies applied")
        return

    sectors = df.sector.unique()

    for sector in sectors:

        df_sector = df[df.sector == sector]
        states = df_sector.state.unique()

        for state in states:

            df_state = df_sector[df_sector.state == state]
            years = [x for x in df_state.year.unique() if x in n.investment_periods]

            if not years:
                logger.warning(f"No co2 policies applied for {sector} in {year}")
                continue

            for year in years:

                df_limit = df_state[df_state.year == year].reset_index(drop=True)
                assert df_limit.shape[0] == 1

                # results calcualted in T CO2, policy given in MMT CO2
                value = df_limit.loc[0, "co2_limit_mmt"] * 1e6

                if state.upper() == "USA":

                    if sector == "all":
                        apply_total_national_limit(n, year, value)
                    else:
                        apply_sector_national_limit(n, year, sector, value)

                else:

                    if sector == "all":
                        apply_total_state_limit(n, year, state, value)
                    else:
                        apply_sector_state_limit(n, year, state, sector, value)


def add_ng_import_export_limits(n, config):

    def _format_link_name(s: str) -> str:
        states = s.split("-")
        return f"{states[0]} {states[1]} gas"

    def _format_domestic_data(
        prod: pd.DataFrame,
        link_suffix: Optional[str] = None,
    ) -> pd.DataFrame:

        df = prod.copy()
        df["link"] = df.state.map(_format_link_name)
        if link_suffix:
            df["link"] = df.link + link_suffix

        # convert mmcf to MWh
        df["value"] = df["value"] * 1000 / NG_MWH_2_MMCF

        return df[["link", "value"]].rename(columns={"value": "rhs"}).set_index("link")

    def _format_international_data(
        prod: pd.DataFrame,
        link_suffix: Optional[str] = None,
    ) -> pd.DataFrame:

        df = prod.copy()
        df = df[["value", "state"]].groupby("state", as_index=False).sum()
        df = df[~(df.state == "USA")].copy()

        df["link"] = df.state.map(_format_link_name)
        if link_suffix:
            df["link"] = df.link + link_suffix

        # convert mmcf to MWh
        df["value"] = df["value"] * 1000 / NG_MWH_2_MMCF

        return df[["link", "value"]].rename(columns={"value": "rhs"}).set_index("link")

    def add_import_limits(n, imports):
        """
        Sets gas import limit over each year.
        """

        weights = n.snapshot_weightings.objective

        links = n.links[n.links.carrier.str.endswith("gas import")].index.to_list()

        for year in n.investment_periods:
            for link in links:
                try:
                    rhs = imports.at[link, "rhs"]
                except KeyError:
                    # logger.warning(f"Can not set gas import limit for {link}")
                    continue
                lhs = n.model["Link-p"].mul(weights).sel(snapshot=year, Link=link).sum()

                n.model.add_constraints(lhs <= rhs, name=f"ng_limit-{year}-{link}")

    def add_export_limits(n, exports):
        """
        Sets maximum export limit over the year.
        """

        weights = n.snapshot_weightings.objective

        links = n.links[n.links.carrier.str.endswith("gas export")].index.to_list()

        for year in n.investment_periods:
            for link in links:
                try:
                    rhs = exports.at[link, "rhs"]
                except KeyError:
                    # logger.warning(f"Can not set gas import limit for {link}")
                    continue
                lhs = n.model["Link-p"].mul(weights).sel(snapshot=year, Link=link).sum()

                n.model.add_constraints(lhs >= rhs, name=f"ng_limit-{year}-{link}")

    api = config["api"]["eia"]
    year = pd.to_datetime(config["snapshots"]["start"]).year

    # add domestic limits

    imports = Trade("gas", False, "imports", year, api).get_data()
    imports = _format_domestic_data(imports, " import")
    exports = Trade("gas", False, "exports", year, api).get_data()
    exports = _format_domestic_data(exports, " export")

    # add_import_limits(n, imports)
    add_export_limits(n, exports)

    # add international limits

    imports = Trade("gas", True, "imports", year, api).get_data()
    imports = _format_international_data(imports, " import")
    exports = Trade("gas", True, "exports", year, api).get_data()
    exports = _format_international_data(exports, " export")

    # add_import_limits(n, imports)
    add_export_limits(n, exports)


def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to
    ``pypsa.optimization.optimize``.

    If you want to enforce additional custom constraints, this is a good
    location to add them. The arguments ``opts`` and
    ``snakemake.config`` are expected to be attached to the network.
    """
    opts = n.opts
    config = n.config
    if "RPS" in opts and n.generators.p_nom_extendable.any():
        add_RPS_constraints(n, config)
    if "REM" in opts and n.generators.p_nom_extendable.any():
        add_regional_co2limit(n, snapshots, config)
    if "BAU" in opts and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, config)
    if "SAFE" in opts and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, config)
    if "SAFER" in opts and n.generators.p_nom_extendable.any():
        add_SAFER_constraints(n, config)
    if "CCL" in opts and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, config)
    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)
    interface_limits = config["lines"].get("interface_transmission_limits", {})
    if interface_limits:
        add_interface_limits(n, snapshots, config)
    if "sector" in opts:
        sector_co2_limits = config["sector"]["co2"].get("policy", {})
        if sector_co2_limits:
            # add_sector_co2_constraints(n, config)
            pass
        if config["sector"]["natural_gas"].get("force_imports_exports", False):
            add_ng_import_export_limits(n, config)

    for o in opts:
        if "EQ" in o:
            add_EQ_constraints(n, o)
    add_battery_constraints(n)
    add_pipe_retrofit_constraint(n)


def solve_network(n, config, solving, opts="", **kwargs):
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    if len(n.investment_periods) > 1:
        kwargs["multi_investment_periods"] = config["foresight"] == "perfect"

    kwargs["solver_options"] = solving["solver_options"][set_of_options] if set_of_options else {}
    kwargs["solver_name"] = solving["solver"]["name"]
    kwargs["extra_functionality"] = extra_functionality
    kwargs["transmission_losses"] = cf_solving.get("transmission_losses", False)
    kwargs["linearized_unit_commitment"] = cf_solving.get(
        "linearized_unit_commitment",
        False,
    )
    kwargs["assign_all_duals"] = cf_solving.get("assign_all_duals", False)

    rolling_horizon = cf_solving.pop("rolling_horizon", False)
    skip_iterations = cf_solving.pop("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    if rolling_horizon:
        kwargs["horizon"] = cf_solving.get("horizon", 365)
        kwargs["overlap"] = cf_solving.get("overlap", 0)
        n.optimize.optimize_with_rolling_horizon(**kwargs)
        status, condition = "", ""
    elif skip_iterations:
        status, condition = n.optimize(**kwargs)
    else:
        kwargs["track_iterations"] = (cf_solving.get("track_iterations", False),)
        kwargs["min_iterations"] = (cf_solving.get("min_iterations", 4),)
        kwargs["max_iterations"] = (cf_solving.get("max_iterations", 6),)
        status, condition = n.optimize.optimize_transmission_expansion_iteratively(
            **kwargs,
        )

    if status != "ok" and not rolling_horizon:
        logger.warning(
            f"Solving status '{status}' with termination condition '{condition}'",
        )
    if "infeasible" in condition:
        raise RuntimeError("Solving status 'infeasible'")

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_network",
            simpl="12",
            opts="48SEG",
            clusters="6",
            ll="v1.0",
            sector_opts="",
            sector="E-G",
            planning_horizons="2030",
            interconnect="western",
        )
    configure_logging(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)
    if "sector_opts" in snakemake.wildcards.keys():
        update_config_with_sector_opts(
            snakemake.config,
            snakemake.wildcards.sector_opts,
        )

    opts = snakemake.wildcards.opts
    if "sector_opts" in snakemake.wildcards.keys():
        opts += "-" + snakemake.wildcards.sector_opts
    opts = [o for o in opts.split("-") if o != ""]
    solve_opts = snakemake.params.solving["options"]

    # sector specific co2 options
    if snakemake.wildcards.sector != "E":
        # sector co2 limits applied via config file, not through Co2L
        opts = [x for x in opts if not x.startswith("Co2L")]
        opts.append("sector")

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)

    n = prepare_network(
        n,
        solve_opts,
        config=snakemake.config,
        foresight=snakemake.params.foresight,
        planning_horizons=snakemake.params.planning_horizons,
        co2_sequestration_potential=snakemake.params["co2_sequestration_potential"],
    )

    n = solve_network(
        n,
        config=snakemake.config,
        solving=snakemake.params.solving,
        opts=opts,
        log_fn=snakemake.log.solver,
    )
    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])

    with open(snakemake.output.config, "w") as file:
        yaml.dump(
            n.meta,
            file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
