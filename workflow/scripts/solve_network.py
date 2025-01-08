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


def get_region_buses(n, region_list):
    return n.buses[
        (
            n.buses.country.isin(region_list)
            | n.buses.reeds_state.isin(region_list)
            | n.buses.interconnect.str.lower().isin(region_list)
            | n.buses.nerc_reg.isin(region_list)
            | (1 if "all" in region_list else 0)
        )
    ]


def filter_components(n, component_type, planning_horizon, carrier_list, region_buses, extendable):
    """
    Filter components based on common criteria.

    Parameters:
    - n: pypsa.Network
        The PyPSA network object.
    - component_type: str
        The type of component (e.g., "Generator", "StorageUnit").
    - planning_horizon: str or int
        The planning horizon to filter active assets.
    - carrier_list: list
        List of carriers to filter.
    - region_buses: pd.Index
        Index of region buses to filter.
    - extendable: bool, optional
        If specified, filters by extendable or non-extendable assets.

    Returns:
    - pd.DataFrame
        Filtered assets.
    """
    component = n.df(component_type)
    if planning_horizon != "all":
        ph = int(planning_horizon)
        iv = n.investment_periods
        active_components = n.get_active_assets(component.index.name, iv[iv >= ph][0])
    else:
        active_components = component.index

    filtered = component.loc[
        active_components
        & component.carrier.isin(carrier_list)
        & component.bus.isin(region_buses)
        & (component.p_nom_extendable == extendable)
    ]

    return filtered


def add_land_use_constraints(n):
    """
    Adds constraint for land-use based on information from the generators
    table.

    Constraint is defined by land-use per carrier and land_region. The
    definition of land_region enables sub-bus level land-use
    constraints.
    """
    model = n.model
    generators = n.generators.query("p_nom_extendable & land_region != '' ").rename_axis(index="Generator-ext")

    if generators.empty:
        return
    p_nom = n.model["Generator-p_nom"].loc[generators.index]

    grouper = pd.concat([generators.carrier, generators.land_region], axis=1)
    lhs = p_nom.groupby(grouper).sum()

    maximum = generators.groupby(["carrier", "land_region"])["p_nom_max"].max()
    maximum = maximum[np.isfinite(maximum)]

    rhs = xr.DataArray(maximum).rename(dim_0="group")
    index = rhs.indexes["group"].intersection(lhs.indexes["group"])

    if not index.empty:
        logger.info("Adding land-use constraints")
        model.add_constraints(lhs.sel(group=index) <= rhs.loc[index], name="land_use_constraint")


def prepare_network(
    n,
    solve_opts=None,
    config=None,
    foresight=None,
    planning_horizons=None,
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

    return n


def add_technology_capacity_target_constraints(n, config):
    """
    Add Technology Capacaity Target (TCT) constraint to the network.

    Add minimum or maximum levels of generator nominal capacity per carrier for individual regions. Each constraint can be designated for a specified planning horizon in multi-period models. Opts and path for technology_capacity_targets.csv must be defined in config.yaml. Default file is available at config/policy_constraints/technology_capacity_targets.csv.

    *** Review to subtract the non-extensible capacity from the rhs? ***

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-TCT-24H]
    electricity:
        technology_capacity_target: config/policy_constraints/technology_capacity_target.csv
    """
    p_nom = n.model["Generator-p_nom"]
    tct_data = pd.read_csv(config["electricity"]["technology_capacity_targets"])
    if tct_data.empty:
        return

    for idx, target in tct_data.iterrows():
        planning_horizon = target.planning_horizon
        region_list = [region_.strip() for region_ in target.region.split(",")]
        carrier_list = [carrier_.strip() for carrier_ in target.carrier.split(",")]
        region_buses = get_region_buses(n, region_list)

        lhs_gens = filter_components(
            n=n,
            component_type="Generator",
            planning_horizon=planning_horizon,
            carrier_list=carrier_list,
            region_buses=region_buses.index,
            extendable=True,
        )

        # rhs_g_non_extendable = filter_components(
        #     n=n,
        #     component_type="Generator",
        #     planning_horizon=planning_horizon,
        #     carrier_list=carrier_list,
        #     region_buses=region_buses.index,
        #     extendable=False,
        # ).p_nom.sum()

        lhs_storage = filter_components(
            n=n,
            component_type="StorageUnit",
            planning_horizon=planning_horizon,
            carrier_list=carrier_list,
            region_buses=region_buses.index,
            extendable=True,
        )

        # rhs_s_non_extendable = filter_components(
        #     n=n,
        #     component_type="StorageUnit",
        #     planning_horizon=planning_horizon,
        #     carrier_list=carrier_list,
        #     region_buses=region_buses.index,
        #     extendable=False,
        # ).p_nom.sum()

        if region_buses.empty or (lhs_gens.empty and lhs_storage.empty):
            continue

        if not lhs_gens.empty:
            grouper_g = pd.concat([lhs_gens.bus.map(n.buses.country), lhs_gens.carrier], axis=1).rename_axis(
                "Generator-ext",
            )
            lhs_g = p_nom.loc[lhs_gens.index].groupby(grouper_g).sum().rename(bus="country")
        else:
            lhs_g = None

        if not lhs_storage.empty:
            grouper_s = pd.concat([lhs_storage.bus.map(n.buses.country), lhs_storage.carrier], axis=1).rename_axis(
                "StorageUnit-ext",
            )
            lhs_s = n.model["StorageUnit-p_nom"].loc[lhs_storage.index].groupby(grouper_s).sum()
        else:
            lhs_s = None

        if lhs_g is None and lhs_s is None:
            continue
        elif lhs_g is None:
            lhs = lhs_s.sum()
        elif lhs_s is None:
            lhs = lhs_g.sum()
        else:
            lhs = (lhs_g + lhs_s).sum()

        lhs_extendable = lhs_gens.p_nom.sum() + lhs_storage.p_nom.sum()
        # rhs_non_extendable = rhs_g_non_extendable + rhs_s_non_extendable

        if target["max"] == "existing":
            target["max"] = lhs_extendable
        if target["min"] == "existing":
            target["min"] = lhs_extendable

        if not np.isnan(target["min"]):

            n.model.add_constraints(
                lhs >= (target["min"]),
                name=f"GlobalConstraint-{target.name}_{target.planning_horizon}_min",
            )
            logger.info(
                "Adding TCT Constraint:\n"
                "Name: %s\n"
                "Planning Horizon: %s\n"
                "Region: %s\n"
                "Carrier: %s\n"
                "Min Value: %s",
                target.name,
                target.planning_horizon,
                target.region,
                target.carrier,
                (target["min"]),
            )

        if not np.isnan(target["max"]):
            n.model.add_constraints(
                lhs <= (target["max"]),
                name=f"GlobalConstraint-{target.name}_{target.planning_horizon}_max",
            )
            logger.info(
                "Adding TCT Constraint:\n"
                "Name: %s\n"
                "Planning Horizon: %s\n"
                "Region: %s\n"
                "Carrier: %s\n"
                "Max Value: %s",
                target.name,
                target.planning_horizon,
                target.region,
                target.carrier,
                (target["max"]),
            )


def add_RPS_constraints(n, config):
    """
    Add Renewable Portfolio Standards (RPS) constraints to the network.

    This function enforces constraints on the percentage of electricity generation
    from renewable energy sources for specific regions and planning horizons.
    It reads the necessary data from configuration files and the network.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object.
    config : dict
        A dictionary containing configuration settings and file paths.

    Returns
    -------
    None
    """

    def process_reeds_data(filepath, carriers, value_col):
        """
        Helper function to process RPS or CES REEDS data.
        """
        reeds = pd.read_csv(filepath)

        # Handle both wide and long formats
        if "rps_all" not in reeds.columns:
            reeds = reeds.melt(id_vars="st", var_name="planning_horizon", value_name=value_col)

        # Standardize column names
        reeds.rename(columns={"st": "region", "t": "planning_horizon", "rps_all": "pct"}, inplace=True)
        reeds["carrier"] = [", ".join(carriers)] * len(reeds)

        # Extract and create new rows for `rps_solar` and `rps_wind`
        additional_rows = []
        for carrier_col, carrier_name in [("rps_solar", "solar"), ("rps_wind", "onwind, offwind, offwind_floating")]:
            if carrier_col in reeds.columns:
                temp = reeds[["region", "planning_horizon", carrier_col]].copy()
                temp.rename(columns={carrier_col: "pct"}, inplace=True)
                temp["carrier"] = carrier_name
                additional_rows.append(temp)

        # Combine original data with additional rows
        if additional_rows:
            additional_rows = pd.concat(additional_rows, ignore_index=True)
            reeds = pd.concat([reeds, additional_rows], ignore_index=True)

        # Ensure the final dataframe has consistent columns
        reeds = reeds[["region", "planning_horizon", "carrier", "pct"]]
        reeds = reeds[reeds["pct"] > 0.0]  # Remove any rows with zero or negative percentages

        return reeds

    # Read portfolio standards data
    portfolio_standards = pd.read_csv(config["electricity"]["portfolio_standards"])

    # Define carriers for RPS and CES
    rps_carriers = ["onwind", "offwind", "offwind_floating", "solar", "hydro", "geothermal", "biomass", "EGS"]
    ces_carriers = rps_carriers + ["nuclear", "SMR"]

    # Process RPS and CES REEDS data
    rps_reeds = process_reeds_data(
        snakemake.input.rps_reeds,
        rps_carriers,
        value_col="pct",
    )
    ces_reeds = process_reeds_data(
        snakemake.input.ces_reeds,
        ces_carriers,
        value_col="pct",
    )

    # Concatenate all portfolio standards
    portfolio_standards = pd.concat([portfolio_standards, rps_reeds, ces_reeds])
    portfolio_standards = portfolio_standards[
        (portfolio_standards.pct > 0.0)
        & (portfolio_standards.planning_horizon.isin(snakemake.params.planning_horizons))
        & (portfolio_standards.region.isin(n.buses.reeds_state.unique()))
    ]

    # Iterate through constraints and add RPS constraints to the model
    for _, constraint_row in portfolio_standards.iterrows():
        region_list = [region.strip() for region in constraint_row.region.split(",")]
        region_buses = get_region_buses(n, region_list)

        if region_buses.empty:
            continue

        carriers = [carrier.strip() for carrier in constraint_row.carrier.split(",")]

        # Filter region generators
        region_gens = n.generators[n.generators.bus.isin(region_buses.index)]
        region_gens_eligible = region_gens[region_gens.carrier.isin(carriers)]

        if not region_gens_eligible.empty:
            # Eligible generation
            p_eligible = n.model["Generator-p"].sel(
                period=constraint_row.planning_horizon,
                Generator=region_gens_eligible.index,
            )
            lhs = p_eligible.sum()

            # Region demand
            region_demand = (
                n.loads_t.p_set.loc[
                    constraint_row.planning_horizon,
                    n.loads.bus.isin(region_buses.index),
                ]
                .sum()
                .sum()
            )

            rhs = constraint_row.pct * region_demand

            # Add constraint
            n.model.add_constraints(
                lhs >= rhs,
                name=f"GlobalConstraint-{constraint_row.name}_{constraint_row.planning_horizon}_rps_limit",
            )
            logger.info(f"Added RPS {constraint_row.name} for {constraint_row.planning_horizon}.")


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

    regional_co2_lims = pd.read_csv(
        config["electricity"]["regional_Co2_limits"],
        index_col=[0],
    )
    logger.info("Adding regional Co2 Limits.")
    regional_co2_lims = regional_co2_lims[regional_co2_lims.planning_horizon.isin(snakemake.params.planning_horizons)]
    weightings = n.snapshot_weightings.loc[n.snapshots]

    for idx, emmission_lim in regional_co2_lims.iterrows():
        region_list = [region.strip() for region in emmission_lim.regions.split(",")]
        region_buses = get_region_buses(n, region_list)

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

        # EF_imports = emmission_lim.get('import_emissions_factor')  # MT CO₂e/MWh_elec
        # if EF_imports > 0.0:
        #     # emissions imported = EF_imports * imports
        #     # imports = Internal Demand - Internal Production
        #     # Emissions imported = EF_imports * Internal Demand - EF_imports * Internal Production

        #     # Full Constraint:
        #     # Emissions Produced + Emissions Imported <= Emissions Limit
        #     # Emissions Produced  - EF_imports * Internal Production  <= Emissions Limit - (EF_imports * Internal Demand)

        #     # Internal Production
        #     p_internal = (
        #         n.model["Generator-p"]
        #         .loc[:, region_gens.index]
        #         .sel(period=planning_horizon)
        #         .mul(weightings.generators.loc[planning_horizon])
        #     )
        #     lhs -= (p_internal * EF_imports).sum()

        #     region_storage = n.storage_units[n.storage_units.bus.isin(region_buses.index)]
        #     if not region_storage.empty:
        #         p_store_discharge = (
        #             n.model["StorageUnit-p_dispatch"]
        #             .loc[:, region_storage.index]
        #             .sel(period=planning_horizon)
        #             .mul(weightings.stores.loc[planning_horizon])
        #         )
        #         lhs -= (p_store_discharge * EF_imports).sum()

        #     # Internal Demand
        #     breakpoint()
        #     region_loads = n.loads[n.loads.bus.isin(region_buses.index)]
        #     region_demand = (
        #         n.loads_t.p_set.loc[planning_horizon,region_loads.index].sum().sum()
        #     )
        #     rhs -= (region_demand * EF_imports)

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
        region_buses = get_region_buses(n, region_list)

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
                logger.warning(
                    f"No co2 policies applied for {sector} due to no defined years",
                )
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


def add_cooling_heat_pump_constraints(n, config):
    """
    Adds constraints to the cooling heat pumps.

    These constraints allow HPs to be used to meet both heating and cooling
    demand within a single timeslice while respecting capacity limits.
    Since we are aggregating (and not modelling individual units)
    this should be fine.

    Two seperate constraints are added:
    - Constrains the cooling HP capacity to equal the heating HP capacity. Since the
    cooling hps do not have a capital cost, this will not effect objective cost
    - Constrains the total generation of Heating and Cooling HPs at each time slice
    to be less than or equal to the max generation of the heating HP. Note, that both
    the cooling and heating HPs have the same COP
    """

    def add_hp_capacity_constraint(n, hp_type):

        assert hp_type in ("ashp", "gshp")

        heating_hps = n.links[n.links.index.str.endswith(hp_type)].index
        if heating_hps.empty:
            return
        cooling_hps = n.links[n.links.index.str.endswith(f"{hp_type}-cool")].index

        assert len(heating_hps) == len(cooling_hps)

        lhs = n.model["Link-p_nom"].loc[heating_hps] - n.model["Link-p_nom"].loc[cooling_hps]
        rhs = 0

        n.model.add_constraints(lhs == rhs, name=f"Link-{hp_type}_cooling_capacity")

    def add_hp_generation_constraint(n, hp_type):

        heating_hps = n.links[n.links.index.str.endswith(hp_type)].index
        if heating_hps.empty:
            return
        cooling_hps = n.links[n.links.index.str.endswith(f"{hp_type}-cooling")].index

        heating_hp_p = n.model["Link-p"].loc[:, heating_hps]
        cooling_hp_p = n.model["Link-p"].loc[:, cooling_hps]

        heating_hps_cop = n.links_t["efficiency"][heating_hps]
        cooling_hps_cop = n.links_t["efficiency"][cooling_hps]

        heating_hps_gen = heating_hp_p.mul(heating_hps_cop)
        cooling_hps_gen = cooling_hp_p.mul(cooling_hps_cop)

        lhs = heating_hps_gen + cooling_hps_gen

        heating_hp_p_nom = n.model["Link-p_nom"].loc[heating_hps]
        max_gen = heating_hp_p_nom.mul(heating_hps_cop)

        rhs = max_gen

        n.model.add_constraints(lhs <= rhs, name=f"Link-{hp_type}_cooling_generation")

    for hp_type in ("ashp", "gshp"):
        add_hp_capacity_constraint(n, hp_type)
        add_hp_generation_constraint(n, hp_type)


def add_gshp_capacity_constraint(n, config):
    """
    Constrains gshp capacity based on population and ashp installations.

    This constraint should be added if rural/urban sectors are combined into
    a single total area. In this case, we need to constrain how much gshp capacity
    can be added to the system.

    For example:
    - If ratio is 0.75 urban and 0.25 rural
    - We want to enforce that at max, only 0.33 unit of GSHP can be installed for every unit of ASHP
    - The constraint is: [ASHP - (urban / rural) * GSHP >= 0]
    - ie. for every unit of GSHP, we need to install 3 units of ASHP
    """

    pop = pd.read_csv(snakemake.input.pop_layout)
    pop["urban_rural_fraction"] = (pop.urban_fraction / pop.rural_fraction).round(2)
    fraction = pop.set_index("name")["urban_rural_fraction"].to_dict()

    ashp = n.links[n.links.index.str.endswith("ashp")].copy()
    gshp = n.links[n.links.index.str.endswith("gshp")].copy()
    if gshp.empty:
        return

    assert len(ashp) == len(gshp)

    gshp["urban_rural_fraction"] = gshp.bus0.map(fraction)

    ashp_capacity = n.model["Link-p_nom"].loc[ashp.index]
    gshp_capacity = n.model["Link-p_nom"].loc[gshp.index]
    gshp_multiplier = gshp["urban_rural_fraction"]

    lhs = ashp_capacity - gshp_capacity.mul(gshp_multiplier.values)
    rhs = 0

    n.model.add_constraints(lhs >= rhs, name=f"Link-gshp_capacity_ratio")


def add_ng_import_export_limits(n, config):

    def _format_link_name(s: str) -> str:
        states = s.split("-")
        return f"{states[0]} {states[1]} gas"

    def _format_data(
        prod: pd.DataFrame,
        link_suffix: Optional[str] = None,
    ) -> pd.DataFrame:

        df = prod.copy()
        df["link"] = df.state.map(_format_link_name)
        if link_suffix:
            df["link"] = df.link + link_suffix

        # convert mmcf to MWh
        df["value"] = df["value"] * NG_MWH_2_MMCF

        return df[["link", "value"]].rename(columns={"value": "rhs"}).set_index("link")

    def add_import_limits(n, data, constraint, multiplier=None):
        """
        Sets gas import limit over each year.
        """

        assert constraint in ("max", "min")

        if not multiplier:
            multiplier = 1

        weights = n.snapshot_weightings.objective

        links = n.links[(n.links.carrier == "gas trade") & (n.links.bus0.str.endswith(" gas trade"))].index.to_list()

        for year in n.investment_periods:
            for link in links:
                try:
                    rhs = data.at[link, "rhs"] * multiplier
                except KeyError:
                    # logger.warning(f"Can not set gas import limit for {link}")
                    continue
                lhs = n.model["Link-p"].mul(weights).sel(snapshot=year, Link=link).sum()

                if constraint == "min":
                    n.model.add_constraints(
                        lhs >= rhs,
                        name=f"ng_limit_import_min-{year}-{link}",
                    )
                else:
                    n.model.add_constraints(
                        lhs <= rhs,
                        name=f"ng_limit_import_max-{year}-{link}",
                    )

    def add_export_limits(n, data, constraint, multiplier=None):
        """
        Sets maximum export limit over the year.
        """

        assert constraint in ("max", "min")

        if not multiplier:
            multiplier = 1

        weights = n.snapshot_weightings.objective

        links = n.links[(n.links.carrier == "gas trade") & (n.links.bus0.str.endswith(" gas"))].index.to_list()

        for year in n.investment_periods:
            for link in links:
                try:
                    rhs = data.at[link, "rhs"] * multiplier
                except KeyError:
                    # logger.warning(f"Can not set gas import limit for {link}")
                    continue
                lhs = n.model["Link-p"].mul(weights).sel(snapshot=year, Link=link).sum()

                if constraint == "min":
                    n.model.add_constraints(
                        lhs >= rhs,
                        name=f"ng_limit_export_min-{year}-{link}",
                    )
                else:
                    n.model.add_constraints(
                        lhs <= rhs,
                        name=f"ng_limit_export_max-{year}-{link}",
                    )

    api = config["api"]["eia"]
    year = pd.to_datetime(config["snapshots"]["start"]).year

    # get limits

    import_min = config["sector"]["natural_gas"]["imports"].get("min", 1)
    import_max = config["sector"]["natural_gas"]["imports"].get("max", 1)
    export_min = config["sector"]["natural_gas"]["exports"].get("min", 1)
    export_max = config["sector"]["natural_gas"]["exports"].get("max", 1)

    # to avoid numerical issues, ensure there is a gap between min/max constraints
    if import_max == "inf":
        pass
    elif abs(import_max - import_min) < 0.0001:
        import_min -= 0.001
        import_max += 0.001
        if import_min < 0:
            import_min = 0

    if export_max == "inf":
        pass
    elif abs(export_max - export_min) < 0.0001:
        export_min -= 0.001
        export_max += 0.001
        if export_min < 0:
            export_min = 0

    # import and export dataframes contain the same information, just in different formats
    # ie. imports from one S1 -> S2 are the same as exports from S2 -> S1
    # we use the exports direction to set limits

    # add domestic limits

    trade = Trade("gas", False, "exports", year, api).get_data()
    trade = _format_data(trade, " trade")

    add_import_limits(n, trade, "min", import_min)
    add_export_limits(n, trade, "min", export_min)

    if not import_max == "inf":
        add_import_limits(n, trade, "max", import_max)
    if not export_max == "inf":
        add_export_limits(n, trade, "max", export_max)

    # add international limits

    trade = Trade("gas", True, "exports", year, api).get_data()
    trade = _format_data(trade, " trade")

    add_import_limits(n, trade, "min", import_min)
    add_export_limits(n, trade, "min", export_min)

    if not import_max == "inf":
        add_import_limits(n, trade, "max", import_max)
    if not export_max == "inf":
        add_export_limits(n, trade, "max", export_max)


def add_water_heater_constraints(n, config):
    """
    Adds constraint so energy to meet water demand must flow through store.
    """

    links = n.links[(n.links.index.str.contains("-water-")) & (n.links.index.str.contains("-discharger"))]

    link_names = links.index
    store_names = [x.replace("-discharger", "") for x in links.index]

    for period in n.investment_periods:

        # first snapshot does not respect constraint
        e_previous = n.model["Store-e"].loc[period, store_names]
        e_previous = e_previous.roll(timestep=1)
        e_previous = e_previous.mul(n.snapshot_weightings.stores.loc[period])

        p_current = n.model["Link-p"].loc[period, link_names]
        p_current = p_current.mul(n.snapshot_weightings.objective.loc[period])

        lhs = e_previous - p_current
        rhs = 0

        n.model.add_constraints(lhs >= rhs, name=f"water_heater-{period}")


def add_demand_response_constraints(n, config):
    """
    Adds demand response equations

    Applies the p_nom_max here, as easier to iterate over different configs
    """

    def add_capacity_constraint(
        n: pypsa.Network,
        sector: str,
        shift: float,  # as a percentage
    ):
        """Adds limit on deferable load

        No need to multiply out snapshot weights here
        """

        dr_links = n.links[n.links.carrier.str.endswith("-dr") & n.links.carrier.str.startswith(f"{sector}-")]

        if dr_links.empty:
            return

        if sector != "trn":

            deferrable_links = dr_links[dr_links.index.str.endswith("-dr-discharger")]

            deferrable_loads = deferrable_links.bus1.unique().tolist()

            lhs = n.model["Link-p"].loc[:, deferrable_links.index].groupby(deferrable_links.bus1).sum()
            rhs = n.loads_t["p_set"][deferrable_loads].mul(shift).div(100).round(2)  # div cause percentage input
            rhs.columns.name = "bus1"

            # force rhs to be same order as lhs
            # idk why but coordinates were not aligning and this gets around that
            bus_order = lhs.vars.bus1.data
            rhs = rhs[bus_order]

            n.model.add_constraints(lhs <= rhs, name=f"demand_response_capacity-{sector}")

        # transport dr is at the aggregation bus
        # sum all outgoing capacity and apply the capacity limit to that
        else:

            inflow_links = dr_links[dr_links.index.str.endswith("-dr-discharger")]
            inflow = n.model["Link-p"].loc[:, inflow_links.index].groupby(inflow_links.bus1).sum()
            inflow = inflow.rename({"bus1": "Bus"})  # align coordinate names

            outflow_links = n.links[n.links.bus0.isin(inflow_links.bus1) & ~n.links.carrier.str.endswith("-dr")]
            outflow = n.model["Link-p"].loc[:, outflow_links.index].groupby(outflow_links.bus0).sum()
            outflow = outflow.rename({"bus0": "Bus"})  # align coordinate names

            lhs = outflow.mul(shift).div(100) - inflow
            rhs = 0

            n.model.add_constraints(
                lhs >= rhs,
                name=f"demand_response_capacity-trn",
            )

    # demand response addition starts here

    sectors = ["res", "com", "ind", "trn"]

    for sector in sectors:

        if sector in ["res", "com"]:
            dr_config = config["sector"]["service_sector"].get("demand_response", {})
        elif sector == "trn":
            dr_config = config["sector"]["transport_sector"].get("demand_response", {})
        elif sector == "ind":
            dr_config = config["sector"]["industrial_sector"].get("demand_response", {})
        else:
            raise ValueError

        shift = dr_config.get("shift", 0)

        if not dr_config or shift < 0.001:
            logger.info(f"Demand response not enabled for {sector}")
            continue

        # capacity constraint

        if shift == "inf":
            pass
        elif shift >= 0.001:  # for tolerance
            add_capacity_constraint(n, sector, shift)
        else:
            logger.info(f"Unknown arguement of {shift} for {sector} DR")
            raise ValueError(shift)


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
    if "TCT" in opts and n.generators.p_nom_extendable.any():
        add_technology_capacity_target_constraints(n, config)
    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)
    interface_limits = config["lines"].get("interface_transmission_limits", {})
    if interface_limits:
        add_interface_limits(n, snapshots, config)
    if "sector" in opts:
        add_cooling_heat_pump_constraints(n, config)
        if not config["sector"]["service_sector"].get("split_urban_rural", False):
            add_gshp_capacity_constraint(n, config)
        if config["sector"]["co2"].get("policy", {}):
            add_sector_co2_constraints(n, config)
        if config["sector"]["natural_gas"].get("imports", False):
            add_ng_import_export_limits(n, config)
        water_config = config["sector"]["service_sector"].get("water_heating", {})
        if not water_config.get("simple_storage", True):
            add_water_heater_constraints(n, config)
        add_demand_response_constraints(n, config)

    for o in opts:
        if "EQ" in o:
            add_EQ_constraints(n, o)
    add_battery_constraints(n)
    add_land_use_constraints(n)


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
        # n.model.print_infeasibilities()
        raise RuntimeError("Solving status 'infeasible'")

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        print("")

        snakemake = mock_snakemake(
            "solve_network",
            interconnect="western",
            simpl="70",
            clusters="4m",
            ll="v1.0",
            opts="3h",
            sector="E-G",
            planning_horizons="2019",
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
