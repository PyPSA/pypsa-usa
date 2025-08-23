import logging  # noqa: D100

import numpy as np
import pandas as pd
import pypsa
from opts._helpers import (
    ceil_precision,
    filter_components,
    floor_precision,
    get_model_horizon,
    get_region_buses,
)
from pypsa.descriptors import get_switchable_as_dense as get_as_dense

logger = logging.getLogger(__name__)

RPS_CARRIERS = [
    "onwind",
    "offwind",
    "offwind_floating",
    "solar",
    "hydro",
    "geothermal",
    "biomass",
    "EGS",
]
CES_CARRIERS = [*RPS_CARRIERS, "nuclear", "SMR", "hydrogen_ct", "CCGT-95CCS", "CCGT-99CCS", "Coal-95CCS"]


def add_technology_capacity_target_constraints(n, config):
    """
    Add Technology Capacity Target (TCT) constraint to the network.

    Add minimum or maximum levels of generator nominal capacity per carrier for individual regions.
    Each constraint can be designated for a specified planning horizon in multi-period models.
    Opts and path for technology_capacity_targets.csv must be defined in config.yaml.
    Default file is available at config/policy_constraints/technology_capacity_targets.csv.

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
    tct_data = pd.read_csv(config["electricity"]["technology_capacity_targets"])
    if tct_data.empty:
        return

    model_horizon = get_model_horizon(n.model)

    for _, target in tct_data.iterrows():
        planning_horizon = target.planning_horizon
        if planning_horizon != "all" and int(planning_horizon) > max(model_horizon):
            continue

        region_list = [region_.strip() for region_ in target.region.split(",")]
        carrier_list = [carrier_.strip() for carrier_ in target.carrier.split(",")]
        region_buses = get_region_buses(n, region_list)

        lhs_gens_ext = filter_components(
            n=n,
            component_type="Generator",
            planning_horizon=planning_horizon,
            carrier_list=carrier_list,
            region_buses=region_buses.index,
            extendable=True,
        )
        lhs_gens_existing = filter_components(
            n=n,
            component_type="Generator",
            planning_horizon=planning_horizon,
            carrier_list=carrier_list,
            region_buses=region_buses.index,
            extendable=False,
        )

        lhs_storage_ext = filter_components(
            n=n,
            component_type="StorageUnit",
            planning_horizon=planning_horizon,
            carrier_list=carrier_list,
            region_buses=region_buses.index,
            extendable=True,
        )
        lhs_storage_existing = filter_components(
            n=n,
            component_type="StorageUnit",
            planning_horizon=planning_horizon,
            carrier_list=carrier_list,
            region_buses=region_buses.index,
            extendable=False,
        )

        lhs_link_ext = filter_components(
            n=n,
            component_type="Link",
            planning_horizon=planning_horizon,
            carrier_list=carrier_list,
            region_buses=region_buses.index,
            extendable=True,
        )
        lhs_link_existing = filter_components(
            n=n,
            component_type="Link",
            planning_horizon=planning_horizon,
            carrier_list=carrier_list,
            region_buses=region_buses.index,
            extendable=False,
        )

        if region_buses.empty or (lhs_gens_ext.empty and lhs_storage_ext.empty and lhs_link_ext.empty):
            continue

        if not lhs_gens_ext.empty:
            grouper_g = pd.concat(
                [lhs_gens_ext.bus.map(n.buses.country), lhs_gens_ext.carrier],
                axis=1,
            ).rename_axis(
                "Generator-ext",
            )
            lhs_g = n.model["Generator-p_nom"].loc[lhs_gens_ext.index].groupby(grouper_g).sum().rename(bus="country")
        else:
            lhs_g = None

        if not lhs_storage_ext.empty:
            grouper_s = pd.concat(
                [lhs_storage_ext.bus.map(n.buses.country), lhs_storage_ext.carrier],
                axis=1,
            ).rename_axis(
                "StorageUnit-ext",
            )
            lhs_s = n.model["StorageUnit-p_nom"].loc[lhs_storage_ext.index].groupby(grouper_s).sum()
        else:
            lhs_s = None

        if not lhs_link_ext.empty:
            grouper_l = pd.concat(
                [lhs_link_ext.bus.map(n.buses.country), lhs_link_ext.carrier],
                axis=1,
            ).rename_axis(
                "Link-ext",
            )
            lhs_l = n.model["Link-p_nom"].loc[lhs_link_ext.index].groupby(grouper_l).sum()
        else:
            lhs_l = None

        if lhs_g is None and lhs_s is None and lhs_l is None:
            continue
        else:
            gen = lhs_g.sum() if lhs_g else 0
            lnk = lhs_l.sum() if lhs_l else 0
            sto = lhs_s.sum() if lhs_s else 0

        lhs = gen + lnk + sto

        lhs_existing = lhs_gens_existing.p_nom.sum() + lhs_storage_existing.p_nom.sum() + lhs_link_existing.p_nom.sum()

        if target["max"] == "existing":
            target["max"] = ceil_precision(lhs_existing, 2)
        else:
            target["max"] = float(target["max"])

        if target["min"] == "existing":
            target["min"] = floor_precision(lhs_existing, 2)
        else:
            target["min"] = float(target["min"])

        if not np.isnan(target["min"]):
            rhs = floor_precision(target["min"] - lhs_existing, 2)

            n.model.add_constraints(
                lhs >= rhs,
                name=f"GlobalConstraint-{target.name}_{target.planning_horizon}_min",
            )

            logger.info(
                f"Adding TCT Constraint: Name: {target.name}, Planning Horizon: {target.planning_horizon}, Region: {target.region}, Carrier: {target.carrier}, Min Value: {target['min']}, Min Value Adj: {rhs}",
            )

        if not np.isnan(target["max"]):
            assert target["max"] >= lhs_existing, (
                f"TCT constraint of {target['max']} MW for {target['carrier']} must be at least {lhs_existing}"
            )

            rhs = ceil_precision(target["max"] - lhs_existing, 2)

            n.model.add_constraints(
                lhs <= rhs,
                name=f"GlobalConstraint-{target.name}_{target.planning_horizon}_max",
            )

            logger.info(
                f"Adding TCT Constraint: Name: {target.name}, Planning Horizon: {target.planning_horizon}, Region: {target.region}, Carrier: {target.carrier}, Max Value: {target['max']}, Max Value Adj: {rhs}",
            )


def _process_reeds_data(filepath, carriers, value_col):
    """Helper function to process RPS or CES REEDS data."""
    reeds = pd.read_csv(filepath)

    # Handle both wide and long formats
    if "rps_all" not in reeds.columns:
        reeds = reeds.melt(
            id_vars="st",
            var_name="planning_horizon",
            value_name=value_col,
        )

    # Standardize column names
    reeds = reeds.rename(
        columns={"st": "region", "t": "planning_horizon", "rps_all": "pct"},
    )
    reeds["carrier"] = [", ".join(carriers)] * len(reeds)

    # Ensure the final dataframe has consistent columns
    reeds = reeds[["region", "planning_horizon", "carrier", "pct"]]
    reeds = reeds[reeds["pct"] > 0.0]  # Remove any rows with zero or negative percentages

    return reeds


def _collapse_portfolio_standards(n: pypsa.Network, planning_horizons: list[int], *args):
    """Collapse portfolio standards into a single row per region, planning horizon, and carrier."""
    expected_columns = ["region", "planning_horizon", "carrier", "pct"]
    dfs = [df[expected_columns] for df in args]
    portfolio_standards = pd.concat(dfs)

    portfolio_standards = portfolio_standards[
        (portfolio_standards.pct > 0.0)
        & (
            portfolio_standards.planning_horizon.isin(
                planning_horizons,
            )
        )
        & (portfolio_standards.region.isin(n.buses.reeds_state.unique()))
    ]

    mapper = n.buses.groupby("reeds_state")["rec_trading_zone"].first().to_dict()
    portfolio_standards["rec_trading_zone"] = portfolio_standards.region.map(mapper).fillna(portfolio_standards.region)

    return portfolio_standards


def add_RPS_constraints(n, config, snakemake=None):
    """
    Add Renewable Portfolio Standards (RPS) constraints to the network.

    This function enforces constraints on the percentage of electricity generation
    from renewable energy sources for specific regions and planning horizons.
    It reads the necessary data from configuration files and the network.

    The differenct between electrical and sector implementation is:
    - Electrical applies RPS against exogenously defined demand
    - Sector applies RPS against endogenously solved power sector generation

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object.
    config : dict
        A dictionary containing configuration settings and file paths.
    snakemake: object, optional
        Snakemake object containing inputs and parameters

    Returns
    -------
    None
    """
    # Get model horizon
    model_horizon = get_model_horizon(n.model)

    # Read portfolio standards data
    portfolio_standards = pd.read_csv(config["electricity"]["portfolio_standards"])

    # Process RPS and CES REEDS data
    rps_reeds = _process_reeds_data(
        snakemake.input.rps_reeds,
        RPS_CARRIERS,
        value_col="pct",
    )
    ces_reeds = _process_reeds_data(
        snakemake.input.ces_reeds,
        CES_CARRIERS,
        value_col="pct",
    )

    # Concatenate all portfolio standards
    portfolio_standards = _collapse_portfolio_standards(
        n,
        snakemake.params.planning_horizons,
        portfolio_standards,
        rps_reeds,
        ces_reeds,
    )

    for _, constraint_row in portfolio_standards.iterrows():
        region_list = [region.strip() for region in constraint_row.region.split(",")]
        region_buses = get_region_buses(n, region_list)
        if region_buses.empty:
            continue

        region_demand = (
            n.loads_t.p_set.loc[constraint_row.planning_horizon]
            .loc[:, n.loads.bus.isin(region_buses.index)]
            .sum()
            .sum()
        )
        region_rps_rhs = int(constraint_row.pct * region_demand)
        portfolio_standards.loc[constraint_row.name, "rps_rhs"] = region_rps_rhs

    # Iterate through constraints and add RPS constraints to the model
    for (rec_trading_zone, planning_horizon, policy_carriers), zone_constraints in portfolio_standards.groupby(
        ["rec_trading_zone", "planning_horizon", "carrier"],
    ):
        if planning_horizon not in model_horizon:
            continue
        region_buses = get_region_buses(n, zone_constraints.region.unique())
        carriers = [carrier.strip() for carrier in policy_carriers.split(",")]

        # Filter region generators
        region_gens = n.generators[n.generators.bus.isin(region_buses.index)]
        region_gens_eligible = region_gens[region_gens.carrier.isin(carriers)]

        if region_gens_eligible.empty:
            return

        # Eligible generation
        p_eligible = n.model["Generator-p"].sel(
            period=planning_horizon,
            Generator=region_gens_eligible.index,
        )
        renewable_gen = zone_constraints.rps_rhs.sum()
        lhs = p_eligible.sum() - renewable_gen
        rhs = 0

        n.model.add_constraints(
            lhs >= rhs,
            name=f"GlobalConstraint-{rec_trading_zone}_{planning_horizon}_rps_limit",
        )

        logger.info(
            f"Added RPS constraint '{rec_trading_zone}' for {planning_horizon} "
            f"requiring {renewable_gen / 1e6:.1f} TWh of {policy_carriers} generation ",
        )


def _get_state_generation(n, planning_horizon, state, carriers):
    """Generation of supply side technologies excluding trade."""
    state_buses = n.buses[(n.buses.reeds_state == state) & (n.buses.carrier == "AC")]
    state_gens = n.generators[n.generators.bus.isin(state_buses.index) & n.generators.carrier.isin(carriers)]
    state_links = n.links[n.links.bus1.isin(state_buses.index) & n.links.carrier.isin(carriers)]

    gens_demand = (
        n.model["Generator-p"]
        .sel(
            period=planning_horizon,
            Generator=state_gens.index,
        )
        .sum()
    )
    links_demand = (
        n.model["Link-p"].sel(period=planning_horizon, Link=state_links.index).mul(state_links.efficiency).sum()
    )

    return gens_demand + links_demand


def add_RPS_constraints_sector(n, config, snakemake=None):
    """Add RPS constraints to the network for sector studies.

    This function enforces constraints on the percentage of electricity generation
    from renewable energy sources for specific regions and planning horizons.
    It reads the necessary data from configuration files and the network.

    The differenct between electrical and sector implementation is:
    - Electrical applies RPS against exogenously defined demand
    - Sector applies RPS against endogenously solved power sector generation as final
    demand is not exogenously availabele.
    """
    # Get model horizon
    model_horizon = get_model_horizon(n.model)

    # Read portfolio standards data
    portfolio_standards = pd.read_csv(f"../{config['electricity']['portfolio_standards']}")

    # Process RPS and CES REEDS data
    rps_reeds = _process_reeds_data(
        snakemake.input.rps_reeds,
        RPS_CARRIERS,
        value_col="pct",
    )
    ces_reeds = _process_reeds_data(
        snakemake.input.ces_reeds,
        CES_CARRIERS,
        value_col="pct",
    )

    # Concatenate all portfolio standards
    portfolio_standards = _collapse_portfolio_standards(
        n,
        snakemake.params.planning_horizons,
        portfolio_standards,
        rps_reeds,
        ces_reeds,
    )

    # get all genertion carriers
    all_carriers = list(
        set(config["electricity"].get("conventional_carriers", []))
        | set(config["electricity"].get("renewable_carriers", []))
        | set(config["electricity"].get("extendable_carriers", {}).get("Generator", [])),
    )
    # Iterate through constraints and add RPS constraints to the model
    for rec_trading_zone in portfolio_standards.rec_trading_zone.unique():
        rtz = portfolio_standards[portfolio_standards.rec_trading_zone == rec_trading_zone]

        for planning_horizon in rtz.planning_horizon.unique():
            # only add constraints for planning horizons in the model horizon
            if planning_horizon not in model_horizon:
                continue

            rtz_planning_horizon = rtz[rtz.planning_horizon == planning_horizon]

            for policy_carriers in rtz_planning_horizon.carrier.unique():
                carriers = [x.strip() for x in policy_carriers.split(",")]

                policy = rtz_planning_horizon[rtz_planning_horizon.carrier == policy_carriers]

                # total supply side demand in the rec zone scaled by state level rps
                demands = []  # linopy sums
                for state, rps in zip(policy.region, policy.pct):
                    demand = _get_state_generation(n, planning_horizon, state, all_carriers)
                    demands.append(demand * rps)
                rps_required_generation = sum(demands)

                # rps eligible generation in the rec zone
                generations = []  # linopy sums
                for state in policy.region.unique():
                    generations.append(_get_state_generation(n, planning_horizon, state, carriers))
                rps_actual_generation = sum(generations)

                lhs = rps_actual_generation - rps_required_generation
                rhs = 0

                # add constraint
                carrier_name = "-".join(carriers)
                n.model.add_constraints(
                    lhs >= rhs,
                    name=f"GlobalConstraint-{rec_trading_zone}_{planning_horizon}_{carrier_name}_limit",
                )
                logger.info(
                    f"Added {rec_trading_zone} for {planning_horizon} for carriers {carrier_name}.",
                )


def add_regional_co2limit(n, config):
    """Adding regional regional CO2 Limits Specified in the config.yaml."""
    model_horizon = get_model_horizon(n.model)
    regional_co2_lims = pd.read_csv(
        config["electricity"]["regional_Co2_limits"],
        index_col=[0],
    )

    regional_co2_lims = regional_co2_lims[regional_co2_lims.planning_horizon.isin(n.investment_periods)]
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
        if planning_horizon not in model_horizon:
            continue

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

        # CO2 Atmospheric Emissions
        if any(n.carriers.index.isin(["co2"])):
            co2_atm = n.stores.loc[["atmosphere" in name for name in n.stores.index]]
            last_timestep = n.snapshots.get_level_values(1)[-1]
            end_co2_atm_storage = (
                n.model["Store-e"].loc[:, co2_atm.index].sel(period=planning_horizon).sel(timestep=last_timestep)
            ).sum()
        else:
            end_co2_atm_storage = 0

        lhs = (p_em * em_pu).sum() + end_co2_atm_storage
        rhs = region_co2lim

        n.model.add_constraints(
            lhs <= rhs,
            name=f"GlobalConstraint-{emmission_lim.name}_{planning_horizon}co2_limit",
        )

        logger.info(
            f"Adding regional Co2 Limit for {emmission_lim.name} in {planning_horizon} with limit {rhs}",
        )
