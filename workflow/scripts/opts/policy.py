import logging  # noqa: D100

import numpy as np
import pandas as pd
from opts._helpers import ceil_precision, filter_components, floor_precision, get_region_buses
from pypsa.descriptors import get_switchable_as_dense as get_as_dense

logger = logging.getLogger(__name__)


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

    for _, target in tct_data.iterrows():
        planning_horizon = target.planning_horizon
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


def add_RPS_constraints(n, config, sector, snakemake=None):
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
    sector: bool
        Sector study
    snakemake: object, optional
        Snakemake object containing inputs and parameters

    Returns
    -------
    None
    """

    def process_reeds_data(filepath, carriers, value_col):
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

        # # Extract and create new rows for `rps_solar` and `rps_wind`
        # additional_rows = []
        # for carrier_col, carrier_name in [
        #     ("rps_solar", "solar"),
        #     ("rps_wind", "onwind, offwind, offwind_floating"),
        # ]:
        #     if carrier_col in reeds.columns:
        #         temp = reeds[["region", "planning_horizon", carrier_col]].copy()
        #         temp = temp.rename(columns={carrier_col: "pct"})
        #         temp["carrier"] = carrier_name
        #         additional_rows.append(temp)

        # # Combine original data with additional rows
        # if additional_rows:
        #     additional_rows = pd.concat(additional_rows, ignore_index=True)
        #     reeds = pd.concat([reeds, additional_rows], ignore_index=True)

        # Ensure the final dataframe has consistent columns
        reeds = reeds[["region", "planning_horizon", "carrier", "pct"]]
        reeds = reeds[reeds["pct"] > 0.0]  # Remove any rows with zero or negative percentages

        return reeds

    # Read portfolio standards data
    portfolio_standards = pd.read_csv(config["electricity"]["portfolio_standards"])

    # Define carriers for RPS and CES
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
    ces_carriers = [*rps_carriers, "nuclear", "SMR", "hydrogen_ct", "CCGT-95CCS", "CCGT-99CCS", "Coal-95CCS"]

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
        & (
            portfolio_standards.planning_horizon.isin(
                snakemake.params.planning_horizons,
            )
        )
        & (portfolio_standards.region.isin(n.buses.reeds_state.unique()))
    ]

    REC_TRADING_ZONE_MAPPER = {  # noqa: N806
        "CA": "WREGIS",
        "OR": "WREGIS",
        "AZ": "WREGIS",
        "WA": "WREGIS",
        "NM": "WREGIS",
        "UT": "WREGIS",
        "CO": "WREGIS",
        "NV": "WREGIS",
        "ID": "WREGIS",
        "WY": "WREGIS",
        "MT": "WREGIS",
        "ND": "MRETS",
        "SD": "MRETS",
        "MN": "MRETS",
        "IA": "MRETS",
        "WI": "MRETS",
        "MI": "MIRECS",
        "MO": "NAR",
        "KS": "NAR",
        "IL": "MRETS",
        "IN": "MRETS",
        "OH": "MRETS",
        "KY": "PJM-GATS",
        "VA": "PJM-GATS",
        "WV": "PJM-GATS",
        "MD": "PJM-GATS",
        "DE": "PJM-GATS",
        "NJ": "PJM-GATS",
        "PA": "PJM-GATS",
        "NY": "NYGATS",
        "CT": "NEPOOL",
        "RI": "NEPOOL",
        "MA": "NEPOOL",
        "NH": "NEPOOL",
        "ME": "NEPOOL",
        "VT": "NEPOOL",
        "NC": "NC-RETS",
        "TX": "ERCOT",
    }
    portfolio_standards["trading_zone"] = portfolio_standards.region.map(REC_TRADING_ZONE_MAPPER).fillna(
        portfolio_standards.region,
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

        if sector:
            # power level buses
            pwr_buses = n.buses[(n.buses.carrier == "AC") & (n.buses.index.isin(region_buses.index))]
            # links delievering power within the region
            # removes any transmission links
            pwr_links = n.links[(n.links.bus0.isin(pwr_buses.index)) & ~(n.links.bus1.isin(pwr_buses.index))]
            region_demand_sector = (
                n.model["Link-p"].sel(period=constraint_row.planning_horizon, Link=pwr_links.index).sum()
            )
            region_rps_rhs_sector = int(constraint_row.pct * region_demand_sector)
            portfolio_standards.loc[constraint_row.name, "rps_rhs_sector"] = region_rps_rhs_sector

    # Iterate through constraints and add RPS constraints to the model
    for (trading_zone, planning_horizon, policy_carriers), zone_constraints in portfolio_standards.groupby(
        ["trading_zone", "planning_horizon", "carrier"],
    ):
        region_buses = get_region_buses(n, zone_constraints.region.unique())
        carriers = [carrier.strip() for carrier in policy_carriers.split(",")]

        # Filter region generators
        region_gens = n.generators[n.generators.bus.isin(region_buses.index)]
        region_gens_eligible = region_gens[region_gens.carrier.isin(carriers)]

        if region_gens_eligible.empty:
            return

        elif not sector:
            # Eligible generation
            p_eligible = n.model["Generator-p"].sel(
                period=planning_horizon,
                Generator=region_gens_eligible.index,
            )
            renewable_gen = zone_constraints.rps_rhs.sum()
            lhs = p_eligible.sum() - renewable_gen
            rhs = 0

        elif sector:
            # generator power contributing
            p_eligible = n.model["Generator-p"].sel(
                period=planning_horizon,
                Generator=region_gens_eligible.index,
            )
            renewable_gen = zone_constraints.rps_rhs_sector.sum()
            lhs = p_eligible.sum() - renewable_gen
            rhs = 0

        else:
            logger.error("Undefined control flow for RPS constraint.")

        n.model.add_constraints(
            lhs >= rhs,
            name=f"GlobalConstraint-{trading_zone}_{planning_horizon}_rps_limit",
        )

        logger.info(
            f"Added RPS constraint '{trading_zone}' for {planning_horizon} "
            f"requiring {renewable_gen:.1f} of {policy_carriers} generation ",
        )


def add_regional_co2limit(n, config):
    """Adding regional regional CO2 Limits Specified in the config.yaml."""
    regional_co2_lims = pd.read_csv(
        config["electricity"]["regional_Co2_limits"],
        index_col=[0],
    )
    logger.info("Adding regional Co2 Limits.")
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

        n.model.add_constraints(
            lhs <= rhs,
            name=f"GlobalConstraint-{emmission_lim.name}_{planning_horizon}co2_limit",
        )

        logger.info(
            f"Adding regional Co2 Limit for {emmission_lim.name} in {planning_horizon}",
        )
