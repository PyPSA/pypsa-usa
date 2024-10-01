"""
Adds extra extendable components to the clustered and simplified network.
"""

import logging
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from _helpers import calculate_annuity, configure_logging
from add_electricity import add_missing_carriers

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def add_co2_emissions(n, costs, carriers):
    """
    Add CO2 emissions to the network's carriers attribute.
    """
    suptechs = n.carriers.loc[carriers].index.str.split("-").str[0]

    missing_carriers = set(suptechs) - set(costs.index)
    if missing_carriers:
        logger.warning(f"CO2 emissions for carriers {missing_carriers} not defined in cost data.")
        suptechs = suptechs.difference(missing_carriers)
    n.carriers.loc[suptechs, "co2_emissions"] = costs.co2_emissions[suptechs].values

    n.carriers.fillna(
        {"co2_emissions": 0},
        inplace=True,
    )  # TODO: FIX THIS ISSUE IN BUILD_COST_DATA- missing co2_emissions for some VRE carriers

    if any("CCS" in carrier for carrier in carriers):
        ccs_factor = (
            1
            - pd.Series(carriers, index=carriers).str.split("-").str[1].str.replace("CCS", "").fillna(0).astype(int)
            / 100
        )
        n.carriers.loc[ccs_factor.index, "co2_emissions"] *= ccs_factor


def add_nice_carrier_names(n, config):
    carrier_i = n.carriers.index
    nice_names = (
        pd.Series(config["plotting"]["nice_names"]).reindex(carrier_i).fillna(carrier_i.to_series().str.title())
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
            capital_cost=costs.at[carrier, "annualized_capex_fom"],
            marginal_cost=0,  # costs.at[carrier, "marginal_cost"], # TODO: FIX THIS ISSUE IN BUILD_COST_DATA
            efficiency_store=costs.at[carrier, "efficiency"] ** roundtrip_correction,
            efficiency_dispatch=costs.at[carrier, "efficiency"] ** roundtrip_correction,
            max_hours=max_hours,
            cyclic_state_of_charge=False,
            build_year=investment_year,
            lifetime=costs.at[carrier, "cost_recovery_period_years"],
        )


def attach_phs_storageunits(n: pypsa.Network, elec_opts, costs: pd.DataFrame):
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

        if region_onshore_psh.empty:
            continue

        region_onshore_psh_grp = (
            region_onshore_psh.groupby(["name", "cost_kw_round"])["potential_mw"].agg("sum").reset_index()
        )

        region_onshore_psh_grp["class"] = region_onshore_psh_grp.groupby(["name"]).cumcount() + 1
        region_onshore_psh_grp["class"] = "c" + region_onshore_psh_grp["class"].astype(
            str,
        )
        region_onshore_psh_grp["tech"] = carrier
        region_onshore_psh_grp["carrier"] = region_onshore_psh_grp[["tech", "class"]].agg("_".join, axis=1)
        region_onshore_psh_grp["Generator"] = region_onshore_psh_grp["name"] + " " + region_onshore_psh_grp["carrier"]
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

        costs.at["PHS", "efficiency"] = efficiency_store
        costs.at["PHS", "co2_emissions"] = 0
        add_missing_carriers(n, ["PHS"])
        add_co2_emissions(n, costs, ["PHS"])
        n.madd(
            "StorageUnit",
            region_onshore_psh_grp.index,
            bus=region_onshore_psh_grp.name,
            carrier="PHS",  # region_onshore_psh_grp.tech,
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
            capital_cost=costs.at["fuel cell", "capital_cost"] * costs.at["fuel cell", "efficiency"],
            marginal_cost=costs.at["fuel cell", "marginal_cost"],
            build_year=investment_year,
            lifetime=costs.at["fuel cell", "lifetime"],
            suffix=str(investment_year),
        )


def split_retirement_gens(
    n: pypsa.Network,
    costs: pd.DataFrame,
    gens: list[str] = None,
    economic: bool = True,
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
    retirement_mask = n.generators["p_nom_extendable"] & (n.generators["carrier"].isin(gens) if gens else True)
    retirement_gens = n.generators[retirement_mask]
    if retirement_gens.empty:
        return

    # Change capex to fixed OM cost for retiring generators
    n.generators["capital_cost"] = n.generators.apply(
        lambda row: (
            row["capital_cost"]
            if not row.name in (retirement_gens.index)
            else costs.at[row["carrier"], "opex_fixed_per_kw"] * 1e3
        ),
        axis=1,
    )

    # Rename retiring generators to include "existing" suffix
    n.generators.index = n.generators.apply(
        lambda row: (row.name if not row.name in (retirement_gens.index) else row.name + " existing"),
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

    n.generators.loc[retirement_mask.values, "p_nom_extendable"] = (
        economic  # if economic retirement is true enable extendable
    )

    n.madd(  # Adding Expanding generators for the first investment period
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
        lifetime=retirement_gens.carrier.map(costs.lifetime).fillna(np.inf),
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
    Adds multiple investment options for generators types that were already
    existing in the network. Function used for all carriers, renewable and
    conventional.

    Specifically this function does the following:
    1. Adds new generators for the given investment year, according that year's costs.
        if this is the first investment period we use the existing generator's p_nom and p_nom_min
    2. Adds time dependent factors for the new generators


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
        capital_cost=gens.carrier.map(costs.annualized_capex_fom),
        build_year=investment_year,
        lifetime=gens.carrier.map(costs.cost_recovery_period_years),
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

    p_max_pu_t = n.generators_t["p_max_pu"][[x for x in gens.index if x in n.generators_t["p_max_pu"].columns]]
    p_max_pu_t = p_max_pu_t.rename(
        columns={x: f"{x} {investment_year}" for x in p_max_pu_t.columns},
    )
    n.generators_t["p_max_pu"] = n.generators_t["p_max_pu"].join(p_max_pu_t)


def attach_newCarrier_generators(n, costs, carriers, investment_year):
    """
    Attaches generators for carriers which did not previously exist in the
    network.

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
            capital_cost=costs.at[carrier, "annualized_capex_fom"],
            marginal_cost=costs.at[carrier, "marginal_cost"],
            efficiency=costs.at[carrier, "efficiency"],
            build_year=investment_year,
            lifetime=costs.at[carrier, "cost_recovery_period_years"],
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
        p_nom = n.generators.p_nom.loc[n.generators.carrier == carrier].sum()
        n.carriers.loc[carrier, "max_growth"] = growth_base.get(carrier) or p_nom
        n.carriers.loc[carrier, "max_relative_growth"] = max_annual_growth_rate[carrier] ** years


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
        n.investment_periods[i]: pd.read_csv(snakemake.input.tech_costs[i]).pivot(
            index="pypsa-name",
            columns="parameter",
            values="value",
        )
        for i in range(len(n.investment_periods))
    }

    new_carriers = list(
        set(elec_config["extendable_carriers"].get("Generator", []))
        - set(elec_config["conventional_carriers"])
        - set(elec_config["renewable_carriers"]),
    )

    if any("PHS" in s for s in elec_config["extendable_carriers"]["StorageUnit"]):
        attach_phs_storageunits(n, elec_config, costs_dict[n.investment_periods[0]])

    if snakemake.params.retirement == "economic":
        economic_retirement_gens = set(elec_config.get("conventional_carriers", None))
        split_retirement_gens(
            n,
            costs_dict[n.investment_periods[0]],
            economic_retirement_gens,
            economic=True,
        )

    split_retirement_gens(
        n,
        costs_dict[n.investment_periods[0]],
        set(elec_config.get("renewable_carriers", None)),
        economic=False,
    )

    multi_horizon_gens = n.generators[
        n.generators["p_nom_extendable"]
        & n.generators["carrier"].isin(elec_config["extendable_carriers"]["Generator"])
        & ~n.generators.index.str.contains("existing")
    ]

    for investment_year in n.investment_periods:
        costs = costs_dict[investment_year]
        attach_storageunits(n, costs, elec_config, investment_year)
        # attach_stores(n, costs, elec_config, investment_year)
        attach_multihorizon_generators(n, costs, multi_horizon_gens, investment_year)
        attach_newCarrier_generators(n, costs, new_carriers, investment_year)

    if not multi_horizon_gens.empty and not len(n.investment_periods) == 1:
        # Remove duplicate generators from first investment period,
        # created by attach_multihorizon_generators
        n.mremove(
            "Generator",
            multi_horizon_gens.index,
        )

    apply_itc(n, snakemake.config["costs"]["itc_modifier"])
    apply_ptc(n, snakemake.config["costs"]["ptc_modifier"])
    apply_max_annual_growth_rate(n, snakemake.config["costs"]["max_growth"])
    add_nice_carrier_names(n, snakemake.config)

    add_co2_emissions(n, costs_dict[n.investment_periods[0]], n.carriers.index)
    n.generators.to_csv("generators_ec.csv")
    n.consistency_check()
    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
