"""Adds extra extendable components to the clustered and simplified network."""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from _helpers import calculate_annuity, configure_logging
from add_electricity import add_missing_carriers
from shapely.geometry import Point

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def add_co2_emissions(n, costs, carriers):
    """Add CO2 emissions to the network's carriers attribute."""
    suptechs = n.carriers.loc[carriers].index.str.split("-").str[0]
    missing_carriers = set(suptechs) - set(costs.index)
    if missing_carriers:
        logger.warning(
            f"CO2 emissions for carriers {missing_carriers} not defined in cost data.",
        )
        suptechs = suptechs.difference(missing_carriers)
    n.carriers.loc[suptechs, "co2_emissions"] = costs.co2_emissions[suptechs].values

    n.carriers = n.carriers.fillna(
        {"co2_emissions": 0},
    )  # TODO: FIX THIS ISSUE IN BUILD_COST_DATA- missing co2_emissions for some VRE carriers

    if any("CCS" in carrier for carrier in carriers):
        ccs_carriers = [carrier for carrier in carriers if "CCS" in carrier]
        for ccs_carrier in ccs_carriers:
            base_carrier = ccs_carrier.split("-")[0]
            base_emissions = n.carriers.loc[base_carrier, "co2_emissions"]
            ccs_level = int(ccs_carrier.split("-")[1].replace("CCS", ""))
            ccs_emissions = (1 - ccs_level / 100) * base_emissions
            n.carriers.loc[ccs_carrier, "co2_emissions"] = ccs_emissions


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
    carriers: list[str] | None = None,
    economic: bool = True,
):
    """
    Seperates extendable conventional generators into existing and new
    generators to support economic or technical retirement.


    Specifically this function does the following:
    1. Creates duplicate generators for any that are tagged as extendable. For
    example, an extendable "CCGT" generator will be split into "CCGT existing" and "CCGT"
    2. Capital costs of existing extendable generators are replaced with fixed costs
    3. p_nom_max of existing extendable generators are set to p_nom
    4. p_nom_min of existing and new generators is set to zero

    Arguments:
    n: pypsa.Network,
    costs: pd.DataFrame,
    carriers: List[str]
        List of generator carriers to apply economic retirment to.
    economic: bool
        If True, enable economic retirement, else only allow lifetime
        retirement for the new generators
    """
    retirement_mask = (
        n.generators["p_nom_extendable"]
        & (n.generators["carrier"].isin(carriers) if carriers else True)
        & n.generators.p_nom
        > 0
    )
    retirement_gens = n.generators[retirement_mask]
    if retirement_gens.empty:
        return

    # Change capex to fixed OM cost for retiring generators
    n.generators["capital_cost"] = n.generators.apply(
        lambda row: (
            row["capital_cost"]
            if row.name not in (retirement_gens.index)
            else costs.at[row["carrier"], "opex_fixed_per_kw"] * 1e3
        ),
        axis=1,
    )

    # Rename retiring generators to include "existing" suffix
    n.generators.index = n.generators.apply(
        lambda row: (row.name if row.name not in (retirement_gens.index) else row.name + " existing"),
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

    n.generators.loc[
        retirement_mask.values,
        "p_nom_extendable",
    ] = economic  # if economic retirement is true enable extendable

    # Adding Expanding generators for the first investment period
    # There are generators that exist today and could expand
    # in the first time horizon
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
        lifetime=retirement_gens.carrier.map(costs.lifetime).fillna(np.inf),
        p_min_pu=retirement_gens.p_min_pu,
        p_max_pu=retirement_gens.p_max_pu,
        land_region=retirement_gens.land_region,
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


def attach_multihorizon_existing_generators(
    n: pypsa.Network,
    costs: dict,
    gens: pd.DataFrame,
    investment_year: int,
):
    """
    Adds multiple investment options for generators types that were already
    existing in the network. Function used for all carriers, renewable and
    conventional. Generators are added only to the nodes where they already exist
    because their cost information is spatially resolved.

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
        land_region=gens.land_region,
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


def attach_multihorizon_egs(
    n: pypsa.Network,
    costs: pd.DataFrame,
    costs_dict: dict,
    gens: pd.DataFrame,
    investment_year: int,
):
    """
    Adds multiple investment options for EGS.
    Arguments:
    n: pypsa.Network,
    costs: pd.DataFrame,
        dataframe with costs of investment year
    costs_dict: dict,
        Dict of costs for each investment period
    carriers: List[str]
        List of carriers to add multiple investment options for.
    """
    if gens.empty or len(n.investment_periods) == 1:
        return

    lifetime = 25  # Following EGS supply curves by Aljubran et al. (2024)
    base_year = n.investment_periods[0]
    learning_ratio = costs.loc["EGS", "capex_per_kw"] / costs_dict[base_year].loc["EGS", "capex_per_kw"]
    capital_cost = learning_ratio * gens["capital_cost"]
    n.madd(
        "Generator",
        gens.index,
        suffix=f" {investment_year}",
        carrier=gens.carrier,
        bus=gens.bus,
        p_nom_min=0,
        p_nom=0,
        p_nom_max=gens.p_nom_max,
        p_nom_extendable=True,
        ramp_limit_up=gens.ramp_limit_up,
        ramp_limit_down=gens.ramp_limit_down,
        efficiency=gens.efficiency,
        marginal_cost=gens.marginal_cost,
        p_min_pu=gens.p_min_pu,
        p_max_pu=gens.p_max_pu,
        capital_cost=capital_cost,
        build_year=investment_year,
        lifetime=lifetime,
    )

    # time dependent factors added after
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

    # shift over time to capture decline
    investment_year_idx = np.where(n.investment_periods == investment_year)[0][0]
    cars = list(
        n.generators_t["p_max_pu"].filter(like="EGS").filter(like=str(investment_year)).columns,
    )
    n.generators_t["p_max_pu"].loc[n.investment_periods[investment_year_idx:], cars] = (
        n.generators_t["p_max_pu"]
        .loc[
            n.investment_periods[: len(n.investment_periods) - investment_year_idx],
            cars,
        ]
        .values
    )


def attach_multihorizon_new_generators(n, costs, carriers, investment_year):
    """
    Attaches generators for carriers which did not previously exist in the
    network (CCS, H2, SMR, etc). These generators do not have spatially resolved
    costs, so they are added to all buses in the network.

    Unlike CT's and CCGT's we include nuclear in this function, since we assume
    they can be built anywhere in the network.

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
    min_years = snakemake.config["costs"].get("min_year")
    buses_i = n.buses.index
    for carrier in carriers:
        p_max_pu_t = None
        if min_years and min_years.get(carrier, 0) > investment_year:
            continue
        existing_gens = n.generators[
            (
                (n.generators.carrier == carrier)
                & ~n.generators.index.str.contains("existing")
                & (n.generators.build_year <= n.investment_periods[0])
            )
        ].copy()

        if not existing_gens.empty:
            p_max_pu_t = n.get_switchable_as_dense("Generator", "p_max_pu")
            p_max_pu_t = (p_max_pu_t[[x for x in existing_gens.index if x in p_max_pu_t.columns]]).mean().mean()

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
            lifetime=costs.at[carrier, "lifetime"],
            p_max_pu=p_max_pu_t if p_max_pu_t is not None else 1,
            ramp_limit_up=existing_gens.ramp_limit_up.mean() or 1,
            ramp_limit_down=existing_gens.ramp_limit_down.mean() or 1,
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
    Applies maximum annual growth rate to components specified in the
    configuration file.

    Arguments:
    n: pypsa.Network,
    max_growth: dict,
        Dict of maximum annual growth rate and base for each carrier.
        Format: #{carrier_name: {base: , rate: }}
    """
    if max_growth is None or len(n.investment_periods) <= 1:
        return

    years = n.investment_period_weightings.index.to_series().diff().dropna().mean()

    for carrier, growth_params in max_growth.items():
        base = growth_params.get("base", None)
        rate = growth_params.get("rate", None)

        if base is None and rate is None:
            continue

        p_nom = n.generators.p_nom.loc[n.generators.carrier == carrier].sum()
        n.carriers.loc[carrier, "max_growth"] = base or p_nom
        n.carriers.loc[carrier, "max_relative_growth"] = rate**years


def add_demand_response(
    n: pypsa.Network,
    dr_config: dict[str, str | float],
) -> None:
    """Add price based demand response to network."""
    n.add("Carrier", "demand_response", color="#dd2e23", nice_name="Demand Response")

    shift = dr_config.get("shift", 0)
    if shift == 0:
        logger.info(f"DR not applied as allowable sift is {shift}")
        return

    marginal_cost_storage = dr_config.get("marginal_cost", 0)
    if marginal_cost_storage == 0:
        logger.warning("No cost applied to demand response")

    # attach dr at all load locations

    buses = n.loads.bus
    df = n.buses[n.buses.index.isin(buses)].copy()

    # two storageunits for forward and backwards load shifting

    n.madd(
        "Bus",
        names=df.index,
        suffix="-fwd-dr",
        x=df.x,
        y=df.y,
        carrier="demand_response",
        unit="MWh",
        country=df.country,
        reeds_zone=df.reeds_zone,
        reeds_ba=df.reeds_ba,
        interconnect=df.interconnect,
        trans_reg=df.trans_reg,
        trans_grp=df.trans_grp,
        reeds_state=df.reeds_state,
        substation_lv=df.substation_lv,
    )

    n.madd(
        "Bus",
        names=df.index,
        suffix="-bck-dr",
        x=df.x,
        y=df.y,
        carrier="demand_response",
        unit="MWh",
        country=df.country,
        reeds_zone=df.reeds_zone,
        reeds_ba=df.reeds_ba,
        interconnect=df.interconnect,
        trans_reg=df.trans_reg,
        trans_grp=df.trans_grp,
        reeds_state=df.reeds_state,
        substation_lv=df.substation_lv,
    )

    # seperate charging/discharging links for easier constraint generation

    n.madd(
        "Link",
        names=df.index,
        suffix="-fwd-dr-charger",
        bus0=df.index,
        bus1=df.index + "-fwd-dr",
        carrier="demand_response",
        p_nom_extendable=False,
        p_nom=np.inf,
    )

    n.madd(
        "Link",
        names=df.index,
        suffix="-fwd-dr-discharger",
        bus0=df.index + "-fwd-dr",
        bus1=df.index,
        carrier="demand_response",
        p_nom_extendable=False,
        p_nom=np.inf,
    )

    n.madd(
        "Link",
        names=df.index,
        suffix="-bck-dr-charger",
        bus0=df.index,
        bus1=df.index + "-bck-dr",
        carrier="demand_response",
        p_nom_extendable=False,
        p_nom=np.inf,
    )

    n.madd(
        "Link",
        names=df.index,
        suffix="-bck-dr-discharger",
        bus0=df.index + "-bck-dr",
        bus1=df.index,
        carrier="demand_response",
        p_nom_extendable=False,
        p_nom=np.inf,
    )

    # backward stores have positive marginal cost storage and postive e
    # forward stores have negative marginal cost storage and negative e

    n.madd(
        "Store",
        names=df.index,
        suffix="-bck-dr",
        bus=df.index + "-bck-dr",
        e_cyclic=True,
        e_nom_extendable=False,
        e_nom=np.inf,
        e_min_pu=0,
        e_max_pu=1,
        carrier="demand_response",
        marginal_cost_storage=marginal_cost_storage,
    )

    n.madd(
        "Store",
        names=df.index,
        suffix="-fwd-dr",
        bus=df.index + "-fwd-dr",
        e_cyclic=True,
        e_nom_extendable=False,
        e_nom=np.inf,
        e_min_pu=-1,
        e_max_pu=0,
        carrier="demand_response",
        marginal_cost_storage=marginal_cost_storage * (-1),
    )


def add_co2_storage(n: pypsa.Network, config: dict, co2_storage_csv: str, sector: bool):
    """Adds node level CO2 (underground) storage."""

    # get node level CO2 (underground) storage potential and cost from CSV file
    co2_storage = pd.read_csv(co2_storage_csv).set_index("node")


    # add buses to represent node level CO2 captured by different processes
    n.madd("Bus",
        co2_storage.index,
        suffix = " co2 capture",
        carrier = "co2",
    )


    # add stores to represent node level CO2 (underground) storage
    n.madd("Store",
        co2_storage.index,
        suffix = " co2 storage",
        bus = co2_storage.index + " co2 capture",
        e_nom_extendable = True,
        e_nom_max = co2_storage["potential [MtCO2]"] * 1e6,   # in tCO2
        marginal_cost = co2_storage["cost [USD/tCO2]"],
        carrier = "co2",
    )


    # add carrier to represent CO2
    n.madd("Carrier",
        ["co2"],
        color = config["plotting"]["tech_colors"]["co2"],
        nice_name = config["plotting"]["nice_names"]["co2"],
    )


    # add carrier to represent CC only (i.e. without S)
    carriers = n.carriers.query("Carrier.str.endswith('CCS')")
    if carriers.empty == False:
        n.madd("Carrier",
            carriers.index.str.replace("CCS", "CC", regex = True),
            color = carriers["color"],
            nice_name = carriers["nice_name"].str.replace("Ccs", "Cc", regex = True),
        )


    if sector is True:

        links = n.links.index.str.contains("CCS")
        if True in links:   # links equipped with CCS exists
            # add bus4 to CCS links to point to their respective CO2 capture bus and specify its efficiency
            n.links.loc[links, "bus4"] = co2_storage.index + " %s co2 capture" % n.links.loc[links]["bus2"].str.split(" ")[-1][-1].split("-")[0]
            n.links.loc[links, "efficiency4"] = [0.95] * len(co2_storage.index)    # send 95% of a tonne of CO2 generated per each MW produced (and sent from bus0 to bus1)


            # remove storage cost from CCS links' capital cost (given that they do not require technology to store CO2 anymore as this is done underground)
            n.links.loc[links, "capital_cost"] *= 0.9   # TODO: replace with concrete storage cost


            # replace substring "CCS" with just "CC" in CCS links' names and carriers
            n.links.loc[links, "carrier"] = n.links.loc[links].carrier.str.replace("CCS", "CC", regex = True)
            n.links.index = n.links.index.str.replace("CCS", "CC", regex = True)

    else:   # sector-less

        generators = n.generators.index.str.contains("CCS")
        if True in generators:   # generators equipped with CCS exists
            # remove storage cost from CCS generators' capital cost (given that they do not require technology to store CO2 anymore as this is done underground)
            n.generators.loc[generators, "capital_cost"] *= 0.9   # TODO: replace with concrete storage cost


            # replace "CCS" with "CC" in CCS generators' indexes/carriers description
            n.generators.loc[generators, "carrier"] = n.generators.loc[generators].carrier.str.replace("CCS", "CC", regex = True)
            n.generators.index = n.generators.index.str.replace("CCS", "CC", regex = True)


            # add buses to represent node level electricity CC generator
            indexes = n.generators.loc[generators].index
            n.madd("Bus",
                indexes,
                carrier = n.generators.loc[generators].carrier,
            )


            # add buses to represent node level emitted CO2 by different processes
            granularity = config["dac"]["granularity"]
            if granularity == "nation":
                buses_atmosphere_unique = ["atmosphere"]
                buses_atmosphere = buses_atmosphere_unique
            else:
                if config["model_topology"]["transmission_network"] == "reeds":
                    elements = 1
                else:   # TAMU
                    elements = 2
                if granularity == "state":
                    buses = n.buses[["x", "y"]].copy()
                    buses["geometry"] = buses.apply(lambda x: Point(x.x, x.y), axis = 1)
                    buses_gdf = gpd.GeoDataFrame(buses, crs = "EPSG:4269")
                    states_gdf = gpd.GeoDataFrame(gpd.read_file(snakemake.input.county_shapes).dissolve("STUSPS")["geometry"])
                    buses_projected = buses_gdf.to_crs("EPSG:3857")
                    states_projected = states_gdf.to_crs("EPSG:3857")                    
                    states = gpd.sjoin_nearest(buses_projected, states_projected, how = "left").query("x != 0 and y != 0")["STUSPS"]   # TODO: remove the query and have it when making a copy of the buses above (this way it will faster to make the join operation)
                    buses_atmosphere_unique = states.unique() + " atmosphere"
                    buses_atmosphere = ["%s atmosphere" % states.loc[" ".join(index.split(" ")[:elements])] for index in indexes]
                else:   # node
                    buses_atmosphere_unique = ["%s atmosphere" % " ".join(index.split(" ")[:elements]) for index in indexes]
                    buses_atmosphere = buses_atmosphere_unique


            # add buses to represent (air) atmosphere where CO2 emissions are sent to
            n.madd("Bus",
                buses_atmosphere_unique,
                carrier = "co2",
            )


            # add stores to represent (air) atmosphere where CO2 emissions are stored
            n.madd("Store",
                buses_atmosphere_unique,
                bus = buses_atmosphere_unique,
                e_nom_extendable = True,
                e_min_pu = -1,
                carrier = "co2",
            )


            # calculate efficiencies
            gas_co2_intensity = costs.loc["gas"]["co2_emissions"]
            coal_co2_intensity = costs.loc["coal"]["co2_emissions"]           
            efficiency2 = []
            efficiency3 = []
            for index in indexes:
                generator_efficiency = n.generators.loc[index]["efficiency"]
                if "CCGT" in index:
                    efficiency = 1 / generator_efficiency * gas_co2_intensity
                elif "coal" in index:
                    efficiency = 1 / generator_efficiency * coal_co2_intensity
                else:
                    logger.warning("Assuming a CO2 intensity equal to 1 given that generator '%s' is not powered by gas or coal" % index)
                    efficiency = 1 / generator_efficiency * 1
                efficiency2.append(efficiency)
                efficiency3.append(efficiency * 0.05 / 0.95)


            # add links to represent the sending of electricity (in MW) to the electricity bus (e.g. "p9" if ReEDS or "p100 0" if TAMU) as well as sending emitted CO2 (by the CC generator) to both the atmosphere bus and the co2 capture bus
            n.madd("Link",
                indexes,
                bus0 = indexes,
                bus1 = n.generators.loc[generators]["bus"],
                bus2 = buses_atmosphere,
                bus3 = co2_storage.index + " co2 capture",
                efficiency = 1,
                efficiency2 = efficiency2,
                efficiency3 = efficiency3,
                carrier = n.generators.loc[generators].carrier,
            )


            # (re-)attach CC generators to new buses (that represent node level CC generator)
            n.generators.loc[generators, "bus"] = indexes


def add_co2_network(n: pypsa.Network, config: dict):
    """Adds CO2 (transportation) network."""

    # get electricity connections
    if config["model_topology"]["transmission_network"] == "reeds":
        connections = n.links.query("carrier == 'AC' and not Link.str.endswith('exp')")
    else:   # TAMU
        connections = n.lines


    # calculate annualized capital cost
    number_years = n.snapshot_weightings.generators.sum() / 8760
    cost = config["co2"]["network"]["capital_cost"] * calculate_annuity(config["co2"]["network"]["lifetime"], config["co2"]["network"]["discount_rate"]) * number_years


    # add links to represent CO2 (transportation) network based on electricity connections layout
    n.madd("Link",
        connections.index,
        suffix = " co2 transport",
        bus0 = connections["bus0"] + " co2 capture",
        bus1 = connections["bus1"] + " co2 capture",
        efficiency = 1,
        p_min_pu = -1,
        p_nom_extendable = True,
        length = connections["length"].values,
        capital_cost = cost * connections["length"].values,
        marginal_cost = config["co2"]["network"]["marginal_cost"],
        carrier = "co2",
        lifetime = config["co2"]["network"]["lifetime"],
    )


def add_dac(n: pypsa.Network, config: dict, sector: bool):
    """Adds node level DAC capabilities."""

    # generate node level buses to represent emitted, captured and accounted CO2 and links to represent DAC in function of whether network is based on sectors or not
    if sector is True:

        # set number of elements based on electricity transmission network type
        if config["model_topology"]["transmission_network"] == "reeds":
            elements = 1
        else:   # TAMU
            elements = 2


        # get links that emit CO2 for all sectors
        links = n.links.query("bus2.str.endswith('-co2')")


        # lorem ipsum
        existing_states = set()
        existing_dac = set()
        buses_atmosphere = []
        buses_atmosphere_unique = []
        buses_atmosphereXXX = []
        buses_co2_capture = []
        buses_ac = []
        buses_co2_account = []
        links_dac = []
        for index in links.index:
            bus2 = links.loc[index]["bus2"]   # e.g. "CA pwr-co2"
            node = " ".join(index.split(" ")[:elements])   # e.g. "p9" if ReEDS or "p100 0" if TAMU
            state = bus2.split(" ")[0]   # e.g. "CA"
            node_sector = bus2.split(" ")[1].split("-")[0]   # "pwr"
            buses_atmosphereXXX.append("%s atmosphere" % state)
            
            
            #if state not in existing_states:
            #    buses_atmosphere_unique.append("%s atmosphere" % state)
            #    buses_co2_account.append(bus2)
            #    existing_states.add(state)
            
            #if node not in existing_dac:
            #    buses_ac.append(node)
            #    links_dac.append("%s dac" % node)
            #    buses_co2_capture.append("%s capture" % node)
            #    existing_dac.add(node)
            #    buses_atmosphere.append("%s atmosphere" % state)


            key = (node, node_sector)
            if key not in existing_states:
                buses_atmosphere_unique.append("%s %s atmosphere" % (node, node_sector))
                buses_co2_account.append(bus2)
                existing_states.add(key)              
                
            if key not in existing_dac:
                buses_ac.append(node)
                links_dac.append("%s %s dac" % (node, node_sector))
                #buses_co2_capture.append("%s capture" % node)
                buses_atmosphere.append("%s %s atmosphere" % (node, node_sector))
                existing_dac.add(key)


        ###################
        buses_co2_capture = n.buses.query("Bus.str.endswith(' co2 capture')").index
        buses_ac = buses_co2_capture.str.replace(" co2 capture", "")
        links_dac = [buses_atmosphere_unique.str.replace(" atmosphere", " dac")]
        ####################

        # add state level buses to represent (air) atmosphere where CO2 emissions are sent to
        n.madd("Bus",
            buses_atmosphere_unique,
            carrier = "co2",
        )


        # add links from node level buses that emit CO2 to state level buses tracking CO2 emissions
        n.madd("Link",
            buses_atmosphere_unique,
            bus0 = buses_atmosphere_unique,
            bus1 = buses_co2_account,
            efficiency = 1,
            p_nom_extendable = True,   # TODO: check if this is necessary
            carrier = "co2",
        )


        # redirect links that emit CO2 to node level buses that emit CO2
        #n.links.loc[links.index, "bus2"] = links.index.str.split(" ").str[0] + " " + links.loc[links.index]["bus2"].str.split(" ").str[1].str.split("-").str[0] + " atmosphere"   # e.g. "p1 trn co2 limit"
        n.links.loc[links.index, "bus2"] = buses_atmosphereXXX

    else:   # sector-less

        buses_atmosphere = n.links.query("bus2.str.endswith('atmosphere')")["bus2"].values
        buses_co2_capture = n.buses.query("Bus.str.endswith(' co2 capture')").index
        buses_ac = buses_co2_capture.str.replace(" co2 capture", "")
        links_dac = buses_co2_capture.str.replace(" co2 capture", " dac")


    # calculate annualized capital cost
    number_years = n.snapshot_weightings.generators.sum() / 8760
    cost = config["dac"]["capital_cost"] * calculate_annuity(config["dac"]["lifetime"], config["dac"]["discount_rate"]) * number_years

    print("links_dac:")
    print(links_dac)
    print("")
    print("buses_atmosphere:")
    print(buses_atmosphere)
    print("")
    print("buses_co2_capture:")
    print(buses_co2_capture)
    print("")
    print("buses_ac:")
    print(buses_ac)
    print("")


    # add links to represent node level DAC capabilities
    n.madd("Link",
        links_dac,
        bus0 = buses_atmosphere,
        bus1 = buses_co2_capture,
        bus2 = buses_ac,
        efficiency = 1,   # in tCO2
        efficiency2 = -config["dac"]["electricity_input"],   # in MWh (for each tCO2)
        p_nom_extendable = True,
        capital_cost = cost,
        carrier = "co2",
        lifetime = config["dac"]["lifetime"],
    )
    #print(88/0)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_extra_components",
            interconnect="western",
            simpl="70",
            clusters="4m",
        )
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)
    elec_config = snakemake.config["electricity"]

    costs_dict = {
        n.investment_periods[i]: pd.read_csv(snakemake.input.tech_costs[i]).pivot(
            index="pypsa-name",
            columns="parameter",
            values="value",
        )
        for i in range(len(n.investment_periods))
    }

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
    # Split renewable generators from the first investement period to support lifetime retirement
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

    multi_horizon_gens = multi_horizon_gens[
        multi_horizon_gens["carrier"].isin(
            [car for car in elec_config["extendable_carriers"]["Generator"] if "EGS" not in car],
        )
    ]
    egs_gens = n.generators[n.generators["p_nom_extendable"]]
    egs_gens = egs_gens.loc[egs_gens["carrier"].str.contains("EGS")]

    new_carriers = list(
        set(elec_config["extendable_carriers"].get("Generator", [])) - set(n.generators.carrier.unique())
        | set(
            ["nuclear"] if "nuclear" in elec_config["extendable_carriers"].get("Generator", []) else [],
        ),
    )

    for investment_year in n.investment_periods:
        costs = costs_dict[investment_year]
        attach_storageunits(n, costs, elec_config, investment_year)
        attach_multihorizon_existing_generators(
            n,
            costs,
            multi_horizon_gens,
            investment_year,
        )
        attach_multihorizon_egs(n, costs, costs_dict, egs_gens, investment_year)
        attach_multihorizon_new_generators(n, costs, new_carriers, investment_year)
        # attach_stores(n, costs, elec_config, investment_year)

    if not multi_horizon_gens.empty and not len(n.investment_periods) == 1:
        # Remove duplicate generators from first investment period,
        # created by attach_multihorizon_generators
        n.mremove("Generator", multi_horizon_gens.index)

    apply_itc(n, snakemake.config["costs"]["itc_modifier"])
    apply_ptc(n, snakemake.config["costs"]["ptc_modifier"])
    apply_max_annual_growth_rate(n, snakemake.config["costs"]["max_growth"])
    add_nice_carrier_names(n, snakemake.config)
    add_co2_emissions(n, costs_dict[n.investment_periods[0]], n.carriers.index)

    dr_config = snakemake.params.demand_response
    if dr_config:
        add_demand_response(n, dr_config)

    if snakemake.config["scenario"]["sector"] == "E":
        # add node level CO2 (underground) storage
        if snakemake.config["co2"]["storage"] is True:
            logger.info("Adding node level CO2 (underground) storage")
            add_co2_storage(n, snakemake.config, snakemake.input.co2_storage, False)

        # add CO2 (transportation) network
        if snakemake.config["co2"]["network"]["enable"] is True:
            if snakemake.config["co2"]["storage"] is True:
                logger.info("Adding CO2 (transportation) network")
                add_co2_network(n, snakemake.config)
            else:
                logger.warning("Not adding CO2 (transportation) network given that CO2 (underground) storage is not enabled")

        # add node level DAC capabilities
        if snakemake.config["dac"]["enable"] is True:
            if snakemake.config["co2"]["storage"] is True:
                logger.info("Adding node level DAC capabilities")
                add_dac(n, snakemake.config, False)
            else:
                logger.warning("Not adding node level DAC capabilities given that CO2 (underground) storage is not enabled")

    n.consistency_check()
    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
