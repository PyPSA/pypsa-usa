"""Adds extra extendable components to the clustered and simplified network."""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from _helpers import calculate_annuity, configure_logging
from add_electricity import add_missing_carriers
from eia import FuelCosts
from opts._helpers import get_region_buses
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
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


def apply_itc(n, itc_modifier, monitization_cost=0.1):
    """
    Applies investment tax credit to all extendable components in the network.

    Arguments:
    n: pypsa.Network,
    itc_modifier: dict,
        Dict of ITC modifiers for each carrier
    """
    for carrier in itc_modifier.keys():
        carrier_mask = n.generators["carrier"] == carrier
        n.generators.loc[carrier_mask, "capital_cost"] *= 1 - ((1 - monitization_cost) * itc_modifier[carrier])

        carrier_mask = n.storage_units["carrier"] == carrier
        n.storage_units.loc[carrier_mask, "capital_cost"] *= 1 - ((1 - monitization_cost) * itc_modifier[carrier])


def apply_ptc(n, ptc_modifier, costs):
    """
    Applies production tax credit to all extendable components in the network.

    Arguments:
    n: pypsa.Network,
    ptc_modifier: dict,
        Dict of PTC modifiers for each carrier
    """

    def discount_ptc(ptc, r, financial_lifetime, credit_lifetime=10, monitization_cost_pct=0.1):
        eff_ptc = (1 - monitization_cost_pct) * ptc
        pv = eff_ptc * (1 - (1 + r) ** (-1 * credit_lifetime)) / r
        crf = (r * (1 + r) ** financial_lifetime) / ((1 + r) ** financial_lifetime - 1)
        return round(pv * crf, 2)

    for carrier in ptc_modifier.keys():
        ptc = ptc_modifier[carrier]
        discounted_ptc = discount_ptc(ptc, costs.at[carrier, "wacc_real"], costs.at[carrier, "lifetime"])
        mask = (n.generators["carrier"] == carrier) & n.generators.p_nom_extendable
        for build_year in n.investment_periods:
            mask_by = (n.generators.build_year == build_year) & mask
            mc = n.get_switchable_as_dense("Generator", "marginal_cost").loc[:, mask_by]
            mc.loc[build_year:, :] -= discounted_ptc
            n.generators_t.marginal_cost.loc[:, mask_by] = mc
            n.generators.loc[mask_by, "marginal_cost"] -= discounted_ptc


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


def trim_network(n, trim_topology):
    """
    Trim_network splits the network into two parts:
        - The internal network, which is the network within the specified zones.
        - The external network, which is the network outside the specified zones.

    The internal network is retained and unchanged. While the external network components are removed. The external buses which are directly connected to the internal network are aggregated to the `nerc_reg` value of their buses.
    The only generators kept are the OCGTs at the external buses, which are set to non-extendable.

    The external OCGT generators are set to the carrier name `imports` and retain the same emissions intensity.

    """
    retain_zones = trim_topology["zone"]
    internal_buses = get_region_buses(n, retain_zones)
    if internal_buses.empty:
        logger.warning("No internal buses found, skipping trim_network")
        return None

    # Get all lines and links connected to internal buses
    retain_lines = n.lines[n.lines.bus0.isin(internal_buses.index) | n.lines.bus1.isin(internal_buses.index)]
    retain_links = n.links[n.links.bus0.isin(internal_buses.index) | n.links.bus1.isin(internal_buses.index)]

    # Find buses to remove (those not connected to internal network)
    buses_to_remove = n.buses[
        ~n.buses.index.isin(retain_lines.bus0)
        & ~n.buses.index.isin(retain_lines.bus1)
        & ~n.buses.index.isin(retain_links.bus0)
        & ~n.buses.index.isin(retain_links.bus1)
    ]

    # Find external buses to keep (connected to internal network but not internal)
    external_buses_to_keep = n.buses.loc[
        ~n.buses.index.isin(buses_to_remove.index) & ~n.buses.index.isin(internal_buses.index)
    ]

    # Remove components at buses that are being removed
    for c in n.one_port_components:
        component = n.df(c)
        rm = component[component.bus.isin(buses_to_remove.index)]
        if not rm.empty:
            n.mremove(c, rm.index)

    # Remove lines and links at buses being removed
    for c in ["Line", "Link"]:
        component = n.df(c)
        rm = component[~component.bus0.isin(internal_buses.index) & ~component.bus1.isin(internal_buses.index)]
        if not rm.empty:
            n.mremove(c, rm.index)

    # Remove the buses
    n.mremove("Bus", buses_to_remove.index)

    # Get OCGT generators and calculate average marginal cost
    ocgt_gens = n.generators[n.generators.carrier == "OCGT"]
    avg_marginal_cost = get_as_dense(n, "Generator", "marginal_cost").loc[:, ocgt_gens.index].mean().mean()
    n.add("Carrier", "imports", co2_emissions=0.428, nice_name="imports")

    # remove existing oneport components at bus
    for c in n.one_port_components:
        component = n.df(c)
        rm = component[component.bus.isin(external_buses_to_keep.index)]
        if not rm.empty:
            logger.info(f"Removing {c} at external buses {external_buses_to_keep.index} with components {rm.index}")
            n.mremove(c, rm.index)

    # Handle external buses and their generators
    for bus in external_buses_to_keep.index:
        # Create new import generator
        bus_name = n.buses.loc[bus].name
        n.add(
            "Generator",
            f"import_{bus_name}",
            bus=bus,
            carrier="imports",
            p_nom=1e4,
            p_nom_extendable=False,
            marginal_cost=avg_marginal_cost,
            efficiency=1,
            build_year=n.investment_periods[0],
            lifetime=100,
        )

        # Change location names of external buses, append imports to the ['reeds_state', 'reeds_zone', 'reeds_ba', 'interconnect', 'trans_reg', 'trans_grp']
        n.buses.loc[bus, "reeds_state"] = f"imports_{n.buses.loc[bus, 'reeds_state']}"
        n.buses.loc[bus, "reeds_zone"] = f"imports_{n.buses.loc[bus, 'reeds_zone']}"
        n.buses.loc[bus, "reeds_ba"] = f"imports_{n.buses.loc[bus, 'reeds_ba']}"
        n.buses.loc[bus, "interconnect"] = f"imports_{n.buses.loc[bus, 'interconnect']}"
        n.buses.loc[bus, "trans_reg"] = f"imports_{n.buses.loc[bus, 'trans_reg']}"
        n.buses.loc[bus, "trans_grp"] = f"imports_{n.buses.loc[bus, 'trans_grp']}"

        # Set all links and lines connected to the bus as non-extendable
        for c in ["Line", "Link"]:
            attr_name = "p_nom_extendable" if c == "Link" else "s_nom_extendable"
            component = n.df(c)
            mask = (component.bus0 == bus) | (component.bus1 == bus)
            if mask.any():
                component.loc[mask, attr_name] = False
                n.df(c).update(component)

        # Remove the links which have "exp" in the name and are connected to the external buses
        links_to_remove = n.links[
            n.links.index.str.contains("exp")
            & (n.links.bus0.isin(external_buses_to_keep.index) | n.links.bus1.isin(external_buses_to_keep.index))
        ]
        n.mremove("Link", links_to_remove.index)

    # Update network topology
    n.determine_network_topology()


def calc_import_export_costs(n: pypsa.Network, carrier: str) -> float:
    """Calculates the average marginal cost for a given carrier."""
    gens = n.generators[n.generators.carrier == carrier]
    component = "Generator"
    if gens.empty:
        gens = n.links[n.links.carrier == carrier]
        component = "Link"
    if gens.empty:
        raise ValueError(f"No generators or links found for carrier to calculate imports/exports costs: {carrier}")
    costs = get_as_dense(n, component, "marginal_cost").loc[:, gens.index].mean().mean()
    if costs <= 0.01:
        raise ValueError(
            f"Average marginal cost for {carrier} is less than or equal to 0.01. Check the fuel costs configuration.",
        )
    return costs


def load_import_export_costs(eia_api: str, year: int) -> pd.DataFrame:
    """Loads fuel costs from EIA."""
    return FuelCosts(fuel="electricity", year=year, api=eia_api).get_data()


def format_import_export_costs(n: pypsa.Network, fuel_costs: pd.DataFrame) -> pd.DataFrame:
    """Formats fuel costs for BA mappings."""
    df = fuel_costs.copy()
    data = []

    buses = n.buses.copy()

    region_mapping = buses.set_index("country")["reeds_state"].to_dict()
    for region, state in region_mapping.items():
        for period in df.index.unique():
            temp = df[(df.index == period) & (df.state == state)]
            value = temp.value.mean()
            data.append([period, region, value, "usd/mwh"])
    return pd.DataFrame(data, columns=["period", "zone", "value", "units"]).set_index("period")


def format_flowgates_for_imports_exports(n: pypsa.Network, flowgates: pd.DataFrame, zone_col: str) -> pd.DataFrame:
    """Formats flowgates for zone mappings."""
    zones_in_model = n.buses[zone_col].unique()
    df = flowgates.copy()

    # only keep flowgates that connect inside to outside model scope
    df = df[df.r.isin(zones_in_model) ^ df.rr.isin(zones_in_model)]

    # reformat to sinlge value column for easier addition to network
    data = []
    for _, row in df.iterrows():
        if row.MW_f0 > 0:
            data.append([row.r, row.rr, row.MW_f0])
        if row.MW_r0 > 0:
            data.append([row.rr, row.r, row.MW_r0])

    return pd.DataFrame(data, columns=["r", "rr", "value"])


def convert_flowgates_to_state(flowgates: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Converts flowgates to state level."""
    mbshp = membership.set_index("ba")
    df = flowgates.copy()

    df["s"] = df.r.map(mbshp["st"])
    df["ss"] = df.rr.map(mbshp["st"])
    df = df.drop(columns=["r", "rr"])
    df = df.rename(columns={"s": "r", "ss": "rr"})
    return df


def add_elec_imports_exports(
    n: pypsa.Network,
    direction: str,
    flowgates: pd.DataFrame,
    fuel_costs: pd.DataFrame | float,
    co2_emissions: float = 0,
    zone_col: str = "reeds_zone",
):
    """Add electricity imports and exports to the network.

    These are capacity constrianed links to/from states outside the model spatial scope.
    """

    def _get_regions_2_add(n: pypsa.Network, flowgates: pd.DataFrame, zone_col: str) -> list[str]:
        """Gets regions to add import and export buses to."""
        unique_regions = set(flowgates.r.unique()) | set(flowgates.rr.unique())
        return [x for x in unique_regions if x not in n.buses[zone_col].unique()]

    def _add_import_export_carriers(n: pypsa.Network, direction: str, co2_emissions: float | None = None) -> None:
        """Adds import and export carriers to the network."""
        if direction == "imports":
            co2_emissions = 0 if not co2_emissions else co2_emissions
            n.add("Carrier", "imports", co2_emissions=co2_emissions, nice_name="Imports")
        elif direction == "exports":
            n.add("Carrier", "exports", co2_emissions=0, nice_name="Exports")
        else:
            raise ValueError(f"direction must be either imports or exports; received: {direction}")

    def _add_import_export_buses(n: pypsa.Network, regions_2_add: list[str], direction: str) -> None:
        """Adds import and export buses to the network."""
        if direction == "imports":
            suffix = "_imports"
            carrier = "imports"
        elif direction == "exports":
            suffix = "_exports"
            carrier = "exports"
        else:
            raise ValueError(f"direction must be either imports or exports; received: {direction}")

        # cant add in the reeds_state, reeds_zone, reeds_ba, interconnect, trans_reg, trans_grp
        # because this information has already been filtered out of the network

        n.madd(
            "Bus",
            regions_2_add,
            suffix=suffix,
            carrier=carrier,
            country=regions_2_add,
        )

    def _add_import_export_stores(n: pypsa.Network, regions_2_add: list[str], direction: str) -> None:
        """Adds import and export stores to the network."""
        if direction == "imports":
            n.madd(
                "Store",
                regions_2_add,
                bus=[f"{x}_imports" for x in regions_2_add],
                suffix="_imports",
                carrier="imports",
                e_nom_extendable=True,
                marginal_cost=0,
                e_nom=0,
                e_nom_max=np.inf,
                e_min=0,
                e_min_pu=-1,
                e_max_pu=0,
            )
        elif direction == "exports":
            n.madd(
                "Store",
                regions_2_add,
                bus=[f"{x}_exports" for x in regions_2_add],
                suffix="_exports",
                carrier="exports",
                e_nom_extendable=True,
                marginal_cost=0,
                e_nom=0,
                e_nom_max=np.inf,
                e_min=0,
                e_min_pu=0,
                e_max_pu=1,
            )
        else:
            raise ValueError(f"direction must be either imports or exports; received: {direction}")

    def _build_cost_timeseries(n: pypsa.Network, costs: pd.DataFrame, zone: str) -> pd.Series:
        """Builds a cost timeseries for a given state."""
        timesteps = n.snapshots.get_level_values("timestep")
        years = n.investment_periods
        cost_by_zone = costs[costs.zone == zone].drop(columns=["zone", "units"])
        dfs = []
        for year in years:
            df = cost_by_zone.copy()
            df.index = pd.to_datetime(df.index).map(lambda x: x.replace(year=year))
            df = df.resample("h").ffill().reindex(timesteps).ffill()
            df["year"] = year
            df = df.set_index(["year", df.index])  # df.index is timestep
            dfs.append(df)
        df = pd.concat(dfs)
        return df.reindex(n.snapshots)

    def _add_import_export_links(
        n: pypsa.Network,
        flowgates: pd.DataFrame,
        fuel_costs: pd.DataFrame | float | str,
        direction: str,
        zone_col: str = "reeds_zone",
    ) -> None:
        """Adds import and export links to the network."""
        costs = {}
        zones_in_model = n.buses[zone_col].dropna().unique()

        for _, row in flowgates.iterrows():
            zone_inside = row.r if row.r in zones_in_model else row.rr
            zone_outside = row.r if row.r not in zones_in_model else row.rr

            # extremely crude cashing for generating cost timeseries :|
            if zone_outside not in costs:
                if isinstance(fuel_costs, float):
                    costs[zone_inside] = fuel_costs
                elif isinstance(fuel_costs, pd.DataFrame):
                    costs[zone_inside] = _build_cost_timeseries(n, fuel_costs, zone_inside)
                else:
                    logger.warning("Setting marginal cost for electricity imports/exports to 0")
                    costs[zone_inside] = 0

            marginal_cost = costs[zone_inside]

            capacity = row.value

            """Structre of flowgates is given by:

                  r   rr     value
            0    p6   p8   488.117
            1    p8   p6   378.458
            2    p6   p9  4800.000
            ...
            """

            if direction == "imports":
                if row.r == zone_inside:  # originating at r is exports (ie r -> rr)
                    continue
                name = f"{zone_inside}_{zone_outside}_imports"
                bus0 = f"{zone_outside}_imports"
                bus1 = zone_inside
                carrier = "imports"
            else:
                if row.r == zone_outside:  # originating at rr is exports (ie rr -> r)
                    continue
                name = f"{zone_inside}_{zone_outside}_exports"
                bus0 = zone_inside
                bus1 = f"{zone_outside}_exports"
                carrier = "exports"
                if isinstance(marginal_cost, pd.Series):
                    marginal_cost = marginal_cost.mul(-1)  # constraint will limit exports

            mc = marginal_cost.value if isinstance(marginal_cost, pd.DataFrame) else marginal_cost

            n.add(
                "Link",
                name,
                bus0=bus0,
                bus1=bus1,
                carrier=carrier,
                p_nom_extendable=False,
                p_min_pu=0,
                p_max_pu=1,
                marginal_cost=mc,
                p_nom=capacity,
            )

    assert direction in ["imports", "exports"], f"direction must be either imports or exports; received: {direction}"

    regions_2_add = _get_regions_2_add(n, flowgates, zone_col)
    _add_import_export_carriers(n, direction, co2_emissions)
    _add_import_export_buses(n, regions_2_add, direction)
    _add_import_export_stores(n, regions_2_add, direction)
    _add_import_export_links(n, flowgates, fuel_costs, direction, zone_col)


def add_co2_storage(n: pypsa.Network, config: dict, co2_storage_csv: str, costs: pd.DataFrame, sector: bool):
    """Adds node level CO2 (underground) storage."""
    # get node level CO2 (underground) storage potential and cost from CSV file
    co2_storage = pd.read_csv(co2_storage_csv).set_index("node")

    # add carrier to represent CO2
    n.madd(
        "Carrier",
        ["co2"],
        color=config["plotting"]["tech_colors"]["co2"],
        nice_name=config["plotting"]["nice_names"]["co2"],
    )

    # add buses to represent node level CO2 captured by different processes
    n.madd(
        "Bus",
        co2_storage.index,
        suffix=" co2 capture",
        carrier="co2",
    )

    # add stores to represent node level CO2 (underground) storage
    n.madd(
        "Store",
        co2_storage.index,
        suffix=" co2 storage",
        bus=co2_storage.index + " co2 capture",
        e_nom_extendable=True,
        e_nom_max=co2_storage["potential [MtCO2]"] * 1e6,  # in tCO2
        marginal_cost=co2_storage["cost [USD/tCO2]"],
        carrier="co2",
    )

    # add carrier to represent CC only (i.e. without S)
    carriers = n.carriers.query("Carrier.str.endswith('CCS')")
    if not carriers.empty:
        n.madd(
            "Carrier",
            carriers.index.str.replace("CCS", "CC", regex=True),
            color=carriers["color"],
            nice_name=carriers["nice_name"].str.replace("Ccs", "Cc", regex=True),
        )

    # get CO2 intensity for gas and coal
    gas_co2_intensity = costs.loc["gas"]["co2_emissions"]
    coal_co2_intensity = costs.loc["coal"]["co2_emissions"]

    if sector:
        links = n.links.index.str.contains("CCS")
        if links.any():  # found links equipped with CCS
            # specify links' bus4 to point to their respective CO2 capture buses
            n.links.loc[links, "bus4"] = co2_storage.index + " co2 capture"

            # calculate efficiencies
            efficiency2 = []  # to node or state atmosphere bus (e.g. "p9 pwr atmosphere", "CA pwr atmosphere")
            efficiency4 = []  # to node co2 capture bus (e.g. "p9 co2 capture")
            for index in n.links.loc[links].index:
                link_efficiency = n.links.loc[index]["efficiency"]
                if "CCGT" in index:
                    efficiency = 1 / link_efficiency * gas_co2_intensity
                elif "coal" in index:
                    efficiency = 1 / link_efficiency * coal_co2_intensity
                else:
                    logger.warning(
                        f"Assuming a CO2 intensity equal to 1 given that link '{index}' is not powered by gas or coal",
                    )
                    efficiency = 1 / link_efficiency * 1
                cc_level = (
                    int(index.split("-")[1].split("CC")[0]) / 100
                )  # extract CC level from index (e.g. index "p1 CCGT-95CCS_2030" returns 0.95)
                efficiency2.append(efficiency * (1 - cc_level) / cc_level)
                efficiency4.append(efficiency)

            # set links' bus2 and bus4 efficiencies
            n.links.loc[links, "efficiency2"] = efficiency2
            n.links.loc[links, "efficiency4"] = efficiency4

            # remove storage cost from links' capital cost (given that they do not require technology to store CO2 anymore as this is done underground)
            n.links.loc[links, "capital_cost"] *= (
                0.95  # TODO: replace with a concrete storage cost (reducing 5% capital cost for the time being)
            )

            # replace substring "CCS" with just "CC" in links' names and carriers
            n.links.loc[links, "carrier"] = n.links.loc[links].carrier.str.replace("CCS", "CC", regex=True)
            n.links.index = n.links.index.str.replace("CCS", "CC", regex=True)

    else:  # sector-less
        generators = n.generators.index.str.contains("CCS")
        if generators.any():  # found generators equipped with CCS
            # remove storage cost from generators' capital cost (given that they do not require technology to store CO2 anymore as this is done underground)
            n.generators.loc[generators, "capital_cost"] *= (
                0.95  # TODO: replace with a concrete storage cost (reducing 5% capital cost for the time being)
            )

            # replace "CCS" with "CC" in generators' indexes/carriers description
            n.generators.loc[generators, "carrier"] = n.generators.loc[generators].carrier.str.replace(
                "CCS",
                "CC",
                regex=True,
            )
            n.generators.index = n.generators.index.str.replace("CCS", "CC", regex=True)

            # add buses to represent node level electricity CC generator
            indexes = n.generators.loc[generators].index
            n.madd(
                "Bus",
                indexes,
                carrier=n.generators.loc[generators].carrier,
            )

            # add buses to represent node level emitted CO2 by different processes
            granularity = config["dac"]["granularity"]
            if granularity == "nation":
                buses_atmosphere_unique = ["atmosphere"]
                buses_atmosphere = buses_atmosphere_unique
            else:
                if config["model_topology"]["transmission_network"] == "reeds":
                    elements = 1
                else:  # TAMU
                    elements = 2
                if granularity == "state":
                    buses = n.buses[["x", "y"]].query("x != 0 and y != 0").copy()
                    buses["geometry"] = buses.apply(lambda x: Point(x.x, x.y), axis=1)
                    buses_gdf = gpd.GeoDataFrame(buses, crs="EPSG:4269")
                    states_gdf = gpd.GeoDataFrame(
                        gpd.read_file(snakemake.input.county_shapes).dissolve("STUSPS")["geometry"],
                    )
                    buses_projected = buses_gdf.to_crs("EPSG:3857")
                    states_projected = states_gdf.to_crs("EPSG:3857")
                    states = gpd.sjoin_nearest(buses_projected, states_projected, how="left")["STUSPS"]
                    buses_atmosphere_unique = states.unique() + " atmosphere"
                    buses_atmosphere = [
                        "{} atmosphere".format(states.loc[" ".join(index.split(" ")[:elements])]) for index in indexes
                    ]
                else:  # node
                    buses_atmosphere_unique = [
                        "{} atmosphere".format(" ".join(index.split(" ")[:elements])) for index in indexes
                    ]
                    buses_atmosphere = buses_atmosphere_unique

            # add buses to represent (air) atmosphere where CO2 emissions are sent to
            n.madd(
                "Bus",
                buses_atmosphere_unique,
                carrier="co2",
            )

            # add stores to represent (air) atmosphere where CO2 emissions are stored
            n.madd(
                "Store",
                buses_atmosphere_unique,
                bus=buses_atmosphere_unique,
                e_nom_extendable=True,
                e_min_pu=-1,
                carrier="co2",
            )

            # calculate efficiencies
            efficiency2 = []  # to node or state atmosphere bus (e.g. "p9 atmosphere", "CA atmosphere")
            efficiency3 = []  # to node co2 capture bus (e.g. "p9 co2 capture")
            for index in indexes:
                generator_efficiency = n.generators.loc[index]["efficiency"]
                if "CCGT" in index:
                    efficiency = 1 / generator_efficiency * gas_co2_intensity
                elif "coal" in index:
                    efficiency = 1 / generator_efficiency * coal_co2_intensity
                else:
                    logger.warning(
                        f"Assuming a CO2 intensity equal to 1 given that generator '{index}' is not powered by gas or coal",
                    )
                    efficiency = 1 / generator_efficiency * 1
                cc_level = (
                    int(index.split("-")[1].split("CC")[0]) / 100
                )  # extract CC level from index (e.g. index "p1 CCGT-95CCS_2030" returns 0.95)
                efficiency2.append(efficiency)
                efficiency3.append(efficiency * (1 - cc_level) / cc_level)

            # add links to represent sending electricity (in MW) to the electricity bus (e.g. "p9" if ReEDS or "p100 0" if TAMU) as well as sending emitted CO2 (by the generator) to both the atmosphere bus and the co2 capture bus
            n.madd(
                "Link",
                indexes,
                bus0=indexes,
                bus1=n.generators.loc[generators]["bus"],
                bus2=buses_atmosphere,
                bus3=co2_storage.index + " co2 capture",
                efficiency=1,
                efficiency2=efficiency2,
                efficiency3=efficiency3,
                p_nom_extendable=True,
                capital_cost=0,
                marginal_cost=0,
                carrier=n.generators.loc[generators].carrier,
            )

            # (re-)attach generators to new buses (that represent node level CC generator)
            n.generators.loc[generators, "bus"] = indexes


def add_co2_network(n: pypsa.Network, config: dict):
    """Adds CO2 (transportation) network."""
    # get electricity connections
    if config["model_topology"]["transmission_network"] == "reeds":
        connections = n.links.query("carrier == 'AC' and not Link.str.endswith('exp')")
    else:  # TAMU
        connections = n.lines

    # calculate annualized capital cost
    number_years = n.snapshot_weightings.generators.sum() / 8760
    cost = (
        config["co2"]["network"]["capital_cost"]
        * calculate_annuity(config["co2"]["network"]["lifetime"], config["co2"]["network"]["discount_rate"])
        * number_years
    )

    # add links to represent CO2 (transportation) network based on electricity connections layout
    n.madd(
        "Link",
        connections.index,
        suffix=" co2 transport",
        bus0=connections["bus0"] + " co2 capture",
        bus1=connections["bus1"] + " co2 capture",
        efficiency=1,
        p_min_pu=-1,
        p_nom_extendable=True,
        length=connections["length"].values,
        capital_cost=cost * connections["length"].values,
        marginal_cost=config["co2"]["network"]["marginal_cost"],
        carrier="co2",
        lifetime=config["co2"]["network"]["lifetime"],
    )


def add_dac(n: pypsa.Network, config: dict, sector: bool):
    """Adds node level DAC capabilities."""
    # generate node level buses to represent emitted, captured and accounted CO2 and links to represent DAC in function of whether network is based on sectors or not
    if sector:
        # get DAC granularity/scope
        granularity = config["dac"]["granularity"]
        if granularity == "nation":
            granularity = "node"
            logger.warning(
                "Nation level DAC capabilities is not applicable for a network based on sectors - defaulting to node level instead",
            )

        # set number of elements based on electricity transmission network type
        if config["model_topology"]["transmission_network"] == "reeds":
            elements = 1
        else:  # TAMU
            elements = 2

        # get links that emit CO2 for all sectors
        links = n.links.query("bus2.str.endswith('-co2')")

        # set buses needed to create DAC links properly afterwards
        exists_atmosphere = set()
        exists_dac = set()
        buses_atmosphere = []
        buses_atmosphere_all = []
        buses_atmosphere_unique = []
        buses_co2_capture = []
        buses_co2_account = []
        buses_ac = []
        links_dac = []
        for index in links.index:
            bus2 = links.loc[index]["bus2"]  # e.g. "CA pwr-co2"
            node = " ".join(index.split(" ")[:elements])  # e.g. "p9" if ReEDS or "p100 0" if TAMU
            state = bus2.split(" ")[0]  # e.g. "CA"
            state_sector = bus2.split(" ")[1].split("-")[0]  # e.g. "pwr"
            if granularity == "node":
                atmosphere = f"{node} {state_sector} atmosphere"
            else:  # state
                atmosphere = f"{state} {state_sector} atmosphere"
            buses_atmosphere_all.append(atmosphere)
            if atmosphere not in exists_atmosphere:
                buses_atmosphere_unique.append(atmosphere)
                buses_co2_account.append(bus2)
                exists_atmosphere.add(atmosphere)
            dac = f"{node} {state_sector} dac"
            if dac not in exists_dac:
                buses_atmosphere.append(atmosphere)
                buses_co2_capture.append(f"{node} co2 capture")
                buses_ac.append(node)
                links_dac.append(dac)
                exists_dac.add(dac)

        # add node or state level buses to represent (air) atmosphere where CO2 emissions are sent to (on a per sector basis)
        n.madd(
            "Bus",
            buses_atmosphere_unique,
            carrier="co2",
        )

        # add links from node or state level buses that represent (air) atmosphere to state level buses tracking CO2 emissions (on a per sector basis)
        n.madd(
            "Link",
            buses_atmosphere_unique,
            bus0=buses_atmosphere_unique,
            bus1=buses_co2_account,
            efficiency=1,
            p_nom_extendable=True,
            capital_cost=0,
            marginal_cost=0,
            carrier="co2",
        )

        # redirect links that emit CO2 to node or state level buses that represent (air) atmosphere   # e.g. "p1 trn atmosphere"
        n.links.loc[links.index, "bus2"] = buses_atmosphere_all

    else:  # sector-less
        # set buses needed to create DAC links properly afterwards
        buses_atmosphere = n.links.query("bus2.str.endswith('atmosphere')")["bus2"].values
        buses_co2_capture = n.buses.query("Bus.str.endswith(' co2 capture')").index
        buses_ac = buses_co2_capture.str.replace(" co2 capture", "")
        links_dac = buses_co2_capture.str.replace(" co2 capture", " dac")

    # add carrier to represent DAC
    n.madd(
        "Carrier",
        ["dac"],
        color=config["plotting"]["tech_colors"]["dac"],
        nice_name=config["plotting"]["nice_names"]["dac"],
    )

    # calculate annualized capital cost
    number_years = n.snapshot_weightings.generators.sum() / 8760
    cost = (
        config["dac"]["capital_cost"]
        * calculate_annuity(config["dac"]["lifetime"], config["dac"]["discount_rate"])
        * number_years
    )

    # add links to represent node level DAC capabilities
    n.madd(
        "Link",
        links_dac,
        bus0=buses_atmosphere,
        bus1=buses_co2_capture,
        bus2=buses_ac,
        efficiency=1,  # in tCO2
        efficiency2=-config["dac"]["electricity_input"],  # in MWh (for each tCO2)
        p_nom_extendable=True,
        capital_cost=cost,
        marginal_cost=0,
        carrier="dac",
        lifetime=config["dac"]["lifetime"],
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_extra_components",
            interconnect="western",
            simpl="100",
            clusters="58m",
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
    apply_ptc(n, snakemake.config["costs"]["ptc_modifier"], costs)
    apply_max_annual_growth_rate(n, snakemake.config["costs"]["max_growth"])
    add_nice_carrier_names(n, snakemake.config)
    add_co2_emissions(n, costs_dict[n.investment_periods[0]], n.carriers.index)

    dr_config = snakemake.params.demand_response
    if dr_config:
        add_demand_response(n, dr_config)

    trim_network_config = snakemake.params.trim_network
    imports_config = snakemake.params.imports
    exports_config = snakemake.params.exports

    assert not (
        snakemake.params.trim_network and (imports_config.get("enable", False) or exports_config.get("enable", False))
    ), "trim_network and imports/exports cannot be used together"

    if snakemake.params.trim_network:
        trim_network(n, trim_network_config)

    if snakemake.params.transmission_network == "reeds":
        # flowgates to limit the capacity (removed later if configured capacity limit is inf)
        flowgates = pd.read_csv(snakemake.input.flowgates)
        if snakemake.params.topological_boundaries == "state":
            zone_col = "reeds_state"
            membership = pd.read_csv(snakemake.input.reeds_memberships)
            flowgates = convert_flowgates_to_state(flowgates, membership)
            flowgates = format_flowgates_for_imports_exports(n, flowgates, zone_col)
            flowgates = flowgates.groupby(["r", "rr"], as_index=False).sum()
        elif snakemake.params.topological_boundaries == "county":
            zone_col = "county"
            flowgates = format_flowgates_for_imports_exports(n, flowgates, zone_col)
        elif snakemake.params.topological_boundaries == "reeds_zone":
            zone_col = "reeds_zone"
            flowgates = format_flowgates_for_imports_exports(n, flowgates, zone_col)
        else:
            raise ValueError(f"Invalid topological boundaries: {snakemake.params.topological_boundaries}")

    # Electricity imports configuration
    if imports_config.get("enable", False) and snakemake.params.transmission_network == "reeds":
        co2_emissions = imports_config.get("co2_emissions", 0)

        weather_year = snakemake.params.weather_year
        if isinstance(weather_year, list):
            year = weather_year[0]

        import_flowgates = flowgates.copy()
        if not imports_config.get("capacity_limit", True):
            import_flowgates["value"] = np.inf

        import_costs = imports_config.get("costs", False)

        if isinstance(import_costs, float):  # user defined value
            fuel_costs = import_costs
        elif isinstance(import_costs, str):  # name of carrier
            fuel_costs = calc_import_export_costs(n, import_costs)
        elif isinstance(import_costs, bool):  # wholesale market cost
            if import_costs:
                fuel_costs = load_import_export_costs(snakemake.params.eia_api, year)
                fuel_costs = format_import_export_costs(n, fuel_costs)
            else:
                fuel_costs = 0
                logger.warning("No imports costs provided, setting to 0. Check the imports configuration.")
        else:
            raise ValueError(f"'imports.costs' must be a float, boolean, or string. Received: {import_costs}")

        add_elec_imports_exports(n, "imports", import_flowgates, fuel_costs, co2_emissions, zone_col)

    # Electricity exports configuration
    if exports_config.get("enable", False) and snakemake.params.transmission_network == "reeds":
        co2_emissions = 0

        weather_year = snakemake.params.weather_year
        if isinstance(weather_year, list):
            year = weather_year[0]

        # flowgates to limit the capacity
        export_flowgates = flowgates.copy()
        if not exports_config.get("capacity_limit", True):
            export_flowgates["value"] = np.inf

        export_costs = exports_config.get("costs", False)

        if isinstance(export_costs, float):  # user defined value
            fuel_costs = export_costs
        elif isinstance(export_costs, str):  # name of carrier
            fuel_costs = calc_import_export_costs(n, export_costs)
        elif isinstance(export_costs, bool):  # wholesale market cost
            if export_costs:
                fuel_costs = load_import_export_costs(snakemake.params.eia_api, year)
                fuel_costs = format_import_export_costs(n, fuel_costs)
            else:
                fuel_costs = 0
                logger.warning("No exports costs provided, setting to 0. Check the exports configuration.")
        else:
            raise ValueError(f"'exports.costs' must be a float, boolean, or string. Received: {export_costs}")

        add_elec_imports_exports(n, "exports", export_flowgates, fuel_costs, co2_emissions)

    if snakemake.config["scenario"]["sector"] == "E":
        # add node level CO2 (underground) storage
        if snakemake.config["co2"]["storage"]:
            logger.info("Adding node level CO2 (underground) storage")
            add_co2_storage(n, snakemake.config, snakemake.input.co2_storage, costs, False)

        # add CO2 (transportation) network
        if snakemake.config["co2"]["network"]["enable"]:
            if snakemake.config["co2"]["storage"]:
                logger.info("Adding CO2 (transportation) network")
                add_co2_network(n, snakemake.config)
            else:
                logger.warning(
                    "Not adding CO2 (transportation) network given that CO2 (underground) storage is not enabled",
                )

        # add node level DAC capabilities
        if snakemake.config["dac"]["enable"]:
            if snakemake.config["co2"]["storage"]:
                logger.info("Adding DAC capabilities")
                add_dac(n, snakemake.config, False)
            else:
                logger.warning(
                    "Not adding DAC capabilities given that CO2 (underground) storage is not enabled",
                )

    n.consistency_check()
    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
