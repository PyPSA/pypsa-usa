# PyPSA USA Authors
"""
**Description**

This module integrates data produced by `build_renewable_profiles`, `build_demand`, `build_cost_data`, `build_fuel_prices`, and `build_base_network` to create a network model that includes generators, demand, and costs. The module attaches generators, storage units, and loads to the network created by `build_base_network`. Each generator is assigned regional capital costs, and regional and daily or monthly marginal costs.

Extendable generators are assigned a maximum capacity based on land-use constraints defined in `build_renewable_profiles`.

**Relevant Settings**

.. code:: yaml

    network_configuration:

    snapshots:
        start:
        end:
        inclusive:

    electricity:

.. seealso::
    Documentation of the configuration file `config/config.yaml` at :ref:`costs_cf`,
    :ref:`electricity_cf`, :ref:`load_cf`, :ref:`renewable_cf`, :ref:`lines_cf`

**Inputs**

- ``resources/costs.csv``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.
- ``resources/demand.csv`` Hourly per-country load profiles.
- ``resources/regions_onshore.geojson``: confer :ref:`busregions`
- ``resources/profile_{}.nc``: all technologies in ``config["renewables"].keys()``, confer :ref:`renewableprofiles`.
- ``networks/elec_base_network.nc``: confer :ref:`base`
- ``resources/ng_fuel_prices.csv``: Natural gas fuel prices by state and BA.

**Outputs**

- ``networks/elec_base_network_l_pp.nc``
"""


import logging
import os
import random
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Union

import constants as const
import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import (
    configure_logging,
    export_network_for_gis_mapping,
    local_to_utc,
    test_network_datatype_consistency,
    update_p_nom_max,
)
from scipy import sparse
from shapely.geometry import Point
from shapely.prepared import prep
from sklearn.neighbors import BallTree

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


# can we get rid of this function and use add_mising_carriers instead?
def _add_missing_carriers_from_costs(n, costs, carriers):
    missing_carriers = pd.Index(carriers).difference(n.carriers.index)
    if missing_carriers.empty:
        return

    emissions_cols = (
        costs.columns.to_series().loc[lambda s: s.str.endswith("_emissions")].values
    )
    suptechs = missing_carriers.str.split("-").str[0]
    emissions = costs.loc[suptechs, emissions_cols].fillna(0.0)
    emissions.index = missing_carriers
    n.import_components_from_dataframe(emissions, "Carrier")


def sanitize_carriers(n, config):
    """
    Sanitize the carrier information in a PyPSA Network object.

    The function ensures that all unique carrier names are present in the network's
    carriers attribute, and adds nice names and colors for each carrier according
    to the provided configuration dictionary.

    Parameters
    ----------
    n : pypsa.Network
        A PyPSA Network object that represents an electrical power system.
    config : dict
        A dictionary containing configuration information, specifically the
        "plotting" key with "nice_names" and "tech_colors" keys for carriers.

    Returns
    -------
    None
        The function modifies the 'n' PyPSA Network object in-place, updating the
        carriers attribute with nice names and colors.

    Warnings
    --------
    Raises a warning if any carrier's "tech_colors" are not defined in the config dictionary.
    """

    for c in n.iterate_components():
        if "carrier" in c.df:
            add_missing_carriers(n, c.df.carrier)

    carrier_i = n.carriers.index
    nice_names = (
        pd.Series(config["plotting"]["nice_names"])
        .reindex(carrier_i)
        .fillna(carrier_i.to_series().str.title())
    )
    n.carriers["nice_name"] = n.carriers.nice_name.where(
        n.carriers.nice_name != "",
        nice_names,
    )
    colors = pd.Series(config["plotting"]["tech_colors"]).reindex(carrier_i)
    if colors.isna().any():
        missing_i = list(colors.index[colors.isna()])
        logger.warning(f"tech_colors for carriers {missing_i} not defined in config.")
    n.carriers["color"] = n.carriers.color.where(n.carriers.color != "", colors)


def add_co2_emissions(n, costs, carriers):
    """
    Add CO2 emissions to the network's carriers attribute.
    """
    suptechs = n.carriers.loc[carriers].index.str.split("-").str[0]
    n.carriers.loc[carriers, "co2_emissions"] = costs.co2_emissions[suptechs].values


def load_costs(
    tech_costs: str,
    config: dict[str, Any],
    max_hours: dict[str, Union[int, float]],
    Nyears: float = 1.0,
) -> pd.DataFrame:

    # set all asset costs and other parameters
    costs = pd.read_csv(tech_costs, index_col=[0, 1]).sort_index()

    # correct units to MW
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
    costs.unit = costs.unit.str.replace("/kW", "/MW")

    # polulate missing values with user provided defaults
    fill_values = config["fill_values"]
    costs = costs.value.unstack().fillna(fill_values)

    costs["capital_cost"] = (
        (
            calculate_annuity(costs["lifetime"], costs["discount rate"])
            + costs["FOM"] / 100.0
        )
        * costs["investment"]
        * Nyears
    )

    costs.at["OCGT", "fuel"] = costs.at["gas", "fuel"]
    costs.at["CCGT", "fuel"] = costs.at["gas", "fuel"]

    costs["marginal_cost"] = costs["VOM"] + costs["fuel"] / costs["efficiency"]

    costs = costs.rename(columns={"CO2 intensity": "co2_emissions"})

    costs.at["OCGT", "co2_emissions"] = costs.at["gas", "co2_emissions"]
    costs.at["CCGT", "co2_emissions"] = costs.at["gas", "co2_emissions"]

    costs.at["solar", "capital_cost"] = (
        config["rooftop_share"] * costs.at["solar-rooftop", "capital_cost"]
        + (1 - config["rooftop_share"]) * costs.at["solar-utility", "capital_cost"]
    )

    def costs_for_storage(store, link1, link2=None, max_hours=1.0):
        capital_cost = link1["capital_cost"] + max_hours * store["capital_cost"]
        if link2 is not None:
            capital_cost += link2["capital_cost"]
        return pd.Series(
            dict(capital_cost=capital_cost, marginal_cost=0.0, co2_emissions=0.0),
        )

    costs.loc["battery"] = costs_for_storage(
        costs.loc["battery storage"],
        costs.loc["battery inverter"],
        max_hours=max_hours["battery"],
    )
    costs.loc["H2"] = costs_for_storage(
        costs.loc["hydrogen storage underground"],
        costs.loc["fuel cell"],
        costs.loc["electrolysis"],
        max_hours=max_hours["H2"],
    )

    for attr in ("marginal_cost", "capital_cost"):
        overwrites = config.get(attr)
        if overwrites is not None:
            overwrites = pd.Series(overwrites)
            costs.loc[overwrites.index, attr] = overwrites

    return costs


def add_annualized_capital_costs(
    costs: pd.DataFrame,
    Nyears: float = 1.0,
) -> pd.DataFrame:
    """
    Adds column to calculate annualized capital costs only.
    """

    costs["investment_annualized"] = (
        calculate_annuity(costs["lifetime"], costs["discount rate"])
        * costs["investment"]
        * Nyears
    )
    return costs


def shapes_to_shapes(orig, dest):
    """
    Adopted from vresutils.transfer.Shapes2Shapes()
    """
    orig_prepped = list(map(prep, orig))
    transfer = sparse.lil_matrix((len(dest), len(orig)), dtype=float)

    for i, j in product(range(len(dest)), range(len(orig))):
        if orig_prepped[j].intersects(dest[i]):
            area = orig[j].intersection(dest[i]).area
            transfer[i, j] = area / dest[i].area

    return transfer


def clean_locational_multiplier(df: pd.DataFrame):
    """
    Updates format of locational multiplier data.
    """
    df = df.fillna(1)
    df = df[["State", "Location Variation"]]
    return df.groupby("State").mean()


def update_capital_costs(
    n: pypsa.Network,
    carrier: str,
    costs: pd.DataFrame,
    multiplier: pd.DataFrame,
    Nyears: float = 1.0,
):
    """
    Applies regional multipliers to capital cost data.
    """

    # map generators to states
    bus_state_mapper = n.buses.to_dict()["state"]
    gen = n.generators[n.generators.carrier == carrier].copy()  # copy with warning
    gen["state"] = gen.bus.map(bus_state_mapper)
    gen = gen[
        gen["state"].isin(multiplier.index)
    ]  # drops any regions that do not have cost multipliers

    # log any states that do not have multipliers attached
    missed = gen[~gen["state"].isin(multiplier.index)]
    if not missed.empty:
        logger.warning(f"CAPEX cost multiplier not applied to {missed.state.unique()}")

    # apply multiplier

    # commented code is if applying multiplier to (capex + fom)
    # gen["capital_cost"] = gen.apply(
    #     lambda x: x["capital_cost"] * multiplier.at[x["state"], "Location Variation"], axis=1)

    # apply multiplier to annualized capital investment cost
    gen["investment"] = gen.apply(
        lambda x: costs.at[carrier, "investment_annualized"]
        * multiplier.at[x["state"], "Location Variation"],
        axis=1,
    )

    # get fixed costs based on overnight capital costs with multiplier applied
    gen["fom"] = gen["investment"] * (costs.at[carrier, "FOM"] / 100.0) * Nyears

    # find final annualized capital cost
    gen["capital_cost"] = gen["investment"] + gen["fom"]

    # overwrite network generator dataframe with updated values
    n.generators.loc[gen.index] = gen


def apply_dynamic_pricing(
    n: pypsa.Network,
    carrier: str,
    geography: str,
    df: pd.DataFrame,
    vom: float = 0,
):
    """
    Applies user-supplied dynamic pricing.

    Arguments
    ---------
    n: pypsa.Network,
    carrier: str,
        carrier to apply fuel cost data to (ie. Gas)
    geography: str,
        column of geography to search over (ie. balancing_area, state, reeds_zone, ...)
    df: pd.DataFrame,
        Fuel costs data
    vom: float = 0
        Additional flat $/MWh cost to add onto the fuel costs
    """

    assert geography in n.buses.columns

    gens = n.generators.copy()
    gens[geography] = gens.bus.map(n.buses[geography])
    gens = gens[(gens.carrier == carrier) & (gens[geography].isin(df.columns))]

    eff = n.get_switchable_as_dense("Generator", "efficiency").T
    eff = eff[eff.index.isin(gens.index)].T
    eff.columns.name = ""

    fuel_cost_per_gen = {gen: df[gens.at[gen, geography]] for gen in gens.index}
    fuel_costs = pd.DataFrame.from_dict(fuel_cost_per_gen)
    fuel_costs.index = pd.to_datetime(fuel_costs.index)

    marginal_costs = fuel_costs.div(eff, axis=1)
    marginal_costs = marginal_costs + vom

    # drop any data that has been assigned at a coarser resolution
    n.generators_t["marginal_cost"] = n.generators_t["marginal_cost"][
        [x for x in n.generators_t["marginal_cost"] if x not in marginal_costs]
    ]

    # assign new marginal costs
    n.generators_t["marginal_cost"] = n.generators_t["marginal_cost"].join(
        marginal_costs,
        how="inner",
    )


def update_transmission_costs(n, costs, length_factor=1.0):
    # TODO: line length factor of lines is applied to lines and links.
    # Separate the function to distinguish

    n.lines["capital_cost"] = (
        n.lines["length"] * length_factor * costs.at["HVAC overhead", "capital_cost"]
    )

    if n.links.empty:
        return

    dc_b = n.links.carrier == "DC"

    # If there are no dc links, then the 'underwater_fraction' column
    # may be missing. Therefore we have to return here.
    if n.links.loc[dc_b].empty:
        return

    costs = (
        n.links.loc[dc_b, "length"]
        * length_factor
        * (
            (1.0 - n.links.loc[dc_b, "underwater_fraction"])
            * costs.at["HVDC overhead", "capital_cost"]
            + n.links.loc[dc_b, "underwater_fraction"]
            * costs.at["HVDC submarine", "capital_cost"]
        )
        + costs.at["HVDC inverter pair", "capital_cost"]
    )
    n.links.loc[dc_b, "capital_cost"] = costs


def attach_hydro(n, costs, ppl, profile_hydro, hydro_capacities, carriers, **params):
    add_missing_carriers(n, carriers)
    add_co2_emissions(n, costs, carriers)

    ppl = (
        ppl.query('carrier == "hydro"')
        .reset_index(drop=True)
        .rename(index=lambda s: str(s) + " hydro")
    )
    ror = ppl.query('technology == "Run-Of-River"')
    phs = ppl.query('technology == "Pumped Storage"')
    hydro = ppl.query('technology == "Reservoir"')

    country = ppl["bus"].map(n.buses.country).rename("country")

    inflow_idx = ror.index.union(hydro.index)
    if not inflow_idx.empty:
        dist_key = ppl.loc[inflow_idx, "p_nom"].groupby(country).transform(normed)

        with xr.open_dataarray(profile_hydro) as inflow:
            inflow_countries = pd.Index(country[inflow_idx])
            missing_c = inflow_countries.unique().difference(
                inflow.indexes["countries"],
            )
            assert missing_c.empty, (
                f"'{profile_hydro}' is missing "
                f"inflow time-series for at least one country: {', '.join(missing_c)}"
            )

            inflow_t = (
                inflow.sel(countries=inflow_countries)
                .rename({"countries": "name"})
                .assign_coords(name=inflow_idx)
                .transpose("time", "name")
                .to_pandas()
                .multiply(dist_key, axis=1)
            )

    if "ror" in carriers and not ror.empty:
        n.madd(
            "Generator",
            ror.index,
            carrier="ror",
            bus=ror["bus"],
            p_nom=ror["p_nom"],
            efficiency=costs.at["ror", "efficiency"],
            capital_cost=costs.at["ror", "capital_cost"],
            weight=ror["p_nom"],
            p_max_pu=(
                inflow_t[ror.index]
                .divide(ror["p_nom"], axis=1)
                .where(lambda df: df <= 1.0, other=1.0)
            ),
        )

    if "PHS" in carriers and not phs.empty:
        # fill missing max hours to params value and
        # assume no natural inflow due to lack of data
        max_hours = params.get("PHS_max_hours", 6)
        phs = phs.replace({"max_hours": {0: max_hours}})
        n.madd(
            "StorageUnit",
            phs.index,
            carrier="PHS",
            bus=phs["bus"],
            p_nom=phs["p_nom"],
            capital_cost=costs.at["PHS", "capital_cost"],
            max_hours=phs["max_hours"],
            efficiency_store=np.sqrt(costs.at["PHS", "efficiency"]),
            efficiency_dispatch=np.sqrt(costs.at["PHS", "efficiency"]),
            cyclic_state_of_charge=True,
        )

    if "hydro" in carriers and not hydro.empty:
        hydro_max_hours = params.get("hydro_max_hours")

        assert hydro_max_hours is not None, "No path for hydro capacities given."

        hydro_stats = pd.read_csv(
            hydro_capacities,
            comment="#",
            na_values="-",
            index_col=0,
        )
        e_target = hydro_stats["E_store[TWh]"].clip(lower=0.2) * 1e6
        e_installed = hydro.eval("p_nom * max_hours").groupby(hydro.country).sum()
        e_missing = e_target - e_installed
        missing_mh_i = hydro.query("max_hours.isnull()").index

        if hydro_max_hours == "energy_capacity_totals_by_country":
            # watch out some p_nom values like IE's are totally underrepresented
            max_hours_country = (
                e_missing / hydro.loc[missing_mh_i].groupby("country").p_nom.sum()
            )

        elif hydro_max_hours == "estimate_by_large_installations":
            max_hours_country = (
                hydro_stats["E_store[TWh]"] * 1e3 / hydro_stats["p_nom_discharge[GW]"]
            )

        max_hours_country.clip(0, inplace=True)

        missing_countries = pd.Index(hydro["country"].unique()).difference(
            max_hours_country.dropna().index,
        )
        if not missing_countries.empty:
            logger.warning(
                "Assuming max_hours=6 for hydro reservoirs in the countries: {}".format(
                    ", ".join(missing_countries),
                ),
            )
        hydro_max_hours = hydro.max_hours.where(
            hydro.max_hours > 0,
            hydro.country.map(max_hours_country),
        ).fillna(6)

        n.madd(
            "StorageUnit",
            hydro.index,
            carrier="hydro",
            bus=hydro["bus"],
            p_nom=hydro["p_nom"],
            max_hours=hydro_max_hours,
            capital_cost=costs.at["hydro", "capital_cost"],
            marginal_cost=costs.at["hydro", "marginal_cost"],
            p_max_pu=1.0,  # dispatch
            p_min_pu=0.0,  # store
            efficiency_dispatch=costs.at["hydro", "efficiency"],
            efficiency_store=0.0,
            cyclic_state_of_charge=True,
            inflow=inflow_t.loc[:, hydro.index],
        )


def attach_breakthrough_renewable_plants(
    n,
    fn_plants,
    renewable_carriers,
    extendable_carriers,
    costs,
):

    _add_missing_carriers_from_costs(n, costs, renewable_carriers)

    plants = pd.read_csv(fn_plants, dtype={"bus_id": str}, index_col=0).query(
        "bus_id in @n.buses.index",
    )
    plants.replace(["wind_offshore"], ["offwind"], inplace=True)

    for tech in renewable_carriers:
        tech_plants = plants.query("type == @tech")
        tech_plants.index = tech_plants.index.astype(str)

        logger.info(f"Adding {len(tech_plants)} {tech} generators to the network.")

        if tech in ["wind", "offwind"]:
            p = pd.read_csv(snakemake.input["wind_breakthrough"], index_col=0)
        else:
            p = pd.read_csv(snakemake.input[f"{tech}_breakthrough"], index_col=0)
        intersection = set(p.columns).intersection(
            tech_plants.index,
        )  # filters by plants ID for the plants of type tech
        p = p[list(intersection)]

        Nhours = len(n.snapshots)
        p = p.iloc[:Nhours, :]  # hotfix to fit 2016 renewable data to load data

        p.index = n.snapshots
        p.columns = p.columns.astype(str)

        if (tech_plants.Pmax == 0).any():
            # p_nom is the maximum of {Pmax, dispatch}
            p_nom = pd.concat([p.max(axis=0), tech_plants["Pmax"]], axis=1).max(axis=1)
            p_max_pu = (p[p_nom.index] / p_nom).fillna(0)  # some values remain 0
        else:
            p_nom = tech_plants.Pmax
            p_max_pu = p[tech_plants.index] / p_nom

        n.madd(
            "Generator",
            tech_plants.index,
            bus=tech_plants.bus_id,
            p_nom_min=p_nom,
            p_nom=p_nom,
            marginal_cost=tech_plants.GenIOB
            * tech_plants.GenFuelCost,  # (MMBTu/MW) * (USD/MMBTu) = USD/MW
            # marginal_cost_quadratic = tech_plants.GenIOC * tech_plants.GenFuelCost,
            capital_cost=costs.at[tech, "capital_cost"],
            p_max_pu=p_max_pu,  # timeseries of max power output pu
            p_nom_extendable=tech in extendable_carriers["Generator"],
            carrier=tech,
            weight=1.0,
            efficiency=costs.at[tech, "efficiency"],
        )
    return n


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


def normed(s):
    """
    Normalize a pandas.Series to sum to 1.
    """
    return s / s.sum()


def calculate_annuity(n, r):
    """
    Calculate the annuity factor for an asset with lifetime n years and.

    discount rate of r, e.g. annuity(20, 0.05) * 20 = 1.6
    """
    if isinstance(r, pd.Series):
        return pd.Series(1 / n, index=r.index).where(
            r == 0,
            r / (1.0 - 1.0 / (1.0 + r) ** n),
        )
    elif r > 0:
        return r / (1.0 - 1.0 / (1.0 + r) ** n)
    else:
        return 1 / n


def add_missing_carriers(n, carriers):
    """
    Function to add missing carriers to the network without raising errors.
    """
    missing_carriers = set(carriers) - set(n.carriers.index)
    if len(missing_carriers) > 0:
        n.madd("Carrier", missing_carriers)


def add_missing_fuel_cost(plants, costs_fn):
    fuel_cost = pd.read_csv(costs_fn, index_col=0, skiprows=3)
    plants["fuel_cost"] = plants.fuel_type.map(fuel_cost.fuel_price_per_mmbtu)
    return plants


def add_missing_heat_rates(plants, heat_rates_fn):
    heat_rates = pd.read_csv(heat_rates_fn, index_col=0, skiprows=3)
    heat_rates = heat_rates.loc[heat_rates.heat_rate_btu_per_kwh > 0]
    hr_mapped = (
        plants.fuel_type.map(heat_rates.heat_rate_btu_per_kwh) / 1000
    )  # convert to mmbtu/mwh
    plants["heat_rate"].fillna(hr_mapped, inplace=True)
    return plants


def match_plant_to_bus(n, plants):
    plants_matched = plants.copy()
    plants_matched["bus_assignment"] = None

    buses = n.buses.copy()
    buses["geometry"] = gpd.points_from_xy(buses["x"], buses["y"])

    # from: https://stackoverflow.com/questions/58893719/find-nearest-point-in-other-dataframe-with-a-lot-of-data
    # Create a BallTree
    tree = BallTree(buses[["x", "y"]].values, leaf_size=2)
    # Query the BallTree on each feature from 'appart' to find the distance
    # to the nearest 'pharma' and its id
    plants_matched["distance_nearest"], plants_matched["id_nearest"] = tree.query(
        plants_matched[
            ["longitude", "latitude"]
        ].values,  # The input array for the query
        k=1,  # The number of nearest neighbors
    )
    plants_matched.bus_assignment = (
        buses.reset_index().iloc[plants_matched.id_nearest].Bus.values
    )
    plants_matched.drop(columns=["id_nearest"], inplace=True)

    return plants_matched


def attach_renewable_capacities_to_atlite(
    n: pypsa.Network,
    plants_df: pd.DataFrame,
    renewable_carriers: list,
):
    plants = plants_df.query(
        "bus_assignment in @n.buses.index",
    )
    for tech in renewable_carriers:
        plants_filt = plants.query("carrier == @tech")
        if plants_filt.empty:
            continue
        generators_tech = n.generators[n.generators.carrier == tech]
        caps_per_bus = (
            plants_filt[["bus_assignment", "p_nom"]]
            .groupby("bus_assignment")
            .sum()
            .p_nom
        )  # namplate capacity per bus
        # TODO: #16 Gens excluded from atlite profiles bc of landuse/etc will not be able to be attached if in the breakthrough network
        if caps_per_bus[~caps_per_bus.index.isin(generators_tech.bus)].sum() > 0:
            missing_capacity = caps_per_bus[
                ~caps_per_bus.index.isin(generators_tech.bus)
            ].sum()
            logger.info(
                f"There are {np.round(missing_capacity/1000,4)} GW of {tech} plants that are not in the network. See git issue #16.",
            )

        logger.info(
            f"{np.round(caps_per_bus.sum()/1000,2)} GW of {tech} capacity added.",
        )
        n.generators.p_nom.update(generators_tech.bus.map(caps_per_bus).dropna())
        n.generators.p_nom_min.update(generators_tech.bus.map(caps_per_bus).dropna())


def attach_demand(n: pypsa.Network, demand_per_bus_fn: str):
    """
    Add demand to network from specified configuration setting.

    Returns network with demand added.
    """
    demand_per_bus = pd.read_csv(demand_per_bus_fn, index_col=0)
    demand_per_bus.index = pd.to_datetime(demand_per_bus.index)
    n.madd(
        "Load",
        demand_per_bus.columns,
        bus=demand_per_bus.columns,
        p_set=demand_per_bus,
        carrier="AC",
    )


def attach_conventional_generators(
    n: pypsa.Network,
    costs: pd.DataFrame,
    plants: pd.DataFrame,
    conventional_carriers: list,
    extendable_carriers: list,
    conventional_params,
    renewable_carriers: list,
    conventional_inputs,
    unit_commitment=None,
    fuel_price=None,
):
    carriers = [
        carrier
        for carrier in set(conventional_carriers)
        | set(extendable_carriers["Generator"])
        if carrier not in renewable_carriers
    ]
    add_missing_carriers(n, carriers)
    add_co2_emissions(n, costs, carriers)

    plants = (
        plants.query("carrier in @carriers")
        .join(costs, on="carrier", rsuffix="_r")
        .rename(index=lambda s: "C" + str(s))
    )

    plants["efficiency"] = plants.efficiency.fillna(plants.efficiency_r)

    if unit_commitment is not None:
        committable_attrs = plants.carrier.isin(unit_commitment).to_frame("committable")
        for attr in unit_commitment.index:
            default = pypsa.components.component_attrs["Generator"].default[attr]
            committable_attrs[attr] = plants.carrier.map(
                unit_commitment.loc[attr],
            ).fillna(
                default,
            )
    else:
        committable_attrs = {}

    marginal_cost = plants.carrier.map(costs.VOM) + plants.marginal_cost

    # Define generators using modified ppl DataFrame
    caps = plants.groupby("carrier").p_nom.sum().div(1e3).round(2)
    logger.info(f"Adding {len(plants)} generators with capacities [GW] \n{caps}")
    n.madd(
        "Generator",
        plants.index,
        carrier=plants.carrier,
        bus=plants.bus_assignment,
        p_nom_min=plants.p_nom.where(
            plants.carrier.isin(conventional_carriers),
            0,
        ),  # enforces that plants cannot be retired/sold-off at their capital cost
        p_nom=plants.p_nom.where(plants.carrier.isin(conventional_carriers), 0),
        p_nom_extendable=plants.carrier.isin(extendable_carriers["Generator"]),
        ramp_limit_up=plants.ramp_limit_up,
        ramp_limit_down=plants.ramp_limit_down,
        efficiency=plants.efficiency.round(3),
        marginal_cost=marginal_cost,
        capital_cost=plants.capital_cost,
        build_year=plants.build_year.fillna(0).astype(int),
        lifetime=(plants.dateout - plants.build_year).fillna(np.inf),
        **committable_attrs,
    )

    # Add fuel and VOM costs to the network
    n.generators.loc[plants.index, "vom_cost"] = plants.carrier.map(costs.VOM)
    n.generators.loc[plants.index, "fuel_cost"] = plants.marginal_cost
    n.generators.loc[plants.index, "heat_rate"] = plants.heat_rate
    n.generators.loc[plants.index, "ba_eia"] = plants.balancing_authority_code
    n.generators.loc[plants.index, "ba_ads"] = plants.ads_balancing_area


def attach_wind_and_solar(
    n: pypsa.Network,
    costs: pd.DataFrame,
    input_profiles: str,
    carriers: list[str],
    extendable_carriers: dict[str, list[str]],
    line_length_factor=1,
):
    """
    Attached Atlite Calculated wind and solar capacity factor profiles to the
    network.
    """
    add_missing_carriers(n, carriers)
    for car in carriers:
        if car == "hydro":
            continue

        with xr.open_dataset(getattr(input_profiles, "profile_" + car)) as ds:
            if ds.indexes["bus"].empty:
                continue

            supcar = car.split("-", 2)[0]
            if supcar == "offwind" or supcar == "offwind_floating":
                if supcar == "offwind_floating":
                    supcar = "offwind"
                underwater_fraction = ds["underwater_fraction"].to_pandas()
                connection_cost = (
                    line_length_factor
                    * ds["average_distance"].to_pandas()
                    * (
                        underwater_fraction
                        * costs.at[supcar + "-ac-connection-submarine", "capital_cost"]
                        + (1.0 - underwater_fraction)
                        * costs.at[
                            supcar + "-ac-connection-underground",
                            "capital_cost",
                        ]
                    )
                )
                capital_cost = (
                    costs.at[car, "capital_cost"]
                    + costs.at[
                        supcar + "-ac-station",
                        "capital_cost",
                    ]  # update to find floating substation costs
                    + connection_cost
                )

                logger.info(
                    "Added connection cost of {:0.0f}-{:0.0f} USD/MW/a to {}".format(
                        connection_cost.min(),
                        connection_cost.max(),
                        car,
                    ),
                )
            else:
                capital_cost = costs.at[car, "capital_cost"]

            bus2sub = (
                pd.read_csv(input_profiles.bus2sub, dtype=str)
                .drop("interconnect", axis=1)
                .rename(columns={"Bus": "bus_id"})
            )
            bus_list = (
                ds.bus.to_dataframe("sub_id").merge(bus2sub).bus_id.astype(str).values
            )
            p_nom_max_bus = (
                ds["p_nom_max"]
                .to_dataframe()
                .merge(bus2sub[["bus_id", "sub_id"]], left_on="bus", right_on="sub_id")
                .set_index("bus_id")
                .p_nom_max
            )
            weight_bus = (
                ds["weight"]
                .to_dataframe()
                .merge(bus2sub[["bus_id", "sub_id"]], left_on="bus", right_on="sub_id")
                .set_index("bus_id")
                .weight
            )
            bus_profiles = (
                ds["profile"]
                .transpose("time", "bus")
                .to_pandas()
                .T.merge(
                    bus2sub[["bus_id", "sub_id"]],
                    left_on="bus",
                    right_on="sub_id",
                )
                .set_index("bus_id")
                .drop(columns="sub_id")
                .T
            )
            if supcar == "offwind":
                capital_cost = capital_cost.to_frame().reset_index()
                capital_cost.bus = capital_cost.bus.astype(int)
                capital_cost = (
                    pd.merge(
                        capital_cost,
                        n.buses.sub_id.reset_index(),
                        left_on="bus",
                        right_on="sub_id",
                        how="left",
                    )
                    .rename(columns={0: "capital_cost"})
                    .set_index("Bus")
                    .capital_cost
                )

            logger.info(f"Adding {car} capacity-factor profiles to the network.")
            # TODO: #24 VALIDATE TECHNICAL POTENTIALS

            n.madd(
                "Generator",
                bus_list,
                " " + car,
                bus=bus_list,
                carrier=car,
                p_nom_extendable=car in extendable_carriers["Generator"],
                p_nom_max=p_nom_max_bus,
                weight=weight_bus,
                marginal_cost=costs.at[car, "marginal_cost"],
                capital_cost=capital_cost,
                efficiency=costs.at[car, "efficiency"],
                p_max_pu=bus_profiles,
            )


# double check to make sure batteries are added regardless if they are extendable.
def attach_battery_storage(
    n: pypsa.Network,
    plants: pd.DataFrame,
    extendable_carriers,
    costs,
):
    """
    Attaches Battery Energy Storage Systems To the Network.
    """
    plants_filt = plants.query("carrier == 'battery' ")
    plants_filt.index = (
        plants_filt.index.astype(str) + "_" + plants_filt.generator_id.astype(str)
    )
    plants_filt.nameplate_energy_capacity_mwh = (
        plants_filt.nameplate_energy_capacity_mwh.astype(float)
    )
    logger.info(
        f"Added Batteries as Storage Units to the network.\n{np.round(plants_filt.p_nom.sum()/1000,2)} GW Power Capacity \n{np.round(plants_filt.nameplate_energy_capacity_mwh.sum()/1000, 2)} GWh Energy Capacity",
    )

    plants_filt = plants_filt.dropna(subset=["nameplate_energy_capacity_mwh"])
    n.madd(
        "StorageUnit",
        plants_filt.index,
        carrier="battery",
        bus=plants_filt.bus_assignment,
        p_nom=plants_filt.p_nom,
        p_nom_min=plants_filt.p_nom,
        p_nom_extendable=False,
        max_hours=plants_filt.nameplate_energy_capacity_mwh / plants_filt.p_nom,
        build_year=plants_filt.operating_year,
        efficiency_store=0.9**0.5,
        efficiency_dispatch=0.9**0.5,
        cyclic_state_of_charge=True,
    )


def load_powerplants_eia(
    eia_dataset: str,
    interconnect: str = None,
) -> pd.DataFrame:

    plants = pd.read_csv(
        eia_dataset,
    )
    if (interconnect is not None) & (interconnect != "usa"):
        plants["interconnection"] = plants["nerc_region"].map(const.NERC_REGION_MAPPER)
        plants = plants[plants.interconnection == interconnect]

    plants["generator_name"] = (
        plants.index.astype(str)
        + "_"
        + plants.plant_code.astype(str)
        + "_"
        + plants.generator_id.astype(str)
    )
    plants.set_index("generator_name", inplace=True)
    plants["p_nom"] = plants.pop("nameplate_capacity_mw")

    # Set Costs
    plants["heat_rate"] = plants.pop("egrid_heat_rate")
    plants = add_missing_fuel_cost(
        plants,
        snakemake.input.fuel_costs,
    )  # Only used for plants that don't have temporal data
    plants = add_missing_heat_rates(
        plants,
        snakemake.input.fuel_costs,
    )  # Only used for plants that not included in ADS data
    plants["marginal_cost"] = (
        plants.heat_rate * plants.fuel_cost
    )  # (MMBTu/MW) * (USD/MMBTu) = USD/MW

    # plants["vom_costs"] = plants.pop("ads_vom_cost")
    # avg_prime_move_vom = plants[['carrier','vom_costs']].groupby('carrier').mean()
    # plants.loc[plants.vom_costs.isna(),'vom_costs'] = plants.loc[plants.vom_costs.isna(),'carrier'].map(avg_prime_move_vom.vom_costs)

    plants["start_up_cost"] = plants["ads_startup_cost_fixed$"].fillna(
        0,
    ) + +plants.ads_startfuelmmbtu.fillna(0) * plants.fuel_cost.fillna(0)

    plants["efficiency"] = 1 / (
        plants["heat_rate"] / 3.412
    )  # MMBTu/MWh to MWh_electric/MWh_thermal

    # Set Ramp Rates
    plants["ramp_limit_up"] = (
        plants.pop("ads_rampup_ratemw/minute") / plants.p_nom * 60
    )  # MW/min to p.u./hour
    plants["ramp_limit_down"] = (
        plants.pop("ads_rampdn_ratemw/minute") / plants.p_nom * 60
    )  # MW/min to p.u./hour
    avg_prime_mover_ramp_rates = (
        plants[["carrier", "ramp_limit_up", "ramp_limit_down"]]
        .groupby("carrier")
        .mean()
    )
    # fill missing ramp rates with average ramp rates of prime movers
    plants.loc[plants.ramp_limit_up.isna(), "ramp_limit_up"] = plants.loc[
        plants.ramp_limit_up.isna(),
        "carrier",
    ].map(avg_prime_mover_ramp_rates.ramp_limit_up)
    plants.loc[plants.ramp_limit_down.isna(), "ramp_limit_down"] = plants.loc[
        plants.ramp_limit_down.isna(),
        "carrier",
    ].map(avg_prime_mover_ramp_rates.ramp_limit_down)

    # Timeline
    plants["build_year"] = plants.operating_year
    plants["dateout"] = (
        np.inf
    )  # plants.planned_retirement_year.replace(' ').astype(int).fillna(np.inf)  # placeholder TODO Add retirement year

    if interconnect:
        plants["interconnection"] = plants["nerc_region"].map(const.NERC_REGION_MAPPER)
        plants = plants[plants.interconnection == interconnect]

    return plants


def apply_seasonal_capacity_derates(
    n: pypsa.Network,
    plants: pd.DataFrame,
    conventional_carriers: list,
    sns: pd.DatetimeIndex,
):
    "Applies rerate factor p_max_pu based on the seasonal capacity derates defined in eia860"
    summer_sns = sns[sns.month.isin([6, 7, 8])]
    winter_sns = sns[~sns.month.isin([6, 7, 8])]

    conv_plants = plants.query("carrier in @conventional_carriers")
    conv_plants.index = "C" + conv_plants.index
    conv_gens = n.generators.query("carrier in @conventional_carriers")

    p_max_pu = pd.DataFrame(1.0, index=sns, columns=conv_gens.index)
    p_max_pu.loc[summer_sns, conv_gens.index] *= conv_plants.loc[
        :,
        "summer_derate",
    ].astype(float)
    p_max_pu.loc[winter_sns, conv_gens.index] *= conv_plants.loc[
        :,
        "winter_derate",
    ].astype(float)
    n.generators_t.p_max_pu = pd.concat([n.generators_t.p_max_pu, p_max_pu], axis=1)

    conv_plants.loc[:, "ads_mustrun"] = conv_plants.ads_mustrun.fillna(False)
    must_run = conv_plants.loc[conv_plants.ads_mustrun, :].copy()
    must_run.loc[:, "minimum_load_mw"] = must_run.minimum_load_mw.astype(float)
    must_run.loc[:, "minimum_cf"] = must_run.minimum_load_mw / must_run.p_nom
    must_run.loc[:, "minimum_cf"] = must_run.minimum_cf.clip(
        upper=np.minimum(must_run.summer_derate, must_run.winter_derate),
    )

    p_min_pu = pd.DataFrame(1.0, index=sns, columns=must_run.index)
    p_min_pu.loc[:, must_run.index] *= must_run.loc[:, "minimum_cf"].astype(float)
    n.generators_t.p_min_pu = pd.concat([n.generators_t.p_min_pu, p_min_pu], axis=1)


def assign_ads_missing_lat_lon(plants, n):
    plants_unmatched = plants[plants.latitude.isna() | plants.longitude.isna()]
    plants_unmatched = plants_unmatched[~plants_unmatched.balancing_area.isna()]
    logger.info(
        f"Assigning lat and lon to {len(plants_unmatched)} plants missing locations.",
    )

    ba_list_map = {
        "CISC": "CISO-SCE",
        "CISD": "CISO-SDGE",
        "VEA": "CISO-VEA",
        "AZPS": "Arizona",
        "SRP": "Arizona",
        "PAID": "PACW",
        "PAUT": "PACW",
        "PAWY": "PACW",
        "IPFE": "IPCO",
        "IPMV": "IPCO",
        "IPTV": "IPCO",
        "TPWR": "BPAT",
        "SCL": "BPAT",
        "CIPV": "CISO-PGAE",
        "CIPB": "CISO-PGAE",
        "SPPC": "CISO-PGAE",
        "TH_PV": "Arizona",
    }

    plants_unmatched["balancing_area"] = plants_unmatched["balancing_area"].replace(
        ba_list_map,
    )
    buses = n.buses.copy()

    # assign lat and lon to the plants_unmatched by choosing the bus within the same balancing_area that has the highest v_nom value.
    # Currently randomly assigned to the top 4 buses in the balancing area by v_nom.
    for i, row in plants_unmatched.iterrows():
        # print(row.balancing_area)
        buses_in_area = buses[buses.balancing_area == row.balancing_area].sort_values(
            by="v_nom",
            ascending=False,
        )
        top_5_buses = buses_in_area.iloc[:4]
        bus = top_5_buses.iloc[random.randint(0, 3)]
        plants_unmatched.loc[i, "longitude"] = bus.x
        plants_unmatched.loc[i, "latitude"] = bus.y

    plants.loc[plants_unmatched.index] = plants_unmatched
    logger.info(
        f"{len(plants[plants.latitude.isna() | plants.longitude.isna()])} plants still missing locations.",
    )
    plants = plants.dropna(
        subset=["latitude", "longitude"],
    )  # drop any plants that still don't have lat/lon

    return plants


def attach_ads_renewables(n, plants_df, renewable_carriers, extendable_carriers, costs):
    """
    Attaches renewable plants from ADS files.
    """
    ads_renewables_path = snakemake.input.ads_renewables

    for tech_type in renewable_carriers:
        plants_filt = plants_df.query("carrier == @tech_type")
        plants_filt.index = plants_filt.ads_name.astype(str)

        logger.info(f"Adding {len(plants_filt)} {tech_type} generators to the network.")

        if tech_type in ["wind", "offwind"]:
            profiles = pd.read_csv(ads_renewables_path + "/wind_2032.csv", index_col=0)
        elif tech_type == "solar":
            profiles = pd.read_csv(ads_renewables_path + "/solar_2032.csv", index_col=0)
            dpv = pd.read_csv(ads_renewables_path + "/btm_solar_2032.csv", index_col=0)
            profiles = pd.concat([profiles, dpv], axis=1)
        else:
            profiles = pd.read_csv(
                ads_renewables_path + f"/{tech_type}_2032.csv",
                index_col=0,
            )

        profiles.columns = profiles.columns.str.replace(".dat: 2032", "")
        profiles.columns = profiles.columns.str.replace(".DAT: 2032", "")

        profiles.index = n.snapshots
        profiles.columns = profiles.columns.astype(str)

        if (
            tech_type == "hydro"
        ):  # matching hydro according to balancing authority specified
            profiles.columns = profiles.columns.str.replace("HY_", "")
            profiles.columns = profiles.columns.str.replace("_2018", "")
            southwest = {"Arizona", "SRP", "WALC", "TH_Mead"}
            northwest = {"DOPD", "CHPD", "WAUW"}
            pge_dict = {"CISO-PGAE": "CIPV", "CISO-SCE": "CISC", "CISO-SDGE": "CISD"}
            plants_filt.balancing_area = plants_filt.balancing_area.map(
                pge_dict,
            ).fillna(plants_filt.balancing_area)
            # {'Arizona', 'CISC', 'IPFE', 'DOPD', 'CISD', 'IPMV', 'CHPD', 'PSCO', 'CISO-SDGE', 'IPTV', 'CIPV', 'TH_Mead', 'CIPB', 'WALC', 'CISO-SCE', 'WAUW', 'SRP', 'CISO-PGAE'}
            # TODO: #34 Add BCHA and AESO hydro profiles in ADS Configuration. Profiles that don't get used: 'AESO', 'IPCO', 'NEVP', 'BCHA'
            # profiles_ba = set(profiles.columns) # available ba hydro profiles
            # bas = set(plants_filt.balancing_area.unique()) # plants that need BA hydro profiles

            # print( need to assign bas for pge bay and valley)
            profiles_new = pd.DataFrame(index=n.snapshots, columns=plants_filt.index)
            for plant in profiles_new.columns:
                ba = plants_filt.loc[plant].balancing_area
                if ba in southwest:
                    ba = "SouthConsolidated"
                elif ba in northwest:
                    ba = "BPAT"  # this is a temp fix. Probably not right to assign all northwest hydro to BPA
                ba_prof = profiles.columns.str.contains(ba)
                if ba_prof.sum() == 0:
                    logger.warning(f"No hydro profile for {ba}.")
                    profiles_new[plant] = 0

                profiles_new[plant] = profiles.loc[:, ba_prof].values
            p_max_pu = profiles_new
            p_max_pu.columns = plants_filt.index
        else:  #  solar + wind + other
            # intersection = set(profiles.columns).intersection(plants_filt.dispatchshapename)
            # missing = set(plants_filt.dispatchshapename) - intersection
            # profiles = profiles[list(intersection)]
            profiles_new = pd.DataFrame(
                index=n.snapshots,
                columns=plants_filt.dispatchshapename,
            )
            for plant in profiles_new.columns:
                profiles_new[plant] = profiles[plant]
            p_max_pu = profiles_new
            p_max_pu.columns = plants_filt.index

        p_nom = plants_filt["maxcap(mw)"]
        n.madd(
            "Generator",
            plants_filt.index,
            bus=plants_filt.bus_assignment,
            p_nom_min=p_nom,
            p_nom=p_nom,
            marginal_cost=0,  # (MMBTu/MW) * (USD/MMBTu) = USD/MW
            capital_cost=costs.at[tech_type, "capital_cost"],
            p_max_pu=p_max_pu,  # timeseries of max power output pu
            p_nom_extendable=tech_type in extendable_carriers["Generator"],
            carrier=tech_type,
            weight=1.0,
            efficiency=costs.at[tech_type, "efficiency"],
        )
    return n


def load_powerplants_ads(
    ads_dataset: str,
    tech_mapper: dict[str, str] = None,
    carrier_mapper: dict[str, str] = None,
    fuel_mapper: dict[str, str] = None,
) -> pd.DataFrame:
    """
    Loads base ADS plants, fills missing data, and applies name mappings.

    Arguments
    ---------
    ads_dataset: str,
    tech_mapper: Dict[str,str],
    carrier_mapper: Dict[str,str],
    fuel_mapper: Dict[str,str],
    """

    # read in data
    plants = pd.read_csv(
        ads_dataset,
        index_col=0,
        dtype={"bus_assignment": "str"},
    ).rename(columns=str.lower)
    plants.rename(columns={"fueltype": "fuel_type_ads"}, inplace=True)

    # apply mappings if required
    if carrier_mapper:
        plants["carrier"] = plants.fuel_type_ads.map(carrier_mapper)
    if fuel_mapper:
        plants["fuel_type"] = plants.fuel_type_ads.map(fuel_mapper)
    if tech_mapper:
        plants["tech_type"] = plants.tech_type.map(tech_mapper)
    plants.rename(columns={"lat": "latitude", "lon": "longitude"}, inplace=True)

    # apply missing data to powerplants
    plants = add_missing_fuel_cost(plants, snakemake.input.fuel_costs)
    plants = add_missing_heat_rates(plants, snakemake.input.fuel_costs)

    plants["generator_name"] = plants.ads_name.astype(str)
    plants["p_nom"] = plants["maxcap(mw)"]
    plants["heat_rate"] = plants["inchr2(mmbtu/mwh)"]
    plants["marginal_cost"] = (
        plants["heat_rate"] * plants.fuel_cost
    )  # (MMBTu/MW) * (USD/MMBTu) = USD/MW
    plants["efficiency"] = 1 / (
        plants["heat_rate"] / 3.412
    )  # MMBTu/MWh to MWh_electric/MWh_thermal
    plants["ramp_limit_up"] = (
        plants["rampup rate(mw/minute)"] / plants["maxcap(mw)"] * 60
    )  # MW/min to p.u./hour
    plants["ramp_limit_down"] = (
        plants["rampdn rate(mw/minute)"] / plants["maxcap(mw)"] * 60
    )  # MW/min to p.u./hour
    return plants


def clean_bus_data(n: pypsa.Network):
    """
    Drops data from the network that are no longer needed in workflow.
    """
    col_list = [
        "poi_bus",
        "poi_sub",
        "poi",
        "Pd",
        "load_dissag",
        "LAF",
        "LAF_state",
        "county",
    ]
    n.buses.drop(columns=[col for col in col_list if col in n.buses], inplace=True)


def main(snakemake):
    params = snakemake.params
    configuration = snakemake.config["network_configuration"]
    interconnection = snakemake.wildcards["interconnect"]
    planning_horizons = snakemake.params["planning_horizons"]

    n = pypsa.Network(snakemake.input.base_network)

    snapshot_config = snakemake.config["snapshots"]
    sns_start = pd.to_datetime(snapshot_config["start"])
    sns_end = pd.to_datetime(snapshot_config["end"])
    sns_inclusive = snapshot_config["inclusive"]

    n.set_snapshots(
        pd.date_range(
            freq="h",
            start=sns_start,
            end=sns_end,
            inclusive=sns_inclusive,
        ),
    )
    Nyears = n.snapshot_weightings.objective.sum() / 8760.0

    costs = load_costs(
        snakemake.input.tech_costs,
        params.costs,
        params.electricity["max_hours"],
        Nyears,
    )

    # calculates annulaized capital costs seperate from the fixed costs to be
    # able to apply regional mulitpliers to only capex
    costs = add_annualized_capital_costs(costs, Nyears)

    # fix for ccgt and ocgt techs
    costs.at["gas", "investment_annualized"] = (
        costs.at["CCGT", "investment_annualized"]
        + costs.at["OCGT", "investment_annualized"]
    ) / 2

    update_transmission_costs(n, costs, params.length_factor)

    renewable_carriers = set(params.electricity["renewable_carriers"])
    extendable_carriers = params.electricity["extendable_carriers"]
    conventional_carriers = params.electricity["conventional_carriers"]
    conventional_inputs = {
        k: v for k, v in snakemake.input.items() if k.startswith("conventional_")
    }

    if params.conventional["unit_commitment"]:
        unit_commitment = pd.read_csv(snakemake.input.unit_commitment, index_col=0)
    else:
        unit_commitment = None

    if configuration == "pypsa-usa":
        plants = load_powerplants_eia(
            snakemake.input["plants_eia"],
            interconnect=interconnection,
        )
    elif configuration == "ads2032":
        plants = load_powerplants_ads(
            snakemake.input["plants_ads"],
            const.ADS_SUB_TYPE_TECH_MAPPER,
            const.ADS_CARRIER_NAME,
            const.ADS_FUEL_MAPPER,
        )
        plants = assign_ads_missing_lat_lon(plants, n)
    else:
        raise ValueError(
            f"Unknown network_configuration {snakemake.config['network_configuration']}",
        )

    # Applying to all configurations
    plants = match_plant_to_bus(n, plants)

    attach_demand(n, snakemake.input.demand)

    attach_conventional_generators(
        n,
        costs,
        plants,
        conventional_carriers,
        extendable_carriers,
        params.conventional,
        renewable_carriers,
        conventional_inputs,
        unit_commitment=unit_commitment,
        fuel_price=None,  # update fuel prices later
    )
    apply_seasonal_capacity_derates(
        n,
        plants,
        conventional_carriers,
        n.snapshots,
    )
    attach_battery_storage(
        n,
        plants,
        extendable_carriers,
        costs,
    )

    if configuration == "ads2032":
        attach_ads_renewables(
            n,
            plants,
            renewable_carriers,
            extendable_carriers,
            costs,
        )
    else:
        attach_wind_and_solar(
            n,
            costs,
            snakemake.input,
            renewable_carriers,
            extendable_carriers,
            params.length_factor,
        )
        renewable_carriers = list(
            set(snakemake.config["electricity"]["renewable_carriers"]).intersection(
                {"onwind", "solar", "offwind", "offwind_floating"},
            ),
        )
        attach_renewable_capacities_to_atlite(
            n,
            plants,
            renewable_carriers,
        )
        # temporarily adding hydro with breakthrough only data until I can correctly import hydro_data
        n = attach_breakthrough_renewable_plants(
            n,
            snakemake.input["plants_breakthrough"],
            ["hydro"],
            extendable_carriers,
            costs,
        )
    update_p_nom_max(n)

    # apply regional multipliers to capital cost data
    for carrier, multiplier_data in const.CAPEX_LOCATIONAL_MULTIPLIER.items():
        if n.generators.query(f"carrier == '{carrier}'").empty:
            continue
        multiplier_file = snakemake.input[f"gen_cost_mult_{multiplier_data}"]
        df_multiplier = pd.read_csv(multiplier_file)
        df_multiplier = clean_locational_multiplier(df_multiplier)
        update_capital_costs(n, carrier, costs, df_multiplier, Nyears)

    if params.conventional["dynamic_fuel_price"]:
        assert params.eia_api, f"Must provide EIA API key for dynamic fuel pricing"

        dynamic_fuel_prices = {
            "OCGT": {
                "state": "state_ng_fuel_prices",
                "balancing_area": "ba_ng_fuel_prices",  # name of file in snakefile
            },
            "CCGT": {
                "state": "state_ng_fuel_prices",
                "balancing_area": "ba_ng_fuel_prices",
            },
            "coal": {"state": "state_coal_fuel_prices"},
        }

        # NOTE: Must go from most to least coarse data (ie. state then ba) to apply the
        # data correctly!

        for carrier, prices in dynamic_fuel_prices.items():
            for area in ("state", "reeds_zone", "balancing_area"):
                # check if data is supplied for the area
                try:
                    datafile = prices[area]
                except KeyError:
                    continue
                # if data should exist, try to read it in
                try:
                    df = pd.read_csv(snakemake.input[datafile], index_col="snapshot")
                except KeyError:
                    logger.warning(f"Can not find dynamic price file {datafile}")
                    continue

                vom = costs.at[carrier, "VOM"]

                apply_dynamic_pricing(
                    n=n,
                    carrier=carrier,
                    geography=area,
                    df=df,
                    vom=vom,
                )
                logger.info(f"Applied dynamic price data for {carrier} from {datafile}")

    # fix p_nom_min for extendable generators
    # The "- 0.001" is just to avoid numerical issues
    n.generators["p_nom_min"] = n.generators.apply(
        lambda x: (
            (x["p_nom"] - 0.001)
            if (x["p_nom_extendable"] and x["p_nom_min"] == 0)
            else x["p_nom_min"]
        ),
        axis=1,
    )

    output_folder = os.path.dirname(snakemake.output[0]) + "/base_network"
    export_network_for_gis_mapping(n, output_folder)

    clean_bus_data(n)
    sanitize_carriers(n, snakemake.config)
    n.meta = snakemake.config
    n.export_to_netcdf(snakemake.output[0])

    logger.info(test_network_datatype_consistency(n))


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("add_electricity", interconnect="western")
    configure_logging(snakemake)
    main(snakemake)
