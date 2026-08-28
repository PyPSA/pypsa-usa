# PyPSA USA Authors
"""
Adds existing conventional generators, renewable generators, and storage devices to the network.

This script will add all generator unit availabilities (capacity-factors) to the network, for all investment horizons.
"""

import logging
import os

import constants as const
import dill as pickle
import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import (
    calculate_annuity,
    configure_logging,
    export_network_for_gis_mapping,
    load_costs,
    log_network_schema,
    update_p_nom_max,
    weighted_avg,
)
from sklearn.neighbors import BallTree

idx = pd.IndexSlice

logger = logging.getLogger(__name__)

# Maximum distance from the model footprint at which a "must add" seam plant is
# still attached. Only applied to footprint-scoped runs (model_topology.include);
# see filter_plants_by_region / _drop_distant_seam_plants.
SEAM_PLANT_MAX_KM = 100.0


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
    for c in n.components:
        if "carrier" in c.static:
            add_missing_carriers(n, c.static.carrier)

    carrier_i = n.carriers.index
    nice_names = (
        pd.Series(config["plotting"]["nice_names"]).reindex(carrier_i).fillna(carrier_i.to_series().str.title())
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


def add_missing_carriers(n, carriers):
    """Function to add missing carriers to the network without raising errors."""
    # sorted: set iteration order is hash-seed dependent, which makes the
    # Carrier table order (and thus the .nc files) differ between otherwise
    # identical runs
    missing_carriers = sorted(set(carriers) - set(n.carriers.index))
    if len(missing_carriers) > 0:
        n.add("Carrier", missing_carriers)


def clean_locational_multiplier(df: pd.DataFrame):
    """Updates format of locational multiplier data."""
    df = df.fillna(1)
    df = df[["State", "Location Variation"]]
    return df.groupby("State").mean()


def update_capital_costs(
    n: pypsa.Network,
    carrier: str,
    costs: pd.DataFrame,
    multiplier: pd.DataFrame,
):
    """Applies regional multipliers to capital cost data."""
    # map generators to states (multiplier CSVs are indexed by full state name;
    # post-aggregation buses only carry reeds_state 2-letter codes)
    bus_state_mapper = n.buses["reeds_state"].map(const.CODE_2_STATE).to_dict()
    gen = n.generators[n.generators.carrier == carrier].copy()
    gen["state"] = gen.bus.map(bus_state_mapper)
    gen = gen[gen["state"].isin(multiplier.index)]  # drops any regions that do not have cost multipliers

    # log any states that do not have multipliers attached
    missed = gen[~gen["state"].isin(multiplier.index)]
    if not missed.empty:
        logger.warning(f"CAPEX cost multiplier not applied to {missed.state.unique()}")

    # apply multiplier to annualized capital investment cost
    gen["annualized_capex_per_mw"] = gen.apply(
        lambda x: costs.at[carrier, "annualized_capex_per_mw"] * multiplier.at[x["state"], "Location Variation"],
        axis=1,
    )

    # get fixed costs based on overnight capital costs with multiplier applied
    gen["fom"] = costs.at[carrier, "opex_fixed_per_kw"] * 1e3

    # find final annualized capital cost
    gen["capital_cost"] = gen["annualized_capex_per_mw"] + gen["fom"]

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

    if gens.empty:
        return

    eff = n.get_switchable_as_dense("Generator", "efficiency").T
    eff = eff[eff.index.isin(gens.index)].T
    eff.columns.name = ""

    fuel_cost_per_gen = {gen: df[gens.at[gen, geography]] for gen in gens.index}
    fuel_costs = pd.DataFrame.from_dict(fuel_cost_per_gen)
    fuel_costs.index = pd.to_datetime(fuel_costs.index)
    fuel_costs = broadcast_investment_horizons_index(n, fuel_costs)

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
        n.lines["length"] * length_factor * costs.at["HVAC overhead", "annualized_capex_per_mw_km"]
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
            (1.0 - n.links.loc[dc_b, "underwater_fraction"]) * costs.at["HVDC overhead", "annualized_capex_per_mw_km"]
            + n.links.loc[dc_b, "underwater_fraction"] * costs.at["HVDC submarine", "annualized_capex_per_mw_km"]
        )
        + costs.at["HVDC inverter pair", "annualized_capex_per_mw"]
    )
    n.links.loc[dc_b, "capital_cost"] = costs


def load_powerplants(
    plants_fn,
    investment_periods: list[int],
    interconnect: str | None = None,
) -> pd.DataFrame:
    plants = pd.read_csv(
        plants_fn,
    )
    plants = plants.set_index("generator_name")

    # Convert date columns to datetime
    plants["current_planned_generator_operating_date"] = pd.to_datetime(
        plants["current_planned_generator_operating_date"],
    )

    plants["generator_retirement_date"] = pd.to_datetime(
        plants["generator_retirement_date"],
    )

    # if operational_status is proposed replace build_year with year of current_planned_generator_operating_date
    plants.loc[plants.operational_status == "proposed", "build_year"] = plants.loc[
        plants.operational_status == "proposed",
        "current_planned_generator_operating_date",
    ].dt.year

    # If operational_status is existing or proposed, replace generator_retirement_date with 1/1/2100
    retirement_date = pd.to_datetime("2100-01-01")
    plants.loc[plants.operational_status.isin(["existing", "proposed"]), "generator_retirement_date"] = retirement_date

    # Handle NaT values
    plants.loc[plants.generator_retirement_date.isna(), "generator_retirement_date"] = pd.to_datetime("1900-01-01")

    # Filter out plants that are not built by first investment period and retired before the first investment period.
    plants = plants[plants.build_year <= investment_periods[0]]
    plants = plants[plants.generator_retirement_date.dt.year > investment_periods[0]]

    # Filter out non-conus plants
    plants = plants[plants.nerc_region != "non-conus"]
    if (interconnect is not None) & (interconnect != "usa"):
        plants["interconnection"] = plants["nerc_region"].map(const.NERC_REGION_MAPPER)
        plants = plants[plants.interconnection == interconnect]
    return plants


def match_nearest_bus(plants_subset, buses_subset):
    """Assign the nearest bus to each plant in the given subsets."""
    if plants_subset.empty or buses_subset.empty:
        return plants_subset

    # Create a BallTree for the given subset of buses
    tree = BallTree(buses_subset[["x", "y"]].values, leaf_size=2)

    # Find nearest bus for each plant in the subset
    distances, indices = tree.query(
        plants_subset[["longitude", "latitude"]].values,
        k=1,
    )

    # Map the nearest bus information back to the plants subset
    plants_subset["bus_assignment"] = buses_subset.index.to_numpy()[indices.flatten()]
    plants_subset["distance_nearest"] = distances.flatten()

    return plants_subset


def match_plant_to_bus(n, plants):
    """
    Matches each plant to it's corresponding bus in the network enfocing a
    match to the correct State.

    Efficient matching taken from:
    https://stackoverflow.com/questions/58893719/find-nearest-point-in-other-dataframe-with-a-lot-of-data
    """
    plants_matched = plants.copy()
    plants_matched["bus_assignment"] = None
    plants_matched["distance_nearest"] = None

    # Get a copy of buses and create a geometry column with GPS coordinates
    buses = n.buses.copy()
    buses["geometry"] = gpd.points_from_xy(buses["x"], buses["y"])

    # First pass: Assign each plant to the nearest bus in the same reeds zone.
    # Only runs when plants carry a `country` column (sjoined from regions_onshore);
    # absent that, fall through to the zone-agnostic second pass.
    if "country" in plants_matched.columns:
        for zone_id in buses["reeds_zone"].unique():
            buses_in_zone = buses[buses["reeds_zone"] == zone_id]
            plants_in_zone = plants_matched[
                (plants_matched["country"] == zone_id) & (plants_matched["bus_assignment"].isnull())
            ]

            # Update plants_matched with the nearest bus within the same REEDS zone
            plants_matched.update(match_nearest_bus(plants_in_zone, buses_in_zone))

    # Second pass: Assign any remaining unmatched plants to the nearest bus regardless of REEDS zone
    unmatched_plants = plants_matched[plants_matched["bus_assignment"].isnull()]
    if not unmatched_plants.empty:
        plants_matched.update(match_nearest_bus(unmatched_plants, buses))

    return plants_matched


def _drop_distant_seam_plants(
    plants_must_add: pd.DataFrame,
    regions_onshore: gpd.GeoDataFrame,
    regions_offshore: gpd.GeoDataFrame,
    max_km: float = SEAM_PLANT_MAX_KM,
) -> pd.DataFrame:
    """
    Drop "must add" seam plants lying further than ``max_km`` from the model footprint.

    ``plants_must_add`` collects plants that fall outside every ReEDS shape of the
    run's interconnect and whose ReEDS membership disagrees with their EIA
    ``interconnection`` column. They are re-added unconditionally so imprecise ReEDS
    shapes never silently delete a legitimate border plant. In a footprint-scoped run
    (model_topology.include) the regions layers only tile the model footprint, so that
    unconditional add-back lets plants thousands of km away survive the filter and then
    attach to the nearest in-footprint bus — match_plant_to_bus applies no distance
    bound. Bounding the population here keeps nearby seam plants while cutting the leak.

    Plants inside the footprint have distance 0 and are always kept.
    """
    if plants_must_add.empty:
        return plants_must_add

    region_geoms = [
        regions.to_crs(epsg=5070).geometry
        for regions in (regions_onshore, regions_offshore)
        if regions is not None and not regions.empty
    ]
    if not region_geoms:
        return plants_must_add
    footprint = pd.concat(region_geoms)
    footprint = footprint[footprint.notna() & ~footprint.is_empty]
    if footprint.empty:
        return plants_must_add

    points = gpd.GeoSeries(
        gpd.points_from_xy(plants_must_add.longitude, plants_must_add.latitude),
        crs="EPSG:4326",
    ).to_crs(epsg=5070)
    # The distance to the footprint is the smallest distance to any one of its
    # regions, so take that minimum directly instead of unioning them first.
    # Reprojecting the region layer into EPSG:5070 can leave coarse cluster
    # polygons invalid (9 of 29 self-intersecting or degenerate at simpl=20,
    # none at simpl=''), and union_all() then dies with a GEOSException
    # "side location conflict"; pairwise distance is robust to that. Where the
    # union does succeed the two agree to 0.0 m, verified on the simpl='' layer.
    distance_km = points.apply(lambda point: footprint.distance(point).min()).to_numpy() / 1e3

    keep = distance_km <= max_km
    if keep.all():
        return plants_must_add

    dropped = plants_must_add[~keep]
    for (name, plant), distance in zip(dropped.iterrows(), distance_km[~keep], strict=False):
        logger.warning(
            f"Out-of-footprint seam plant dropped: '{name}' "
            f"(carrier={plant.get('carrier', 'n/a')}, state={plant.get('state', 'n/a')}, "
            f"{plant.get('p_nom', float('nan')):.1f} MW) sits {distance:.0f} km from the model "
            f"regions, beyond the {max_km:.0f} km seam bound.",
        )
    dropped_mw = float(dropped["p_nom"].sum()) if "p_nom" in dropped.columns else float("nan")
    logger.warning(
        f"Footprint-scoped run: dropped {int((~keep).sum())} of {len(plants_must_add)} 'must add' "
        f"seam plants ({dropped_mw:.1f} MW) further than {max_km:.0f} km from the model regions. "
        "Without this bound match_plant_to_bus attaches them to the nearest in-footprint bus at "
        "unbounded distance.",
    )
    return plants_must_add[keep]


def filter_plants_by_region(
    plants: pd.DataFrame,
    regions_onshore: gpd.GeoDataFrame,
    regions_offshore: gpd.GeoDataFrame,
    reeds_shapes: gpd.GeoDataFrame,
    all_reeds_shapes: gpd.GeoDataFrame,
    reeds_memberships: pd.DataFrame,
    footprint_scoped: bool = False,
) -> pd.DataFrame:
    """
    Filters the plants dataframe to remove plants not within the onshore and
    offshore geometries.

    ``footprint_scoped`` must be set when the run was scoped with
    model_topology.include. It bounds the "must add" seam-plant fallback below to
    SEAM_PLANT_MAX_KM of the (footprint-sized) regions. Left false, the fallback keeps
    its legacy unconditional behavior, so unfiltered interconnect/usa runs are
    unchanged.
    """
    plants = plants.copy()
    plants["geometry"] = gpd.points_from_xy(
        plants.longitude,
        plants.latitude,
        crs="EPSG:4326",
    )
    gdf_plants = gpd.GeoDataFrame(plants, geometry="geometry")
    plants_onshore = gpd.sjoin(gdf_plants, regions_onshore, how="inner")
    plants_offshore = gpd.sjoin(gdf_plants, regions_offshore, how="inner")
    if not plants_offshore.empty:
        logger.warning(f"Offshore plants: {plants_offshore}")
    plants_filt = pd.concat([plants_onshore, plants_offshore])

    # Some plants like Diablo Canyon near oceans don't have region due to
    # imprecise ReEDS Shapes. We filter plants that have no reeds regions,
    # then search these points again.
    plants_in_regions = gpd.sjoin(
        gdf_plants,
        reeds_shapes,
        how="inner",
        predicate="intersects",
    )
    plants_no_region = gdf_plants[~gdf_plants.index.isin(plants_in_regions.index)]
    if not plants_no_region.empty:
        # identify the plants for which the interconnection according to the reeds membership is different than the interconnection according to the EIA data. We need to include these plants since these are plants where the reeds shapes are not precise enough to assign a region.
        plants_no_region = plants_no_region.to_crs(epsg=3857)
        plants_no_region_all_shapes = gpd.sjoin(
            plants_no_region.reset_index(),
            all_reeds_shapes,
            how="inner",
            predicate="intersects",
        )
        plants_no_region_all_shapes = plants_no_region_all_shapes.to_crs(epsg=4326)
        reeds_memberships.loc[reeds_memberships.interconnect == "ercot", "interconnect"] = "texas"
        plants_no_region_all_shapes = plants_no_region_all_shapes.merge(
            reeds_memberships[["ba", "interconnect"]],
            left_on="rb",
            right_on="ba",
            how="left",
        )
        # Handles non-US wide interconnection cases (western, eastern, texas)
        if "interconnection" in plants_no_region_all_shapes.columns:
            plants_must_add = plants_no_region_all_shapes[
                plants_no_region_all_shapes.interconnect != plants_no_region_all_shapes.interconnection
            ]
            remaining_plants = plants_no_region_all_shapes[
                plants_no_region_all_shapes.interconnect == plants_no_region_all_shapes.interconnection
            ]
        # Handles US wide interconnection cases
        else:
            plants_must_add = plants_no_region_all_shapes
            remaining_plants = pd.DataFrame()
        plants_must_add.set_index("generator_name", inplace=True)

        # The regions layers only tile the model footprint when the run is scoped with
        # model_topology.include, so the unconditional add-back above leaks far-away
        # plants into the model. Bound it — but only for scoped runs, so unfiltered
        # interconnect/usa runs stay byte-identical.
        if footprint_scoped:
            plants_must_add = _drop_distant_seam_plants(
                plants_must_add,
                regions_onshore,
                regions_offshore,
            )

        if not remaining_plants.empty:
            remaining_clean = remaining_plants.drop(columns=["index_right"], errors="ignore")
            plants_nearshore = gpd.sjoin_nearest(
                remaining_clean,
                regions_onshore.to_crs(epsg=3857),
                how="inner",
                max_distance=2000,
                distance_col="distance",
            )
            plants_nearshore = plants_nearshore.to_crs(epsg=4326)
            plants_filt = pd.concat([plants_filt, plants_nearshore, plants_must_add])
        else:
            plants_filt = pd.concat([plants_filt, plants_must_add])

    plants_filt = plants_filt.drop(columns=["geometry"])
    plants_filt = plants_filt[~plants_filt.index.duplicated()]
    return pd.DataFrame(plants_filt)


def attach_renewable_capacities_to_atlite(
    n: pypsa.Network,
    plants_df: pd.DataFrame,
    renewable_carriers: list,
):
    plants = plants_df.query(
        "bus_assignment in @n.buses.index",
    )
    if "prime_mover_code" in plants:
        plants = plants[plants.prime_mover_code != "PS"]

    for tech in renewable_carriers:
        plants_filt = plants.query("carrier == @tech").copy()
        if plants_filt.empty:
            continue

        generators_tech = n.generators[n.generators.carrier == tech].copy()
        # After aggregate_to_substations, each bus is one substation, so sub_id is
        # collapsed into the bus index; use bus identity directly for grouping.
        generators_tech["sub_assignment"] = generators_tech.bus
        plants_filt["sub_assignment"] = plants_filt.bus_assignment

        build_year_avg = plants_filt.groupby(["sub_assignment"])[plants_filt.columns].apply(
            lambda x: pd.Series(
                {field: weighted_avg(x, field, "p_nom") for field in ["build_year"]},
            ),
        )

        caps_per_bus = (
            plants_filt[["sub_assignment", "p_nom"]].groupby("sub_assignment").sum().p_nom
        )  # namplate capacity per sub_id

        if caps_per_bus[~caps_per_bus.index.isin(generators_tech.sub_assignment)].sum() > 0:
            # p_all = plants_filt[["sub_assignment", "p_nom", "latitude", "longitude"]]
            # missing_plants = p_all[~p_all.sub_assignment.isin(generators_tech.sub_assignment)]
            missing_capacity = caps_per_bus[~caps_per_bus.index.isin(generators_tech.sub_assignment)].sum()
            # missing_plants.to_csv(f"missing_{tech}_plants.csv",)

            logger.info(
                f"There are {np.round(missing_capacity / 1000, 4)} GW of {tech} plants that are not in the network. See git issue #16.",
            )

        logger.info(
            f"{np.round(caps_per_bus.sum() / 1000, 2)} GW of {tech} capacity added.",
        )
        mapped_values = generators_tech.sub_assignment.map(caps_per_bus).dropna()
        n.generators.loc[mapped_values.index, "p_nom"] = mapped_values
        n.generators.loc[mapped_values.index, "p_nom_min"] = mapped_values
        mapped_values = generators_tech.sub_assignment.map(
            build_year_avg.build_year,
        ).dropna()
        n.generators.loc[mapped_values.index, "build_year"] = mapped_values.astype(int)


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
        for carrier in set(conventional_carriers) | set(extendable_carriers["Generator"])
        if carrier not in renewable_carriers
    ]
    add_missing_carriers(n, carriers)

    plants = (
        plants.query("carrier in @carriers")
        .join(costs, on="carrier", rsuffix="_r")
        .rename(index=lambda s: "C" + str(s))
    )

    plants["efficiency"] = plants.efficiency.astype(float).fillna(plants.efficiency_r)

    committable_fields = ["start_up_cost", "min_down_time", "min_up_time"]
    defaults = n.components["Generator"].defaults["default"]
    if unit_commitment:
        for attr in committable_fields:
            plants[attr] = plants[attr].astype(float).fillna(defaults[attr])
        plants["p_min_pu"] = (
            (plants.minimum_load_mw / plants.p_nom)
            .clip(
                upper=np.minimum(plants.summer_derate, plants.winter_derate),
                lower=0,
            )
            .astype(float)
            .fillna(0)
            .mul(0.95)
        )
    else:
        for attr in committable_fields:
            plants[attr] = defaults[attr]
    committable_attrs = {attr: plants[attr] for attr in committable_fields}

    # Define generators using modified ppl DataFrame
    caps = plants.groupby("carrier").p_nom.sum().div(1e3).round(2)
    logger.info(f"Adding {len(plants)} generators with capacities [GW] \n{caps}")
    n.add(
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
        marginal_cost=plants.marginal_cost,
        capital_cost=plants.annualized_capex_fom,
        build_year=plants.build_year.astype(int).fillna(0),
        lifetime=plants.carrier.map(costs.lifetime),
        committable=unit_commitment,
        **committable_attrs,
    )

    # Add fuel and VOM costs to the network
    n.generators.loc[plants.index, "vom_cost"] = plants.carrier.map(
        costs.opex_variable_per_mwh,
    )
    n.generators.loc[plants.index, "fuel_cost"] = plants.fuel_cost
    n.generators.loc[plants.index, "heat_rate"] = plants.heat_rate_mmbtu_per_mwh
    n.generators.loc[plants.index, "ba_eia"] = plants.balancing_authority_code


def normed(s):
    return s / s.sum()


def attach_wind_and_solar(
    n: pypsa.Network,
    costs: pd.DataFrame,
    input_profiles: str,
    carriers: list[str],
    extendable_carriers: dict[str, list[str]],
    config: dict,
):
    add_missing_carriers(n, carriers)

    # Check if we're using horizon-specific profiles
    godeeep_future = (
        config.get("renewable", {}).get("dataset") == "godeeep" and config["renewable_scenarios"][0] != "historical"
    )

    for car in carriers:
        if car in ["hydro", "EGS"]:
            continue

        capital_cost = costs.at[car, "annualized_capex_fom"]

        # Profile bus index already matches the network bus index after
        # cluster_simpl runs upstream — both are keyed by simpl-cluster bus.

        # For GODEEEP future scenarios, load horizon-specific profiles
        if godeeep_future:
            # Load horizon-specific profiles and concatenate
            logger.info(f"Loading multi-horizon {car} profiles for planning horizons: {n.investment_periods.tolist()}")

            all_profiles = []
            p_nom_max_bus = None
            weight_bus = None
            bus_list = None

            for horizon in n.investment_periods:
                profile_attr = f"profile_{car}_{horizon}"
                if not hasattr(input_profiles, profile_attr):
                    raise ValueError(f"Missing profile for {car} at horizon {horizon}")

                with xr.open_dataset(getattr(input_profiles, profile_attr)) as ds:
                    if ds.indexes["bus"].empty:
                        continue

                    # Get bus list (profile bus = network bus, both at cluster level)
                    if bus_list is None:
                        bus_list = ds.bus.values.astype(str)

                        p_nom_max_bus = (
                            ds["p_nom_max"]
                            .to_pandas()
                            .rename(
                                index=lambda b: str(b),
                            )
                        )
                        weight_bus = (
                            ds["weight"]
                            .to_pandas()
                            .rename(
                                index=lambda b: str(b),
                            )
                        )

                    # Get profile for this horizon — index already at bus level
                    horizon_profile = ds["profile"].transpose("time", "bus").to_pandas()
                    horizon_profile.columns = horizon_profile.columns.astype(str)

                    # Update timestamps to match the horizon year
                    horizon_profile.index = horizon_profile.index.map(lambda x: x.replace(year=int(horizon)))
                    all_profiles.append(horizon_profile)

            # No horizon contributed any buses (e.g. no eligible sites for
            # this carrier in the modeled region) — skip the carrier, matching
            # the single-profile branch's empty-bus `continue`.
            if not all_profiles:
                logger.warning(
                    f"No {car} profile buses found in any planning horizon; skipping {car}.",
                )
                continue

            # Concatenate all horizon profiles
            bus_profiles = pd.concat(all_profiles)

            # Align with network snapshots (which should already be multi-indexed)
            bus_profiles = bus_profiles.reindex(n.snapshots.get_level_values(1))
            bus_profiles.index = n.snapshots

        else:
            # Single profile
            with xr.open_dataset(getattr(input_profiles, "profile_" + car)) as ds:
                if ds.indexes["bus"].empty:
                    continue

                bus_list = ds.bus.values.astype(str)
                p_nom_max_bus = (
                    ds["p_nom_max"]
                    .to_pandas()
                    .rename(
                        index=lambda b: str(b),
                    )
                )
                weight_bus = (
                    ds["weight"]
                    .to_pandas()
                    .rename(
                        index=lambda b: str(b),
                    )
                )
                bus_profiles = ds["profile"].transpose("time", "bus").to_pandas()
                bus_profiles.columns = bus_profiles.columns.astype(str)
                # Broadcast single profile across all horizons
                bus_profiles = broadcast_investment_horizons_index(n, bus_profiles)

        logger.info(f"Adding {car} capacity-factor profiles to the network.")

        n.add(
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
            efficiency=1,
            build_year=n.investment_periods[0],
            lifetime=costs.at[car, "lifetime"],
            p_max_pu=bus_profiles,
        )


def attach_egs(
    n: pypsa.Network,
    costs: pd.DataFrame,
    input_profiles: str,
    carriers: list[str],
    extendable_carriers: dict[str, list[str]],
    line_length_factor=1,
):
    """
    Attached STM Calculated wind and solar capacity factor profiles to the
    network.
    """
    car = "EGS"
    if (car not in carriers) and (car not in extendable_carriers["Generator"]):
        return
    add_missing_carriers(n, carriers)
    capital_recovery_period = 25  # Following EGS supply curves by Aljubran et al. (2024)
    discount_rate = 0.07  # load_costs(snakemake.input.tech_costs).loc["geothermal", "wacc_real"]
    drilling_cost = snakemake.config["renewable"]["EGS"]["drilling_cost"]

    with (
        xr.open_dataset(
            getattr(input_profiles, "specs_egs"),
        ) as ds_specs,
        xr.open_dataset(
            getattr(input_profiles, "profile_egs"),
        ) as ds_profile,
    ):
        # After aggregate_egs runs, the ``sub_id`` dimension contains the
        # simpl-cluster bus IDs already, so it can be used as ``bus_id`` directly.
        df_specs = ds_specs.to_dataframe().reset_index().dropna()
        df_specs = df_specs.rename(columns={"sub_id": "bus_id"})
        df_specs["bus_id"] = df_specs["bus_id"].astype(str)
        df_specs = df_specs.set_index("bus_id")

        # columns must be renamed to refer to the right quantities for pypsa to read it correctly
        logger.info(f"Using {drilling_cost} EGS drilling costs.")
        df_specs = df_specs.rename(
            columns={
                ("advanced_capex_usd_kw" if drilling_cost == "advanced" else "capex_usd_kw"): "capital_cost",
                "avail_capacity_mw": "p_nom_max",
                "fixed_om": "fixed_om",
            },
        )

        # TODO: come up with proper values for these params

        df_specs["capital_cost"] = 1000 * (
            df_specs["capital_cost"] * calculate_annuity(capital_recovery_period, discount_rate) + df_specs["fixed_om"]
        )  # convert and annualize USD/kW to USD/MW-year
        df_specs["efficiency"] = 1.0

        df_specs = df_specs.loc[~(df_specs.index == "nan")]

        seismic_path = getattr(input_profiles, "seismic_exclusion", [])
        if seismic_path:
            seismic_gdf = gpd.read_file(seismic_path).rename(
                columns={"seismic risk": "seismic_risk"},
            )
            egs_buses = df_specs.index.intersection(n.buses.index)
            bus_coords = n.buses.loc[egs_buses, ["x", "y"]]
            bus_gdf = gpd.GeoDataFrame(
                bus_coords,
                geometry=gpd.points_from_xy(bus_coords["x"], bus_coords["y"]),
                crs="EPSG:4326",
            ).to_crs(seismic_gdf.crs)
            bus_with_risk = gpd.sjoin_nearest(
                bus_gdf[["geometry"]],
                seismic_gdf[["geometry", "seismic_risk"]],
                how="left",
            )
            excluded_buses = bus_with_risk[bus_with_risk["seismic_risk"] == 1].index
            df_specs = df_specs[~df_specs.index.isin(excluded_buses)]
            logger.info(
                f"Seismic risk mask excluded {len(excluded_buses)} EGS buses. {len(df_specs)} buses remaining.",
            )

        qualities = snakemake.config["renewable"]["EGS"].get("quality", [1])

        for q in qualities:
            suffix = " " + car + f" Q{q}"
            df_q = df_specs[df_specs["Quality"] == q]

            bus_list = df_q.index.values
            capital_cost = df_q["capital_cost"]
            p_nom_max_bus = df_q["p_nom_max"]
            efficiency = df_q["efficiency"]  # for now.

            df_q_profile = ds_profile.sel(Quality=q).to_dataframe().dropna().reset_index()
            df_q_profile = df_q_profile.rename(columns={"sub_id": "bus_id"})
            df_q_profile["bus_id"] = df_q_profile["bus_id"].astype(str)
            bus_profiles = pd.pivot_table(
                df_q_profile,
                columns="bus_id",
                index=["year", "Date"],
                values="capacity_factor",
            )

            # Align bus_profiles to network snapshots.
            # The EGS data has year=float (2030.0) and Date using the
            # investment year (2030-01-01...) at hourly resolution.
            # n.snapshots has period=int (2030) at the network's resolution.
            # Cast the outer level to int and reindex so that:
            #   (a) the float→int type is resolved, and
            #   (b) the hourly profile is downsampled to the network's timestep.
            # This also preserves the per-period production decline since the
            # EGS data already has different CF values for 2030/2040/2050.
            if hasattr(n.snapshots, "levels"):
                bus_profiles.index = bus_profiles.index.set_levels(
                    bus_profiles.index.levels[0].astype(int),
                    level=0,
                )
                bus_profiles = bus_profiles.reindex(n.snapshots)
            bus_profiles = bus_profiles.reindex(columns=bus_list)

            logger.info(
                f"Adding EGS (Resource Quality-{q}) capacity-factor profiles to the network.",
            )

            n.add(
                "Generator",
                bus_list,
                suffix,
                bus=bus_list,
                carrier=car,
                p_nom_extendable=car in extendable_carriers["Generator"],
                p_nom_max=p_nom_max_bus,
                capital_cost=capital_cost,
                efficiency=efficiency,
                p_max_pu=bus_profiles,
                build_year=n.investment_periods[0],
                lifetime=capital_recovery_period,
                land_region=bus_list,
            )


def attach_battery_storage(
    n: pypsa.Network,
    costs: pd.DataFrame,
    plants: pd.DataFrame,
):
    """Attaches Existing Battery Energy Storage Systems To the Network."""
    plants_filt = plants.query("carrier == 'battery' ")
    plants_filt.index = plants_filt.index.astype(str) + "_" + plants_filt.generator_id.astype(str)
    plants_filt.loc[:, "energy_storage_capacity_mwh"] = plants_filt.energy_storage_capacity_mwh.astype(float)
    plants_filt = plants_filt.dropna(subset=["energy_storage_capacity_mwh"])

    logger.info(
        f"Added Batteries as Storage Units to the network.\n{np.round(plants_filt.p_nom.sum() / 1000, 2)} GW Power Capacity \n{np.round(plants_filt.energy_storage_capacity_mwh.sum() / 1000, 2)} GWh Energy Capacity",
    )

    plants_filt = plants_filt.dropna(subset=["energy_storage_capacity_mwh"])
    n.add(  # Adds storage units which can retire economically or at their lifetime
        "StorageUnit",
        plants_filt.index,
        carrier="battery",
        bus=plants_filt.bus_assignment,
        p_nom=plants_filt.p_nom,
        p_nom_max=plants_filt.p_nom,
        p_nom_min=0,
        p_nom_extendable=False,  # Only Allow lifetime retirments for existing BESS
        capital_cost=costs.at["4hr_battery_storage", "opex_fixed_per_kw"] * 1e3,
        max_hours=plants_filt.energy_storage_capacity_mwh / plants_filt.p_nom / 0.85**0.5,
        build_year=plants_filt.build_year,
        lifetime=costs.at["4hr_battery_storage", "lifetime"],
        efficiency_store=0.85**0.5,
        efficiency_dispatch=0.85**0.5,
        cyclic_state_of_charge=True,
        cyclic_state_of_charge_per_period=True,  # pypsa v1 flipped this default to False
    )


def attach_phs_storage(
    n: pypsa.Network,
    plants: pd.DataFrame,
):
    """Attach existing pumped hydro storage from EIA prime mover PS units."""
    efficiency_dispatch = 0.894427191
    plants_filt = plants.query(
        "prime_mover_code == 'PS' and bus_assignment in @n.buses.index",
    ).copy()
    plants_filt = plants_filt.dropna(subset=["p_nom"])
    if plants_filt.empty:
        logger.info("No PHS storage units found in powerplants.csv.")
        return

    logger.info(
        f"Added PHS as Storage Units to the network.\n{np.round(plants_filt.p_nom.sum() / 1000, 2)} GW Power Capacity",
    )

    n.add(
        "StorageUnit",
        plants_filt.index,
        carrier="PHS",
        bus=plants_filt.bus_assignment,
        p_nom=plants_filt.p_nom,
        p_nom_extendable=False,
        build_year=plants_filt.build_year.astype(int),
        lifetime=np.inf,
        max_hours=24.0 / efficiency_dispatch,
        efficiency_store=efficiency_dispatch,
        efficiency_dispatch=efficiency_dispatch,
        cyclic_state_of_charge=True,
        cyclic_state_of_charge_per_period=True,  # pypsa v1 flipped this default to False
    )


def broadcast_investment_horizons_index(n: pypsa.Network, df: pd.DataFrame):
    """
    Broadcast the index of a dataframe to match the potentially multi-indexed
    investment periods of a PyPSA network.
    """
    sns = n.snapshots
    if not len(df.index) == len(sns):  # if broadcasting is necessary
        df.index = pd.to_datetime(df.index)
        dfs = []
        for planning_horizon in n.investment_periods.to_list():
            period_data = df.copy()
            period_data.index = df.index.map(lambda x: x.replace(year=planning_horizon))
            dfs.append(period_data)
        df = pd.concat(dfs)
        df = pd.merge(
            df,
            sns.to_frame().droplevel(0),
            left_index=True,
            right_index=True,
        ).drop(columns=["period", "timestep"])
        assert len(df.index) == len(sns)
    df.index = sns
    return df


def apply_seasonal_capacity_derates(
    n: pypsa.Network,
    plants: pd.DataFrame,
    conventional_carriers: list,
    sns: pd.DatetimeIndex,
):
    """Applies conventional rerate factor p_max_pu based on the seasonal capacity derates defined in eia860."""
    sns_dt = sns.get_level_values(1)
    summer_sns = sns_dt[sns_dt.month.isin([6, 7, 8])]
    winter_sns = sns_dt[~sns_dt.month.isin([6, 7, 8])]

    # conventional_carriers = ['geothermal'] # testing override impact

    conv_plants = plants.query("carrier in @conventional_carriers")
    conv_plants.index = "C" + conv_plants.index
    conv_gens = n.generators.query("carrier in @conventional_carriers")

    p_max_pu = pd.DataFrame(1.0, index=sns_dt, columns=conv_gens.index)
    p_max_pu.loc[summer_sns, conv_gens.index] *= conv_plants.loc[
        :,
        "summer_derate",
    ].astype(float)
    p_max_pu.loc[winter_sns, conv_gens.index] *= conv_plants.loc[
        :,
        "winter_derate",
    ].astype(float)

    p_max_pu = broadcast_investment_horizons_index(n, p_max_pu)
    n.generators_t.p_max_pu = pd.concat(
        [n.generators_t.p_max_pu, p_max_pu],
        axis=1,
    ).round(3)


def apply_must_run_ratings(
    n: pypsa.Network,
    plants: pd.DataFrame,
    conventional_carriers: list,
    sns: pd.DatetimeIndex,
):
    """Applies Minimum Loading Capacities only to WECC ADS designated Plants."""
    conv_plants = plants.query("carrier in @conventional_carriers").copy()
    conv_plants.index = "C" + conv_plants.index

    conv_plants.loc[:, "ads_mustrun"] = conv_plants.ads_mustrun.infer_objects(
        copy=False,
    ).fillna(False)

    conv_plants.loc[:, "minimum_load_pu"] = conv_plants.minimum_load_mw / conv_plants.p_nom
    conv_plants.loc[:, "minimum_load_pu"] = (
        conv_plants.minimum_load_pu.clip(
            upper=np.minimum(conv_plants.summer_derate, conv_plants.winter_derate),
            lower=0,
        )
        .astype(float)
        .fillna(0)
    )
    must_run = conv_plants.query("ads_mustrun == True")
    n.generators.loc[must_run.index, "p_min_pu"] = must_run.minimum_load_pu.round(3) * 0.95


def clean_bus_data(n: pypsa.Network):
    """Drops data from the network that are no longer needed in workflow."""
    col_list = [
        # "Pd",
        "load_dissag",
        "LAF",
        "LAF_state",
    ]
    n.buses = n.buses.drop(columns=[col for col in col_list if col in n.buses])


def attach_breakthrough_renewable_plants(
    n,
    fn_plants,
    renewable_carriers,
    extendable_carriers,
    costs,
):
    add_missing_carriers(n, renewable_carriers)

    plants = pd.read_csv(fn_plants, dtype={"bus_id": str}, index_col=0)
    plants = plants.replace(["wind_offshore"], ["offwind"])

    # The network at this stage is substation-level (post aggregate_to_substations
    # and cluster_simpl), while the breakthrough base-grid files reference RAW
    # base-network bus_ids. Remap every plant through the busmap chain
    # (raw bus_id -> sub_id -> cluster bus) before matching against n.buses,
    # patterned after aggregate_egs. Raw ids belonging to other interconnects are
    # absent from bus2sub and drop out naturally, which also removes accidental
    # attachments where a foreign raw id collides with a local substation id.
    # All ids are compared as plain integer-strings ("35827", never "35827.0").
    bus2sub = pd.read_csv(snakemake.input.bus2sub, dtype=str)
    raw_to_sub = bus2sub.assign(
        sub_id=bus2sub["sub_id"].str.replace(r"\.0$", "", regex=True),
    ).set_index("Bus")["sub_id"]
    busmap_s = pd.read_csv(snakemake.input.busmap_s, dtype=str)
    sub_to_cluster = busmap_s.assign(
        sub_id=busmap_s["sub_id"].str.replace(r"\.0$", "", regex=True),
        cluster_bus=busmap_s["cluster_bus"].str.replace(r"\.0$", "", regex=True),
    ).set_index("sub_id")["cluster_bus"]
    n_plants_raw = len(plants)
    plants["bus_id"] = plants["bus_id"].map(raw_to_sub).map(sub_to_cluster)
    plants = plants.dropna(subset=["bus_id"]).query("bus_id in @n.buses.index")
    logger.info(
        f"Remapped breakthrough plants through bus2sub/busmap_s: kept {len(plants)} of {n_plants_raw} plants on this network.",
    )

    for tech in renewable_carriers:
        assert tech == "hydro"
        tech_plants = plants.query("type == @tech")
        tech_plants.index = tech_plants.index.astype(str)
        logger.info(f"Adding {len(tech_plants)} {tech} generators to the network.")

        p_nom_be = pd.read_csv(snakemake.input[f"{tech}_breakthrough"], index_col=0)

        intersection = set(p_nom_be.columns).intersection(
            tech_plants.index,
        )  # filters by plants ID for the plants of type tech
        p_nom_be = p_nom_be[list(intersection)]

        p_nom_be.columns = p_nom_be.columns.astype(str)

        if (tech_plants.Pmax == 0).any():
            # p_nom is the maximum of {Pmax, dispatch}
            p_nom = pd.concat([p_nom_be.max(axis=0), tech_plants["Pmax"]], axis=1).max(
                axis=1,
            )
            p_max_pu = (p_nom_be[p_nom.index] / p_nom).astype(float).fillna(0)  # some values remain 0
        else:
            p_nom = tech_plants.Pmax
            p_max_pu = p_nom_be[tech_plants.index] / p_nom

        leap_day = p_max_pu.loc["2016-02-29 00:00:00":"2016-02-29 23:00:00"]
        p_max_pu = p_max_pu.drop(leap_day.index)
        p_max_pu = broadcast_investment_horizons_index(n, p_max_pu)

        n.add(
            "Generator",
            tech_plants.index,
            bus=tech_plants.bus_id,
            p_nom_min=p_nom,
            p_nom=p_nom,
            marginal_cost=0,
            p_max_pu=p_max_pu,  # timeseries of max power output pu
            p_nom_extendable=False,
            carrier=tech,
            weight=1.0,
            build_year=n.investment_periods[0],
            lifetime=np.inf,
        )
    return n


def apply_pudl_fuel_costs(
    n,
    plants,
    costs,
):
    # Apply PuDL Fuel Costs for plants where listed
    pudl_fuel_costs = pd.read_csv(snakemake.input["pudl_fuel_costs"], index_col=0)

    # Check if any of the plants are in the pudl fuel costs
    if not set(plants.index).intersection(pudl_fuel_costs.columns):
        return n

    # Construct the VOM table for each generator by carrier
    vom = pd.DataFrame(index=pudl_fuel_costs.columns)
    for gen in pudl_fuel_costs.columns:
        if gen not in plants.index:
            continue
        carrier = plants.loc[gen, "carrier"]
        if carrier not in costs.index:
            continue
        vom.loc[gen, "VOM"] = costs.at[carrier, "opex_variable_per_mwh"]

    # Apply the VOM to the fuel costs
    pudl_fuel_costs = pudl_fuel_costs + vom.squeeze()
    pudl_fuel_costs = broadcast_investment_horizons_index(n, pudl_fuel_costs)

    # Drop any columns that are not in the network
    pudl_fuel_costs.columns = "C" + pudl_fuel_costs.columns
    pudl_fuel_costs = pudl_fuel_costs[[x for x in pudl_fuel_costs.columns if x in n.generators.index]]

    # drop any data that has been assigned at a coarser resolution
    n.generators_t["marginal_cost"] = n.generators_t["marginal_cost"][
        [x for x in n.generators_t["marginal_cost"] if x not in pudl_fuel_costs]
    ]

    # assign new marginal costs
    n.generators_t["marginal_cost"] = n.generators_t["marginal_cost"].join(
        pudl_fuel_costs,
    )
    logger.info(
        f"Applied PuDL fuel costs to {len(pudl_fuel_costs.columns)} generators.",
    )
    return n


def main(snakemake):
    params = snakemake.params
    interconnection = snakemake.wildcards["interconnect"]

    n = pypsa.Network(snakemake.input.base_network)
    schema_entry = log_network_schema(n, stage="entry")

    regions_onshore = gpd.read_file(snakemake.input.regions_onshore)
    regions_offshore = gpd.read_file(snakemake.input.regions_offshore)
    reeds_shapes = gpd.read_file(snakemake.input.reeds_shapes)
    all_reeds_shapes = gpd.read_file(snakemake.input.all_reeds_shapes)
    reeds_memberships = pd.read_csv(snakemake.input.reeds_memberships)

    costs = load_costs(snakemake.input.tech_costs, params.costs)
    # In the simplify-early DAG this network comes from aggregate_to_substations,
    # whose assign_line_lengths already folded lines.length_factor into `length`.
    # Passing the factor again here would compound it (25% CAPEX inflation on
    # every line) — the pre-refactor pipeline applied it exactly once.
    # DECISION (user, 2026-08-18): keep length_factor=1.0 here so this stage
    # never edits the length data. NOTE: these capital costs only reach the
    # solve on the TAMU (line-preserving) network; under the reeds transport
    # model, lines/DC links are dropped at clustering and ITL link costs are
    # rebuilt from the ReEDS distance-cost tables (see deltas ledger DL-1/DL-2).
    update_transmission_costs(n, costs, length_factor=1.0)

    renewable_carriers = set(params.renewable_carriers)
    extendable_carriers = params.extendable_carriers
    conventional_carriers = params.conventional_carriers
    conventional_inputs = {k: v for k, v in snakemake.input.items() if k.startswith("conventional_")}

    plants = load_powerplants(
        snakemake.input["powerplants"],
        n.investment_periods,
        interconnect=interconnection,
    )
    # A run scoped with model_topology.include tiles regions over the footprint only;
    # the seam-plant fallback in filter_plants_by_region must then be distance-bounded.
    include_filter = snakemake.config.get("model_topology", {}).get("include") or {}
    plants = filter_plants_by_region(
        plants,
        regions_onshore,
        regions_offshore,
        reeds_shapes,
        all_reeds_shapes,
        reeds_memberships,
        footprint_scoped=bool(include_filter),
    )
    plants = match_plant_to_bus(n, plants)

    attach_egs(
        n,
        costs,
        snakemake.input,
        renewable_carriers,
        extendable_carriers,
        params.length_factor,
    )

    attach_conventional_generators(
        n,
        costs,
        plants,
        conventional_carriers,
        extendable_carriers,
        params.conventional,
        renewable_carriers,
        conventional_inputs,
        unit_commitment=params.conventional["unit_commitment"],
        fuel_price=None,  # update fuel prices later
    )
    apply_seasonal_capacity_derates(
        n,
        plants,
        conventional_carriers,
        n.snapshots,
    )

    if params.conventional.get("must_run", False):
        # TODO (@ktehranchi): In the future the plants that are must-run should
        # not be clustered and instead retire according to lifetime
        apply_must_run_ratings(
            n,
            plants,
            conventional_carriers,
            n.snapshots,
        )

    attach_battery_storage(
        n,
        costs,
        plants,
    )
    attach_phs_storage(
        n,
        plants,
    )

    attach_wind_and_solar(
        n,
        costs,
        snakemake.input,
        renewable_carriers,
        extendable_carriers,
        snakemake.config,
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
        update_capital_costs(n, carrier, costs, df_multiplier)

    if params.conventional["dynamic_fuel_price"].get("enable", False):
        logger.info("Applying dynamic fuel pricing to conventional generators")
        if params.conventional["dynamic_fuel_price"]["wholesale"]:
            assert params.eia_api, "Must provide EIA API key for dynamic fuel pricing"

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
                        df = pd.read_csv(
                            snakemake.input[datafile],
                            index_col="snapshot",
                        )
                        if df.empty:
                            logger.warning(f"No data provided for {datafile}")
                            continue
                    except KeyError:
                        logger.warning(f"Can not find dynamic price file {datafile}")
                        continue

                    vom = costs.at[carrier, "opex_variable_per_mwh"]

                    apply_dynamic_pricing(
                        n=n,
                        carrier=carrier,
                        geography=area,
                        df=df,
                        vom=vom,
                    )
                    logger.info(
                        f"Applied dynamic price data for {carrier} from {datafile}",
                    )

        if params.conventional["dynamic_fuel_price"]["pudl"]:
            n = apply_pudl_fuel_costs(n, plants, costs)

    # fix p_nom_min for extendable generators
    # The "- 0.001" is just to avoid numerical issues
    n.generators["p_nom_min"] = n.generators.apply(
        lambda x: (x["p_nom"] - 0.001) if (x["p_nom_extendable"] and x["p_nom_min"] == 0) else x["p_nom_min"],
        axis=1,
    )

    output_folder = os.path.dirname(snakemake.output[0]) + "/base_network"
    export_network_for_gis_mapping(n, output_folder)

    clean_bus_data(n)
    sanitize_carriers(n, snakemake.config)
    n.meta = snakemake.config

    log_network_schema(n, stage="exit", baseline=schema_entry)
    # n.export_to_netcdf(snakemake.output[0])
    pickle.dump(n, open(snakemake.output[0], "wb"))


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("add_electricity", interconnect="western")
    configure_logging(snakemake)
    main(snakemake)
