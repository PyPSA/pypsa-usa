# By PyPSA-USA Authors
"""
Region-resolved capital costs from NREL ReEDS.

Component overnight-capex multipliers (region)
----------------------------------------------
``reg_cap_cost_diff_default.csv`` is indexed by county (``p`` + 5-digit FIPS) with
``|``-delimited technology-group columns whose values are relative differences
centred on 0 (ReEDS applies them as ``1 + x``). They scale the **overnight** capex
only -- never FOM/VOM -- and are dimensionless. The group -> pypsa-carrier mapping
comes from config ``costs: regional_multipliers: column_to_tech``. Buses are mapped
to counties geographically (point in polygon, nearest polygon as fallback), because
clustered buses no longer carry a reliable ``county``.

Wind and solar interconnection cost (site)
------------------------------------------
``reg_cap_cost_diff`` has no wind or UPV column. Wind and solar instead get the
ReEDS per-site interconnection cost (spur line + point of interconnection +
network reinforcement), capacity-weighted per bus in ``build_renewable_profiles``
and stored in the profile as ``constants.RENEWABLE_INTERCONNECTION_COL``. It replaces
the ATB's uniform ``capex_grid_connection_per_kw``, which ``annualized_capex_fom``
already holds, so the generic connection cost is not counted twice. It is switched
on with ``costs: regional_multipliers: renewable_interconnection`` and works with
every ``renewable.dataset``. A carrier priced this way does not also take the EIA
state locational multiplier.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Equal-area CONUS projection: nearest-neighbour distances must be metric.
EQUAL_AREA_CRS = 5070


# ---------------------------------------------------------------------------
# Component overnight-capex multipliers
# ---------------------------------------------------------------------------


def load_reg_cap_cost_diff(path: str) -> pd.DataFrame:
    """Load the ReEDS relative-cost-difference file, indexed by county code ``r``."""
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "r"}).set_index("r")
    return df.apply(pd.to_numeric, errors="coerce")


def county_multiplier_table(reg_cap_diff: pd.DataFrame, column_to_tech: dict[str, list[str]]) -> pd.DataFrame:
    """Expand ReEDS group columns to a (county x pypsa-carrier) table of ``1 + x``.

    Carriers absent from ``column_to_tech`` are not produced; callers default them
    to 1.0.
    """
    if not column_to_tech:
        raise ValueError("column_to_tech is required (config costs: regional_multipliers: column_to_tech).")
    out = {}
    for column, carriers in column_to_tech.items():
        if column not in reg_cap_diff.columns:
            logger.warning("reg_cap_cost_diff column %r not found; skipping.", column)
            continue
        mult = 1.0 + reg_cap_diff[column].fillna(0.0)
        for carrier in carriers:
            out[carrier] = mult
    table = pd.DataFrame(out)
    table.index.name = "r"
    return table


def assign_bus_counties(buses: pd.DataFrame, county_shapes_path: str, valid_counties=None) -> pd.Series:
    """Map each bus to a county code (``p`` + FIPS) by geographic proximity.

    A bus inside a county polygon gets that county; one outside every eligible
    polygon (offshore, or in a county ``valid_counties`` excludes) takes the
    nearest one. Accepts both the raw census shapefile (bare ``GEOID``) and
    ``build_shapes``'s ``p``-prefixed ``county_shapes.geojson``.
    """
    import geopandas as gpd

    counties = gpd.read_file(county_shapes_path)
    geoid = counties["GEOID"].astype(str)
    counties["r"] = geoid.where(geoid.str.startswith("p"), "p" + geoid)

    if valid_counties is not None:
        counties = counties[counties["r"].isin(set(valid_counties))]
    if counties.empty:
        raise ValueError("No county polygons remain after filtering.")

    points = gpd.GeoDataFrame(
        index=buses.index,
        geometry=gpd.points_from_xy(buses["x"].astype(float), buses["y"].astype(float)),
        crs="EPSG:4326",
    ).to_crs(EQUAL_AREA_CRS)
    joined = gpd.sjoin_nearest(points, counties[["r", "geometry"]].to_crs(EQUAL_AREA_CRS), how="left")
    # sjoin_nearest emits one row per tied polygon; collapse deterministically.
    return joined.groupby(level=0)["r"].first().reindex(buses.index)


def bus_multiplier_table(buses: pd.DataFrame, county_table: pd.DataFrame, county_shapes_path: str) -> pd.DataFrame:
    """(bus x pypsa-carrier) overnight-capex multiplier table; unmatched buses get 1.0."""
    bus_county = assign_bus_counties(buses, county_shapes_path, valid_counties=county_table.index)
    unresolved = int(bus_county.isna().sum())
    if unresolved:
        logger.warning("%d bus(es) could not be matched to a county; using multiplier 1.0.", unresolved)

    table = county_table.reindex(bus_county.to_numpy())
    table.index = buses.index
    table = table.astype(float).fillna(1.0)
    for carrier in table.columns:
        col = table[carrier]
        logger.info(
            "Regional overnight-capex multiplier for %r: min=%.3f mean=%.3f max=%.3f over %d bus(es).",
            carrier,
            col.min(),
            col.mean(),
            col.max(),
            len(col),
        )
    return table


def build_bus_multipliers(n, cfg: dict | None, reg_cap_cost_diff_fn: str | None, county_shapes_fn: str | None):
    """Bus multiplier table for the AC buses of ``n``, or ``None`` when disabled."""
    cfg = cfg or {}
    if not cfg.get("enable", False):
        return None
    if not reg_cap_cost_diff_fn or not county_shapes_fn:
        raise ValueError(
            "costs.regional_multipliers.enable is true but the reg_cap_cost_diff or county_shapes input is missing.",
        )
    county_table = county_multiplier_table(load_reg_cap_cost_diff(reg_cap_cost_diff_fn), cfg.get("column_to_tech"))
    ac_buses = n.buses[n.buses.carrier == "AC"] if "carrier" in n.buses.columns else n.buses
    return bus_multiplier_table(ac_buses, county_table, county_shapes_fn)


def carrier_multiplier(bus_multipliers, carrier: str, buses_i):
    """Per-bus multiplier array for ``carrier`` aligned to ``buses_i``; scalar 1.0 if not mapped."""
    if bus_multipliers is None or carrier not in bus_multipliers.columns:
        return 1.0
    return bus_multipliers[carrier].reindex(buses_i).fillna(1.0).to_numpy()


def carrier_capital_charge_rate(costs: pd.DataFrame, carrier: str) -> float:
    """Annual charge per unit of capex ``build_cost_data`` annualized ``carrier`` at.

    That is its ``capital_charge_rate`` (the ATB fixed charge rate, with the ITC of a
    ``+ TC`` case), or the capital recovery factor of its WACC for a cost table without
    one (``load_costs`` zero-fills a missing parameter).
    """
    from _helpers import calculate_annuity

    rate = costs.at[carrier, "capital_charge_rate"] if "capital_charge_rate" in costs.columns else np.nan
    if pd.notna(rate) and rate > 0:
        return float(rate)
    return float(calculate_annuity(costs.at[carrier, "cost_recovery_period_years"], costs.at[carrier, "wacc_real"]))


def overnight_delta_capital_cost(annualized_capex_fom, capex_overnight_per_kw, annuity_factor, multiplier):
    """Capital cost with only the overnight portion scaled by ``multiplier``.

    ``capital_cost = annualized_capex_fom + annuity * (m - 1) * capex_overnight_per_kw * 1e3``

    FOM, grid connection and construction finance bundled in
    ``annualized_capex_fom`` are left untouched. ``capex_overnight_per_kw`` is in
    USD/kW, hence the ``1e3`` to USD/MW.
    """
    return annualized_capex_fom + annuity_factor * (multiplier - 1.0) * capex_overnight_per_kw * 1e3


def regional_capital_cost(costs: pd.DataFrame, carrier: str, buses, bus_multipliers):
    """Per-bus ``capital_cost`` for an ATB carrier with the regional overnight multiplier.

    ``buses`` are the bus names the components sit at (one per component, repeats
    allowed). Returns the uniform ``annualized_capex_fom`` whenever the feature is
    off, the carrier is not mapped, or the cost table has no overnight capex. The
    overnight delta is annualized at the carrier's capital charge rate, like the
    rest of its capex.
    """
    base = costs.at[carrier, "annualized_capex_fom"]
    multiplier = carrier_multiplier(bus_multipliers, carrier, pd.Index(buses))
    if np.isscalar(multiplier):
        return base
    overnight = costs.at[carrier, "capex_overnight_per_kw"] if "capex_overnight_per_kw" in costs.columns else np.nan
    if not np.isfinite(overnight) or overnight == 0:
        logger.warning("No usable capex_overnight_per_kw for %r; skipping its regional multiplier.", carrier)
        return base
    return overnight_delta_capital_cost(base, overnight, carrier_capital_charge_rate(costs, carrier), multiplier)


def renewable_interconnection_enabled(cfg: dict | None) -> bool:
    """True when ``costs: regional_multipliers`` switches the ReEDS wind/solar interconnection cost on."""
    cfg = cfg or {}
    return bool(cfg.get("enable", False) and cfg.get("renewable_interconnection", False))


def renewable_interconnection_capital_cost(costs: pd.DataFrame, carrier: str, interconnection_capex) -> pd.Series:
    """Wind/solar ``capital_cost`` with the ATB grid connection replaced by the ReEDS cost.

    ``capital_cost = annualized_capex_fom + annuity * (interconnection - capex_grid_connection_per_kw * 1e3)``

    ``annualized_capex_fom`` is ``rate * (overnight + grid connection +
    construction finance) * 1e3 + FOM * 1e3`` with ``rate`` the carrier's capital
    charge rate, so this swaps only the ATB's generic
    grid connection for the site-resolved spur/POI/reinforcement cost; overnight
    capex, construction finance and FOM are untouched. ``interconnection_capex`` is
    overnight USD/MW of the same dollar year as ``costs`` (a Series, one value per
    component). A missing or non-positive value keeps the uniform ATB cost.
    """
    interconnection = pd.Series(interconnection_capex, dtype=float)
    base = float(costs.at[carrier, "annualized_capex_fom"])
    grid_connection = (
        costs.at[carrier, "capex_grid_connection_per_kw"] if "capex_grid_connection_per_kw" in costs else 0.0
    )
    grid_connection = 0.0 if pd.isna(grid_connection) else float(grid_connection)
    rate = carrier_capital_charge_rate(costs, carrier)
    priced = base + rate * (interconnection - grid_connection * 1e3)
    return priced.where(interconnection > 0, base)
