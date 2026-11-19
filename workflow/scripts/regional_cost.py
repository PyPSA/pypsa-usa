# By PyPSA-USA Authors
"""
Region-resolved capital costs from NREL ReEDS.

Two independent features live here.

Transmission (region x voltage)
-------------------------------
A single national USD/MW-km cannot express that a 69 kV line costs several times
more per MW-km than a 500 kV line, nor that a corridor in Manhattan costs more
than one across Kansas. Two ReEDS tables supply those dimensions:

* ``transmission_distance_cost_500kVac_county.csv`` -- 500 kV AC cost for ~799k
  county pairs. Its ``length_miles`` is the great-circle distance between county
  centroids, so ReEDS's terrain and route-detour costs live entirely in the
  *price*. Dividing gives a USD/MW **per great-circle km** that already carries
  regional base cost, terrain and detour.
* ``rev_transmission_basecost.csv`` -- USD/MW-mile by voltage class for four ReEDS
  cost regions. Each voltage's ratio to that region's own 500 kV AC entry is taken
  first, then the median across regions, which isolates the voltage relationship
  from the regional level (already carried by the county table). The ratios are
  reduced to one power law ``ratio = (kV / 500)^b`` fitted through 500 kV.

Both tables are in 2004 USD and are converted to ``costs: dollar_year`` with
:mod:`money`.

Line voltages are collapsed to 230 kV in ``aggregate_to_substations``, and the
``county`` bus attribute is dropped there for most topologies, so the costing is
done on the base network *before* that step: :func:`assign_branch_capex` writes the
**overnight** cost per MW per great-circle km into :data:`BRANCH_CAPEX_COL` of every
line and HVDC link. That intensive column then rides through every clustering pass
with :data:`LINE_CAPEX_STRATEGIES` (capacity-weighted mean, exactly how pypsa
aggregates ``capital_cost / length``), and ``add_electricity.update_transmission_costs``
turns it into ``capital_cost`` = annuity x unit cost x great-circle length of the
current buses.

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

Clustering
----------
``capital_cost`` is linear in the county multiplier and in the interconnection
cost, and clustering takes the capacity-weighted mean of all three (see
``constants.GENERATOR_CAPEX_STRATEGIES``). ``add_extra_components`` therefore
rebuilds the capital cost of every investment period from the columns a merged
generator carries, and gets the value the single-period network holds. A generator
without a ReEDS site keeps the ATB grid connection, so a merged generator also
carries the share of its capacity that has a site
(``constants.RENEWABLE_INTERCONNECTION_SHARE_COL``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Equal-area CONUS projection: nearest-neighbour distances must be metric.
EQUAL_AREA_CRS = 5070

# ---------------------------------------------------------------------------
# Transmission line costs: voltage level and route region
# ---------------------------------------------------------------------------

MILES_TO_KM = 1.609344

#: Currency year of both ReEDS transmission tables.
REEDS_TRANSMISSION_COST_YEAR = 2004

#: The county cost field for one county is the median over its ``K`` shortest
#: incident county pairs. A county carries ~500 pairs reaching ~700 km and the
#: per-km cost falls with pair length, so a whole-county average blurs the local
#: corridor and biases short lines low. K=8 is about the number of neighbours a
#: US county has.
NEAR_NEIGHBOUR_PAIRS = 8

#: Cost-region columns of ``rev_transmission_basecost.csv``.
REEDS_BASECOST_REGIONS = ("TEPPC", "SCE", "MISO", "Southeast")

#: Voltage the county cost table is quoted at, and the denominator of the ratios.
REFERENCE_VOLTAGE_KV = 500.0

#: Line/Link column holding the overnight transmission cost in ``dollar_year``
#: USD per MW per great-circle km, already region- and voltage-resolved.
BRANCH_CAPEX_COL = "capex_overnight_per_mw_km"

#: pypsa clustering strategy for :data:`BRANCH_CAPEX_COL`. Without it pypsa's
#: default "consense" strategy raises as soon as two merged lines differ.
LINE_CAPEX_STRATEGIES = {BRANCH_CAPEX_COL: "capacity_weighted_average"}

#: Link carriers priced as transmission corridors.
TRANSMISSION_LINK_CARRIERS = ("AC", "DC")


def load_transmission_basecost(path: str) -> pd.DataFrame:
    """Load ``rev_transmission_basecost.csv``, indexed by voltage-class label.

    The file carries a units row under the header (``kV``, ``MW``, ...) which is
    dropped, and is written with a BOM.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["Voltage"] != "kV"].set_index("Voltage")
    numeric = ["Capacity", *REEDS_BASECOST_REGIONS]
    missing = set(numeric).difference(df.columns)
    if missing:
        raise ValueError(f"rev_transmission_basecost is missing columns {sorted(missing)}.")
    return df[numeric].apply(pd.to_numeric)


def voltage_cost_anchors(basecost: pd.DataFrame) -> pd.Series:
    """Per-voltage cost ratios relative to 500 kV AC, indexed by kV, ascending.

    Ratios are formed inside each cost region first, then the median is taken
    across regions (69 kV/500 kV runs 2.97 in Southeast to 6.62 in MISO, so the
    median keeps one study from dominating).
    """
    ac_rows = basecost.index[basecost.index.str.contains("AC")]
    if ac_rows.empty:
        raise ValueError("rev_transmission_basecost has no AC voltage rows.")
    ac = basecost.loc[ac_rows, list(REEDS_BASECOST_REGIONS)]
    ac.index = ac_rows.str.extract(r"(\d+)", expand=False).astype(float)

    if REFERENCE_VOLTAGE_KV not in ac.index:
        raise ValueError(f"rev_transmission_basecost has no {REFERENCE_VOLTAGE_KV:.0f} kV AC row.")
    ratios = ac.div(ac.loc[REFERENCE_VOLTAGE_KV])
    anchors = ratios.median(axis=1).sort_index()
    anchors.index.name = "v_nom"
    logger.info(
        "Voltage cost anchors relative to %.0f kV AC: %s",
        REFERENCE_VOLTAGE_KV,
        ", ".join(f"{kv:.0f}kV={ratio:.4f}" for kv, ratio in anchors.items()),
    )
    return anchors


def dc_ac_line_cost_ratio(basecost: pd.DataFrame) -> float:
    """500 kV DC bipole line cost as a fraction of 500 kV AC single-circuit.

    All four ReEDS cost regions give the same ratio (0.4137); the agreement is
    asserted so a future data refresh cannot silently collapse a regional spread.
    """
    ac_label = f"{REFERENCE_VOLTAGE_KV:.0f}ACsingle"
    dc_label = f"{REFERENCE_VOLTAGE_KV:.0f}DCbipole"
    for label in (ac_label, dc_label):
        if label not in basecost.index:
            raise ValueError(f"rev_transmission_basecost has no {label!r} row.")

    regions = list(REEDS_BASECOST_REGIONS)
    per_region = basecost.loc[dc_label, regions] / basecost.loc[ac_label, regions]
    if not np.allclose(per_region, per_region.iloc[0], rtol=1e-4):
        raise ValueError(
            "DC/AC line cost ratio differs across ReEDS cost regions "
            f"({per_region.round(6).to_dict()}); the county cost table only resolves AC, "
            "so a per-region DC ratio would need its own regional mapping.",
        )
    ratio = float(per_region.mean())
    logger.info("DC/AC line cost ratio at %.0f kV: %.6f", REFERENCE_VOLTAGE_KV, ratio)
    return ratio


def voltage_cost_exponent(anchors: pd.Series) -> float:
    """Fit ``ratio = (kV / 500)^b`` to :func:`voltage_cost_anchors`, returning ``b``.

    Log-log least squares **constrained through the reference voltage**, so
    ``ratio(500 kV) = 1`` exactly: the county table is quoted at 500 kV AC, and a
    free intercept would rescale every line in the network. Worst anchor residual
    is about -10% (230 kV); in exchange every voltage class -- including the
    100/115/161/345/765 kV classes ReEDS does not publish -- is priced by one
    smooth, monotone law.
    """
    log_kv = np.log(anchors.index.to_numpy(dtype=float) / REFERENCE_VOLTAGE_KV)
    log_ratio = np.log(anchors.to_numpy(dtype=float))
    exponent = float((log_kv * log_ratio).sum() / (log_kv * log_kv).sum())

    residuals = np.exp(exponent * log_kv) / anchors.to_numpy(dtype=float) - 1.0
    logger.info(
        "Voltage cost exponent b = %.4f in ratio = (kV/%.0f)^b; anchor residuals %s.",
        exponent,
        REFERENCE_VOLTAGE_KV,
        ", ".join(f"{kv:.0f}kV={res:+.1%}" for kv, res in zip(anchors.index, residuals)),
    )
    return exponent


def voltage_ratio(v_nom, exponent: float) -> np.ndarray:
    """Cost ratio relative to 500 kV AC; missing/non-positive voltages price at 500 kV."""
    kv = np.asarray(v_nom, dtype=float)
    unresolved = ~np.isfinite(kv) | (kv <= 0)
    if unresolved.any():
        logger.warning(
            "%d branch(es) have no usable v_nom; pricing them at %.0f kV.",
            int(unresolved.sum()),
            REFERENCE_VOLTAGE_KV,
        )
        kv = np.where(unresolved, REFERENCE_VOLTAGE_KV, kv)
    return (kv / REFERENCE_VOLTAGE_KV) ** exponent


def load_transmission_pair_costs(path: str) -> pd.DataFrame:
    """Load the ReEDS county-pair 500 kV AC costs as a symmetric long table.

    Returns columns ``r``, ``rr``, ``length_km``, ``unit_cost`` (2004 USD/MW per
    great-circle km). Both orderings of every pair are present.
    """
    df = pd.read_csv(path)
    required = {"r", "rr", "length_miles", "USD2004perMW"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Transmission distance-cost file is missing columns {sorted(missing)}.")

    df["length_km"] = pd.to_numeric(df["length_miles"], errors="coerce") * MILES_TO_KM
    df["unit_cost"] = pd.to_numeric(df["USD2004perMW"], errors="coerce") / df["length_km"]
    df = df[["r", "rr", "length_km", "unit_cost"]].replace([np.inf, -np.inf], np.nan).dropna()

    flipped = df.rename(columns={"r": "rr", "rr": "r"})
    both = pd.concat([df, flipped[df.columns]], ignore_index=True)
    logger.info("Loaded %d county-pair transmission costs over %d counties.", len(df), both["r"].nunique())
    return both


def pair_unit_costs(pair_table: pd.DataFrame) -> pd.Series:
    """``(r, rr) -> unit_cost`` lookup over both orderings of each pair."""
    series = pair_table.set_index(["r", "rr"])["unit_cost"]
    return series[~series.index.duplicated()]


def county_unit_cost_field(pair_table: pd.DataFrame, n_nearest: int = NEAR_NEIGHBOUR_PAIRS) -> pd.Series:
    """Per-county unit cost: median over the ``n_nearest`` shortest incident pairs."""
    nearest = pair_table.sort_values(["r", "length_km"]).groupby("r", sort=False).head(n_nearest)
    field = nearest.groupby("r")["unit_cost"].median()
    logger.info(
        "County unit-cost field over %d counties (%d nearest pairs each): min=%.0f median=%.0f max=%.0f "
        "USD%d/MW/km at %.0f kV AC.",
        len(field),
        n_nearest,
        field.min(),
        field.median(),
        field.max(),
        REEDS_TRANSMISSION_COST_YEAR,
        REFERENCE_VOLTAGE_KV,
    )
    return field


def _state_fips(counties: pd.Series) -> pd.Series:
    """State FIPS from a ReEDS county code (``p`` + 5-digit FIPS)."""
    return counties.astype("string").str.slice(1, 3)


def transmission_unit_costs(
    county0: pd.Series,
    county1: pd.Series,
    pair_costs: pd.Series,
    field: pd.Series,
) -> pd.Series:
    """Per-branch 500 kV AC unit cost in 2004 USD/MW per great-circle km.

    Resolved in four steps, each catching what the previous one cannot:

    1. the exact ``(county0, county1)`` pair (the measured corridor);
    2. the mean of the two counties' :func:`county_unit_cost_field` values -- the
       main path, since most lines have both ends in one county and ReEDS publishes
       no same-county pair;
    3. the median field value of the state (covers Connecticut, where ReEDS uses
       the post-2022 planning regions and the census shapes the legacy FIPS);
    4. the national median (no county at all, e.g. offshore).
    """
    index = county0.index
    c0 = county0.astype("string")
    c1 = county1.astype("string")

    lookup = pd.MultiIndex.from_arrays([c0.fillna(""), c1.fillna("")])
    unit = pd.Series(pair_costs.reindex(lookup).to_numpy(), index=index, dtype=float)
    from_pair = int(unit.notna().sum())

    endpoint_mean = pd.concat(
        [
            pd.Series(field.reindex(c0.to_numpy()).to_numpy(), index=index, dtype=float),
            pd.Series(field.reindex(c1.to_numpy()).to_numpy(), index=index, dtype=float),
        ],
        axis=1,
    ).mean(axis=1)
    unit = unit.fillna(endpoint_mean)
    from_field = int(unit.notna().sum()) - from_pair

    state_field = field.groupby(_state_fips(pd.Series(field.index, index=field.index))).median()
    branch_state = _state_fips(c0).fillna(_state_fips(c1))
    unit = unit.fillna(pd.Series(state_field.reindex(branch_state.to_numpy()).to_numpy(), index=index, dtype=float))
    from_state = int(unit.notna().sum()) - from_pair - from_field

    unit = unit.fillna(float(field.median()))
    from_national = len(unit) - from_pair - from_field - from_state

    logger.info(
        "Transmission unit costs for %d branch(es): %d from the county pair, %d from the endpoint "
        "county field, %d from the state median, %d from the national median.",
        len(unit),
        from_pair,
        from_field,
        from_state,
        from_national,
    )
    return unit


@dataclass
class TransmissionCostModel:
    """ReEDS transmission cost tables reduced to what pricing a branch needs.

    ``to_dollar_year`` converts the 2004 USD of the tables to the model's
    ``costs: dollar_year``. With ``voltage_factor=False`` every AC line is priced
    at the 500 kV reference cost.
    """

    pair_costs: pd.Series
    field: pd.Series
    voltage_exponent: float
    dc_ratio: float
    to_dollar_year: float
    voltage_factor: bool = True

    @classmethod
    def from_files(
        cls,
        distance_cost_fn: str,
        basecost_fn: str,
        dollar_year: int,
        voltage_factor: bool = True,
    ) -> TransmissionCostModel:
        """Build the model from the ReEDS county-pair and voltage-class cost tables."""
        from money import conversion_factor

        basecost = load_transmission_basecost(basecost_fn)
        pair_table = load_transmission_pair_costs(distance_cost_fn)
        return cls(
            pair_costs=pair_unit_costs(pair_table),
            field=county_unit_cost_field(pair_table),
            voltage_exponent=voltage_cost_exponent(voltage_cost_anchors(basecost)),
            dc_ratio=dc_ac_line_cost_ratio(basecost),
            to_dollar_year=conversion_factor(REEDS_TRANSMISSION_COST_YEAR, dollar_year),
            voltage_factor=voltage_factor,
        )

    def _reference_cost(self, county0: pd.Series, county1: pd.Series) -> pd.Series:
        return transmission_unit_costs(county0, county1, self.pair_costs, self.field) * self.to_dollar_year

    def ac_unit_cost(self, county0: pd.Series, county1: pd.Series, v_nom) -> pd.Series:
        """Overnight AC line cost, ``dollar_year`` USD/MW per great-circle km."""
        ratio = voltage_ratio(v_nom, self.voltage_exponent) if self.voltage_factor else 1.0
        return self._reference_cost(county0, county1) * ratio

    def dc_unit_cost(self, county0: pd.Series, county1: pd.Series) -> pd.Series:
        """Overnight 500 kV HVDC bipole line cost (no converters), same units."""
        return self._reference_cost(county0, county1) * self.dc_ratio


def _endpoint_counties(n, branches: pd.DataFrame, bus_county: pd.Series) -> tuple[pd.Series, pd.Series]:
    c0 = bus_county.reindex(branches["bus0"]).set_axis(branches.index)
    c1 = bus_county.reindex(branches["bus1"]).set_axis(branches.index)
    return c0, c1


def assign_branch_capex(n, model: TransmissionCostModel, bus_county: pd.Series | None = None) -> None:
    """Write :data:`BRANCH_CAPEX_COL` on every line and transmission link of ``n``.

    Must run while lines still carry their native ``v_nom`` (i.e. before
    ``aggregate_to_substations`` converts everything to 230 kV). ``bus_county``
    defaults to ``n.buses.county``. Lines without a ``v_nom`` column fall back to
    the voltage of their ``bus0``.
    """
    if bus_county is None:
        if "county" not in n.buses.columns:
            raise ValueError("n.buses has no 'county' column; pass bus_county explicitly.")
        bus_county = n.buses["county"]
    bus_county = bus_county.where(bus_county.astype("string").str.startswith("p"))

    if not n.lines.empty:
        v_nom = n.lines["v_nom"] if "v_nom" in n.lines.columns else n.lines.bus0.map(n.buses.v_nom)
        c0, c1 = _endpoint_counties(n, n.lines, bus_county)
        n.lines[BRANCH_CAPEX_COL] = model.ac_unit_cost(c0, c1, v_nom).to_numpy()
        logger.info(
            "Overnight AC line cost (USD/MW per great-circle km) by voltage: %s",
            n.lines.groupby(v_nom)[BRANCH_CAPEX_COL].median().round(0).to_dict(),
        )

    if not n.links.empty:
        n.links[BRANCH_CAPEX_COL] = np.nan
        transmission = n.links.index[n.links.carrier.isin(TRANSMISSION_LINK_CARRIERS)]
        if len(transmission):
            links = n.links.loc[transmission]
            c0, c1 = _endpoint_counties(n, links, bus_county)
            ac_cost = model.ac_unit_cost(c0, c1, REFERENCE_VOLTAGE_KV)
            dc_cost = model.dc_unit_cost(c0, c1)
            n.links.loc[transmission, BRANCH_CAPEX_COL] = dc_cost.where(links.carrier == "DC", ac_cost).to_numpy()


def great_circle_km(n, branches: pd.DataFrame) -> pd.Series:
    """Great-circle distance between the current coordinates of each branch's buses."""
    from pypsa.geo import haversine_pts

    coords = n.buses[["x", "y"]].astype(float)
    km = haversine_pts(
        coords.loc[branches.bus0].to_numpy(),
        coords.loc[branches.bus1].to_numpy(),
    )
    return pd.Series(km, index=branches.index, dtype=float)


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
    to 1.0. An empty or null ``column_to_tech`` gives a table without carriers.
    """
    out = {}
    for column, carriers in (column_to_tech or {}).items():
        if column not in reg_cap_diff.columns:
            logger.warning("reg_cap_cost_diff column %r not found; skipping.", column)
            continue
        mult = 1.0 + reg_cap_diff[column].fillna(0.0)
        for carrier in carriers:
            out[carrier] = mult
    table = pd.DataFrame(out, index=reg_cap_diff.index)
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
    """Bus multiplier table for the AC buses of ``n``, or ``None`` when disabled.

    Also ``None`` when ``column_to_tech`` maps no carrier, which is how the ReEDS
    wind/solar interconnection cost is used without the county multipliers.
    """
    cfg = cfg or {}
    if not cfg.get("enable", False):
        return None
    if not cfg.get("column_to_tech"):
        logger.info("costs.regional_multipliers.column_to_tech is empty; no county overnight-capex multipliers.")
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


def regional_capital_cost(costs: pd.DataFrame, carrier: str, buses, bus_multipliers, own_multiplier=None):
    """Per-bus ``capital_cost`` for an ATB carrier with the regional overnight multiplier.

    ``buses`` are the bus names the components sit at (one per component, repeats
    allowed). Returns the uniform ``annualized_capex_fom`` whenever the feature is
    off, the carrier is not mapped, or the cost table has no overnight capex. The
    overnight delta is annualized at the carrier's capital charge rate, like the
    rest of its capex.

    ``own_multiplier`` (one value per component) is the multiplier a component
    already carries (``constants.REGIONAL_MULTIPLIER_COL``). Where it is set it
    replaces the multiplier of the bus, so a generator that clustering merged is
    priced with the capacity-weighted mean of its members and not with the county
    of the clustered bus.
    """
    base = costs.at[carrier, "annualized_capex_fom"]
    multiplier = carrier_multiplier(bus_multipliers, carrier, pd.Index(buses))
    if np.isscalar(multiplier):
        return base
    if own_multiplier is not None:
        own = pd.to_numeric(pd.Series(own_multiplier), errors="coerce").to_numpy(dtype=float)
        multiplier = np.where(own > 0, own, multiplier)
    overnight = costs.at[carrier, "capex_overnight_per_kw"] if "capex_overnight_per_kw" in costs.columns else np.nan
    if not np.isfinite(overnight) or overnight == 0:
        logger.warning("No usable capex_overnight_per_kw for %r; skipping its regional multiplier.", carrier)
        return base
    return overnight_delta_capital_cost(base, overnight, carrier_capital_charge_rate(costs, carrier), multiplier)


def renewable_interconnection_enabled(cfg: dict | None) -> bool:
    """True when ``costs: regional_multipliers`` switches the ReEDS wind/solar interconnection cost on."""
    cfg = cfg or {}
    return bool(cfg.get("enable", False) and cfg.get("renewable_interconnection", False))


def renewable_interconnection_capital_cost(
    costs: pd.DataFrame,
    carrier: str,
    interconnection_capex,
    share=None,
) -> pd.Series:
    """Wind/solar ``capital_cost`` with the ATB grid connection replaced by the ReEDS cost.

    ``capital_cost = annualized_capex_fom + annuity * (interconnection - share * capex_grid_connection_per_kw * 1e3)``

    ``annualized_capex_fom`` is ``rate * (overnight + grid connection +
    construction finance) * 1e3 + FOM * 1e3`` with ``rate`` the carrier's capital
    charge rate, so this swaps only the ATB's generic
    grid connection for the site-resolved spur/POI/reinforcement cost; overnight
    capex, construction finance and FOM are untouched. ``interconnection_capex`` is
    overnight USD/MW of the same dollar year as ``costs`` (a Series, one value per
    component). A missing or non-positive value keeps the uniform ATB cost.

    ``share`` (one value per component, 1 where not given) is the part of the
    capacity the interconnection cost covers. It is below 1 on a generator that
    clustering merged from members with and without a ReEDS site; there
    ``interconnection_capex`` is the mean over the whole capacity with the members
    without a site counted as zero, and only the covered share gives up the ATB
    grid connection. That is the capacity-weighted mean of the members' costs.
    """
    interconnection = pd.Series(interconnection_capex, dtype=float)
    share = pd.Series(1.0 if share is None else share, index=interconnection.index, dtype=float).fillna(1.0)
    base = float(costs.at[carrier, "annualized_capex_fom"])
    grid_connection = (
        costs.at[carrier, "capex_grid_connection_per_kw"] if "capex_grid_connection_per_kw" in costs else 0.0
    )
    grid_connection = 0.0 if pd.isna(grid_connection) else float(grid_connection)
    rate = carrier_capital_charge_rate(costs, carrier)
    priced = base + rate * (interconnection - share * grid_connection * 1e3)
    return priced.where(interconnection > 0, base)
