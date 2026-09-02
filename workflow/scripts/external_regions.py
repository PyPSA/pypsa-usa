"""External (out-of-footprint) regions: imports, exports and contracted units.

A regionally scoped run (California, say) cuts the synchronous grid at a
political boundary. Everything behind that cut still serves load inside the
footprint, and this module is the single place where it is represented. Two
representations are available, selected by ``electricity.imports.representation``:

``store`` (default, unchanged behaviour)
    Per external flowgate zone, a ``{zone}_imports`` bus carrying a bottomless
    ``Store`` (``e_nom_max`` 1e9, ``e_min_pu`` -1) and one-way ``Link`` s into
    the internal zone buses. Each link is rated at the NARIS flowgate capacity
    and PRICED at the import price (a ``wholesale`` EIA timeseries, a carrier's
    average marginal cost, or a flat float). Emissions are carried by the
    ``imports`` carrier that the Store belongs to. Exports mirror this with an
    absorbing Store and negatively-priced links.

``generator``
    The external zone is modelled as a place with generation rather than as an
    infinite energy tank:

    * the ``{zone}_imports`` bus carries a generic import ``Generator``
      (carrier ``unspecified_imports``) sized at the zone's total inbound
      interface capacity and priced with the same machinery as ``store`` mode;
    * California's CPUC-contracted out-of-state units (Palo Verde, Intermountain,
      Hoover, Apex, the AZ/NV solar and battery contracts) are attached at that
      same external bus instead of at a California bus, i.e. BEHIND the boundary;
    * the ``Link`` s into the footprint are UNPRICED and rated at the flowgate
      capacity, so the price sits on the generator and the link is pure transfer
      capacity.

    The point of the second mode is accounting: because deliveries now traverse
    a carrier-``imports`` link, they are counted by
    :func:`opts.interchange.add_interchange_constraints` against the import
    volume cap, and bounded by the interface capacity, exactly like generic
    imports. In ``store`` mode the contracted units sit inside the footprint and
    look like in-state generation, bypassing both limits.

Both modes keep the link carriers ``imports`` / ``exports`` untouched, which is
what ``opts/interchange.py`` and ``opts/interfaces.py`` key off.

External-bus naming
-------------------
Both modes use SEPARATE ``{zone}_imports`` and ``{zone}_exports`` buses rather
than one shared ``{zone}_external`` bus. Beyond keeping ``store`` mode
byte-identical, the separation is load-bearing in ``generator`` mode: a shared
bus would let the priced import generator (and the contracted units) sell
straight into the export sink, collecting the export price for free whenever the
export price exceeds the import cost — a pure arbitrage loop with no physical
meaning.

Export pricing
--------------
Export pricing is IDENTICAL in both modes: the negative price stays on the
export ``Link`` and the absorbing ``Store`` behind it is free. On the export
side there is no double-charge to worry about, because the only way energy can
reach the export Store is through exactly one export link — nothing else injects
into a ``{zone}_exports`` bus — so the negative price is earned exactly once per
exported MWh. That is what makes it safe to leave the export half of the
construction untouched by the representation switch, and it is only safe because
the import generator lives on a different bus (see above).

CO2
---
In ``store`` mode the ``imports`` carrier carries ``imports.co2_emissions`` and
the Store's withdrawal is what the global CO2 constraint sees. In ``generator``
mode there is no import Store, so the emission factor moves onto the
``unspecified_imports`` carrier of the generic import generator and ``imports``
is set to zero. PyPSA attributes primary-energy emissions to generators and
stores, never to links, so the carrier-``imports`` links never double-count.
"""

import logging
import re

import dill
import pandas as pd
import pypsa
from add_electricity import add_missing_carriers, attach_remote_units
from eia import FuelCosts

logger = logging.getLogger(__name__)

REPRESENTATIONS = ("store", "generator")

#: Carrier of the generic import generator in ``generator`` mode. Deliberately
#: NOT ``imports``: that carrier is reserved for the transfer links which
#: ``opts/interchange.py`` sums over.
GENERIC_IMPORT_CARRIER = "unspecified_imports"


# ---------------------------------------------------------------------------
# Flowgate formatting
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Import / export pricing
# ---------------------------------------------------------------------------


def calc_import_export_costs(n: pypsa.Network, carrier: str) -> float:
    """Calculates the average marginal cost for a given carrier."""
    gens = n.generators[n.generators.carrier == carrier]
    component = "Generator"
    if gens.empty:
        gens = n.links[n.links.carrier == carrier]
        component = "Link"
    if gens.empty:
        raise ValueError(f"No generators or links found for carrier to calculate imports/exports costs: {carrier}")
    costs = n.get_switchable_as_dense(component, "marginal_cost").loc[:, gens.index].mean().mean()
    if costs <= 0.01:
        raise ValueError(
            f"Average marginal cost for {carrier} is less than or equal to 0.01. Check the fuel costs configuration.",
        )
    return costs


def load_import_export_costs(eia_api: str, year: int) -> pd.DataFrame:
    """Loads fuel costs from EIA."""
    # EIA retail-sales electricity prices begin 2001; earlier years return an
    # empty payload that crashes format_data. No date shifting is needed: the
    # downstream _build_cost_timeseries relabels the index onto the network's
    # investment periods anyway.
    data_year = max(year, 2001)
    if data_year != year:
        logger.warning(
            f"No EIA electricity prices before 2001; using {data_year} prices for year {year}",
        )
    return FuelCosts(fuel="electricity", year=data_year, api=eia_api).get_data()


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
    formatted = pd.DataFrame(data, columns=["period", "zone", "value", "units"]).set_index("period")
    return formatted[~formatted.value.isna()]  # regions outside of model scope


def resolve_trade_costs(
    n: pypsa.Network,
    trade_config: dict,
    direction: str,
    eia_api: str,
    year: int,
) -> pd.DataFrame | float:
    """Resolve ``imports.costs`` / ``exports.costs`` into a price the network can use.

    ``wholesale`` pulls the monthly EIA electricity price per state, a carrier
    name averages that carrier's marginal cost, and a float is taken at face
    value. Export prices are negated: exporting earns money.
    """
    if direction not in ("imports", "exports"):
        raise ValueError(f"direction must be either imports or exports; received: {direction}")

    sign = 1 if direction == "imports" else -1
    costs = trade_config.get("costs", False)

    if isinstance(costs, float | int):  # user defined value
        return costs * sign
    if isinstance(costs, str):  # 'wholesale' or name of carrier
        if costs == "wholesale":
            fuel_costs = load_import_export_costs(eia_api, year)
            fuel_costs = format_import_export_costs(n, fuel_costs)
            if sign < 0:
                fuel_costs["value"] = fuel_costs.value.mul(sign)  # make money by exporting
            return fuel_costs
        return calc_import_export_costs(n, costs) * sign
    raise ValueError(
        f"'{direction}.costs' must be 'wholesale', name of a carrier, or a float/int. Received: {costs}",
    )


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


# ---------------------------------------------------------------------------
# Shared construction helpers
# ---------------------------------------------------------------------------


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

    n.add(
        "Bus",
        regions_2_add,
        suffix=suffix,
        carrier=carrier,
        country=regions_2_add,
    )


def _add_import_export_stores(n: pypsa.Network, regions_2_add: list[str], direction: str) -> None:
    """Adds import and export stores to the network."""
    if direction == "imports":
        n.add(
            "Store",
            regions_2_add,
            bus=[f"{x}_imports" for x in regions_2_add],
            suffix="_imports",
            carrier="imports",
            e_nom=0,
            e_nom_extendable=True,
            capital_cost=0,
            e_nom_min=0,
            e_nom_max=1e9,
            e_min_pu=-1,
            e_max_pu=0,
            e_cyclic_per_period=False,
            marginal_cost=0,
        )
    elif direction == "exports":
        n.add(
            "Store",
            regions_2_add,
            bus=[f"{x}_exports" for x in regions_2_add],
            suffix="_exports",
            carrier="exports",
            e_nom_extendable=True,
            marginal_cost=0,
            e_nom=0,
            e_nom_max=1e9,
            e_min=0,
            e_min_pu=0,
            e_max_pu=1,
        )
    else:
        raise ValueError(f"direction must be either imports or exports; received: {direction}")


def _add_import_export_links(
    n: pypsa.Network,
    flowgates: pd.DataFrame,
    fuel_costs: pd.DataFrame | float | str,
    direction: str,
    zone_col: str = "reeds_zone",
    priced: bool = True,
) -> None:
    """Adds import and export links to the network.

    ``priced`` is what distinguishes the two representations on the import side:
    in ``store`` mode the link carries the import price, in ``generator`` mode
    the price sits on the external generator instead and the link is a pure
    transfer capacity.
    """
    costs = {}
    zones_in_model = n.buses[zone_col].dropna().unique()

    for _, row in flowgates.iterrows():
        zone_inside = row.r if row.r in zones_in_model else row.rr
        zone_outside = row.r if row.r not in zones_in_model else row.rr

        # extremely crude caching for generating cost timeseries :|
        # keyed by the INSIDE zone — checking the outside zone here skipped
        # the write whenever an earlier row's inside zone happened to match,
        # leaving costs[zone_inside] unset (KeyError on county networks).
        if zone_inside not in costs:
            if isinstance(fuel_costs, float | int):
                costs[zone_inside] = fuel_costs
            elif isinstance(fuel_costs, pd.DataFrame):
                costs[zone_inside] = _build_cost_timeseries(n, fuel_costs, zone_inside)
            else:
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
            if not priced:
                marginal_cost = 0
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


# ---------------------------------------------------------------------------
# `store` representation
# ---------------------------------------------------------------------------


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
    assert direction in ["imports", "exports"], f"direction must be either imports or exports; received: {direction}"

    regions_2_add = _get_regions_2_add(n, flowgates, zone_col)
    _add_import_export_carriers(n, direction, co2_emissions)
    _add_import_export_buses(n, regions_2_add, direction)
    _add_import_export_stores(n, regions_2_add, direction)
    _add_import_export_links(n, flowgates, fuel_costs, direction, zone_col)


# ---------------------------------------------------------------------------
# `generator` representation
# ---------------------------------------------------------------------------


def inbound_capacity_by_zone(
    n: pypsa.Network,
    flowgates: pd.DataFrame,
    zone_col: str = "reeds_zone",
) -> pd.Series:
    """Total interface capacity flowing INTO the footprint, per external zone."""
    zones_in_model = n.buses[zone_col].dropna().unique()
    inbound = flowgates[~flowgates.r.isin(zones_in_model) & flowgates.rr.isin(zones_in_model)]
    return inbound.groupby("r")["value"].sum()


def _internal_zones_served(
    n: pypsa.Network,
    flowgates: pd.DataFrame,
    zone_col: str,
) -> dict[str, list[str]]:
    """Map each external zone onto the internal zones it can deliver into."""
    zones_in_model = n.buses[zone_col].dropna().unique()
    inbound = flowgates[~flowgates.r.isin(zones_in_model) & flowgates.rr.isin(zones_in_model)]
    return {zone: sorted(set(rows.rr)) for zone, rows in inbound.groupby("r")}


def _external_generator_cost(
    n: pypsa.Network,
    fuel_costs: pd.DataFrame | float,
    internal_zones: list[str],
) -> pd.Series | float:
    """Price for the generic import generator of one external zone.

    The wholesale price table is keyed by the zones INSIDE the model (it is built
    from bus attributes), so an external zone has no price of its own. Its
    generator is priced at the mean of the internal zones it can reach, which is
    the same set of prices the ``store``-mode links would have carried.
    """
    if isinstance(fuel_costs, float | int):
        return fuel_costs
    if not isinstance(fuel_costs, pd.DataFrame):
        return 0
    series = [_build_cost_timeseries(n, fuel_costs, zone)["value"] for zone in internal_zones]
    series = [s for s in series if not s.isna().all()]
    if not series:
        return 0
    return pd.concat(series, axis=1).mean(axis=1)


def _add_generic_import_generators(
    n: pypsa.Network,
    regions_2_add: list[str],
    flowgates: pd.DataFrame,
    fuel_costs: pd.DataFrame | float,
    co2_emissions: float,
    zone_col: str,
) -> None:
    """One priced, non-extendable import generator per external bus."""
    n.add("Carrier", GENERIC_IMPORT_CARRIER, co2_emissions=co2_emissions, nice_name="Unspecified Imports")

    capacity = inbound_capacity_by_zone(n, flowgates, zone_col)
    served = _internal_zones_served(n, flowgates, zone_col)

    for zone in regions_2_add:
        p_nom = float(capacity.get(zone, 0.0))
        if p_nom <= 0:
            logger.info(f"External zone '{zone}' has no inbound interface capacity; no import generator added.")
            continue
        marginal_cost = _external_generator_cost(n, fuel_costs, served.get(zone, []))
        n.add(
            "Generator",
            f"{zone}_imports {GENERIC_IMPORT_CARRIER}",
            bus=f"{zone}_imports",
            carrier=GENERIC_IMPORT_CARRIER,
            p_nom=p_nom,
            p_nom_extendable=False,
            efficiency=1,
            marginal_cost=marginal_cost,
        )


def map_remote_units_to_zones(
    unit_states: pd.Series,
    external_zones: list[str],
    inbound_capacity: pd.Series,
    zone_col: str,
    membership: pd.DataFrame | None = None,
) -> pd.Series:
    """Assign each contracted remote unit to the external zone it sits behind.

    Candidate zones are the boundary zones whose state matches the unit's
    physical state (from ``powerplants.csv``); the candidate with the largest
    inbound interface capacity wins. A unit whose state has no direct interface
    with the footprint (Utah's Intermountain, say) falls back to the boundary
    zone with the largest inbound capacity overall, with a warning.
    """
    ranked = [z for z in external_zones if inbound_capacity.get(z, 0.0) > 0]
    ranked = sorted(ranked, key=lambda z: (-float(inbound_capacity.get(z, 0.0)), z))
    if not ranked:
        raise ValueError("No external zone with inbound interface capacity; cannot place remote contracted units.")

    zone_state = _external_zone_states(ranked, zone_col, membership)
    fallback = ranked[0]

    assignment = {}
    for unit, state in unit_states.items():
        candidates = [z for z in ranked if zone_state.get(z) == state]
        if candidates:
            assignment[unit] = candidates[0]
        else:
            assignment[unit] = fallback
            logger.warning(
                f"Remote contracted unit '{unit}' sits in state '{state}', which has no direct interface with the "
                f"model footprint; placing it behind external zone '{fallback}' (largest inbound capacity).",
            )
    return pd.Series(assignment, dtype=object)


# County zones are "p" + 5-digit county FIPS; the first two digits are the state.
_STATE_BY_FIPS = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}

_COUNTY_ZONE_RE = re.compile(r"^p?(\d{2})\d{3}$")


def _external_zone_states(
    external_zones: list[str],
    zone_col: str,
    membership: pd.DataFrame | None,
) -> dict[str, str]:
    """State code of each external zone.

    With ``topological_boundaries: state`` the zone IS the state code; county
    zones ("p" + county FIPS) resolve through the state FIPS prefix; ReEDS
    balancing areas resolve through the membership table.
    """
    if zone_col == "reeds_state":
        return {zone: zone for zone in external_zones}
    mbshp = membership.set_index("ba")["st"] if membership is not None else pd.Series(dtype=object)

    def state_of(zone: str) -> str | None:
        m = _COUNTY_ZONE_RE.match(zone)
        if m and zone not in mbshp.index:
            return _STATE_BY_FIPS.get(m.group(1))
        return mbshp.get(zone)

    states = {zone: state_of(zone) for zone in external_zones}
    if membership is None and not all(states.values()):
        logger.warning("No ReEDS membership table supplied; cannot map external zones onto states.")
    return states


def attach_remote_contracted_units_externally(
    n: pypsa.Network,
    bundle: dict,
    flowgates: pd.DataFrame,
    zone_col: str,
    membership: pd.DataFrame | None,
) -> pd.Series:
    """Attach the serialized CPUC contracted units at their external buses.

    ``bundle`` is the ``add_electricity`` output described in
    :func:`add_electricity.build_remote_unit_bundle`: the fully-derived unit
    table (still keyed to the California bus whose VRE profile it borrowed) plus
    the borrowed profiles and the cost table needed to resolve capital costs.
    Only the ``bus`` column is rewritten here.
    """
    units = bundle["units"].copy()
    if units.empty:
        return pd.Series(dtype=object)

    external_zones = [b[: -len("_imports")] for b in n.buses.index[n.buses.index.str.endswith("_imports")]]
    inbound = inbound_capacity_by_zone(n, flowgates, zone_col)
    zones = map_remote_units_to_zones(units["state"], external_zones, inbound, zone_col, membership)

    units["zone"] = zones
    units["bus"] = zones.map(lambda z: f"{z}_imports")

    for name, unit in units.iterrows():
        logger.info(
            f"Remote contracted unit '{name}' ({unit['carrier']}, {unit['p_nom']:.1f} MW, state {unit['state']}) "
            f"attached behind external zone '{unit['zone']}' at bus '{unit['bus']}'.",
        )

    add_missing_carriers(n, sorted(set(units["carrier"])))
    attach_remote_units(
        n,
        units.drop(columns=["zone"]),
        bundle["costs"],
        bundle["conventional_carriers"],
        bundle["unit_commitment"],
        profiles=bundle.get("vre_profiles"),
    )
    return units["zone"]


def _add_generator_representation(
    n: pypsa.Network,
    flowgates: pd.DataFrame,
    fuel_costs: pd.DataFrame | float,
    co2_emissions: float,
    zone_col: str,
    remote_bundle: dict | None,
    membership: pd.DataFrame | None,
) -> None:
    """Import side of the ``generator`` representation (exports are unchanged)."""
    regions_2_add = _get_regions_2_add(n, flowgates, zone_col)

    # emissions move onto the generic import generator's carrier; the `imports`
    # carrier now only labels links, which PyPSA never charges emissions to.
    _add_import_export_carriers(n, "imports", 0)
    _add_import_export_buses(n, regions_2_add, "imports")
    _add_generic_import_generators(n, regions_2_add, flowgates, fuel_costs, co2_emissions, zone_col)
    _add_import_export_links(n, flowgates, fuel_costs, "imports", zone_col, priced=False)

    if remote_bundle:
        attach_remote_contracted_units_externally(n, remote_bundle, flowgates, zone_col, membership)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def add_external_regions(
    n: pypsa.Network,
    direction: str,
    representation: str,
    flowgates: pd.DataFrame,
    fuel_costs: pd.DataFrame | float,
    co2_emissions: float = 0,
    zone_col: str = "reeds_zone",
    remote_bundle: dict | None = None,
    membership: pd.DataFrame | None = None,
) -> None:
    """Add the external-region representation for one direction of trade.

    Parameters
    ----------
    direction
        ``imports`` or ``exports``.
    representation
        ``store`` or ``generator`` — see the module docstring.
    flowgates
        Formatted NARIS flowgates (columns ``r``, ``rr``, ``value``), already
        restricted to interfaces that cross the model boundary.
    fuel_costs
        Output of :func:`resolve_trade_costs`.
    remote_bundle, membership
        Only used for ``direction='imports'`` in ``generator`` mode: the CPUC
        contracted-unit bundle written by ``add_electricity`` and the ReEDS
        ``membership.csv`` used to map external zones onto states.
    """
    if direction not in ("imports", "exports"):
        raise ValueError(f"direction must be either imports or exports; received: {direction}")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}; received: {representation}")

    if representation == "store" or direction == "exports":
        add_elec_imports_exports(n, direction, flowgates, fuel_costs, co2_emissions, zone_col)
        return

    _add_generator_representation(n, flowgates, fuel_costs, co2_emissions, zone_col, remote_bundle, membership)


def load_remote_unit_bundle(path: str | None) -> dict | None:
    """Read the contracted-unit bundle written by ``add_electricity``.

    The rule always declares the file, so a run with contracted resources
    disabled (or in ``store`` mode) writes a sentinel ``None``; that is not an
    error, it just means there is nothing to place behind the boundary.
    """
    if not path:
        return None
    with open(path, "rb") as f:
        bundle = dill.load(f)
    if not bundle or bundle.get("units") is None or bundle["units"].empty:
        return None
    return bundle


__all__ = [
    "add_elec_imports_exports",
    "add_external_regions",
    "calc_import_export_costs",
    "convert_flowgates_to_state",
    "format_flowgates_for_imports_exports",
    "format_import_export_costs",
    "inbound_capacity_by_zone",
    "load_import_export_costs",
    "load_remote_unit_bundle",
    "map_remote_units_to_zones",
    "resolve_trade_costs",
]
