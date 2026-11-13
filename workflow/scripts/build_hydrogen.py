"""
Module for building the hydrogen supply chain of sector studies.

Hydrogen is balanced at state level, like natural gas: every state in the model
gets a ``{state} H2`` bus carrying the exogenous, non-power hydrogen demand. It
is produced from electricity at every AC bus (electrolysis) or from the state
gas bus (steam methane reforming, with or without carbon capture); stored
underground in the state; and piped between states joined by AC transmission
or a natural gas pipeline. With the natural gas network, gas storage reservoirs
and pipelines can also be retrofitted to hydrogen. The turbines burning it (OCHT,
CCHT) are converted from generators to links in ``add_sectors``.

New capacity gets one vintage per investment period. Costs and performance data
are documented in ``repo_data/costs/hydrogen_costs.csv``.
"""

import logging
import sqlite3
from contextlib import closing

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import shapely
from constants import CODE_2_STATE, HOURS_PER_YEAR, STATE_2_CODE, STATES_INTERCONNECT_MAPPER

logger = logging.getLogger(__name__)

# A storage type is available to a state when its sites cover at least this share
# of the state's area (ReEDS inputs/scalars.csv h2_storage_area_threshold)
H2_STORAGE_AREA_THRESHOLD = 0.01

# Geological storage types of h2_storage_sites.gpkg, the cheapest first. A state
# with neither builds the more expensive storage in underground pipes ("pipe").
H2_STORAGE_SITES = ("salt", "hardrock")

# Equal area projection for the overlap of states and storage sites
EQUAL_AREA_CRS = "EPSG:5070"

# Natural gas storage formations (EIA-191 field types) that can be retrofitted to
# hydrogen, with the suffix of their costs
RETROFITTABLE_FORMATIONS = {"salt-dome": "salt", "depleted-field": "depleted", "aquifer": "aquifer"}


def build_hydrogen(
    n: pypsa.Network,
    costs: dict[int, pd.DataFrame],
    demand_path: str,
    storage_sites_path: str,
    gas_pipelines: pd.DataFrame,
    states_geometry: gpd.GeoDataFrame,
    center_points: pd.DataFrame,
    options: dict,
) -> None:
    """
    Adds the hydrogen supply chain to the network.

    Arguments
    ---------
    n: pypsa.Network
        Network with state assigned AC buses ('STATE') and state gas buses
    costs: dict[int, pd.DataFrame]
        Technology costs for every investment period
    demand_path: str
        ``repo_data/hydrogen/adp2024_h2_demand.csv``
    storage_sites_path: str
        ``repo_data/hydrogen/h2_storage_sites.gpkg``
    gas_pipelines: pd.DataFrame
        State to state natural gas pipelines ('STATE_FROM', 'STATE_TO'), see
        ``get_gas_pipelines``
    states_geometry: gpd.GeoDataFrame
        State boundaries with a 'STATE' column
    center_points: pd.DataFrame
        State center coordinates ('x', 'y') indexed by state
    options: dict
        ``sector: hydrogen`` config
    """
    states = get_hydrogen_states(n)
    production = options.get("production", [])

    add_hydrogen_buses(n, states, center_points)
    add_hydrogen_demand(n, demand_path, options["demand_scenario"], states)

    if options.get("storage", False):
        storage_types = get_storage_types(states_geometry, storage_sites_path, states)
        add_hydrogen_storage_buses(n, storage_types.index, center_points)

    if options.get("pipelines", False):
        routes = get_pipeline_routes(n, states, gas_pipelines)
        length_factor = options.get("pipeline_length_factor", 1.25)

    for year in n.investment_periods:
        year_costs = costs[year]

        if "h2_electrolysis" in production:
            add_electrolysis(n, year_costs, year)
        for carrier in ("h2_smr", "h2_smr_ccs"):
            if carrier in production:
                add_steam_methane_reforming(n, year_costs, year, carrier)

        if options.get("storage", False):
            add_hydrogen_storage(n, year_costs, year, storage_types)
        if options.get("pipelines", False):
            add_hydrogen_pipelines(n, year_costs, year, routes, length_factor)

    # the natural gas infrastructure is sized once for all investment periods, so
    # it is retrofitted once, in the first
    retrofit = options.get("retrofit", {})
    first_year = n.investment_periods[0]
    if retrofit.get("storage", False):
        add_retrofitted_storage(n, costs[first_year], first_year, center_points)
    if retrofit.get("pipelines", False):
        add_retrofitted_pipelines(n, costs[first_year], first_year)


def get_hydrogen_states(n: pypsa.Network) -> list[str]:
    """States of the AC buses."""
    return sorted(n.buses.loc[n.buses.carrier == "AC", "STATE"].dropna().unique())


def _ac_buses(n: pypsa.Network) -> pd.DataFrame:
    return n.buses[(n.buses.carrier == "AC") & n.buses.STATE.notna()]


def _state_bus_attrs(states: list[str], center_points: pd.DataFrame) -> dict:
    points = center_points.reindex(states).fillna(0)
    return {
        "x": points.x.values,
        "y": points.y.values,
        "unit": "MWh_th",
        "interconnect": [STATES_INTERCONNECT_MAPPER.get(x) for x in states],
        "country": states,  # for consistency with the other state level buses
        "STATE": states,
        "STATE_NAME": [CODE_2_STATE.get(x) for x in states],
    }


def add_hydrogen_buses(n: pypsa.Network, states: list[str], center_points: pd.DataFrame) -> None:
    """Adds a hydrogen bus for every state."""
    names = [f"{x} H2" for x in states]
    n.add("Bus", names, carrier="H2", **_state_bus_attrs(states, center_points))


def add_hydrogen_demand(n: pypsa.Network, demand_path: str, scenario: str, states: list[str]) -> None:
    """
    Adds the exogenous, non-power hydrogen demand of each state.

    The annual ADP demand is interpolated linearly between the ADP years (and
    held constant outside them) and spread evenly over the hours of the year.
    """
    df = pd.read_csv(demand_path)
    df = df[df.scenario == scenario]
    if df.empty:
        raise ValueError(f"No hydrogen demand for scenario '{scenario}' in {demand_path}")

    annual = df.pivot(index="year", columns="state", values="demand_mwh").sort_index()

    missing = [x for x in states if x not in annual.columns]
    if missing:
        logger.warning(f"No hydrogen demand for {missing}; setting it to zero")
    annual = annual.reindex(columns=states, fill_value=0)

    names = [f"{x} H2" for x in states]
    p_set = pd.DataFrame(index=n.snapshots, columns=names, dtype=float)
    for period in n.investment_periods:
        if not annual.index.min() <= period <= annual.index.max():
            logger.warning(
                f"Investment period {period} outside of the ADP years {annual.index.min()}-{annual.index.max()}; "
                "holding the nearest year's hydrogen demand constant",
            )
        demand = annual.apply(lambda x: np.interp(period, annual.index, x))
        p_set.loc[period] = (demand / HOURS_PER_YEAR).values

    n.add("Load", names, bus=names, carrier="H2", p_set=p_set)


def add_electrolysis(n: pypsa.Network, costs: pd.DataFrame, year: int) -> None:
    """Adds electrolyzers drawing from every AC bus, rated on electricity input."""
    carrier = "h2_electrolysis"
    buses = _ac_buses(n)
    n.add(
        "Link",
        buses.index,
        suffix=f" {carrier}_{year}",
        bus0=buses.index,
        bus1=buses.STATE + " H2",
        carrier=carrier,
        efficiency=costs.at[carrier, "efficiency"],
        capital_cost=costs.at[carrier, "annualized_capex_fom"],
        p_nom_extendable=True,
        lifetime=costs.at[carrier, "lifetime"],
        build_year=year,
    )


def add_steam_methane_reforming(n: pypsa.Network, costs: pd.DataFrame, year: int, carrier: str) -> None:
    """
    Adds steam methane reformers at every AC bus, rated on gas input.

    Reformers draw gas from the state gas bus (port 0) and their electricity from
    the AC bus (port 4). The CO2 they emit is tracked on the state 'h2-co2' bus
    (port 2); the CO2 they capture goes to the bus' CO2 storage if it is modelled
    (port 5). Port 3 is left to the methane leakage of the gas they draw, like
    the other gas users (``build_emission_tracking.build_ch4_tracking``).
    """
    buses = _ac_buses(n)
    efficiency = 1 / costs.at[carrier, "gas-input"]  # MWh_H2 / MWh_gas

    n.add(
        "Link",
        buses.index,
        suffix=f" {carrier}_{year}",
        bus0=(buses.STATE + " gas").values,
        bus1=(buses.STATE + " H2").values,
        bus2=(buses.STATE + " h2-co2").values,
        bus4=buses.index,
        carrier=carrier,
        efficiency=efficiency,
        efficiency2=costs.at[carrier, "co2-emitted"] * efficiency,
        efficiency4=-costs.at[carrier, "electricity-input"] * efficiency,
        marginal_cost=costs.at[carrier, "variable-om"] * efficiency,
        capital_cost=costs.at[carrier, "annualized_capex_fom"] * efficiency,
        p_nom_extendable=True,
        lifetime=costs.at[carrier, "lifetime"],
        build_year=year,
        **_captured_co2(n, buses.index, costs.at[carrier, "co2-captured"] * efficiency, port=5),
    )


def _captured_co2(n: pypsa.Network, buses: pd.Index, captured: float, port: int) -> dict:
    """
    Link port sending the captured CO2 to the '{bus} co2 capture' bus of the CO2 storage.

    Empty if CO2 storage is not modelled (``co2: storage``); links at buses
    without CO2 storage leave the port unconnected.
    """
    capture_buses = buses + " co2 capture"
    if not captured or not capture_buses.isin(n.buses.index).any():
        return {}
    return {
        f"bus{port}": np.where(capture_buses.isin(n.buses.index), capture_buses, ""),
        f"efficiency{port}": captured,
    }


def get_storage_types(states_geometry: gpd.GeoDataFrame, storage_sites_path: str, states: list[str]) -> pd.Series:
    """
    Underground hydrogen storage type of each state, following ReEDS.

    ReEDS (``reeds/input_processing/h2_storage.py``) lets each region build only
    the cheapest storage available to it: salt caverns, else hard rock caverns
    where at least 1% of its area overlaps the respective sites (Lord et al.,
    2014), else underground pipes.
    """
    regions = states_geometry.set_index("STATE").loc[states].to_crs(EQUAL_AREA_CRS)
    regions_area = regions.area

    storage_type = pd.Series("pipe", index=regions.index, name="storage_type")
    for site in reversed(H2_STORAGE_SITES):  # the cheaper type overrides
        sites = read_geopackage(storage_sites_path, layer=site).to_crs(EQUAL_AREA_CRS)
        sites = sites.geometry.make_valid().union_all()
        overlap = regions.geometry.make_valid().intersection(sites).area / regions_area
        storage_type[overlap >= H2_STORAGE_AREA_THRESHOLD] = site

    logger.info(f"Hydrogen storage types: {storage_type.value_counts().to_dict()}")
    return storage_type


def read_geopackage(path: str, layer: str) -> gpd.GeoDataFrame:
    """
    Geometries of a GeoPackage layer, read with sqlite3 rather than GDAL.

    A GeoPackage is a SQLite database whose geometries are WKB behind a short
    header. Read through GDAL (``gpd.read_file``), it left the E-G-H
    ``add_sectors`` process to crash on exit (access violation while garbage
    collecting, on Windows), with or without Arrow.
    """
    with closing(sqlite3.connect(path)) as con:
        column, srs_id = con.execute(
            "SELECT column_name, srs_id FROM gpkg_geometry_columns WHERE table_name = ?",
            (layer,),
        ).fetchone()
        (crs,) = con.execute("SELECT definition FROM gpkg_spatial_ref_sys WHERE srs_id = ?", (srs_id,)).fetchone()
        blobs = [row[0] for row in con.execute(f'SELECT "{column}" FROM "{layer}"') if row[0] is not None]

    # GeoPackage binary header: 'GP', version, flags, srs id (8 bytes), then an
    # envelope whose size the flags' bits 1-3 give
    envelope_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    wkb = [blob[8 + envelope_bytes[(blob[3] >> 1) & 0b111] :] for blob in blobs]
    return gpd.GeoDataFrame(geometry=shapely.from_wkb(wkb), crs=crs)


def add_hydrogen_storage_buses(n: pypsa.Network, states: pd.Index, center_points: pd.DataFrame) -> None:
    """Adds an underground hydrogen storage bus for every state."""
    states = list(states)
    names = [f"{x} H2 storage" for x in states]
    n.add("Bus", names, carrier="h2_storage", **_state_bus_attrs(states, center_points))


def add_hydrogen_storage(n: pypsa.Network, costs: pd.DataFrame, year: int, storage_types: pd.Series) -> None:
    """
    Adds underground hydrogen storage in every state.

    The storage capacity, the injection (charger) and the withdrawal
    (discharger) are sized separately. The store keeps a ``cushion`` share of its
    capacity; its cost is per MWh of working capacity. The energy used to compress
    the injected hydrogen is counted as a loss of hydrogen.
    """
    states = storage_types.index
    storage_bus = states + " H2 storage"
    storage_carrier = "h2_storage_" + storage_types
    cushion = storage_carrier.map(lambda x: costs.at[x, "cushion"]).values

    n.add(
        "Store",
        storage_bus,
        suffix=f"_{year}",
        bus=storage_bus,
        carrier=storage_carrier.values,
        e_nom_extendable=True,
        e_min_pu=cushion,
        e_cyclic_per_period=True,  # pypsa v1 flipped this default to False
        capital_cost=storage_carrier.map(lambda x: costs.at[x, "annualized_capex_fom"]).values * (1 - cushion),
        lifetime=storage_carrier.map(lambda x: costs.at[x, "lifetime"]).values,
        build_year=year,
    )

    carrier = "h2_storage_charger"
    n.add(
        "Link",
        states,
        suffix=f" H2 storage charger_{year}",
        bus0=states + " H2",
        bus1=storage_bus,
        carrier=carrier,
        efficiency=1 - costs.at[carrier, "compression-loss"],
        capital_cost=costs.at[carrier, "annualized_capex_fom"],
        p_nom_extendable=True,
        lifetime=costs.at[carrier, "lifetime"],
        build_year=year,
    )

    carrier = "h2_storage_discharger"
    n.add(
        "Link",
        states,
        suffix=f" H2 storage discharger_{year}",
        bus0=storage_bus,
        bus1=states + " H2",
        carrier=carrier,
        efficiency=1,
        capital_cost=costs.at[carrier, "annualized_capex_fom"],
        p_nom_extendable=True,
        lifetime=costs.at[carrier, "lifetime"],
        build_year=year,
    )


def get_pipeline_routes(n: pypsa.Network, states: list[str], gas_pipelines: pd.DataFrame) -> list[tuple[str, str]]:
    """
    Pairs of states in the model joined by AC transmission or a natural gas pipeline.

    AC transmission: a line, or a link of carrier 'AC' (transport models), between
    AC buses of the two states in the network. DC links do not count.
    Gas: a pipeline between the two states in ``gas_pipelines`` (``get_gas_pipelines``).
    """
    ac_state = n.buses.loc[n.buses.carrier == "AC", "STATE"]
    branches = pd.concat([n.lines[["bus0", "bus1"]], n.links.loc[n.links.carrier == "AC", ["bus0", "bus1"]]])
    branches = branches[branches.bus0.isin(ac_state.index) & branches.bus1.isin(ac_state.index)]
    electricity = _state_pairs(branches.bus0.map(ac_state), branches.bus1.map(ac_state), states)

    gas = _state_pairs(gas_pipelines.STATE_FROM, gas_pipelines.STATE_TO, states)

    routes = sorted(electricity | gas)
    logger.info(
        f"Found {len(routes)} hydrogen pipeline routes: {len(electricity)} along AC transmission, "
        f"{len(gas)} along natural gas pipelines",
    )
    return routes


def get_gas_pipelines(path: str) -> pd.DataFrame:
    """
    State to state natural gas pipelines with capacity in the latest year of the
    EIA state to state capacity data, as used by the natural gas sector.
    """
    df = pd.read_excel(path, sheet_name="Pipeline State2State Capacity", skiprows=1, index_col=0)
    df.columns = df.columns.str.strip()
    df = df[(df.index == df.index.max()) & (df["Capacity (mmcfd)"] > 0)]
    return pd.DataFrame(
        {
            "STATE_FROM": df["State From"].astype(str).str.strip().map(STATE_2_CODE),
            "STATE_TO": df["State To"].astype(str).str.strip().map(STATE_2_CODE),
        },
    )


def _state_pairs(a: pd.Series, b: pd.Series, states: list[str]) -> set[tuple[str, str]]:
    """Unordered pairs of different states, both in ``states``."""
    pairs = pd.DataFrame({"a": a.values, "b": b.values})
    pairs = pairs[pairs.a.isin(states) & pairs.b.isin(states) & (pairs.a != pairs.b)]
    return {tuple(sorted(x)) for x in zip(pairs.a, pairs.b)}


def add_hydrogen_pipelines(
    n: pypsa.Network,
    costs: pd.DataFrame,
    year: int,
    routes: list[tuple[str, str]],
    length_factor: float,
) -> None:
    """
    Adds new hydrogen pipelines along the routes of ``get_pipeline_routes``.

    The route length is the distance between the state centers times
    ``length_factor``.
    """
    if not routes:
        return

    carrier = "h2_pipeline"
    a, b = (pd.Index([x[i] for x in routes]) for i in (0, 1))
    bus0, bus1 = a + " H2", b + " H2"

    length = (
        pypsa.geo.haversine_pts(
            n.buses.loc[bus0, ["x", "y"]].to_numpy(),
            n.buses.loc[bus1, ["x", "y"]].to_numpy(),
        )
        * length_factor
    )
    _add_pipeline_pairs(n, costs, carrier, year, a + " " + b + f" {carrier}", bus0, bus1, length)


def _add_pipeline_pairs(
    n: pypsa.Network,
    costs: pd.DataFrame,
    carrier: str,
    year: int,
    names: pd.Index,
    bus0: pd.Index,
    bus1: pd.Index,
    length: np.ndarray,
    fwd: dict | None = None,
    **kwargs,
) -> None:
    """
    Adds hydrogen pipelines as pairs of links, one per direction.

    The bidirectional link constraint keeps both at the same capacity. Pipes are
    costed per MW and km of ``carrier``, compressor stations included. As for
    natural gas, the hydrogen transported loses a share per distance: the energy
    used for compression and the leakage. ``fwd`` holds attributes of the forward
    links only, ``kwargs`` those of both.
    """
    loss_per_1000km = (costs.at["h2_pipeline", "compression-loss"] + costs.at["h2_pipeline", "leakage"]) / 100
    efficiency = 1 - loss_per_1000km * length / 1000
    # split over the two links of a route
    capital_cost = costs.at[carrier, "annualized_capex_fom"] * length / 2

    for direction, (start, end), attrs in (("fwd", (bus0, bus1), fwd or {}), ("rev", (bus1, bus0), {})):
        n.add(
            "Link",
            names,
            suffix=f"_{direction}_{year}",
            bus0=start,
            bus1=end,
            carrier=carrier,
            length=length,
            efficiency=efficiency,
            capital_cost=capital_cost,
            p_nom_extendable=True,
            lifetime=costs.at[carrier, "lifetime"],
            build_year=year,
            **attrs,
            **kwargs,
        )


###
# natural gas infrastructure retrofitted to hydrogen
###


def add_retrofitted_storage(n: pypsa.Network, costs: pd.DataFrame, year: int, center_points: pd.DataFrame) -> None:
    """
    Adds hydrogen storage retrofitted from the natural gas storage reservoirs.

    Salt dome, depleted field and aquifer reservoirs can be retrofitted in part.
    The share retrofitted holds ``energy-conversion`` times its natural gas energy
    as hydrogen, with the same cushion share, and withdraws hydrogen at
    ``power-conversion`` times its natural gas withdrawal. New compressors inject
    the hydrogen through the same wells, at most at that rate. A storage of several
    formations (``natural_gas: storage_aggregation: state``) takes the mean of
    their conversion factors and costs: weighted by their working gas for the
    store, by their withdrawal for the rates. ``opts.retrofit`` takes the share
    from the natural gas storage.
    """
    gas = n.stores[n.stores.carrier == "gas storage"]
    if gas.empty:
        logger.warning("No natural gas storage to retrofit to hydrogen; it needs the natural gas sector (E-G-H)")
        return
    working = _formation_shares(gas, "working")
    withdrawal = _formation_shares(gas, "withdrawal")
    # a storage without working gas or withdrawal holds nothing to retrofit
    usable = working.sum(axis=1).gt(0) & withdrawal.sum(axis=1).gt(0)
    gas, working, withdrawal = gas[usable], working[usable], withdrawal[usable]

    def by_formation(technology: str, parameter: str) -> pd.Series:
        return pd.Series(
            {x: costs.at[f"{technology}_{suffix}", parameter] for x, suffix in RETROFITTABLE_FORMATIONS.items()}
        )

    def mean(weights: pd.DataFrame, values: pd.Series) -> np.ndarray:
        return (weights * values).sum(axis=1).div(weights.sum(axis=1)).to_numpy()

    energy = by_formation("h2_storage_retrofitted", "energy-conversion")
    power = by_formation("h2_storage_retrofitted", "power-conversion")
    # the hydrogen each formation holds, and withdraws, weighs its costs
    h2_working = working * energy
    h2_withdrawal = withdrawal * power
    energy_factor = mean(working, energy)
    power_factor = mean(withdrawal, power)

    links = n.links[n.links.carrier == "gas storage"]
    charge = pd.Index(links.index[links.bus1.isin(gas.bus)])
    discharge = pd.Index(links.index[links.bus0.isin(gas.bus)])
    # in the order of the reservoirs
    charge = charge[pd.Index(n.links.loc[charge, "bus1"]).get_indexer(gas.bus)]
    discharge = discharge[pd.Index(n.links.loc[discharge, "bus0"]).get_indexer(gas.bus)]

    states = pd.Index(n.links.loc[charge, "bus0"].map(n.buses.STATE))
    existing_energy = _existing_capacity(n, "Store", gas.index)

    buses = pd.Index(gas.bus.str.removesuffix(" gas storage") + " H2 storage retrofitted")
    n.add("Bus", buses, carrier="h2_storage", **_state_bus_attrs(list(states), center_points))
    n.add(
        "Store",
        buses,
        suffix=f"_{year}",
        bus=buses,
        carrier="h2_storage_retrofitted",
        e_nom_extendable=True,
        e_nom_max=energy_factor * existing_energy,
        e_min_pu=gas.e_min_pu.to_numpy(),
        e_cyclic_per_period=True,
        # the cost is per MWh of working capacity, the store spans the cushion too
        capital_cost=mean(h2_working, by_formation("h2_storage_retrofitted", "annualized_capex_fom"))
        * (1 - gas.e_min_pu.to_numpy()),
        lifetime=mean(h2_working, by_formation("h2_storage_retrofitted", "lifetime")),
        build_year=year,
        retrofit_from=gas.index,
        retrofit_factor=energy_factor,
    )

    # the wells withdraw the hydrogen at a share of the gas rate; new compressors
    # inject it through the same wells, at most at that rate. The gas injection,
    # kept to the wells left to gas, contracts too
    _existing_capacity(n, "Link", charge)
    withdrawal = power_factor * _existing_capacity(n, "Link", discharge)
    for kind, bus0, bus1, efficiency, retrofitted in (
        ("charger", states + " H2", buses, 1 - costs.at["h2_storage_charger", "compression-loss"], {}),
        ("discharger", buses, states + " H2", 1, {"retrofit_from": discharge, "retrofit_factor": power_factor}),
    ):
        # the injection of each formation is the same share of its withdrawal
        technology = f"h2_storage_{kind}_retrofitted"
        n.add(
            "Link",
            buses,
            suffix=f" {kind}_{year}",
            bus0=bus0,
            bus1=bus1,
            carrier=f"h2_storage_{kind}",
            efficiency=efficiency,
            capital_cost=mean(h2_withdrawal, by_formation(technology, "annualized_capex_fom")),
            p_nom_extendable=True,
            p_nom_max=withdrawal,
            lifetime=mean(h2_withdrawal, by_formation(technology, "lifetime")),
            build_year=year,
            **retrofitted,
        )

    logger.info(f"Natural gas storage that can be retrofitted to hydrogen: {gas.index.tolist()}")


def _formation_shares(gas: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    Share of each formation in the ``kind`` ('working' gas or 'withdrawal') of the
    natural gas storage, as ``build_natural_gas.GasStorage`` gives it.
    """
    columns = gas.columns[gas.columns.str.startswith(f"{kind}_share_")]
    if columns.empty:
        raise ValueError("The natural gas storage has no formation shares to retrofit it to hydrogen by")
    shares = gas[columns].fillna(0)
    shares.columns = columns.str.removeprefix(f"{kind}_share_")
    unknown = shares.columns.difference(list(RETROFITTABLE_FORMATIONS))
    if shares[unknown].gt(0).any(axis=None):
        raise ValueError(f"No hydrogen retrofitting costs for the gas storage formations {unknown.tolist()}")
    return shares.reindex(columns=list(RETROFITTABLE_FORMATIONS), fill_value=0)


def add_retrofitted_pipelines(n: pypsa.Network, costs: pd.DataFrame, year: int) -> None:
    """
    Adds hydrogen pipelines retrofitted from the natural gas pipelines between states.

    A retrofitted pipeline carries ``capacity-conversion`` times the natural gas
    capacity it takes, in either direction, along the length of the gas
    pipeline. ``opts.retrofit`` takes the capacity from the natural gas
    pipeline.
    """
    gas = n.links[n.links.carrier == "gas pipeline"]
    if gas.empty:
        logger.warning("No natural gas pipelines to retrofit to hydrogen; they need the natural gas sector (E-G-H)")
        return

    carrier = "h2_pipeline_retrofitted"
    factor = costs.at[carrier, "capacity-conversion"]
    bus0 = pd.Index(gas.bus0.map(n.buses.STATE) + " H2")
    bus1 = pd.Index(gas.bus1.map(n.buses.STATE) + " H2")
    in_model = bus0.isin(n.buses.index) & bus1.isin(n.buses.index)
    gas, bus0, bus1 = gas[in_model], bus0[in_model], bus1[in_model]

    _add_pipeline_pairs(
        n,
        costs,
        carrier,
        year,
        gas.index + f" {carrier}",
        bus0,
        bus1,
        gas.length.to_numpy(),
        # the reverse link follows the forward one, which takes the gas capacity
        fwd={"retrofit_from": gas.index, "retrofit_factor": factor},
        p_nom_max=factor * _existing_capacity(n, "Link", gas.index),
    )


def _existing_capacity(n: pypsa.Network, component: str, names: pd.Index) -> np.ndarray:
    """
    Existing capacity of natural gas components that contract, from which what is
    retrofitted is taken (``_helpers.get_gas_infra_contraction``).
    """
    static = n.components[component].static
    attr = "e_nom" if component == "Store" else "p_nom"
    fixed = names[~static.loc[names, f"{attr}_extendable"].to_numpy()]
    if not fixed.empty:
        raise ValueError(f"Natural gas components to retrofit to hydrogen must contract: {fixed.tolist()}")
    return static.loc[names, f"{attr}_max"].to_numpy()
