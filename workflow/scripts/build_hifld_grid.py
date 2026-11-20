"""
Builds a Breakthrough-Energy-format base grid out of the HIFLD transmission layer.

Why
---
The Breakthrough Energy / TAMU grid that PyPSA-USA ships is, in its authors' own
words, "a fictitious configuration": its corridors are synthesised, not surveyed.
Comparing it against HIFLD shows the discrepancy is not cosmetic. BE's 345 kV
lines sit a median 1.8 km from *some* real line but 19.3 km from any real line of
345 kV or above, i.e. they are drawn along real sub-transmission corridors while
carrying a synthetic EHV label. Loop geometry built on those corridors has no
physical original, which matters for any study whose answer depends on where the
parallel paths are.

This script keeps HIFLD's geometry and voltages and borrows only BE's *typical
values*: per-km r/x/b and per-circuit MVA, taken as the median of BE's own lines
at each voltage class. BE's impedances are internally consistent -- its label-free
surge impedance loading, 100*sqrt(b/x) MW, comes out at 22/74/182/460/1008/2239 MW
at its six voltage labels against textbook values of 12/50/140/420/1000/2300 -- so
rebuilding a real corridor from BE's medians keeps the electrical parameters
coherent instead of splicing in a foreign parameter set.

The output is written in BE's CSV schema so that ``build_base_network`` and every
rule downstream of it read it unchanged. The grid is built once for the whole
country -- interconnection labels and the seam cut need the national graph --
and ``build_base_network`` then filters it by ``{interconnect}`` exactly as it
filters the BE grid.

Selected with ``model_topology: transmission_network: hifld``. The input is the
HIFLD "Electric Power Transmission Lines" layer as downloaded by
``rule retrieve_hifld_transmission_lines``; ``reduce_lines`` keeps one row per
line record with its voltage, DC flag, endpoints and route length.

The five things HIFLD cannot supply on its own
----------------------------------------------
1. *Nodes.* HIFLD names substations in SUB_1/SUB_2, but the names are not keys:
   "NOT AVAILABLE" alone appears 17,510 times spanning the continent, and even at
   a 50 m clustering radius 29% of sites carry two different known names. So the
   endpoints are clustered geometrically (``SITE_RADIUS_KM``) and names are
   discarded.

2. *Interconnections.* HIFLD is pure AC geometry with no interconnection field,
   and the three US interconnections meet at back-to-back DC stations that are a
   single site on the ground. Clustered naively the whole country becomes one
   synchronous island of 880,000 km. Each site is therefore labelled from the
   nearest BE substation, the labels are smoothed over the graph so they are
   edge-consistent rather than merely nearest-neighbour, and every line whose two
   ends land in different interconnections is cut. See ``assign_interconnects``.

3. *Load.* HIFLD carries no demand. ``Pd`` -- read downstream only when
   ``electricity.demand.bus_allocation: breakthrough``; the default
   ``population`` allocation recomputes its own weight in ``build_base_network``
   -- is built from county population (2020 Decennial Census, the same file the
   ``population`` allocation reads) spread uniformly over each county's area and
   collected by nearest site, i.e. a Voronoi allocation evaluated on a grid.

4. *Offshore points of interconnection.* The BE path finds them by looking for
   onshore buses wired to BE's own offshore substations, which HIFLD does not
   have. The same BE sites (see ``be_offshore_poi_sites``) are matched here to
   the nearest HIFLD bus.

5. *Generators.* Nothing to do for the main fleet: ``add_electricity`` matches
   ``powerplants.csv`` (PUDL/EIA, carries lat/lon) to buses by coordinate, so it
   follows the topology automatically. Only BE's own ``plant.csv`` -- used solely
   for hydro -- is keyed by BE bus id, so it is re-keyed here by coordinate.
"""

import itertools
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from _helpers import configure_logging
from build_base_network import BE_OFFSHORE_SUB_ID_MIN, TEXAS_POI_SUB_IDS
from build_bus_population import load_county_population
from build_shapes import load_na_shapes
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import BallTree

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0088

# HIFLD's voltage classes and the bin edges that map a nominal kV onto one.
VOLTAGE_CLASSES = (69, 138, 230, 345, 500, 765)
CLASS_EDGES = (100, 200, 300, 400, 700)

# Two line endpoints closer than this are the same substation site.
#
# There is no natural threshold to read off: the nearest-neighbour distance
# between distinct endpoint coordinates peaks at 25-50 m and then falls smoothly
# all the way out to the kilometre scale with no valley, so the within-yard and
# between-substation populations overlap continuously. Three measurements bracket
# it instead, and they agree:
#
#   * 68,042 endpoints over 8.08e6 km^2 give a density of 0.0084/km^2, so if
#     substations were independent only ~40 pairs would fall within 150 m. There
#     are 21,478. Below 150 m is essentially all one yard's own busbars.
#   * Where HIFLD names both endpoints, P(the two share a name) sits on a plateau
#     of ~0.53 from 25 m out to 175 m, breaks below 0.5 at 175-200 m and decays to
#     its far-field baseline of 0.20 by ~300 m. (The plateau is at 0.53 rather
#     than 1.0 because the naming is about half noise -- the same yard often
#     carries two different names -- which biases the crossing point downwards, so
#     185 m is a floor.)
#   * Swept end to end, the route-km lost to the two opposing errors -- a yard
#     split into unconnected fragments, and a real short line collapsed to a
#     self-loop -- totals 12,714 / 12,699 / 12,653 / 12,574 / 12,816 / 13,845 km
#     at 0.05 / 0.10 / 0.15 / 0.20 / 0.30 / 0.50 km. The minimum is at 0.20.
#
# One further reason to err small: merging here destroys a line outright -- it
# becomes a self-loop and is dropped -- whereas the downstream {simpl}
# clustering (cluster_resources) merges nearby sites anyway where the requested
# resolution calls for it, so borderline pairs are better left to that pass.
SITE_RADIUS_KM = 0.2

# How far outside a state polygon a substation may sit and still count as being in
# the country. It exists only for the coastline: a waterfront or Great Lakes
# substation routinely lands a kilometre or two offshore of a generalised
# shoreline. The value is not delicate -- of the sites outside every state polygon
# and outside Canada and Mexico, 199 lie within 5 km, exactly one more within
# 50 km, and the remaining 384 are Alaska, Hawaii, Puerto Rico and Guam, thousands
# of kilometres away -- so anything in the 5-50 km gap gives the same grid.
COAST_TOLERANCE_KM = 5.0

# Sweeps of graph-consistency smoothing applied to the interconnection labels.
LABEL_SWEEPS = 30
# A site whose nearest BE substation is this far away (km) has its own opinion
# discounted by half, so isolated sites follow their neighbours instead.
ANCHOR_HALF_LIFE_KM = 5.0

# Grid spacing (m, in EPSG:5070) used to spread county population over its area.
POPULATION_GRID_M = 5000

# Id offsets, chosen to clear every hard-coded id in build_base_network: BE's
# offshore substations start at 41012, and the synthetic offshore grid built later
# uses sub ids from 50000 and bus ids from 3008161.
SUB_ID_OFFSET = 100_000
BUS_ID_OFFSET = 200_000

# How far a BE-keyed record (a DC link end, a hydro plant, a POI) may be moved
# onto the nearest HIFLD bus before it is dropped instead. The DC allowance is the
# loosest of the three: a back-to-back tie sits *on* the seam, so the nearest bus
# on the far side can be tens of kilometres away -- 50.5 km at Miles City, the
# worst of the 17 -- and dropping a tie would sever an interconnection exchange
# outright, which is a far worse error than siting it one substation off.
MAX_REKEY_KM = {"dcline": 75.0, "plant": 100.0, "poi": 100.0}


#: Voltage (kV) taken for a record whose ``VOLTAGE`` is missing, by ``VOLT_CLASS``.
CLASS_KV = {
    "UNDER 100": 69.0,
    "100-161": 138.0,
    "220-287": 230.0,
    "345": 345.0,
    "500": 500.0,
    "735 AND ABOVE": 765.0,
}

#: ``STATUS`` values of lines that do not carry power.
DROP_STATUS = ("UNDER CONSTRUCTION", "INACTIVE")

COLUMNS = ["id", "kv", "dc", "lon1", "lat1", "lon2", "lat2", "route_km"]


def _parts(geometry):
    return list(geometry.geoms) if geometry.geom_type == "MultiLineString" else [geometry]


def endpoints(geometry) -> tuple[float, float, float, float]:
    """First point of the first part and last point of the last part."""
    parts = _parts(geometry)
    (lon1, lat1), (lon2, lat2) = parts[0].coords[0][:2], parts[-1].coords[-1][:2]
    return lon1, lat1, lon2, lat2


def route_km(geometry) -> float:
    """Length of a (multi)line string in km, as the sum of great-circle segments."""
    total = 0.0
    for part in _parts(geometry):
        xy = np.radians(np.asarray(part.coords)[:, :2])
        lon, lat = xy[:, 0], xy[:, 1]
        a = np.sin(np.diff(lat) / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2
        total += float((2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))).sum())
    return total


def reduce_lines(raw: gpd.GeoDataFrame) -> pd.DataFrame:
    """The HIFLD layer as one row per line record with the :data:`COLUMNS`.

    ``kv`` is ``VOLTAGE`` or, where that is missing (``-999999``), the
    representative voltage of ``VOLT_CLASS`` (:data:`CLASS_KV`); ``dc`` is 1 if
    ``TYPE`` names DC; the endpoints are the first point of the first part and the
    last point of the last part; ``route_km`` sums great-circle segments. Records
    with neither a voltage nor a voltage class, :data:`DROP_STATUS` records and
    empty geometries are dropped.
    """
    geometry = raw.geometry.name
    lines = raw.rename(columns={c: c.upper() for c in raw.columns if c != geometry})
    if lines.crs is not None:
        lines = lines.to_crs(4326)
    lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty]

    voltage = pd.to_numeric(lines["VOLTAGE"], errors="coerce")
    kv = voltage.where(voltage > 0, lines["VOLT_CLASS"].map(CLASS_KV))
    keep = kv.notna() & ~lines["STATUS"].isin(DROP_STATUS)
    logger.info(
        "Keeping %d of %d records; dropping %d without a voltage or voltage class and %d %s.",
        int(keep.sum()),
        len(lines),
        int(kv.isna().sum()),
        int((kv.notna() & lines["STATUS"].isin(DROP_STATUS)).sum()),
        " or ".join(DROP_STATUS),
    )
    lines, kv = lines[keep], kv[keep]

    ends = np.array([endpoints(g) for g in lines.geometry])
    out = pd.DataFrame(
        {
            "id": pd.to_numeric(lines["ID"]).astype(np.int64).to_numpy(),
            "kv": kv.astype(float).to_numpy(),
            "dc": lines["TYPE"].astype(str).str.contains("DC").astype(int).to_numpy(),
            "lon1": ends[:, 0],
            "lat1": ends[:, 1],
            "lon2": ends[:, 2],
            "lat2": ends[:, 3],
            "route_km": [route_km(g) for g in lines.geometry],
        },
    )
    return out.sort_values("id").reset_index(drop=True)[COLUMNS]


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(v, dtype=float)) for v in (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def class_of(kv):
    """Map a nominal voltage onto its HIFLD voltage class."""
    return np.select(
        [np.asarray(kv, dtype=float) < edge for edge in CLASS_EDGES],
        list(VOLTAGE_CLASSES[:-1]),
        default=VOLTAGE_CLASSES[-1],
    ).astype(int)


def cluster_sites(lon, lat, radius_km=SITE_RADIUS_KM):
    """Union-find over every pair of endpoints closer than ``radius_km``.

    Returns one integer site id per input point. Coordinates are de-duplicated
    first: of ~186k line endpoints only ~75k are distinct, the rest being lines
    that genuinely terminate on the very same point.
    """
    frame = pd.DataFrame({"lon": np.asarray(lon), "lat": np.asarray(lat)})
    frame["key"] = frame.lon.astype(str) + "_" + frame.lat.astype(str)
    uniq = frame.drop_duplicates("key").reset_index(drop=True)

    radians = np.c_[np.radians(uniq.lat.to_numpy()), np.radians(uniq.lon.to_numpy())]
    neighbours = BallTree(radians, metric="haversine").query_radius(radians, r=radius_km / EARTH_RADIUS_KM)

    parent = np.arange(len(uniq))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, group in enumerate(neighbours):
        for j in group:
            if j <= i:
                continue
            a, b = find(i), find(j)
            if a != b:
                parent[max(a, b)] = min(a, b)

    label = pd.factorize(np.array([find(i) for i in range(len(uniq))]))[0]
    return frame.key.map(pd.Series(label, index=uniq.key.to_numpy())).to_numpy()


def _within(points, shapes):
    """Boolean per point: whether it falls inside any of ``shapes``."""
    hit = gpd.sjoin(points, shapes[["geometry"]].to_crs(points.crs), how="left", predicate="within")
    return hit.groupby(level=0).index_right.first().reindex(range(len(points))).notna().to_numpy()


def in_united_states(site_xy, state_shapes, foreign_shapes, tolerance_km=COAST_TOLERANCE_KM):
    """True for the sites that stand in the United States.

    A site can fall outside the 48-state polygon for two quite different reasons,
    and only one of them means it is abroad, so the two are tested separately.

    *Coastline.* A waterfront or Great Lakes substation routinely lands a
    kilometre or two offshore of a generalised shoreline -- Florida, Washington,
    Michigan, Maryland, New York account for most of them. Those are American and
    are kept, which is what ``tolerance_km`` is for.

    *Foreign soil.* Slack alone would also admit the cross-border spurs HIFLD
    carries, because Nuevo Laredo, Piedras Negras, Matamoros and Ciudad Juarez all
    sit within 2 km of the Texas polygon: along the Rio Grande the border *is* the
    water, so no distance threshold can tell them from a US waterfront. Foreign
    territory is therefore excluded outright, by testing the Canadian provinces
    and Mexican states rather than inferring anything from distance. A site inside
    one of those is out however close to the US it lies. (The two datasets do not
    disagree: no site falls inside both.)

    A line is dropped when *either* end fails, so a tie that crosses the border
    leaves the model rather than dangling at a foreign bus.
    """
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(site_xy.lon, site_xy.lat),
        crs="EPSG:4326",
    )
    abroad = _within(points, foreign_shapes)

    projected = points.to_crs("EPSG:5070")
    nearest = gpd.sjoin_nearest(projected, state_shapes[["geometry"]].to_crs("EPSG:5070"), distance_col="_m")
    distance_km = nearest.groupby(level=0)["_m"].min().reindex(range(len(site_xy))).to_numpy() / 1000

    logger.info(
        "Border test: %d site(s) on Canadian or Mexican soil, %d more further than %.0f km "
        "from any state (Alaska, Hawaii, Puerto Rico, Guam).",
        int(abroad.sum()),
        int((~abroad & (distance_km > tolerance_km)).sum()),
        tolerance_km,
    )
    return (~abroad) & (distance_km <= tolerance_km)


def assign_interconnects(site_xy, bus0, bus1, be_substations):
    """Label each site Eastern/Western/Texas and flag the lines that bridge two.

    HIFLD has no interconnection field, so the label is seeded from the nearest BE
    substation. That alone is not enough: near a seam the nearest-substation rule
    flips back and forth over a few kilometres and would cut a ragged line through
    real corridors. The seed is therefore smoothed by repeated majority vote over
    the graph -- each site counts its own seed, weighted down the further its
    nearest BE substation is, plus one vote per incident line -- which pulls the
    boundary onto the small set of edges that genuinely bridge the two systems.
    """
    tree = BallTree(np.c_[np.radians(be_substations.lat), np.radians(be_substations.lon)], metric="haversine")
    distance, index = tree.query(np.c_[np.radians(site_xy.lat), np.radians(site_xy.lon)], k=1)
    distance_km = distance[:, 0] * EARTH_RADIUS_KM

    names = np.array(["Eastern", "Western", "Texas"])
    code = {name: i for i, name in enumerate(names)}
    seed = np.array([code[name] for name in be_substations.interconnect.to_numpy()[index[:, 0]]])
    label = seed.copy()

    anchor = np.clip(1.0 / (1.0 + distance_km / ANCHOR_HALF_LIFE_KM), 0.05, 1.0)
    n_site = len(site_xy)
    changed = 0
    for sweep in range(LABEL_SWEEPS):
        votes = np.zeros((n_site, len(names)))
        votes[np.arange(n_site), seed] += 2.0 * anchor
        for here, there in ((bus0, bus1), (bus1, bus0)):
            np.add.at(votes, (here, label[there]), 1.0)
        new = votes.argmax(axis=1)
        changed = int((new != label).sum())
        label = new
        if changed == 0:
            logger.info("Interconnection labels converged after %d sweep(s).", sweep + 1)
            break
    else:
        logger.info("Interconnection labels still moving for %d site(s) after %d sweeps.", changed, LABEL_SWEEPS)

    logger.info(
        "Interconnection labels: %s; %d site(s) moved off their nearest BE substation.",
        {name: int((label == i).sum()) for i, name in enumerate(names)},
        int((label != seed).sum()),
    )
    return names[label], label[bus0] != label[bus1]


def largest_component_per_interconnect(n_site, bus0, bus1, interconnect):
    """Keep, for each interconnection, only its largest connected component.

    Once the seam is cut the three interconnections separate cleanly, but HIFLD
    also leaves several hundred small fragments -- stubs whose connecting line is
    missing from the layer, plus genuinely asynchronous pockets such as northern
    Maine. A fragment carrying load and no generation is simply infeasible, so
    they are dropped rather than patched. With the non-CONUS sites already gone
    this costs of order 1% of route-km.
    """
    graph = coo_matrix((np.ones(len(bus0)), (bus0, bus1)), shape=(n_site, n_site))
    _, component = connected_components(graph + graph.T, directed=False)

    keep_site = np.zeros(n_site, dtype=bool)
    for name in np.unique(interconnect):
        mask = interconnect == name
        counts = pd.Series(component[mask]).value_counts()
        keep_site |= mask & (component == counts.index[0])
        logger.info(
            "%s: %d site(s) in %d component(s); keeping the largest (%d sites, %.1f%%), second largest has %d.",
            name,
            int(mask.sum()),
            len(counts),
            int(counts.iloc[0]),
            100 * counts.iloc[0] / mask.sum(),
            int(counts.iloc[1]) if len(counts) > 1 else 0,
        )
    return keep_site


def class_reference_parameters(be_bus, be_branch):
    """Per-km r/x/b and per-circuit MVA for each voltage class, from BE's medians.

    BE's per-unit values are on a 100 MVA base, so a line's ohmic value is
    ``r_pu * kV**2 / 100``; dividing by the great-circle distance between its two
    substations gives the per-km figure. Only lines over 1 km with a positive
    rating are used, which excludes bus couplers.
    """
    lines = be_branch.query("branch_device_type == 'Line'").copy()
    for side in ("from", "to"):
        lines[f"{side}_kv"] = lines[f"{side}_bus_id"].map(be_bus.baseKV)
        lines[f"{side}_lat"] = lines[f"{side}_bus_id"].map(be_bus.lat)
        lines[f"{side}_lon"] = lines[f"{side}_bus_id"].map(be_bus.lon)
    lines["km"] = haversine_km(lines.from_lon, lines.from_lat, lines.to_lon, lines.to_lat)
    usable = lines.query("km > 1 and x > 0 and b > 0 and rateA > 0 and from_kv == to_kv")
    usable = usable.assign(kV=usable.from_kv, cls=class_of(usable.from_kv))

    reference = {}
    for level in VOLTAGE_CLASSES:
        group = usable.loc[usable.cls == level]
        if len(group) < 20:
            raise ValueError(f"Only {len(group)} usable BE lines in the {level} kV class; cannot take a median.")
        reference[level] = {
            "r_ohm_km": float((group.r * group.kV**2 / 100 / group.km).median()),
            "x_ohm_km": float((group.x * group.kV**2 / 100 / group.km).median()),
            "b_s_km": float((group.b * 100 / group.kV**2 / group.km).median()),
            "mva": float(group.rateA.median()),
        }
    return reference


def transformer_reference_parameters(be_bus, be_branch):
    """Median per-unit r/x and MVA for each ordered (low class, high class) pair.

    Expressed on the low-voltage side's base, because ``add_branches_from_file``
    turns per-unit into ohms using the *from* bus's nominal voltage and the
    transformers written here always run low -> high.
    """
    transformers = be_branch.query("branch_device_type in ['Transformer', 'TransformerWinding']").copy()
    from_kv = transformers.from_bus_id.map(be_bus.baseKV).to_numpy(dtype=float)
    c0 = class_of(transformers.from_bus_id.map(be_bus.baseKV))
    c1 = class_of(transformers.to_bus_id.map(be_bus.baseKV))
    low, high = np.minimum(c0, c1), np.maximum(c0, c1)
    scale = (from_kv**2) / (low.astype(float) ** 2)
    transformers["r_lv"] = transformers.r * scale
    transformers["x_lv"] = transformers.x * scale
    transformers["low"], transformers["high"] = low, high

    usable = transformers.query("low != high and x_lv > 0 and rateA > 0")
    grouped = usable.groupby(["low", "high"]).agg(r=("r_lv", "median"), x=("x_lv", "median"), mva=("rateA", "median"))
    return grouped.to_dict("index")


def population_weights(site_xy, county_shapes, county_population):
    """Population nearest to each site, from county totals spread over county area.

    County population is treated as uniform within the county, sampled on a
    ``POPULATION_GRID_M`` grid in an equal-area projection, and each sample is
    collected by its nearest site. That is a Voronoi allocation of population to
    substations, evaluated numerically so no Voronoi polygons have to be built and
    clipped.

    ``county_population`` is indexed by the ``p``-prefixed county id that
    ``build_shapes`` writes into the county shapes' ``GEOID`` column (see
    ``build_bus_population.load_county_population``).
    """
    counties = county_shapes.assign(population=county_shapes["GEOID"].map(county_population))
    missing = int(counties.population.isna().sum())
    if missing:
        logger.info("No population for %d of %d county shapes (territories); treated as zero.", missing, len(counties))
    counties = counties.dropna(subset=["population"]).to_crs("EPSG:5070")

    samples, weights = [], []
    for geometry, population in zip(counties.geometry, counties.population):
        minx, miny, maxx, maxy = geometry.bounds
        xs = np.arange(minx + POPULATION_GRID_M / 2, maxx, POPULATION_GRID_M)
        ys = np.arange(miny + POPULATION_GRID_M / 2, maxy, POPULATION_GRID_M)
        chosen = np.empty((0, 2))
        if len(xs) and len(ys):
            grid = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
            chosen = grid[shapely.contains_xy(geometry, grid[:, 0], grid[:, 1])]
        if len(chosen) == 0:
            # a county smaller than one grid cell still holds its people
            point = geometry.representative_point()
            chosen = np.array([[point.x, point.y]])
        samples.append(chosen)
        weights.append(np.full(len(chosen), population / len(chosen)))

    stacked = np.concatenate(samples)
    points = gpd.GeoSeries(gpd.points_from_xy(stacked[:, 0], stacked[:, 1]), crs="EPSG:5070").to_crs("EPSG:4326")
    weight = np.concatenate(weights)
    logger.info("Spread %.0f people over %d grid sample(s).", weight.sum(), len(weight))

    tree = BallTree(np.c_[np.radians(site_xy.lat), np.radians(site_xy.lon)], metric="haversine")
    _, nearest = tree.query(np.c_[np.radians(points.y.to_numpy()), np.radians(points.x.to_numpy())], k=1)
    return np.bincount(nearest[:, 0], weights=weight, minlength=len(site_xy))


def rekey_by_coordinate(lat, lon, bus_table, max_km, restrict_to=None):
    """Re-point a record keyed by BE bus id at the nearest bus of the new network.

    ``restrict_to`` gives, per record, an interconnection the match must lie in.
    A back-to-back DC station is two substations a kilometre apart in BE but a
    single site in HIFLD, so without the restriction both ends of such a tie land
    on the same bus and the link collapses; with it, each end finds the nearest bus
    on its own side of the seam.

    Returns the new bus ids -- ``-1`` where nothing suitable lies within
    ``max_km`` -- and the distance moved.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    new_id = np.full(len(lat), -1, dtype=np.int64)
    distance_km = np.full(len(lat), np.inf)

    if restrict_to is None:
        groups = [(None, np.ones(len(lat), dtype=bool), bus_table)]
    else:
        restrict_to = np.asarray(restrict_to)
        groups = [
            (name, restrict_to == name, bus_table[bus_table.interconnect == name]) for name in np.unique(restrict_to)
        ]

    for name, rows, candidates in groups:
        if not rows.any() or candidates.empty:
            if rows.any():
                logger.warning("No bus in interconnection %s to re-key %d record(s) onto.", name, int(rows.sum()))
            continue
        tree = BallTree(np.c_[np.radians(candidates.lat), np.radians(candidates.lon)], metric="haversine")
        distance, index = tree.query(np.c_[np.radians(lat[rows]), np.radians(lon[rows])], k=1)
        km = distance[:, 0] * EARTH_RADIUS_KM
        distance_km[rows] = km
        new_id[rows] = np.where(km <= max_km, candidates.index.to_numpy()[index[:, 0]], -1)
    return new_id, distance_km


def be_offshore_poi_sites(be_bus2sub, be_sub, be_branch):
    """The BE substations ``build_base_network`` uses as offshore points of interconnection.

    Those are the onshore ends of BE's lines from its offshore substations
    (``identify_osw_poi``) plus :data:`TEXAS_POI_SUB_IDS` (``assign_texas_poi``).
    Returns ``sub_id``, ``name``, ``lat``, ``lon``, ``interconnect`` and ``source``.
    """
    sub_of = be_bus2sub.set_index("bus_id").sub_id
    lines = be_branch.query("branch_device_type == 'Line'")
    from_sub = lines.from_bus_id.map(sub_of)
    to_sub = lines.to_bus_id.map(sub_of)
    landfall = to_sub[(from_sub >= BE_OFFSHORE_SUB_ID_MIN) & (to_sub < BE_OFFSHORE_SUB_ID_MIN)].unique()

    sites = pd.concat(
        [
            pd.DataFrame({"sub_id": landfall, "source": "be_offshore_line"}),
            pd.DataFrame({"sub_id": TEXAS_POI_SUB_IDS, "source": "texas_hardcoded"}),
        ],
        ignore_index=True,
    ).drop_duplicates("sub_id")
    sites = sites.merge(be_sub[["sub_id", "name", "lat", "lon", "interconnect"]], on="sub_id", how="left")
    return sites[["sub_id", "name", "lat", "lon", "interconnect", "source"]]


def build_sites(ac):
    """Cluster line endpoints into substation sites and drop the station jumpers."""
    site = cluster_sites(np.concatenate([ac.lon1, ac.lon2]), np.concatenate([ac.lat1, ac.lat2]))
    bus0, bus1 = site[: len(ac)], site[len(ac) :]
    site_xy = (
        pd.DataFrame(
            {
                "site": site,
                "lon": np.concatenate([ac.lon1, ac.lon2]),
                "lat": np.concatenate([ac.lat1, ac.lat2]),
            },
        )
        .groupby("site")[["lon", "lat"]]
        .mean()
    )

    self_loop = bus0 == bus1
    logger.info(
        "Clustered %d endpoint(s) into %d site(s) at %.2f km; dropping %d self-loop record(s) (%.0f km of station jumpers).",
        2 * len(ac),
        len(site_xy),
        SITE_RADIUS_KM,
        int(self_loop.sum()),
        ac.route_km.to_numpy()[self_loop].sum(),
    )
    return ac[~self_loop].reset_index(drop=True), bus0[~self_loop], bus1[~self_loop], site_xy


def build_branches(ac, bus0, bus1, buses, bus_of, interconnect, reference, transformer_reference):
    """Synthesise line and transformer electrical parameters onto HIFLD's geometry.

    Impedance is built from HIFLD's *surveyed route* length, which is what the
    conductor actually measures. ``build_base_network`` separately sets
    ``Line.length`` to the great-circle distance between the two substations, which
    is 12% shorter over the whole grid; that figure drives per-km cost and
    clustering, not the electrical parameters, and nothing downstream recomputes
    r/x/b from it.
    """
    km = ac.route_km.to_numpy()
    kv = ac.cls.to_numpy().astype(float)
    per_km = {field: np.array([reference[c][field] for c in ac.cls]) for field in ("r_ohm_km", "x_ohm_km", "b_s_km")}

    lines = pd.DataFrame(
        {
            "from_bus_id": bus_of.loc[bus0 * 1000 + ac.cls.to_numpy()].to_numpy(),
            "to_bus_id": bus_of.loc[bus1 * 1000 + ac.cls.to_numpy()].to_numpy(),
            # back to per-unit on a 100 MVA base, which is what BE's schema carries
            # and what add_branches_from_file expects
            "r": per_km["r_ohm_km"] * km / (kv**2 / 100),
            "x": per_km["x_ohm_km"] * km / (kv**2 / 100),
            "b": per_km["b_s_km"] * km * (kv**2 / 100),
            "rateA": np.array([reference[c]["mva"] for c in ac.cls]),
            "branch_device_type": "Line",
            "interconnect": interconnect[bus0],
        },
    )

    # One transformer per adjacent pair of voltage classes present at a site, so a
    # 69/138/345 kV site gets 69-138 and 138-345 rather than a full mesh.
    multi = buses[buses.site.isin(buses.site.value_counts().loc[lambda s: s > 1].index)]
    rows = []
    for site_id, group in multi.sort_values("cls").groupby("site"):
        record = group.to_dict("records")
        for low, high in itertools.pairwise(record):
            rows.append((site_id, low["bus_id"], high["bus_id"], low["cls"], high["cls"], low["interconnect"]))
    transformers = pd.DataFrame(rows, columns=["site", "from_bus_id", "to_bus_id", "c_lo", "c_hi", "interconnect"])

    def lookup(row, field, fallback):
        entry = transformer_reference.get((row.c_lo, row.c_hi))
        return entry[field] if entry else fallback

    # A transformer must at least pass what the lines on its low side can deliver,
    # otherwise the site becomes an artificial bottleneck that HIFLD never implied.
    low_side = lines.groupby("from_bus_id").rateA.sum().add(lines.groupby("to_bus_id").rateA.sum(), fill_value=0)
    transformers["r"] = [lookup(row, "r", 0.0005) for row in transformers.itertuples()]
    transformers["x"] = [lookup(row, "x", 0.02) for row in transformers.itertuples()]
    transformers["b"] = 0.0
    transformers["rateA"] = np.maximum(
        [lookup(row, "mva", 500.0) for row in transformers.itertuples()],
        transformers.from_bus_id.map(low_side).fillna(0.0).to_numpy(),
    )
    transformers["branch_device_type"] = "Transformer"
    logger.info("Transformers: %d at %d multi-voltage site(s).", len(transformers), transformers.site.nunique())

    branch = pd.concat([lines, transformers.drop(columns=["site", "c_lo", "c_hi"])], ignore_index=True)
    for column, value in (
        ("rateB", 0.0),
        ("rateC", 0.0),
        ("ratio", 1.0),
        ("angle", 0.0),
        ("status", 1),
        ("angmin", 0.0),
        ("angmax", 0.0),
        ("Pf", 0.0),
        ("Qf", 0.0),
        ("Pt", 0.0),
        ("Qt", 0.0),
        ("mu_Sf", 0.0),
        ("mu_St", 0.0),
        ("mu_angmin", 0.0),
        ("mu_angmax", 0.0),
    ):
        branch[column] = value
    branch.index.name = "branch_id"
    return branch


def main(snakemake):
    lines = reduce_lines(gpd.read_file(snakemake.input.hifld_lines))
    ac = lines.query("dc == 0").reset_index(drop=True)
    logger.info("Read %d HIFLD line record(s), %d of them AC.", len(lines), len(ac))

    ac, bus0, bus1, site_xy = build_sites(ac)

    # ---- United States only ------------------------------------------------
    # Done before anything else looks at the graph: a spur that runs out of the
    # country and back would otherwise carry votes across the interconnection
    # labelling below, and would hold two US fragments together through a foreign
    # bus. A line goes only if *both* its ends are in the country.
    in_us = in_united_states(
        site_xy,
        gpd.read_file(snakemake.input.state_shapes),
        load_na_shapes(countries=["CA", "MX"]),
    )
    keep = in_us[bus0] & in_us[bus1]
    logger.info(
        "Dropping %d site(s) outside the United States (AK / HI / PR / GU and the Canadian "
        "and Mexican spurs) and the %d line(s) (%.0f km) with an end among them.",
        int((~in_us).sum()),
        int((~keep).sum()),
        ac.route_km.to_numpy()[~keep].sum(),
    )
    ac, bus0, bus1 = ac[keep].reset_index(drop=True), bus0[keep], bus1[keep]

    # ---- interconnections and the seam ------------------------------------
    be_substations = pd.read_csv(snakemake.input.be_sub)
    interconnect, crosses_seam = assign_interconnects(site_xy, bus0, bus1, be_substations)
    logger.info(
        "Cutting %d AC line(s) (%.0f km) that bridge two interconnections; on the ground these are DC ties.",
        int(crosses_seam.sum()),
        ac.route_km.to_numpy()[crosses_seam].sum(),
    )
    ac, bus0, bus1 = ac[~crosses_seam].reset_index(drop=True), bus0[~crosses_seam], bus1[~crosses_seam]

    keep_site = largest_component_per_interconnect(len(site_xy), bus0, bus1, interconnect)
    keep = keep_site[bus0] & keep_site[bus1]
    logger.info(
        "Dropping %d further line(s) (%.0f km) left in disconnected fragments.",
        int((~keep).sum()),
        ac.route_km.to_numpy()[~keep].sum(),
    )
    ac, bus0, bus1 = ac[keep].reset_index(drop=True), bus0[keep], bus1[keep]

    # renumber the surviving sites densely
    surviving = np.unique(np.concatenate([bus0, bus1]))
    renumber = np.full(len(site_xy), -1)
    renumber[surviving] = np.arange(len(surviving))
    bus0, bus1 = renumber[bus0], renumber[bus1]
    site_xy = site_xy.iloc[surviving].reset_index(drop=True)
    interconnect = interconnect[surviving]
    logger.info("Final grid: %d site(s), %d AC line(s), %.0f route-km.", len(site_xy), len(ac), ac.route_km.sum())

    # ---- buses: one per (site, voltage class) -----------------------------
    ac["cls"] = class_of(ac.kv)
    buses = (
        pd.DataFrame({"site": np.concatenate([bus0, bus1]), "cls": np.concatenate([ac.cls, ac.cls])})
        .drop_duplicates()
        .sort_values(["site", "cls"])
        .reset_index(drop=True)
    )
    buses["bus_id"] = BUS_ID_OFFSET + np.arange(len(buses))
    buses["sub_id"] = SUB_ID_OFFSET + buses.site
    buses["lon"] = site_xy.lon.to_numpy()[buses.site]
    buses["lat"] = site_xy.lat.to_numpy()[buses.site]
    buses["interconnect"] = interconnect[buses.site]
    # (site, class) -> bus id, keyed on a single integer so the 80k lookups below
    # are a vectorised map rather than a MultiIndex scan
    bus_key = buses.site.to_numpy() * 1000 + buses.cls.to_numpy()
    bus_of = pd.Series(buses.bus_id.to_numpy(), index=bus_key)
    logger.info("Buses: %d over %d site(s).", len(buses), buses.site.nunique())

    # ---- electrical parameters borrowed from BE ---------------------------
    be_bus = pd.read_csv(snakemake.input.be_bus, index_col=0)
    be_bus2sub = pd.read_csv(snakemake.input.be_bus2sub).set_index("bus_id")
    be_sub = be_substations.set_index("sub_id")
    be_bus["lat"] = be_bus2sub.sub_id.map(be_sub.lat)
    be_bus["lon"] = be_bus2sub.sub_id.map(be_sub.lon)
    be_branch = pd.read_csv(snakemake.input.be_branch, index_col=0)

    reference = class_reference_parameters(be_bus, be_branch)
    for level, values in reference.items():
        logger.info(
            "%3d kV reference: r %.4f ohm/km, x %.4f ohm/km, b %.3e S/km, %.0f MVA per circuit",
            level,
            values["r_ohm_km"],
            values["x_ohm_km"],
            values["b_s_km"],
            values["mva"],
        )
    branch = build_branches(
        ac,
        bus0,
        bus1,
        buses,
        bus_of,
        interconnect,
        reference,
        transformer_reference_parameters(be_bus, be_branch),
    )

    # ---- load allocation weights from population --------------------------
    site_population = population_weights(
        site_xy,
        gpd.read_file(snakemake.input.county_shapes),
        load_county_population(snakemake.input.county_population),
    )
    logger.info(
        "Population allocated to sites: %.0f in total; %d site(s) collected none.",
        site_population.sum(),
        int((site_population == 0).sum()),
    )
    # Load taps off the transmission system at the lowest voltage present at a
    # site, so the whole of a site's weight goes on its lowest-voltage bus. Only
    # the within-state ratio of Pd is ever used downstream, never its magnitude.
    buses["Pd"] = 0.0
    lowest = buses.groupby("site").bus_id.idxmin()
    buses.loc[lowest, "Pd"] = site_population[buses.loc[lowest, "site"].to_numpy()]

    # ---- BE-schema output -------------------------------------------------
    bus_out = pd.DataFrame(
        {
            "bus_id": buses.bus_id,
            "type": 1,
            "Pd": buses.Pd,
            "Qd": 0.0,
            "Gs": 0.0,
            "Bs": 0.0,
            "zone_id": 1,
            "Vm": 1.0,
            "Va": 0.0,
            "baseKV": buses.cls.astype(float),
            "loss_zone": 1,
            "Vmax": 1.1,
            "Vmin": 0.9,
            "lam_P": 0.0,
            "lam_Q": 0.0,
            "mu_Vmax": 0.0,
            "mu_Vmin": 0.0,
            "interconnect": buses.interconnect,
        },
    ).set_index("bus_id")

    sub_out = (
        buses.groupby("sub_id")
        .agg(lat=("lat", "first"), lon=("lon", "first"), interconnect=("interconnect", "first"))
        .reset_index()
    )
    sub_out["name"] = "HIFLD " + sub_out.sub_id.astype(str)
    sub_out["interconnect_sub_id"] = sub_out.groupby("interconnect").cumcount() + 1
    sub_out = sub_out[["sub_id", "name", "interconnect_sub_id", "lat", "lon", "interconnect"]]

    located = buses.set_index("bus_id")[["lat", "lon", "interconnect"]]

    # ---- DC links, re-keyed by coordinate ---------------------------------
    be_dcline = pd.read_csv(snakemake.input.be_dcline, index_col=0)
    new_from, d_from = rekey_by_coordinate(
        be_dcline.from_bus_id.map(be_bus.lat),
        be_dcline.from_bus_id.map(be_bus.lon),
        located,
        MAX_REKEY_KM["dcline"],
        restrict_to=be_dcline.from_interconnect,
    )
    new_to, d_to = rekey_by_coordinate(
        be_dcline.to_bus_id.map(be_bus.lat),
        be_dcline.to_bus_id.map(be_bus.lon),
        located,
        MAX_REKEY_KM["dcline"],
        restrict_to=be_dcline.to_interconnect,
    )
    dcline_out = be_dcline.assign(from_bus_id=new_from, to_bus_id=new_to)
    dcline_out = dcline_out[(new_from >= 0) & (new_to >= 0) & (new_from != new_to)]
    dcline_out.index.name = "dcline_id"
    logger.info(
        "DC links: kept %d of %d BE links, median end moved %.1f km.",
        len(dcline_out),
        len(be_dcline),
        float(np.median(np.r_[d_from, d_to])),
    )

    # ---- hydro plants, re-keyed by coordinate ------------------------------
    be_plant = pd.read_csv(snakemake.input.be_plant, index_col=0)
    new_bus, d_plant = rekey_by_coordinate(
        be_plant.bus_id.map(be_bus.lat),
        be_plant.bus_id.map(be_bus.lon),
        located,
        MAX_REKEY_KM["plant"],
    )
    plant_out = be_plant.assign(bus_id=new_bus)
    plant_out = plant_out[new_bus >= 0]
    logger.info(
        "Plants: re-keyed %d of %d BE records, median move %.1f km.",
        len(plant_out),
        len(be_plant),
        float(np.median(d_plant)),
    )

    # ---- offshore points of interconnection --------------------------------
    poi = be_offshore_poi_sites(be_bus2sub.reset_index(), be_substations, be_branch)
    poi_bus, d_poi = rekey_by_coordinate(poi.lat, poi.lon, located, MAX_REKEY_KM["poi"])
    poi_out = poi.assign(bus_id=poi_bus, distance_km=d_poi.round(2))
    poi_out = poi_out[poi_out.bus_id >= 0]
    poi_out["sub_id"] = poi_out.bus_id.map(buses.set_index("bus_id").sub_id)
    logger.info(
        "Offshore POI: matched %d of %d BE sites, median move %.1f km.",
        len(poi_out),
        len(poi),
        float(np.median(d_poi)),
    )

    bus_out.to_csv(snakemake.output.bus)
    branch.to_csv(snakemake.output.branch)
    buses[["bus_id", "sub_id", "interconnect"]].to_csv(snakemake.output.bus2sub, index=False)
    sub_out.to_csv(snakemake.output.sub, index=False)
    dcline_out.to_csv(snakemake.output.dcline)
    plant_out.to_csv(snakemake.output.plant)
    poi_out[["sub_id", "bus_id", "lat", "lon", "interconnect", "source", "distance_km"]].to_csv(
        snakemake.output.poi,
        index=False,
    )

    inventory = ac.groupby("cls").agg(n_lines=("id", "size"), route_km=("route_km", "sum"))
    inventory["TW_km"] = inventory.route_km * [reference[c]["mva"] for c in inventory.index] / 1e6
    logger.info("Final inventory by voltage class:\n%s", inventory.to_string())


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_hifld_grid")
    configure_logging(snakemake)
    main(snakemake)
