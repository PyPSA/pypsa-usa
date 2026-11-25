# By PyPSA-USA Authors
"""
Wind and solar resource from the NREL ReEDS supply curves.

``renewable.dataset: reeds`` builds the per-bus capacity-factor profiles from the
ReEDS hourly CF tables, and ``costs.regional_multipliers.renewable_interconnection``
takes the per-bus interconnection cost from the ReEDS interconnection tables. Both
start from the same supply-curve sites, so both are built here.

Data
----
* ``sc_<tech>_<access>.csv``: one row per supply-curve site (``sc_point_gid``)
  with its installable ``capacity`` (MW, AC) and mean ``cf``. ``access`` is the
  reV land-access scenario (``reference`` | ``limited`` | ``open``), the same
  scenarios ``renewable_land_access`` selects for GODEEEP.
* ``sitemap.csv``: site coordinates.
* ``cf_<tech>_<access>.h5``: one ``cf_profile_<year>`` block per weather year,
  uint16 scaled by 1e4, columns keyed by ``/columns`` (``sc_point_gid``), and a
  matching ``time_index_<year>`` block of timestamps.
* ``interconnection_{land,offshore}.h5``: per-site spur line, POI and network
  reinforcement cost (``cost_total_trans_usd_per_mw``) with its dollar year in the
  ``/data`` group attributes.

Time axis
---------
Model snapshots are naive UTC without daylight saving (``constants.UTC_OFFSET_HOURS``).
The CF tables carry explicit offsets and were checked to be true UTC: solar output
peaks within 0.05 h of solar noon in UTC across every longitude band. The time
stamps are still parsed and converted to UTC, never assumed. If a table were
published at a non-zero offset, the year block is rolled onto the UTC hours of the
weather year by that offset, wrapping the overflow hours, which is how the EER
demand reader rolls its CST data (``build_demand.ReadEer``).

Rows are labelled by position from 1 January 00:00 UTC of the weather year. Leap
weather years hold HOURS_PER_YEAR rows (1 January to 30 December, Feb 29 included), so their
Feb 29 hours land on the 1 March labels and every later hour moves one day. That
is the convention of ``fix_godeeep_time`` (deltas ledger DL-17) and of the
positional EER demand reader, so wind, solar and demand stay on the same hours.
"""

from __future__ import annotations

import calendar
import logging
import re

import numpy as np
import pandas as pd
from constants import (
    HOURS_PER_YEAR,
    REEDS_CF_SCALE,
    REEDS_DEFAULT_FLOATING_DEPTH_M,
    REEDS_INTERCONNECTION_COST_FIELD,
    REEDS_SOLAR_INVERTER_LOADING_RATIO,
)

logger = logging.getLogger(__name__)

#: An explicit UTC offset at the end of an ISO time stamp.
_OFFSET = re.compile(r"(Z|[+-]\d{2}(:?\d{2})?)$")


def interconnection_file_kind(tech: str) -> str:
    """``land`` or ``offshore``: which ReEDS interconnection table covers ``tech``."""
    return "offshore" if tech.startswith("offwind") else "land"


def floating_depth_threshold(tech: str, tech_config: dict) -> float:
    """Depth (m) splitting fixed from floating offshore sites.

    Fixed-bottom ``offwind`` keeps sites at or above its ``max_depth``, floating
    keeps sites below its ``min_depth``. Both are 60 m in config.common.yaml, the
    same split the atlite depth screens use.
    """
    key = "max_depth" if tech == "offwind" else "min_depth"
    return float(tech_config.get(key) or REEDS_DEFAULT_FLOATING_DEPTH_M)


def load_supply_curve(sc_path: str, sitemap_path: str) -> pd.DataFrame:
    """Supply-curve sites with coordinates: ``sc_point_gid, capacity, cf, latitude, longitude``."""
    supply_curve = pd.read_csv(sc_path)
    sitemap = pd.read_csv(sitemap_path)
    sites = supply_curve.merge(sitemap, on="sc_point_gid", how="inner", validate="one_to_one")
    dropped = len(supply_curve) - len(sites)
    if dropped:
        logger.warning("%d ReEDS supply-curve sites have no coordinates in %s; dropped.", dropped, sitemap_path)
    return sites[sites["capacity"] > 0].reset_index(drop=True)


def load_interconnection_costs(path: str, tech: str, dollar_year: int) -> pd.DataFrame:
    """Per-site interconnection capital cost in USD/MW of ``dollar_year``.

    Returns a frame indexed by ``sc_point_gid`` with ``interconnection_cost``
    (spur line + POI + network reinforcement, overnight) and ``dist_spur_km``.
    The table's own dollar year is read from its ``/data`` attributes, so a
    re-published table in other dollars cannot slip through unconverted.
    """
    import tables
    from money import to_dollars

    field = REEDS_INTERCONNECTION_COST_FIELD[tech]
    with tables.open_file(path, "r") as h5:
        attrs = h5.get_node("/data")._v_attrs
        if "dollaryear" not in attrs._f_list():
            raise ValueError(f"{path} does not state its dollar year (/data attribute 'dollaryear').")
        table_year = int(attrs["dollaryear"])
        gids = h5.get_node("/data/sc_point_gid")[:].astype(np.int64)
        cost = h5.get_node(f"/data/{field}")[:].astype(float)
        spur = h5.get_node("/data/dist_spur_km")[:].astype(float)
    costs = pd.DataFrame(
        {
            "interconnection_cost": to_dollars(cost, year=table_year, to_year=int(dollar_year)),
            "dist_spur_km": spur,
        },
        index=pd.Index(gids, name="sc_point_gid"),
    )
    logger.info(
        "ReEDS %s interconnection cost (%s, %d USD -> %d USD): median %.0f USD/MW.",
        tech,
        field,
        table_year,
        int(dollar_year),
        costs["interconnection_cost"].median(),
    )
    return costs


def split_offshore_by_depth(sites: pd.DataFrame, gebco_path: str, floating: bool, threshold_m: float) -> pd.DataFrame:
    """Keep the fixed-bottom (depth <= threshold) or floating (deeper) offshore sites.

    ReEDS publishes one offshore curve for both. Sites off the GEBCO raster or on
    its nodata count as floating: they lie beyond the raster's western edge, in
    deep Pacific water.
    """
    import rasterio

    with rasterio.open(gebco_path) as src:
        west_edge = src.bounds.left
        elevation = np.array([v[0] for v in src.sample(zip(sites.longitude, sites.latitude))], dtype=float)
        nodata = src.nodata
    outside = (sites.longitude.to_numpy() < west_edge) | (elevation == nodata) | (np.abs(elevation) > 1e4)
    is_floating = np.where(outside, True, -elevation > threshold_m)
    return sites.loc[is_floating if floating else ~is_floating].reset_index(drop=True)


def map_sites_to_regions(sites: pd.DataFrame, regions, tech: str = "") -> pd.DataFrame:
    """Assign each site the bus whose region polygon contains it.

    ``regions`` is the GeoDataFrame of bus regions indexed by bus name. Sites in
    no region (outside the modelled footprint) are dropped and their capacity is
    logged.
    """
    import geopandas as gpd

    points = gpd.GeoDataFrame(sites, geometry=gpd.points_from_xy(sites.longitude, sites.latitude), crs=4326)
    polygons = regions.to_crs(4326)[["geometry"]].rename_axis("bus").reset_index()
    joined = gpd.sjoin(points, polygons, how="inner", predicate="within")
    # a site on a shared boundary would otherwise be counted twice
    joined = joined[~joined.index.duplicated(keep="first")]
    logger.info(
        "ReEDS %s: %d of %d supply-curve sites (%.0f of %.0f MW) fall in the %d bus regions.",
        tech,
        len(joined),
        len(sites),
        joined["capacity"].sum(),
        sites["capacity"].sum(),
        joined["bus"].nunique(),
    )
    out = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))
    out["bus"] = out["bus"].astype(str)
    return out.reset_index(drop=True)


def bus_weighted_mean(sites: pd.DataFrame, column: str) -> pd.Series:
    """Capacity-weighted mean of ``column`` per bus over the sites where it is known."""
    known = sites[np.isfinite(sites[column].to_numpy(dtype=float))]
    weight = known["capacity"]
    total = weight.groupby(known["bus"]).sum()
    return (known[column] * weight).groupby(known["bus"]).sum() / total.where(total > 0)


def _parse_utc(raw) -> pd.DatetimeIndex:
    """Decode ReEDS time stamps and convert them to naive UTC.

    Every stamp must carry an explicit UTC offset: a naive stamp could be local
    time, and guessing would shift every profile by hours.
    """
    text = [v.decode() if isinstance(v, bytes) else str(v) for v in raw]
    naive = [t for t in text if not _OFFSET.search(t.strip())]
    if naive:
        raise ValueError(
            f"{len(naive)} ReEDS time stamps carry no UTC offset (first: {naive[0]!r}); "
            "they cannot be placed on the model's UTC snapshots.",
        )
    stamps = pd.DatetimeIndex(pd.to_datetime(text, utc=True, format="ISO8601"))
    return stamps.tz_convert("UTC").tz_localize(None)


def utc_year_axis(stamps: pd.DatetimeIndex, weather_year: int, n_rows: int) -> tuple[pd.DatetimeIndex, int, np.ndarray]:
    """Place one weather-year block of hourly rows on the UTC calendar of that year.

    Parameters
    ----------
    stamps
        The block's time stamps, naive UTC, one per row.
    weather_year
        The weather year the block belongs to.
    n_rows
        Rows in the CF block (must equal ``len(stamps)``).

    Returns
    -------
    labels : DatetimeIndex
        Naive UTC labels of the kept rows, on a 365-day calendar (no Feb 29).
    shift : int
        Hours to roll the rows (``np.roll(rows, shift, axis=0)``) so that row ``j``
        holds UTC hour ``j`` of the year. Zero for tables published in UTC.
    keep : ndarray of bool
        Mask over the rolled rows; drops 31 December when the block is a full leap
        year, which no snapshot reads (``_helpers.weather_year_hours``).
    """
    year = int(weather_year)
    if len(stamps) != n_rows:
        raise ValueError(f"ReEDS {year}: {len(stamps)} time stamps for {n_rows} profile rows.")
    if not ((stamps[1:] - stamps[:-1]) == pd.Timedelta(hours=1)).all():
        raise ValueError(f"ReEDS {year} time stamps are not one contiguous hourly series.")

    offset = (stamps[0] - pd.Timestamp(year=year, month=1, day=1)) / pd.Timedelta(hours=1)
    if offset != round(offset) or abs(offset) >= 24:
        raise ValueError(
            f"ReEDS {year} block starts at {stamps[0]} UTC, not within a day of the start of {year}; "
            "it does not belong to that weather year.",
        )
    shift = round(offset)
    if shift:
        logger.warning(
            "ReEDS %d block starts %+d h from 1 January 00:00 UTC; rolling %d hour(s) across the year "
            "boundary onto the UTC snapshots (as the EER demand reader does).",
            year,
            shift,
            abs(shift),
        )

    leap = calendar.isleap(year)
    calendar_hours = pd.date_range(
        pd.Timestamp(year=year, month=1, day=1), periods=HOURS_PER_YEAR + 24 if leap else HOURS_PER_YEAR, freq="h"
    )
    is_feb29 = (calendar_hours.month == 2) & (calendar_hours.day == 29)
    # Positional, as _helpers.weather_year_hours lays every weather input on the
    # model calendar: row i is model hour i, so in a leap year the Feb 29 rows take
    # the 1 March labels and every later row moves one day on.
    if n_rows == HOURS_PER_YEAR:
        return calendar_hours[~is_feb29], shift, np.ones(n_rows, dtype=bool)
    if leap and n_rows == len(calendar_hours):
        # a full leap year: its 31 December is the one day no snapshot reads
        return calendar_hours[~is_feb29], shift, np.arange(n_rows) < HOURS_PER_YEAR
    raise ValueError(
        f"ReEDS {year} block has {n_rows} rows; expected {HOURS_PER_YEAR} (or {HOURS_PER_YEAR + 24} in a leap year)."
    )


def available_weather_years(cf_path: str) -> list[int]:
    """Weather years published in a ReEDS CF table."""
    import tables

    with tables.open_file(cf_path, "r") as h5:
        return sorted(
            int(node._v_name.rsplit("_", 1)[1]) for node in h5.list_nodes("/") if node._v_name.startswith("cf_profile_")
        )


def read_time_axis(cf_path: str, weather_year: int) -> tuple[pd.DatetimeIndex, int, np.ndarray]:
    """:func:`utc_year_axis` for the ``weather_year`` block of a ReEDS CF table."""
    import tables

    year = int(weather_year)
    with tables.open_file(cf_path, "r") as h5:
        if f"/cf_profile_{year}" not in h5 or f"/time_index_{year}" not in h5:
            raise ValueError(
                f"{cf_path} has no weather year {year}; renewable.dataset: reeds supports "
                f"{available_weather_years(cf_path)}.",
            )
        raw = h5.get_node(f"/time_index_{year}")[:]
        n_rows = h5.get_node(f"/cf_profile_{year}").shape[0]
    return utc_year_axis(_parse_utc(raw), year, n_rows)


def cf_table_gids(cf_path: str) -> pd.Index:
    """``sc_point_gid`` of every column of a ReEDS CF table."""
    import tables

    with tables.open_file(cf_path, "r") as h5:
        return pd.Index(h5.get_node("/columns")[:].astype(np.int64))


def solar_ac_transform(block: np.ndarray) -> np.ndarray:
    """Per-site UPV CF from MW_AC/MW_DC to MW_AC/MW_AC.

    Multiplies by ``constants.REEDS_SOLAR_INVERTER_LOADING_RATIO`` and clips at 1
    (output cannot exceed the inverter rating). The clip is non-linear, so it acts
    on site CFs, before any aggregation.
    """
    return np.minimum(block * REEDS_SOLAR_INVERTER_LOADING_RATIO, 1.0)


def aggregate_cf_rows(
    cf_path: str,
    weather_year: int,
    sites: pd.DataFrame,
    buses: pd.Index,
    transform=None,
    row_block: int = 2048,
) -> np.ndarray:
    """Capacity-weighted mean CF per bus for every row of one weather-year block.

    ``sites`` needs ``sc_point_gid``, ``bus`` and ``capacity``; every site must be a
    column of the table and every ``bus`` must be in ``buses``. Rows are read in
    blocks and collapsed through a sparse (site x bus) weight matrix, so peak
    memory is one block, not the full (hours x sites) table. ``transform`` acts on
    site-level CFs before averaging. Returns (rows x buses), in the table's row
    order.
    """
    import tables
    from scipy import sparse

    buses = pd.Index(buses)
    bus_pos = buses.get_indexer(sites["bus"])
    if (bus_pos < 0).any():
        raise ValueError("Every site's bus must be one of `buses`.")
    capacity = sites["capacity"].to_numpy(dtype=float)
    bus_capacity = np.bincount(bus_pos, weights=capacity, minlength=len(buses))
    if (bus_capacity[bus_pos] <= 0).any():
        raise ValueError("Every site must carry positive capacity.")
    weights = sparse.csc_matrix(
        (capacity / bus_capacity[bus_pos], (bus_pos, np.arange(len(sites)))),
        shape=(len(buses), len(sites)),
    )

    with tables.open_file(cf_path, "r") as h5:
        columns = pd.Index(h5.get_node("/columns")[:].astype(np.int64)).get_indexer(sites["sc_point_gid"])
        if (columns < 0).any():
            raise ValueError(f"{int((columns < 0).sum())} sites are not columns of {cf_path}.")
        node = h5.get_node(f"/cf_profile_{int(weather_year)}")
        out = np.empty((node.shape[0], len(buses)), dtype=np.float32)
        for start in range(0, node.shape[0], row_block):
            stop = min(start + row_block, node.shape[0])
            block = node[start:stop, :][:, columns].astype(np.float64) * REEDS_CF_SCALE
            if transform is not None:
                block = transform(block)
            out[start:stop] = (weights @ block.T).T
    return out


def bus_profiles(
    cf_path: str,
    weather_year: int,
    sites: pd.DataFrame,
    buses: pd.Index,
    transform=None,
) -> pd.DataFrame:
    """Hourly per-bus CF on naive UTC labels of the weather year (365-day calendar)."""
    labels, shift, keep = read_time_axis(cf_path, weather_year)
    rows = aggregate_cf_rows(cf_path, weather_year, sites, buses, transform=transform)
    rows = np.roll(rows, shift, axis=0)[keep]
    return pd.DataFrame(rows, index=labels.rename("time"), columns=pd.Index(buses, name="bus"))
