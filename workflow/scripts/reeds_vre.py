# By PyPSA-USA Authors
"""
Wind and solar supply-curve sites from NREL ReEDS.

``costs.regional_multipliers.renewable_interconnection`` takes the per-bus
interconnection cost of wind and solar from the ReEDS supply-curve sites and
interconnection tables.

Data
----
* ``sc_<tech>_<access>.csv``: one row per supply-curve site (``sc_point_gid``)
  with its installable ``capacity`` (MW, AC) and mean ``cf``. ``access`` is the
  reV land-access scenario (``reference`` | ``limited`` | ``open``), the same
  scenarios ``renewable_land_access`` selects for GODEEEP.
* ``sitemap.csv``: site coordinates.
* ``interconnection_{land,offshore}.h5``: per-site spur line, POI and network
  reinforcement cost (``cost_total_trans_usd_per_mw``) with its dollar year in the
  ``/data`` group attributes.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from constants import REEDS_DEFAULT_FLOATING_DEPTH_M, REEDS_INTERCONNECTION_COST_FIELD

logger = logging.getLogger(__name__)


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
