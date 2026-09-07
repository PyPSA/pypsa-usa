"""Adds aggregate inter-regional transmission interface limits (RESOLVE/NARIS).

An interface is a bundle of transmission paths between two groups of regions,
capped in aggregate rather than path-by-path. The limits are read from a CSV
with the columns ``interface, region_1, region_2, flow_12, flow_21``, where
``flow_12`` is the MW cap on flow out of ``region_1`` into ``region_2`` and
``flow_21`` the cap on flow in the opposite direction, for example::

    interface,region_1,region_2,flow_12,flow_21,Notes
    CAISO_Imports,"p9, p10, p11","p2, p5, p6, ...",9728,10208,RESOLVE

The caps are applied to the import/export ``Link`` components created by
``external_regions.add_external_regions`` and are therefore a no-op
when ``electricity.imports``/``electricity.exports`` are disabled.
"""

import logging

import pandas as pd
import pypsa
from opts._helpers import get_region_buses

logger = logging.getLogger(__name__)

TRADE_CARRIERS = ("imports", "exports")


def _parse_regions(cell: str) -> list[str]:
    """Split a comma separated region cell into a list of region names."""
    return [region.strip() for region in str(cell).split(",") if region.strip()]


def _boundary_links(
    n: pypsa.Network,
    inside_regions: list[str],
    outside_regions: list[str],
    direction: str,
) -> pd.Index:
    """Get the trade links crossing an interface.

    Links are selected by bus membership and carrier, never by parsing link
    names. Imports run from an external ``{zone}_imports`` bus into a bus inside
    ``inside_regions``; exports run the other way into a ``{zone}_exports`` bus.
    """
    if direction not in TRADE_CARRIERS:
        raise ValueError(f"direction must be either imports or exports; received: {direction}")

    links = n.links[n.links.carrier == direction]
    if links.empty:
        return links.index

    # The external trade buses carry the *outside* zone name in their `country`
    # field, which `get_region_buses` also matches on, so drop them here.
    inside_buses = get_region_buses(n, inside_regions)
    inside_buses = inside_buses[~inside_buses.carrier.isin(TRADE_CARRIERS)]

    external_names = {f"{zone}_{direction}" for zone in outside_regions}
    external_buses = n.buses[
        (n.buses.carrier == direction) & (n.buses.index.isin(external_names) | n.buses.country.isin(outside_regions))
    ]

    if direction == "imports":
        crossing = links.bus0.isin(external_buses.index) & links.bus1.isin(inside_buses.index)
    else:
        crossing = links.bus0.isin(inside_buses.index) & links.bus1.isin(external_buses.index)

    return links[crossing].index


def add_interface_transmission_limits(n: pypsa.Network, limits_csv_path: str) -> None:
    """Cap the aggregate per-snapshot flow across each transmission interface.

    ``flow_21`` limits total imports into ``region_1``, ``flow_12`` total
    exports out of it. Rows without any matching link are skipped.

    Note that only the import/export links are constrained. Region_2 entries
    that are inside the network (e.g. `p8` in a California-only run is itself a
    California zone) contribute no trade links, so internal AC lines such as
    p8-p9 escape the cap. The resulting understatement is documented, not
    corrected here.
    """
    limits = pd.read_csv(limits_csv_path)

    for _, row in limits.iterrows():
        region_1 = _parse_regions(row.region_1)
        region_2 = _parse_regions(row.region_2)

        for direction, cap in (("imports", row.flow_21), ("exports", row.flow_12)):
            if pd.isna(cap):
                continue

            links = _boundary_links(n, region_1, region_2, direction)
            if links.empty:
                logger.info(f"No {direction} links cross interface {row.interface}; skipping limit")
                continue

            lhs = n.model["Link-p"].sel(name=links.tolist()).sum("name")

            n.model.add_constraints(
                lhs <= float(cap),
                name=f"interface_limit-{row.interface}-{direction}",
            )
            logger.info(f"Added {direction} limit of {cap} MW on interface {row.interface} over {len(links)} links")
