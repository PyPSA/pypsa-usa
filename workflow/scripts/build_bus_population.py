"""
Census-population demand-allocation weights for base-network buses.

Provides the ``bus_allocation: population`` source for ``n.buses.load_weight``:
2020 Decennial Census (DEC DHC, table P1) county populations are split evenly
across the substations in each county, then evenly across the buses of each
substation. The weight is only ever consumed as a within-zone ratio (LAF_state,
zone->bus demand disaggregation, clustering weights), so its absolute scale
(persons) is irrelevant downstream.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_county_population(csv_path: str) -> pd.Series:
    """
    County population indexed by ReEDS-style county id (``p`` + 5-digit GEOID).

    Parses the data.census.gov export ``DECENNIALDHC2020.P1-Data.csv`` (first
    row is the machine-readable header, second the human-readable one used
    here, matching ``build_population_layouts.load_population``).
    """
    df = pd.read_csv(csv_path, skiprows=1)
    df = df.set_index("Geography")
    # county GEOID is the tail of e.g. "0500000US01001"; buses carry the
    # "p"-prefixed form assigned in build_shapes
    df.index = "p" + df.index.str[-5:]
    df.index.name = "county"
    df = df.rename(columns={x: x.strip() for x in df.columns})
    return df["!!Total"].astype(float)


def assign_load_weight(
    gdf_bus: pd.DataFrame,
    county_population: pd.Series,
) -> pd.Series:
    """
    Per-bus demand-allocation weight from county populations.

    Each county's population is split evenly across its substations and each
    substation's share evenly across its buses, so multi-bus substations do
    not soak up a multiple of a single-bus substation's weight. Buses without
    a county assignment (offshore, unmapped) get weight 0.
    """
    df = gdf_bus[["sub_id", "county"]].copy()
    df = df[df.county.notna() & (df.county != "")]

    missing = df.county[~df.county.isin(county_population.index)].unique()
    if len(missing):
        logger.warning(
            "%d bus counties missing from census population data (weight 0): %s",
            len(missing),
            list(missing[:10]),
        )

    empty_counties = county_population.index.difference(df.county.unique())
    if len(empty_counties):
        share = county_population[empty_counties].sum() / county_population.sum()
        logger.info(
            "%d counties have no substations; their population (%.2f%% of total) "
            "is redistributed within each demand zone by normalization.",
            len(empty_counties),
            100 * share,
        )

    subs_per_county = df.groupby("county")["sub_id"].nunique()
    buses_per_sub = df.groupby("sub_id").size()

    weight = (
        df.county.map(county_population).fillna(0.0) / df.county.map(subs_per_county) / df.sub_id.map(buses_per_sub)
    )
    return weight.reindex(gdf_bus.index, fill_value=0.0)
