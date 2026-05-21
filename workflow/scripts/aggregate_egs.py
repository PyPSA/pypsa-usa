# BY PyPSA-USA Authors
"""Remap precomputed EGS supply-curve data from substation to ``{simpl}`` cluster bus.

The EGS dataset (NREL-derived supply curves shipped in ``data/EGS/{interconnect}/``)
is keyed by ``(sub_id, Quality)`` for specs and ``(sub_id, Quality, year, Date)`` for
profiles. After the simplify-early refactor the network buses are simpl-cluster IDs,
not substation IDs, so the substation-keyed EGS data has to be aggregated through
the ``cluster_simpl`` busmap before ``add_electricity`` can attach it.

Per ``(cluster_bus, Quality)`` group:
* ``avail_capacity_mw`` → sum of constituent substations' capacities
* ``capex_usd_kw``, ``advanced_capex_usd_kw``, ``fixed_om`` → capacity-weighted mean
* ``capacity_factor(year, Date)`` → capacity-weighted mean per timestep

The output schema keeps ``sub_id`` as the dimension name (now containing cluster
bus IDs) so ``attach_egs`` in :mod:`add_electricity` can consume it with minimal
changes.
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr
from _helpers import configure_logging

logger = logging.getLogger(__name__)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    w = weights.sum()
    if w == 0:
        return np.nan
    return float((values * weights).sum() / w)


def aggregate_specs(
    df_specs: pd.DataFrame,
    busmap: pd.Series,
) -> pd.DataFrame:
    """Aggregate (sub_id, Quality) specs to (cluster_bus, Quality)."""
    df = df_specs.copy()
    df["sub_id"] = df["sub_id"].astype(str)
    df["cluster_bus"] = df["sub_id"].map(busmap)
    df = df.dropna(subset=["cluster_bus"])

    def _reduce(group: pd.DataFrame) -> pd.Series:
        caps = group["avail_capacity_mw"]
        return pd.Series(
            {
                "avail_capacity_mw": caps.sum(),
                "capex_usd_kw": _weighted_mean(group["capex_usd_kw"], caps),
                "advanced_capex_usd_kw": _weighted_mean(
                    group["advanced_capex_usd_kw"],
                    caps,
                ),
                "fixed_om": _weighted_mean(group["fixed_om"], caps),
            },
        )

    agg = (
        df.groupby(["cluster_bus", "Quality"], group_keys=False)
        .apply(_reduce)
        .reset_index()
        .rename(columns={"cluster_bus": "sub_id"})
    )
    return agg


def aggregate_profile(
    df_profile: pd.DataFrame,
    specs_caps: pd.Series,
    busmap: pd.Series,
) -> pd.DataFrame:
    """Aggregate (sub_id, Quality, year, Date) profile to cluster bus.

    ``specs_caps`` is the per-(sub_id, Quality) capacity used as the weight.
    """
    df = df_profile.copy()
    df["sub_id"] = df["sub_id"].astype(str)
    df["cluster_bus"] = df["sub_id"].map(busmap)
    df = df.dropna(subset=["cluster_bus"])

    weights = df.set_index(["sub_id", "Quality"]).index.map(specs_caps)
    df["weight"] = pd.Series(weights, index=df.index).fillna(0.0)
    df["weighted_cf"] = df["capacity_factor"] * df["weight"]

    agg = (
        df.groupby(["cluster_bus", "Quality", "year", "Date"], group_keys=False)[["weighted_cf", "weight"]]
        .sum()
        .reset_index()
    )
    nonzero = agg["weight"] > 0
    agg["capacity_factor"] = 0.0
    agg.loc[nonzero, "capacity_factor"] = agg.loc[nonzero, "weighted_cf"] / agg.loc[nonzero, "weight"]

    agg = agg[["cluster_bus", "Quality", "year", "Date", "capacity_factor"]].rename(
        columns={"cluster_bus": "sub_id"},
    )
    return agg


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "aggregate_egs",
            interconnect="texas",
            simpl="50",
        )
    configure_logging(snakemake)

    busmap = pd.read_csv(snakemake.input.busmap, index_col=0, dtype=str)
    busmap.index = busmap.index.astype(str)
    busmap = busmap.iloc[:, 0]
    logger.info(
        "Loaded busmap with %d substations mapping to %d clusters.",
        len(busmap),
        busmap.nunique(),
    )

    with xr.open_dataset(snakemake.input.specs) as ds_specs:
        df_specs = ds_specs.to_dataframe().reset_index().dropna()
    agg_specs = aggregate_specs(df_specs, busmap)
    logger.info(
        "Aggregated specs: %d (cluster_bus, Quality) rows from %d substation rows.",
        len(agg_specs),
        len(df_specs),
    )

    specs_caps = (
        agg_specs.set_index(["sub_id", "Quality"])["avail_capacity_mw"]
        # Use the post-aggregation capacities as weights so re-running the
        # aggregation against the output reproduces it (idempotent).
    )
    # For profile weighting we need the PRE-aggregation per-(sub_id, Quality)
    # capacity. Rebuild it from the original dataframe.
    pre_caps = df_specs.assign(sub_id=df_specs["sub_id"].astype(str)).set_index(["sub_id", "Quality"])[
        "avail_capacity_mw"
    ]

    with xr.open_dataset(snakemake.input.profile) as ds_profile:
        df_profile = ds_profile.to_dataframe().reset_index().dropna()
    agg_profile = aggregate_profile(df_profile, pre_caps, busmap)
    logger.info(
        "Aggregated profile: %d rows (cluster_bus x Quality x time).",
        len(agg_profile),
    )

    out_specs = agg_specs.set_index(["sub_id", "Quality"]).to_xarray()
    out_specs.to_netcdf(snakemake.output.specs)

    out_profile = agg_profile.set_index(
        ["sub_id", "Quality", "year", "Date"],
    )[["capacity_factor"]].to_xarray()
    out_profile.to_netcdf(snakemake.output.profile)
