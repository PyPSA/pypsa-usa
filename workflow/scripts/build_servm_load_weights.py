# BY PyPSA-USA Authors
"""Fractional load-allocation factors mapping SERVM load regions onto clustered buses.

The CPUC SERVM demand dataset is published for six California load regions (IID,
LADWP, NCNC, PGE, SCE, SDGE). Attaching it to the model needs, per region, the
share of that region's demand that lands on each bus of the simplified network.

The share cannot be read off the network the demand is attached to:
``aggregate_to_substations`` drops ``balancing_area`` from the buses, and a single
substation/cluster bus may straddle two SERVM regions anyway (LA County holds both
LDWP and CISO-SCE buses). Both facts are handled by going back to
``elec_base_network.nc`` — the last network that still carries ``balancing_area``
and the population-based ``load_weight`` per bus — and composing the two busmaps
(base bus -> substation -> ``{simpl}`` cluster bus) to find where each base bus's
weight ends up. A straddling cluster therefore appears once per SERVM region it
overlaps, each row carrying only that region's share.

The resulting long-format table (``bus``, ``servm_region``, ``laf``) sums to 1.0
within every region, so it is scale-free: it disaggregates whatever regional
demand the downstream rule reads.
"""

import logging

import pandas as pd
import pypsa
from _helpers import configure_logging

logger = logging.getLogger(__name__)

SERVM_REGIONS = ("IID", "LADWP", "NCNC", "PGE", "SCE", "SDGE")


def load_servm_region_map(path: str) -> pd.Series:
    """
    Balancing-area -> SERVM region lookup read from ``servm_region_map.csv``.

    Balancing areas that are deliberately out of scope (CISO-VEA, whose footprint
    is in Nevada) keep their row with a null region rather than being dropped, so
    that :func:`build_load_weights` can tell a known-but-excluded balancing area
    from one that upstream relabeling has made unknown.
    """
    df = pd.read_csv(path, dtype=str)
    ba = df["balancing_area"].str.strip()
    region = df["servm_region"].str.strip().replace("", None)

    duplicated = ba[ba.duplicated()].unique()
    if len(duplicated):
        raise ValueError(f"Duplicate balancing areas in {path}: {list(duplicated)}")

    return pd.Series(region.to_numpy(), index=pd.Index(ba, name="balancing_area"), name="servm_region")


def compose_busmaps(busmap_b: pd.Series, busmap_s: pd.Series) -> pd.Series:
    """
    Chain the base-bus -> substation and substation -> cluster busmaps.

    ``busmap_b`` is written by ``aggregate_to_substations`` (index: base network
    bus, value: ``sub_id``) and ``busmap_s`` by ``cluster_simpl`` (index:
    ``sub_id``, value: ``cluster_bus``). ``cluster_simpl`` runs for every
    ``{simpl}`` value, emitting an identity map when the wildcard is empty, so the
    second leg is always available. Substation ids round-trip through CSV as
    integers, hence the explicit string normalization on both keys and values.

    Base buses whose substation is absent from ``busmap_s`` map to NaN; whether
    that is fatal depends on whether they carry demand weight, which is decided in
    :func:`build_load_weights`.
    """
    busmap_b = busmap_b.astype(str)
    busmap_b.index = busmap_b.index.astype(str)
    busmap_s = busmap_s.astype(str)
    busmap_s.index = busmap_s.index.astype(str)

    composed = busmap_b.map(busmap_s)
    composed.name = "cluster_bus"
    composed.index.name = "bus"

    unmapped = int(composed.isna().sum())
    if unmapped:
        logger.info(
            "%d of %d base buses have no cluster bus (substation missing from the simpl busmap).",
            unmapped,
            len(composed),
        )
    return composed


def build_load_weights(
    buses: pd.DataFrame,
    region_map: pd.Series,
    busmap: pd.Series,
) -> pd.DataFrame:
    """
    Per-(SERVM region, cluster bus) load-allocation factors in long format.

    ``buses`` are the base-network buses, which must still carry
    ``balancing_area`` and ``load_weight``. Returns columns ``bus``,
    ``servm_region``, ``laf`` with ``laf`` summing to 1.0 within each region.
    """
    missing_cols = {"balancing_area", "load_weight"}.difference(buses.columns)
    if missing_cols:
        raise ValueError(
            f"Base network buses are missing {sorted(missing_cols)}. SERVM weights must be built from "
            "elec_base_network.nc, the last network that carries the balancing area and load weight.",
        )

    # the index name is dropped so the added "bus" column cannot collide with it
    df = pd.DataFrame(
        {
            "balancing_area": buses.balancing_area.fillna("").astype(str).str.strip(),
            "load_weight": pd.to_numeric(buses.load_weight, errors="coerce").fillna(0.0),
        },
    ).rename_axis(None)
    weighted = df.load_weight > 0

    blank_ba = weighted & (df.balancing_area == "")
    if blank_ba.any():
        offenders = df.index[blank_ba]
        raise ValueError(
            f"{blank_ba.sum()} buses carry demand weight but have no balancing area, so their load "
            f"cannot be assigned to a SERVM region: {list(offenders[:10])}",
        )

    unknown_ba = weighted & ~df.balancing_area.isin(region_map.index)
    if unknown_ba.any():
        raise ValueError(
            f"Balancing areas carrying demand weight are absent from the SERVM region map: "
            f"{sorted(df.balancing_area[unknown_ba].unique())}. Add them to "
            "repo_data/CPUC/servm_region_map.csv (with an empty region to exclude them).",
        )

    excluded_bas = region_map.index[region_map.isna()]
    excluded = df.balancing_area.isin(excluded_bas) & weighted
    total_weight = df.load_weight.sum()
    if excluded.any() and total_weight > 0:
        logger.info(
            "Dropping %d buses in balancing areas excluded from SERVM (%s), carrying %.3f%% of total bus load weight.",
            int(excluded.sum()),
            ", ".join(sorted(df.balancing_area[excluded].unique())),
            100 * df.load_weight[excluded].sum() / total_weight,
        )

    df["servm_region"] = df.balancing_area.map(region_map)
    df = df[df.servm_region.notna() & weighted].copy()

    df["bus"] = df.index.to_series().astype(str).map(busmap)
    unmapped = df.bus.isna()
    if unmapped.any():
        offenders = df.index[unmapped]
        raise ValueError(
            f"{unmapped.sum()} buses carrying SERVM demand weight are absent from the composed busmap: "
            f"{list(offenders[:10])}",
        )

    weights = df.groupby(["servm_region", "bus"], as_index=False).load_weight.sum()
    weights["laf"] = weights.load_weight / weights.groupby("servm_region").load_weight.transform("sum")
    weights = weights[["bus", "servm_region", "laf"]].sort_values(["servm_region", "bus"]).reset_index(drop=True)

    if weights.empty:
        raise ValueError(
            "No bus carries load weight for any SERVM region. Check that the network covers California "
            "and that its balancing areas match repo_data/CPUC/servm_region_map.csv.",
        )

    region_sums = weights.groupby("servm_region").laf.sum()
    off = region_sums[(region_sums - 1.0).abs() > 1e-9]
    if len(off):
        raise ValueError(f"Load allocation factors do not sum to 1 per SERVM region: {off.to_dict()}")

    absent = [r for r in SERVM_REGIONS if r not in region_sums.index]
    if absent:
        logger.warning(
            "No buses found for SERVM regions %s; their demand cannot be allocated in this network.",
            ", ".join(absent),
        )

    return weights


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_servm_load_weights",
            interconnect="western",
            simpl="",
        )
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)

    busmap_b = pd.read_csv(snakemake.input.busmap_b, index_col=0, dtype=str).iloc[:, 0]
    busmap_s = pd.read_csv(snakemake.input.busmap_s, index_col=0, dtype=str).iloc[:, 0]
    busmap = compose_busmaps(busmap_b, busmap_s)

    region_map = load_servm_region_map(snakemake.input.region_map)
    weights = build_load_weights(n.buses, region_map, busmap)

    logger.info(
        "Built %d (SERVM region, bus) allocation weights across %d regions and %d buses.",
        len(weights),
        weights.servm_region.nunique(),
        weights.bus.nunique(),
    )
    weights.to_csv(snakemake.output.weights, index=False)
