# BY PyPSA-USA Authors
"""Benchmark the modelled California fleet against the CPUC Baseline Generator List.

The CPUC publishes, alongside the SERVM hourly load, a unit-level baseline
generator list covering the whole WECC. Its California rows are the reference
fleet the CPUC's own SERVM/RESOLVE runs start from, so comparing installed
capacity against PyPSA-USA's fleet is a direct check on whether the model starts
a California study from a plausible resource mix.

Two axes have to be reconciled before the sides are comparable.

Region
------
The CPUC tags every unit with a ``SERVM Region`` (PGE, SCE, SDGE, IID, LADWP,
NCNC). PyPSA-USA's ``powerplants.csv`` carries ``balancing_authority_code_eia``,
which is the *EIA* balancing-authority code — and EIA reports every CAISO plant
under the single code ``CISO`` with no sub-BA breakdown. There is therefore no
column on the model side that separates PGE from SCE from SDGE, and inventing
one (point-in-polygon against the clustered BA shapes) would need a
``resources/`` artifact that this rule does not consume.

So the benchmark is run at the coarsest region resolution both sides support:

* ``CAISO``  — CPUC PGE + SCE + SDGE  vs. model BA ``CISO`` (the CPUC data
  dictionary defines CAISO as exactly PGE + SCE + SDGE)
* ``LADWP``  — CPUC LADWP            vs. model BA ``LDWP``
* ``IID``    — CPUC IID              vs. model BA ``IID``
* ``NCNC``   — CPUC NCNC             vs. model BAs ``BANC`` + ``TIDC``

The mapping lives in ``repo_data/CPUC/servm_benchmark_regions.csv`` so the
collapse is explicit and editable rather than hard-coded here. The model side is
additionally restricted to ``state == "CA"``: EIA's ``CISO`` also covers the
Valley Electric (CISO-VEA) footprint, which sits in Nevada and is excluded from
the SERVM California regions.

Technology
----------
SERVM splits technology far finer than PyPSA carriers do (nine solar buckets, a
paired/hybrid storage split, a CHP bucket). Both sides are therefore mapped into
a common ``compare_category`` by ``repo_data/CPUC/servm_tech_map.csv``, which
carries a ``side`` column (``cpuc`` / ``pypsa``) so one file documents both
directions. Anything unmapped on either side becomes ``UNMAPPED:<name>`` and is
reported as its own row — a category is never silently dropped, because a silent
drop is exactly how a benchmark starts agreeing with itself.

Two mappings deserve to be flagged when reading the output:

* ``Gas Cogen/CHP`` is CPUC-only. EIA technology descriptions have no CHP
  concept, so California gas cogeneration lands in ``CCGT``/``OCGT`` by prime
  mover. Expect the model to be short in ``Gas Cogen/CHP`` and long in the two
  gas buckets by roughly the same amount.
* ``Demand Response`` and ``Pumping Load`` are CPUC-only resources with no
  PyPSA-USA counterpart; they are kept as rows with ``model_mw == 0``.

Vintage and retirement
----------------------
Both sides are filtered by the same rule (:func:`filter_active`): in service by
December 31 of the horizon, and not retired by then. The model side reproduces
``add_electricity.load_powerplants`` exactly (see
:func:`prepare_model_plants`) so this benchmark measures the fleet the model
actually builds, not a differently-filtered idealisation of it.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns
from _helpers import configure_logging

logger = logging.getLogger(__name__)

sns.set_theme("paper", style="whitegrid")

DPI = 300

CPUC_SHEET = "BaselineGeneratorList"
#: The workbook's first row is blank; the real header is the second one.
CPUC_HEADER_ROW = 1

#: SERVM regions that make up California. Every other ``SERVM Region`` value in
#: the workbook is a non-CA WECC balancing area.
CA_SERVM_REGIONS = ("IID", "LADWP", "NCNC", "PGE", "SCE", "SDGE")

CPUC_CAPACITY_COL = "Capmax MW"
CPUC_REGION_COL = "SERVM Region"
CPUC_TECH_COL = "SERVM Tech Category"
CPUC_INSV_COL = "Insvdt"
CPUC_RETIRE_COL = "RetireDate"

#: Sentinel used by :func:`filter_active` for "no retirement date on record".
NEVER_RETIRES = pd.Timestamp("2262-01-01")

UNMAPPED_PREFIX = "UNMAPPED:"

#: Pseudo-region prefix for :func:`reconcile_totals`' informational rows.
RECONCILE_PREFIX = "RECONCILE:"


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #


def read_cpuc_baseline(xlsx_path: str) -> pd.DataFrame:
    """
    Read the California rows of the CPUC Baseline Generator List.

    The workbook opens with a blank row, so the header is row 2 (``header=1``).
    The published sheet covers all of WECC; only units whose ``SERVM Region`` is
    one of :data:`CA_SERVM_REGIONS` are kept.

    ``Capmax MW`` is the nameplate capacity of the unit as SERVM sees it. Paired
    and hybrid resources are published as separate rows per component (a
    ``Paired_Solar_1Axis`` row and a ``Paired_BattStorage`` row), so summing
    ``Capmax MW`` by technology counts each component once and never
    double-counts the point of interconnection.
    """
    df = pd.read_excel(xlsx_path, sheet_name=CPUC_SHEET, header=CPUC_HEADER_ROW)

    missing = [
        c
        for c in (CPUC_CAPACITY_COL, CPUC_REGION_COL, CPUC_TECH_COL, CPUC_INSV_COL, CPUC_RETIRE_COL)
        if c not in df.columns
    ]
    if missing:
        raise ValueError(
            f"{xlsx_path} sheet '{CPUC_SHEET}' is missing {missing}. The CPUC has changed the "
            "workbook layout; update benchmark_cpuc_baseline.py's column constants.",
        )

    df = df.copy()
    df[CPUC_REGION_COL] = df[CPUC_REGION_COL].astype("object").where(df[CPUC_REGION_COL].notna()).str.strip()
    df[CPUC_TECH_COL] = df[CPUC_TECH_COL].astype("object").where(df[CPUC_TECH_COL].notna()).str.strip()
    df[CPUC_CAPACITY_COL] = pd.to_numeric(df[CPUC_CAPACITY_COL], errors="coerce")
    for col in (CPUC_INSV_COL, CPUC_RETIRE_COL):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    ca = df[df[CPUC_REGION_COL].isin(CA_SERVM_REGIONS)].copy()
    if ca.empty:
        raise ValueError(
            f"No rows in {xlsx_path} carry a California SERVM Region ({', '.join(CA_SERVM_REGIONS)}). "
            "Either the wrong sheet was read or the region labels have changed.",
        )

    logger.info(
        "Read %d CPUC baseline units, %d of them in California across %d SERVM regions.",
        len(df),
        len(ca),
        ca[CPUC_REGION_COL].nunique(),
    )
    return ca


def load_tech_map(path: str) -> dict[str, dict[str, str]]:
    """
    ``{side: {source_category: compare_category}}`` read from ``servm_tech_map.csv``.

    One file holds both directions, keyed by a ``side`` column of ``cpuc`` or
    ``pypsa``, so the two halves of a comparison category can never drift apart
    across separate files.
    """
    df = pd.read_csv(path, dtype=str)
    for col in ("side", "source_category", "compare_category"):
        if col not in df.columns:
            raise ValueError(
                f"{path} must have columns side, source_category, compare_category; got {list(df.columns)}",
            )
        df[col] = df[col].str.strip()

    unknown_sides = sorted(set(df.side.unique()) - {"cpuc", "pypsa"})
    if unknown_sides:
        raise ValueError(f"{path} has unknown side values {unknown_sides}; expected 'cpuc' or 'pypsa'.")

    out: dict[str, dict[str, str]] = {}
    for side, grp in df.groupby("side"):
        duplicated = grp.source_category[grp.source_category.duplicated()].unique()
        if len(duplicated):
            raise ValueError(f"{path} maps {list(duplicated)} more than once on side '{side}'.")
        out[side] = dict(zip(grp.source_category, grp.compare_category))

    for side in ("cpuc", "pypsa"):
        out.setdefault(side, {})
    return out


def load_benchmark_regions(path: str) -> tuple[pd.Series, pd.Series]:
    """
    Return ``(servm_region -> benchmark_region, eia_ba_code -> benchmark_region)``.

    Both lookups come from the same file so the CAISO collapse (PGE/SCE/SDGE and
    EIA ``CISO`` landing on one ``CAISO`` row) is stated once.
    """
    df = pd.read_csv(path, dtype=str)
    for col in ("servm_region", "eia_ba_code", "benchmark_region"):
        if col not in df.columns:
            raise ValueError(
                f"{path} must have columns servm_region, eia_ba_code, benchmark_region; got {list(df.columns)}",
            )
        df[col] = df[col].str.strip()

    servm = df.drop_duplicates("servm_region").set_index("servm_region").benchmark_region
    dupes = servm.index[servm.index.duplicated()].unique()
    if len(dupes):
        raise ValueError(f"{path} maps SERVM regions {list(dupes)} to more than one benchmark region.")

    ba = df.drop_duplicates("eia_ba_code")
    conflicting = ba.groupby("eia_ba_code").benchmark_region.nunique()
    conflicting = conflicting[conflicting > 1]
    if len(conflicting):
        raise ValueError(f"{path} maps EIA BA codes {list(conflicting.index)} to more than one benchmark region.")

    return servm, ba.set_index("eia_ba_code").benchmark_region


# --------------------------------------------------------------------------- #
# shared filtering / mapping
# --------------------------------------------------------------------------- #


def filter_active(
    df: pd.DataFrame,
    horizon: int,
    insv_col: str,
    retire_col: str,
) -> pd.DataFrame:
    """
    Units in service by the end of ``horizon`` and not yet retired.

    A unit counts if it enters service on or before December 31 of the horizon
    year and its retirement year is strictly greater than the horizon — the same
    inequality ``add_electricity.load_powerplants`` applies to the model fleet,
    so the two sides of the benchmark are filtered identically.

    A missing retirement date means "never retires" (the workbook leaves the
    field empty for units with no announced retirement; it also uses a
    ``2050-12-01`` placeholder for many, which needs no special handling since a
    2050 retirement genuinely keeps the unit alive through any earlier horizon).
    A missing in-service date is treated as already in service: the workbook is a
    list of the *existing* baseline fleet, so an absent date is a data gap rather
    than a future project.
    """
    insv = pd.to_datetime(df[insv_col], errors="coerce")
    retire = pd.to_datetime(df[retire_col], errors="coerce").fillna(NEVER_RETIRES)

    n_missing_insv = int(insv.isna().sum())
    if n_missing_insv:
        logger.warning(
            "%d of %d units have no in-service date; treating them as already in service.",
            n_missing_insv,
            len(df),
        )

    in_service = insv.isna() | (insv <= pd.Timestamp(year=horizon, month=12, day=31))
    not_retired = retire.dt.year > horizon
    return df[in_service & not_retired]


def map_compare_category(values: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """
    Map source technology labels onto comparison categories.

    Anything absent from ``mapping`` becomes ``UNMAPPED:<label>`` rather than
    NaN, so unrecognised technologies survive into the output table with their
    capacity attached and show up as a one-sided row instead of quietly
    improving the agreement.
    """
    labels = values.astype("object").where(values.notna(), "").astype(str).str.strip()
    mapped = labels.map(mapping)
    unmapped = mapped.isna()
    if unmapped.any():
        logger.warning(
            "Unmapped technology categories (reported as %s… rows): %s",
            UNMAPPED_PREFIX,
            sorted(labels[unmapped].unique()),
        )
    return mapped.where(~unmapped, UNMAPPED_PREFIX + labels)


# --------------------------------------------------------------------------- #
# CPUC side
# --------------------------------------------------------------------------- #


def cpuc_capacity_by_region_tech(
    df: pd.DataFrame,
    tech_map: dict[str, str],
    horizon: int,
    region_map: pd.Series | None = None,
) -> pd.DataFrame:
    """
    CPUC installed capacity at ``horizon``, by benchmark region and category.

    ``region_map`` collapses SERVM regions onto benchmark regions (PGE/SCE/SDGE
    -> CAISO); passing ``None`` keeps the raw SERVM regions, which is what the
    unit tests exercise when the collapse itself is not under test.
    """
    active = filter_active(df, horizon, CPUC_INSV_COL, CPUC_RETIRE_COL)

    regions = active[CPUC_REGION_COL].astype(str)
    if region_map is not None:
        mapped = regions.map(region_map)
        if mapped.isna().any():
            raise ValueError(
                f"CPUC SERVM regions absent from the benchmark region map: {sorted(regions[mapped.isna()].unique())}",
            )
        regions = mapped

    out = pd.DataFrame(
        {
            "region": regions,
            "compare_category": map_compare_category(active[CPUC_TECH_COL], tech_map),
            "cpuc_mw": pd.to_numeric(active[CPUC_CAPACITY_COL], errors="coerce").fillna(0.0),
        },
    )
    return out.groupby(["region", "compare_category"], as_index=False).cpuc_mw.sum()


# --------------------------------------------------------------------------- #
# model side
# --------------------------------------------------------------------------- #


def prepare_model_plants(plants: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ``add_electricity.load_powerplants``' vintage/retirement rules.

    Mirrors ``workflow/scripts/add_electricity.py::load_powerplants`` (the
    ``operational_status`` handling around its build-year and retirement-date
    blocks): proposed units take their build year from
    ``current_planned_generator_operating_date``; existing and proposed units are
    pinned to a 2100 retirement (i.e. the model does not retire them on the EIA
    planned date); and units with no retirement date on record are pinned to
    1900 so the ``> horizon`` test drops them.

    Deviating from that here would benchmark a fleet the model never builds, so
    the mirroring is deliberate even where the upstream rule is coarse — the
    2100 pin in particular means announced retirements do not shrink the model
    fleet, which is one of the things this benchmark is meant to make visible.
    """
    df = plants.copy()

    for col in ("current_planned_generator_operating_date", "generator_retirement_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            df[col] = pd.NaT

    if "operational_status" not in df.columns:
        df["operational_status"] = pd.NA
    status = df.operational_status.astype("object")

    proposed = status == "proposed"
    if proposed.any():
        df.loc[proposed, "build_year"] = df.loc[proposed, "current_planned_generator_operating_date"].dt.year

    df.loc[status.isin(["existing", "proposed"]), "generator_retirement_date"] = pd.Timestamp("2100-01-01")
    df.loc[df.generator_retirement_date.isna(), "generator_retirement_date"] = pd.Timestamp("1900-01-01")

    # filter_active() reads an in-service *date*; build_year is a year integer.
    years = pd.to_numeric(df.get("build_year"), errors="coerce")
    build_date = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    known = years.notna()
    build_date[known] = pd.to_datetime(years[known].astype(int).astype(str) + "-01-01")
    df["build_date"] = build_date

    # ``load_powerplants`` compares ``build_year <= investment_periods[0]``, which is
    # False for NaN, so units with no build year never enter the model. Drop them
    # here too rather than letting filter_active's "no date means in service"
    # convention (right for the CPUC baseline of existing units) let them in.
    if (~known).any():
        logger.info("Dropping %d plants with no build year, matching load_powerplants.", int((~known).sum()))
        df = df[known]
    return df


def model_capacity_by_region_tech(
    powerplants_df: pd.DataFrame,
    region_alloc: pd.Series,
    tech_map: dict[str, str],
    horizon: int,
) -> pd.DataFrame:
    """
    Modelled installed capacity at ``horizon``, by benchmark region and category.

    ``region_alloc`` is the ``eia_ba_code -> benchmark_region`` lookup from
    :func:`load_benchmark_regions`. Only California plants are considered, since
    EIA's ``CISO`` code also spans the Nevada Valley Electric footprint that the
    SERVM California regions exclude. California plants whose balancing authority
    is outside the benchmark regions (a handful sit in WALC or NEVP) are kept as
    ``UNMAPPED:<ba>`` regions rather than dropped.

    Pumped hydro carries the ``hydro`` carrier in ``powerplants.csv``; the model
    splits it out by ``prime_mover_code == "PS"`` in
    ``add_electricity.attach_phs_storage``, so the same split is applied here
    before the carrier is mapped.
    """
    df = prepare_model_plants(powerplants_df)

    if "state" in df.columns:
        df = df[df.state.astype("object").where(df.state.notna(), "").astype(str).str.strip() == "CA"]
    else:
        logger.warning("powerplants.csv has no 'state' column; the CA restriction cannot be applied.")

    active = filter_active(df, horizon, "build_date", "generator_retirement_date")

    ba = active.get("balancing_authority_code_eia", pd.Series(index=active.index, dtype="object"))
    ba = ba.astype("object").where(ba.notna(), "").astype(str).str.strip()
    regions = ba.map(region_alloc)
    regions = regions.where(regions.notna(), UNMAPPED_PREFIX + ba)

    carriers = active.carrier.astype("object").where(active.carrier.notna(), "").astype(str).str.strip()
    if "prime_mover_code" in active.columns:
        is_phs = active.prime_mover_code.astype("object").where(active.prime_mover_code.notna(), "") == "PS"
        carriers = carriers.where(~is_phs, "PHS")

    out = pd.DataFrame(
        {
            "region": regions,
            "compare_category": map_compare_category(carriers, tech_map),
            "model_mw": pd.to_numeric(active.p_nom, errors="coerce").fillna(0.0),
        },
    )
    return out.groupby(["region", "compare_category"], as_index=False).model_mw.sum()


def reconcile_totals(n: pypsa.Network, fleet_table: pd.DataFrame, tech_map: dict[str, str]) -> pd.DataFrame:
    """
    Coarse cross-check of the assembled network against the fleet table.

    ``powerplants.csv`` is an input to ``add_electricity``, not its output: units
    are dropped when no bus matches, batteries without an energy capacity are
    skipped, and clustering aggregates the rest. Totalling ``p_nom`` over the
    network's generators and storage units by comparison category and setting it
    beside the fleet-table total therefore says how much of the powerplants table
    actually survived into the model.

    This is informational and carries no tolerance: the network is
    interconnect-wide while the fleet table is California-only, so the two
    totals are expected to differ. It rides along in the same long-format CSV as
    two pseudo-region rows per category, ``RECONCILE: fleet table (CA)`` and
    ``RECONCILE: network total``, both carrying their capacity in ``model_mw``
    with ``cpuc_mw`` left at zero — the CPUC is not one of the two things being
    compared here, so its column stays empty rather than being borrowed.
    """
    frames = []
    for component in ("generators", "storage_units"):
        c = getattr(n, component)
        if c.empty or "p_nom" not in c.columns:
            continue
        frames.append(
            pd.DataFrame({"carrier": c.carrier.astype(str), "p_nom": pd.to_numeric(c.p_nom, errors="coerce")}),
        )

    if not frames:
        logger.warning("Network carries no generators or storage units; skipping the reconciliation rows.")
        return pd.DataFrame(columns=["region", "compare_category", "cpuc_mw", "model_mw"])

    comps = pd.concat(frames, ignore_index=True)
    comps["compare_category"] = map_compare_category(comps.carrier, tech_map)

    fleet_mw = fleet_table.groupby("compare_category").model_mw.sum()
    network_mw = comps.groupby("compare_category").p_nom.sum()
    # Reindex both onto the union so a category that reaches only one side shows
    # up as an explicit zero — that gap is the whole point of the cross-check.
    categories = fleet_mw.index.union(network_mw.index)

    rows = {
        f"{RECONCILE_PREFIX} fleet table (CA)": fleet_mw.reindex(categories).fillna(0.0),
        f"{RECONCILE_PREFIX} network total": network_mw.reindex(categories).fillna(0.0),
    }
    return pd.concat(
        [
            pd.DataFrame({"region": label, "compare_category": categories, "cpuc_mw": 0.0, "model_mw": mw.to_numpy()})
            for label, mw in rows.items()
        ],
        ignore_index=True,
    )


# --------------------------------------------------------------------------- #
# comparison + plot
# --------------------------------------------------------------------------- #


def build_comparison(
    cpuc: pd.DataFrame,
    model: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Outer-join the two sides into the long comparison table for one horizon.

    Columns: ``horizon``, ``region``, ``compare_category``, ``cpuc_mw``,
    ``model_mw``, ``delta_mw``, ``delta_pct``. The join is outer so a category
    present on only one side keeps its row with a zero on the other.

    ``delta_pct`` is the model's deviation from the CPUC reference. It is NaN
    where the reference is zero (a model-only category has no percentage
    deviation to report) so the heatmap leaves that cell blank instead of
    painting an infinity.
    """
    merged = cpuc.merge(model, on=["region", "compare_category"], how="outer")
    merged[["cpuc_mw", "model_mw"]] = merged[["cpuc_mw", "model_mw"]].fillna(0.0)
    merged["delta_mw"] = merged.model_mw - merged.cpuc_mw
    merged["delta_pct"] = np.where(
        merged.cpuc_mw > 0,
        100.0 * merged.delta_mw / merged.cpuc_mw.replace(0.0, np.nan),
        np.nan,
    )
    merged.insert(0, "horizon", horizon)
    return merged.sort_values(["region", "compare_category"]).reset_index(drop=True)


def plot_deviation_heatmap(df: pd.DataFrame, path: str) -> None:
    """
    Region x category deviation heatmap, one panel per planning horizon.

    Cells are ``delta_pct`` on a diverging colormap centered at zero, clipped to
    +/-100% so a single model-only category cannot flatten the rest of the panel.
    Reconciliation rows are excluded — they compare the network against the
    fleet table, not against the CPUC, so they do not belong on a CPUC-deviation
    scale.
    """
    plot_df = df[~df.region.astype(str).str.startswith(RECONCILE_PREFIX)]
    horizons = sorted(plot_df.horizon.unique())

    if plot_df.empty or not horizons:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no CPUC benchmark rows to plot", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        return

    categories = sorted(plot_df.compare_category.unique())
    regions = sorted(plot_df.region.unique())

    fig, axes = plt.subplots(
        len(horizons),
        1,
        figsize=(1.0 * len(categories) + 4, 1.1 * len(regions) * len(horizons) + 2),
        squeeze=False,
    )
    for ax, horizon in zip(axes[:, 0], horizons):
        pivot = (
            plot_df[plot_df.horizon == horizon]
            .pivot_table(index="region", columns="compare_category", values="delta_pct")
            .reindex(index=regions, columns=categories)
        )
        sns.heatmap(
            pivot,
            ax=ax,
            annot=True,
            fmt=".0f",
            center=0.0,
            vmin=-100,
            vmax=100,
            cmap="coolwarm",
            linewidths=0.5,
            cbar_kws={"label": "model - CPUC [%]"},
        )
        ax.set_title(f"{horizon}: modelled capacity deviation from the CPUC baseline")
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("benchmark_cpuc_baseline")
    configure_logging(snakemake)

    # run.benchmark_cpuc_horizons decouples the benchmark year from the study
    # years: the default (2026) compares today's fleet against today's CPUC
    # baseline, which is the only pure fleet-vs-fleet comparison — later
    # horizons measure model expansion against a mostly-static baseline.
    horizons = sorted(set(snakemake.params.benchmark_horizons or snakemake.params.planning_horizons))

    tech_map = load_tech_map(snakemake.input.tech_map)
    servm_regions, ba_regions = load_benchmark_regions(snakemake.input.region_map)

    cpuc_raw = read_cpuc_baseline(snakemake.input.cpuc_baseline)
    plants = pd.read_csv(snakemake.input.powerplants)

    tables = []
    fleet_frames = []
    for horizon in horizons:
        cpuc = cpuc_capacity_by_region_tech(cpuc_raw, tech_map["cpuc"], horizon, region_map=servm_regions)
        model = model_capacity_by_region_tech(plants, ba_regions, tech_map["pypsa"], horizon)
        fleet_frames.append(model)
        tables.append(build_comparison(cpuc, model, horizon))

    comparison = pd.concat(tables, ignore_index=True)

    # Network reconciliation intentionally does not run here — the rule is
    # network-free so the benchmark never forces a model build. Use
    # reconcile_totals() interactively against a built network when needed.
    network_path = getattr(snakemake.input, "network", None)
    if network_path:
        n = pypsa.Network(network_path)
        recon = reconcile_totals(n, fleet_frames[0], tech_map["pypsa"])
        if not recon.empty:
            # Informational rows: there is no CPUC reference behind them, so
            # both delta columns stay empty rather than restating model_mw.
            recon.insert(0, "horizon", horizons[0])
            recon["delta_mw"] = np.nan
            recon["delta_pct"] = np.nan
            comparison = pd.concat([comparison, recon[comparison.columns]], ignore_index=True)

    logger.info(
        "Benchmarked %d horizons across %d regions and %d comparison categories.",
        len(horizons),
        comparison.region.nunique(),
        comparison.compare_category.nunique(),
    )
    comparison.to_csv(snakemake.output.comparison, index=False)
    plot_deviation_heatmap(comparison, snakemake.output.heatmap)
