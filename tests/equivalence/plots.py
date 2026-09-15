"""Figure renderer for the master-vs-develop equivalence benchmark.

PNG figures are the primary review artifact (PROJECT.md section 3.2), so this
module is a **renderer over the frames that** ``metrics.py`` **returns** and
holds no statistics of its own. Every figure is written through
:func:`save_figure`, which writes the PNG **and the CSV behind it with the same
stem** — a PNG without its CSV is a bug, and ``test_every_figure_has_a_csv``
asserts the two stem sets are equal.

Conventions, applied to every figure:

- **Two series, two fixed colours.** ``master-benchmark`` is blue,
  ``develop`` is orange, everywhere, with a legend on every figure that shows
  both. Identity is never colour-alone: axis labels name the side too.
- **One palette per carrier**, keyed on the carrier name via
  :func:`carrier_color` (the repo's own ``config.plotting.yaml`` ``tech_colors``
  when it is readable, a built-in fallback otherwise), so "solar" is the same
  hue in every figure it appears in.
- **Delta means develop minus master**, in the metric's own unit, and is shown
  on its own panel with a zero line — never as a second y-axis.
- Axes are labelled with their unit; grids are recessive.

Zone joins use the ``reeds_zone`` BUS ATTRIBUTE, never string surgery on cluster
names: cluster ids are ``p{zone}{subcluster} {i}`` with no separator ("p101 1"
is zone p10, sub-cluster 1), so any prefix split silently produces labels that
match almost nothing in reeds_shapes.geojson — the bug behind the first,
mostly-grey report maps.

- develop profile bus -> zone: the s{simpl} clustered network's ``buses.reeds_zone``.
- master profile bus -> zone: SUBSTATION id -> busmap_s{simpl} -> cluster ->
  ``reeds_zone``. The baseline's profile buses are substation ids ('39762.0'),
  NOT its base-network bus ids: the two numbering spaces overlap numerically but
  a direct join lands on the right zone only ~2% of the time (verified
  2026-09-01). The busmap chain is the same join develop's caps remap uses.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.ticker import MaxNLocator

from . import metrics
from .paths import EQ, INTERCONNECT, REPO, SIMPL2, prong_pairs

DPI = 150

LABELS = {"develop": "develop", "master": "master-benchmark (baseline)"}

#: The two branches, as two fixed categorical slots. Never reassigned by rank.
SIDE_COLORS = {"master": "#2a78d6", "develop": "#eb6834"}

#: Diverging pair for signed quantities, with a neutral midpoint for maps.
DELTA_CMAP = "RdBu_r"
GRID_KW = {"color": "#e3e3e0", "lw": 0.6}
MISSING_KW = {"color": "#dddddd"}
TEXT_MUTED = "#52514e"

#: Fallback categorical slots for carriers the project palette does not name.
#: Fixed order, assigned by a stable hash of the carrier name so the colour
#: follows the entity and never its rank in a particular figure.
_FALLBACK_SLOTS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)

_PALETTE: dict[str, str] | None = None


def _project_palette() -> dict[str, str]:
    """``tech_colors`` from the repo's plotting config; ``{}`` if unreadable.

    Using the project's own carrier colours keeps these figures legible beside
    every other pypsa-usa figure. The file is a tracked config, not an input
    artifact, so reading it needs no ``data/`` or ``resources/`` tree.
    """
    global _PALETTE
    if _PALETTE is not None:
        return _PALETTE
    _PALETTE = {}
    try:
        import yaml

        cfg = yaml.safe_load((REPO / "workflow" / "repo_data" / "config" / "config.plotting.yaml").read_text())
        colors = (cfg or {}).get("plotting", {}).get("tech_colors", {})
        _PALETTE = {str(k).lower(): str(v) for k, v in colors.items()}
    except (OSError, ValueError, AttributeError):
        _PALETTE = {}
    return _PALETTE


def carrier_color(carrier: str) -> str:
    """A stable colour for a carrier, consistent across every figure.

    The project's ``tech_colors`` win where they name the carrier. Anything else
    gets a fixed categorical slot chosen by a stable hash of the name, so the
    same carrier keeps the same hue between figures and between runs. Distinct
    carriers can collide on a slot; that is acceptable because in every figure
    here the carrier is also named on an axis, never encoded by colour alone.
    """
    key = str(carrier).lower()
    pal = _project_palette()
    if key in pal:
        return pal[key]
    return _FALLBACK_SLOTS[zlib.crc32(key.encode()) % len(_FALLBACK_SLOTS)]


def _style(ax, xlabel: str = "", ylabel: str = "") -> None:
    ax.grid(True, axis="both", **GRID_KW)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=TEXT_MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=TEXT_MUTED)


def save_figure(fig, data: pd.DataFrame, name: str, outdir: Path) -> tuple[Path, Path]:
    """Write ``<outdir>/figures/<name>.png`` **and** ``<name>.csv``.

    The CSV holds exactly the values that were plotted, so every figure has its
    data table (PROJECT.md section 3.2). Every figure goes through here.
    """
    fdir = Path(outdir) / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    png, csv = fdir / f"{name}.png", fdir / f"{name}.csv"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    frame.to_csv(csv, index=not isinstance(frame.index, pd.RangeIndex))
    return png, csv


def _empty_figure(title: str, name: str, outdir: Path, note: str = "no data") -> tuple[Path, Path]:
    """A labelled placeholder, so a missing solve cannot abort the export."""
    fig, ax = plt.subplots(figsize=(6, 2.4))
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)
    ax.text(0.5, 0.5, note, ha="center", va="center", fontsize=12, color=TEXT_MUTED)
    return save_figure(fig, metrics.empty_frame(), name, outdir)


def paired_bar(df: pd.DataFrame, title: str, unit: str, name: str, outdir: Path) -> tuple[Path, Path]:
    """Master/develop bars beside a develop-minus-master percentage panel.

    ``df`` is a metrics frame (``master``/``develop``/``delta``/``delta_pct``).
    An empty frame yields a labelled placeholder rather than an exception.
    """
    if df is None or df.empty:
        return _empty_figure(title, name, outdir)
    keys = [metrics_key_label(k) for k in df.index]
    y = np.arange(len(keys))
    h = 0.38
    height = max(2.6, 0.34 * len(keys) + 1.6)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, height), width_ratios=[2.4, 1], sharey=True)
    ax1.barh(y + h / 2, df["master"].to_numpy(), height=h, color=SIDE_COLORS["master"], label=LABELS["master"])
    ax1.barh(y - h / 2, df["develop"].to_numpy(), height=h, color=SIDE_COLORS["develop"], label=LABELS["develop"])
    ax1.set_yticks(y)
    ax1.set_yticklabels(keys, fontsize=8)
    ax1.invert_yaxis()
    _style(ax1, xlabel=f"{title} [{unit}]")
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_title(title, fontsize=11)

    pct = df["delta_pct"].to_numpy(dtype=float)
    ax2.barh(y, np.nan_to_num(pct, nan=0.0), height=0.6, color=[carrier_color(k) for k in keys])
    ax2.axvline(0, color="#999999", lw=0.8)
    for i, v in enumerate(pct):
        if not np.isfinite(v):
            ax2.text(0, i, "  new on develop", va="center", fontsize=7, color=TEXT_MUTED)
    _style(ax2, xlabel="develop - master [%]")
    ax2.set_title("relative difference", fontsize=10)
    fig.tight_layout()
    return save_figure(fig, df, name, outdir)


def metrics_key_label(key) -> str:
    """Flatten a (possibly tuple) metric key to one readable label."""
    return " | ".join(str(k) for k in key) if isinstance(key, tuple) else str(key)


def objective_figure(row: pd.Series, name: str, outdir: Path) -> tuple[Path, Path]:
    """Total system cost on both sides, with the HF-13 normalisation visible.

    Both the normalised totals (what is compared) and the raw reported
    ``objective``/``objective_constant`` (what each pypsa version printed) are
    drawn, so the reader can see that the offset was moved and not assumed away.
    """
    if row is None or len(row) == 0:
        return _empty_figure("total system cost", name, outdir, note="no solved network")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(2)
    vals = [row["master"] / 1e9, row["develop"] / 1e9]
    ax.bar(x, vals, width=0.5, color=[SIDE_COLORS["master"], SIDE_COLORS["develop"]])
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS["master"], LABELS["develop"]], fontsize=9)
    for xi, v in zip(x, vals):
        ax.text(xi, v, f"{v:,.4f}", ha="center", va="bottom", fontsize=9, color=TEXT_MUTED)
    _style(ax, ylabel="objective + objective_constant [bn $]")
    ax.set_title(
        f"total system cost - delta {row['delta']:,.1f} $ ({row['delta_pct']:.4g} %)\n"
        f"raw objective: master {row['master_raw']:,.1f} + constant {row['master_constant']:,.1f}; "
        f"develop {row['develop_raw']:,.1f} + constant {row['develop_constant']:,.1f} (HF-13)",
        fontsize=9,
    )
    fig.tight_layout()
    return save_figure(fig, row.to_frame("value"), name, outdir)


def _zone_frame(v_master: pd.Series, v_develop: pd.Series) -> pd.DataFrame:
    return metrics.frame(v_master, v_develop, name="zone")


def choropleth_triptych(
    zones,
    v_master: pd.Series,
    v_develop: pd.Series,
    title: str,
    unit: str,
    name: str,
    outdir: Path,
) -> tuple[Path, Path]:
    """Master, develop and develop-minus-master, side by side, on one scale.

    ``zones`` is either a GeoDataFrame of zone polygons indexed by zone name, or
    a DataFrame with ``x``/``y`` columns (zone or bus coordinates taken from the
    network) for the point-map fallback when no shapes are available. Zones with
    no value are grey on every panel, never silently dropped.
    """
    data = _zone_frame(v_master, v_develop)
    if data.empty or zones is None or len(zones) == 0:
        return _empty_figure(title, name, outdir, note="no zone data")
    vmax = float(np.nanmax([data["master"].max(), data["develop"].max()])) or 1e-9
    dmax = float(np.nanmax(np.abs(data["delta"].to_numpy()))) or 1e-9
    panels = (
        ("master", LABELS["master"], "viridis", 0.0, vmax),
        ("develop", LABELS["develop"], "viridis", 0.0, vmax),
        ("delta", f"{LABELS['develop']} - {LABELS['master']}\n(grey = no data)", DELTA_CMAP, -dmax, dmax),
    )
    is_geo = "geometry" in getattr(zones, "columns", [])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (col, lab, cmap, vmin, vhi) in zip(axes, panels):
        joined = zones.join(data[col].rename("v"))
        if is_geo:
            joined.plot(column="v", ax=ax, cmap=cmap, vmin=vmin, vmax=vhi, legend=True, missing_kwds=MISSING_KW)
        else:
            miss = joined["v"].isna()
            size = max(40.0, min(260.0, 6000.0 / max(len(joined), 1)))
            ax.scatter(
                joined.loc[miss, "x"], joined.loc[miss, "y"],
                c=MISSING_KW["color"], s=size, edgecolors="white", linewidths=0.5,
            )
            sc = ax.scatter(
                joined.loc[~miss, "x"],
                joined.loc[~miss, "y"],
                c=joined.loc[~miss, "v"],
                cmap=cmap,
                vmin=vmin,
                vmax=vhi,
                s=size,
                edgecolors="white",
                linewidths=0.5,
            )
            ax.set_aspect("equal", adjustable="box")
            ax.margins(0.15)
            fig.colorbar(sc, ax=ax, shrink=0.7, fraction=0.045, pad=0.02)
        ax.set_title(f"{lab}\n{title} [{unit}]", fontsize=9)
        if is_geo:
            ax.set_axis_off()
        else:
            # The point fallback has no coastline to orient against, so keep
            # the coordinate axes as the spatial reference.
            _style(ax, xlabel="x [deg]", ylabel="y [deg]")
            ax.tick_params(labelsize=7)
    fig.tight_layout()
    return save_figure(fig, data, name, outdir)


def duration_curve(
    s_master: pd.Series,
    s_develop: pd.Series,
    title: str,
    unit: str,
    name: str,
    outdir: Path,
    points: int = 201,
    color: str = "#a03030",
) -> tuple[Path, Path]:
    """Sorted-descending distribution of two value sets on a common percentile axis.

    Sampling both sides at the same percentiles is what makes them comparable
    when their bus spaces differ, which they do at prong 2.
    """
    vm = np.asarray(pd.Series(s_master, dtype=float).dropna().to_numpy())
    vd = np.asarray(pd.Series(s_develop, dtype=float).dropna().to_numpy())
    if vm.size == 0 or vd.size == 0:
        return _empty_figure(title, name, outdir, note="no profile data")
    q = np.linspace(0.0, 100.0, points)
    ym = np.percentile(vm, 100.0 - q)
    yd = np.percentile(vd, 100.0 - q)
    data = pd.DataFrame(
        {"master": ym, "develop": yd, "delta": yd - ym},
        index=pd.Index(q, name="percent_of_time_at_or_above"),
    )
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, height_ratios=[3, 1])
    ax1.plot(q, ym, color=SIDE_COLORS["master"], lw=2.0, label=LABELS["master"])
    ax1.plot(q, yd, color=SIDE_COLORS["develop"], lw=2.0, ls="--", label=LABELS["develop"])
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_title(title, fontsize=11)
    _style(ax1, ylabel=f"{title} [{unit}]")
    ax2.axhline(0, color="#999999", lw=0.8)
    ax2.plot(q, yd - ym, color=color, lw=1.6)
    _style(ax2, xlabel="percent of samples at or above [%]", ylabel=f"develop - master\n[{unit}]")
    fig.tight_layout()
    return save_figure(fig, data, name, outdir)


def timeseries_pair(
    s_master: pd.Series,
    s_develop: pd.Series,
    title: str,
    unit: str,
    name: str,
    outdir: Path,
) -> tuple[Path, Path]:
    """Two time series plus their relative difference.

    A full benchmark year is resampled to daily means to stay legible; a short
    series (fewer than three days, as in the unit tests) is drawn at its native
    resolution, since a one-point daily mean plots nothing at all.
    """
    if s_master is None or s_develop is None or len(s_master) == 0 or len(s_develop) == 0:
        return _empty_figure(title, name, outdir, note="no profile data")
    resampled = False
    dm, dd = s_master, s_develop
    try:
        rm, rd = s_master.resample("D").mean(), s_develop.resample("D").mean()
        if min(len(rm), len(rd)) >= 3:
            dm, dd, resampled = rm, rd, True
    except (TypeError, ValueError):  # non-datetime index
        pass
    n = min(len(dm), len(dd))
    idx = dm.index[:n]
    ym, yd = dm.to_numpy()[:n], dd.to_numpy()[:n]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(ym != 0, 100.0 * (yd - ym) / ym, np.nan)
    data = pd.DataFrame({"master": ym, "develop": yd, "delta": yd - ym, "delta_pct": rel}, index=idx)
    marker = "o" if n <= 40 else None
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True, height_ratios=[3, 1])
    ax1.plot(idx, ym, color=SIDE_COLORS["master"], lw=1.6, marker=marker, ms=4, label=LABELS["master"])
    ax1.plot(idx, yd, color=SIDE_COLORS["develop"], lw=1.6, ls="--", marker=marker, ms=4, label=LABELS["develop"])
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_title(title, fontsize=11)
    _style(ax1, ylabel=f"{title}\n[{unit}{', daily mean' if resampled else ''}]")
    ax2.axhline(0, color="#999999", lw=0.8)
    ax2.plot(idx, rel, color="#a03030", lw=1.2, marker=marker, ms=3)
    _style(ax2, xlabel="date", ylabel="develop - master [%]")
    fig.tight_layout()
    return save_figure(fig, data, name, outdir)


def findings_by_stage(findings: list[dict], name: str, outdir: Path) -> tuple[Path, Path]:
    """Stacked bar of live vs waived findings per comparison stage."""
    if not findings:
        return _empty_figure("findings by stage", name, outdir, note="no findings")
    df = pd.DataFrame(findings)
    if "waived" not in df.columns:
        df["waived"] = False
    df["waived"] = df["waived"].fillna(False).astype(bool)
    counts = (
        df.groupby([df["stage"].astype(str), df["waived"]]).size().unstack(fill_value=0).rename(
            columns={False: "live", True: "waived"},
        )
    )
    for col in ("live", "waived"):
        if col not in counts:
            counts[col] = 0
    counts = counts[["live", "waived"]].sort_values("live", ascending=False)
    y = np.arange(len(counts))
    fig, ax = plt.subplots(figsize=(9, max(2.6, 0.34 * len(counts) + 1.6)))
    ax.barh(y, counts["live"], color="#e34948", label="live")
    ax.barh(y, counts["waived"], left=counts["live"], color="#b9b8b2", label="waived")
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index, fontsize=8)
    ax.invert_yaxis()
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("equivalence findings by stage", fontsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style(ax, xlabel="number of findings")
    fig.tight_layout()
    return save_figure(fig, counts, name, outdir)


# --------------------------------------------------------------------------
# Artifact loading: the only part of the harness's numeric path that touches
# files. ``metrics.py`` stays file-free so its tests need no data.
# --------------------------------------------------------------------------


@dataclass
class Artifacts:
    """Loaded objects for one prong, with ``None`` for anything not built yet."""

    prong: int
    develop_root: Path
    master_root: Path
    n_master: object | None = None
    n_develop: object | None = None
    solved_master: object | None = None
    solved_develop: object | None = None
    zone_master: pd.Series | None = None
    zone_develop: pd.Series | None = None
    zones: object | None = None

    @property
    def profile_pairs(self) -> list:
        """The prong's renewable-profile artifact pairs."""
        return [p for p in prong_pairs(self.prong) if p.kind == "profile"]


def _zone_maps(develop_root: Path):
    """(develop bus->zone, master substation->zone, zone shapes or None)."""
    import geopandas as gpd
    import pypsa

    zc = pd.Series(dtype=object)
    za = pd.Series(dtype=object)
    zones = None
    shp = develop_root / f"{EQ}/geospatial/{INTERCONNECT}/reeds_shapes.geojson"
    if shp.exists():
        zones = gpd.read_file(shp).set_index("name")
    net = develop_root / f"{EQ}/networks/{INTERCONNECT}/elec_s{SIMPL2}.nc"
    if net.exists():
        nc = pypsa.Network(str(net))
        zc = nc.buses["reeds_zone"].astype(str)
        if zones is None:
            # No shapes: fall back to a point map built from the network's own
            # bus coordinates, averaged per zone.
            zones = nc.buses.groupby(zc)[["x", "y"]].mean()
        bm = develop_root / f"{EQ}/busmaps/{INTERCONNECT}/busmap_s{SIMPL2}.csv"
        if bm.exists():
            busmap = pd.read_csv(bm, index_col=0, dtype=str).iloc[:, 0]
            busmap.index = busmap.index.astype(str)
            za = busmap.map(zc)  # substation id -> cluster -> reeds_zone
    return zc, za, zones


def load_artifacts(prong: int, develop_root: Path, master_root: Path) -> Artifacts:
    """Load the networks and zone maps a figure export needs; skip what is absent."""
    from .compare import load_network

    art = Artifacts(prong=prong, develop_root=Path(develop_root), master_root=Path(master_root))
    for pair in prong_pairs(prong):
        if pair.kind not in ("network", "network_pkl_vs_nc"):
            continue
        dp, mp = art.develop_root / pair.develop, art.master_root / pair.master
        if not (dp.exists() and mp.exists()):
            continue
        if pair.solve_stage:
            art.solved_develop, art.solved_master = load_network(dp), load_network(mp)
        elif pair.stage == "clustered_network":
            art.n_develop, art.n_master = load_network(dp), load_network(mp)
    art.zone_develop, art.zone_master, art.zones = _zone_maps(art.develop_root)
    return art


def _safe(metric: str, missing: list[dict], fn, *a, **kw):
    """Run a metric; on failure record it in ``missing`` and return ``None``.

    A metric that cannot be computed is a criterion vanishing from the
    comparison, which is worse than a difference — so it is never swallowed.
    ``tables.comparison_table`` turns every entry of ``missing`` into a
    ``MISSING`` row, and ``run.py`` exits non-zero while any remain.
    """
    try:
        return fn(*a, **kw)
    except Exception as exc:
        # Deliberately broad. The four-exception list this replaced let an
        # IndexError or an OSError from one metric abort the whole export,
        # losing every other metric AND the MISSING row that would have
        # explained it. KeyboardInterrupt and SystemExit derive from
        # BaseException, not Exception, so they still propagate.
        reason = f"{type(exc).__name__}: {exc}"
        print(f"[plots] MISSING {metric}: {reason}")
        missing.append({"metric": metric, "reason": reason})
        return None


def collect_metrics(art: Artifacts, missing: list[dict] | None = None) -> dict[str, object]:
    """Every metric frame the comparison table and the figures need.

    A metric whose *artifacts* were never built is simply absent (a pre-solve run
    legitimately has no dispatch). A metric whose artifacts exist but which
    **raises** is appended to ``missing`` and surfaces as a ``MISSING`` row in
    the comparison table — that is the difference between "not run yet" and
    "the criterion broke".
    """
    out: dict[str, object] = {}
    miss = missing if missing is not None else []
    pre = (art.n_master, art.n_develop)
    solved = (art.solved_master, art.solved_develop)
    if all(x is not None for x in pre):
        out["capacity_existing_by_carrier"] = _safe(
            "capacity_existing_by_carrier", miss, metrics.capacity_by_carrier, *pre, attr="p_nom",
        )
        out["p_nom_existing_by_zone_carrier"] = _safe(
            "p_nom_existing_by_zone_carrier", miss, metrics.capacity_by_zone_carrier, *pre, attr="p_nom",
        )
        out["demand_by_zone"] = _safe("demand_by_zone", miss, metrics.demand_by_zone, *pre)
    if all(x is not None for x in solved):
        out["objective"] = _safe("objective", miss, metrics.objective_row, *solved)
        out["capacity_opt_by_carrier"] = _safe(
            "capacity_opt_by_carrier", miss, metrics.capacity_by_carrier, *solved, attr="p_nom_opt",
        )
        out["dispatch_by_carrier"] = _safe("dispatch_by_carrier", miss, metrics.dispatch_by_carrier, *solved)
        out["capacity_factor_by_carrier"] = _safe(
            "capacity_factor_by_carrier", miss, metrics.capacity_factor_by_carrier, *solved,
        )
    for pair in art.profile_pairs:
        tech = pair.stage.replace("profile_", "")
        dp, mp = art.develop_root / pair.develop, art.master_root / pair.master
        if not (dp.exists() and mp.exists()):
            continue
        with xr.open_dataset(dp) as dsd, xr.open_dataset(mp) as dsm:
            out[f"p_max_pu_quantiles_{tech}"] = _safe(
                f"p_max_pu_quantiles_{tech}", miss, metrics.p_max_pu_quantiles, dsm, dsd,
            )
            out[f"p_nom_max_by_zone_{tech}"] = _safe(
                f"p_nom_max_by_zone_{tech}", miss, metrics.p_nom_max_by_zone,
                dsm, dsd, art.zone_master, art.zone_develop,
            )
            out[f"mean_cf_by_zone_{tech}"] = _safe(
                f"mean_cf_by_zone_{tech}", miss, metrics.mean_cf_by_zone,
                dsm, dsd, art.zone_master, art.zone_develop,
            )
    return {k: v for k, v in out.items() if v is not None}


#: Per-process memo of ``(artifacts, metrics, missing)`` keyed by the run's
#: prong and roots. ``tables.export_all`` and :func:`export_all` are called back
#: to back by ``run.py`` and would otherwise load every network twice.
_RUN_METRICS: dict[tuple[str, ...], tuple] = {}


def run_metrics(
    prong: int,
    develop_root: Path | None = None,
    master_root: Path | None = None,
) -> tuple[Artifacts, dict[str, object], list[dict]]:
    """``(artifacts, metric_frames, missing)`` for one run, computed once.

    The result is memoised for the process, so the comparison table and the
    figures are drawn from the *same* numbers and the networks are read once.
    """
    from .build import BASELINE_WORKTREE

    develop_root = Path(develop_root or REPO / "workflow")
    master_root = Path(master_root or BASELINE_WORKTREE / "workflow")
    key = (str(prong), str(develop_root), str(master_root))
    if key not in _RUN_METRICS:
        art = load_artifacts(prong, develop_root, master_root)
        missing: list[dict] = []
        frames = collect_metrics(art, missing)
        _RUN_METRICS[key] = (art, frames, missing)
    return _RUN_METRICS[key]


def _carrier_zone_slice(df: pd.DataFrame, carrier: str) -> tuple[pd.Series, pd.Series]:
    """(master, develop) zone Series for one carrier of a (zone, carrier) frame."""
    sub = df.xs(carrier, level="carrier")
    return sub["master"], sub["develop"]


def export_all(
    run_dir: Path,
    ctx: object | None = None,
    artifacts: Artifacts | None = None,
    metric_frames: dict[str, object] | None = None,
    findings: list[dict] | None = None,
    missing: list[dict] | None = None,
    zone_carriers: int = 6,
) -> Path:
    """Render every required figure under ``<run_dir>/figures/``.

    Called by ``run.py`` as ``export_all(ctx.run_dir, ctx)``; ``ctx`` is the
    :class:`context.RunContext`, from which only ``prong`` is read. The
    remaining arguments exist for the unit tests, which supply their own
    fixtures instead of reading a run directory.

    Metrics that failed to compute are written to
    ``<run_dir>/missing_metrics.json`` so the failure survives the run even if
    nobody reads stdout. Returns the figures directory.
    """
    run_dir = Path(run_dir)
    prong = int(getattr(ctx, "prong", 2))
    if artifacts is None:
        artifacts, computed, computed_missing = run_metrics(prong)
        metric_frames = computed if metric_frames is None else metric_frames
        missing = computed_missing if missing is None else missing
    missing = [] if missing is None else missing
    m = collect_metrics(artifacts, missing) if metric_frames is None else metric_frames
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "missing_metrics.json").write_text(json.dumps(missing, indent=1))

    objective_figure(m.get("objective"), "objective", run_dir)
    paired_bar(m.get("capacity_existing_by_carrier"), "existing capacity", "MW", "capacity_existing_by_carrier", run_dir)
    paired_bar(m.get("capacity_opt_by_carrier"), "optimised capacity", "MW", "capacity_opt_by_carrier", run_dir)
    paired_bar(m.get("dispatch_by_carrier"), "annual dispatch", "MWh", "dispatch_by_carrier", run_dir)
    paired_bar(m.get("capacity_factor_by_carrier"), "realised capacity factor", "-", "capacity_factor_by_carrier", run_dir)
    paired_bar(m.get("demand_by_zone"), "demand by zone", "MW", "demand_zones", run_dir)

    zc = m.get("p_nom_existing_by_zone_carrier")
    if zc is not None and not zc.empty:
        top = zc.groupby(level="carrier")["master"].sum().sort_values(ascending=False)
        for carrier in list(top.index)[:zone_carriers]:
            vm, vd = _carrier_zone_slice(zc, carrier)
            choropleth_triptych(
                artifacts.zones, vm, vd, f"{carrier} existing capacity", "MW",
                f"p_nom_existing_zones_{carrier}", run_dir,
            )

    for pair in artifacts.profile_pairs:
        tech = pair.stage.replace("profile_", "")
        dp, mp = artifacts.develop_root / pair.develop, artifacts.master_root / pair.master
        if not (dp.exists() and mp.exists()):
            print(f"[plots] skipping {tech}: artifact missing")
            continue
        with xr.open_dataset(dp) as dsd, xr.open_dataset(mp) as dsm:
            pot = m.get(f"p_nom_max_by_zone_{tech}")
            cf = m.get(f"mean_cf_by_zone_{tech}")
            if pot is not None:
                choropleth_triptych(
                    artifacts.zones, pot["master"] / 1e3, pot["develop"] / 1e3,
                    f"{tech} installable potential", "GW", f"{tech}_potential_zones", run_dir,
                )
            if cf is not None:
                choropleth_triptych(
                    artifacts.zones, cf["master"], cf["develop"],
                    f"{tech} potential-weighted mean CF", "-", f"{tech}_meancf_zones", run_dir,
                )
            duration_curve(
                pd.Series(metrics.profile_values(dsm)),
                pd.Series(metrics.profile_values(dsd)),
                f"{tech} capacity factor", "-", f"p_max_pu_duration_{tech}", run_dir,
                color=carrier_color(tech),
            )
            timeseries_pair(
                metrics.available_power(dsm) / 1e3,
                metrics.available_power(dsd) / 1e3,
                f"{tech}: sum over buses of profile x p_nom_max", "GW",
                f"{tech}_national_available_power", run_dir,
            )

    if findings is None:
        findings = _read_findings(run_dir, artifacts.prong)
    findings_by_stage(findings, "findings_by_stage", run_dir)
    return run_dir / "figures"


def _read_findings(run_dir: Path, prong: int) -> list[dict]:
    """Findings from the run directory, or ``[]`` when they are not there yet.

    One run directory per run (plan D5): ``compare.run_comparison`` writes
    ``<run_dir>/findings_<prong>.json`` beside ``run_meta.json`` and both
    manifests, so the figures read it from the same place.
    """
    p = Path(run_dir) / f"findings_{prong}.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text()).get("findings", [])
    except (OSError, ValueError):
        return []
