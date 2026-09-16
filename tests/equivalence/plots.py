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

At **prong 2** every profile metric is computed after master's nodal file has
been rolled up onto develop's cluster bus space by
``metrics.aggregate_profile_to_clusters`` (the same ``busmap_s{simpl}``), so the
two sides are compared at one resolution. Master's bus ids are then cluster ids,
so the master side of those metrics takes the *develop* cluster->zone map, not
the substation->zone chain above. Clusters only one side carries are then
removed by ``metrics.common_cluster_subset`` — the pooled statistics are over
the shared population, and the one-sided clusters are reported as their own
``cluster_set`` finding rather than smeared across every quantile row.

**The figures take that same object.** :func:`prepare_profiles` is the ONE place
the rollup, the common-cluster subset and the master zone-map switch happen;
:func:`collect_metrics` calls it, caches the result on
:attr:`Artifacts.profiles`, and :func:`export_all` draws every profile figure
from the cache. Reading the raw master file again for a figure is how
``p_max_pu_duration_onwind`` came to plot 544 pooled substations against 19
clusters while the ``p_max_pu_quantiles_onwind`` table rows beside it were
equivalent — one run, two answers, and the picture was the wrong one. Figures
whose master side was rolled up say so, in the legend and under the title.
"""

from __future__ import annotations

import json
import zlib
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.ticker import MaxNLocator

from . import metrics
from .paths import (
    EQ,
    INTERCONNECT,
    REPO,
    SIMPL2,
    assembled_target,
    baseline_assembled_target,
    prong_pairs,
)

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
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (col, lab, cmap, vmin, vhi) in zip(axes, panels):
        _zone_panel(fig, ax, zones, data[col], cmap, vmin, vhi, f"{lab}\n{title} [{unit}]")
    fig.tight_layout()
    return save_figure(fig, data, name, outdir)


def _zone_panel(fig, ax, zones, values: pd.Series, cmap, vmin: float, vmax: float, label: str) -> None:
    """One zone panel: a choropleth when ``zones`` has geometry, else a point map.

    Extracted from :func:`choropleth_triptych` so the HF-26 reconstruction map
    draws zones with exactly the same styling and the same "grey means no data"
    rule, rather than a second look-alike implementation that drifts.
    """
    is_geo = "geometry" in getattr(zones, "columns", [])
    joined = zones.join(values.rename("v"))
    if is_geo:
        joined.plot(column="v", ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, legend=True, missing_kwds=MISSING_KW)
    else:
        miss = joined["v"].isna()
        size = max(40.0, min(260.0, 6000.0 / max(len(joined), 1)))
        ax.scatter(
            joined.loc[miss, "x"],
            joined.loc[miss, "y"],
            c=MISSING_KW["color"],
            s=size,
            edgecolors="white",
            linewidths=0.5,
        )
        sc = ax.scatter(
            joined.loc[~miss, "x"],
            joined.loc[~miss, "y"],
            c=joined.loc[~miss, "v"],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=size,
            edgecolors="white",
            linewidths=0.5,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.margins(0.15)
        fig.colorbar(sc, ax=ax, shrink=0.7, fraction=0.045, pad=0.02)
    ax.set_title(label, fontsize=9)
    if is_geo:
        ax.set_axis_off()
    else:
        # The point fallback has no coastline to orient against, so keep the
        # coordinate axes as the spatial reference.
        _style(ax, xlabel="x [deg]", ylabel="y [deg]")
        ax.tick_params(labelsize=7)


#: Bar colour for a reconstruction row whose residual is outside its gate.
RECON_FAIL_COLOR = "#c0392b"
#: Fill of the +/- gate_tol_mw band on the residual panel.
RECON_BAND_COLOR = "#dfe7ee"
#: The reconstruction's own prediction of the fleet, drawn beside the two sides.
RECON_FLEET_COLOR = "#7f7f78"
#: How many zones a reconstruction figure plots before pooling the rest. The USA
#: leg has 134 ReEDS zones; the CSV twin is always complete.
RECON_ZONE_CAP = 25

_RECON_NUMERIC = ("fleet_mw", "profiled_mw", "dropped_mw", "master_mw", "develop_mw", "residual_mw", "gate_tol_mw")


def _recon_frame(recon) -> tuple[pd.DataFrame | None, str]:
    """``(frame, note)`` for a reconstruction: the frame, or why there is none."""
    if recon is None:
        return None, "no reconstruction was computed for this run"
    error = getattr(recon, "error", None)
    if error:
        return None, f"reconstruction unavailable: {error}"
    frame = getattr(recon, "frame", None)
    if frame is None or frame.empty:
        return None, "reconstruction produced no rows"
    out = frame.reset_index(drop=True).copy()
    for col in _RECON_NUMERIC:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["zone"] = out["zone"].astype(str)
    out["carrier"] = out["carrier"].astype(str)
    return out, ""


def _recon_plot_rows(sub: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Top ``cap`` zones by ``|dropped_mw|``, with the remainder pooled.

    Pooling rather than truncating: a figure that silently stops at 25 of 134
    zones invites the reader to add up what is drawn and get a different national
    total than the title states.
    """
    ordered = sub.assign(_o=sub["dropped_mw"].abs()).sort_values("_o", ascending=False).drop(columns="_o")
    if len(ordered) <= cap:
        return ordered
    head, tail = ordered.iloc[:cap], ordered.iloc[cap:]
    pooled = {c: float(tail[c].sum()) for c in _RECON_NUMERIC}
    pooled.update(zone=f"other ({len(tail)} zones)", carrier=str(head["carrier"].iloc[0]))
    return pd.concat([head, pd.DataFrame([pooled])], ignore_index=True)


def _recon_title(df: pd.DataFrame) -> str:
    """National fleet / master / develop / dropped per carrier, and max |residual|."""
    parts = []
    for carrier, g in df.groupby("carrier", sort=True):
        parts.append(
            f"{carrier}: fleet {g['fleet_mw'].sum():,.0f} / master {g['master_mw'].sum():,.0f} / "
            f"develop {g['develop_mw'].sum():,.0f} / dropped {g['dropped_mw'].sum():,.0f} MW",
        )
    worst = float(df["residual_mw"].abs().max()) if len(df) else 0.0
    return (
        "HF-26 reconstruction: existing renewable MW master drops at substations its profile file does not cover\n"
        + "; ".join(parts)
        + f"\nmax |residual| {worst:,.1f} MW (residual = (develop - master) - dropped)"
    )


def hf26_dropped_zone_figure(
    recon,
    name: str,
    outdir: Path,
    cap: int = RECON_ZONE_CAP,
) -> tuple[Path, Path]:
    """Per-zone bars of the reconstruction, with a residual panel beside them.

    Left panel per carrier: ``master``, the reconstructed ``dropped_mw`` hatched
    and stacked on top of it, ``develop``, and the reconstructed ``fleet_mw``. The
    hatched segment is the claim: ``master + dropped`` must reach ``develop``.

    Right panel: ``residual_mw`` against a shaded +/-``gate_tol_mw`` band, so a
    zone the mechanism does NOT explain is the one bar sticking out of its band
    (drawn in the failure colour). That is the HF-27 relocation shape.

    The CSV twin is the FULL, uncapped frame, whatever the plot pooled.
    """
    df, note = _recon_frame(recon)
    if df is None:
        return _empty_figure("HF-26 reconstruction", name, outdir, note=note)
    carriers = sorted(df["carrier"].unique())
    rows = [_recon_plot_rows(df[df["carrier"] == c], cap) for c in carriers]
    height = sum(max(2.6, 0.30 * len(r) + 1.6) for r in rows)
    fig, axes = plt.subplots(
        len(carriers),
        2,
        figsize=(14, height),
        width_ratios=[2.4, 1],
        squeeze=False,
    )
    for (ax1, ax2), carrier, plot_df in zip(axes, carriers, rows):
        y = np.arange(len(plot_df))
        h = 0.26
        ax1.barh(y + h, plot_df["master_mw"], height=h, color=SIDE_COLORS["master"], label=LABELS["master"])
        ax1.barh(
            y + h,
            plot_df["dropped_mw"],
            height=h,
            left=plot_df["master_mw"],
            color=SIDE_COLORS["master"],
            alpha=0.35,
            hatch="///",
            edgecolor="white",
            label="dropped by master (reconstructed)",
        )
        ax1.barh(y, plot_df["develop_mw"], height=h, color=SIDE_COLORS["develop"], label=LABELS["develop"])
        ax1.barh(y - h, plot_df["fleet_mw"], height=h, color=RECON_FLEET_COLOR, label="reconstructed fleet")
        ax1.set_yticks(y)
        ax1.set_yticklabels(plot_df["zone"], fontsize=8)
        ax1.invert_yaxis()
        _style(ax1, xlabel="existing capacity [MW]")
        ax1.legend(fontsize=7, frameon=False)
        ax1.set_title(f"{carrier}: master + dropped should meet develop", fontsize=10)

        tol = plot_df["gate_tol_mw"].to_numpy(dtype=float)
        res = plot_df["residual_mw"].to_numpy(dtype=float)
        ax2.barh(y, 2.0 * tol, left=-tol, height=0.82, color=RECON_BAND_COLOR, zorder=0, label="gate tolerance")
        colors = [RECON_FAIL_COLOR if abs(r) > t else carrier_color(carrier) for r, t in zip(res, tol)]
        ax2.barh(y, res, height=0.52, color=colors, zorder=2)
        ax2.axvline(0, color="#999999", lw=0.8, zorder=3)
        ax2.set_yticks(y)
        # Repeat the zone labels: the two panels are not sharey (their x scales
        # differ by three orders of magnitude), so a bare 0..n index here would
        # leave the reader guessing which bar is which zone.
        ax2.set_yticklabels(plot_df["zone"], fontsize=8)
        ax2.invert_yaxis()
        _style(ax2, xlabel="residual [MW]")
        ax2.legend(fontsize=7, frameon=False)
        ax2.set_title("(develop - master) - dropped", fontsize=10)
    fig.suptitle(_recon_title(df), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, df, name, outdir)


def hf26_dropped_map_figure(
    zones,
    recon,
    name: str,
    outdir: Path,
) -> tuple[Path, Path]:
    """Zone choropleth of ``dropped_mw`` per carrier, plus a residual panel.

    Where the reconstruction says master lost capacity, and where the prediction
    fails to account for the difference. The residual panel is on a diverging
    scale on purpose: the relocation shape is equal-and-opposite neighbours, and
    a sequential ramp hides the pairing.
    """
    df, note = _recon_frame(recon)
    if df is None:
        return _empty_figure("HF-26 dropped capacity by zone", name, outdir, note=note)
    if zones is None or len(zones) == 0:
        return _empty_figure("HF-26 dropped capacity by zone", name, outdir, note="no zone data")
    carriers = sorted(df["carrier"].unique())
    dropped = df.pivot_table(index="zone", columns="carrier", values="dropped_mw", aggfunc="sum")
    residual = df.groupby("zone")["residual_mw"].sum()
    vmax = float(np.nanmax(dropped.to_numpy(dtype=float))) if dropped.size else 0.0
    vmax = vmax if vmax > 0 else 1e-9
    # The residual scale is floored at the largest gate tolerance, never at the
    # largest residual. Without the floor a run where every residual is float
    # round-off (1e-13 MW, which is what an exact reconstruction looks like)
    # renders as a saturated red map: correct arithmetic drawn as a catastrophe.
    # With it, "inside its own tolerance" is pale by construction and only a
    # real relocation reaches the ends of the ramp.
    gate = float(np.nanmax(df["gate_tol_mw"].to_numpy(dtype=float))) if len(df) else 0.0
    dmax = float(np.nanmax(np.abs(residual.to_numpy(dtype=float)))) if len(residual) else 0.0
    dmax = max(dmax, gate)
    dmax = dmax if dmax > 0 else 1e-9

    fig, axes = plt.subplots(1, len(carriers) + 1, figsize=(5.4 * (len(carriers) + 1), 4.6), squeeze=False)
    flat = list(axes[0])
    for ax, carrier in zip(flat, carriers):
        _zone_panel(
            fig,
            ax,
            zones,
            dropped[carrier],
            "viridis",
            0.0,
            vmax,
            f"{carrier} dropped by master\n[MW] (grey = no data)",
        )
    _zone_panel(
        fig,
        flat[-1],
        zones,
        residual,
        DELTA_CMAP,
        -dmax,
        dmax,
        "residual: (develop - master) - dropped\n[MW] (grey = no data)",
    )
    fig.suptitle(_recon_title(df), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    data = dropped.add_suffix("_dropped_mw").join(residual.rename("residual_mw"))
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
    master_label: str | None = None,
    subtitle: str = "",
) -> tuple[Path, Path]:
    """Sorted-descending distribution of two value sets on a common percentile axis.

    Sampling both sides at the same percentiles is what makes them comparable
    when their bus spaces differ, which they do at prong 2.

    ``master_label`` and ``subtitle`` carry the provenance of the master series
    onto the figure — at prong 2 it is not the file master wrote but that file
    rolled up to ``s{simpl}``, and a reader comparing two curves has to be told
    which object the blue one is.
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
    ax1.plot(q, ym, color=SIDE_COLORS["master"], lw=2.0, label=master_label or LABELS["master"])
    ax1.plot(q, yd, color=SIDE_COLORS["develop"], lw=2.0, ls="--", label=LABELS["develop"])
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_title(title + (f"\n{subtitle}" if subtitle else ""), fontsize=11)
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
    master_label: str | None = None,
    subtitle: str = "",
) -> tuple[Path, Path]:
    """Two time series plus their relative difference.

    A full benchmark year is resampled to daily means to stay legible; a short
    series (fewer than three days, as in the unit tests) is drawn at its native
    resolution, since a one-point daily mean plots nothing at all.

    ``master_label`` / ``subtitle`` name the object the master series came from;
    see :func:`duration_curve`.
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
    ax1.plot(idx, ym, color=SIDE_COLORS["master"], lw=1.6, marker=marker, ms=4, label=master_label or LABELS["master"])
    ax1.plot(idx, yd, color=SIDE_COLORS["develop"], lw=1.6, ls="--", marker=marker, ms=4, label=LABELS["develop"])
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_title(title + (f"\n{subtitle}" if subtitle else ""), fontsize=11)
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
        df.groupby([df["stage"].astype(str), df["waived"]])
        .size()
        .unstack(fill_value=0)
        .rename(
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
class PreparedProfile:
    """One tech's two profile Datasets, put on ONE bus space, plus how.

    This is the object every prong-2 profile number — table row and figure
    alike — is taken from. ``master`` is the rolled-up master when
    :attr:`rolled_up`, and the raw file otherwise; ``zone_master`` is the bus
    -> ``reeds_zone`` map that addresses whichever it is.
    """

    master: object
    develop: object
    zone_master: pd.Series | None = None
    rolled_up: bool = False
    #: What :func:`metrics.cluster_sets` reported, or ``None`` at prong 1.
    info: dict | None = None

    @property
    def master_label(self) -> str:
        """Legend text for the master series, naming the rollup when there was one."""
        if not self.rolled_up:
            return LABELS["master"]
        return f"{LABELS['master']}, rolled up to s{SIMPL2}"

    @property
    def subtitle(self) -> str:
        """One line under the title saying which object the master side is."""
        if not self.rolled_up:
            return ""
        n = (self.info or {}).get("n_common")
        common = f"; {n} common clusters" if n is not None else ""
        return f"master rolled up to s{SIMPL2} (p_nom_max-weighted){common}"


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
    #: The ASSEMBLED networks — develop's ``elec_s{simpl}_l_pp.pkl`` (dill) and
    #: master's ``elec_s{simpl}.nc``. They are the earliest stage at which the
    #: existing fleet is attached, so they are where existing capacity is
    #: compared when the run stops before ``cluster_network``
    #: (``EQ_UNTIL=assembled``) and no clustered network exists. They are NOT
    #: compared cell by cell: at prong 2 the two sides' simpl-stage kmeans
    #: differ by design, so only aggregate, clustering-invariant sums are taken
    #: from them (total MW by carrier, and MW by reeds_zone x carrier).
    assembled_master: object | None = None
    assembled_develop: object | None = None
    zone_master: pd.Series | None = None
    zone_develop: pd.Series | None = None
    zones: object | None = None
    #: develop's ``busmap_s{simpl}`` (substation id -> cluster bus). Prong 2
    #: only; it is what rolls master's nodal profiles onto develop's bus space.
    busmap: pd.Series | None = None
    #: ``{tech: metrics.cluster_sets(...)}`` — what the two cluster sets were
    #: after the prong-2 rollup, and what the one-sided clusters are worth in
    #: MW. Filled by :func:`collect_metrics`; the pooled profile metrics are
    #: computed over the common clusters only, so this is the record of what
    #: they left out.
    cluster_sets: dict[str, dict] = field(default_factory=dict)
    #: ``{tech: PreparedProfile}`` — the datasets the profile metrics were taken
    #: from, cached by :func:`prepare_profiles` at prong 2 so ``export_all``
    #: draws its figures from the same object instead of reopening the raw
    #: master file. Cluster-resolution, so keeping it costs almost nothing.
    profiles: dict[str, PreparedProfile] = field(default_factory=dict)

    @property
    def profile_pairs(self) -> list:
        """The prong's renewable-profile artifact pairs."""
        return [p for p in prong_pairs(self.prong) if p.kind == "profile"]


def _zone_maps(develop_root: Path):
    """(develop bus->zone, master substation->zone, zone shapes or None)."""
    import geopandas as gpd
    import pypsa

    from .compare import load_busmap

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
        busmap = load_busmap(develop_root)
        if busmap is not None:
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
    ad = art.develop_root / assembled_target(prong)
    am = art.master_root / baseline_assembled_target(prong)
    if ad.exists() and am.exists():
        # The assembled pair is not in prong_pairs at prong 2 (the two sides'
        # simpl-stage kmeans differ, so a cell-by-cell comparison is
        # meaningless), but the existing fleet it carries is a sum, and a sum
        # over a different partition of the same buses is the same number.
        # Without this, `EQ_UNTIL=assembled` produced no existing-capacity
        # comparison at all and a 3.5 GW onwind difference (HF-26) was invisible
        # to the table.
        art.assembled_develop, art.assembled_master = load_network(ad), load_network(am)
    art.zone_develop, art.zone_master, art.zones = _zone_maps(art.develop_root)
    if prong == 2:
        from .compare import load_busmap

        art.busmap = load_busmap(art.develop_root)
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


def prepare_profiles(
    art: Artifacts,
    tech: str,
    ds_develop,
    ds_master_raw,
    missing: list[dict],
) -> PreparedProfile | None:
    """Put one tech's two profile Datasets on a single bus space. The only place.

    At prong 2 master's file is NODAL (substation buses) and develop's is at
    s{simpl}. Every profile statistic is resolution-sensitive — pooled
    ``(time, bus)`` quantiles over 544 sites and over the 19 clusters they roll
    up into are different numbers no matter what the refactor did — so master is
    aggregated onto develop's bus space first, and the clusters only one side
    carries are then removed (they are a row-set difference, reported once as
    the ``cluster_set`` finding, not smeared over every quantile row).

    Master's buses are cluster ids afterwards, so the returned
    ``zone_master`` is develop's cluster->zone map rather than the
    substation->zone chain that addressed the nodal file.

    Returns ``None`` only when the rollup itself failed; ``_safe`` has then
    recorded a ``MISSING`` row, which is the honest outcome — computing the
    metrics at mismatched resolutions would publish numbers that mean nothing.
    Prong 1 (and a prong-2 run with no busmap) passes both sides through
    untouched, with ``rolled_up=False``.

    The result is cached on ``art.profiles[tech]`` when it was rolled up, and
    the two Datasets are then in memory, so ``export_all`` can draw from it
    after the files are closed. A prong-1 pass-through is NOT cached: those
    Datasets are the nodal files themselves, and holding a whole-USA one open
    for the length of a run costs hundreds of MB for no gain.
    """
    if not (art.prong == 2 and art.busmap is not None and "bus" in getattr(ds_master_raw, "dims", ())):
        return PreparedProfile(
            master=ds_master_raw,
            develop=ds_develop,
            zone_master=art.zone_master,
            rolled_up=False,
        )
    agg = _safe(
        f"aggregate_master_profile_{tech}",
        missing,
        metrics.aggregate_profile_to_clusters,
        ds_master_raw,
        art.busmap,
    )
    if agg is None:
        return None
    dsm, dsd_cmp, info = metrics.common_cluster_subset(agg, ds_develop)
    prep = PreparedProfile(
        # ``load()``: ``dsd_cmp`` is a view on a file-backed Dataset, and the
        # cache outlives the ``open_dataset`` context it was built in.
        master=dsm,
        develop=dsd_cmp.load(),
        zone_master=art.zone_develop,
        rolled_up=True,
        info=info,
    )
    art.cluster_sets[tech] = info
    art.profiles[tech] = prep
    return prep


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
    # Existing capacity is attached at the assembled stage and is conserved by
    # clustering, so either stage answers the question. The clustered networks
    # are preferred when they exist (unchanged behaviour for a full run); the
    # assembled ones are what an `EQ_UNTIL=assembled` run has, and without them
    # the geographic-assignment criterion is simply absent from the table.
    existing = pre if all(x is not None for x in pre) else (art.assembled_master, art.assembled_develop)
    if all(x is not None for x in existing):
        out["capacity_existing_by_carrier"] = _safe(
            "capacity_existing_by_carrier",
            miss,
            metrics.capacity_by_carrier,
            *existing,
            attr="p_nom",
        )
        out["p_nom_existing_by_zone_carrier"] = _safe(
            "p_nom_existing_by_zone_carrier",
            miss,
            metrics.capacity_by_zone_carrier,
            *existing,
            attr="p_nom",
        )
    if all(x is not None for x in pre):
        out["demand_by_zone"] = _safe("demand_by_zone", miss, metrics.demand_by_zone, *pre)
    if all(x is not None for x in solved):
        out["objective"] = _safe("objective", miss, metrics.objective_row, *solved)
        out["capacity_opt_by_carrier"] = _safe(
            "capacity_opt_by_carrier",
            miss,
            metrics.capacity_by_carrier,
            *solved,
            attr="p_nom_opt",
        )
        out["dispatch_by_carrier"] = _safe("dispatch_by_carrier", miss, metrics.dispatch_by_carrier, *solved)
        out["capacity_factor_by_carrier"] = _safe(
            "capacity_factor_by_carrier",
            miss,
            metrics.capacity_factor_by_carrier,
            *solved,
        )
    for pair in art.profile_pairs:
        tech = pair.stage.replace("profile_", "")
        dp, mp = art.develop_root / pair.develop, art.master_root / pair.master
        if not (dp.exists() and mp.exists()):
            continue
        with xr.open_dataset(dp) as dsd, xr.open_dataset(mp) as dsm_raw:
            # One preparation, shared with the figures (see prepare_profiles).
            prep = prepare_profiles(art, tech, dsd, dsm_raw, miss)
            if prep is None:
                continue
            dsm, dsd_cmp, zone_m = prep.master, prep.develop, prep.zone_master
            out[f"p_max_pu_quantiles_{tech}"] = _safe(
                f"p_max_pu_quantiles_{tech}",
                miss,
                metrics.p_max_pu_quantiles,
                dsm,
                dsd_cmp,
            )
            out[f"p_nom_max_by_zone_{tech}"] = _safe(
                f"p_nom_max_by_zone_{tech}",
                miss,
                metrics.p_nom_max_by_zone,
                dsm,
                dsd_cmp,
                zone_m,
                art.zone_develop,
            )
            out[f"mean_cf_by_zone_{tech}"] = _safe(
                f"mean_cf_by_zone_{tech}",
                miss,
                metrics.mean_cf_by_zone,
                dsm,
                dsd_cmp,
                zone_m,
                art.zone_develop,
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
        _record_cluster_sets(art)
    return _RUN_METRICS[key]


def _record_cluster_sets(art: Artifacts) -> None:
    """Put the rollup facts behind the TABLE numbers into ``run_meta.json``.

    ``compare.run_comparison`` records the same thing from the findings side,
    but the two sides do not always both run: at ``EQ_UNTIL=full`` the profile
    pairs are dropped from the comparison while the table still computes every
    profile metric. Whichever ran last is the one that describes the numbers a
    reader is holding, so both write the field and they agree by construction —
    same busmap, same files, same ``metrics.cluster_sets``.
    """
    if art.prong != 2 or not art.cluster_sets:
        return
    from . import context

    records = [
        {"kind": "profile_rollup", "stage": f"profile_{tech}", "rolled_up": True, **info}
        for tech, info in sorted(art.cluster_sets.items())
    ]
    try:
        context.update_run_meta(
            master_profile_stage=context.master_profile_stage(art.prong, rolled_up=True),
            profile_cluster_sets=records,
        )
    except OSError as exc:  # pragma: no cover - provenance must never kill a run
        print(f"[plots] could not update run_meta.json: {exc}")


def _hf26_reconstruction(artifacts, frames, reconstructions=None):
    """The HF-26 reconstruction for this run, or ``None``.

    ``None`` when the run has no existing-capacity criterion to reconstruct (a
    solve-only export), or when ``EQ_RECONSTRUCTIONS=0`` emptied the registry.
    Both cases draw a labelled placeholder rather than skipping the figure, so a
    missing reconstruction is visible in ``figures/`` instead of absent from it.
    """
    if (frames or {}).get("p_nom_existing_by_zone_carrier") is None:
        return None
    from . import reconstructions as recon_mod

    reg = recon_mod.registry(artifacts, frames) if reconstructions is None else reconstructions
    return reg.get("hf26_existing_renewable_drop") if reg else None


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
    reconstructions=None,
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
    paired_bar(
        m.get("capacity_existing_by_carrier"),
        "existing capacity",
        "MW",
        "capacity_existing_by_carrier",
        run_dir,
    )
    paired_bar(m.get("capacity_opt_by_carrier"), "optimised capacity", "MW", "capacity_opt_by_carrier", run_dir)
    paired_bar(m.get("dispatch_by_carrier"), "annual dispatch", "MWh", "dispatch_by_carrier", run_dir)
    paired_bar(
        m.get("capacity_factor_by_carrier"),
        "realised capacity factor",
        "-",
        "capacity_factor_by_carrier",
        run_dir,
    )
    paired_bar(m.get("demand_by_zone"), "demand by zone", "MW", "demand_zones", run_dir)

    # HF-26's computed explanation, drawn whatever the verdicts were. The table
    # only resolves a reconstruction when a row needs it; the figures are a
    # deliverable in their own right (PROJECT.md 3.2), so they ask for it
    # explicitly and the memoised registry keeps that to one computation.
    recon = _hf26_reconstruction(artifacts, m, reconstructions)
    hf26_dropped_zone_figure(recon, "hf26_dropped_mw_by_zone", run_dir)
    hf26_dropped_map_figure(artifacts.zones, recon, "hf26_dropped_mw_map", run_dir)

    zc = m.get("p_nom_existing_by_zone_carrier")
    if zc is not None and not zc.empty:
        top = zc.groupby(level="carrier")["master"].sum().sort_values(ascending=False)
        for carrier in list(top.index)[:zone_carriers]:
            vm, vd = _carrier_zone_slice(zc, carrier)
            choropleth_triptych(
                artifacts.zones,
                vm,
                vd,
                f"{carrier} existing capacity",
                "MW",
                f"p_nom_existing_zones_{carrier}",
                run_dir,
            )

    for pair in artifacts.profile_pairs:
        tech = pair.stage.replace("profile_", "")
        # The prepared pair the TABLE metrics were taken from. Reopening the raw
        # master file here is what made p_max_pu_duration_onwind plot 544 pooled
        # substations against 19 clusters while the quantile rows beside it were
        # equivalent; the figure and its row must be one object.
        prep = artifacts.profiles.get(tech)
        with ExitStack() as stack:
            if prep is None:
                dp, mp = artifacts.develop_root / pair.develop, artifacts.master_root / pair.master
                if not (dp.exists() and mp.exists()):
                    print(f"[plots] skipping {tech}: artifact missing")
                    continue
                dsd = stack.enter_context(xr.open_dataset(dp))
                dsm_raw = stack.enter_context(xr.open_dataset(mp))
                prep = prepare_profiles(artifacts, tech, dsd, dsm_raw, missing)
                if prep is None:
                    print(f"[plots] skipping {tech}: master profile could not be rolled up")
                    continue
            dsm, dsd = prep.master, prep.develop
            pot = m.get(f"p_nom_max_by_zone_{tech}")
            cf = m.get(f"mean_cf_by_zone_{tech}")
            if pot is not None:
                choropleth_triptych(
                    artifacts.zones,
                    pot["master"] / 1e3,
                    pot["develop"] / 1e3,
                    f"{tech} installable potential",
                    "GW",
                    f"{tech}_potential_zones",
                    run_dir,
                )
            if cf is not None:
                choropleth_triptych(
                    artifacts.zones,
                    cf["master"],
                    cf["develop"],
                    f"{tech} potential-weighted mean CF",
                    "-",
                    f"{tech}_meancf_zones",
                    run_dir,
                )
            duration_curve(
                pd.Series(metrics.profile_values(dsm)),
                pd.Series(metrics.profile_values(dsd)),
                f"{tech} capacity factor",
                "-",
                f"p_max_pu_duration_{tech}",
                run_dir,
                color=carrier_color(tech),
                master_label=prep.master_label,
                subtitle=prep.subtitle,
            )
            timeseries_pair(
                metrics.available_power(dsm) / 1e3,
                metrics.available_power(dsd) / 1e3,
                f"{tech}: sum over buses of profile x p_nom_max",
                "GW",
                f"{tech}_national_available_power",
                run_dir,
                master_label=prep.master_label,
                subtitle=prep.subtitle,
            )

    # Rewritten: a rollup that failed inside the loop above (only possible when
    # the caller did not go through collect_metrics first) is a criterion lost,
    # and the file written before the loop would not name it.
    (run_dir / "missing_metrics.json").write_text(json.dumps(missing, indent=1))

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
