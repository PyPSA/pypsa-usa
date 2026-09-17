"""Figures for the "simplify-early" deck.

Every figure writes a PNG and a CSV twin holding exactly the plotted numbers,
into ``figures/`` next to this file. Deltas are always develop minus master and
always live on their own panel with a zero line - never on a second y-axis.

Colour scheme (fixed, two categorical slots, validated colourblind-safe):
    master  #2a78d6 (blue)
    develop #eb6834 (orange)
Delta bars use the diverging pair blue / red about a neutral zero line, where
blue means develop is lower than master and red means develop is higher.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import measurements as M

FIGDIR = Path(__file__).parent / "figures"

# --- colour roles ---------------------------------------------------------
MASTER = "#2a78d6"
DEVELOP = "#eb6834"
NEG = "#2a78d6"  # develop lower than master
POS = "#e34948"  # develop higher than master
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#dedcd6"
SURFACE = "#fcfcfb"
NEUTRAL_FILL = "#eceae4"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "axes.labelcolor": INK2,
        "axes.edgecolor": GRID,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _write_csv(name: str, header: list[str], rows: list[list]) -> Path:
    path = FIGDIR / f"{name}.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def _save(fig, name: str) -> Path:
    path = FIGDIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ==========================================================================
# 1 + 2. DAG diagrams
# ==========================================================================

# Each column is (title, [box labels], bus badge, kind)
#   kind: "plain" | "heavy" | "reduce" | "solve"
MASTER_DAG = [
    ("build_base_network", ["build_base\nnetwork"], "82,549", "plain"),
    ("build_bus_regions", ["build_bus\nregions"], "82,549", "plain"),
    (
        "per-bus heavy rules",
        ["build_renewable\nprofiles", "build_electrical\ndemand"],
        "82,549",
        "heavy",
    ),
    ("add_demand", ["add_demand"], "82,549", "heavy"),
    ("add_electricity", ["add_electricity"], "82,549", "heavy"),
    ("simplify_network", ["simplify\nnetwork"], "82,549 -> 300", "reduce"),
    ("cluster_network", ["cluster\nnetwork"], "300 -> 134", "reduce"),
    ("solve", ["prepare\nnetwork", "solve\nnetwork"], "134", "solve"),
]

DEVELOP_DAG = [
    ("build_base_network", ["build_base\nnetwork"], "82,549", "plain"),
    ("build_bus_regions", ["build_bus\nregions"], "82,549", "plain"),
    ("aggregate_to_substations", ["aggregate_to\nsubstations"], "82,549 -> 41,012", "reduce"),
    ("cluster_resources", ["cluster\nresources"], "41,012 -> 300", "reduce"),
    (
        "per-bus heavy rules",
        ["build_renewable\nprofiles", "build_electrical\ndemand"],
        "300",
        "heavy",
    ),
    ("add_demand", ["add_demand"], "300", "heavy"),
    ("add_electricity", ["add_electricity"], "300", "heavy"),
    ("cluster_network", ["cluster\nnetwork"], "300 -> 134", "reduce"),
    ("solve", ["prepare\nnetwork", "solve\nnetwork"], "134", "solve"),
]


def _dag_figure(spec, accent: str, title: str, subtitle: str, name: str) -> tuple[Path, Path]:
    n = len(spec)
    fig, ax = plt.subplots(figsize=(13.0, 5.4))
    ax.set_axis_off()
    ax.grid(False)

    colw = 1.0 / n
    box_w = colw * 0.80
    y_top = 0.72  # vertical centre of every column
    badge_y = 0.455  # fixed row for the bus-count badges, below the tallest column

    fills = {
        "plain": NEUTRAL_FILL,
        "heavy": accent,
        "reduce": "#ffffff",
        "solve": NEUTRAL_FILL,
    }
    edges = {
        "plain": MUTED,
        "heavy": accent,
        "reduce": accent,
        "solve": MUTED,
    }

    centers = []
    for i, (_, labels, badge, kind) in enumerate(spec):
        cx = colw * (i + 0.5)
        centers.append(cx)
        nb = len(labels)
        bh = 0.19
        gap = 0.04
        total_h = nb * bh + (nb - 1) * gap
        y0 = y_top - total_h / 2

        for j, lab in enumerate(labels):
            yy = y0 + j * (bh + gap)
            fc = fills[kind]
            ec = edges[kind]
            lw = 2.0 if kind in ("heavy", "reduce") else 1.2
            box = FancyBboxPatch(
                (cx - box_w / 2, yy),
                box_w,
                bh,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                facecolor=fc,
                edgecolor=ec,
                linewidth=lw,
                zorder=3,
            )
            ax.add_patch(box)
            txt_col = "#ffffff" if kind == "heavy" else INK
            ax.text(
                cx,
                yy + bh / 2,
                lab,
                ha="center",
                va="center",
                fontsize=7.6,
                color=txt_col,
                zorder=4,
                linespacing=1.35,
            )

        # bus-count badge, on a fixed row so columns of different heights line up
        ax.text(
            cx,
            badge_y,
            badge,
            ha="center",
            va="top",
            fontsize=8.0,
            color=INK,
            fontweight="bold",
            zorder=4,
        )
        ax.text(
            cx,
            badge_y - 0.058,
            "buses",
            ha="center",
            va="top",
            fontsize=6.8,
            color=MUTED,
            zorder=4,
        )

    # arrows between columns
    for i in range(n - 1):
        a = FancyArrowPatch(
            (centers[i] + box_w / 2 + 0.004, y_top),
            (centers[i + 1] - box_w / 2 - 0.004, y_top),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.1,
            color=MUTED,
            zorder=2,
        )
        ax.add_patch(a)

    # band under the heavy columns
    heavy_idx = [i for i, s in enumerate(spec) if s[3] == "heavy"]
    if heavy_idx:
        x0 = centers[heavy_idx[0]] - box_w / 2
        x1 = centers[heavy_idx[-1]] + box_w / 2
        ax.add_patch(
            FancyBboxPatch(
                (x0, 0.275),
                x1 - x0,
                0.062,
                boxstyle="round,pad=0.004,rounding_size=0.01",
                facecolor=accent,
                edgecolor="none",
                alpha=0.20,
                zorder=1,
            )
        )
        buses = spec[heavy_idx[0]][2]
        ax.text(
            (x0 + x1) / 2,
            0.306,
            f"per-bus heavy rules run at {buses} buses",
            ha="center",
            va="center",
            fontsize=9.5,
            color=INK,
            fontweight="bold",
            zorder=4,
        )

    # reduction-step legend marks
    ax.text(0.0, 0.205, subtitle, ha="left", va="top", fontsize=9.5, color=INK2)
    ax.text(0.0, 0.98, title, ha="left", va="top", fontsize=14.5, color=INK, fontweight="bold")

    # a small legend so the fills are never colour-alone
    ax.add_patch(
        FancyBboxPatch(
            (0.0, 0.02),
            0.022,
            0.045,
            boxstyle="round,pad=0.002,rounding_size=0.008",
            facecolor=accent,
            edgecolor=accent,
            zorder=3,
        )
    )
    ax.text(0.028, 0.042, "per-bus heavy rule", fontsize=8, color=INK2, va="center")
    ax.add_patch(
        FancyBboxPatch(
            (0.19, 0.02),
            0.022,
            0.045,
            boxstyle="round,pad=0.002,rounding_size=0.008",
            facecolor="#ffffff",
            edgecolor=accent,
            linewidth=2.0,
            zorder=3,
        )
    )
    ax.text(0.218, 0.042, "resolution-reduction rule", fontsize=8, color=INK2, va="center")
    ax.add_patch(
        FancyBboxPatch(
            (0.42, 0.02),
            0.022,
            0.045,
            boxstyle="round,pad=0.002,rounding_size=0.008",
            facecolor=NEUTRAL_FILL,
            edgecolor=MUTED,
            zorder=3,
        )
    )
    ax.text(0.448, 0.042, "other rule", fontsize=8, color=INK2, va="center")

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.0, 1.0)

    png = _save(fig, name)
    csvp = _write_csv(
        name,
        ["order", "stage", "rules", "buses_seen", "kind"],
        [[i + 1, s[0], " + ".join(s[1]), s[2], s[3]] for i, s in enumerate(spec)],
    )
    return png, csvp


def fig_dag_master():
    return _dag_figure(
        MASTER_DAG,
        MASTER,
        "master: reduce LAST",
        "Every per-bus heavy rule runs on the full nodal network. simplify_network does the substation\n"
        "aggregation and the {simpl} kmeans together, in one rule, after all the per-bus work is finished.",
        "fig1_dag_master",
    )


def fig_dag_develop():
    return _dag_figure(
        DEVELOP_DAG,
        DEVELOP,
        "develop: reduce FIRST (the simplify-early refactor)",
        "simplify_network is split in two and moved to the front: aggregate_to_substations (topology),\n"
        "then cluster_resources ({simpl} kmeans). Every per-bus heavy rule then consumes elec_s300.",
        "fig2_dag_develop",
    )


# ==========================================================================
# 3. Resolution ladder
# ==========================================================================


def fig_resolution_ladder():
    lad = M.USA_LADDER
    labels = [r.label for r in lad]
    vals = [r.buses for r in lad]

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    xs = range(len(lad))
    bars = ax.bar(xs, vals, width=0.5, color=[MUTED, MUTED, DEVELOP, MASTER], zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("buses in the network (log scale)")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(
        "USA case: where the network shrinks, and where each branch's heavy rules sit",
        fontsize=13,
        color=INK,
        loc="left",
        pad=14,
    )
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v * 1.18,
            f"{v:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=INK,
        )

    # annotate which branch's heavy rules run where
    # arrows land inside the bar so they never cross the value label above it
    ax.annotate(
        "master's heavy rules\nrun here",
        xy=(0, vals[0] * 0.45),
        xytext=(0.72, vals[0] * 2.2),
        fontsize=9.5,
        color=MASTER,
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color=MASTER, lw=1.6,
                        connectionstyle="arc3,rad=0.15"),
    )
    ax.annotate(
        "develop's heavy rules\nrun here",
        xy=(2, vals[2] * 0.62),
        xytext=(2.0, vals[2] * 18),
        fontsize=9.5,
        color=DEVELOP,
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color=DEVELOP, lw=1.6),
    )
    ax.text(
        0.99,
        0.96,
        f"reduction seen by the heavy rules: {M.REDUCTION_FACTOR:,.0f}x",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=INK,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#ffffff", edgecolor=GRID),
    )
    ax.set_ylim(60, vals[0] * 8)
    ax.grid(axis="x", visible=False)

    png = _save(fig, "fig3_resolution_ladder")
    csvp = _write_csv(
        "fig3_resolution_ladder",
        ["stage", "buses", "source"],
        [[r.label.replace("\n", " "), r.buses, r.source] for r in lad],
    )
    return png, csvp


# ==========================================================================
# 4 + 5. Per-rule wall time and peak memory
# ==========================================================================


def _paired_bar_with_delta(
    rules,
    value_master,
    value_develop,
    ylabel,
    delta_label,
    title,
    name,
    unit,
    log=False,
):
    pairs = [r for r in rules if r.measured_both]
    labels = [r.stage for r in pairs]
    mv = [value_master(r) for r in pairs]
    dv = [value_develop(r) for r in pairs]
    delta = [d - m for m, d in zip(mv, dv)]

    fig, (ax, axd) = plt.subplots(
        2,
        1,
        figsize=(12.6, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.0], "hspace": 0.12},
    )

    xs = range(len(pairs))
    w = 0.38
    b1 = ax.bar([x - w / 2 for x in xs], mv, w, label="master", color=MASTER, zorder=3)
    b2 = ax.bar([x + w / 2 for x in xs], dv, w, label="develop", color=DEVELOP, zorder=3)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="upper left", ncols=2, fontsize=10)
    ax.set_title(title, fontsize=13, color=INK, loc="left", pad=12)
    ax.grid(axis="x", visible=False)

    for bars, vals in ((b1, mv), (b2, dv)):
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() * (1.06 if log else 1.0) + (0 if log else max(mv + dv) * 0.015),
                f"{v:,.0f}",
                ha="center",
                va="bottom",
                fontsize=7.6,
                color=INK2,
            )

    cols = [NEG if d < 0 else POS for d in delta]
    axd.bar(list(xs), delta, 0.55, color=cols, zorder=3)
    axd.axhline(0, color=INK2, linewidth=1.1, zorder=4)
    axd.set_ylabel(delta_label)
    axd.set_xticks(list(xs))
    axd.set_xticklabels(labels, fontsize=8.2)
    axd.grid(axis="x", visible=False)
    span = max(abs(min(delta)), abs(max(delta))) or 1.0
    for x, d in zip(xs, delta):
        axd.text(
            x,
            d + span * (0.045 if d >= 0 else -0.045),
            f"{d:+,.0f}",
            ha="center",
            va="bottom" if d >= 0 else "top",
            fontsize=7.6,
            color=INK2,
        )
    # headroom so the outermost value label is never clipped
    axd.set_ylim(min(0, min(delta)) - span * 0.30, max(0, max(delta)) + span * 0.30)
    axd.text(
        0.995,
        0.06,
        f"blue: develop lower  |  red: develop higher   ({unit})",
        transform=axd.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )

    png = _save(fig, name)
    csvp = _write_csv(
        name,
        ["stage", f"master_{unit}", f"develop_{unit}", f"delta_develop_minus_master_{unit}", "ratio_master_over_develop", "master_benchmark_file", "develop_benchmark_file", "master_mtime", "develop_mtime"],
        [
            [
                r.stage.replace("\n", " "),
                m,
                d,
                round(d - m, 4),
                round(m / d, 3) if d else "",
                r.master_file,
                r.develop_file,
                r.master_mtime,
                r.develop_mtime,
            ]
            for r, m, d in zip(pairs, mv, dv)
        ],
    )
    return png, csvp


def fig_runtime_usa():
    return _paired_bar_with_delta(
        M.USA_RULES,
        lambda r: r.master_s,
        lambda r: r.develop_s,
        "wall time (s, log scale)",
        "develop - master (s)",
        "USA per-rule wall time, master vs develop (snakemake benchmark TSVs)",
        "fig4_runtime_usa",
        "s",
        log=True,
    )


def fig_memory_usa():
    return _paired_bar_with_delta(
        M.USA_RULES,
        lambda r: r.master_rss_mib,
        lambda r: r.develop_rss_mib,
        "peak RSS (MiB)",
        "develop - master (MiB)",
        "USA per-rule peak memory, master vs develop (snakemake benchmark TSVs)",
        "fig5_memory_usa",
        "MiB",
        log=False,
    )


# ==========================================================================
# 6. Artifact sizes
# ==========================================================================


def fig_artifact_sizes():
    arts = M.USA_ARTIFACTS
    labels = [a.label for a in arts]
    mv = [a.master_bytes / 1e6 for a in arts]
    dv = [a.develop_bytes / 1e6 for a in arts]
    ratio = [a.master_bytes / a.develop_bytes for a in arts]

    fig, (ax, axr) = plt.subplots(
        2,
        1,
        figsize=(12.0, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.12},
    )
    xs = range(len(arts))
    w = 0.38
    ax.bar([x - w / 2 for x in xs], mv, w, label="master", color=MASTER, zorder=3)
    ax.bar([x + w / 2 for x in xs], dv, w, label="develop", color=DEVELOP, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("artifact size (MB, log scale)")
    ax.legend(frameon=False, loc="upper left", ncols=2, fontsize=10)
    ax.set_ylim(min(dv) * 0.45, max(mv) * 4.0)
    ax.set_title(
        "The arrays the heavy rules write: same rule, same config, different resolution",
        fontsize=13,
        color=INK,
        loc="left",
        pad=12,
    )
    ax.grid(axis="x", visible=False)
    for x, v in zip(xs, mv):
        ax.text(x - w / 2, v * 1.14, f"{v:,.0f}", ha="center", va="bottom", fontsize=7.8, color=INK2)
    for x, v in zip(xs, dv):
        ax.text(x + w / 2, v * 1.14, f"{v:,.1f}", ha="center", va="bottom", fontsize=7.8, color=INK2)

    axr.bar(list(xs), ratio, 0.55, color=MUTED, zorder=3)
    axr.axhline(1, color=INK2, linewidth=1.1, zorder=4)
    axr.set_ylabel("master / develop\n(x smaller)")
    axr.set_xticks(list(xs))
    axr.set_xticklabels(labels, fontsize=8.6)
    axr.grid(axis="x", visible=False)
    for x, v in zip(xs, ratio):
        axr.text(x, v + max(ratio) * 0.04, f"{v:,.1f}x", ha="center", va="bottom", fontsize=9, fontweight="bold", color=INK)
    axr.set_ylim(0, max(ratio) * 1.28)

    png = _save(fig, "fig6_artifact_sizes")
    csvp = _write_csv(
        "fig6_artifact_sizes",
        ["artifact", "produced_by", "master_bytes", "develop_bytes", "master_MB", "develop_MB", "ratio_master_over_develop", "master_path", "develop_path", "note"],
        [
            [
                a.label.replace("\n", " "),
                a.produced_by,
                a.master_bytes,
                a.develop_bytes,
                round(a.master_bytes / 1e6, 3),
                round(a.develop_bytes / 1e6, 3),
                round(a.master_bytes / a.develop_bytes, 3),
                a.master_path,
                a.develop_path,
                a.note,
            ]
            for a in arts
        ],
    )
    return png, csvp


# ==========================================================================
# 7. Segment totals, with the unmeasured master rule made explicit
# ==========================================================================


def fig_segment_totals():
    t = M.heavy_totals(M.USA_RULES)
    new_rules = M.USA_DEVELOP_ONLY
    new_s = sum(r.develop_s for r in new_rules)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.6), gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.28})

    # -- left: stacked segment wall time
    labels = ["master", "develop"]
    heavy = [t["master_s"], t["develop_s"]]
    reduce_ = [0.0, new_s]

    ax.bar(labels, heavy, 0.5, label="per-bus heavy rules (5 rules)", color=[MASTER, DEVELOP], zorder=3)
    ax.bar(
        labels,
        reduce_,
        0.5,
        bottom=heavy,
        label="develop's new reduction rules",
        color="#ffffff",
        edgecolor=DEVELOP,
        linewidth=2.0,
        hatch="///",
        zorder=3,
    )
    # the unmeasured master piece
    ax.bar(
        ["master"],
        [600],
        0.5,
        bottom=[t["master_s"]],
        color="#ffffff",
        edgecolor=MUTED,
        linewidth=1.6,
        linestyle="--",
        hatch="xx",
        zorder=3,
        label="master simplify_network: NOT MEASURED",
    )
    ax.text(
        0,
        t["master_s"] + 300,
        "NOT\nMEASURED",
        ha="center",
        va="center",
        fontsize=8.5,
        color=INK2,
        fontweight="bold",
        zorder=5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="none"),
    )
    ax.text(0, t["master_s"] / 2, f"{t['master_s']:,.0f} s", ha="center", va="center", fontsize=12, color="#ffffff", fontweight="bold")
    ax.text(1, t["develop_s"] / 2, f"{t['develop_s']:,.0f} s", ha="center", va="center", fontsize=10, color="#ffffff", fontweight="bold")
    ax.text(1, t["develop_s"] + new_s / 2, f"{new_s:,.0f} s", ha="center", va="center", fontsize=10, color=INK)
    ax.set_ylabel("wall time (s)")
    ax.set_title("Build segment: heavy rules + the reduction step", fontsize=12, loc="left", color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8.4, loc="upper right", bbox_to_anchor=(1.0, 0.80))
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, (t["master_s"] + 600) * 1.16)

    # -- right: peak RSS across the same segment
    seg_m = t["master_peak_rss"]
    seg_d = max(t["develop_peak_rss"], max(r.develop_rss_mib for r in new_rules))
    bars = ax2.bar(
        ["master", "develop"],
        [seg_m, seg_d],
        0.5,
        color=[MASTER, DEVELOP],
        zorder=3,
    )
    for b, v, who in zip(bars, [seg_m, seg_d], ["add_demand", "aggregate_to_substations"]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 900, f"{v:,.0f} MiB", ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK)
        ax2.text(b.get_x() + b.get_width() / 2, v / 2, who, ha="center", va="center", fontsize=8.6, color="#ffffff", rotation=90)
    ax2.set_ylabel("peak RSS of the worst rule in the segment (MiB)")
    ax2.set_title("Segment memory peak moved, it did not fall", fontsize=12, loc="left", color=INK, pad=10)
    ax2.set_ylim(0, max(seg_m, seg_d) * 1.30)
    ax2.grid(axis="x", visible=False)
    ax2.text(
        0.03,
        0.985,
        "develop's peak is its new aggregate_to_substations rule,\n"
        "which master does inside the unmeasured simplify_network",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color=INK2,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor=GRID),
    )

    png = _save(fig, "fig7_segment_totals")
    csvp = _write_csv(
        "fig7_segment_totals",
        ["quantity", "master", "develop", "unit", "note"],
        [
            ["heavy per-bus rules, summed wall time", round(t["master_s"], 2), round(t["develop_s"], 2), "s", f"{t['n_rules']} rules measured on both branches"],
            ["develop-only reduction rules, summed wall time", 0, round(new_s, 2), "s", "aggregate_to_substations + cluster_resources"],
            ["master simplify_network wall time", "NOT MEASURED", "", "s", M.MASTER_UNMEASURED["why"]],
            ["segment wall time as measured", round(t["master_s"], 2), round(t["develop_s"] + new_s, 2), "s", "master value is a lower bound: simplify_network is missing from it"],
            ["worst-rule peak RSS in segment", round(seg_m, 2), round(seg_d, 2), "MiB", "master = add_demand; develop = aggregate_to_substations"],
        ],
    )
    return png, csvp


# ==========================================================================
# 8. Scale dependence: the western (California) counter-example
# ==========================================================================


def fig_scale_dependence():
    usa = M.heavy_totals(M.USA_RULES)
    west = M.heavy_totals(M.WESTERN_RULES)

    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    cases = [
        f"western (California)\n4,249 -> 20 buses",
        f"USA\n82,549 -> 300 buses",
    ]
    mv = [west["master_s"], usa["master_s"]]
    dv = [west["develop_s"], usa["develop_s"]]
    xs = range(2)
    w = 0.34
    ax.bar([x - w / 2 for x in xs], mv, w, label="master", color=MASTER, zorder=3)
    ax.bar([x + w / 2 for x in xs], dv, w, label="develop", color=DEVELOP, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("summed wall time of the 5 heavy rules (s, log scale)")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(cases, fontsize=10)
    ax.legend(frameon=False, ncols=2, fontsize=10, loc="upper left")
    ax.set_title("The saving scales with the problem: small case, small win", fontsize=13, loc="left", color=INK, pad=12)
    ax.grid(axis="x", visible=False)
    for x, m, d in zip(xs, mv, dv):
        ax.text(x - w / 2, m * 1.10, f"{m:,.0f} s", ha="center", va="bottom", fontsize=9.5, color=INK2)
        ax.text(x + w / 2, d * 1.10, f"{d:,.0f} s", ha="center", va="bottom", fontsize=9.5, color=INK2)
        ax.text(x, max(m, d) * 2.3, f"{m / d:,.1f}x", ha="center", va="bottom", fontsize=13, fontweight="bold", color=INK)
    ax.set_ylim(min(mv + dv) * 0.5, max(mv + dv) * 9)

    png = _save(fig, "fig8_scale_dependence")
    csvp = _write_csv(
        "fig8_scale_dependence",
        ["case", "master_summed_s", "develop_summed_s", "speedup_master_over_develop", "n_rules", "note"],
        [
            ["western (CA, 4249->20)", round(west["master_s"], 2), round(west["develop_s"], 2), round(west["master_s"] / west["develop_s"], 3), west["n_rules"], "smoke leg; per-rule TSVs under benchmarks/equivalence/western"],
            ["usa (82549->300)", round(usa["master_s"], 2), round(usa["develop_s"], 2), round(usa["master_s"] / usa["develop_s"], 3), usa["n_rules"], "acceptance case; per-rule TSVs under benchmarks/equivalence/usa"],
        ],
    )
    return png, csvp


ALL = [
    fig_dag_master,
    fig_dag_develop,
    fig_resolution_ladder,
    fig_runtime_usa,
    fig_memory_usa,
    fig_artifact_sizes,
    fig_segment_totals,
    fig_scale_dependence,
]


def build_all() -> dict[str, Path]:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for fn in ALL:
        png, csvp = fn()
        out[fn.__name__] = png
        print(f"  {png.name}  +  {csvp.name}")
    return out


if __name__ == "__main__":
    build_all()
