"""
Render the technology-capacity-target (TCT) map for the documentation.

Draws the three policy families shipped in
``repo_data/config/policy_constraints/technology_capacity_targets.csv`` as three
CONUS panels: the ReEDS nuclear no-build regions (``max = existing``), the ReEDS
forced retirements (``max = 0``) and the ReEDS storage mandates (annual ``min``
MW of ``4hr_battery_storage``). A target's ``region`` may be a state code or a
ReEDS zone id, and both are drawn.

Only the shipped defaults are mapped; a user pointing
``electricity: technology_capacity_targets`` at another CSV gets other targets.

Needs only the ReEDS zone shapes and membership shipped in ``repo_data``.
Run via ``snakemake docs_tct_targets`` (see ``workflow/Snakefile``).
"""

import logging

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)

CRS = "EPSG:5070"
# categorical fills follow plot_rec_trading_zones.ZONE_COLORS
STATE_COLOR = "#2f6fd6"
ZONE_COLOR = "#1b998b"
RETIRE_COLOR = "#a52a2a"
NEUTRAL_COLOR = "#e6eaee"
STORAGE_CMAP = "Blues"
STORAGE_HORIZON = 2030

# small north-eastern states get their label pushed off the map body
LABEL_OFFSETS = {
    "MA": (44, 30),
    "NJ": (52, -8),
    "MD": (46, -34),
}
# forced-retirement callout placement, so the three labels do not collide
RETIRE_OFFSETS = {"IL": (-30, 52), "NY": (-8, 58), "VA": (36, -46)}
CARRIERS_PER_LINE = 3


def _wrap_carriers(carriers):
    """'CCGT/OCGT/coal/' + newline + 'coal-95CCS/oil' — long lists overflow the panel."""
    names = sorted(carriers)
    chunks = [names[i : i + CARRIERS_PER_LINE] for i in range(0, len(names), CARRIERS_PER_LINE)]
    return "\n".join("/".join(c) + ("/" if i < len(chunks) - 1 else "") for i, c in enumerate(chunks))


def _retirement_label(state, group):
    """One block per horizon: 'coal/coal-95CCS 2024', newest last."""
    lines = [
        f"{_wrap_carriers(g.carrier)} {h}" for h, g in sorted(group.groupby("planning_horizon"), key=lambda kv: kv[0])
    ]
    return "\n".join([state, *lines])


def _fmt_mw(value):
    return f"{value:g}" if value < 10 else f"{value:,.0f}"


def _label_point(geom):
    """Representative point of the largest part, so Michigan is labelled in its peninsula."""
    parts = list(getattr(geom, "geoms", [geom]))
    return max(parts, key=lambda g: g.area).representative_point()


def _annotate(ax, geom, text, offset=None, color="#1f2933", fontsize=7, weight="normal"):
    """Label a geometry at its representative point, optionally offset with a leader."""
    p = _label_point(geom)
    if offset is None:
        ax.annotate(
            text,
            (p.x, p.y),
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            fontweight=weight,
        )
        return
    ax.annotate(
        text,
        (p.x, p.y),
        xytext=offset,
        textcoords="offset points",
        ha="center" if offset[0] == 0 else "left" if offset[0] > 0 else "right",
        va="bottom" if offset[1] > 0 else "top" if offset[1] < 0 else "center",
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#b0b8c0", "linewidth": 0.5},
        arrowprops={"arrowstyle": "-", "color": "#5a6672", "linewidth": 0.6},
    )


def _panel_legend(ax, handles):
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8,
        frameon=False,
        ncol=1,
    )


def _load_geometry(shapes_path, membership_path):
    """Return (states dissolved by code, ReEDS zones), both CONUS and EPSG:5070."""
    zones = gpd.read_file(shapes_path).to_crs(CRS)
    membership = pd.read_csv(membership_path).rename(columns={"ba": "rb"})
    zones = zones.merge(membership[["rb", "st", "country"]], on="rb", how="left")
    zones = zones[zones.country == "USA"]
    states = zones.dissolve(by="st").reset_index()
    # close the hairline gaps between ReEDS zone polygons so the outlines draw clean
    states["geometry"] = states.geometry.buffer(500).buffer(-500)
    zones = zones.dissolve(by="rb").reset_index()
    zones["geometry"] = zones.geometry.buffer(500).buffer(-500)
    return states, zones


def _panel_nuclear(ax, targets, states, zones):
    rows = targets[targets.name == "reeds_nuclear_no_build"]
    state_regions = sorted(set(rows.region) & set(states.st))
    zone_regions = sorted(set(rows.region) & set(zones.rb))

    states.plot(ax=ax, color=NEUTRAL_COLOR, edgecolor="white", linewidth=0.7)
    hit_states = states[states.st.isin(state_regions)]
    hit_zones = zones[zones.rb.isin(zone_regions)]
    hit_states.plot(ax=ax, color=STATE_COLOR, edgecolor="white", linewidth=0.7)
    hit_zones.plot(ax=ax, color=ZONE_COLOR, edgecolor="white", linewidth=0.5)

    for _, s in hit_states.iterrows():
        offset = LABEL_OFFSETS.get(s.st)
        _annotate(ax, s.geometry, s.st, offset, color="#1f2933" if offset else "white")
    # the target zones are small New England polygons: one callout beats six stacked labels
    if not hit_zones.empty:
        _annotate(ax, hit_zones.union_all(), f"{len(zone_regions)} ReEDS zones", (40, 34), fontsize=6.5)

    _panel_legend(
        ax,
        [
            Patch(
                facecolor=STATE_COLOR,
                edgecolor="white",
                label=f"state region ({len(state_regions)}): {', '.join(state_regions)}",
            ),
            Patch(
                facecolor=ZONE_COLOR,
                edgecolor="white",
                label=f"ReEDS zone region ({len(zone_regions)}): {', '.join(zone_regions)}",
            ),
            Patch(facecolor=NEUTRAL_COLOR, edgecolor="white", label="no target"),
        ],
    )
    ax.set_title(
        f"Nuclear no-build (max = existing, all horizons)\n{len(rows)} rows: "
        f"{len(state_regions)} states, {len(zone_regions)} ReEDS zones; carrier 'nuclear, SMR'",
        fontsize=10,
        loc="left",
        fontweight="bold",
    )


def _panel_retirements(ax, targets, states):
    rows = targets[targets.name == "reeds_forced_retirements"]
    hit = states[states.st.isin(rows.region.unique())]

    states.plot(ax=ax, color=NEUTRAL_COLOR, edgecolor="white", linewidth=0.7)
    hit.plot(ax=ax, color=RETIRE_COLOR, edgecolor="white", linewidth=0.7)

    for _, s in hit.iterrows():
        label = _retirement_label(s.st, rows[rows.region == s.st])
        _annotate(ax, s.geometry, label, RETIRE_OFFSETS.get(s.st, (40, 30)), fontsize=6.5)

    _panel_legend(
        ax,
        [
            Patch(facecolor=RETIRE_COLOR, edgecolor="white", label=f"forced retirement ({len(hit)} states)"),
            Patch(facecolor=NEUTRAL_COLOR, edgecolor="white", label="no target"),
        ],
    )
    horizons = ", ".join(sorted(rows.planning_horizon.astype(str).unique()))
    ax.set_title(
        f"Forced retirements (max = 0)\n{len(rows)} rows over {len(hit)} states; horizons {horizons}",
        fontsize=10,
        loc="left",
        fontweight="bold",
    )


def _panel_storage(ax, targets, states):
    rows = targets[targets.name == "reeds_storage_mandate"].copy()
    rows["horizon"] = rows.planning_horizon.astype(int)
    by_state = rows.pivot_table(index="region", columns="horizon", values="min")
    if STORAGE_HORIZON in by_state.columns and by_state[STORAGE_HORIZON].notna().all():
        values, shown = by_state[STORAGE_HORIZON], STORAGE_HORIZON
    else:
        values, shown = by_state[by_state.columns.max()], int(by_state.columns.max())

    states.plot(ax=ax, color=NEUTRAL_COLOR, edgecolor="white", linewidth=0.7)
    hit = states[states.st.isin(values.index)].copy()
    hit["value"] = hit.st.map(values)
    norm = Normalize(vmin=0, vmax=values.max())
    cmap = plt.get_cmap(STORAGE_CMAP)
    hit.plot(ax=ax, color=[cmap(norm(v)) for v in hit.value], edgecolor="white", linewidth=0.7)

    for _, s in hit.iterrows():
        offset = LABEL_OFFSETS.get(s.st)
        dark = norm(s.value) > 0.55 and offset is None
        _annotate(
            ax,
            s.geometry,
            f"{s.st}\n{_fmt_mw(s.value)}",
            offset,
            color="white" if dark else "#1f2933",
            fontsize=6.5,
        )

    # an inset below the map keeps this panel the same size as the other two
    cax = ax.inset_axes([0.18, -0.06, 0.6, 0.035])
    cbar = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cbar.set_label(f"{shown} minimum capacity [MW]", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    horizons = f"{rows.horizon.min()}-{rows.horizon.max()}"
    ax.set_title(
        f"Storage mandates (min MW, 4hr_battery_storage)\n{len(rows)} rows over {len(values)} states, "
        f"horizons {horizons};\nshaded by the {shown} minimum",
        fontsize=10,
        loc="left",
        fontweight="bold",
    )


def plot_tct_targets(tct_path, shapes_path, membership_path, out_path):
    targets = pd.read_csv(tct_path)
    states, zones = _load_geometry(shapes_path, membership_path)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.4), dpi=150)
    _panel_nuclear(axes[0], targets, states, zones)
    _panel_retirements(axes[1], targets, states)
    _panel_storage(axes[2], targets, states)
    xmin, ymin, xmax, ymax = states.total_bounds
    padx, pady = 0.04 * (xmax - xmin), 0.06 * (ymax - ymin)
    for ax in axes:
        # identical framing and top-anchored boxes, so the three maps and titles line up
        ax.set_xlim(xmin - padx, xmax + padx)
        ax.set_ylim(ymin - pady, ymax + pady)
        ax.set_anchor("N")
        ax.set_axis_off()

    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.12, wspace=0.04)
    fig.suptitle(
        "Default technology capacity targets — "
        "repo_data/config/policy_constraints/technology_capacity_targets.csv "
        f"({len(targets)} rows, {targets.name.nunique()} policy families)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("docs_tct_targets")
    logging.basicConfig(level=logging.INFO)
    plot_tct_targets(
        snakemake.input.tct,
        snakemake.input.reeds_shapes,
        snakemake.input.reeds_memberships,
        snakemake.output.tct_targets,
    )
