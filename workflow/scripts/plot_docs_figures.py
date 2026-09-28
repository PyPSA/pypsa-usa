"""
Render the canonical documentation figures from real workflow artifacts.

Produces the images embedded in the Model Description docs pages
(``docs/source/_static/generated/``) so that documentation figures are
regenerated from the current pipeline rather than hand-maintained:

- ``network_aggregation.png`` — the same system at nodal, ``{simpl}``, and
  ``{clusters}`` resolution, illustrating the two-stage spatial aggregation.
- ``cluster_suffixes.png`` — the same ``{simpl}`` network clustered with the
  four ``{clusters}`` wildcard variants (``N``, ``Nm``, ``Nc``, ``Na``),
  showing which generator carriers each variant aggregates.
- ``example_outputs.png`` — optimal capacity by carrier and a week of dispatch
  from a solved network.

Run via ``snakemake docs_figures`` (see ``workflow/Snakefile``).
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)

NET_KW = dict(geomap=False, bus_colors="firebrick")

# Used only if the rule does not hand in electricity:conventional_carriers.
FALLBACK_CONVENTIONAL_CARRIERS = [
    "nuclear",
    "oil",
    "OCGT",
    "CCGT",
    "coal",
    "geothermal",
    "biomass",
    "waste",
]

# The four `{clusters}` wildcard variants, in the order they are drawn, with
# the behaviour `cluster_network.py` implements for each.
SUFFIX_DESCRIPTIONS = {
    "": "all carriers aggregated",
    "m": "conventional aggregated; renewables keep {simpl} zones",
    "c": "renewables aggregated; conventional keep {simpl} zones",
    "a": "nothing aggregated",
}

GEN_COLORS = {"conventional": "#1f3a93", "renewable": "#2e9e5b"}


def _plot_network_panel(ax, n, shapes, title, extent=None):
    """Draw one network on ``ax``.

    ``extent`` is ``(xmin, xmax, ymin, ymax)``; when given, the panel is framed
    on it rather than on the network's own bus bounding box, so a heavily
    clustered network (a handful of buses) still shows the whole footprint.
    """
    if shapes is not None:
        shapes.plot(ax=ax, facecolor="#f0f0f0", edgecolor="white", linewidth=0.6)
    n.plot(
        ax=ax,
        bus_sizes=0.004,
        line_widths=n.lines.s_nom / n.lines.s_nom.max() * 2.2 if len(n.lines) else 0,
        link_widths=0.88,
        **NET_KW,
    )
    n_branches = len(n.lines) + len(n.links)
    if title is not None:
        ax.set_title(f"{title}\n({len(n.buses)} buses, {n_branches} branches)", fontsize=10)
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.axis("off")


def _footprint_extent(*networks, pad=0.06):
    """Padded bounding box of every bus in ``networks`` — the modelled footprint.

    The shapes file is not used for this: it covers the whole interconnect, while
    the model may be a slice of it (the tutorial is California only).
    """
    xs = pd.concat([n.buses.x for n in networks])
    ys = pd.concat([n.buses.y for n in networks])
    xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
    dx, dy = (xmax - xmin) * pad, (ymax - ymin) * pad
    return (xmin - dx, xmax + dx, ymin - dy, ymax + dy)


def plot_network_aggregation(base_path, simpl_path, clusters_path, shapes_path, out_path):
    """Three-panel map: nodal base network -> {simpl} zones -> {clusters} zones."""
    shapes = None
    if shapes_path:
        import geopandas as gpd

        shapes = gpd.read_file(shapes_path)

    networks = [pypsa.Network(path) for path in (base_path, simpl_path, clusters_path)]
    extent = _footprint_extent(*networks)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, n, title in zip(
        axes,
        networks,
        ["Nodal base network", "After cluster_simpl ({simpl})", "After cluster_network ({clusters})"],
    ):
        _plot_network_panel(ax, n, shapes, title, extent=extent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def _cluster_suffix(cluster_wildcard):
    """``"4m" -> "m"``, ``"4" -> ""`` — the carrier-aggregation suffix."""
    last = str(cluster_wildcard)[-1]
    return last if last in ("m", "c", "a") else ""


def _draw_generators(ax, n, simpl_buses, conventional, cap, jitter, rng):
    """Scatter ``n``'s generators: aggregated ones at their cluster bus, the rest at their {simpl} zone."""
    gens = n.generators
    if not len(gens):
        return
    # non-aggregated generators keep land_region = their {simpl} bus: draw them there
    land_region = gens.get("land_region", pd.Series(index=gens.index, dtype=object)).fillna("")
    at_simpl = land_region.isin(simpl_buses.index) & (land_region != gens.bus)
    anchor = gens.bus.where(~at_simpl, land_region)
    lookup = pd.concat([n.buses[["x", "y"]], simpl_buses[["x", "y"]]])
    lookup = lookup[~lookup.index.duplicated()]
    xy = lookup.loc[anchor, ["x", "y"]].to_numpy(dtype=float)
    offsets = rng.normal(scale=jitter, size=xy.shape)
    crowded = (anchor.map(anchor.value_counts()) > 1).to_numpy()
    offsets[~crowded] = 0.0
    xy = xy + offsets
    is_conventional = gens.carrier.isin(conventional).to_numpy()
    sizes = 6.0 + 34.0 * np.sqrt(np.clip(gens.p_nom.fillna(0.0).to_numpy(dtype=float), 0, cap) / cap)
    for mask, key in ((is_conventional, "conventional"), (~is_conventional, "renewable")):
        if mask.any():
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=sizes[mask],
                c=GEN_COLORS[key],
                alpha=0.75,
                linewidths=0.3,
                edgecolors="white",
                zorder=5,
            )


def _generator_legend(fig, title):
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=GEN_COLORS["conventional"], label="conventional generator"),
        plt.Line2D(
            [], [], marker="o", linestyle="", color=GEN_COLORS["renewable"], label="renewable / other generator"
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9, title=title, title_fontsize=8)


def plot_cluster_suffixes(simpl_path, suffix_paths, shapes_path, out_path, conventional_carriers=None):
    """Four-panel map: one ``{simpl}`` network clustered with each ``{clusters}`` suffix.

    ``suffix_paths`` maps a ``{clusters}`` wildcard value (``"4"``, ``"4m"``,
    ``"4c"``, ``"4a"``) to the clustered network built with it. Every panel
    draws the same clustered buses and branches; what differs is the
    generators. Every generator of the clustered network is attached to a
    cluster bus, but a carrier that was *not* aggregated keeps one generator per
    ``{simpl}`` zone, recorded in ``land_region``. Those are drawn at the
    coordinates of their ``{simpl}`` bus, so they spread over the resource zones
    they still resolve, while an aggregated carrier collapses to one marker per
    cluster bus (jittered slightly where several carriers share a bus).
    """
    conventional = set(conventional_carriers or FALLBACK_CONVENTIONAL_CARRIERS)

    shapes = None
    if shapes_path:
        import geopandas as gpd

        shapes = gpd.read_file(shapes_path)

    panels = {}
    for wildcard, path in suffix_paths.items():
        panels[_cluster_suffix(wildcard)] = (str(wildcard), pypsa.Network(path))
    ordered = [(suffix, *panels[suffix]) for suffix in SUFFIX_DESCRIPTIONS if suffix in panels]

    # One capacity scale for the whole figure, so the same p_nom is the same
    # marker size in every panel. Capped at the 95th percentile: a handful of
    # very large plants would otherwise set the scale for all the rest.
    p_nom = pd.concat([n.generators.p_nom for _, _, n in ordered]).fillna(0.0)
    cap = max(float(p_nom.quantile(0.95)), 1.0)

    # Jitter radius from the map extent, so the clouds read the same at any zoom.
    buses = ordered[0][2].buses
    span = max(float(buses.x.max() - buses.x.min()), float(buses.y.max() - buses.y.min()), 1e-6)
    jitter = 0.015 * span

    simpl_network = pypsa.Network(simpl_path)
    simpl_buses = simpl_network.buses
    n_simpl_buses = len(simpl_buses)
    extent = _footprint_extent(simpl_network, *[n for _, _, n in ordered])

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(ordered), figsize=(4.3 * len(ordered), 5.6))
    for ax, (suffix, wildcard, n) in zip(np.atleast_1d(axes), ordered):
        _plot_network_panel(ax, n, shapes, None, extent=extent)
        gens = n.generators
        _draw_generators(ax, n, simpl_buses, conventional, cap, jitter, rng)
        ax.set_title(
            f"{{clusters}} = {wildcard}\n{SUFFIX_DESCRIPTIONS[suffix]}\n({len(gens)} generators)",
            fontsize=9,
        )

    fig.suptitle(
        f"The same {n_simpl_buses}-zone {{simpl}} network clustered to "
        f"{len(buses)} zones by each {{clusters}} suffix",
        fontsize=11,
    )
    _generator_legend(
        fig,
        "marker area ∝ p_nom (capped); non-aggregated generators drawn at their {simpl} zone, aggregated ones at the cluster bus",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("wrote %s", out_path)


SIMPL_DESCRIPTIONS = {
    "": "identity: every substation bus kept",
    "county": "county fast-path: one bus per county",
}


def plot_simpl_resolutions(panels, shapes_path, out_path, conventional_carriers=None):
    """One panel per ``{simpl}`` value, all clustered with the same ``<N>a`` wildcard.

    ``panels`` maps a ``{simpl}`` wildcard value (``"75"``, ``"county"``, ``""``)
    to ``(simpl_network_path, clustered_network_path)``. With ``a`` nothing is
    aggregated, so every generator is drawn at its ``{simpl}`` zone and the
    panels show the resource resolution each ``{simpl}`` mode gives. The
    cluster count may differ per panel: ``county`` requires
    ``topological_boundaries: county`` and therefore one cluster per county.
    """
    conventional = set(conventional_carriers or FALLBACK_CONVENTIONAL_CARRIERS)
    shapes = None
    if shapes_path:
        import geopandas as gpd

        shapes = gpd.read_file(shapes_path)

    loaded = {str(k): (pypsa.Network(sp), pypsa.Network(cp)) for k, (sp, cp) in panels.items()}
    order = sorted(loaded, key=lambda k: (not k.isdigit(), k != "county"))  # numeric, county, identity
    p_nom = pd.concat([c.generators.p_nom for _, c in loaded.values()]).fillna(0.0)
    cap = max(float(p_nom.quantile(0.95)), 1.0)
    extent = _footprint_extent(*[n for pair in loaded.values() for n in pair])
    span = max(extent[1] - extent[0], extent[3] - extent[2], 1e-6)
    jitter = 0.015 * span
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, len(order), figsize=(4.3 * len(order), 5.6))
    for ax, key in zip(np.atleast_1d(axes), order):
        simpl_network, clustered = loaded[key]
        _plot_network_panel(ax, clustered, shapes, None, extent=extent)
        _draw_generators(ax, clustered, simpl_network.buses, conventional, cap, jitter, rng)
        what = SIMPL_DESCRIPTIONS.get(key, f"k-means to {key} zones")
        ax.set_title(
            f"{{simpl}} = {key!r}, {{clusters}} = {_clusters_wildcard(clustered)}\n{what}\n"
            f"({len(simpl_network.buses)} {{simpl}} buses, {len(clustered.generators)} generators)",
            fontsize=9,
        )
    fig.suptitle(
        "The same footprint at each {simpl} resolution, clustered with the a suffix (nothing aggregated)",
        fontsize=11,
    )
    _generator_legend(fig, "marker area ∝ p_nom (capped); every generator drawn at its {simpl} zone (nothing aggregated)")
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def _clusters_wildcard(n):
    """Best-effort ``{clusters}`` label for a clustered network: its bus count plus the ``a`` suffix."""
    return f"{len(n.buses)}a"


def _carrier_colors(n, carriers):
    colors = n.carriers.color.reindex(carriers)
    fallback = pd.Series(
        [plt.get_cmap("tab20")(i % 20) for i in range(len(carriers))],
        index=carriers,
    )
    return colors.where(colors.notna() & (colors != ""), fallback)


def plot_example_outputs(solved_path, out_path):
    """Two-panel figure from a solved network: capacity by carrier + a dispatch week."""
    n = pypsa.Network(solved_path)

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(13, 4.5),
        gridspec_kw={"width_ratios": [1, 2]},
    )

    cap = (
        n.generators.groupby("carrier")
        .p_nom_opt.sum()
        .add(n.storage_units.groupby("carrier").p_nom_opt.sum(), fill_value=0)
        .div(1e3)
        .sort_values()
    )
    cap = cap[cap > 1e-3]
    colors = _carrier_colors(n, cap.index)
    cap.plot.barh(ax=ax1, color=list(colors))
    ax1.set_xlabel("Optimal capacity [GW]")
    ax1.set_ylabel("")
    ax1.set_title("Capacity by carrier", fontsize=10)

    p = n.generators_t.p.T.groupby(n.generators.carrier).sum().T.div(1e3)
    # plain timestamp index (snapshots may be (period, timestep) MultiIndex)
    timestamps = pd.DatetimeIndex(n.snapshots.get_level_values(-1))
    steps_per_day = max(1, round(pd.Timedelta("1D") / (timestamps[1] - timestamps[0])))
    start = len(timestamps) // 2
    week = slice(start, start + 7 * steps_per_day)
    p_week = p.iloc[week].set_axis(timestamps[week])
    p_week = p_week.loc[:, p_week.abs().max() > 1e-3]
    colors = _carrier_colors(n, p_week.columns)
    ax2.stackplot(
        p_week.index,
        p_week.clip(lower=0).T.values,
        labels=p_week.columns,
        colors=list(colors),
        linewidth=0,
    )
    load = n.loads_t.p_set.sum(axis=1).div(1e3).iloc[week]
    ax2.plot(
        p_week.index,
        load.values,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="load",
    )
    ax2.set_xlim(p_week.index[0], p_week.index[-1])
    ax2.set_ylabel("Generation [GW]")
    ax2.set_xlabel("")
    ax2.set_title("Dispatch, example week", fontsize=10)
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, ncol=3, fontsize=7, loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("docs_figures")
    logging.basicConfig(level=logging.INFO)

    if hasattr(snakemake.output, "simpl_resolutions"):
        panels = {
            str(snakemake.params.numeric_simpl): (snakemake.input.numeric_simpl, snakemake.input.numeric_clustered),
            "": (snakemake.input.identity_simpl, snakemake.input.identity_clustered),
        }
        county = list(snakemake.input.county)
        if len(county) == 2:
            panels["county"] = (county[0], county[1])
        plot_simpl_resolutions(
            panels,
            snakemake.input.onshore_shapes,
            snakemake.output.simpl_resolutions,
            conventional_carriers=snakemake.params.conventional_carriers,
        )
        raise SystemExit(0)

    plot_network_aggregation(
        snakemake.input.base_network,
        snakemake.input.simpl_network,
        snakemake.input.clustered_network,
        snakemake.input.onshore_shapes,
        snakemake.output.network_aggregation,
    )
    # Key each clustered network by its own {clusters} wildcard, read off the
    # filename, so the panel titles cannot drift from the file that was built.
    suffix_paths = {}
    for key in ("clustered_plain", "clustered_m", "clustered_c", "clustered_a"):
        path = getattr(snakemake.input, key)
        suffix_paths[Path(path).stem.rsplit("_c", 1)[-1]] = path
    plot_cluster_suffixes(
        snakemake.input.simpl_network,
        suffix_paths,
        snakemake.input.onshore_shapes,
        snakemake.output.cluster_suffixes,
        conventional_carriers=snakemake.params.conventional_carriers,
    )
    plot_example_outputs(
        snakemake.input.solved_network,
        snakemake.output.example_outputs,
    )
