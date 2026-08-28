"""
Render the canonical documentation figures from real workflow artifacts.

Produces the images embedded in the Model Description docs pages
(``docs/source/_static/generated/``) so that documentation figures are
regenerated from the current pipeline rather than hand-maintained:

- ``network_aggregation.png`` — the same system at nodal, ``{simpl}``, and
  ``{clusters}`` resolution, illustrating the two-stage spatial aggregation.
- ``example_outputs.png`` — optimal capacity by carrier and a week of dispatch
  from a solved network.

Run via ``snakemake docs_figures`` (see ``workflow/Snakefile``).
"""

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)

NET_KW = dict(geomap=False, bus_colors="firebrick")


def _plot_network_panel(ax, n, shapes, title):
    if shapes is not None:
        shapes.plot(ax=ax, facecolor="#f0f0f0", edgecolor="white", linewidth=0.6)
    n.plot(
        ax=ax,
        bus_sizes=0.004,
        line_widths=n.lines.s_nom / n.lines.s_nom.max() * 2.0 if len(n.lines) else 0,
        link_widths=0.8,
        **NET_KW,
    )
    n_branches = len(n.lines) + len(n.links)
    ax.set_title(f"{title}\n({len(n.buses)} buses, {n_branches} branches)", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_network_aggregation(base_path, simpl_path, clusters_path, shapes_path, out_path):
    """Three-panel map: nodal base network -> {simpl} zones -> {clusters} zones."""
    shapes = None
    if shapes_path:
        import geopandas as gpd

        shapes = gpd.read_file(shapes_path)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, path, title in zip(
        axes,
        [base_path, simpl_path, clusters_path],
        ["Nodal base network", "After cluster_simpl ({simpl})", "After cluster_network ({clusters})"],
    ):
        _plot_network_panel(ax, pypsa.Network(path), shapes, title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("wrote %s", out_path)


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
    plot_network_aggregation(
        snakemake.input.base_network,
        snakemake.input.simpl_network,
        snakemake.input.clustered_network,
        snakemake.input.onshore_shapes,
        snakemake.output.network_aggregation,
    )
    plot_example_outputs(
        snakemake.input.solved_network,
        snakemake.output.example_outputs,
    )
