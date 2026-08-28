"""
Plots static and interactive charts to analyze system results.

**Inputs**

A solved network

**Outputs**

Capacity maps for:
    - Base capacity
    - New capacity
    - Optimal capacity (does not show existing unused capacity)
    - Optimal browfield capacity
    - Renewable potential capacity

    .. image:: _static/plots/capacity-map.png
        :scale: 33 %

Emission charts for:
    - Emissions map by node

    .. image:: _static/plots/emissions-map.png
        :scale: 33 %
"""

import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns
from _helpers import configure_logging
from add_electricity import sanitize_carriers
from cartopy import crs as ccrs
from matplotlib.colors import LinearSegmentedColormap, Normalize
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches
from summary import (
    get_node_emissions_timeseries,
)

logger = logging.getLogger(__name__)

# Global Plotting Settings
TITLE_SIZE = 16


def _get_investment_periods(n: pypsa.Network) -> list:
    """Return the list of investment periods, or a single dummy [None] when not multi-period."""
    if hasattr(n, "investment_periods") and len(n.investment_periods) > 0:
        return list(n.investment_periods)
    return [None]


def _active_mask(df: pd.DataFrame, horizon) -> pd.Series:
    """Boolean mask: asset active in `horizon` (build_year <= horizon < build_year + lifetime).

    When `horizon` is None (single-period case), all rows are considered active.
    """
    if horizon is None or "build_year" not in df.columns:
        return pd.Series(True, index=df.index)
    lifetime = df["lifetime"] if "lifetime" in df.columns else pd.Series(np.inf, index=df.index)
    return (df["build_year"] <= horizon) & (horizon < df["build_year"] + lifetime)


def _capacity_by_bus_carrier(
    n: pypsa.Network,
    horizon,
    attr: str,
) -> pd.Series:
    """Sum `attr` (e.g. p_nom, p_nom_opt) of active assets at `horizon` by (bus, carrier).

    Mirrors `summary.get_capacity_brownfield` but filters by activity in `horizon`.
    """
    parts = []
    for c in (n.components[name] for name in ["Generator", "StorageUnit", "Link"]):
        mask = _active_mask(c.static, horizon)
        if not mask.any():
            continue
        df = c.static.loc[mask]
        if c.name == "Link":
            parts.append(df[attr].groupby([df.bus0, df.carrier]).sum().rename_axis(index={"bus0": "bus"}))
            parts.append(df[attr].groupby([df.bus1, df.carrier]).sum().rename_axis(index={"bus1": "bus"}))
        else:
            parts.append(df[attr].groupby([df.bus, df.carrier]).sum())
    if not parts:
        return pd.Series(dtype=float, index=pd.MultiIndex.from_tuples([], names=["bus", "carrier"]))
    return pd.concat(parts)


def _line_link_capacity(
    n: pypsa.Network,
    horizon,
    attr_line: str,
    attr_link: str,
) -> tuple[pd.Series, pd.Series]:
    """Return (line_values, link_values) for active lines/AC-links at `horizon`."""
    line_mask = _active_mask(n.lines, horizon)
    line_values = n.lines.loc[line_mask, attr_line] if line_mask.any() else pd.Series(0, index=n.lines.index)

    ac_links = n.links[n.links.carrier == "AC"]
    link_mask = _active_mask(ac_links, horizon)
    link_values = (
        ac_links.loc[link_mask, attr_link].replace(to_replace={pd.NA: 0})
        if link_mask.any()
        else pd.Series(
            0,
            index=ac_links.index,
        )
    )
    return line_values, link_values


def get_color_palette(n: pypsa.Network) -> pd.Series:
    """Returns colors based on nice name."""
    colors = (n.carriers.reset_index().set_index("nice_name")).color

    # additional = {
    #     "Battery Charge": n.carriers.loc["battery"].color,
    #     "Battery Discharge": n.carriers.loc["battery"].color,
    #     "battery_discharger": n.carriers.loc["battery"].color,
    #     "battery_charger": n.carriers.loc["battery"].color,
    #     "4hr_battery_storage_discharger": n.carriers.loc["4hr_battery_storage"].color,
    #     "4hr_battery_storage_charger": n.carriers.loc["4hr_battery_storage"].color,
    #     "8hr_PHS_charger": n.carriers.loc["8hr_PHS"].color,
    #     "8hr_PHS_discharger": n.carriers.loc["8hr_PHS"].color,
    #     "10hr_PHS_charger": n.carriers.loc["10hr_PHS"].color,
    #     "10hr_PHS_discharger": n.carriers.loc["10hr_PHS"].color,
    #     "co2": "k",
    # }

    # Initialize the additional dictionary
    additional = {
        "co2": "k",
    }

    # Loop through the carriers DataFrame
    for index, row in n.carriers.iterrows():
        if "battery" in index or "PHS" in index:
            color = row.color
            additional.update(
                {
                    f"{index}_charger": color,
                    f"{index}_discharger": color,
                },
            )

    return pd.concat([colors, pd.Series(additional)]).to_dict()


def get_bus_scale(interconnect: str) -> float:
    """Scales lines based on interconnect size."""
    if interconnect != "usa":
        return 1e5
    else:
        return 4e4


def get_line_scale(interconnect: str) -> float:
    """Scales lines based on interconnect size."""
    if interconnect != "usa":
        return 2e3
    else:
        return 3e3


def create_title(title: str, **wildcards) -> str:
    """
    Standardizes wildcard writing in titles.

    Arguments:
        title: str
            Title of chart to plot
        **wildcards
            any wildcards to add to title
    """
    w = []
    for wildcard, value in wildcards.items():
        if wildcard == "interconnect":
            w.append(f"interconnect = {value}")
        elif wildcard == "clusters":
            w.append(f"#clusters = {value}")
        elif wildcard == "ll":
            w.append(f"ll = {value}")
        elif wildcard == "opts":
            w.append(f"opts = {value}")
        elif wildcard == "sector":
            w.append(f"sectors = {value}")
    wildcards_joined = " | ".join(w)
    return f"{title} \n ({wildcards_joined})"


def remove_sector_buses(df: pd.DataFrame) -> pd.DataFrame:
    """Removes buses for sector coupling."""
    num_levels = df.index.nlevels

    if num_levels > 1:
        condition = (df.index.get_level_values("bus").str.endswith(" gas")) | (
            df.index.get_level_values("bus").str.endswith(" gas storage")
        )
    else:
        condition = (
            (df.index.str.endswith(" gas"))
            | (df.index.str.endswith(" gas storage"))
            | (df.index.str.endswith(" gas import"))
            | (df.index.str.endswith(" gas export"))
        )
    return df.loc[~condition].copy()


def plot_emissions_map(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    save: str,
    **wildcards,
) -> None:
    # get data

    emissions = (
        get_node_emissions_timeseries(n)
        # pandas 3 removed groupby(axis=1); transpose-group-transpose groups
        # the columns while preserving the (snapshots x bus) orientation.
        .T.groupby(level=0)  # group columns
        .sum()
        .T.sum()  # collaps rows
        .mul(1e-6)  # T -> MT
    )
    emissions = remove_sector_buses(emissions.T).T
    emissions.index.name = "bus"

    # plot data

    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw={"projection": ccrs.EqualEarth(n.buses.x.mean())},
    )

    # Use fixed scale - same as get_bus_scale() for capacity maps
    bus_scale = 1e3  # Emissions in MT - divide to get reasonable circle sizes
    legend_sizes_mt = [1, 5, 10]  # Legend in MT

    # First draw regions as background
    regions.plot(
        ax=ax,
        facecolor="whitesmoke",
        edgecolor="white",
        aspect="equal",
        transform=ccrs.PlateCarree(),
        linewidth=1.2,
    )

    with plt.rc_context({"patch.linewidth": 0.1}):
        n.plot(
            bus_sizes=emissions / bus_scale,
            bus_colors="k",
            bus_alpha=0.6,
            line_widths=0,
            link_widths=0,
            ax=ax,
            margin=0.2,
            color_geomap=None,
        )

    ax.set_extent(regions.total_bounds[[0, 2, 1, 3]])

    # Add legend for emission circle sizes
    legend_kwargs = {"loc": "upper left", "frameon": False}
    add_legend_circles(
        ax,
        [s / bus_scale for s in legend_sizes_mt],
        [f"{s:.0f} MT" for s in legend_sizes_mt],
        legend_kw={"bbox_to_anchor": (1, 1), "labelspacing": 3, **legend_kwargs},
        patch_kw={"facecolor": "k", "edgecolor": "black", "alpha": 0.7},
    )

    title = create_title("Emissions (MTonne)", **wildcards)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=20)
    fig.savefig(save, bbox_inches="tight")
    plt.close()


def _plot_capacity_on_ax(
    n: pypsa.Network,
    bus_values: pd.Series,
    line_values: pd.Series,
    link_values: pd.Series,
    regions: gpd.GeoDataFrame,
    ax,
    bus_scale=1,
    line_scale=1,
    line_colors="teal",
    link_colors="green",
    line_cmap="viridis",
    line_norm=None,
    flow=None,
) -> dict:
    """Draw the capacity pie-chart map onto an existing axis.

    Returns a dict with bus_colors and nice_names so the caller can attach a
    shared legend at the figure level when used inside a subplot grid.
    """
    carrier_exclusion = ["imports", "exports", "demand_response"]
    bus_colors = n.carriers.color[~n.carriers.color.index.isin(carrier_exclusion)].fillna("#000000")
    nice_names = n.carriers.nice_name[~n.carriers.nice_name.index.isin(carrier_exclusion)].fillna("Other")

    # Drop any (bus, carrier) rows whose carrier isn't in bus_colors — n.plot()
    # raises ValueError("Colors not defined for all elements...") otherwise.
    if not bus_values.empty and "carrier" in bus_values.index.names:
        bus_values = bus_values[bus_values.index.get_level_values("carrier").isin(bus_colors.index)]

    line_width = line_values / line_scale
    link_width = link_values / line_scale

    # Draw region tiles first as the geographic backdrop.
    regions.plot(
        ax=ax,
        facecolor="whitesmoke",
        edgecolor="white",
        aspect="equal",
        transform=ccrs.PlateCarree(),
        linewidth=1.2,
    )

    # Empty bus_values would make n.plot()'s bbox compute crash on an empty
    # array. Skip the network draw and leave the region backdrop in place.
    if not bus_values.empty:
        with plt.rc_context({"patch.linewidth": 0.1}):
            n.plot(
                bus_sizes=bus_values / bus_scale,
                bus_colors=bus_colors,
                bus_alpha=0.7,
                line_widths=line_width,
                link_widths=0 if link_width.empty else link_width,
                line_colors=line_colors,
                link_colors=link_colors,
                ax=ax,
                margin=0.2,
                color_geomap=True,
                flow=flow,
                line_cmap=line_cmap,
                line_norm=line_norm,
            )

    ax.set_extent(regions.total_bounds[[0, 2, 1, 3]])

    return {"bus_colors": bus_colors, "nice_names": nice_names}


def _add_capacity_legends(
    fig,
    ax,
    bus_colors,
    nice_names,
    bus_scale,
    line_scale,
    *,
    bbox_anchor=(1, 1),
) -> None:
    """Attach the standard capacity-map legends (sizes + carrier patches)."""
    legend_kwargs = {"loc": "upper left", "frameon": False}
    bus_sizes = [5000, 10e3, 50e3]  # MW
    line_sizes = [2000, 5000]  # MW

    add_legend_circles(
        ax,
        [s / bus_scale for s in bus_sizes],
        [f"{s / 1000:.0f} GW" for s in bus_sizes],
        legend_kw={"bbox_to_anchor": bbox_anchor, "labelspacing": 3, **legend_kwargs},
    )
    add_legend_lines(
        ax,
        [s / line_scale for s in line_sizes],
        [f"{s / 1000:.0f} GW" for s in line_sizes],
        legend_kw={"bbox_to_anchor": (bbox_anchor[0], bbox_anchor[1] - 0.2), **legend_kwargs},
    )
    add_legend_patches(
        ax,
        bus_colors,
        nice_names,
        legend_kw={"bbox_to_anchor": (bbox_anchor[0], 0), **legend_kwargs, "loc": "lower left"},
    )


def plot_capacity_map(
    n: pypsa.Network,
    bus_values: pd.DataFrame,
    line_values: pd.DataFrame,
    link_values: pd.DataFrame,
    regions: gpd.GeoDataFrame,
    bus_scale=1,
    line_scale=1,
    title=None,
    flow=None,
    line_colors="teal",
    link_colors="green",
    line_cmap="viridis",
    line_norm=None,
) -> tuple[plt.figure, plt.axes]:
    """Generic single-axis capacity pie-chart map."""
    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw={"projection": ccrs.EqualEarth(n.buses.x.mean())},
    )

    artifacts = _plot_capacity_on_ax(
        n,
        bus_values,
        line_values,
        link_values,
        regions,
        ax,
        bus_scale=bus_scale,
        line_scale=line_scale,
        line_colors=line_colors,
        link_colors=link_colors,
        line_cmap=line_cmap,
        line_norm=line_norm,
        flow=flow,
    )
    _add_capacity_legends(fig, ax, artifacts["bus_colors"], artifacts["nice_names"], bus_scale, line_scale)

    ax.set_title(title or "Capacity (MW)", fontsize=TITLE_SIZE, pad=20)
    fig.tight_layout()
    return fig, ax


def plot_capacity_map_by_horizon(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    carriers: list[str],
    save: str,
    kind: str,
    **wildcards,
) -> None:
    """Plot a capacity-pie-chart map per investment period.

    `kind` selects the data:
        - "base"     : pre-solve installed capacity active in each horizon
        - "optimal"  : optimized capacity active in each horizon
        - "new"      : capacity vintaged to that specific horizon (built then)
    """
    horizons = _get_investment_periods(n)
    if len(horizons) == 1:
        # Fall back to the single-axis layout when only one horizon exists.
        bus_attr_for_kind = {"base": "p_nom", "optimal": "p_nom_opt", "new": "p_nom_opt"}
        attr = bus_attr_for_kind[kind]
        bus_values = _capacity_by_bus_carrier(n, horizons[0], attr)
        if kind == "new":
            # `new` is p_nom_opt - p_nom for assets vintaged in this horizon.
            base_bv = _capacity_by_bus_carrier(n, horizons[0], "p_nom")
            bus_values = (bus_values - base_bv.reindex(bus_values.index, fill_value=0)).clip(lower=0)
        bus_values = bus_values[bus_values.index.get_level_values("carrier").isin(carriers)]
        bus_values = remove_sector_buses(bus_values).groupby(["bus", "carrier"]).sum()

        line_attr = "s_nom_opt" if kind != "base" else "s_nom"
        link_attr = "p_nom_opt" if kind != "base" else "p_nom"
        line_values, link_values = _line_link_capacity(n, horizons[0], line_attr, link_attr)
        if kind == "new":
            line_base, link_base = _line_link_capacity(n, horizons[0], "s_nom", "p_nom")
            line_values = (line_values - line_base).clip(lower=0)
            link_values = (link_values - link_base).clip(lower=0)

        kind_titles = {"base": "Base", "optimal": "Optimal", "new": "New"}
        title = create_title(f"{kind_titles[kind]} Network Capacities", **wildcards)
        interconnect = wildcards.get("interconnect")
        bus_scale = get_bus_scale(interconnect) if interconnect else 1
        line_scale = get_line_scale(interconnect) if interconnect else 1

        if bus_values.empty and kind == "new":
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.text(0.5, 0.5, "No new capacity built", ha="center", va="center", fontsize=14)
            ax.set_title(title, fontsize=TITLE_SIZE, pad=20)
            ax.axis("off")
        else:
            fig, _ = plot_capacity_map(
                n=n,
                bus_values=bus_values,
                line_values=line_values,
                link_values=link_values,
                regions=regions,
                bus_scale=bus_scale,
                line_scale=line_scale,
                title=title,
            )
        fig.savefig(save)
        plt.close()
        return

    # Multi-horizon: one column per horizon, shared legend on the right.
    n_h = len(horizons)
    fig, axes = plt.subplots(
        1,
        n_h,
        figsize=(8 * n_h, 8.5),
        subplot_kw={"projection": ccrs.EqualEarth(n.buses.x.mean())},
    )
    if n_h == 1:
        axes = [axes]

    interconnect = wildcards.get("interconnect")
    bus_scale = get_bus_scale(interconnect) if interconnect else 1
    line_scale = get_line_scale(interconnect) if interconnect else 1

    artifacts = None
    for ax, horizon in zip(axes, horizons):
        if kind == "base":
            bus_values = _capacity_by_bus_carrier(n, horizon, "p_nom")
            line_values, link_values = _line_link_capacity(n, horizon, "s_nom", "p_nom")
        elif kind == "optimal":
            bus_values = _capacity_by_bus_carrier(n, horizon, "p_nom_opt")
            line_values, link_values = _line_link_capacity(n, horizon, "s_nom_opt", "p_nom_opt")
        elif kind == "new":
            # Restrict to assets vintaged to this horizon: built_year == horizon.
            parts = []
            for c in (n.components[name] for name in ["Generator", "StorageUnit", "Link"]):
                df = (
                    c.static[c.static.get("build_year") == horizon]
                    if "build_year" in c.static.columns
                    else c.static.iloc[0:0]
                )
                if df.empty:
                    continue
                if c.name == "Link":
                    parts.append(
                        df["p_nom_opt"].groupby([df.bus0, df.carrier]).sum().rename_axis(index={"bus0": "bus"}),
                    )
                    parts.append(
                        df["p_nom_opt"].groupby([df.bus1, df.carrier]).sum().rename_axis(index={"bus1": "bus"}),
                    )
                else:
                    parts.append(df["p_nom_opt"].groupby([df.bus, df.carrier]).sum())
            bus_values = (
                pd.concat(parts)
                if parts
                else pd.Series(
                    dtype=float,
                    index=pd.MultiIndex.from_tuples([], names=["bus", "carrier"]),
                )
            )
            # New transmission for this horizon: lines/links vintaged here.
            line_mask = n.lines.get("build_year", pd.Series(np.nan, index=n.lines.index)) == horizon
            line_values = n.lines.loc[line_mask, "s_nom_opt"] if line_mask.any() else pd.Series(0, index=n.lines.index)
            ac_links = n.links[n.links.carrier == "AC"]
            link_mask = ac_links.get("build_year", pd.Series(np.nan, index=ac_links.index)) == horizon
            link_values = (
                ac_links.loc[link_mask, "p_nom_opt"].replace(to_replace={pd.NA: 0})
                if link_mask.any()
                else pd.Series(0, index=ac_links.index)
            )
        else:
            raise ValueError(f"Unknown kind: {kind}")

        bus_values = bus_values[bus_values.index.get_level_values("carrier").isin(carriers)]
        bus_values = (
            remove_sector_buses(bus_values).groupby(["bus", "carrier"]).sum() if not bus_values.empty else bus_values
        )

        artifacts = _plot_capacity_on_ax(
            n,
            bus_values,
            line_values,
            link_values,
            regions,
            ax,
            bus_scale=bus_scale,
            line_scale=line_scale,
        )
        ax.set_title(f"{horizon}", fontsize=TITLE_SIZE)

    # Attach a single shared legend on the rightmost axis.
    if artifacts is not None:
        _add_capacity_legends(
            fig,
            axes[-1],
            artifacts["bus_colors"],
            artifacts["nice_names"],
            bus_scale,
            line_scale,
        )

    kind_titles = {"base": "Base", "optimal": "Optimal", "new": "New"}
    fig.suptitle(create_title(f"{kind_titles[kind]} Network Capacities by Horizon", **wildcards), fontsize=TITLE_SIZE)
    fig.tight_layout(rect=[0, 0, 0.9, 0.97])
    fig.savefig(save, bbox_inches="tight")
    plt.close()


def _bus_value_per_horizon(
    n: pypsa.Network,
    horizon,
    series_t: pd.DataFrame,
    aggregator: str = "mean",
) -> pd.Series:
    """Average (or sum) a time-indexed bus quantity over snapshots in `horizon`.

    `series_t` columns are bus names; index is `n.snapshots` (MultiIndex of
    (period, timestep) in the multi-investment case). When `horizon` is None,
    aggregate over the entire snapshots index.
    """
    if horizon is None or not isinstance(series_t.index, pd.MultiIndex):
        sub = series_t
    else:
        sub = series_t.loc[series_t.index.get_level_values(0) == horizon]
    if aggregator == "sum":
        return sub.sum(axis=0)
    return sub.mean(axis=0)


def _plot_choropleth_on_ax(
    n: pypsa.Network,
    values: pd.Series,
    regions: gpd.GeoDataFrame,
    ax,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    show_lines: bool = True,
) -> None:
    """Draw a bus-value choropleth onto `ax`, joining `values` (indexed by bus) to `regions`."""
    region_col = regions.set_index("name")
    region_col = region_col.join(values.rename("value"))
    region_col["value"] = region_col["value"].fillna(np.nan)

    # Background for buses with no data: light gray.
    regions.plot(
        ax=ax,
        facecolor="whitesmoke",
        edgecolor="white",
        aspect="equal",
        transform=ccrs.PlateCarree(),
        linewidth=0.5,
    )

    region_col.plot(
        column="value",
        ax=ax,
        cmap=cmap,
        edgecolor="white",
        linewidth=0.3,
        aspect="equal",
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
        missing_kwds={"color": "whitesmoke"},
    )

    if show_lines and not n.lines.empty:
        with plt.rc_context({"patch.linewidth": 0.05}):
            n.plot(
                bus_sizes=0,
                line_widths=0.4,
                link_widths=0,
                line_colors="black",
                ax=ax,
                margin=0.2,
                color_geomap=None,
            )

    ax.set_extent(regions.total_bounds[[0, 2, 1, 3]])


def plot_demand_map(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    carriers: list[str],
    save: str,
    **wildcards,
) -> None:
    """Plot mean nodal demand as a region-tile choropleth, one subplot per horizon."""
    horizons = _get_investment_periods(n)

    # AC-load (MW) at each bus per snapshot, summed across all loads at that bus.
    load_per_bus = n.loads_t.p_set.rename(columns=n.loads.bus).T.groupby(level=0).sum().T

    # Compute a shared color scale across horizons so subplots are comparable.
    horizon_values = {h: _bus_value_per_horizon(n, h, load_per_bus, aggregator="mean") for h in horizons}
    finite_vals = pd.concat(horizon_values.values()).replace([np.inf, -np.inf], np.nan).dropna()
    vmin = float(finite_vals.min()) if not finite_vals.empty else 0.0
    vmax = float(finite_vals.max()) if not finite_vals.empty else 1.0

    n_h = len(horizons)
    fig, axes = plt.subplots(
        1,
        n_h,
        figsize=(8 * n_h, 8.5),
        subplot_kw={"projection": ccrs.EqualEarth(n.buses.x.mean())},
    )
    if n_h == 1:
        axes = [axes]

    # Use a truncated Blues palette so the low end starts at a slightly darker
    # blue (skipping the near-white portion of matplotlib's built-in Blues).
    base_blues = plt.colormaps["Blues"]
    blues_truncated = LinearSegmentedColormap.from_list(
        "BluesTruncated",
        base_blues(np.linspace(0.25, 1.0, 256)),
    )

    for ax, horizon in zip(axes, horizons):
        values = horizon_values[horizon]
        _plot_choropleth_on_ax(n, values, regions, ax, cmap=blues_truncated, vmin=vmin, vmax=vmax)
        h_label = "" if horizon is None else f" — {horizon}"
        ax.set_title(f"Mean Nodal Demand (MW){h_label}", fontsize=TITLE_SIZE - 2)

    # Shared colorbar on the right edge.
    sm = plt.cm.ScalarMappable(cmap=blues_truncated, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label("Mean Demand (MW)")

    fig.suptitle(create_title("Mean Nodal Demand by Horizon", **wildcards), fontsize=TITLE_SIZE)
    fig.savefig(save, bbox_inches="tight")
    plt.close()


def plot_lmp_choropleth_map(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    save: str,
    **wildcards,
) -> None:
    """Plot mean nodal LMP as a region-tile choropleth, one subplot per horizon."""
    if not hasattr(n, "buses_t") or "marginal_price" not in n.buses_t or n.buses_t.marginal_price.empty:
        logger.warning("No marginal_price data available; skipping LMP map.")
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.text(0.5, 0.5, "No LMP data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        fig.savefig(save)
        plt.close()
        return

    horizons = _get_investment_periods(n)
    lmps_t = n.buses_t.marginal_price

    horizon_values = {h: _bus_value_per_horizon(n, h, lmps_t, aggregator="mean") for h in horizons}
    finite_vals = pd.concat(horizon_values.values()).replace([np.inf, -np.inf], np.nan).dropna()
    if finite_vals.empty:
        vmin, vmax = 0.0, 1.0
    else:
        # Clip extreme outliers for color stability (5th/95th pct).
        vmin = float(np.nanpercentile(finite_vals.values, 5))
        vmax = float(np.nanpercentile(finite_vals.values, 95))

    n_h = len(horizons)
    fig, axes = plt.subplots(
        1,
        n_h,
        figsize=(8 * n_h, 8.5),
        subplot_kw={"projection": ccrs.EqualEarth(n.buses.x.mean())},
    )
    if n_h == 1:
        axes = [axes]

    for ax, horizon in zip(axes, horizons):
        values = horizon_values[horizon]
        _plot_choropleth_on_ax(n, values, regions, ax, cmap="plasma", vmin=vmin, vmax=vmax)
        h_label = "" if horizon is None else f" — {horizon}"
        ax.set_title(f"Mean LMP ($/MWh){h_label}", fontsize=TITLE_SIZE - 2)

    sm = plt.cm.ScalarMappable(cmap="plasma", norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label("Mean LMP ($/MWh)")

    fig.suptitle(create_title("Mean Locational Marginal Price by Horizon", **wildcards), fontsize=TITLE_SIZE)
    fig.savefig(save, bbox_inches="tight")
    plt.close()


def plot_base_capacity_map(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    carriers: list[str],
    save: str,
    **wildcards,
) -> None:
    """Plot installed (base) capacity active in each planning horizon."""
    plot_capacity_map_by_horizon(n, regions, carriers, save, kind="base", **wildcards)


def plot_opt_capacity_map(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    carriers: list[str],
    save: str,
    **wildcards,
) -> None:
    """Plot optimal (brownfield) capacity active in each planning horizon."""
    plot_capacity_map_by_horizon(n, regions, carriers, save, kind="optimal", **wildcards)


def plot_new_capacity_map(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    carriers: list[str],
    save: str,
    **wildcards,
) -> None:
    """Plot capacity newly built in each planning horizon (vintaged by build_year)."""
    plot_capacity_map_by_horizon(n, regions, carriers, save, kind="new", **wildcards)


def plot_renewable_potential(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    save: str,
    **wildcards,
) -> None:
    """Plots wind and solar resource potential by node."""
    # get data
    renew = n.generators[
        (n.generators.p_nom_max != np.inf)
        & (n.generators.build_year == n.investment_periods[0])
        & (
            n.generators.carrier.isin(
                ["onwind", "offwind", "offwind_floating", "solar", "EGS"],
            )
        )
    ]

    bus_values = renew.groupby(["bus", "carrier"]).p_nom_max.sum()

    # do not show lines or links
    line_values = pd.Series(0, index=n.lines.s_nom.index)
    link_values = pd.Series(0, index=n.links.p_nom.index)

    # plot data
    title = create_title("Renewable Capacity Potential", **wildcards)
    interconnect = wildcards.get("interconnect", None)
    bus_scale = get_bus_scale(interconnect) if interconnect else 1

    bus_scale *= 15  # since potential capacity is so big

    fig, ax = plot_capacity_map(
        n=n,
        bus_values=bus_values,
        line_values=line_values,
        link_values=link_values,
        regions=regions,
        bus_scale=bus_scale,
        title=title,
    )

    # only show renewables in legend
    fig.artists[-2].remove()  # remove line width legend
    fig.artists[-1].remove()  # remove existing colour legend
    renew_carriers = n.carriers[n.carriers.index.isin(["onwind", "offwind", "offwind_floating", "solar", "EGS"])]
    add_legend_patches(
        ax,
        renew_carriers.color,
        renew_carriers.nice_name,
        legend_kw={"bbox_to_anchor": (1, 0), "frameon": False, "loc": "lower left"},
    )

    fig.savefig(save)
    plt.close()


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_network_maps",
            interconnect="western",
            clusters="4m",
            simpl="70",
            ll="v1.0",
            opts="1h-TCT",
            sector="E",
        )
    configure_logging(snakemake)

    # extract shared plotting files
    n = pypsa.Network(snakemake.input.network)
    onshore_regions = gpd.read_file(snakemake.input.regions_onshore)

    sanitize_carriers(n, snakemake.config)

    # carriers to plot
    carriers = (
        snakemake.params.electricity["conventional_carriers"]
        + snakemake.params.electricity["renewable_carriers"]
        + snakemake.params.electricity["extendable_carriers"]["Generator"]
        + snakemake.params.electricity["extendable_carriers"]["StorageUnit"]
        + snakemake.params.electricity["extendable_carriers"]["Store"]
        + snakemake.params.electricity["extendable_carriers"]["Link"]
    )
    carriers = list(set(carriers))  # remove any duplicates

    # plotting theme
    sns.set_theme("paper", style="darkgrid")

    # create plots
    plot_base_capacity_map(
        n,
        onshore_regions,
        carriers,
        snakemake.output["capacity_map_base.pdf"],
        **snakemake.wildcards,
    )
    plot_opt_capacity_map(
        n,
        onshore_regions,
        carriers,
        **snakemake.wildcards,
        save=snakemake.output["capacity_map_optimized.pdf"],
    )
    plot_new_capacity_map(
        n,
        onshore_regions,
        carriers,
        **snakemake.wildcards,
        save=snakemake.output["capacity_map_new.pdf"],
    )
    plot_demand_map(
        n,
        onshore_regions,
        carriers,
        snakemake.output["demand_map.pdf"],
        **snakemake.wildcards,
    )
    plot_emissions_map(
        n,
        onshore_regions,
        snakemake.output["emissions_map.pdf"],
        **snakemake.wildcards,
    )
    plot_renewable_potential(
        n,
        onshore_regions,
        snakemake.output["renewable_potential_map.pdf"],
        **snakemake.wildcards,
    )
    # `lmp_map.pdf` was added after the rule's outputs are sometimes cached in
    # a stale DAG — fall back to skipping with a warning rather than crashing
    # the whole rule. Force-recreate the DAG to pick it up: `snakemake -F` or
    # remove the cached rule graph.
    try:
        lmp_save = snakemake.output["lmp_map.pdf"]
    except (AttributeError, KeyError):
        logger.warning(
            "Output 'lmp_map.pdf' not registered in this rule invocation; "
            "skipping LMP map. Force-rerun snakemake (e.g. with -F) to refresh outputs.",
        )
    else:
        plot_lmp_choropleth_map(
            n,
            onshore_regions,
            lmp_save,
            **snakemake.wildcards,
        )
