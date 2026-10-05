"""
Plot per-bus p_nom_max map and capacity-vs-access-scenario bar chart.

Mirrors the style of the existing solar plots 05_p_nom_max_per_bus.png and
08_capacity_totals_by_access.png under nrel_exclusion_work/, but parameterized
over tech so the same code produces onwind and offwind figures.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm

TECH_LABEL = {
    "solar": "Solar",
    "onwind": "Onshore wind",
    "offwind": "Offshore wind",
    "offwind_floating": "Offshore wind (floating)",
}


def load_caps(caps_dir: Path, tech: str, access: str) -> pd.DataFrame:
    ds = xr.open_dataset(caps_dir / f"caps_{tech}_{access}.nc")
    df = pd.DataFrame(
        {"name": ds["bus"].values.astype(str), "p_nom_max": ds["p_nom_max"].values},
    )
    ds.close()
    return df


def plot_pnom_map(
    caps_df: pd.DataFrame,
    shapes: gpd.GeoDataFrame,
    tech: str,
    access: str,
    out_path: Path,
    xlim: tuple[float, float] = (-130, -65),
    ylim: tuple[float, float] = (22, 52),
) -> None:
    shapes = shapes.copy()
    shapes["name"] = shapes["name"].astype(str)
    merged = shapes.merge(caps_df, on="name", how="left")
    n_buses = int(merged["p_nom_max"].notna().sum())
    total_gw = float(merged["p_nom_max"].sum(skipna=True)) / 1000.0

    fig, ax = plt.subplots(figsize=(10, 6))
    vals = merged["p_nom_max"].replace(0, np.nan)
    vmin = max(1.0, float(np.nanmin(vals))) if vals.notna().any() else 1.0
    vmax = float(np.nanmax(vals)) if vals.notna().any() else 10.0
    merged.plot(
        column="p_nom_max",
        cmap="plasma",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        linewidth=0.1,
        edgecolor="0.7",
        ax=ax,
        legend=True,
        legend_kwds={"label": "p_nom_max per bus (MW, log scale)"},
        missing_kwds={"color": "white", "edgecolor": "0.7", "linewidth": 0.1},
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{TECH_LABEL[tech]} capacity potential per bus "
        f"(NREL {access} supply curve)\n"
        f"Σ = {total_gw:,.1f} GW across {n_buses} western buses",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_totals_by_access(
    caps_dir: Path,
    tech: str,
    out_path: Path,
    accesses: tuple[str, ...] = ("limited", "reference", "open"),
    colors: tuple[str, ...] = ("#C44E52", "#4C72B0", "#2CA02C"),
) -> None:
    totals_gw = {}
    for a in accesses:
        df = load_caps(caps_dir, tech, a)
        totals_gw[a] = float(df["p_nom_max"].sum()) / 1000.0

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(list(totals_gw.keys()), list(totals_gw.values()), color=list(colors))
    for bar, v in zip(bars, totals_gw.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:,.0f} GW",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_ylabel(f"Western US {TECH_LABEL[tech].lower()} p_nom_max (GW)")
    ax.set_title(
        f"{TECH_LABEL[tech]} capacity potential vs NREL access scenario\n"
        f"(Sum across western county buses, from NREL supply curves)",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(totals_gw.values()) * 1.12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tech",
        required=True,
        choices=["solar", "onwind", "offwind", "offwind_floating"],
    )
    ap.add_argument(
        "--caps-dir",
        default="/scratch/groups/iazevedo/asia/nrel_avail",
    )
    ap.add_argument(
        "--onshore-shapes",
        default="/home/groups/iazevedo/asia/pypsa-usa/workflow/resources/godeeep/"
        "geospatial/county_rcp45hotter/western/regions_onshore.geojson",
    )
    ap.add_argument(
        "--offshore-shapes",
        default="/home/groups/iazevedo/asia/pypsa-usa/workflow/resources/godeeep/"
        "geospatial/county_rcp45hotter/western/regions_offshore.geojson",
    )
    ap.add_argument("--access-for-map", default="reference")
    ap.add_argument(
        "--out-dir",
        default="/home/groups/iazevedo/asia/nrel_exclusion_work",
    )
    args = ap.parse_args()

    caps_dir = Path(args.caps_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decide which bus polygons to use for the map. For onwind/solar we use
    # onshore buses; for offwind we use offshore buses, and also draw onshore
    # buses as faint context so the coastline is visible.
    onshore = gpd.read_file(args.onshore_shapes).to_crs(4326)
    if args.tech.startswith("offwind"):
        offshore = gpd.read_file(args.offshore_shapes).to_crs(4326)
        shapes = offshore
        xlim = (-130, -115)
        ylim = (30, 50)
    else:
        shapes = onshore
        xlim = (-130, -100)
        ylim = (24, 52)

    caps_df = load_caps(caps_dir, args.tech, args.access_for_map)

    # Map
    map_path = out_dir / f"p_nom_max_per_bus_{args.tech}.png"
    if args.tech.startswith("offwind"):
        # Draw onshore context then overlay offshore p_nom_max
        shapes_plot = shapes.copy()
        shapes_plot["name"] = shapes_plot["name"].astype(str)
        merged = shapes_plot.merge(caps_df, on="name", how="left")
        n_buses = int(merged["p_nom_max"].notna().sum())
        total_gw = float(merged["p_nom_max"].sum(skipna=True)) / 1000.0

        fig, ax = plt.subplots(figsize=(10, 6))
        onshore.plot(ax=ax, color="0.93", edgecolor="0.7", linewidth=0.1)
        vals = merged["p_nom_max"].replace(0, np.nan)
        if vals.notna().any():
            vmin = max(1.0, float(np.nanmin(vals)))
            vmax = float(np.nanmax(vals))
            merged.plot(
                column="p_nom_max",
                cmap="plasma",
                norm=LogNorm(vmin=vmin, vmax=vmax),
                linewidth=0.3,
                edgecolor="0.3",
                ax=ax,
                legend=True,
                legend_kwds={"label": "p_nom_max per bus (MW, log scale)"},
                missing_kwds={"color": "white", "edgecolor": "0.5", "linewidth": 0.2},
            )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(
            f"{TECH_LABEL[args.tech]} capacity potential per bus "
            f"(NREL {args.access_for_map} supply curve)\n"
            f"Σ = {total_gw:,.1f} GW across {n_buses} western offshore buses",
        )
        fig.tight_layout()
        fig.savefig(map_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {map_path}")
    else:
        plot_pnom_map(
            caps_df=caps_df,
            shapes=shapes,
            tech=args.tech,
            access=args.access_for_map,
            out_path=map_path,
            xlim=xlim,
            ylim=ylim,
        )

    # Bar chart
    bar_path = out_dir / f"capacity_totals_by_access_{args.tech}.png"
    plot_totals_by_access(
        caps_dir=caps_dir,
        tech=args.tech,
        out_path=bar_path,
    )


if __name__ == "__main__":
    main()
