"""
Plots EGS site availability with seismic risk exclusion overlay.

**Inputs**

- ``specs_egs``: EGS supply curve specifications (NetCDF), indexed by sub_id
- ``bus2sub``: Bus-to-substation mapping (CSV) with x/y coordinates per sub_id
- ``seismic_exclusion``: Seismic risk mask (GeoJSON, EPSG:3857 point grid, ``seismic risk`` 0/1)
- ``regions_onshore``: Onshore region shapes for map background (GeoJSON)

**Outputs**

- ``plot``: PNG map of EGS sites colored by inclusion/exclusion status
"""

import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from _helpers import configure_logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_egs_seismic_availability",
            interconnect="western",
            simpl="10",
            clusters="30",
            ll="v1.0",
            opts="REM-4h",
            sector="",
        )
    configure_logging(snakemake)

    # Load EGS specs and resolve sub_id → (x, y) via bus2sub
    with xr.open_dataset(snakemake.input.specs_egs) as ds:
        df_specs = ds.sel(Quality=1).to_dataframe().reset_index().dropna(subset=["avail_capacity_mw"])
    df_specs["sub_id"] = df_specs["sub_id"].astype(str)

    bus2sub = pd.read_csv(snakemake.input.bus2sub, dtype=str)
    bus2sub["sub_id"] = bus2sub["sub_id"].apply(lambda x: str(int(float(x))))
    bus2sub["x"] = bus2sub["x"].astype(float)
    bus2sub["y"] = bus2sub["y"].astype(float)

    df = pd.merge(
        df_specs[["sub_id", "avail_capacity_mw"]],
        bus2sub[["sub_id", "x", "y"]].drop_duplicates("sub_id"),
        on="sub_id",
        how="inner",
    ).dropna(subset=["x", "y"])

    egs_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["x"], df["y"]),
        crs="EPSG:4326",
    )

    # Load seismic mask and classify each EGS site by nearest grid point
    seismic_gdf = gpd.read_file(snakemake.input.seismic_exclusion).rename(
        columns={"seismic risk": "seismic_risk"},
    )

    egs_projected = egs_gdf.to_crs(seismic_gdf.crs)
    egs_with_risk = gpd.sjoin_nearest(
        egs_projected[["sub_id", "avail_capacity_mw", "geometry"]],
        seismic_gdf[["geometry", "seismic_risk"]],
        how="left",
    )

    included = egs_with_risk[egs_with_risk["seismic_risk"] == 0]
    excluded = egs_with_risk[egs_with_risk["seismic_risk"] == 1]
    logger.info(f"EGS sites: {len(included)} included, {len(excluded)} excluded by seismic risk.")

    # Background regions
    regions = gpd.read_file(snakemake.input.regions_onshore).to_crs(seismic_gdf.crs)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 9))

    regions.plot(ax=ax, color="whitesmoke", edgecolor="gray", linewidth=0.5, zorder=1)

    seismic_gdf[seismic_gdf["seismic_risk"] == 1].plot(
        ax=ax,
        color="lightcoral",
        alpha=0.4,
        markersize=0.5,
        zorder=2,
        label="High seismic risk zone",
    )

    if not included.empty:
        included.plot(
            ax=ax,
            color="steelblue",
            markersize=6,
            zorder=3,
            label=f"EGS included ({len(included)})",
        )

    if not excluded.empty:
        excluded.plot(
            ax=ax,
            color="crimson",
            markersize=6,
            zorder=4,
            label=f"EGS excluded — seismic risk ({len(excluded)})",
        )

    ax.set_title("EGS Site Availability: Seismic Risk Exclusion")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.legend(loc="lower left")

    fig.savefig(snakemake.output.plot, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved EGS seismic availability plot to {snakemake.output.plot}")
