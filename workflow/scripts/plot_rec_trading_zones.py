"""
Render the REC trading zone map for the documentation.

Draws every CONUS state coloured by the REC trading zone that
``build_base_network`` assigns to its buses (``constants.REC_TRADING_ZONE_MAPPER``;
a state absent from the mapper is its own zone). Portfolio standards are
enforced per zone, so this is the geography over which RPS/CES compliance is
pooled.

Needs only the ReEDS zone shapes and membership shipped in ``repo_data``.
Run via ``snakemake docs_rec_trading_zones`` (see ``workflow/Snakefile``).
"""

import logging

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from constants import REC_TRADING_ZONE_MAPPER
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)

CRS = "EPSG:5070"
# one colour per multi-state tracking system; single-state zones share a neutral fill
ZONE_COLORS = {
    "WREGIS": "#2f6fd6",
    "MRETS": "#1b998b",
    "PJM-GATS": "#a52a2a",
    "NEPOOL": "#b07aa1",
    "MIRECS": "#e07a5f",
    "NAR": "#f4c542",
    "NYGATS": "#8fb8ff",
    "NC-RETS": "#2c5f2d",
    "ERCOT": "#ff8c00",
}
OWN_ZONE_COLOR = "#d9dee3"


def plot_rec_trading_zones(shapes_path, membership_path, out_path):
    zones = gpd.read_file(shapes_path).to_crs(CRS)
    membership = pd.read_csv(membership_path).rename(columns={"ba": "rb"})
    zones = zones.merge(membership[["rb", "st", "country"]], on="rb", how="left")
    states = zones[zones.country == "USA"].dissolve(by="st").reset_index()
    # close the hairline gaps between ReEDS zone polygons so state and zone outlines draw clean
    states["geometry"] = states.geometry.buffer(500).buffer(-500)

    # same rule as build_base_network: mapped states join a tracking system, the rest stand alone
    states["rec_trading_zone"] = states.st.map(REC_TRADING_ZONE_MAPPER).fillna(states.st)
    states["own_zone"] = ~states.st.isin(REC_TRADING_ZONE_MAPPER)
    states["color"] = states.rec_trading_zone.map(ZONE_COLORS).fillna(OWN_ZONE_COLOR)

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    states.plot(ax=ax, color=states.color, edgecolor="white", linewidth=0.8)
    states.dissolve(by="rec_trading_zone").boundary.plot(ax=ax, color="#1f2933", linewidth=1.2)
    for _, s in states.iterrows():
        p = s.geometry.representative_point()
        ax.annotate(s.st, (p.x, p.y), ha="center", va="center", fontsize=7, color="#1f2933")

    handles = [Patch(facecolor=c, edgecolor="white", label=z) for z, c in ZONE_COLORS.items()]
    own = ", ".join(sorted(states.st[states.own_zone]))
    handles.append(Patch(facecolor=OWN_ZONE_COLOR, edgecolor="white", label=f"own zone (state only): {own}"))
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
        frameon=False,
        title="rec_trading_zone",
        title_fontsize=9,
    )
    ax.set_title(
        f"REC trading zones: {states.rec_trading_zone.nunique()} zones over {len(states)} states "
        "(constants.REC_TRADING_ZONE_MAPPER)",
        fontsize=11,
        loc="left",
        fontweight="bold",
    )
    ax.set_axis_off()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("docs_rec_trading_zones")
    logging.basicConfig(level=logging.INFO)
    plot_rec_trading_zones(
        snakemake.input.reeds_shapes,
        snakemake.input.reeds_memberships,
        snakemake.output.rec_trading_zones,
    )
