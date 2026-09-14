"""Standalone PNG plot exporter for the equivalence harness.

Replaces the HTML report (user decision 2026-09-01): each figure is written as
an individual PNG under ``workflow/results/equivalence/plots{suffix}/`` so it
can be viewed directly.

Zone joins use the ``reeds_zone`` BUS ATTRIBUTE, never string surgery on
cluster names: cluster IDs are ``p{zone}{subcluster} {i}`` with no separator
("p101 1" is zone p10, sub-cluster 1), so any prefix split silently produces
labels that match almost nothing in reeds_shapes.geojson (the bug behind the
first, mostly-grey report maps).

- develop profile bus -> zone: the s{simpl} clustered network's
  ``buses.reeds_zone``.
- master profile bus -> zone: SUBSTATION id -> busmap_s{simpl} -> cluster ->
  ``reeds_zone``. The baseline's profile buses are substation ids ('39762.0'),
  NOT its base-network bus ids — the two numbering spaces overlap numerically
  (base buses are 1..82549) but a direct join lands on the right zone only
  ~2% of the time (verified 2026-09-01). The busmap chain is the same join
  develop's caps remap uses, which matches 100%.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .build import BASELINE_WORKTREE
from .paths import EQ, INTERCONNECT, SIMPL2, prong_pairs

REPO = Path(__file__).resolve().parents[2]
DEVELOP_ROOT = REPO / "workflow"
MASTER_ROOT = BASELINE_WORKTREE / "workflow"
_SUF = "" if INTERCONNECT == "western" else f"_{INTERCONNECT}"
OUTDIR = DEVELOP_ROOT / "results" / "equivalence" / f"plots{_SUF}"

LABELS = {"develop": "develop", "master": "master-benchmark (baseline)"}
DPI = 150


def _zone_maps() -> tuple[pd.Series, pd.Series, gpd.GeoDataFrame]:
    """(develop bus->zone, master sub->zone, zone shapes)."""
    import pypsa

    zones = gpd.read_file(DEVELOP_ROOT / f"{EQ}/geospatial/{INTERCONNECT}/reeds_shapes.geojson").set_index("name")
    nc = pypsa.Network(DEVELOP_ROOT / f"{EQ}/networks/{INTERCONNECT}/elec_s{SIMPL2}.nc")
    zc = nc.buses["reeds_zone"].astype(str)
    busmap = pd.read_csv(
        DEVELOP_ROOT / f"{EQ}/busmaps/{INTERCONNECT}/busmap_s{SIMPL2}.csv", index_col=0, dtype=str
    ).iloc[:, 0]
    busmap.index = busmap.index.astype(str)
    za = busmap.map(zc)  # substation id -> cluster -> reeds_zone
    return zc, za, zones


def _norm_ids(idx, mapper: pd.Series) -> pd.Series:
    """Map profile bus ids onto a bus->zone Series, tolerating '39762.0' vs '39762'."""
    ids = pd.Index([str(b) for b in idx])
    out = pd.Series(ids, index=ids).map(mapper)
    miss = out.isna()
    if miss.any():
        norm = []
        for b in out.index[miss]:
            try:
                norm.append(str(int(float(b))))
            except (TypeError, ValueError):
                norm.append(b)
        out.loc[miss] = pd.Series(norm, index=out.index[miss]).map(mapper).values
    return out


def _save(fig, name: str) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = OUTDIR / f"{name}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {p}")
    return p


def _choropleth_row(zones, vc: pd.Series, va: pd.Series, title: str, unit: str, name: str):
    gc, ga = zones.join(vc.rename("v")), zones.join(va.rename("v"))
    diff = gc["v"] - ga["v"]
    vmax = float(np.nanmax([gc["v"].max(), ga["v"].max()]))
    dmax = float(np.nanmax(np.abs(diff))) or 1e-9
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, g, lab in ((axes[0], gc, LABELS["develop"]), (axes[1], ga, LABELS["master"])):
        g.plot(column="v", ax=ax, cmap="viridis", vmin=0, vmax=vmax, legend=True, missing_kwds={"color": "#dddddd"})
        ax.set_title(f"{lab}\n{title} [{unit}]", fontsize=9)
    zones.join(diff.rename("v")).plot(
        column="v", ax=axes[2], cmap="RdBu_r", vmin=-dmax, vmax=dmax, legend=True, missing_kwds={"color": "#dddddd"}
    )
    axes[2].set_title(f"{LABELS['develop']} − {LABELS['master']}\n(grey = no data)", fontsize=9)
    for ax in axes:
        ax.set_axis_off()
    _save(fig, name)


def export_profile_plots() -> None:
    zc, za, zones = _zone_maps()
    pairs = [p for p in prong_pairs(2) if p.kind == "profile"]
    for pair in pairs:
        tech = pair.stage.replace("profile_", "")
        pc, pa = DEVELOP_ROOT / pair.develop, MASTER_ROOT / pair.master
        if not (pc.exists() and pa.exists()):
            print(f"[plots] skipping {tech}: artifact missing")
            continue
        with xr.open_dataset(pc) as dsc, xr.open_dataset(pa) as dsa:
            stats = {}
            for side, ds, zmap in (("develop", dsc, zc), ("master", dsa, za)):
                pnom = ds["p_nom_max"].to_pandas()
                pnom.index = pnom.index.map(str)
                cf = ds["profile"].mean("time").to_pandas()
                cf.index = cf.index.map(str)
                z = _norm_ids(pnom.index, zmap)
                unmapped = z.isna().sum()
                if unmapped:
                    print(f"[plots] {tech}/{side}: {unmapped}/{len(z)} profile buses have no reeds_zone")
                pot = pnom.groupby(z.values).sum()
                wcf = (cf * pnom).groupby(z.values).sum() / pot.replace(0.0, np.nan)
                stats[side] = (pot / 1e3, wcf)
            avail_c = (dsc["profile"] * dsc["p_nom_max"]).sum("bus").to_pandas() / 1e3
            avail_a = (dsa["profile"] * dsa["p_nom_max"]).sum("bus").to_pandas() / 1e3

        _choropleth_row(zones, stats["develop"][0], stats["master"][0], f"{tech} installable potential", "GW", f"{tech}_potential_zones")
        _choropleth_row(zones, stats["develop"][1], stats["master"][1], f"{tech} potential-weighted mean CF", "-", f"{tech}_meancf_zones")

        day_c, day_a = avail_c.resample("D").mean(), avail_a.resample("D").mean()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True, height_ratios=[3, 1])
        ax1.plot(day_c.index, day_c.values, label=LABELS["develop"], lw=1.0)
        ax1.plot(day_a.index, day_a.values, label=LABELS["master"], lw=1.0, alpha=0.75)
        ax1.set_ylabel("national available power [GW]\n(daily mean)")
        ax1.legend(fontsize=8)
        ax1.set_title(f"{tech}: Σ profile·p_nom_max over all buses")
        rel = 100.0 * (day_c - day_a) / day_a.replace(0.0, np.nan)
        ax2.axhline(0, color="#999999", lw=0.5)
        ax2.plot(rel.index, rel.values, color="#a03030", lw=0.8)
        ax2.set_ylabel("rel diff [%]")
        _save(fig, f"{tech}_national_available_power")


def export_all() -> Path:
    export_profile_plots()
    return OUTDIR


if __name__ == "__main__":
    export_all()
