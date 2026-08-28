"""Legacy per-stage plot helpers reused by the revamped report sections."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..compare import load_network
from ..paths import INTERCONNECT as IC
from ..paths import ArtifactPair

LABELS = {"candidate": "V1-epic", "anchor": "anchor"}


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _img(fig, title: str) -> str:
    return f'<figure><img src="{_png(fig)}" style="max-width:100%"><figcaption>{title}</figcaption></figure>'


def demand_overlay(pair: ArtifactPair, cand_root: Path, anch_root: Path) -> str:
    fc = pd.read_csv(cand_root / pair.candidate, index_col=0, parse_dates=True)
    fa = pd.read_csv(anch_root / pair.anchor, index_col=0, parse_dates=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(fc.index, fc.sum(axis=1), label=LABELS["candidate"], lw=1.2)
    axes[0].plot(fa.index, fa.sum(axis=1), label=LABELS["anchor"], lw=1.2, ls="--")
    axes[0].set_ylabel("System demand (MW)")
    axes[0].legend()
    diff = fc.sum(axis=1) - fa.sum(axis=1).reindex(fc.index)
    axes[1].plot(fc.index, diff, color="crimson", lw=1)
    axes[1].set_ylabel("candidate - anchor (MW)")
    fig.suptitle("Demand: system total overlay")
    return _img(fig, "Demand overlay (per-bus deltas appear in findings)")


def profile_curves(pair: ArtifactPair, cand_root: Path, anch_root: Path) -> str:
    import xarray as xr

    with (
        xr.open_dataset(cand_root / pair.candidate) as dc,
        xr.open_dataset(
            anch_root / pair.anchor,
        ) as da,
    ):
        pc = dc["profile"].transpose("time", "bus").to_pandas()
        pa = da["profile"].transpose("time", "bus").to_pandas()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for name, df, ax_scatter in ((("candidate"), pc, None), (("anchor"), pa, None)):
        cf = df.mean(axis=1).sort_values(ascending=False).reset_index(drop=True)
        axes[0].plot(cf.index / len(cf) * 100, cf, label=name)
    axes[0].set_xlabel("% of hours")
    axes[0].set_ylabel("mean CF")
    axes[0].legend()
    axes[0].set_title("Duration curve (bus-mean)")
    common = sorted(set(pc.columns.map(str)) & set(pa.columns.map(str)))
    pc.columns, pa.columns = pc.columns.map(str), pa.columns.map(str)
    axes[1].scatter(pa[common].mean(), pc[common].mean(), s=6, alpha=0.5)
    lim = [0, max(0.01, pa[common].mean().max(), pc[common].mean().max())]
    axes[1].plot(lim, lim, "k--", lw=0.8)
    axes[1].set_xlabel("anchor mean CF/bus")
    axes[1].set_ylabel("candidate mean CF/bus")
    axes[1].set_title("Per-bus mean CF")
    fig.suptitle(pair.stage)
    return _img(fig, pair.stage)


def capacity_bars(pair: ArtifactPair, cand_root: Path, anch_root: Path) -> str:
    nc = load_network(cand_root / pair.candidate)
    na = load_network(anch_root / pair.anchor)
    cap = "p_nom_opt" if pair.solve_stage else "p_nom"
    gc = nc.generators.groupby("carrier")[cap].sum()
    ga = na.generators.groupby("carrier")[cap].sum()
    df = pd.DataFrame({LABELS["candidate"]: gc, LABELS["anchor"]: ga}).fillna(0.0) / 1e3
    fig, ax = plt.subplots(figsize=(8, 0.4 * max(len(df), 4) + 1))
    df.sort_values("anchor").plot.barh(ax=ax)
    ax.set_xlabel(f"{cap} (GW)")
    ax.set_title(f"{pair.stage}: generator {cap} by carrier")
    return _img(fig, f"{pair.stage} capacity by carrier")


def scatter_network(pair: ArtifactPair, cand_root: Path, anch_root: Path) -> str:
    nc = load_network(cand_root / pair.candidate)
    na = load_network(anch_root / pair.anchor)
    panels = [
        ("Load p_set mean/bus", nc.loads_t.p_set.mean(), na.loads_t.p_set.mean()),
        ("Generator p_nom", nc.generators.p_nom, na.generators.p_nom),
        ("Generator marginal_cost", nc.generators.marginal_cost, na.generators.marginal_cost),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.6))
    for ax, (title, a, b) in zip(np.atleast_1d(axes), panels):
        a.index, b.index = a.index.map(str), b.index.map(str)
        common = sorted(set(a.index) & set(b.index))
        if not common:
            ax.set_title(f"{title}: no common ids")
            continue
        ax.scatter(b[common], a[common], s=5, alpha=0.4)
        lim = [0, max(1e-9, float(np.nanmax(b[common])), float(np.nanmax(a[common])))]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel(LABELS["anchor"])
        ax.set_ylabel(LABELS["candidate"])
        ax.set_title(title, fontsize=9)
    fig.suptitle(pair.stage)
    return _img(fig, f"{pair.stage} per-element scatters")


def _pnom_grouped(n, attr: str):
    """Generator p_nom summed by an attribute (carrier / zone / decade)."""
    g = n.generators
    if g.empty:
        return pd.Series(dtype=float)
    if attr == "carrier":
        key = g.carrier
    elif attr == "reeds_zone":
        if "reeds_zone" not in n.buses.columns:
            return pd.Series(dtype=float)
        key = g.bus.map(n.buses.reeds_zone)
    elif attr == "build_year":
        key = (g.build_year.fillna(0).astype(float) // 10 * 10).astype(int).astype(str) + "s"
    else:
        return pd.Series(dtype=float)
    return g.groupby(key).p_nom.sum()


def pnom_attribute_section(prong: int, cand_root: Path, anch_root: Path) -> str:
    """Paired p_nom-by-attribute bars for the add_electricity outputs and the
    simplify-stage networks (candidate assembled pkl vs each anchor stage).
    """
    s = "" if prong == 1 else "20"
    pairings = [
        (
            "add_electricity outputs",
            cand_root / f"resources/equivalence/networks/{IC}/elec_s{s}_l_pp.pkl",
            anch_root / f"resources/equivalence/{IC}/elec_base_network_l_pp.pkl",
            LABELS["candidate"] + " (substation)",
            "anchor (nodal)",
        ),
        (
            "simplify_network stage",
            cand_root / f"resources/equivalence/networks/{IC}/elec_s{s}_l_pp.pkl",
            anch_root / f"resources/equivalence/{IC}/elec_s{s}.nc",
            LABELS["candidate"] + " (assembled)",
            "anchor (simplified)",
        ),
    ]
    out: list[str] = []
    for title, pc, pa, lc, la in pairings:
        if not (pc.exists() and pa.exists()):
            out.append(f"<p>p_nom section: missing artifact for {title}</p>")
            continue
        try:
            nc, na = load_network(pc), load_network(pa)
            attrs = ["carrier", "reeds_zone", "build_year"]
            fig, axes = plt.subplots(1, len(attrs), figsize=(5.2 * len(attrs), 4.2))
            for ax, attr in zip(np.atleast_1d(axes), attrs):
                gc, ga = _pnom_grouped(nc, attr), _pnom_grouped(na, attr)
                if gc.empty and ga.empty:
                    ax.set_title(f"by {attr}: n/a")
                    continue
                df = pd.DataFrame({lc: gc, la: ga}).fillna(0.0) / 1e3
                df.sort_index().plot.barh(ax=ax, width=0.8)
                ax.set_xlabel("p_nom (GW)")
                ax.set_title(f"by {attr}", fontsize=10)
                ax.tick_params(labelsize=7)
            fig.suptitle(f"Generator p_nom by attribute - {title}")
            fig.tight_layout()
            out.append(_img(fig, f"p_nom by attribute: {title}"))
        except Exception as e:
            out.append(f"<p>p_nom plot failed for {title}: {e}</p>")
    return "".join(out)
