"""Self-contained HTML visual report for one equivalence run (spec D4/D11).

Single file, matplotlib PNGs embedded as base64 data URIs, no external
assets. Sections: run header, per-stage pass/fail summary, demand overlays,
profile duration curves, capacity-by-carrier bars, per-bus scatters,
worst-offender findings, waivers, benchmark table.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .compare import load_network, load_waivers
from .paths import ArtifactPair, prong_pairs

REPO = Path(__file__).resolve().parents[2]


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
    axes[0].plot(fc.index, fc.sum(axis=1), label="candidate", lw=1.2)
    axes[0].plot(fa.index, fa.sum(axis=1), label="anchor", lw=1.2, ls="--")
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
    df = pd.DataFrame({"candidate": gc, "anchor": ga}).fillna(0.0) / 1e3
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
        ax.set_xlabel("anchor")
        ax.set_ylabel("candidate")
        ax.set_title(title, fontsize=9)
    fig.suptitle(pair.stage)
    return _img(fig, f"{pair.stage} per-element scatters")


def benchmark_table() -> str:
    rows = []
    for side in ("candidate", "anchor"):
        p = REPO / "workflow" / "results" / "equivalence" / f"manifest_{side}.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        for rule, bench in sorted(m.get("benchmarks", {}).items()):
            rows.append(
                {
                    "side": side,
                    "rule": rule,
                    "wall_s": bench.get("s", ""),
                    "max_rss_mb": bench.get("max_rss", ""),
                },
            )
    if not rows:
        return "<p>No benchmark manifests found.</p>"
    df = pd.DataFrame(rows)
    piv = df.pivot_table(
        index="rule",
        columns="side",
        values="wall_s",
        aggfunc="first",
    ).fillna("")
    return piv.to_html(border=0)


def findings_tables(findings: dict) -> str:
    rows = findings.get("findings", [])
    if not rows:
        return "<p><b>No findings.</b></p>"
    df = pd.DataFrame(rows)
    df["detail"] = df["detail"].astype(str).str.slice(0, 220)
    return df.to_html(border=0, index=False)


def build_report(prong: int) -> Path:
    cand_root = REPO / "workflow"
    anch_root = REPO / ".worktrees" / "anchor-e7f8bd70" / "workflow"
    fpath = REPO / "workflow" / "results" / "equivalence" / f"findings_{prong}.json"
    findings = json.loads(fpath.read_text()) if fpath.exists() else {"findings": []}
    sections: list[str] = []
    for pair in prong_pairs(prong):
        pc, pa = cand_root / pair.candidate, anch_root / pair.anchor
        if not (pc.exists() and pa.exists()):
            continue
        try:
            if pair.kind in ("demand_csv", "demand_total"):
                sections.append(demand_overlay(pair, cand_root, anch_root))
            elif pair.kind == "profile":
                sections.append(profile_curves(pair, cand_root, anch_root))
            elif pair.kind in ("network", "network_pkl_vs_nc"):
                sections.append(capacity_bars(pair, cand_root, anch_root))
                if not pair.solve_stage:
                    sections.append(scatter_network(pair, cand_root, anch_root))
        except Exception as e:  # report must render even when a plot fails
            sections.append(f"<p>plot failed for {pair.stage}: {e}</p>")
    status = "PASS" if findings.get("pass") else "FAIL"
    n_live = findings.get("n_live", "?")
    waivers = load_waivers()
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Equivalence report — prong {prong}</title>
<style>body{{background:#ffffff;color:#111111;font-family:system-ui;margin:2rem;max-width:1100px}}
a{{color:#0b5cad}}h1,h2{{color:#111111}}
table{{border-collapse:collapse;font-size:12px;background:#ffffff;color:#111111}}td,th{{padding:3px 8px;border-bottom:1px solid #ddd;text-align:left;color:#111111}}
figure{{margin:1rem 0;background:#ffffff}}figcaption{{color:#555555;font-size:12px}}</style></head><body>
<h1>Equivalence run — prong {prong}: <span style="color:{"green" if status == "PASS" else "crimson"}">{status}</span></h1>
<p>candidate = v1-epic · anchor = upstream/develop e7f8bd70 · live findings: {n_live}
· waivers active: {len(waivers)}</p>
<h2>Findings</h2>{findings_tables(findings)}
<h2>Visual comparison</h2>{"".join(sections)}
<h2>Benchmarks (wall seconds per rule)</h2>{benchmark_table()}
</body></html>"""
    out = REPO / "workflow" / "results" / "equivalence" / f"report_prong{prong}.html"
    out.write_text(html)
    return out
