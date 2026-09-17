"""Per-cluster p5 / p50 / p95 of ``p_max_pu`` for solar and onwind, master vs develop.

Three views per carrier and quantile, on the USA s300 artifacts:

1. **Unpaired** — each side's own clusters as they sit: develop's
   ``profile_{tech}_s300.nc``, master's own ``elec_s300.nc`` generators (master
   aggregates nodal CFs with its own rule, existing-``p_nom`` weighting with a
   uniform fallback — HF-25 — on its own s300 partition), and master's NODAL
   profile rolled up onto develop's clusters through ``busmap_s300`` with
   develop's rule (``p_nom_max`` weighting; HF-24's dropped substations are simply
   absent from master's file).
2. **Paired** — the reconstruction: master-rolled-up vs develop, one point per
   common cluster. If the only differences between the sides are HF-24 / HF-25,
   the points sit on the 1:1 line.
3. **Residual** — the per-cluster delta of the paired view.

Outputs (PNG + CSV twin) go to ``--out``. Run on a compute node:
``uv run python tests/equivalence/diagnostics/p_max_pu_quantiles.py --out <dir>``
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pypsa  # noqa: E402
import xarray as xr  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tests.equivalence.metrics import aggregate_profile_to_clusters  # noqa: E402

log = logging.getLogger("p_max_pu_quantiles")

DEV = REPO / "workflow/resources/equivalence"
MAS = REPO / ".worktrees/master-benchmark/workflow/resources/equivalence/usa"
TECHS = ("solar", "onwind")
Q = (5, 50, 95)

# categorical slots, fixed order: develop, master own, master reconstructed
C_DEV, C_MAS, C_REC = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e6e5e0"


def quantiles_over_time(da: xr.DataArray) -> pd.DataFrame:
    """``(bus, time)`` -> one row per bus with columns q5, q50, q95 (NaN-safe)."""
    da = da.transpose("bus", ...)
    vals = np.asarray(da.values, dtype=float)
    out = np.nanpercentile(vals, Q, axis=1).T
    return pd.DataFrame(out, index=[str(b) for b in da.indexes["bus"]], columns=[f"q{q}" for q in Q])


def load_busmap() -> pd.Series:
    p = DEV / "busmaps/usa/busmap_s300.csv"
    bm = pd.read_csv(p, index_col=0, dtype=str).iloc[:, 0]
    bm.index = bm.index.astype(str)
    return bm.astype(str)


def cluster_zone_map() -> pd.Series:
    n = pypsa.Network(str(DEV / "networks/usa/elec_s300.nc"))
    return n.buses["reeds_zone"].astype(str)


def master_own_s300(tech: str) -> pd.DataFrame:
    n = pypsa.Network(str(MAS / "elec_s300.nc"))
    gens = n.generators[n.generators.carrier == tech]
    ts = n.generators_t.p_max_pu
    cols = [g for g in gens.index if g in ts.columns]
    log.info("master elec_s300 %s: %d generators, %d with a p_max_pu series", tech, len(gens), len(cols))
    da = xr.DataArray(ts[cols].to_numpy().T, dims=("bus", "time"), coords={"bus": gens.loc[cols, "bus"].astype(str).values})
    return quantiles_over_time(da)


def one_tech(tech: str, busmap: pd.Series, zones: pd.Series, out: Path) -> pd.DataFrame:
    ds_dev = xr.open_dataset(DEV / f"profiles/usa/profile_{tech}_s300.nc")
    ds_mas = xr.open_dataset(MAS / f"profile_{tech}.nc")
    log.info("%s: develop %d clusters, master %d nodal buses", tech, ds_dev.sizes["bus"], ds_mas.sizes["bus"])

    q_dev = quantiles_over_time(ds_dev["profile"])
    ds_rec = aggregate_profile_to_clusters(ds_mas, busmap, weight="p_nom_max")
    q_rec = quantiles_over_time(ds_rec["profile"])
    q_own = master_own_s300(tech)

    common = q_dev.index.intersection(q_rec.index)
    paired = q_dev.loc[common].join(q_rec.loc[common], lsuffix="_develop", rsuffix="_master_recon")
    paired = paired[paired.notna().all(axis=1)]
    for q in Q:
        paired[f"q{q}_delta"] = paired[f"q{q}_master_recon"] - paired[f"q{q}_develop"]
    paired.insert(0, "reeds_zone", zones.reindex(paired.index).values)
    paired.index.name = "cluster"
    paired.to_csv(out / f"p_max_pu_quantiles_{tech}.csv")
    q_own.index.name = "master_cluster"
    q_own.to_csv(out / f"p_max_pu_quantiles_{tech}_master_own_s300.csv")

    dropped = int(ds_rec.attrs.get("eq_dropped_buses", 0))
    log.info(
        "%s: %d common clusters (develop %d, recon %d, master own %d); master nodal buses absent from busmap: %d",
        tech, len(paired), len(q_dev), len(q_rec), len(q_own), dropped,
    )

    fig, axes = plt.subplots(3, 3, figsize=(15, 11.5), facecolor="#fcfcfb")
    fig.suptitle(
        f"USA s300 — {tech} p_max_pu per resource cluster, master vs develop\n"
        f"paired view: master's nodal profile rolled onto develop's {len(paired)} clusters "
        f"(busmap_s300, p_nom_max-weighted = develop's rule, HF-25; HF-24-dropped substations absent)",
        fontsize=12, color=INK,
    )
    summary_rows = []
    for i, q in enumerate(Q):
        col = f"q{q}"
        # --- 1. unpaired sorted curves
        ax = axes[i, 0]
        for series, color, label, ls in (
            (q_dev[col], C_DEV, f"develop s300 ({len(q_dev)} clusters)", "-"),
            (q_own[col], C_MAS, f"master own s300, HF-25 rule ({len(q_own)} clusters)", "-"),
            (q_rec[col], C_REC, f"master rolled onto develop clusters ({len(q_rec)})", "--"),
        ):
            s = np.sort(series.dropna().to_numpy())
            ax.plot(np.linspace(0, 1, len(s)), s, color=color, lw=2, ls=ls, label=label)
        ax.set_xlabel("clusters, ranked (fraction)", color=MUTED)
        ax.set_ylabel(f"p{q} of p_max_pu over the year", color=INK)
        ax.set_title(f"p{q}: each side's own clusters (unpaired)", fontsize=10, color=INK)
        if i == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")
        # --- 2. paired scatter
        ax = axes[i, 1]
        x, y = paired[f"{col}_develop"], paired[f"{col}_master_recon"]
        lim = (0, max(float(x.max()), float(y.max())) * 1.05 + 1e-9)
        ax.plot(lim, lim, color=GRID, lw=1.5, zorder=1)
        ax.scatter(x, y, s=22, color=C_REC, edgecolor="#fcfcfb", linewidth=0.6, zorder=2)
        d = paired[f"{col}_delta"]
        rmse = float(np.sqrt((d**2).mean()))
        within = int((d.abs() <= 0.005).sum())
        ax.text(
            0.03, 0.97,
            f"n = {len(d)}\nmax |Δ| = {d.abs().max():.4f}\nRMSE = {rmse:.5f}\n|Δ| ≤ 0.005: {within}/{len(d)}",
            transform=ax.transAxes, va="top", fontsize=9, color=INK,
        )
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"develop p{q}", color=INK)
        ax.set_ylabel(f"master, reconstructed p{q}", color=INK)
        ax.set_title(f"p{q}: paired per cluster after reconstruction", fontsize=10, color=INK)
        # --- 3. residual bars
        ax = axes[i, 2]
        ds = d.sort_values()
        ax.bar(np.arange(len(ds)), ds.to_numpy(), width=1.0, color=C_REC, edgecolor="none")
        ax.axhline(0, color=INK, lw=0.8)
        ax.axhspan(-0.005, 0.005, color=GRID, alpha=0.6, zorder=0)
        worst = ds.abs().sort_values(ascending=False).head(3)
        ax.text(
            0.03, 0.97,
            "largest |Δ|: " + ", ".join(f"{c} ({paired.loc[c, 'reeds_zone']}) {ds[c]:+.4f}" for c in worst.index),
            transform=ax.transAxes, va="top", fontsize=8, color=INK, wrap=True,
        )
        ax.set_xlabel("clusters, ranked by Δ", color=MUTED)
        ax.set_ylabel(f"Δ p{q} = master recon − develop", color=INK)
        ax.set_title(f"p{q}: residual per cluster (band = ±0.005)", fontsize=10, color=INK)
        summary_rows.append(
            dict(tech=tech, quantile=f"p{q}", n_clusters=len(d), max_abs_delta=float(d.abs().max()),
                 rmse=rmse, median_abs_delta=float(d.abs().median()), n_within_0p005=within,
                 develop_median=float(x.median()), master_recon_median=float(y.median()),
                 master_own_s300_median=float(q_own[col].median()), master_own_n=len(q_own),
                 master_nodal_buses_dropped=dropped)
        )
    for ax in axes.ravel():
        ax.set_facecolor("#fcfcfb")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / f"p_max_pu_quantiles_{tech}.png", dpi=150)
    plt.close(fig)
    return pd.DataFrame(summary_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    a.out.mkdir(parents=True, exist_ok=True)
    busmap = load_busmap()
    zones = cluster_zone_map()
    summaries = [one_tech(t, busmap, zones, a.out) for t in TECHS]
    s = pd.concat(summaries, ignore_index=True)
    s.to_csv(a.out / "p_max_pu_quantiles_summary.csv", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
