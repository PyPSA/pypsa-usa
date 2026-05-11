"""
Compare LEGACY vs NREL profile_*.nc files on the shared bus set.

For each tech:
  1) annual mean CF per bus
  2) p_nom_max per bus (MW)
  3) annual energy potential per bus (MWh) = (profile * p_nom_max).sum(time)

Writes one PNG per tech with three LEGACY-vs-NREL scatter panels, plus a
summary bar chart of fleet-wide totals across techs, plus a CSV with
per-bus metrics on the intersection.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


TECH_LABEL = {
    "solar": "Solar",
    "onwind": "Onshore wind",
    "offwind_floating": "Offshore wind (floating)",
}


def load_pair(profiles_dir: Path, tech: str):
    leg = xr.open_dataset(profiles_dir / f"profile_{tech}_LEGACY.nc")
    nrl = xr.open_dataset(profiles_dir / f"profile_{tech}_NREL.nc")
    return leg, nrl


def metrics_on_intersection(leg: xr.Dataset, nrl: xr.Dataset) -> pd.DataFrame:
    leg_buses = set(leg["bus"].values.astype(str))
    nrl_buses = set(nrl["bus"].values.astype(str))
    shared = sorted(leg_buses & nrl_buses)

    leg = leg.assign_coords(bus=leg["bus"].astype(str)).sel(bus=shared)
    nrl = nrl.assign_coords(bus=nrl["bus"].astype(str)).sel(bus=shared)

    leg_cf = leg["profile"].mean("time").values
    nrl_cf = nrl["profile"].mean("time").values
    leg_pnom = leg["p_nom_max"].values
    nrl_pnom = nrl["p_nom_max"].values
    leg_energy = (leg["profile"] * leg["p_nom_max"]).sum("time").values
    nrl_energy = (nrl["profile"] * nrl["p_nom_max"]).sum("time").values

    return pd.DataFrame(
        {
            "bus": shared,
            "cf_legacy": leg_cf,
            "cf_nrel": nrl_cf,
            "pnom_legacy_mw": leg_pnom,
            "pnom_nrel_mw": nrl_pnom,
            "energy_legacy_mwh": leg_energy,
            "energy_nrel_mwh": nrl_energy,
        }
    )


def _scatter(ax, x, y, *, log: bool, xlabel: str, ylabel: str, title: str):
    mask = np.isfinite(x) & np.isfinite(y)
    if log:
        mask &= (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    ax.scatter(x, y, s=10, alpha=0.5, edgecolor="none")
    if x.size > 0:
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        if log:
            lo = max(lo, 1e-6)
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
            ax.set_xscale("log"); ax.set_yscale("log")
        else:
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_tech(df: pd.DataFrame, tech: str, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    n_buses = len(df)

    _scatter(
        axes[0], df["cf_legacy"].values, df["cf_nrel"].values,
        log=False,
        xlabel="LEGACY annual mean CF",
        ylabel="NREL annual mean CF",
        title=f"Annual mean CF\n(median LEG={df['cf_legacy'].median():.3f}, "
              f"NREL={df['cf_nrel'].median():.3f})",
    )
    _scatter(
        axes[1], df["pnom_legacy_mw"].values, df["pnom_nrel_mw"].values,
        log=True,
        xlabel="LEGACY p_nom_max (MW, log)",
        ylabel="NREL p_nom_max (MW, log)",
        title=f"p_nom_max per bus\n(Σ LEG={df['pnom_legacy_mw'].sum()/1e3:,.0f} GW, "
              f"NREL={df['pnom_nrel_mw'].sum()/1e3:,.0f} GW)",
    )
    _scatter(
        axes[2], df["energy_legacy_mwh"].values, df["energy_nrel_mwh"].values,
        log=True,
        xlabel="LEGACY annual energy (MWh, log)",
        ylabel="NREL annual energy (MWh, log)",
        title=f"Annual energy potential\n(Σ LEG={df['energy_legacy_mwh'].sum()/1e6:,.1f} TWh, "
              f"NREL={df['energy_nrel_mwh'].sum()/1e6:,.1f} TWh)",
    )

    fig.suptitle(
        f"{TECH_LABEL[tech]}: LEGACY vs NREL on {n_buses} shared buses",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_totals(summary: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    techs = summary["tech"].tolist()
    x = np.arange(len(techs))
    w = 0.38

    pnom_leg = summary["pnom_legacy_gw"].values
    pnom_nrl = summary["pnom_nrel_gw"].values
    axes[0].bar(x - w/2, pnom_leg, w, color="#4C72B0", label="LEGACY (Zenodo)")
    axes[0].bar(x + w/2, pnom_nrl, w, color="#C44E52", label="NREL (reference)")
    for xi, v in zip(x - w/2, pnom_leg):
        axes[0].text(xi, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x + w/2, pnom_nrl):
        axes[0].text(xi, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels([TECH_LABEL[t] for t in techs])
    axes[0].set_ylabel("Σ p_nom_max on shared buses (GW)")
    axes[0].set_title("Capacity potential")
    axes[0].grid(axis="y", alpha=0.3); axes[0].set_axisbelow(True)
    axes[0].legend()

    e_leg = summary["energy_legacy_twh"].values
    e_nrl = summary["energy_nrel_twh"].values
    axes[1].bar(x - w/2, e_leg, w, color="#4C72B0", label="LEGACY (Zenodo)")
    axes[1].bar(x + w/2, e_nrl, w, color="#C44E52", label="NREL (reference)")
    for xi, v in zip(x - w/2, e_leg):
        axes[1].text(xi, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x + w/2, e_nrl):
        axes[1].text(xi, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(x); axes[1].set_xticklabels([TECH_LABEL[t] for t in techs])
    axes[1].set_ylabel("Σ annual energy on shared buses (TWh)")
    axes[1].set_title("Annual energy potential")
    axes[1].grid(axis="y", alpha=0.3); axes[1].set_axisbelow(True)
    axes[1].legend()

    fig.suptitle("LEGACY vs NREL totals on shared bus sets", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profiles-dir",
        default="/home/groups/iazevedo/asia/nrel_exclusion_work/profiles",
    )
    ap.add_argument(
        "--out-dir",
        default="/home/groups/iazevedo/asia/nrel_exclusion_work",
    )
    ap.add_argument(
        "--techs",
        nargs="+",
        default=["solar", "onwind", "offwind_floating"],
    )
    args = ap.parse_args()

    profiles_dir = Path(args.profiles_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for tech in args.techs:
        leg, nrl = load_pair(profiles_dir, tech)
        n_leg, n_nrl = leg.sizes["bus"], nrl.sizes["bus"]
        df = metrics_on_intersection(leg, nrl)
        leg.close(); nrl.close()

        df.to_csv(out_dir / f"compare_legacy_vs_nrel_{tech}.csv", index=False)

        plot_tech(df, tech, out_dir / f"compare_legacy_vs_nrel_{tech}.png")

        rows.append({
            "tech": tech,
            "n_legacy": n_leg,
            "n_nrel": n_nrl,
            "n_shared": len(df),
            "cf_legacy_med": df["cf_legacy"].median(),
            "cf_nrel_med": df["cf_nrel"].median(),
            "pnom_legacy_gw": df["pnom_legacy_mw"].sum() / 1e3,
            "pnom_nrel_gw": df["pnom_nrel_mw"].sum() / 1e3,
            "energy_legacy_twh": df["energy_legacy_mwh"].sum() / 1e6,
            "energy_nrel_twh": df["energy_nrel_mwh"].sum() / 1e6,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "compare_legacy_vs_nrel_summary.csv", index=False)
    plot_totals(summary, out_dir / "compare_legacy_vs_nrel_totals.png")

    print("\n=== Summary on shared buses ===")
    with pd.option_context("display.float_format", "{:,.3f}".format,
                           "display.width", 160):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
