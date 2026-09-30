r"""Does the demand that reaches the buses add up to the demand table? (HF-28).

``build_demand`` splits a zonal demand table across buses with per-bus load
allocation factors. ``build_base_network`` normalises ``LAF_state`` over
``full_state``; the factors are consumed against ``n.buses.reeds_state``
(develop) or ``n.buses.state`` (master-benchmark). Neither reproduces
``full_state`` -- the District of Columbia is its own ``full_state`` group and
no bus carries a DC key -- so ``sum(laf)`` inside a demand key was not 1 and the
model got more demand than the table holds.

This checker measures that, per key and nationally, from a built network and the
resolved zonal demand table. Run it on a `develop` artifact before and after the
fix; the baseline it was written against is

    usa, s300, develop, 2026-09-15 (elec_s300_dem.nc):
      allocated 470,193.2 MW vs table 461,636.2 MW -> +1.85 %
      Maryland sum(LAF_state) 2.008, allocated +100.8 %

Usage::

    python -m tests.equivalence.diagnostics.demand_conservation \
        --network  workflow/resources/equivalence/networks/usa/elec_s300_dem.nc \
        --demand-table workflow/resources/equivalence/demand/usa/power_zonal_components_s300.parquet \
        --outdir  /tmp/demand_conservation

Writes ``demand_conservation.csv`` and ``demand_conservation.png`` (per-key
bars, table vs allocated, with a signed error panel) into ``--outdir`` and
prints the table and the national error.

The demand table is the reader's own output, persisted by the ``build_*_demand``
rules as ``resources/<run>/demand/<interconnect>/power_zonal_components_s{simpl}.parquet``
(a snapshot x sector x subsector x fuel frame with ONE COLUMN PER SOURCE ZONE).
If a run predates that artifact there is no other file that holds the
pre-disaggregation table -- say so rather than reconstructing it from the
allocation, which is the quantity under test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pypsa

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "workflow" / "scripts"))

import constants as const  # noqa: E402

CODE_2_STATE = const.CODE_2_STATE

MASTER_COLOR = "#1f77b4"
DEVELOP_COLOR = "#ff7f0e"


def bus_keys(n: pypsa.Network, key: str = "auto") -> tuple[pd.Series, str]:
    """Map every bus to the demand key its load was allocated from.

    ``master-benchmark`` disaggregates on ``n.buses.state`` (full state names)
    at the nodal stage; ``develop`` disaggregates on ``n.buses.reeds_state``
    (two-letter codes) at the ``s{simpl}`` stage, because
    ``aggregate_to_substations`` drops ``state``. Returns the mapping and the
    name of the column it came from.
    """
    has_state = "state" in n.buses.columns and n.buses.state.notna().any()
    has_reeds = "reeds_state" in n.buses.columns and n.buses.reeds_state.notna().any()

    if key == "auto":
        key = "state" if has_state else "reeds_state"
    if key == "state":
        if not has_state:
            raise SystemExit("network has no usable 'state' column; pass --key reeds_state")
        return n.buses.state.astype(str), "state"
    if key == "reeds_state":
        if not has_reeds:
            raise SystemExit("network has no usable 'reeds_state' column; pass --key state")
        return n.buses.reeds_state.map(CODE_2_STATE), "reeds_state"
    raise SystemExit(f"unknown --key {key!r}")


def allocated_by_key(n: pypsa.Network, keys: pd.Series) -> pd.Series:
    """Mean MW of load attached to the buses of each demand key."""
    per_bus = n.loads_t.p_set.mean()
    bus_of_load = n.loads.bus.reindex(per_bus.index)
    return per_bus.groupby(bus_of_load.map(keys)).sum()


def table_by_key(path: Path, snapshots: pd.DatetimeIndex) -> pd.Series:
    """Mean MW per demand key in the resolved zonal table, on the network's snapshots.

    The table carries every sector/subsector/fuel component on its index; the
    model's load is their sum. It is hourly for a whole calendar year while the
    network may be sampled (``3h``), so it is restricted to the network's own
    timestamps before the mean -- otherwise the comparison measures the sampling.
    """
    if path.suffix == ".parquet":
        table = pd.read_parquet(path)
    else:
        table = pd.read_csv(path, index_col=[0, 1, 2, 3])
    levels = [name for name in table.index.names if name != "snapshot"]
    hourly = table.groupby(level="snapshot").sum() if levels else table

    stamps = pd.DatetimeIndex(pd.to_datetime(snapshots))
    index = pd.DatetimeIndex(pd.to_datetime(hourly.index))
    hourly.index = index
    common = index.intersection(stamps)
    if len(common) == 0:
        raise SystemExit(
            f"no overlap between the demand table ({index.min()}..{index.max()}) and the "
            f"network snapshots ({stamps.min()}..{stamps.max()}); wrong table for this network?",
        )
    if len(common) < len(stamps):
        print(
            f"[warn] {len(stamps) - len(common)} of {len(stamps)} network snapshots are absent "
            f"from the demand table; comparing on the {len(common)} they share.",
        )
    return hourly.loc[common].mean()


def conservation_frame(allocated: pd.Series, table: pd.Series, laf_sum: pd.Series | None) -> pd.DataFrame:
    """One row per demand key: what the table holds, what the buses received."""
    keys = sorted(set(allocated.index.dropna()) | set(table.index.dropna()))
    df = pd.DataFrame(index=pd.Index(keys, name="demand_key"))
    df["table_mw"] = table.reindex(df.index).astype(float).fillna(0.0)
    df["allocated_mw"] = allocated.reindex(df.index).astype(float).fillna(0.0)
    df["delta_mw"] = df.allocated_mw - df.table_mw
    df["delta_pct"] = 100.0 * df.delta_mw / df.table_mw.where(df.table_mw != 0)
    if laf_sum is not None:
        df["laf_sum"] = laf_sum.reindex(df.index).astype(float)
    df["in_table"] = df.index.isin(table.index)
    df["has_bus"] = df.index.isin(allocated.index)
    return df


def plot(df: pd.DataFrame, path: Path, title: str) -> None:
    """Per-key bars (table vs allocated) over a signed error panel."""
    ordered = df.sort_values("table_mw", ascending=False)
    x = range(len(ordered))
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(max(10, 0.28 * len(ordered)), 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    width = 0.42
    top.bar([i - width / 2 for i in x], ordered.table_mw, width, label="demand table", color=MASTER_COLOR)
    top.bar([i + width / 2 for i in x], ordered.allocated_mw, width, label="allocated to buses", color=DEVELOP_COLOR)
    top.set_ylabel("mean demand [MW]")
    top.set_title(title)
    top.legend(frameon=False)
    top.grid(axis="y", alpha=0.25)

    colors = ["#c0392b" if value > 0 else "#2471a3" for value in ordered.delta_pct.fillna(0.0)]
    bottom.bar(list(x), ordered.delta_pct.fillna(0.0), 0.7, color=colors)
    bottom.axhline(0.0, color="black", linewidth=0.8)
    bottom.set_ylabel("allocated - table [%]")
    bottom.set_xticks(list(x))
    bottom.set_xticklabels(ordered.index, rotation=90, fontsize=7)
    bottom.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Run the checker over one network and its demand table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--network", required=True, type=Path, help="a network carrying loads (*_dem.nc or later)")
    parser.add_argument("--demand-table", required=True, type=Path, help="power_zonal_components_s{simpl}.parquet")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--key", default="auto", choices=("auto", "state", "reeds_state"))
    parser.add_argument("--label", default="", help="extra text for the figure title")
    args = parser.parse_args(argv)

    n = pypsa.Network(str(args.network))
    keys, key_column = bus_keys(n, args.key)
    allocated = allocated_by_key(n, keys)

    snapshots = n.snapshots
    if isinstance(snapshots, pd.MultiIndex):
        snapshots = snapshots.get_level_values(-1)
    table = table_by_key(args.demand_table, snapshots)

    laf_sum = None
    if "LAF_state" in n.buses.columns:
        laf_sum = pd.to_numeric(n.buses.LAF_state, errors="coerce").groupby(keys).sum()

    df = conservation_frame(allocated, table, laf_sum)

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "demand_conservation.csv"
    png_path = args.outdir / "demand_conservation.png"
    df.to_csv(csv_path)

    total_table = float(df.table_mw.sum())
    total_alloc = float(df.allocated_mw.sum())
    error_pct = 100.0 * (total_alloc - total_table) / total_table if total_table else float("nan")

    label = f" — {args.label}" if args.label else ""
    plot(df, png_path, f"Demand conservation, key '{key_column}'{label}: national error {error_pct:+.2f} %")

    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(df.round(3).to_string())
    print()
    print(f"network       : {args.network}")
    print(f"demand table  : {args.demand_table}")
    print(f"key column    : {key_column}")
    print(f"table total   : {total_table:,.1f} MW mean")
    print(f"allocated     : {total_alloc:,.1f} MW mean")
    print(f"national error: {total_alloc - total_table:+,.1f} MW ({error_pct:+.4f} %)")

    no_bus = df.index[~df.has_bus & df.in_table].tolist()
    no_demand = df.index[~df.in_table & df.has_bus].tolist()
    if no_bus:
        print(f"keys in the table with NO BUS (dropped): {no_bus} = {df.loc[no_bus, 'table_mw'].sum():,.1f} MW")
    if no_demand:
        print(f"keys on buses with NO DEMAND COLUMN    : {no_demand}")
    print(f"wrote {csv_path} and {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
