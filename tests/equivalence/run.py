"""CLI orchestrator: resolve the run, gate the config, build both sides, compare.

    uv run python -m tests.equivalence.run --prong 2 [--skip-solve] [--side both]

Sides are ``master`` (the ``master-benchmark`` baseline, built in
``.worktrees/master-benchmark``) and ``develop`` (the main checkout).

Order matters and is the point of this module:

1. :func:`context.build_context` resolves the three shas, both environments and
   the translated wildcards, and mints the run directory.
2. :func:`context.assert_config_equivalent` refuses to go further if the two
   sides would not be on the same config — *before* either build, so a long run
   is never spent comparing two different configurations.
3. ``run_meta.json`` is written before the builds, so a run that dies halfway
   still says what it was.
4. Both sides build, sequentially and never concurrently: they share one
   ``data/`` cache, and two concurrent DAGs would race on the same retrieve
   targets.
5. Compare, then tabulate and plot.

Exit 0 only when there are no unwaived findings and no ``UNEXPLAINED`` rows in
the comparison table.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tests.equivalence import context  # noqa: E402
from tests.equivalence.build import BASELINE_WORKTREE, build_side  # noqa: E402
from tests.equivalence.compare import run_comparison  # noqa: E402
from tests.equivalence.paths import (  # noqa: E402
    UNTIL,
    assembled_target,
    baseline_assembled_target,
    baseline_final_target,
    final_target,
)
from tests.equivalence.plots import export_all  # noqa: E402


def _export_figures(ctx: context.RunContext) -> Path:
    """Call ``plots.export_all`` through whichever signature it currently has.

    T3 of the harness plan gives it ``(run_dir, ctx)``; before that lands it is
    zero-arg and writes to its own module-level ``OUTDIR``. Bridging here keeps
    the run directory authoritative without editing a file T3 owns.
    """
    if len(inspect.signature(export_all).parameters) >= 2:
        return export_all(ctx.run_dir, ctx)
    return export_all()


def _export_tables(ctx: context.RunContext, result: dict) -> None:
    """Hand the run off to T3's table layer, if it is present.

    The hook is ``tables.export_all(run_dir, ctx, findings)``; until T3 lands
    there is no ``tables`` module and the run completes on the findings alone.
    Nothing here interprets the table — the verdict counts are read back from
    the artifact it writes, so this module and that one agree by construction.
    """
    try:
        from tests.equivalence import tables
    except ImportError:
        print("[equivalence] tables: module not present; skipping the comparison table")
        return
    export = getattr(tables, "export_all", None)
    if export is None:
        print("[equivalence] tables: no export_all hook; skipping the comparison table")
        return
    export(ctx.run_dir, ctx, result)


def _verdict_counts(run_dir: Path) -> dict[str, int]:
    """``{verdict: count}`` read back from ``tables/comparison.csv``.

    Reads the artifact rather than the code that made it, so an empty dict means
    "no table was written", not "the table said nothing".
    """
    path = run_dir / "tables" / "comparison.csv"
    if not path.exists():
        return {}
    import csv

    counts: dict[str, int] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            verdict = row.get("verdict", "")
            counts[verdict] = counts.get(verdict, 0) + 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prong", type=int, choices=(1, 2), required=True)
    ap.add_argument("--skip-solve", action="store_true")
    ap.add_argument(
        "--side",
        choices=("develop", "master", "both", "none"),
        default="both",
        help="which builds to run before comparing",
    )
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument(
        "--timeout",
        type=int,
        default=10800,
        help="per-side snakemake wall-clock cap in seconds (USA-scale builds need far more than the 3h default)",
    )
    ap.add_argument(
        "--tables",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write the comparison table (tables/comparison.{csv,md})",
    )
    args = ap.parse_args()

    ctx = context.build_context(args.prong)
    allowed = context.assert_config_equivalent(ctx)
    ctx = context.with_config_diff(ctx, allowed)
    meta_path = context.write_run_meta(ctx)
    print(f"[equivalence] run {ctx.run_id}: {meta_path}")

    solve = not args.skip_solve
    # EQ_UNTIL=assembled already stops the compared pairs at the assembled
    # stage (paths.prong_pairs); build the matching targets too, or the run
    # would still drive the whole chain through the solve it is not comparing.
    if UNTIL == "assembled":
        dev_target, mas_target = assembled_target(args.prong), baseline_assembled_target(args.prong)
    else:
        dev_target, mas_target = final_target(args.prong, solve), baseline_final_target(args.prong, solve)
    # Sequential, never concurrent: one shared data/ cache.
    if args.side in ("develop", "both"):
        build_side("develop", dev_target, args.jobs, timeout=args.timeout)
    if args.side in ("master", "both"):
        build_side("master", mas_target, args.jobs, timeout=args.timeout)

    result = run_comparison(
        args.prong,
        REPO / "workflow",
        BASELINE_WORKTREE / "workflow",
    )
    if args.tables:
        _export_tables(ctx, result)
    verdicts = _verdict_counts(ctx.run_dir)
    comparison_md = ctx.run_dir / "tables" / "comparison.md"
    # PNG figures instead of the HTML report (user decision 2026-09-01: the
    # HTML wrapper added nothing over the figures themselves).
    figures_dir = _export_figures(ctx)

    unexplained = verdicts.get("UNEXPLAINED", 0)
    ok = result["pass"] and unexplained == 0
    print(f"[equivalence] run id         : {ctx.run_id}  (prong {ctx.prong}, {ctx.interconnect})")
    print(f"[equivalence] master         : {ctx.master_sha[:12]}")
    print(f"[equivalence] master-benchmark: {ctx.baseline_sha[:12]} (+{ctx.baseline_commits_on_top_of_master})")
    print(
        f"[equivalence] develop        : {ctx.develop_sha[:12]} (+{ctx.develop_commits_ahead_of_master})"
        + ("  DIRTY" if ctx.develop_dirty else ""),
    )
    print(
        f"[equivalence] verdicts       : {verdicts.get('equivalent', 0)} equivalent / "
        f"{verdicts.get('explained', 0)} explained / {unexplained} UNEXPLAINED; "
        f"{result['n_live']} live of {result['n_findings']} findings",
    )
    print(f"[equivalence] table          : {comparison_md if comparison_md.exists() else '(not written)'}")
    print(f"[equivalence] figures        : {figures_dir}")
    print(f"[equivalence] prong {args.prong}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
