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
3. ``run_meta.json`` is written BEFORE the gate and rewritten after it, so
   even a run the gate refuses leaves the three shas and the reason behind.
4. Both sides build, sequentially and never concurrently: they share one
   ``data/`` cache, and two concurrent DAGs would race on the same retrieve
   targets.
5. Compare, then tabulate and plot.

Exit 0 only when there are no unwaived findings and no ``UNEXPLAINED`` or
``MISSING`` rows in the comparison table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tests.equivalence import context, plots, tables  # noqa: E402
from tests.equivalence.build import BASELINE_WORKTREE, build_side  # noqa: E402
from tests.equivalence.compare import run_comparison  # noqa: E402
from tests.equivalence.paths import (  # noqa: E402
    UNTIL,
    assembled_target,
    baseline_assembled_target,
    baseline_final_target,
    final_target,
)


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
    print(f"[equivalence] run {ctx.run_id}: {ctx.run_dir / 'run_meta.json'}")
    # run_gate writes run_meta.json before the gate and rewrites it after, so a
    # gate failure still leaves the provenance of the run it refused.
    ctx = context.run_gate(ctx)

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
    verdicts = dict.fromkeys(tables.VERDICT_ORDER, 0)
    if args.tables:
        verdicts = tables.verdict_counts(tables.export_all(ctx.run_dir, ctx, result))
    comparison_md = ctx.run_dir / "tables" / "comparison.md"
    # PNG figures instead of the HTML report (user decision 2026-09-01: the
    # HTML wrapper added nothing over the figures themselves).
    figures_dir = plots.export_all(ctx.run_dir, ctx)

    # UNEXPLAINED is a difference nobody can account for; MISSING is a
    # criterion that failed to compute at all. Both must fail the run — a
    # vanished criterion is worse than a difference, not better.
    failing = sum(verdicts.get(v, 0) for v in tables.FAILING_VERDICTS)
    ok = result["pass"] and failing == 0
    print(f"[equivalence] run id         : {ctx.run_id}  (prong {ctx.prong}, {ctx.interconnect})")
    print(f"[equivalence] master         : {ctx.master_sha[:12]}")
    print(f"[equivalence] master-benchmark: {ctx.baseline_sha[:12]} (+{ctx.baseline_commits_on_top_of_master})")
    print(
        f"[equivalence] develop        : {ctx.develop_sha[:12]} (+{ctx.develop_commits_ahead_of_master})"
        + ("  DIRTY" if ctx.develop_dirty else ""),
    )
    print(
        f"[equivalence] verdicts       : {verdicts.get('equivalent', 0)} equivalent / "
        f"{verdicts.get('explained', 0)} explained / {verdicts.get('one-sided', 0)} one-sided / "
        f"{verdicts.get('undefined', 0)} undefined / {verdicts.get('UNEXPLAINED', 0)} UNEXPLAINED / "
        f"{verdicts.get('MISSING', 0)} MISSING; "
        f"{result['n_live']} live of {result['n_findings']} findings",
    )
    print(f"[equivalence] table          : {comparison_md if comparison_md.exists() else '(not written)'}")
    print(f"[equivalence] figures        : {figures_dir}")
    print(f"[equivalence] prong {args.prong}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
