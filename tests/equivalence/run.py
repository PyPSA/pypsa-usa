"""CLI orchestrator: build both sides, compare, report.

uv run python -m tests.equivalence.run --prong 1 [--skip-solve] [--side both]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tests.equivalence.build import ANCHOR_WORKTREE, build_side  # noqa: E402
from tests.equivalence.compare import run_comparison  # noqa: E402
from tests.equivalence.paths import (  # noqa: E402
    UNTIL,
    anchor_assembled_target,
    anchor_final_target,
    assembled_target,
    final_target,
)
from tests.equivalence.plots import export_all  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prong", type=int, choices=(1, 2), required=True)
    ap.add_argument("--skip-solve", action="store_true")
    ap.add_argument(
        "--side",
        choices=("candidate", "anchor", "both", "none"),
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
    args = ap.parse_args()

    solve = not args.skip_solve
    # EQ_UNTIL=assembled already stops the compared pairs at the assembled
    # stage (paths.prong_pairs); build the matching targets too, or the run
    # would still drive the whole chain through the solve it is not comparing.
    if UNTIL == "assembled":
        cand_target, anch_target = assembled_target(args.prong), anchor_assembled_target(args.prong)
    else:
        cand_target, anch_target = final_target(args.prong, solve), anchor_final_target(args.prong, solve)
    if args.side in ("candidate", "both"):
        build_side("candidate", cand_target, args.jobs, timeout=args.timeout)
    if args.side in ("anchor", "both"):
        build_side("anchor", anch_target, args.jobs, timeout=args.timeout)

    result = run_comparison(
        args.prong,
        REPO / "workflow",
        ANCHOR_WORKTREE / "workflow",
    )
    # PNG plots instead of the HTML report (user decision 2026-09-01: the
    # HTML wrapper added nothing over the figures themselves).
    plots_dir = export_all()
    print(
        f"[equivalence] prong {args.prong}: "
        f"{'PASS' if result['pass'] else 'FAIL'} "
        f"({result['n_live']} live / {result['n_findings']} total findings)",
    )
    print(f"[equivalence] plots: {plots_dir}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
