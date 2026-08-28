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
from tests.equivalence.paths import anchor_final_target, final_target  # noqa: E402
from tests.equivalence.report import build_report  # noqa: E402


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
    args = ap.parse_args()

    solve = not args.skip_solve
    if args.side in ("candidate", "both"):
        build_side("candidate", final_target(args.prong, solve), args.jobs)
    if args.side in ("anchor", "both"):
        build_side("anchor", anchor_final_target(args.prong, solve), args.jobs)

    result = run_comparison(
        args.prong,
        REPO / "workflow",
        ANCHOR_WORKTREE / "workflow",
    )
    report = build_report()
    print(
        f"[equivalence] prong {args.prong}: "
        f"{'PASS' if result['pass'] else 'FAIL'} "
        f"({result['n_live']} live / {result['n_findings']} total findings)",
    )
    print(f"[equivalence] report: {report}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
