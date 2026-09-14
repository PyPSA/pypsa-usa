"""CLI orchestrator: build both sides, compare, tabulate, report.

uv run python -m tests.equivalence.run --prong 1 [--skip-solve] [--side both]

Sides are ``master`` (the ``master-benchmark`` baseline, built in
``.worktrees/master-benchmark``) and ``develop`` (the main checkout).

Outputs land in one run directory, ``workflow/results/equivalence/<run_id>/``:
``tables/comparison.csv`` and ``tables/comparison.md`` (the verdict table) and
``figures/`` (one PNG per figure, each beside the CSV it was drawn from).
``EQ_RUN_ID`` names the directory; T4's ``context.build_context`` will supply it
from the ``RunContext`` once that lands.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tests.equivalence import plots, tables  # noqa: E402
from tests.equivalence.build import BASELINE_WORKTREE, build_side  # noqa: E402
from tests.equivalence.compare import run_comparison  # noqa: E402
from tests.equivalence.paths import (  # noqa: E402
    INTERCONNECT,
    UNTIL,
    assembled_target,
    baseline_assembled_target,
    baseline_final_target,
    final_target,
)

HOTFIXES_PATH = Path(__file__).parent / "hotfixes.yaml"


def run_dir_for(prong: int) -> Path:
    """The run directory for this invocation (``EQ_RUN_ID`` wins)."""
    run_id = os.environ.get("EQ_RUN_ID") or f"{INTERCONNECT}-p{prong}"
    return REPO / "workflow" / "results" / "equivalence" / run_id


def emit_results(prong: int, run_dir: Path, findings: list[dict]) -> dict[str, int]:
    """Metrics -> comparison table -> figures. Returns the verdict counts.

    A metric that raises lands in ``missing`` and becomes a ``MISSING`` row, so
    a criterion cannot drop out of the comparison and still exit 0.
    """
    art = plots.load_artifacts(prong, REPO / "workflow", BASELINE_WORKTREE / "workflow")
    missing: list[dict] = []
    metric_frames = plots.collect_metrics(art, missing)
    hotfixes = tables.load_hotfixes(HOTFIXES_PATH)
    from tests.equivalence.compare import load_waivers

    comparison = tables.comparison_table(metric_frames, hotfixes, load_waivers(), missing)
    written = tables.write_tables({**metric_frames, "comparison": comparison}, run_dir)
    print(f"[equivalence] wrote {len(written)} table(s) under {run_dir / 'tables'}")
    figdir = plots.export_all(
        run_dir, artifacts=art, metric_frames=metric_frames, findings=findings, missing=missing,
    )
    print(f"[equivalence] figures: {figdir}")
    return tables.verdict_counts(comparison)


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
        help="build the metrics, the comparison table and the figures (default: on)",
    )
    args = ap.parse_args()

    solve = not args.skip_solve
    # EQ_UNTIL=assembled already stops the compared pairs at the assembled
    # stage (paths.prong_pairs); build the matching targets too, or the run
    # would still drive the whole chain through the solve it is not comparing.
    if UNTIL == "assembled":
        dev_target, mas_target = assembled_target(args.prong), baseline_assembled_target(args.prong)
    else:
        dev_target, mas_target = final_target(args.prong, solve), baseline_final_target(args.prong, solve)
    if args.side in ("develop", "both"):
        build_side("develop", dev_target, args.jobs, timeout=args.timeout)
    if args.side in ("master", "both"):
        build_side("master", mas_target, args.jobs, timeout=args.timeout)

    result = run_comparison(
        args.prong,
        REPO / "workflow",
        BASELINE_WORKTREE / "workflow",
    )
    run_dir = run_dir_for(args.prong)
    counts = dict.fromkeys(tables.VERDICT_ORDER, 0)
    if args.tables:
        counts = emit_results(args.prong, run_dir, result["findings"])
    print(
        f"[equivalence] prong {args.prong}: "
        f"{'PASS' if result['pass'] else 'FAIL'} "
        f"({result['n_live']} live / {result['n_findings']} total findings)",
    )
    print(
        f"[equivalence] verdicts: {counts['equivalent']} equivalent, "
        f"{counts['explained']} explained, {counts['one-sided']} one-sided, "
        f"{counts['undefined']} undefined, {counts['UNEXPLAINED']} UNEXPLAINED, "
        f"{counts['MISSING']} MISSING",
    )
    print(f"[equivalence] comparison: {run_dir / 'tables' / 'comparison.md'}")
    failing = sum(counts.get(v, 0) for v in tables.FAILING_VERDICTS)
    return 0 if (result["pass"] and failing == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
