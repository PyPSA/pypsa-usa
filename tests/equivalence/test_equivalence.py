"""Pytest wrapper for the Tier C equivalence harness.

Marked ``equivalence`` — excluded from fast/integration runs. Assumes both
sides are already built (run ``python -m tests.equivalence.run`` first);
skips when artifacts are absent so CI without builds stays green.

Findings land in this run's directory (``paths.run_dir()``, plan D5), so set
``EQ_RUN_ID`` to the run you are re-checking or a fresh directory is minted.
"""

from pathlib import Path

import pytest

from tests.equivalence.build import BASELINE_WORKTREE
from tests.equivalence.compare import run_comparison
from tests.equivalence.paths import baseline_final_target, final_target, run_dir

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.equivalence
@pytest.mark.parametrize("prong", [1, 2])
def test_equivalence(prong):
    dev = REPO / "workflow" / final_target(prong)
    mas = BASELINE_WORKTREE / "workflow" / baseline_final_target(prong)
    if not dev.exists() or not mas.exists():
        pytest.skip(
            f"prong {prong} artifacts not built (develop={dev.exists()}, "
            f"master={mas.exists()}) — run python -m tests.equivalence.run",
        )
    result = run_comparison(prong, REPO / "workflow", BASELINE_WORKTREE / "workflow")
    live = [f for f in result["findings"] if not f["waived"]]
    assert not live, f"{len(live)} unwaived findings; see {run_dir(prong) / f'findings_{prong}.json'}"
