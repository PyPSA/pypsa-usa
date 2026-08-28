"""Pytest wrapper for the Tier C equivalence harness.

Marked ``equivalence`` — excluded from fast/integration runs. Assumes both
sides are already built (run ``python -m tests.equivalence.run`` first);
skips when artifacts are absent so CI without builds stays green.
"""

from pathlib import Path

import pytest

from tests.equivalence.build import ANCHOR_WORKTREE
from tests.equivalence.compare import run_comparison
from tests.equivalence.paths import anchor_final_target, final_target

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.equivalence
@pytest.mark.parametrize("prong", [1, 2])
def test_equivalence(prong):
    cand = REPO / "workflow" / final_target(prong)
    anch = ANCHOR_WORKTREE / "workflow" / anchor_final_target(prong)
    if not cand.exists() or not anch.exists():
        pytest.skip(
            f"prong {prong} artifacts not built (candidate={cand.exists()}, "
            f"anchor={anch.exists()}) — run python -m tests.equivalence.run",
        )
    result = run_comparison(prong, REPO / "workflow", ANCHOR_WORKTREE / "workflow")
    live = [f for f in result["findings"] if not f["waived"]]
    assert not live, f"{len(live)} unwaived findings; see results/equivalence/findings_{prong}.json"
