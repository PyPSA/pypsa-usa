"""Grep-guards for the retired 'anchor' nomenclature and the deleted patch machinery.

Tier A (``fast``): pure text inspection of files already in the repo. No data,
no network, no imports of the heavy scientific stack.

The baseline of this harness is the ``master-benchmark`` branch off ``master``
(project brain ``memory/plans/harness-master-vs-develop.md``, T1/T2), not a
pinned upstream commit patched at build time. These tests keep the vocabulary
and the deletion from creeping back.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "tests" / "equivalence"

# Unrelated uses of the word that must NOT be flagged: matplotlib legend
# placement, SVG text alignment and the myst docs extension. They are stripped
# from a line before the 'anchor' search runs.
_ALLOWED = re.compile(r"bbox_to_anchor|bbox_anchor|text-anchor|myst_heading_anchors")

_DELETED_SYMBOLS = (
    "ANCHOR_SHA",
    "ANCHOR_WORKTREE",
    "INFRA_PATCH_MARK",
    "ADOPTED_FIX_MARK",
    "POWERPLANTS_FIX_MARK",
    "POWERPLANTS_SCRIPT",
    "SEAM_FIX_MARK",
    "ADD_ELECTRICITY_SCRIPT",
    "LEAP_FIX_MARK",
    "BUILD_PROFILES_SCRIPT",
    "FORCE_RERUN_MARKER",
    "apply_infra_patches",
    "apply_adopted_fix_patches",
    "apply_powerplants_adoption",
    "apply_seam_adoption",
    "apply_leap_day_adoption",
    "mark_force_rerun",
    "snakemake_attrs",
)
_PATCH_BANNER = re.compile(r"EQUIVALENCE-HARNESS .* PATCH")


# This module is the only file allowed to say 'anchor': a guard has to be able
# to name what it forbids.
_SELF = Path(__file__).name


def _scanned_files() -> list[Path]:
    files = sorted(
        p
        for ext in ("py", "yaml", "sbatch")
        for p in HARNESS.glob(f"*.{ext}")
        if p.name != _SELF
    )
    files += sorted((REPO / "workflow" / "repo_data" / "config").glob("config.equivalence*.yaml"))
    files.append(REPO / "CONTEXT.md")
    return [p for p in files if p.exists()]


def _pytest_ini_block() -> str:
    """The ``[tool.pytest.ini_options]`` table of pyproject.toml, as text."""
    text = (REPO / "pyproject.toml").read_text()
    start = text.index("[tool.pytest.ini_options]")
    rest = text[start + len("[tool.pytest.ini_options]") :]
    nxt = re.search(r"^\[", rest, flags=re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


@pytest.mark.fast
def test_no_anchor_nomenclature():
    """'anchor' names nothing in the harness any more — the baseline is a branch."""
    hits: list[str] = []
    for path in _scanned_files():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if "anchor" in _ALLOWED.sub("", line).lower():
                hits.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    block = _pytest_ini_block()
    for lineno, line in enumerate(block.splitlines(), start=1):
        if "anchor" in _ALLOWED.sub("", line).lower():
            hits.append(f"pyproject.toml [tool.pytest.ini_options]+{lineno}: {line.strip()}")
    assert not hits, (
        "retired 'anchor' nomenclature found; the baseline is master-benchmark:\n"
        + "\n".join(hits)
    )


@pytest.mark.fast
def test_no_patch_machinery():
    """build.py provisions a worktree; it never rewrites the baseline's source."""
    text = (HARNESS / "build.py").read_text()
    present = [s for s in _DELETED_SYMBOLS if s in text]
    assert not present, (
        "build.py still carries deleted build-time patch machinery: "
        f"{present}. Those changes belong on master-benchmark as commits (T1)."
    )
    banner = _PATCH_BANNER.search(text)
    assert banner is None, f"build.py still writes a patch banner: {banner.group(0)!r}"


@pytest.mark.fast
@pytest.mark.parametrize(
    ("develop_wc", "expected"),
    [("4", "4m"), ("4s", "4"), ("134", "134m"), ("all", "all")],
)
def test_baseline_clusters_translation(develop_wc, expected):
    """Develop's {clusters} dialect maps onto master's; see paths.baseline_clusters."""
    from tests.equivalence.paths import baseline_clusters

    assert baseline_clusters(develop_wc) == expected


@pytest.mark.fast
@pytest.mark.parametrize("prong", [1, 2])
def test_artifact_pair_fields(prong):
    """Pairs are (develop, master) paths — no 'anchor' anywhere, and they differ."""
    from tests.equivalence.paths import prong_pairs

    pairs = prong_pairs(prong)
    assert pairs, f"prong {prong} produced no artifact pairs"
    for pair in pairs:
        assert pair.develop, f"{pair.stage}: empty develop path"
        assert pair.master, f"{pair.stage}: empty master path"
        assert pair.develop != pair.master, f"{pair.stage}: both sides share a path"
        assert "anchor" not in pair.develop.lower()
        assert "anchor" not in pair.master.lower()
