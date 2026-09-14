"""Baseline-ref resolution and worktree provisioning, on a throwaway git repo.

Tier A (``fast``): every test runs against a ~10-line repository created in
``tmp_path`` (``git init``, two commits). No pypsa, no ``data/``, no network.

The behaviour under test is the F2 failure mode recorded in the harness plan:
``.worktrees/<name>`` can survive a move of the checkout, carrying a ``.git``
file whose ``gitdir:`` points at *another* repository. Provisioning must refuse
such a directory instead of adopting it and running git against the wrong repo.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.equivalence import build

pytestmark = pytest.mark.fast


def _git(repo: Path, *args: str) -> str:
    cp = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return cp.stdout.strip()


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """A two-commit git repository with a ``master-benchmark`` branch.

    It carries the minimum pypsa-usa shape provisioning touches — a
    ``workflow/repo_data/config/`` with one template in it, and a ``.gitignore``
    matching master's bare ``config/`` rule — so provisioning can run to
    completion here without any of the real workflow or data.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "master")
    _git(repo, "config", "user.email", "harness@example.invalid")
    _git(repo, "config", "user.name", "harness")
    (repo / ".gitignore").write_text("config/\ndata/\ncutouts\n")
    cfg = repo / "workflow" / "repo_data" / "config"
    cfg.mkdir(parents=True)
    (cfg / "config.cluster.yaml").write_text("cluster: {}\n")
    (repo / "a.txt").write_text("one\n")
    _git(repo, "add", "-f", ".gitignore", "a.txt", "workflow")
    _git(repo, "commit", "-qm", "first")
    (repo / "b.txt").write_text("two\n")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-qm", "second")
    _git(repo, "branch", "master-benchmark")
    return repo


@pytest.fixture
def harness(monkeypatch, tmp_repo: Path) -> Path:
    """Point build.REPO / build.BASELINE_WORKTREE at the throwaway repo."""
    monkeypatch.setattr(build, "REPO", tmp_repo)
    monkeypatch.setattr(build, "BASELINE_WORKTREE", tmp_repo / ".worktrees" / "master-benchmark")
    return tmp_repo


def test_resolve_baseline_sha(harness: Path):
    sha = build.resolve_baseline_sha()
    assert len(sha) == 40, sha
    assert sha == _git(harness, "rev-parse", "master-benchmark")


def test_resolve_baseline_sha_honours_env(monkeypatch, harness: Path):
    """EQ_BASELINE_REF is read at call time, not frozen at import."""
    monkeypatch.setenv("EQ_BASELINE_REF", "master")
    assert build.baseline_ref() == "master"
    assert build.resolve_baseline_sha() == _git(harness, "rev-parse", "master")


def test_baseline_ref_defaults_to_master_benchmark(monkeypatch):
    """With no env override the baseline ref is master-benchmark."""
    monkeypatch.delenv("EQ_BASELINE_REF", raising=False)
    monkeypatch.setattr(build, "BASELINE_REF", "master-benchmark")
    assert build.baseline_ref() == "master-benchmark"


def test_resolve_baseline_sha_missing_ref_names_t1(monkeypatch, harness: Path):
    monkeypatch.setenv("EQ_BASELINE_REF", "no-such-branch")
    with pytest.raises(RuntimeError) as exc:
        build.resolve_baseline_sha()
    msg = str(exc.value)
    assert "no-such-branch" in msg
    assert "T1" in msg and "master-benchmark" in msg


def test_dangling_worktree_raises(harness: Path):
    """A .git file pointing at a nonexistent gitdir is refused, not adopted."""
    wt = build.BASELINE_WORKTREE
    wt.mkdir(parents=True)
    dangling = "/nonexistent/other-checkout/.git/worktrees/master-benchmark"
    (wt / ".git").write_text(f"gitdir: {dangling}\n")

    with pytest.raises(RuntimeError) as exc:
        build.provision_baseline_worktree()
    msg = str(exc.value)
    assert dangling in msg
    assert str(wt) in msg
    # It must not have been turned into a real worktree behind our back.
    assert "master-benchmark" not in _git(harness, "worktree", "list", "--porcelain")


def test_unregistered_plain_directory_raises(harness: Path):
    """Even a directory with no .git at all is refused rather than built in."""
    wt = build.BASELINE_WORKTREE
    wt.mkdir(parents=True)
    (wt / "stray.txt").write_text("not a worktree\n")

    with pytest.raises(RuntimeError) as exc:
        build.provision_baseline_worktree()
    assert "does not list it as a worktree" in str(exc.value)


def test_registered_worktree_is_accepted(harness: Path):
    """A correctly registered worktree is reused and repointed, not refused."""
    wt = build.BASELINE_WORKTREE
    sha = build.resolve_baseline_sha()
    first = _git(harness, "rev-parse", "HEAD~1")
    wt.parent.mkdir(parents=True, exist_ok=True)
    _git(harness, "worktree", "add", "--detach", str(wt), first)

    registered = build._registered_worktrees()
    assert os.path.realpath(wt) in registered
    assert registered[os.path.realpath(wt)] == first

    assert build.provision_baseline_worktree() == wt
    assert _git(wt, "rev-parse", "HEAD") == sha
    # Seeding ran: the layered template landed in the gitignored config/ overlay.
    assert (wt / "workflow" / "config" / "config.cluster.yaml").exists()
    # And it left the checkout clean, so a build is allowed to start.
    assert build.checkout_dirt(wt) == []


def test_clean_checkout_passes(harness: Path):
    assert build.assert_clean_checkout("develop", harness) == []


def test_dirty_tracked_file_refused(harness: Path):
    """A modified TRACKED file blocks the build on either side."""
    (harness / "a.txt").write_text("edited\n")
    with pytest.raises(RuntimeError) as exc:
        build.assert_clean_checkout("develop", harness)
    msg = str(exc.value)
    assert "a.txt" in msg
    assert "EQ_ALLOW_DIRTY" in msg


def test_dirty_allowed_by_env(monkeypatch, harness: Path):
    """EQ_ALLOW_DIRTY=1 downgrades the refusal to a recorded warning."""
    (harness / "a.txt").write_text("edited\n")
    monkeypatch.setenv("EQ_ALLOW_DIRTY", "1")
    dirt = build.assert_clean_checkout("develop", harness)
    assert any("a.txt" in line for line in dirt)


def test_untracked_files_do_not_count_as_dirty(harness: Path):
    """The harness creates untracked files on both sides; they are not dirt."""
    (harness / "workflow" / "cutouts").mkdir(parents=True, exist_ok=True)
    (harness / "stray_artifact.txt").write_text("untracked\n")
    assert build.checkout_dirt(harness) == []
    assert build.assert_clean_checkout("master", harness) == []
