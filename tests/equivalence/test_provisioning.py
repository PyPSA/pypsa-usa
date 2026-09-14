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
    """A two-commit git repository with a ``master-benchmark`` branch."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "master")
    _git(repo, "config", "user.email", "harness@example.invalid")
    _git(repo, "config", "user.name", "harness")
    (repo / "a.txt").write_text("one\n")
    _git(repo, "add", "a.txt")
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
    monkeypatch.delenv("EQ_BASELINE_REF", raising=False)
    from tests.equivalence import paths

    assert paths.BASELINE_REF == "master-benchmark"
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


def test_registered_worktree_is_accepted(harness: Path, monkeypatch):
    """A correctly registered worktree is reused and repointed, not refused.

    The repo-specific seeding steps (symlinks, config copies) need a real
    pypsa-usa layout, so only the git half of provisioning is exercised here.
    """
    wt = build.BASELINE_WORKTREE
    sha = build.resolve_baseline_sha()
    first = _git(harness, "rev-parse", "HEAD~1")
    wt.parent.mkdir(parents=True, exist_ok=True)
    _git(harness, "worktree", "add", "--detach", str(wt), first)

    registered = build._registered_worktrees()
    assert os.path.realpath(wt) in registered
    assert registered[os.path.realpath(wt)] == first

    # Provisioning fails later (no workflow/ in the throwaway repo) but must
    # first have checked the worktree out at the resolved baseline sha.
    with pytest.raises((RuntimeError, FileNotFoundError, NotADirectoryError)):
        build.provision_baseline_worktree()
    assert _git(wt, "rev-parse", "HEAD") == sha
