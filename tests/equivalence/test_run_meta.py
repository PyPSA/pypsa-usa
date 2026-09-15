"""``RunContext`` and ``run_meta.json``, on a throwaway git repository.

Tier A (``fast``): a ~10-line repo in ``tmp_path`` with ``master``,
``master-benchmark`` and a branch standing in for ``develop``. No pypsa, no
``data/``, no build, and ``probe_env=False`` so nothing shells out to ``uv``.

What has to hold, because ``run_meta.json`` is the only record of what produced
a number:

- three distinct 40-char shas, one per ref, resolved from the repo rather than
  pinned anywhere;
- the develop side's dirtiness is *recorded*, not refused — a dirty run is
  readable as long as it says it was dirty (plan D2);
- the run id is a directory name: no ``/``, no spaces.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests.equivalence import build, context, paths

pytestmark = pytest.mark.fast


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Build master -> master-benchmark (+1) and a develop-like branch (+1)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "master")
    _git(repo, "config", "user.email", "harness@example.invalid")
    _git(repo, "config", "user.name", "harness")
    (repo / "a.txt").write_text("one\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-qm", "master tip")

    _git(repo, "switch", "-q", "-c", "master-benchmark")
    (repo / "ported.txt").write_text("port(HF-8)\n")
    _git(repo, "add", "ported.txt")
    _git(repo, "commit", "-qm", "port(HF-8): something")

    _git(repo, "switch", "-q", "-c", "feat/harness", "master")
    (repo / "b.txt").write_text("two\n")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-qm", "develop tip")
    return repo


@pytest.fixture
def harness(monkeypatch, tmp_repo: Path) -> Path:
    """Point every module's REPO at the throwaway repo and mint a fresh id."""
    monkeypatch.setattr(build, "REPO", tmp_repo)
    monkeypatch.setattr(build, "BASELINE_WORKTREE", tmp_repo / ".worktrees" / "master-benchmark")
    monkeypatch.setattr(paths, "REPO", tmp_repo)
    monkeypatch.setattr(context, "REPO", tmp_repo)
    # setenv-then-delenv so monkeypatch records the pre-existing value and
    # undoes it at teardown: paths.run_id() FREEZES the id into the environment,
    # and a leaked EQ_RUN_ID would redirect every later test's run directory.
    monkeypatch.setenv("EQ_RUN_ID", "placeholder")
    monkeypatch.delenv("EQ_RUN_ID")
    monkeypatch.setenv("EQ_BASELINE_REF", "placeholder")
    monkeypatch.delenv("EQ_BASELINE_REF")
    monkeypatch.setattr(paths, "BASELINE_REF", "master-benchmark")
    return tmp_repo


def test_run_id_is_filesystem_safe(harness, monkeypatch):
    monkeypatch.setenv("EQ_RUN_ID", "usa/p2 REM-3h")
    rid = paths.run_id(2)
    assert "/" not in rid and " " not in rid
    assert rid == "usa-p2-REM-3h"


def test_run_id_is_frozen_after_the_first_call(harness, monkeypatch):
    """A later call that does not know the prong resolves the same directory."""
    first = paths.run_id(2)
    assert "-p2-" in first
    assert paths.run_id() == first
    assert paths.run_dir() == paths.run_dir(2)


def test_run_dir_is_under_the_develop_checkout(harness):
    rd = paths.run_dir(1)
    assert rd.parent == harness / "workflow" / "results" / "equivalence"


def test_context_carries_three_distinct_shas(harness):
    ctx = context.build_context(2, probe_env=False)
    for name in ("master_sha", "baseline_sha", "develop_sha"):
        sha = getattr(ctx, name)
        assert len(sha) == 40, f"{name}={sha!r}"
    assert ctx.master_sha == _git(harness, "rev-parse", "master")
    assert ctx.baseline_sha == _git(harness, "rev-parse", "master-benchmark")
    assert ctx.develop_sha == _git(harness, "rev-parse", "HEAD")
    assert len({ctx.master_sha, ctx.baseline_sha, ctx.develop_sha}) == 3
    assert ctx.baseline_commits_on_top_of_master == 1
    assert ctx.develop_commits_ahead_of_master == 1
    assert ctx.baseline_ref == "master-benchmark"
    assert ctx.develop_ref == "feat/harness"


def test_context_records_the_translated_clusters(harness):
    """Both dialects are recorded, because the two sides ran different literals."""
    ctx = context.build_context(2, probe_env=False)
    assert ctx.clusters_develop == paths.CLUSTERS
    assert ctx.clusters_baseline == paths.baseline_clusters()


def test_dirty_develop_is_recorded_not_refused(harness):
    (harness / "b.txt").write_text("edited\n")
    ctx = context.build_context(2, probe_env=False)
    assert ctx.develop_dirty is True


def test_clean_develop_is_recorded_clean(harness):
    ctx = context.build_context(2, probe_env=False)
    assert ctx.develop_dirty is False


def test_run_id_slug_is_written_back_to_the_environment(harness, monkeypatch):
    """The sbatch driver builds the run-dir path from $EQ_RUN_ID itself.

    If slugging changed anything and we kept the raw value in the environment,
    bash and python would disagree about where the run lives.
    """
    monkeypatch.setenv("EQ_RUN_ID", "usa/p2 REM-3h")
    import os

    assert paths.run_id(2) == "usa-p2-REM-3h"
    assert os.environ["EQ_RUN_ID"] == "usa-p2-REM-3h"


@pytest.mark.parametrize(
    "raw",
    ["../escape", "..", "a/../../b", "..\\windows", "....", "./."],
)
def test_run_id_cannot_climb_out_of_the_results_directory(harness, monkeypatch, raw):
    monkeypatch.setenv("EQ_RUN_ID", raw)
    rid = paths.run_id(2)
    assert "/" not in rid and "\\" not in rid
    assert ".." not in rid
    assert paths.run_dir(2).parent == harness / "workflow" / "results" / "equivalence"


def test_gate_status_starts_pending(harness):
    assert context.build_context(2, probe_env=False).config_gate == "pending"


def test_run_gate_leaves_provenance_when_the_gate_fails(harness, monkeypatch):
    """A refused run still says which three shas it was about to compare."""
    ctx = context.build_context(2, probe_env=False)

    def _boom(_ctx=None):
        raise RuntimeError("pudl_path differs")

    monkeypatch.setattr(context, "assert_config_equivalent", _boom)
    with pytest.raises(RuntimeError):
        context.run_gate(ctx)

    meta = json.loads((ctx.run_dir / "run_meta.json").read_text())
    assert meta["config_gate"] == "failed"
    assert "pudl_path differs" in meta["config_gate_error"]
    assert len(meta["baseline_sha"]) == 40
    assert len(meta["develop_sha"]) == 40


def test_run_gate_records_pass(harness, monkeypatch):
    ctx = context.build_context(2, probe_env=False)
    monkeypatch.setattr(context, "assert_config_equivalent", lambda _ctx=None: [{"key": "k", "reason": "r"}])
    ctx = context.run_gate(ctx)
    meta = json.loads((ctx.run_dir / "run_meta.json").read_text())
    assert meta["config_gate"] == "passed"
    assert meta["config_diff_allowed"][0]["key"] == "k"


def test_run_meta_round_trips(harness):
    ctx = context.build_context(1, probe_env=False)
    ctx = context.with_config_diff(ctx, [{"key": "scenario.clusters", "reason": "translated"}])
    path = context.write_run_meta(ctx)

    assert path == ctx.run_dir / "run_meta.json"
    meta = json.loads(path.read_text())
    assert meta["run_id"] == ctx.run_id
    assert meta["prong"] == 1
    for name in ("master_sha", "baseline_sha", "develop_sha"):
        assert len(meta[name]) == 40
    assert meta["config_diff_allowed"][0]["key"] == "scenario.clusters"
    assert meta["run_dir"] == str(ctx.run_dir)
    # known_code_differences comes from the hot-fix registry and must never
    # contain a hot-fix that is ported onto the baseline.
    assert meta["known_code_differences"], "no candidate explanations recorded"
    assert meta["known_code_differences"][0]["key"].startswith("{clusters}")


def test_slurm_job_id_is_captured(harness, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "43441927")
    assert context.build_context(2, probe_env=False).slurm_job_id == "43441927"


def test_missing_baseline_branch_points_at_t1(harness, monkeypatch):
    monkeypatch.setenv("EQ_BASELINE_REF", "no-such-branch")
    with pytest.raises(RuntimeError) as exc:
        context.build_context(2, probe_env=False)
    assert "T1" in str(exc.value)


def test_master_profile_stage_is_recorded_per_prong(harness):
    """A CF quantile is meaningless without the bus population it was taken over.

    Prong 2 compares master's SUBSTATION-resolution profile file against
    develop's s{simpl} file, so the harness rolls master up first; run_meta.json
    has to say so, or a reader cannot tell which object produced the number.
    ``build_context`` runs before either side builds, so all it can honestly
    record is that the rollup is PLANNED; the comparison rewrites it.
    """
    assert context.build_context(1, probe_env=False).master_profile_stage == "nodal"
    meta = json.loads(
        context.write_run_meta(context.build_context(2, probe_env=False)).read_text(),
    )
    assert meta["master_profile_stage"] == f"nodal; rollup to s{paths.SIMPL2} planned, not yet run"


def test_master_profile_stage_reports_what_actually_happened():
    """The field is derived from the observed rollup, never from the prong alone.

    A prong-2 run whose ``busmap_s{simpl}.csv`` was missing compared two
    different bus resolutions. Saying 'nodal->s20' because the prong is 2 would
    describe a rollup that never ran.
    """
    assert context.master_profile_stage(1) == "nodal"
    assert context.master_profile_stage(1, rolled_up=False) == "nodal"
    assert context.master_profile_stage(2, rolled_up=True) == f"nodal->s{paths.SIMPL2} (p_nom_max-weighted)"
    not_run = context.master_profile_stage(2, rolled_up=False)
    assert "did NOT run" in not_run
    assert "planned" in context.master_profile_stage(2)


def test_update_run_meta_merges_into_the_existing_record(harness, tmp_path):
    """Post-comparison provenance lands in the SAME file as the shas."""
    ctx = context.build_context(2, probe_env=False)
    context.write_run_meta(ctx)

    path = context.update_run_meta(
        run_dir=ctx.run_dir,
        master_profile_stage="nodal->s20 (p_nom_max-weighted)",
        profile_cluster_sets=[{"stage": "profile_onwind", "only_develop": {"p87 0": 96.0}}],
    )
    meta = json.loads(path.read_text())

    assert path == ctx.run_dir / "run_meta.json"
    assert meta["master_profile_stage"] == "nodal->s20 (p_nom_max-weighted)"
    assert meta["profile_cluster_sets"][0]["only_develop"] == {"p87 0": 96.0}
    # Everything already in the record survives the merge.
    assert meta["develop_sha"] == ctx.develop_sha
    assert meta["run_id"] == ctx.run_id


def test_update_run_meta_creates_the_file_when_there_is_none(tmp_path):
    path = context.update_run_meta(run_dir=tmp_path / "fresh", profile_cluster_sets=[])
    assert json.loads(path.read_text()) == {"profile_cluster_sets": []}


def test_run_context_declares_the_cluster_set_field(harness):
    ctx = context.build_context(2, probe_env=False)
    assert ctx.profile_cluster_sets == []
    meta = json.loads(context.write_run_meta(ctx).read_text())
    assert meta["profile_cluster_sets"] == []
