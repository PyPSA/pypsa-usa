"""Tier A — the Sherlock controller, profile and helper scripts.

These are the only files in the repository that are *not* exercised by any
other test: a broken `--cluster` format string or a missing `-o` directory is
discovered as a Slurm job that never starts, hours into a campaign. Everything
here is static (``bash -n``, YAML parsing, a stubbed ``sacct``) except one
opt-in dry run, so the whole module runs in seconds without a scheduler.
"""

from __future__ import annotations

import functools
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / "workflow"
PROFILE_DIR = WORKFLOW / "snakemake_profiles" / "sherlock"
PROFILE = PROFILE_DIR / "config.yaml"
STATUS_HOOK = PROFILE_DIR / "slurm-status.sh"
OPTIONS_FIXTURE = Path(__file__).parent / "fixtures" / "snakemake-7.32.4-options.txt"

# The HPC scripts. `bash -n` and the no-absolute-paths rule apply to all of
# them; run_equivalence.sbatch is included because the executor switch was
# wired into it (it is submitted with `sbatch`, never executed directly, so it
# is not required to carry the executable bit).
OWNED_SCRIPTS = [
    WORKFLOW / "run_slurm.sbatch",
    WORKFLOW / "run_slurm_all.sh",
    WORKFLOW / "run_report.sbatch",
    WORKFLOW / "run_slurm.sh",
    WORKFLOW / "collect_solve_statistics.sh",
    STATUS_HOOK,
]
SHELL_SCRIPTS = [*OWNED_SCRIPTS, REPO / "tests" / "equivalence" / "run_equivalence.sbatch"]

# A site path baked into a tracked script is the failure mode that killed both
# CH2's controller (an absolute `cd` into one checkout) and this repo's
# run_usa.sbatch (a REPO= that had stopped existing). Environment variables and
# $SLURM_SUBMIT_DIR are the supported way to say the same thing.
ABSOLUTE_PATH_RE = r"(^|[^A-Za-z_$])/(oak|scratch|home)/"


@pytest.mark.fast
@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_script_exists(script):
    assert script.is_file(), f"{script} is missing"


@pytest.mark.fast
@pytest.mark.parametrize("script", OWNED_SCRIPTS, ids=lambda p: p.name)
def test_script_is_executable(script):
    """slurm-status.sh in particular: snakemake execs it directly, and a
    non-executable hook makes every status check fail.
    """
    assert script.stat().st_mode & stat.S_IXUSR, f"{script} is not executable"


@pytest.mark.fast
@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_script_parses(script):
    """Bash -n: syntax only, nothing is executed."""
    cp = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert cp.returncode == 0, f"{script.name} failed bash -n:\n{cp.stderr}"


@pytest.mark.fast
@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_no_absolute_site_paths(script):
    import re

    bad = [
        (i, line)
        for i, line in enumerate(script.read_text().splitlines(), 1)
        if re.search(ABSOLUTE_PATH_RE, line)
    ]
    assert not bad, (
        f"{script.name} hard-codes a site path; use $SLURM_SUBMIT_DIR or an env "
        f"var instead:\n" + "\n".join(f"  {i}: {line.strip()}" for i, line in bad)
    )


# --------------------------------------------------------------------------- #
# The profile
# --------------------------------------------------------------------------- #


def _load_profile() -> dict:
    return yaml.safe_load(PROFILE.read_text())


@functools.lru_cache(maxsize=1)
def _valid_snakemake_options() -> frozenset[str]:
    """Long-option names the installed snakemake accepts.

    Prefers the live ``snakemake --help`` (the venv is 7.32.4); falls back to
    the captured list so the test still means something where snakemake is not
    importable.
    """
    import re

    text = ""
    if shutil.which("snakemake"):
        cp = subprocess.run(
            ["snakemake", "--help"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if cp.returncode == 0:
            text = cp.stdout
    if not text:
        text = OPTIONS_FIXTURE.read_text()
    return frozenset(
        m.lstrip("-") for m in re.findall(r"(?:^|[ ,])(--[a-zA-Z0-9][a-zA-Z0-9-]*)", text)
    )


@pytest.mark.fast
def test_profile_parses():
    cfg = _load_profile()
    assert isinstance(cfg, dict) and cfg, "profile is empty or not a mapping"


@pytest.mark.fast
def test_profile_keys_are_real_snakemake_options():
    """A typo'd or 8.x-only key makes snakemake abort with 'unrecognized'."""
    valid = _valid_snakemake_options()
    assert "cluster" in valid, "could not recover snakemake's option list"
    unknown = sorted(k for k in _load_profile() if k not in valid)
    assert not unknown, (
        f"profile keys that snakemake does not accept: {unknown}. "
        "`local-rules` and `executor` in particular are Snakemake 8 only; this "
        "repo pins 7.32.4."
    )


@pytest.mark.fast
def test_profile_declares_the_pieces_that_make_retries_work():
    cfg = _load_profile()
    # cluster-status is what lets snakemake see an OOM/TIMEOUT kill at all;
    # without it restart-times is dead and the controller hangs (audit S1).
    assert "cluster-status" in cfg
    assert cfg.get("restart-times", 0) >= 1
    assert cfg.get("cluster-cancel") == "scancel"
    hook = WORKFLOW / cfg["cluster-status"]
    assert hook.resolve() == STATUS_HOOK.resolve(), (
        f"cluster-status must resolve from workflow/; got {cfg['cluster-status']}"
    )


@pytest.mark.fast
def test_cluster_string_formats_with_snakemake_placeholders():
    """The `--cluster` string is run through str.format() before the shell.

    Literal braces must therefore be doubled. If they are not, submission fails
    with a KeyError on the first job and nothing runs.
    """
    cluster = _load_profile()["cluster"]

    class _Res:
        mem_mb = 8000
        walltime = "02:00:00"

    rendered = cluster.format(rule="solve_network", threads=8, resources=_Res())
    assert rendered.startswith("sbatch ")
    assert "--parsable" in rendered, "cluster-status needs a bare job id"
    assert "smk-" in rendered, "children must be named so scancel can scope them"
    assert "solve_network" in rendered
    assert "--mem 8000" in rendered
    assert "--time 02:00:00" in rendered
    assert "--cpus-per-task 8" in rendered
    # Shell defaults survive the format pass as ${VAR:-default}, not ${{...}}.
    assert "${PYPSA_SLURM_PARTITION:-serc}" in rendered
    assert "${PYPSA_SLURM_LOGDIR:-logs/slurm}" in rendered


@pytest.mark.fast
def test_cluster_string_is_shell_parseable_once_rendered():
    class _Res:
        mem_mb = 8000
        walltime = "02:00:00"

    rendered = _load_profile()["cluster"].format(
        rule="build_shapes", threads=1, resources=_Res(),
    )
    cp = subprocess.run(
        ["bash", "-n"],
        input=f"true {rendered} 'jobscript'\n",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert cp.returncode == 0, f"rendered cluster string is not valid shell:\n{cp.stderr}"


@pytest.mark.fast
def test_default_resources_cover_the_undeclared_rules():
    """~30 rules declare no walltime and 5 no mem_mb; without a floor the
    `--cluster` string fails to format and nothing is submitted.
    """
    defaults = " ".join(_load_profile().get("default-resources", []))
    assert "mem_mb=" in defaults
    assert "walltime=" in defaults


# --------------------------------------------------------------------------- #
# The status hook
# --------------------------------------------------------------------------- #

# (sacct State string, expected verdict). The point of the hook is the bottom
# half of this table: every one of those is a job Slurm killed, which without a
# --cluster-status hook is indistinguishable from a job still running.
SACCT_STATES = [
    ("COMPLETED", "success"),
    ("RUNNING", "running"),
    ("PENDING", "running"),
    ("COMPLETING", "running"),
    ("SUSPENDED", "running"),
    ("REQUEUED", "running"),
    ("", "running"),
    ("FAILED", "failed"),
    ("TIMEOUT", "failed"),
    ("OUT_OF_MEMORY", "failed"),
    ("NODE_FAIL", "failed"),
    ("PREEMPTED", "failed"),
    ("BOOT_FAIL", "failed"),
    ("DEADLINE", "failed"),
    ("CANCELLED", "failed"),
    # sacct spells a user cancellation with a trailing "by <uid>".
    ("CANCELLED by 12345", "failed"),
    # and truncates long states when the field is narrow.
    ("OUT_OF_ME+", "failed"),
]


def _run_status_hook(tmp_path: Path, sacct_out: str, squeue_out: str = "", jobid="4242"):
    """Call slurm-status.sh with stub sacct/squeue ahead of it on $PATH."""
    stub = tmp_path / "bin"
    stub.mkdir(exist_ok=True)
    for name, out in (("sacct", sacct_out), ("squeue", squeue_out)):
        f = stub / name
        f.write_text("#!/usr/bin/env bash\nprintf '%s' \"$OUT\"\n".replace("$OUT", out))
        f.chmod(0o755)
    env = dict(os.environ, PATH=f"{stub}:{os.environ['PATH']}")
    cp = subprocess.run(
        ["bash", str(STATUS_HOOK), jobid],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert cp.returncode == 0, cp.stderr
    return cp.stdout.strip()


@pytest.mark.fast
@pytest.mark.parametrize("state,expected", SACCT_STATES, ids=lambda v: str(v).replace(" ", "_"))
def test_status_hook_maps_sacct_states(tmp_path, state, expected):
    out = _run_status_hook(tmp_path, f"{state}\n" if state else "")
    assert out == expected, f"sacct State={state!r} -> {out!r}, expected {expected!r}"


@pytest.mark.fast
@pytest.mark.parametrize(
    "noise",
    [
        "slurm_load_jobs error: Invalid job id specified",
        "Socket timed out on send/recv operation",
        "sacct: error: slurm_persist_conn_open_without_init",
    ],
    ids=["invalid-id", "socket-timeout", "persist-conn"],
)
def test_status_hook_ignores_an_error_message_on_stdout(tmp_path, noise):
    """Slurm client tools print some errors to stdout. Reading one as a state
    maps a healthy job to `failed` and kills it, so it must be discarded and
    the caller left to poll again.
    """
    assert _run_status_hook(tmp_path, noise + "\n", squeue_out=noise + "\n") == "running"


@pytest.mark.fast
def test_status_hook_falls_back_to_squeue_when_sacct_is_silent(tmp_path):
    """Slurmdbd lags a few seconds behind sbatch; squeue knows immediately."""
    assert _run_status_hook(tmp_path, "", squeue_out="RUNNING\n") == "running"


@pytest.mark.fast
def test_status_hook_prints_exactly_one_word(tmp_path):
    """Anything else aborts the whole run with 'unknown job status'."""
    for state, _ in SACCT_STATES:
        out = _run_status_hook(tmp_path, f"{state}\n" if state else "")
        assert out in {"success", "failed", "running"}
        assert "\n" not in out


@pytest.mark.fast
def test_status_hook_strips_a_federated_cluster_suffix(tmp_path):
    """`sbatch --parsable` prints '<id>;<cluster>' on a federation."""
    stub = tmp_path / "bin"
    stub.mkdir()
    # Echo the id sacct was asked about, so a bad strip shows up as a mismatch.
    (stub / "sacct").write_text(
        '#!/usr/bin/env bash\nfor a; do case "$a" in -j) next=1;; *) '
        '[ "${next:-}" = 1 ] && { echo "$a" > "$TMPDIR/seen"; next=0; };; esac; done\n'
        "echo COMPLETED\n",
    )
    (stub / "sacct").chmod(0o755)
    (stub / "squeue").write_text("#!/usr/bin/env bash\ntrue\n")
    (stub / "squeue").chmod(0o755)
    env = dict(os.environ, PATH=f"{stub}:{os.environ['PATH']}", TMPDIR=str(tmp_path))
    cp = subprocess.run(
        ["bash", str(STATUS_HOOK), "987654;sherlock"],
        capture_output=True, text=True, env=env, timeout=30,
    )
    assert cp.stdout.strip() == "success"
    assert (tmp_path / "seen").read_text().strip() == "987654"


@pytest.mark.fast
def test_status_hook_without_an_argument_does_not_hang(tmp_path):
    cp = subprocess.run(
        ["bash", str(STATUS_HOOK)], capture_output=True, text=True, timeout=30,
    )
    assert cp.stdout.strip() in {"failed", "running"}


# --------------------------------------------------------------------------- #
# The launchers
# --------------------------------------------------------------------------- #


@pytest.mark.fast
def test_run_slurm_all_refuses_a_duplicate_run_name(tmp_path):
    """--nolock is safe only when every concurrent config has its own
    run.name; the launcher must check that before it submits anything.
    """
    cfg = WORKFLOW / "repo_data" / "config" / "config.tutorial.yaml"
    env = dict(os.environ, PYPSA_DRYRUN="1", PYPSA_REPO=str(REPO))
    cp = subprocess.run(
        ["bash", str(WORKFLOW / "run_slurm_all.sh"), str(cfg), str(cfg)],
        capture_output=True, text=True, env=env, timeout=60, cwd=str(REPO),
    )
    assert cp.returncode != 0, "duplicate run.name must be refused"
    assert "run.name" in cp.stderr


@pytest.mark.fast
def test_run_slurm_all_dry_run_submits_nothing():
    cfg = WORKFLOW / "repo_data" / "config" / "config.tutorial.yaml"
    env = dict(os.environ, PYPSA_DRYRUN="1", PYPSA_REPO=str(REPO))
    cp = subprocess.run(
        ["bash", str(WORKFLOW / "run_slurm_all.sh"), str(cfg)],
        capture_output=True, text=True, env=env, timeout=60, cwd=str(REPO),
    )
    assert cp.returncode == 0, cp.stderr
    assert "DRYRUN" in cp.stderr
    assert "run_slurm.sbatch" in cp.stderr
    # The run.name really has to be READ, not just defaulted: the duplicate
    # check above is vacuous if every config resolves to the same placeholder.
    assert "run.name=Tutorial" in cp.stdout, cp.stdout


@pytest.mark.fast
def test_run_slurm_all_chains_with_afterok():
    """Ordering is a Slurm dependency, not CH2's watch_and_replot.sh, which
    grepped a controller log every 300 s for a hard-coded step count.
    """
    cfgs = [
        str(WORKFLOW / "repo_data" / "config" / "config.tutorial.yaml"),
        str(WORKFLOW / "repo_data" / "config" / "config.california.yaml"),
    ]
    env = dict(
        os.environ, PYPSA_DRYRUN="1", PYPSA_CHAIN="1", PYPSA_REPORT="1", PYPSA_REPO=str(REPO),
    )
    cp = subprocess.run(
        ["bash", str(WORKFLOW / "run_slurm_all.sh"), *cfgs],
        capture_output=True, text=True, env=env, timeout=60, cwd=str(REPO),
    )
    assert cp.returncode == 0, cp.stderr
    assert "--dependency=afterok:" in cp.stderr
    assert "run_report.sbatch" in cp.stderr


@pytest.mark.fast
def test_retired_launcher_fails_loudly():
    """run_slurm.sh ran the scheduler on a login node; the stub must not
    silently do nothing, and must name its replacement.
    """
    cp = subprocess.run(
        ["bash", str(WORKFLOW / "run_slurm.sh")],
        capture_output=True, text=True, timeout=30,
    )
    assert cp.returncode != 0
    assert "run_slurm.sbatch" in cp.stderr


@pytest.mark.fast
def test_controller_needs_a_configfile():
    cp = subprocess.run(
        ["bash", str(WORKFLOW / "run_slurm.sbatch")],
        capture_output=True, text=True, timeout=30, cwd=str(REPO),
    )
    assert cp.returncode == 2
    assert "usage" in cp.stderr.lower()


# --------------------------------------------------------------------------- #
# The statistics collector
# --------------------------------------------------------------------------- #

BENCHMARK_HEADER = (
    "s\th:m:s\tmax_rss\tmax_vms\tmax_uss\tmax_pss\tio_in\tio_out\tmean_load\tcpu_time"
)


@pytest.mark.fast
def test_collect_solve_statistics_reads_benchmark_tsvs(tmp_path):
    """One tidy CSV out of snakemake's benchmark TSVs, with the rule name
    recovered from the path or the stem.
    """
    bench = tmp_path / "benchmarks"
    cases = {
        # rule name is a path component
        "equivalence/usa/cluster_resources/elec_s300": "cluster_resources",
        # rule name is the stem prefix
        "equivalence/aggregate_to_substations": "aggregate_to_substations",
        # rule name is buried inside the stem (wildcard debris on both sides)
        "equivalence/elec_s300_add_electricity": "add_electricity",
    }
    for rel in cases:
        f = bench / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(
            BENCHMARK_HEADER
            + "\n305.6691\t0:05:05\t27827.10\t1\t1\t1\t0\t0\t92.97\t284.58\n",
        )

    out = tmp_path / "stats.csv"
    env = dict(
        os.environ,
        PYPSA_REPO=str(REPO),
        PYPSA_BENCHMARKS=str(bench),
        # A window with no jobs in it, so the sacct half contributes nothing
        # and the test does not depend on this user's job history.
        PYPSA_SACCT_SINCE="2000-01-01",
        PYPSA_JOB_TAG="no-such-tag",
    )
    cp = subprocess.run(
        ["bash", str(WORKFLOW / "collect_solve_statistics.sh"), str(out)],
        capture_output=True, text=True, env=env, timeout=120, cwd=str(REPO),
    )
    assert cp.returncode == 0, cp.stderr
    rows = out.read_text().splitlines()
    assert rows[0].startswith("source,rule,run,stem,jobid,state,wall_s")
    got = {r.split(",")[1] for r in rows[1:] if r.startswith("benchmark,")}
    assert got == set(cases.values()), got
    assert "27827.10" in out.read_text()


# --------------------------------------------------------------------------- #
# The equivalence executor switch
# --------------------------------------------------------------------------- #


@pytest.mark.fast
def test_eq_executor_default_is_local(monkeypatch):
    from tests.equivalence import build

    monkeypatch.delenv("EQ_EXECUTOR", raising=False)
    assert build.executor_profile() is None
    assert "--profile" not in build.snakemake_cmd("target")


@pytest.mark.fast
def test_eq_executor_slurm_adds_the_sherlock_profile(monkeypatch):
    from tests.equivalence import build

    monkeypatch.setenv("EQ_EXECUTOR", "slurm")
    profile = build.executor_profile()
    assert profile is not None and profile.is_absolute()
    assert (profile / "config.yaml").is_file()
    cmd = build.snakemake_cmd("target")
    assert cmd[cmd.index("--profile") + 1] == str(profile)
    # The explicit flags must still be there: snakemake applies a profile as
    # argparse defaults, so -j and friends win.
    assert "-j" in cmd and "--rerun-incomplete" in cmd


@pytest.mark.fast
def test_eq_executor_rejects_an_unknown_value(monkeypatch):
    from tests.equivalence import build

    monkeypatch.setenv("EQ_EXECUTOR", "kubernetes")
    with pytest.raises(ValueError, match="EQ_EXECUTOR"):
        build.executor_profile()


# --------------------------------------------------------------------------- #
# End to end, on a compute node only
# --------------------------------------------------------------------------- #


@pytest.mark.fast
@pytest.mark.skipif(
    shutil.which("snakemake") is None,
    reason="needs snakemake on PATH (run from the venv on a compute node)",
)
@pytest.mark.skipif(
    os.environ.get("SLURM_JOB_ID") is None,
    reason="needs a compute node; snakemake must not be run on a login node",
)
def test_profile_dry_run_resolves():
    """`snakemake -n --profile ...` must build the DAG and accept every key.

    A dry run never submits, so this exercises argument parsing and the DAG,
    not the scheduler.
    """
    cp = subprocess.run(
        [
            "snakemake", "-n", "--quiet",
            "--profile", str(PROFILE_DIR),
            "--configfile", "repo_data/config/config.tutorial.yaml",
            "--until", "cluster_network",
        ],
        capture_output=True, text=True, timeout=300, cwd=str(WORKFLOW),
    )
    assert cp.returncode == 0, (
        f"dry run with the sherlock profile failed:\n{cp.stdout[-3000:]}\n{cp.stderr[-3000:]}"
    )
