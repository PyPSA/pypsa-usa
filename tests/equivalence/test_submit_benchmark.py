"""submit_benchmark.sh: one press yields a smoke job and a full job chained on it.

Runs the real script with a fake ``sbatch`` on PATH, so the argument plumbing
(env knobs, dependency, mail, partition) is exercised without touching Slurm.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tests" / "equivalence" / "submit_benchmark.sh"


@pytest.fixture()
def fake_sbatch(tmp_path: Path) -> Path:
    """A stand-in ``sbatch`` that logs its argv + selected env and prints an id."""
    log = tmp_path / "sbatch.log"
    shim = tmp_path / "sbatch"
    shim.write_text(
        "#!/bin/bash\n"
        f"n=$(( $(grep -c '^ARGS' '{log}' 2>/dev/null || echo 0) + 1 ))\n"
        f"echo \"ARGS $*\" >> '{log}'\n"
        f"echo \"ENV EQ_INTERCONNECT=$EQ_INTERCONNECT EQ_UNTIL=$EQ_UNTIL EQ_RUN_ID=$EQ_RUN_ID\" >> '{log}'\n"
        'echo "1000$n"\n'
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    return log


def _run(tmp_path: Path, *args: str) -> dict[str, str]:
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}")
    out = subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env, check=True).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def test_default_chains_full_after_smoke(tmp_path: Path, fake_sbatch: Path) -> None:
    facts = _run(tmp_path)
    log = fake_sbatch.read_text()
    assert facts["smoke_job"] == "10001"
    assert facts["full_job"] == "10002"
    assert facts["dependency"] == "afterok:10001"
    assert "--dependency=afterok:10001" in log
    assert log.count("--mail-type=END,FAIL") == 2
    assert "ENV EQ_INTERCONNECT=western EQ_UNTIL=assembled" in log
    assert "ENV EQ_INTERCONNECT=usa EQ_UNTIL= " in log
    assert facts["run_id"].startswith("eq-usa-") and len(facts["run_id"]) == len("eq-usa-") + 12
    assert facts["develop_sha"] and facts["config_sha12"] != "missing"


def test_smoke_only_submits_once(tmp_path: Path, fake_sbatch: Path) -> None:
    facts = _run(tmp_path, "--smoke-only")
    assert "full_job" not in facts
    assert fake_sbatch.read_text().count("ARGS") == 1


def test_full_only_has_no_dependency(tmp_path: Path, fake_sbatch: Path) -> None:
    facts = _run(tmp_path, "--full-only")
    assert "smoke_job" not in facts
    assert "dependency" not in facts
    assert "--dependency" not in fake_sbatch.read_text()


def test_dry_run_submits_nothing(tmp_path: Path, fake_sbatch: Path) -> None:
    facts = _run(tmp_path, "--dry-run")
    assert not fake_sbatch.exists()
    assert facts["smoke_job"].startswith("dry-")
