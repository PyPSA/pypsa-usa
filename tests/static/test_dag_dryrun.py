"""Tier A — verify the snakemake DAG resolves on representative configs.

Catches: missing inputs, typo'd rule names, wildcard mismatches, and
syntactically broken ``.smk`` files. Runs ``snakemake -n`` (dry-run only,
no rule actually executes).

No seeding step: ``workflow/Snakefile`` reads its whole layered base and the
``policy_constraints/`` static inputs out of the tracked
``workflow/repo_data/config/`` tree, and treats the per-user files under
``workflow/config/`` as optional overlays. A fresh clone resolves without
``init_pypsa_usa.sh`` having been run.
"""

import subprocess
from pathlib import Path

import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[2] / "workflow"


@pytest.mark.fast
@pytest.mark.parametrize(
    "configfile,target",
    [
        ("repo_data/config/config.tutorial.yaml", "cluster_network"),
        ("repo_data/config/config.tutorial.yaml", "solve_network"),
        ("repo_data/config/config.default.yaml", "cluster_network"),
        ("repo_data/config/config.test.yaml", "cluster_network"),
    ],
)
def test_snakemake_dryrun_resolves(configfile, target):
    result = subprocess.run(
        [
            "snakemake",
            "-n",
            "--configfile",
            configfile,
            "--until",
            target,
            "--quiet",
        ],
        cwd=WORKFLOW_DIR,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"snakemake -n failed for {configfile} --until {target}\n"
        f"stderr:\n{result.stderr}\n"
        f"stdout (last 50 lines):\n" + "\n".join(result.stdout.splitlines()[-50:])
    )
