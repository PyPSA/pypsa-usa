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

import re
import subprocess
from pathlib import Path

import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[2] / "workflow"


# County-resolution overrides for the California config. `config.california.yaml`
# ships this same block commented out; snakemake's ``--config`` performs a
# recursive dict update, so only the listed sub-keys are replaced. California has
# 58 counties, and ``simpl: county`` selects the county-FIPS fast path in
# ``cluster_simpl``.
CALIFORNIA_COUNTY_OVERRIDE = [
    "scenario={clusters: [58], simpl: ['county']}",
    "model_topology={topological_boundaries: 'county'}",
]


@pytest.mark.fast
@pytest.mark.parametrize(
    "configfile,target,overrides",
    [
        ("repo_data/config/config.tutorial.yaml", "cluster_network", []),
        ("repo_data/config/config.tutorial.yaml", "solve_network", []),
        ("repo_data/config/config.default.yaml", "cluster_network", []),
        ("repo_data/config/config.test.yaml", "cluster_network", []),
        ("repo_data/config/config.california.yaml", "cluster_network", []),
        ("repo_data/config/config.california.yaml", "solve_network", []),
        (
            "repo_data/config/config.california.yaml",
            "cluster_network",
            CALIFORNIA_COUNTY_OVERRIDE,
        ),
    ],
    ids=lambda v: "+".join(v) if isinstance(v, list) else v,
)
def test_snakemake_dryrun_resolves(configfile, target, overrides):
    cmd = [
        "snakemake",
        "-n",
        "--configfile",
        configfile,
        "--until",
        target,
        "--quiet",
    ]
    if overrides:
        cmd += ["--config", *overrides]
    result = subprocess.run(
        cmd,
        cwd=WORKFLOW_DIR,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"snakemake -n failed for {configfile} --until {target} (overrides={overrides})\n"
        f"stderr:\n{result.stderr}\n"
        f"stdout (last 50 lines):\n" + "\n".join(result.stdout.splitlines()[-50:])
    )


def _solve_network_inputs(overrides):
    """Return the ``solve_network`` input paths snakemake resolves in a dry run."""
    cmd = [
        "snakemake",
        "-n",
        "--configfile",
        "repo_data/config/config.tutorial.yaml",
        "--until",
        "solve_network",
    ]
    if overrides:
        cmd += ["--config", *overrides]
    result = subprocess.run(
        cmd,
        cwd=WORKFLOW_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"snakemake -n failed (overrides={overrides})\nstderr:\n{result.stderr}\n"
        f"stdout (last 50 lines):\n" + "\n".join(result.stdout.splitlines()[-50:])
    )
    match = re.search(r"^rule solve_network:\n\s*input: (.*)$", result.stdout, re.MULTILINE)
    assert match, f"no solve_network job in the dry run output:\n{result.stdout}"
    return [path.strip() for path in match.group(1).split(",")]


@pytest.mark.fast
def test_interface_limits_input_follows_config_key(tmp_path):
    """``electricity: transmission_interface_limits`` selects the interface CSV.

    The key was declared in the schema, documented and defaulted, but
    ``solve_network`` hard-coded the repo path as its ``interface_limits``
    input, so pointing the key elsewhere had no effect. Assert the rule input
    tracks the config value: the default path by default, an overridden path
    when the key is overridden.
    """
    default_path = "repo_data/config/policy_constraints/transmission_interface_limits.csv"
    assert default_path in _solve_network_inputs([])

    custom = tmp_path / "custom_interface_limits.csv"
    custom.write_text((WORKFLOW_DIR / default_path).read_text())
    inputs = _solve_network_inputs([f"electricity={{transmission_interface_limits: '{custom}'}}"])
    assert str(custom) in inputs
    assert default_path not in inputs
