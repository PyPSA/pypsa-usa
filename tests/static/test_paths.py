"""Tier A — assert every rule path uses a category constant and resolves.

After PR #12 reorganized ``resources/`` into category-first subfolders
(``NETWORKS``, ``BUSMAPS``, ``PROFILES``, ``GEOSPATIAL``, ``COSTS``,
``PRICES``, ``DEMAND``, ``POWERPLANTS``, ``HEATING_COP``, ``TEMPERATURE``,
``POPULATION``, ``CO2``), rule inputs/outputs must use these constants
rather than hard-coded ``RESOURCES + "{interconnect}/..."`` strings.
This test scans every ``.smk`` file (and the root Snakefile) for the
forbidden pattern. ``.smk`` files are not pure Python and cannot be AST
parsed without a Snakemake-aware preprocessor, so we walk the source text
directly with a regex that matches ``RESOURCES + "<...>"`` concatenations
where the literal contains ``{interconnect}/``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[2] / "workflow"
SMK_FILES = list(WORKFLOW_DIR.glob("Snakefile")) + list(
    WORKFLOW_DIR.glob("rules/*.smk"),
)

CATEGORY_CONSTANTS = {
    "NETWORKS",
    "BUSMAPS",
    "PROFILES",
    "GEOSPATIAL",
    "COSTS",
    "PRICES",
    "POWERPLANTS",
    "DEMAND",
    "HEATING_COP",
    "TEMPERATURE",
    "POPULATION",
    "CO2",
}

# Match  RESOURCES + "...{interconnect}/..."  or
#        RESOURCES + f"...{{interconnect}}/..."  (f-strings escape braces)
# across a logical concatenation (allowing whitespace / line continuations
# between the operands). The leading boundary (\b not after a letter) keeps
# this from matching e.g. ``MY_RESOURCES``.
HARDCODED_RE = re.compile(
    r"(?<![A-Za-z0-9_])RESOURCES\s*\+\s*f?\"([^\"]*\{interconnect\}/[^\"]*)\"",
)


@pytest.mark.fast
@pytest.mark.parametrize("smk_path", SMK_FILES, ids=lambda p: p.name)
def test_no_hardcoded_resources_interconnect_paths(smk_path):
    """RESOURCES + "{interconnect}/..." is forbidden — use a category constant."""
    source = smk_path.read_text()
    violations = []
    for match in HARDCODED_RE.finditer(source):
        line_no = source.count("\n", 0, match.start()) + 1
        violations.append((line_no, match.group(1)))
    assert not violations, (
        f"{smk_path.name}: found RESOURCES + '{{interconnect}}/...' literals — "
        f"use a category constant ({', '.join(sorted(CATEGORY_CONSTANTS))}):\n"
        + "\n".join(f"  line {ln}: {lit!r}" for ln, lit in violations)
    )


# --- The Sherlock driver -----------------------------------------------------
# One parameterised sbatch script replaces run_usa.sbatch and run_usa_cf.sbatch.
# Both of those died on their first line because they hard-coded
# ``REPO=/oak/.../refactor/pypsa-usa``, a path that no longer exists — the same
# class of bug as a hard-coded RESOURCES path above, and the reason the driver
# now resolves its root from $SLURM_SUBMIT_DIR.

SBATCH_DRIVERS = sorted(
    (Path(__file__).resolve().parents[1] / "equivalence").glob("*.sbatch"),
)

# Site software paths (gurobi's licence, the module tree) are not repository
# paths and are allowed; a checkout location and a per-user scratch directory
# are not. $OAK, $SCRATCH and $GROUP_SCRATCH exist precisely so these never have
# to be written down, and a literal one silently fails the job for anyone whose
# SUNet id is not the one it was written for.
PER_USER_PATH_RE = re.compile(r"/oak/\S+|/scratch/groups/\S+|/scratch/users/\S+|/home/(users|groups)/\S+")


@pytest.mark.fast
def test_exactly_one_sbatch_driver():
    """run_usa.sbatch and run_usa_cf.sbatch are gone, not renamed alongside."""
    names = [p.name for p in SBATCH_DRIVERS]
    assert names == ["run_equivalence.sbatch"], names


@pytest.mark.fast
@pytest.mark.parametrize("sbatch_path", SBATCH_DRIVERS, ids=lambda p: p.name)
def test_sbatch_uses_the_serc_partition(sbatch_path):
    """The project's partition is serc (PROJECT.md §4), not normal."""
    source = sbatch_path.read_text()
    assert re.search(r"^#SBATCH\s+-p\s+serc\b", source, re.MULTILINE), (
        f"{sbatch_path.name} does not request -p serc"
    )
    assert not re.search(r"^#SBATCH\s+-p\s+normal\b", source, re.MULTILINE), (
        f"{sbatch_path.name} still requests -p normal"
    )


@pytest.mark.fast
@pytest.mark.parametrize("sbatch_path", SBATCH_DRIVERS, ids=lambda p: p.name)
def test_sbatch_has_no_per_user_absolute_path(sbatch_path):
    """No ``/oak/...`` or per-user scratch literal, in a directive or the body.

    The checkout is resolved from ``$SLURM_SUBMIT_DIR``, the caches from
    ``$GROUP_SCRATCH``, and the Slurm log paths are relative. A literal here is
    the exact bug that made both predecessors of this driver fail on their first
    line.
    """
    violations = []
    for line_no, line in enumerate(sbatch_path.read_text().splitlines(), start=1):
        for match in PER_USER_PATH_RE.finditer(line):
            violations.append((line_no, match.group(0)))
    assert not violations, (
        f"{sbatch_path.name} hard-codes a per-user absolute path — resolve it from "
        f"$SLURM_SUBMIT_DIR, $GROUP_SCRATCH or an env knob instead:\n"
        + "\n".join(f"  line {ln}: {lit}" for ln, lit in violations)
    )


@pytest.mark.fast
@pytest.mark.parametrize("sbatch_path", SBATCH_DRIVERS, ids=lambda p: p.name)
def test_sbatch_log_directives_are_relative(sbatch_path):
    """``#SBATCH -o/-e`` must not name an absolute directory.

    sbatch refuses the job outright when the log directory does not exist, and
    an absolute site path is one someone else will not have. Relative means the
    submit directory, which by construction exists.
    """
    bad = [
        line.strip()
        for line in sbatch_path.read_text().splitlines()
        if re.match(r"^#SBATCH\s+-[oe]\s+/", line)
    ]
    assert not bad, f"{sbatch_path.name} writes Slurm logs to an absolute path: {bad}"
