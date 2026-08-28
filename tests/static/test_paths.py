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
