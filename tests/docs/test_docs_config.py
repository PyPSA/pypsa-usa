"""Guardrails for the configuration reference docs.

Two failure modes are covered:

1. ``test_literalinclude_scoping`` — every ``literalinclude`` directive in the
   config reference pages must slice out exactly the YAML block it documents.
   The old ``:start-at: <key>:`` convention matched by substring, so e.g.
   ``:start-at: costs:`` latched onto the indented ``costs: wholesale`` line
   inside ``electricity: imports:`` instead of the top-level ``costs:`` block.
   Slices are re-computed here with Sphinx's semantics and parsed with
   ``yaml.safe_load`` to assert the top-level key set is exactly the expected
   one for each ``# docs : <NAME>`` marker.

2. ``test_config_tree_sync`` — ``workflow/repo_data/config/`` is canonical and
   ``init_pypsa_usa.sh`` copies it to ``workflow/config/``. The copies drift;
   this test asserts they are byte-identical (excluding per-user files).

Dependency-light on purpose: stdlib + PyYAML only.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_SOURCE = REPO_ROOT / "docs" / "source"

DOC_PAGES = [
    DOCS_SOURCE / "config-configuration.md",
    DOCS_SOURCE / "config-sectors.md",
]

# Expected top-level YAML keys per "# docs : <NAME>" marker used by
# config-configuration.md. Most markers map to a single key named after the
# marker; entries here either override that mapping or extend it (a slice may
# legitimately span the marker's key plus trailing sibling keys that share the
# same docs section, e.g. scenario + foresight).
EXPECTED_TOPLEVEL_OVERRIDES = {
    "PUDL": {"pudl_path"},
    "SCENARIO": {"scenario", "foresight"},
    "CLUSTERING": {"clustering", "focus_weights"},
    "NREL_EXCLUSION": {
        "renewable_land_access",
        "apply_cec_basescreen",
        "apply_boem_osw",
        "godeeep_wind_height",
    },
}

# Markers that select an indented fragment (sub-keys of ``renewable:`` in
# config.common.yaml) rather than a top-level block. Value = the single key
# the fragment must define.
FRAGMENT_MARKERS = {
    "SOLAR": "solar",
    "ONWIND": "onwind",
    "OFFWIND": "offwind",
    "OFFWIND_FLOATING": "offwind_floating",
    "HYDRO": "hydro",
    "EGS": "EGS",
}

MARKER_RE = re.compile(r"#\s*docs\s*:\s*(?P<name>\S+)")


def parse_literalincludes(page: Path) -> list[dict]:
    """Extract literalinclude directives (path + options) from a MyST page."""
    directives = []
    lines = page.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        match = re.match(r"\s*\.\.\s+literalinclude::\s+(\S+)", lines[i])
        if not match:
            i += 1
            continue
        options = {}
        i += 1
        while i < len(lines):
            opt = re.match(r"\s+:([\w-]+):\s?(.*)$", lines[i])
            if not opt:
                break
            options[opt.group(1)] = opt.group(2).strip()
            i += 1
        directives.append(
            {"page": page, "source": match.group(1), "options": options},
        )
    return directives


def slice_lines(lines: list[str], options: dict) -> list[str]:
    """Re-implement Sphinx's LiteralIncludeReader start/end filters.

    - ``start-after`` drops everything up to AND INCLUDING the first line
      containing the value; ``start-at`` keeps the matching line.
    - ``end-before`` truncates (exclusively) at the first line, searched after
      the start filter has been applied, containing the value.
    Matching is plain substring containment, exactly like Sphinx.
    """
    start_after = options.get("start-after")
    start_at = options.get("start-at")
    end_before = options.get("end-before")

    if start_after is not None or start_at is not None:
        needle = start_after if start_after is not None else start_at
        for lineno, line in enumerate(lines):
            if needle in line:
                lines = lines[lineno + 1 :] if start_after is not None else lines[lineno:]
                break
        else:
            raise AssertionError(f"start marker {needle!r} not found")

    if end_before is not None:
        for lineno, line in enumerate(lines):
            if end_before in line:
                lines = lines[:lineno]
                break
        else:
            raise AssertionError(f"end marker {end_before!r} not found after start")

    return lines


def assert_slice_matches_marker(text: str, marker_name: str, context: str) -> None:
    """Assert a sliced YAML snippet contains exactly its documented key(s)."""
    assert text.strip(), f"{context}: slice is empty"

    if marker_name in FRAGMENT_MARKERS:
        # Indented fragment: must stay inside its parent block, i.e. contain
        # no column-0 keys. Wrap under a dummy parent so it parses standalone.
        parsed = yaml.safe_load("_wrap_:\n" + text)
        assert isinstance(parsed, dict) and set(parsed) == {"_wrap_"}, (
            f"{context}: fragment escaped its parent block (top-level keys: {sorted(set(parsed) - {'_wrap_'})})"
        )
        expected = {FRAGMENT_MARKERS[marker_name]}
        got = set(parsed["_wrap_"] or {})
        assert got == expected, f"{context}: fragment keys {sorted(got)} != {sorted(expected)}"
        return

    expected = EXPECTED_TOPLEVEL_OVERRIDES.get(marker_name, {marker_name.lower()})
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict), f"{context}: slice does not parse to a mapping"
    got = set(parsed)
    assert got == expected, f"{context}: top-level keys {sorted(got)} != {sorted(expected)}"


def collect_directives() -> list[dict]:
    directives = []
    for page in DOC_PAGES:
        assert page.exists(), f"missing docs page {page}"
        directives.extend(parse_literalincludes(page))
    assert directives, "no literalinclude directives found in the config docs"
    return directives


@pytest.mark.parametrize(
    "directive",
    collect_directives(),
    ids=lambda d: f"{d['page'].name}:{d['options'].get('start-after') or d['options'].get('start-at') or 'whole-file'}",
)
def test_literalinclude_scoping(directive):
    source = (directive["page"].parent / directive["source"]).resolve()
    assert source.exists(), f"literalinclude source {source} does not exist"
    options = directive["options"]

    if "start-after" not in options and "start-at" not in options:
        # Whole-file include (e.g. config.plotting.yaml): just require valid YAML.
        assert isinstance(yaml.safe_load(source.read_text(encoding="utf-8")), dict)
        return

    lines = source.read_text(encoding="utf-8").splitlines()
    sliced = slice_lines(lines, options)
    text = "\n".join(sliced)
    context = f"{directive['page'].name} -> {source.name} ({options})"

    start = options.get("start-after") or options.get("start-at")
    marker = MARKER_RE.search(start) if start.lstrip().startswith("#") else None

    if marker and " : " in start:
        # "# docs : NAME" convention (config-configuration.md).
        assert_slice_matches_marker(text, marker.group("name"), context)
    else:
        # config-sectors.md style: ":start-at: # docs-<name>" on indented
        # markers inside the single top-level ``sector:`` block. The slice
        # must be non-empty and must not escape the sector block.
        assert text.strip(), f"{context}: slice is empty"
        for line in sliced:
            if line and not line.startswith((" ", "#", "\t")):
                raise AssertionError(f"{context}: slice escaped the sector block at {line!r}")
        parsed = yaml.safe_load("_wrap_:\n" + text)
        assert isinstance(parsed, dict) and parsed.get("_wrap_"), (
            f"{context}: sliced fragment does not parse to a non-empty mapping"
        )


def test_scoping_check_catches_old_costs_bug():
    """The pre-fix directive (:start-at: costs:) must fail the scoping check.

    ``costs:`` matches the indented ``costs: wholesale`` line inside
    ``electricity: imports:`` long before the real top-level ``costs:`` block,
    so the old directive rendered the wrong content. Recreate that slice and
    assert the checker rejects it.
    """
    source = REPO_ROOT / "workflow" / "repo_data" / "config" / "config.default.yaml"
    lines = source.read_text(encoding="utf-8").splitlines()
    sliced = slice_lines(lines, {"start-at": "costs:", "end-before": "# docs"})
    with pytest.raises((AssertionError, yaml.YAMLError)):
        assert_slice_matches_marker("\n".join(sliced), "COSTS", "old-costs-bug-pattern")


def _init_script_dirs() -> tuple[Path, Path]:
    """Read the template/destination dirs out of init_pypsa_usa.sh."""
    script = (REPO_ROOT / "init_pypsa_usa.sh").read_text(encoding="utf-8")
    templates = re.search(r'templates="([^"]+)"', script)
    destination = re.search(r'destination="([^"]+)"', script)
    assert templates and destination, "could not parse init_pypsa_usa.sh"
    return REPO_ROOT / templates.group(1), REPO_ROOT / destination.group(1)


# Per-user files: gitignored working copies that legitimately diverge from the
# repo_data templates (API keys, personal HPC account/walltime settings).
PER_USER_FILES = {"config.api.yaml", "config.cluster.yaml"}


def test_config_tree_sync():
    templates, destination = _init_script_dirs()
    assert templates.is_dir(), f"{templates} missing"

    template_yamls = sorted(p for p in templates.rglob("*.yaml") if p.name not in PER_USER_FILES)
    assert template_yamls, "no template yamls found"

    if not destination.is_dir() or not any(destination.rglob("*.yaml")):
        pytest.skip("workflow/config is empty — init_pypsa_usa.sh has not been run")

    stale = []
    for template in template_yamls:
        copy = destination / template.relative_to(templates)
        if not copy.exists():
            stale.append(f"{copy.relative_to(REPO_ROOT)} missing")
        elif copy.read_text(encoding="utf-8") != template.read_text(encoding="utf-8"):
            stale.append(f"{copy.relative_to(REPO_ROOT)} differs from template")

    assert not stale, f"workflow/config has drifted from workflow/repo_data/config (re-copy the templates): {stale}"
