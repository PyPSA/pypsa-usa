"""The waiver file and the hot-fix registry, checked against each other.

Tier A (``fast``): pure YAML, no git, no data, no build.

Two files carry the project's explanations for a difference — ``waivers.yaml``
("this delta was signed off") and ``hotfixes.yaml`` ("this change on develop can
produce a delta"). They are only useful if every id in one resolves in the
other, and if the registry's own shape is trustworthy: a full ``HF-1``…``HF-n``
range with no gaps, so a reader who sees ``HF-17`` referenced knows it is a real
row and not a typo for ``HF-1``.

The sharpest check here is :func:`test_ported_hotfix_is_not_an_explanation`. A
hot-fix ported onto ``master-benchmark`` runs on BOTH sides, so it cannot
explain a difference between them; citing one hides a broken port behind a
plausible story.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from tests.equivalence import hotfixes as hf

pytestmark = pytest.mark.fast

WAIVERS = Path(__file__).parent / "waivers.yaml"
HOTFIXES = hf.HOTFIXES_PATH

DL_RE = re.compile(r"^DL-\d+$")
HF_RE = re.compile(r"^HF-\d+$")


@pytest.fixture(scope="module")
def registry() -> dict[str, dict]:
    reg = hf.load_hotfixes()
    assert reg, f"{HOTFIXES} is missing or empty"
    return reg


@pytest.fixture(scope="module")
def waivers() -> list[dict]:
    rows = yaml.safe_load(WAIVERS.read_text()) or []
    assert rows, f"{WAIVERS} is missing or empty"
    return rows


# --- the registry's own shape ------------------------------------------------


def test_ids_are_bare_and_unique(registry):
    rows = yaml.safe_load(HOTFIXES.read_text())
    ids = [str(r["id"]) for r in rows]
    bad = [i for i in ids if not HF_RE.match(i)]
    assert not bad, f"ids must match ^HF-\\d+$ (bare, never zero-padded): {bad}"
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    assert not dupes, f"duplicate hot-fix ids: {dupes}"


def test_id_range_has_no_gaps(registry):
    """HF-1..HF-n, every integer present.

    A gap means a ledger row was transcribed away, and the reader of a
    difference tagged with the missing id has nowhere to go.
    """
    numbers = sorted(int(i.split("-")[1]) for i in registry)
    assert numbers[0] == 1, f"the registry starts at HF-{numbers[0]}, not HF-1"
    expected = list(range(1, numbers[-1] + 1))
    missing = sorted(set(expected) - set(numbers))
    assert not missing, f"gaps in the hot-fix range: {[f'HF-{n}' for n in missing]}"


def test_every_row_has_the_required_fields(registry):
    bad: list[str] = []
    for hid, row in sorted(registry.items()):
        for f in hf.REQUIRED_FIELDS:
            if f not in row:
                bad.append(f"{hid}: missing field {f!r}")
        commit = str(row.get("commit", ""))
        if len(commit) not in (7, 40) or not re.fullmatch(r"[0-9a-f]+", commit):
            bad.append(f"{hid}: commit={commit!r}, expected a 7- or 40-char hex sha")
        pr = row.get("pr")
        # null only for a harness-branch change that never had an upstream PR.
        if pr is not None and not isinstance(pr, int):
            bad.append(f"{hid}: pr={pr!r}, expected an integer or null")
        if not str(row.get("title", "")).strip():
            bad.append(f"{hid}: title is empty")
        if row.get("confidence") not in hf.CONFIDENCES:
            bad.append(f"{hid}: confidence={row.get('confidence')!r}, expected one of {hf.CONFIDENCES}")
        for flag in ("ported", "usa_noop"):
            if not isinstance(row.get(flag), bool):
                bad.append(f"{hid}: {flag}={row.get(flag)!r}, expected a boolean")
    assert not bad, "hot-fix registry shape violations:\n" + "\n".join(bad)


def test_ported_rows_name_their_baseline_commit(registry):
    """A ported row says WHICH master-benchmark commit carries it."""
    bad = [
        hid
        for hid, row in sorted(registry.items())
        if row.get("ported") and not str(row.get("ported_sha", "")).strip()
    ]
    assert not bad, f"ported rows with no ported_sha: {bad}"


# --- the join with the waivers -----------------------------------------------


def test_every_waiver_names_a_deltas_ledger_row(waivers):
    bad = [w for w in waivers if not DL_RE.match(str(w.get("ledger", "")))]
    assert not bad, "waivers whose ledger id is missing or not DL-<n>:\n" + "\n".join(
        f"  {w.get('component')}.{w.get('column')}: ledger={w.get('ledger')!r}" for w in bad
    )


def test_every_waiver_hotfix_resolves(waivers, registry):
    bad: list[str] = []
    for w in waivers:
        hid = w.get("hotfix")
        if hid is None:
            continue
        hid = str(hid)
        if not HF_RE.match(hid):
            bad.append(f"{w.get('component')}.{w.get('column')}: hotfix={hid!r} is not ^HF-\\d+$")
        elif hid not in registry:
            bad.append(f"{w.get('component')}.{w.get('column')}: hotfix={hid} has no row in {HOTFIXES.name}")
    assert not bad, "waiver hot-fix references that do not resolve:\n" + "\n".join(bad)


# --- the rule the whole registry exists for ----------------------------------


def test_ported_hotfix_is_not_an_explanation(registry):
    """``explains`` rejects every ported id, and accepts the live ones.

    A ported hot-fix is on ``master-benchmark``, so BOTH sides run it. If a
    difference traces to one, the port is broken — that is a finding, not an
    explanation. The rejection message has to say so, because it is what a
    reader of the comparison table sees.
    """
    ported = hf.ported_ids(registry)
    assert ported, "no ported rows at all; T1 marked nothing, or the registry is stale"

    for hid in sorted(ported):
        ok, reason = hf.explains(hid, registry)
        assert not ok, f"{hid} is ported to master-benchmark but was accepted as an explanation"
        assert "ported" in reason and "both sides" in reason.lower(), reason

    live = sorted(set(registry) - ported)
    assert live, "every row is ported; nothing is left as a candidate explanation"
    for hid in live:
        ok, reason = hf.explains(hid, registry)
        assert ok, f"{hid} is live on develop but was rejected: {reason}"


def test_unknown_hotfix_is_not_an_explanation(registry):
    ok, reason = hf.explains("HF-9999", registry)
    assert not ok
    assert "HF-9999" in reason


def test_live_code_differences_excludes_ported_and_usa_noops(registry):
    live = hf.live_code_differences(registry)
    ids = [d["hotfix"] for d in live]
    assert ids, "no live code differences at all; the USA run would have no suspects"
    for hid in ids:
        assert not registry[hid]["ported"], f"{hid} is ported but listed as a live code difference"
        assert not registry[hid]["usa_noop"], f"{hid} is a USA no-op but listed as a live code difference"
    assert set(live[0]) == {"key", "master", "develop", "hotfix"}


def test_missing_registry_file_yields_empty(tmp_path):
    assert hf.load_hotfixes(tmp_path / "nope.yaml") == {}
