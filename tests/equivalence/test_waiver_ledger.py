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

from tests.equivalence import compare, tables
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
        hid for hid, row in sorted(registry.items()) if row.get("ported") and not str(row.get("ported_sha", "")).strip()
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

    live = sorted(hid for hid in set(registry) - ported if not registry[hid]["usa_noop"])
    assert live, "every row is ported or a no-op; nothing is left as a candidate explanation"
    for hid in live:
        ok, reason = hf.explains(hid, registry)
        assert ok, f"{hid} is live on develop but was rejected: {reason}"


def test_usa_noop_hotfix_is_not_an_explanation(registry):
    """A row that does nothing under the standing config explains nothing.

    HF-12 is the pypsa 0.30 -> 1.3 stack move: result-bearing only when
    ``conventional.unit_commitment`` is true, which the USA config leaves false.
    The ledger's instruction is not to chase these whatever their confidence, so
    the registry must not offer one as an explanation either.
    """
    noops = [hid for hid, row in registry.items() if row.get("usa_noop") and not row.get("ported")]
    assert "HF-12" in noops, "HF-12 should be an unported USA no-op in the shipped registry"
    for hid in noops:
        ok, reason = hf.explains(hid, registry)
        assert not ok, f"{hid} is a USA no-op but was accepted as an explanation"
        assert "no-op" in reason, reason


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


# --- the waiver file's own shape ---------------------------------------------

#: Every field a waiver entry may carry. Selection: which row or cell it is
#: about. Scope: which run it was measured on. Bounds: how far the explanation
#: reaches. Provenance: where the sign-off is written down.
#:
#: The two bound sets are taken from the code that ENFORCES them, never
#: restated here: a bound this file allows but nothing checks is the exact
#: failure mode these tests exist to prevent.
WAIVER_SELECT_FIELDS = ("stage", "component", "column", "kind", "metric", "key", "family")
WAIVER_SCOPE_FIELDS = ("interconnect", "prong")
TABLE_BOUND_FIELDS = tuple(tables.WAIVER_BOUND_KEYS)
CELL_BOUND_FIELDS = tuple(compare.CELL_BOUND_KEYS)
WAIVER_BOUND_FIELDS = tuple(dict.fromkeys(TABLE_BOUND_FIELDS + CELL_BOUND_FIELDS))
#: Bounds whose value is the NAME of a computed check rather than a number or a
#: direction. They are bounds in every other sense -- a waiver carrying one is
#: bounded, and ``_bounds_violation`` enforces it -- but ``float()`` on them is
#: nonsense, so they are held out of :data:`NUMERIC_BOUNDS` and validated
#: against the reconstruction registry instead.
WAIVER_COMPUTED_FIELDS = tuple(tables.WAIVER_COMPUTED_KEYS)
#: Bounds that belong to exactly one kind. ``expect_sign`` and ``max_abs_pct``
#: are shared: on a table waiver they are the sign and size of the row's delta,
#: on a cell waiver the sign and size of whatever that component measures.
TABLE_ONLY_BOUNDS = tuple(k for k in TABLE_BOUND_FIELDS if k not in CELL_BOUND_FIELDS)
CELL_ONLY_BOUNDS = tuple(k for k in CELL_BOUND_FIELDS if k not in TABLE_BOUND_FIELDS)
#: Bounds whose value is a direction, not a magnitude.
DIRECTION_BOUNDS = ("expect_sign", "expect_side")
#: Numeric bounds, which must parse and be strictly positive.
NUMERIC_BOUNDS = tuple(k for k in WAIVER_BOUND_FIELDS if k not in DIRECTION_BOUNDS + WAIVER_COMPUTED_FIELDS)
#: The bounds a cell waiver on each bounded component MUST carry. Anything a
#: component's registry entry allows but this map omits is optional
#: (``max_one_sided_pct``: a relative cap on top of the absolute MW one).
REQUIRED_CELL_BOUNDS = {
    "system_available_mw": ("expect_sign", "max_total_pct", "max_mean_rel_pct"),
    "system_potential_mw": ("expect_sign", "max_abs_pct"),
    "cluster_set": ("expect_side", "max_one_sided_mw"),
}
WAIVER_PROVENANCE_FIELDS = ("ledger", "hotfix", "reason", "justification")
WAIVER_FIELDS = frozenset(
    WAIVER_SELECT_FIELDS + WAIVER_SCOPE_FIELDS + WAIVER_BOUND_FIELDS + WAIVER_PROVENANCE_FIELDS,
)


def _is_table_waiver(w: dict) -> bool:
    return any(w.get(k) is not None for k in ("metric", "key", "family"))


def test_no_waiver_carries_an_unknown_field(waivers):
    """A misspelled field is silently ignored by the matcher, so it is a failure here."""
    bad = sorted({k for w in waivers for k in w if k not in WAIVER_FIELDS})
    assert not bad, f"unknown waiver fields (typo, or add them to WAIVER_FIELDS): {bad}"


def test_every_waiver_selects_something(waivers):
    bad = [w for w in waivers if not any(w.get(k) is not None for k in WAIVER_SELECT_FIELDS)]
    assert not bad, f"waivers that name neither a cell nor a table row: {bad}"


def test_waiver_bounds_are_well_formed(waivers):
    """Every bound must be usable, and must belong to what carries it.

    A bound the matcher cannot parse is worse than none, because it reads as a
    limit while enforcing nothing. A bound on the wrong KIND of waiver is the
    same failure by another route: ``max_total_pct`` on a table waiver is read
    by nobody.

    For a cell waiver "the right kind" is per COMPONENT, because each component's
    finding carries different fields: ``max_total_pct`` reads
    ``develop_total_mwh``, which only ``system_available_mw`` has. A bound named
    on a component that cannot read it is rejected HERE rather than silently
    evaluating to "unmeasurable" during a 20-hour run.
    """
    bad: list[str] = []
    for w in waivers:
        bounds = [k for k in WAIVER_BOUND_FIELDS if k in w]
        if not bounds:
            continue
        table = _is_table_waiver(w)
        label = f"{w.get('metric')}/{w.get('key')}" if table else f"{w.get('stage')}.{w.get('component')}"
        if table:
            wrong = [k for k in CELL_ONLY_BOUNDS if k in w]
            if wrong:
                bad.append(f"{label}: {wrong} is a cell bound on a table waiver; nothing enforces it")
        else:
            readable = compare.cell_bound_keys_for(w.get("component"))
            wrong = [k for k in bounds if k not in readable]
            if wrong:
                bad.append(
                    f"{label}: {wrong} cannot be read from a {w.get('component')!r} finding "
                    f"(readable: {list(readable) or 'none'}); nothing enforces it",
                )
        if "expect_sign" in w and w["expect_sign"] not in ("+", "-"):
            bad.append(f"{label}: expect_sign={w['expect_sign']!r}, expected '+' or '-'")
        if "expect_side" in w and w["expect_side"] not in compare.CLUSTER_SIDES:
            bad.append(f"{label}: expect_side={w['expect_side']!r}, expected one of {compare.CLUSTER_SIDES}")
        for key in NUMERIC_BOUNDS:
            if key not in w:
                continue
            # `max_total_pct: true` floats to 1.0 and reads as a real cap. A
            # bound that means something other than what it says is the failure
            # mode this whole test exists for.
            if isinstance(w[key], bool):
                bad.append(f"{label}: {key}={w[key]!r} is a boolean, not a number")
                continue
            try:
                cap = float(w[key])
            except (TypeError, ValueError):
                bad.append(f"{label}: {key}={w[key]!r} is not a number")
            else:
                if not cap > 0:
                    bad.append(f"{label}: {key}={cap} must be > 0")
    assert not bad, "malformed waiver bounds:\n" + "\n".join(bad)


def test_every_table_waiver_is_bounded(waivers):
    """A table waiver with no bounds explains any sign and any magnitude.

    That is how an HF-24 waiver written for +1.1 % of extra potential would have
    signed off a -50 % row. New table waivers state what they measured.
    """
    unbounded = [
        f"{w.get('metric')}/{w.get('key')}"
        for w in waivers
        if _is_table_waiver(w) and not any(k in w for k in TABLE_BOUND_FIELDS)
    ]
    assert not unbounded, f"table waivers with no {'/'.join(TABLE_BOUND_FIELDS)}: {unbounded}"


def test_every_reconstruction_waiver_names_a_known_reconstruction(waivers):
    """A ``reconstruction:`` value that resolves to nothing is an unbounded waiver.

    ``_bounds_violation`` refuses an unresolvable reconstruction at run time, so
    the failure mode is a row that stays UNEXPLAINED for 20 hours with no hint
    that the waiver's own name was the typo. Catch it here, in a fast test.
    """
    from tests.equivalence import reconstructions

    unknown = sorted(
        {
            str(w["reconstruction"])
            for w in waivers
            if w.get("reconstruction") and str(w["reconstruction"]) not in reconstructions.RECONSTRUCTIONS
        },
    )
    known = sorted(reconstructions.RECONSTRUCTIONS)
    assert not unknown, f"waivers naming unknown reconstructions: {unknown}; known: {known}"


def test_every_bounded_component_is_decided_here():
    """Adding a component to the bound registry forces a decision in this file.

    Otherwise a new component arrives with a checker, nothing requires its
    waivers to use it, and the next blank cheque is written by default.
    """
    assert set(REQUIRED_CELL_BOUNDS) == set(compare.BOUNDED_COMPONENTS)
    for component, required in REQUIRED_CELL_BOUNDS.items():
        readable = compare.cell_bound_keys_for(component)
        assert set(required) <= set(readable), f"{component}: {required} not all readable from {readable}"


def test_every_prong2_cell_waiver_is_bounded(waivers):
    """A waiver on a prong-2 profile finding says how far it reaches.

    All three of these are findings that would catch a real construction
    difference, and each was a blank cheque until it could be bounded:

    - ``system_available_mw`` is ``sum_bus(profile * p_nom_max)``, exactly
      invariant under the prong-2 rollup, so nothing but physics moves it;
    - ``system_potential_mw`` is ``sum(p_nom_max)`` — the capacity the model may
      build;
    - ``cluster_set`` is the ONLY report of a cluster one side does not have.
      Since master is rolled up to the common set, a develop-only cluster moves
      no comparison-table row at all: unbounded, this waiver hides capacity
      appearing or vanishing at any scale.

    ``system_potential_mw`` and ``cluster_set`` carried bound keys that were
    only ever evaluated against a ``system_available_mw`` detail, so before the
    per-component registry they could not be bounded even in principle.
    """
    bad = [
        f"{w.get('stage')}.{w.get('component')}: missing {sorted(set(REQUIRED_CELL_BOUNDS[w['component']]) - set(w))}"
        for w in waivers
        if not _is_table_waiver(w)
        and w.get("component") in REQUIRED_CELL_BOUNDS
        and not set(REQUIRED_CELL_BOUNDS[w["component"]]) <= set(w)
    ]
    assert not bad, "under-bounded cell waivers:\n" + "\n".join(bad)


def test_a_bound_the_component_cannot_read_is_malformed():
    """The registry is what makes ``max_total_pct`` on a cluster_set a typo."""
    assert compare.cell_bound_keys_for("cluster_set") == (
        "expect_side",
        "max_one_sided_mw",
        "max_one_sided_pct",
    )
    assert compare.cell_bound_keys_for("system_potential_mw") == ("expect_sign", "max_abs_pct")
    assert compare.cell_bound_keys_for("Bus") == ()
    assert compare.cell_bound_keys_for(None) == ()

    unreadable = [
        {
            "stage": "profile_solar",
            "component": "cluster_set",
            "column": "<index>",
            "kind": "row_set",
            "ledger": "DL-18",
            "max_total_pct": 0.5,
        },
    ]
    finding = {
        "stage": "profile_solar",
        "component": "cluster_set",
        "column": "<index>",
        "kind": "row_set",
        "detail": {"only_develop": {}, "only_master": {}},
    }
    waived, note = compare.waiver_status(finding, unreadable)
    assert waived is False
    assert "unreadable bound" in note and "max_total_pct" in note


# --- the two waiver kinds must not be confused for each other ----------------


def test_table_waivers_name_a_known_metric(waivers):
    """A table waiver whose metric matches nothing is dead configuration."""
    from tests.equivalence.tables import KNOWN_METRICS

    bad = [
        str(w.get("metric"))
        for w in waivers
        if w.get("metric") not in (None, "*") and str(w.get("metric")) not in KNOWN_METRICS
    ]
    assert not bad, f"waiver metrics that match no known metric: {sorted(set(bad))}"


@pytest.mark.parametrize(
    "named",
    [
        {"metric": "p_nom_max_by_zone_onwind", "key": "p8"},
        # The wildcard form is the dangerous one: `metric: '*'` used to read as
        # "names no metric", so the entry fell through to the cell branch where
        # its four absent cell fields wildcard onto EVERY finding in the run.
        {"metric": "*", "key": "*"},
        {"metric": "*"},
        {"family": "p_nom_max"},
    ],
)
def test_a_table_waiver_never_silences_a_finding(named):
    """``compare.is_waived`` must ignore any waiver that names a table row.

    A table waiver carries no stage/component/column/kind, and ``is_waived``
    treats an absent field as a wildcard — so without the guard, one
    ``{metric: ..., prong: 2, interconnect: western}`` row would waive EVERY
    western prong-2 finding, including the live ones it says nothing about.
    Naming ``metric``/``key``/``family`` at all is what makes it a table waiver;
    the VALUE, wildcard or not, does not change that.
    """
    from tests.equivalence import compare

    table_waiver = [{"interconnect": "western", "prong": 2, "ledger": "DL-18", "hotfix": "HF-24", **named}]
    finding = {
        "stage": "profile_onwind",
        "component": "system_available_mw",
        "column": "sum_bus(profile*p_nom_max)",
        "kind": "value",
        "prong": 2,
        "interconnect": "western",
    }
    assert compare.is_waived(finding, table_waiver) is False


#: The two findings the HF-24 available-power waivers were measured against
#: (western-p2-3h-20260915-1429, 2026-09-15). Develop is above master by a
#: quarter of a percent of annual energy, from the capacity master drops.
SHIPPED_AVAILABLE_MW = {
    "onwind": {"develop_total_mwh": 622_371_840.0, "master_total_mwh": 620_823_223.0, "mean_rel_pct": 0.386},
    "solar": {"develop_total_mwh": 4_513_095_168.0, "master_total_mwh": 4_506_760_381.0, "mean_rel_pct": 0.984},
}


def _available_mw_finding(tech: str, **detail) -> dict:
    return {
        "stage": f"profile_{tech}",
        "component": "system_available_mw",
        "column": "sum_bus(profile*p_nom_max)",
        "kind": "value",
        "prong": 2,
        "interconnect": "western",
        "detail": {**SHIPPED_AVAILABLE_MW[tech], **detail},
    }


def test_the_shipped_waivers_bound_system_available_mw():
    """HF-24 waives the measured available-power delta, and only that.

    ``system_available_mw`` is the metric that would catch a real capacity-factor
    construction difference, so waiving it unbounded would hollow out prong 2.
    Bounded, it is waived at the measured +0.25 % / +0.14 % and comes straight
    back the moment the delta changes sign or grows.
    """
    shipped = compare.load_waivers()
    for tech in SHIPPED_AVAILABLE_MW:
        as_measured = _available_mw_finding(tech)
        assert compare.is_waived(as_measured, shipped) is True, tech

        master = as_measured["detail"]["master_total_mwh"]
        # Develop BELOW master: the opposite of HF-24's direction.
        flipped = _available_mw_finding(tech, develop_total_mwh=master * 0.9975)
        waived, note = compare.waiver_status(flipped, shipped)
        assert waived is False, tech
        assert "expect_sign" in note, note

        # Right direction, ten times the measured size.
        big = _available_mw_finding(tech, develop_total_mwh=master * 1.03)
        waived, note = compare.waiver_status(big, shipped)
        assert waived is False, tech
        assert "max_total_pct" in note, note

        # The per-hour mean error blowing up while the annual total does not.
        noisy = _available_mw_finding(tech, mean_rel_pct=9.0)
        waived, note = compare.waiver_status(noisy, shipped)
        assert waived is False, tech
        assert "max_mean_rel_pct" in note, note

        # A dawn/dusk worst case is NOT a bound: 100 % of a few MW is noise.
        assert compare.is_waived(_available_mw_finding(tech, worst_rel_pct=100.0), shipped) is True, tech


#: What the two ``system_potential_mw`` findings measured on the same run:
#: develop's retained NREL caps capacity, a fifth of a percent of the total.
SHIPPED_POTENTIAL_MW = {
    "onwind": {"develop": 287_546.0, "master": 287_000.0, "rel_pct": 0.1902},
    "solar": {"develop": 1_802_566.0, "master": 1_800_000.0, "rel_pct": 0.1426},
}
#: And what the two ``cluster_set`` findings measured: one develop-only cluster.
SHIPPED_CLUSTER_SET = {
    "onwind": {"n_master": 18, "n_develop": 19, "n_common": 18, "only_develop": {"p87 0": 96.0}},
    "solar": {"n_master": 19, "n_develop": 20, "n_common": 19, "only_develop": {"p87 0": 2158.0}},
}


def _potential_finding(tech: str, **detail) -> dict:
    return {
        "stage": f"profile_{tech}",
        "component": "system_potential_mw",
        "column": "sum(p_nom_max)",
        "kind": "value",
        "prong": 2,
        "interconnect": "western",
        "detail": {**SHIPPED_POTENTIAL_MW[tech], **detail},
    }


def _cluster_set_finding(tech: str, **detail) -> dict:
    base = {"only_master": {}, "common_total_mw": 1_000_000.0, **SHIPPED_CLUSTER_SET[tech]}
    return {
        "stage": f"profile_{tech}",
        "component": "cluster_set",
        "column": "<index>",
        "kind": "row_set",
        "prong": 2,
        "interconnect": "western",
        "detail": {**base, **detail},
    }


def test_the_shipped_waivers_bound_system_potential_mw():
    """HF-24 only ever ADDS potential, and by a fraction of a percent.

    The finding carries ``develop``/``master``/``rel_pct`` and no annual totals,
    so before the per-component registry these two entries could not be bounded
    at all: a -44 % potential delta — capacity develop LOST — was waived by the
    entry written for +0.19 %.
    """
    shipped = compare.load_waivers()
    for tech in SHIPPED_POTENTIAL_MW:
        assert compare.is_waived(_potential_finding(tech), shipped) is True, tech

        master = _potential_finding(tech)["detail"]["master"]
        flipped = _potential_finding(tech, develop=master * 0.56, rel_pct=44.0)
        waived, note = compare.waiver_status(flipped, shipped)
        assert waived is False, tech
        assert "expect_sign" in note, note

        big = _potential_finding(tech, develop=master * 1.03, rel_pct=3.0)
        waived, note = compare.waiver_status(big, shipped)
        assert waived is False, tech
        assert "max_abs_pct" in note, note

        blind = _potential_finding(tech)
        blind["detail"] = {"note": "no totals recorded"}
        waived, note = compare.waiver_status(blind, shipped)
        assert waived is False, tech
        assert "unmeasurable" in note, note


def test_the_shipped_waivers_bound_the_cluster_set():
    """A develop-only cluster is waived; a master-only one, or a huge one, is not.

    The rollup restricts master to the common cluster set, so a one-sided
    cluster moves NO comparison-table row. This finding is the only place its
    MW is reported, which is exactly why waiving it unbounded was a blank
    cheque.
    """
    shipped = compare.load_waivers()
    for tech in SHIPPED_CLUSTER_SET:
        assert compare.is_waived(_cluster_set_finding(tech), shipped) is True, tech

        # Capacity on MASTER alone is the opposite phenomenon: develop lost it.
        wrong_side = _cluster_set_finding(tech, only_master={"p1 0": 30_000.0})
        waived, note = compare.waiver_status(wrong_side, shipped)
        assert waived is False, tech
        assert "expect_side" in note, note

        # Right side, 50 GW of it.
        huge = _cluster_set_finding(tech, only_develop={"p99 0": 50_000.0})
        waived, note = compare.waiver_status(huge, shipped)
        assert waived is False, tech
        assert "max_one_sided_mw" in note, note

        # A one-sided cluster whose MW nobody recorded is not a small one.
        unknown = _cluster_set_finding(tech, only_develop={"p99 0": float("nan")})
        waived, note = compare.waiver_status(unknown, shipped)
        assert waived is False, tech
        assert "unmeasurable" in note, note

        # And a detail with no cluster lists at all cannot be measured either.
        blind = _cluster_set_finding(tech)
        blind["detail"] = {"n_common": 18}
        waived, note = compare.waiver_status(blind, shipped)
        assert waived is False, tech
        assert "unmeasurable" in note, note


def test_max_one_sided_pct_is_measured_against_the_common_total():
    """The optional relative cap, and what it does without ``common_total_mw``."""
    waiver = [
        {
            "stage": "profile_solar",
            "component": "cluster_set",
            "column": "<index>",
            "kind": "row_set",
            "ledger": "DL-18",
            "expect_side": "develop",
            "max_one_sided_pct": 1.0,
        },
    ]
    # 2,158 MW of a 1,000,000 MW common set is 0.22 %: inside the cap.
    assert compare.is_waived(_cluster_set_finding("solar"), waiver) is True

    over = _cluster_set_finding("solar", common_total_mw=100_000.0)
    waived, note = compare.waiver_status(over, waiver)
    assert waived is False
    assert "max_one_sided_pct" in note, note

    blind = _cluster_set_finding("solar", common_total_mw=None)
    waived, note = compare.waiver_status(blind, waiver)
    assert waived is False
    assert "unmeasurable" in note, note


def test_expect_side_either_still_bounds_the_magnitude():
    """``either`` waives a side, never a size."""
    waiver = [
        {
            "stage": "profile_solar",
            "component": "cluster_set",
            "column": "<index>",
            "kind": "row_set",
            "ledger": "DL-18",
            "expect_side": "either",
            "max_one_sided_mw": 3000,
        },
    ]
    assert compare.is_waived(_cluster_set_finding("solar", only_master={"p1 0": 100.0}), waiver) is True
    waived, note = compare.waiver_status(
        _cluster_set_finding("solar", only_master={"p1 0": 30_000.0}),
        waiver,
    )
    assert waived is False
    assert "max_one_sided_mw" in note, note


def test_a_bounded_waiver_refuses_a_finding_it_cannot_measure():
    """No totals in the detail means the bound cannot be shown to hold."""
    shipped = compare.load_waivers()
    blind = _available_mw_finding("onwind")
    blind["detail"] = {"hours_mismatched": 8371}
    waived, note = compare.waiver_status(blind, shipped)
    assert waived is False
    assert "unmeasurable" in note


def test_an_unbounded_cell_waiver_still_waives_as_before():
    """Bounds are opt-in; every pre-existing entry keeps its old meaning."""
    unbounded = [{"stage": "*", "component": "Bus", "column": "control", "kind": "value", "ledger": "DL-4"}]
    finding = {"stage": "assembled_substation_network", "component": "Bus", "column": "control", "kind": "value"}
    assert compare.is_waived(finding, unbounded) is True
    assert compare.waiver_status(finding, unbounded) == (True, None)


def test_hf24_is_a_live_explanation(registry):
    """HF-24 is a develop-only fix, so it may explain a difference."""
    assert "HF-24" in registry
    ok, reason = hf.explains("HF-24", registry)
    assert ok, reason
