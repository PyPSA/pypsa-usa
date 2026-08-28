"""Stage-ordered narrative spine — the heart of the equivalence report.

For each prong, walks the build-order stage spine and tells the story of each
comparison stage in plain language: an auto-composed verdict paragraph, a
born/inherited/masked lineage analysis of delta classes (prong 1), the stage's
visuals (via legacy_plots), and a compact findings table.  Prong 2 gets the
simpler treatment its aggregate-invariant design calls for.

Contract: ``render(ctx) -> str`` (see report_sections/__init__.py).
"""

from __future__ import annotations

import html
import re

from ..compare import is_waived
from . import legacy_plots as lp

PRONG_DESCRIPTORS = {
    1: 'exact comparison, simpl="" — identical clustering by construction',
    2: "aggregate invariants, simpl=20 — clustering differs by design",
}

PRONG_INTROS = {
    1: (
        "Both sides run the pass-through granularity (no simpl-stage kmeans), so every "
        "artifact should match element-for-element within tolerance. Any difference is a "
        "real code-path delta and must be either fixed or explicitly waived against a "
        "signed deltas-ledger entry."
    ),
    2: (
        "At simpl=20 the two branches cluster at different points in the DAG, so their "
        "intermediate networks differ by design. Only clustering-invariant aggregates "
        "(system totals, per-carrier capacity) and the solved outcome are compared."
    ),
}


def _stage_names(clusters: str) -> dict[str, str]:
    return {
        "demand": "Demand (system total of the build-demand CSVs)",
        "profile_onwind": "Onshore-wind capacity-factor profile",
        "profile_solar": "Solar capacity-factor profile",
        "assembled_substation_network": (
            "Assembled substation network (add_electricity output vs anchor simplify output)"
        ),
        "clustered_network": f"Clustered network ({clusters} clusters)",
        "extra_components": "Extra components (storage carriers added)",
        "prepared_network": "Prepared network (line limits and 3h resolution applied)",
        "sectored_network": "Sectored network (electricity-only sector wrap)",
        "solved_network": "Solved network (capacity-expansion optimum)",
        "prong2_aggregates": "Pre-solve aggregate invariants (clustered-network totals)",
    }


_KIND_WORDS = {
    "value": ("value difference", "value differences"),
    "row_set": ("row-set difference", "row-set differences"),
    "column_set": ("column-set difference", "column-set differences"),
    "suppressed": ("suppression marker", "suppression markers"),
    "missing_artifact": ("missing artifact", "missing artifacts"),
    "duplicate_labels": ("duplicate-label note", "duplicate-label notes"),
    "load_rekey": ("load re-key note", "load re-key notes"),
}


def _esc(x) -> str:
    return html.escape(str(x))


def _user_side_text(s: str, labels: dict) -> str:
    """Map internal side tokens to user-facing labels in display text.

    Internal keys stay 'candidate' in manifests/findings; only rendered text
    is rewritten (e.g. 'candidate-only' -> 'V1-epic-only').
    """
    return re.sub(r"\bcandidate\b(?=-only)", labels["candidate"], s)


def _kind_phrase(kind: str, n: int) -> str:
    one, many = _KIND_WORDS.get(kind, (f"{kind} finding", f"{kind} findings"))
    return f"{n} {one if n == 1 else many}"


def _num(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v == int(v) and abs(v) < 1e15:
        return f"{int(v):,}"
    if abs(v) >= 1000:
        return f"{v:,.1f}"
    return f"{v:.4g}"


def _ledger_ids(stage_findings: list[dict], waivers: list[dict]) -> list[str]:
    """Distinct ledger ids of waivers matching these findings (is_waived logic)."""
    ids: set[str] = set()
    for f in stage_findings:
        for w in waivers:
            if is_waived(f, [w]) and w.get("ledger"):
                ids.add(str(w["ledger"]))

    def _key(i: str):
        m = re.search(r"(\d+)", i)
        return (int(m.group(1)) if m else 0, i)

    return sorted(ids, key=_key)


def _worst_value_sentence(value_findings: list[dict], labels: dict) -> str:
    """Characterize the worst value finding: component.column, n_diff/n_total, max_abs."""
    best: tuple[float, float] | None = None
    sentence = ""
    for f in value_findings:
        det = f.get("detail")
        name = f"{f.get('component')}.{f.get('column')}"
        rank = None
        s = ""
        if isinstance(det, dict) and "max_abs" in det:
            rank = (1.0, float(det["max_abs"]))
            ex = (det.get("examples") or [{}])[0]
            s = (
                f"The largest value delta is <b>{_esc(name)}</b>: "
                f"{_num(det.get('n_diff'))} of {_num(det.get('n_total'))} entries differ, "
                f"max |&Delta;| = {_num(det.get('max_abs'))}"
            )
            if det.get("max_rel_pct") is not None:
                s += f" ({_num(det['max_rel_pct'])}% max relative)"
            if ex.get("id") is not None:
                s += (
                    f" (e.g. {_esc(ex.get('id'))}: {_esc(labels['candidate'])} "
                    f"{_num(ex.get('candidate'))} vs {_esc(labels['anchor'])} "
                    f"{_num(ex.get('anchor'))}"
                )
                s += f", {_num(ex['rel_pct'])}%)" if ex.get("rel_pct") is not None else ")"
            s += "."
        elif isinstance(det, list) and det:
            deltas = [abs(float(d.get("candidate", 0)) - float(d.get("anchor", 0))) for d in det]
            i = max(range(len(det)), key=lambda k: deltas[k])
            d = det[i]
            rank = (1.0, deltas[i])
            pct = f", {_num(d['rel_pct'])}%" if d.get("rel_pct") is not None else ""
            s = (
                f"The largest value delta is <b>{_esc(name)}</b>: {len(det)} cells outside "
                f"tolerance; worst is {_esc(d.get('carrier', d.get('id', '?')))} "
                f"({_esc(labels['candidate'])} {_num(d.get('candidate'))} vs "
                f"{_esc(labels['anchor'])} {_num(d.get('anchor'))}{pct})."
            )
        elif isinstance(det, dict) and "rel" in det:
            rank = (1.0, abs(float(det.get("candidate", 0)) - float(det.get("anchor", 0))))
            s = (
                f"The largest value delta is <b>{_esc(name)}</b>: "
                f"{_esc(labels['candidate'])} {_num(det.get('candidate'))} vs "
                f"{_esc(labels['anchor'])} {_num(det.get('anchor'))} "
                f"(relative difference {float(det['rel']) * 100:.4g}%)."
            )
        elif isinstance(det, dict) and "n_diff" in det:
            rank = (0.0, float(det["n_diff"]))
            ex = (det.get("examples") or [{}])[0]
            s = (
                f"The most widespread non-numeric difference is <b>{_esc(name)}</b>: "
                f"{_num(det['n_diff'])} of {_num(det.get('n_total'))} entries differ"
            )
            if ex.get("id") is not None:
                s += (
                    f" (e.g. {_esc(ex.get('id'))}: {_esc(labels['candidate'])} "
                    f"'{_esc(ex.get('candidate'))}' vs {_esc(labels['anchor'])} "
                    f"'{_esc(ex.get('anchor'))}')"
                )
            s += "."
        if rank is not None and (best is None or rank > best):
            best, sentence = rank, s
    return sentence


def _structural_sentences(stage_findings: list[dict], labels: dict) -> list[str]:
    """Summarize what kind of elements differ for row_set / column_set findings."""
    out: list[str] = []
    row = [f for f in stage_findings if f["kind"] == "row_set"]
    if row:
        parts = []
        for f in row:
            det = str(f.get("detail", ""))
            ns = re.findall(r"n=(\d+)", det)
            extra = f" ({' / '.join(ns)} one-side-only elements)" if ns else ""
            parts.append(f"{f['component']}{extra}")
        out.append(
            "Row-set differences (element names present on one side only) affect "
            + "; ".join(_esc(p) for p in parts)
            + " — a naming-scheme difference (per-plant vs pre-aggregated names), with "
            "content guarded by the aggregate comparisons.",
        )
    cols = [f for f in stage_findings if f["kind"] == "column_set"]
    if cols:
        groups: dict[tuple[str, str], list[str]] = {}
        for f in cols:
            groups.setdefault((str(f["component"]), str(f.get("detail"))), []).append(
                str(f["column"]),
            )
        bits = []
        for (comp, side), names in sorted(groups.items()):
            side_label = _user_side_text(side, labels)
            label = f"{len(names)} {side_label} column{'s' if len(names) > 1 else ''} on {comp}"
            if len(names) <= 3:
                label += f" ({', '.join(names)})"
            bits.append(label)
        out.append("Column-set differences: " + "; ".join(_esc(b) for b in bits) + ".")
    sup = [f for f in stage_findings if f["kind"] == "suppressed"]
    for f in sup:
        out.append(
            f"A suppression marker on {_esc(f['component'])} caps that frame's fan-out ({_esc(f.get('detail'))}).",
        )
    return out


def _verdict_html(stage_findings: list[dict], waivers: list[dict], labels: dict) -> str:
    if not stage_findings:
        return (
            '<div class="verdict"><span class="badge pass">identical</span> '
            "Identical within tolerance — no differences found.</div>"
        )
    counts: dict[str, int] = {}
    for f in stage_findings:
        counts[f["kind"]] = counts.get(f["kind"], 0) + 1
    count_words = ", ".join(_kind_phrase(k, n) for k, n in sorted(counts.items()))
    n = len(stage_findings)
    all_waived = all(f.get("waived") for f in stage_findings)
    ledgers = _ledger_ids(stage_findings, waivers)
    if all_waived:
        badge = '<span class="badge delta">waived deltas</span>'
        lead = "The single finding is waived" if n == 1 else f"All {n} findings are waived"
        waive_sentence = (
            lead
            + (
                f" under ledger entr{'ies' if len(ledgers) > 1 else 'y'} {', '.join(_esc(i) for i in ledgers)}"
                if ledgers
                else ""
            )
            + "."
        )
    else:
        n_live = sum(1 for f in stage_findings if not f.get("waived"))
        badge = '<span class="badge fail">live deltas</span>'
        waive_sentence = (
            f"{n_live} of {n} findings are NOT waived (live)"
            + (f"; waived ones fall under {', '.join(_esc(i) for i in ledgers)}" if ledgers else "")
            + "."
        )
    sentences = [f"This stage shows {count_words}.", waive_sentence]
    worst = _worst_value_sentence([f for f in stage_findings if f["kind"] == "value"], labels)
    if worst:
        sentences.append(worst)
    sentences += _structural_sentences(stage_findings, labels)
    return f'<div class="verdict">{badge} ' + " ".join(sentences) + "</div>"


def _fmt_class(c: tuple) -> str:
    comp, col, kind = c
    return f"{comp}.{col} ({kind})"


def _lineage_html(prong_findings: list[dict], stage: str, stage_list: list[str]) -> str:
    """Born / inherited / resolved-masked analysis of delta classes at a stage."""
    classes_at = {
        s: {(f["component"], f["column"], f["kind"]) for f in prong_findings if f["stage"] == s} for s in stage_list
    }
    idx = stage_list.index(stage)
    here = classes_at[stage]
    earlier: set = set()
    for s in stage_list[:idx]:
        earlier |= classes_at[s]
    born = sorted(here - earlier)
    inherited = sorted(here & earlier)
    prev = stage_list[idx - 1] if idx > 0 else None
    masked = sorted(classes_at[prev] - here) if prev else []

    def _fmt_list(classes: list[tuple], cap: int = 12) -> str:
        shown = [_esc(_fmt_class(c)) for c in classes[:cap]]
        more = len(classes) - cap
        return ", ".join(shown) + (f" &hellip; +{more} more" if more > 0 else "")

    items = []
    if born:
        items.append(f"<li><b>Born here:</b> {_fmt_list(born)}</li>")
    if inherited:
        items.append(f"<li><b>Inherited from earlier stages:</b> {_fmt_list(inherited)}</li>")
    if masked:
        items.append(f"<li><b>Resolved/masked before this stage:</b> {_fmt_list(masked)}</li>")
    if not items:
        return ""
    return "<ul>" + "".join(items) + "</ul>"


def _findings_table(stage_findings: list[dict], pd, labels: dict) -> str:
    if not stage_findings:
        return ""
    rows = [
        {
            "component": f.get("component"),
            "column": f.get("column"),
            "kind": f.get("kind"),
            "waived": "yes" if f.get("waived") else "NO (live)",
            "detail": _user_side_text(str(f.get("detail")), labels)[:160],
        }
        for f in stage_findings
    ]
    df = pd.DataFrame(rows, columns=["component", "column", "kind", "waived", "detail"])
    return df.to_html(index=False, escape=True, border=0)


def _stage_visuals(stage: str, pair, prong: int, ctx: dict) -> str:
    """Dispatch the stage's plots; skip on missing files; guard each figure."""
    if pair is None:
        return ""
    cand_root, anch_root = ctx["cand_root"], ctx["anch_root"]
    if not (cand_root / pair.candidate).exists() or not (anch_root / pair.anchor).exists():
        return "<p><i>Visuals skipped: artifact missing on one side.</i></p>"
    calls = []
    if stage == "demand":
        calls = [lambda: lp.demand_overlay(pair, cand_root, anch_root)]
    elif stage.startswith("profile_"):
        calls = [lambda: lp.profile_curves(pair, cand_root, anch_root)]
    elif stage == "assembled_substation_network":
        calls = [
            lambda: lp.capacity_bars(pair, cand_root, anch_root),
            lambda: lp.scatter_network(pair, cand_root, anch_root),
            lambda: lp.pnom_attribute_section(prong, cand_root, anch_root),
        ]
    elif stage in ("clustered_network", "extra_components", "prepared_network", "sectored_network"):
        calls = [lambda: lp.capacity_bars(pair, cand_root, anch_root)]
    elif stage == "solved_network":
        # pair.solve_stage=True switches capacity_bars to p_nom_opt.
        calls = [lambda: lp.capacity_bars(pair, cand_root, anch_root)]
    parts = []
    for call in calls:
        try:
            parts.append(call())
        except Exception as e:
            parts.append(f"<p><i>Figure failed for {_esc(stage)}: {_esc(e)}</i></p>")
    return "".join(parts)


def _prong_stage_list(prong: int, ctx: dict) -> list[str]:
    """Stages to walk: stage_order restricted to this prong's pairs/findings,
    plus finding-only stage keys (prong 2's prong2_aggregates) in build order.
    """
    pair_stages = [p.stage for p in ctx["prong_pairs"](prong)]
    finding_stages = {f["stage"] for f in ctx["findings"][prong].get("findings", [])}
    if prong == 1:
        present = set(pair_stages) | finding_stages
        return [s for s in ctx["stage_order"] if s in present]
    # Prong 2: only stage keys that actually carry findings (aggregate design).
    ordered = ["prong2_aggregates", "solved_network"]
    out = [s for s in ordered if s in finding_stages]
    out += sorted(finding_stages - set(ordered))
    return out or ordered


def _render_prong1(ctx: dict) -> str:
    pd = ctx["pd"]
    labels = ctx["labels"]
    findings = ctx["findings"][1].get("findings", [])
    stage_list = _prong_stage_list(1, ctx)
    pair_by_stage = {p.stage: p for p in ctx["prong_pairs"](1)}
    names = _stage_names(str(ctx.get("clusters", "")))
    parts = [
        f"<h2>Prong 1: {_esc(PRONG_DESCRIPTORS[1])}</h2>",
        f"<p>{_esc(PRONG_INTROS[1])}</p>",
    ]
    for stage in stage_list:
        sf = [f for f in findings if f["stage"] == stage]
        parts.append(f"<h3>{_esc(names.get(stage, stage))}</h3>")
        parts.append(_verdict_html(sf, ctx["waivers"], labels))
        parts.append(_lineage_html(findings, stage, stage_list))
        parts.append(_stage_visuals(stage, pair_by_stage.get(stage), 1, ctx))
        parts.append(_findings_table(sf, pd, labels))
    return "".join(parts)


def _render_prong2(ctx: dict) -> str:
    pd = ctx["pd"]
    labels = ctx["labels"]
    findings = ctx["findings"][2].get("findings", [])
    names = _stage_names(str(ctx.get("clusters", "")))
    parts = [
        f"<h2>Prong 2: {_esc(PRONG_DESCRIPTORS[2])}</h2>",
        f"<p>{_esc(PRONG_INTROS[2])}</p>",
    ]
    for stage in _prong_stage_list(2, ctx):
        sf = [f for f in findings if f["stage"] == stage]
        parts.append(f"<h3>{_esc(names.get(stage, stage))}</h3>")
        parts.append(_verdict_html(sf, ctx["waivers"], labels))
        parts.append(_findings_table(sf, pd, labels))
    # DL-9 root cause, one plain-language sentence (ledger-sourced when available).
    dl9 = next((r for r in ctx.get("ledger_rows", []) if r.get("id") == "DL-9"), None)
    explanation = (
        f"All of prong 2's findings share one root cause, ledger entry <b>DL-9</b>: shared "
        f"attach code silently drops existing wind/solar capacity whose aggregation group has "
        f"no renewable-profile generator, and the number of profile-less groups depends on "
        f"cluster geometry — so the two by-design different clusterings start from different "
        f"existing-RE baselines ({_esc(labels['candidate'])} keeps more and is closer to "
        f"plant-data ground truth), which propagates into the objective and per-carrier "
        f"p_nom_opt; each side is internally consistent and conserved through clustering."
    )
    parts.append(f"<p>{explanation}</p>")
    if dl9:
        dl9_delta = re.sub(r"\bcandidate\b", labels["candidate"], str(dl9.get("delta", "")))
        parts.append(
            "<p><i>Ledger DL-9 (delta): " + _esc(dl9_delta[:300]) + "</i></p>",
        )
    return "".join(parts)


def render(ctx: dict) -> str:
    parts = ["<h2>Stage-by-stage story</h2>"]
    parts.append(
        "<p>The harness compares the two branches at every shared logical state of the "
        "build, in DAG order. Each stage below leads with a plain-language verdict, then "
        "shows where each delta class first appears (born) or carries over (inherited), "
        "the stage's visuals, and the raw findings.</p>",
    )
    for prong in ctx["prongs"]:
        parts.append(_render_prong1(ctx) if prong == 1 else _render_prong2(ctx))
    return "".join(parts)
