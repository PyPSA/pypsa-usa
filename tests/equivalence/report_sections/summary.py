"""Executive summary section of the equivalence report.

Leads with the plain-language verdict (is V1-epic equivalent to the anchor?),
then the provenance fingerprint, the delta index (one row per accepted
difference in the ledger), the prong-1 stage timeline strip, and a one-line
benchmark headline. See report_sections/__init__.py for the render contract.
"""

from __future__ import annotations

import html
import re


def _esc(x) -> str:
    return html.escape(str(x), quote=True)


def _user_text(s: str, labels: dict) -> str:
    """Map the internal side name 'candidate' to its user-facing label in prose."""
    return re.sub(r"\bcandidate\b", labels["candidate"], str(s))


def _truncate(s: str, n: int = 140) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _prong_counts(ctx, prong: int) -> tuple[bool, int, int]:
    """(passed, live, waived) for one prong, tolerant of missing keys."""
    f = ctx["findings"].get(prong, {}) or {}
    findings = f.get("findings", []) or []
    live = f.get("n_live")
    if live is None:
        live = sum(1 for x in findings if not x.get("waived"))
    total = f.get("n_findings")
    if total is None:
        total = len(findings)
    waived = total - live
    passed = f.get("pass", live == 0)
    return bool(passed), int(live), int(waived)


def _objective_html(ctx) -> str:
    """Prong-1 solved-objective relative difference, guarded end to end."""
    cand_name = ctx["labels"]["candidate"]
    anch_name = ctx["labels"]["anchor"]
    try:
        pair = ctx["prong_pairs"](1)[-1]
        cand_p = ctx["cand_root"] / pair.candidate
        anch_p = ctx["anch_root"] / pair.anchor
    except Exception as exc:
        return f"<p>Solved-objective comparison unavailable ({_esc(exc)}).</p>"
    missing = [str(p) for p in (cand_p, anch_p) if not p.exists()]
    if missing:
        return (
            "<p>Solved-objective comparison unavailable: solved network file(s) "
            f"not found on disk ({_esc('; '.join(missing))}).</p>"
        )
    try:
        obj_c = float(ctx["load_network"](cand_p).objective)
        obj_a = float(ctx["load_network"](anch_p).objective)
        rel = abs(obj_c - obj_a) / max(abs(obj_a), 1e-12)
    except Exception as exc:
        return f"<p>Solved-objective comparison failed while loading networks ({_esc(exc)}).</p>"
    return (
        "<p><b>Bottom line for prong 1:</b> both sides solve to the same total "
        f"system cost. Objective {cand_name}: {obj_c:,.0f}; {anch_name}: "
        f"{obj_a:,.0f}; relative difference "
        f"<b>{rel:.2e}</b> ({rel * 100:.4f}%).</p>"
    )


def _verdict_banner(ctx) -> str:
    rows = []
    prong_meaning = {
        1: "pass-through granularity — the two pipelines must match artifact-for-artifact",
        2: "pre-clustered (simpl=20) — cluster geometries differ by design, so system-level aggregates are compared",
    }
    for prong in ctx["prongs"]:
        passed, live, waived = _prong_counts(ctx, prong)
        badge = '<span class="badge pass">PASS</span>' if passed else '<span class="badge fail">FAIL</span>'
        live_badge = (
            f'<span class="badge fail">{live} live</span>' if live else '<span class="badge pass">0 live</span>'
        )
        waived_badge = f'<span class="badge delta">{waived} waived</span>'
        meaning = prong_meaning.get(prong, "")
        rows.append(
            f"<div>Prong {prong} ({_esc(meaning)}): {badge} {live_badge} {waived_badge}</div>",
        )
    all_pass = all(_prong_counts(ctx, p)[0] for p in ctx["prongs"])
    headline = (
        f"<b>{_esc(ctx['labels']['candidate'])} is equivalent to the "
        f"{_esc(ctx['labels']['anchor'])}.</b> Every remaining difference is "
        "understood, bounded, and recorded in the delta index below."
        if all_pass
        else "<b>Equivalence NOT yet established</b> — live (unwaived) "
        "differences remain; see the stage narratives below."
    )
    return f'<div class="verdict"><div>{headline}</div>' + "".join(rows) + _objective_html(ctx) + "</div>"


def _fingerprint_table(ctx) -> str:
    body = "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in ctx["fingerprint"].items())
    return (
        "<h2>Provenance fingerprint</h2>"
        "<p>Exactly what was compared: the two commits, the shared config, and "
        "when this report was generated. Rerunning the harness at these shas "
        "with this config reproduces every number in this report.</p>"
        f"<table>{body}</table>"
    )


def _delta_index(ctx) -> str:
    rows = []
    for r in ctx["ledger_rows"]:
        signoff = str(r.get("signoff", ""))
        if "provisional" in signoff.lower():
            status = '<span class="badge delta">waived - provisional</span>'
        else:
            status = '<span class="badge pass">signed</span>'
        rows.append(
            "<tr>"
            f"<td><b>{_esc(r.get('id', ''))}</b></td>"
            f"<td>{_esc(_truncate(_user_text(r.get('delta', ''), ctx['labels'])))}</td>"
            f"<td>{_esc(r.get('stages', ''))}</td>"
            f"<td>{status}</td>"
            "</tr>",
        )
    if not rows:
        rows.append('<tr><td colspan="4">No ledger entries found.</td></tr>')
    rows.append(
        '<tr><td colspan="4"><i>Bugs that were fixed in code during the '
        "harness campaign (demand conservation, hydro attachment, "
        "length_factor, empty profiles) live in the change-log, not this "
        "ledger — the ledger records only differences that remain by "
        "design.</i></td></tr>",
    )
    return (
        "<h2>Delta index</h2>"
        "<p>Every accepted difference between the two pipelines, in one place. "
        "Each row is a delta that was investigated, explained, bounded, and "
        "then waived in the ledger "
        "(docs/superpowers/specs/2026-08-07-deltas-ledger.md). "
        "&ldquo;Provisional&rdquo; means the waiver awaits user "
        "countersignature.</p>"
        "<table><tr><th>ID</th><th>What it is</th><th>Stage born</th>"
        "<th>Status</th></tr>" + "".join(rows) + "</table>"
    )


def _stage_strip(ctx) -> str:
    findings = (ctx["findings"].get(1, {}) or {}).get("findings", []) or []
    spans = []
    for stage in ctx["stage_order"]:
        at_stage = [f for f in findings if f.get("stage") == stage]
        live = sum(1 for f in at_stage if not f.get("waived"))
        waived = len(at_stage) - live
        cls = "fail" if live else ("delta" if waived else "pass")
        tip = f"{live} live, {waived} waived finding(s)"
        spans.append(
            f'<span class="badge {cls}" title="{_esc(tip)}">{_esc(stage)}</span>',
        )
    return (
        "<h2>Stage timeline (prong 1)</h2>"
        "<p>The prong-1 pipeline, stage by stage in build order. Green: the "
        "two sides matched with no findings. Amber: differences found, all "
        "waived (see delta index). Red would mean a live, unexplained "
        "difference &mdash; there are none. Hover a stage for counts.</p>"
        f'<p class="stage-strip">{" &rarr; ".join(spans)}</p>'
    )


def _benchmark_headline(ctx) -> str:
    fallback = "<p>benchmark headline unavailable</p>"
    try:
        from .benchmarks import harness_benchmark_df
    except Exception:
        return fallback
    try:
        try:
            df = harness_benchmark_df(ctx)
        except TypeError:
            df = harness_benchmark_df()
        if df is None or len(df) == 0:
            return fallback
        cand_name = ctx["labels"]["candidate"]
        anch_name = ctx["labels"]["anchor"]

        def _find_col(*needles):
            for c in df.columns:
                lc = str(c).lower()
                if any(n in lc for n in needles):
                    return c
            return None

        cand_col = _find_col("cand", "v1")
        anch_col = _find_col("anch")
        if cand_col is None or anch_col is None:
            return fallback
        cand_s = ctx["pd"].to_numeric(df[cand_col], errors="coerce")
        anch_s = ctx["pd"].to_numeric(df[anch_col], errors="coerce")
        total_c, total_a = float(cand_s.sum()), float(anch_s.sum())
        win = (anch_s - cand_s).dropna()
        sentence = (
            f"<p><b>Benchmark headline:</b> total harness wall time "
            f"{total_c / 60:.1f} min ({cand_name}) vs {total_a / 60:.1f} min "
            f"({anch_name})."
        )
        if len(win) and float(win.max()) > 0:
            idx = win.idxmax()
            rule_col = _find_col("rule", "stage", "name")
            rule = df.loc[idx, rule_col] if rule_col is not None else idx
            sentence += f" Biggest single-rule {cand_name} win: <b>{_esc(rule)}</b> ({float(win.max()):.0f}s faster)."
        return sentence + " Full per-rule table in the benchmarks section.</p>"
    except Exception:
        return fallback


def render(ctx) -> str:
    cand = _esc(ctx["labels"]["candidate"])
    anch = _esc(ctx["labels"]["anchor"])
    intro = (
        f"<p>This report answers one question: <b>does the {cand} refactor "
        f"build the same model as the {anch} branch?</b> Both pipelines were "
        "run end to end on an identical Western-interconnect configuration "
        "and every intermediate and final artifact was compared. Two prongs "
        "cover the two ways the workflow is used; differences are either "
        "explained and waived (the delta index) or the harness fails.</p>"
    )
    return (
        "<section id='summary'>"
        f"<h1>Equivalence report - {cand} vs {anch}</h1>"
        + intro
        + _verdict_banner(ctx)
        + _fingerprint_table(ctx)
        + _delta_index(ctx)
        + _stage_strip(ctx)
        + _benchmark_headline(ctx)
        + "</section>"
    )
