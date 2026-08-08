"""Unified diff-DAG section: one overlaid rulegraph for both sides.

Runs ``snakemake --rulegraph`` in each side's workflow/ (candidate = V1-epic
checkout, anchor = worktree), parses the dot output into rule-name-level node
and edge sets, and renders a single union graph where color encodes which side
each rule/dependency belongs to. Per-rule walltimes (from the concurrently
written ``benchmarks`` section module) are annotated on the node labels when
available.
"""

from __future__ import annotations

import html
import re
import subprocess
from pathlib import Path

from ..paths import anchor_final_target, final_target

_NODE_RE = re.compile(r'(\d+)\[label = "([^"]+)"')
_EDGE_RE = re.compile(r"(\d+) -> (\d+)")

_FILL_BOTH = "#eeeeee"
_FILL_CAND = "#d7f0d7"  # V1-epic-only
_FILL_ANCH = "#f8d2d2"  # anchor-only
_EDGE_BOTH = "#999999"
_EDGE_CAND = "#2f8f2f"
_EDGE_ANCH = "#c23b3b"


def _rulegraph_dot(cwd: Path, target: str) -> str:
    """Return the dot text of ``snakemake --rulegraph`` run in ``cwd``.

    Gurobi/pulp preamble and snakemake config chatter pollute stdout, so the
    dot source is extracted from the first line starting with ``digraph``.
    """
    cp = subprocess.run(
        ["uv", "run", "snakemake", "--rulegraph", target, "--configfile", "config/config.equivalence.yaml"],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
    )
    m = re.search(r"^digraph", cp.stdout, flags=re.MULTILINE)
    if not m:
        raise RuntimeError(
            f"no 'digraph' in rulegraph output (cwd={cwd}, rc={cp.returncode}): "
            f"stdout[:300]={cp.stdout[:300]!r} stderr[:300]={cp.stderr[:300]!r}",
        )
    return cp.stdout[m.start() :]


def _parse(dot: str) -> tuple[set[str], set[tuple[str, str]]]:
    """Rule-name-level (nodes, edges) from one side's dot text."""
    ids = {m.group(1): m.group(2) for m in _NODE_RE.finditer(dot)}
    edges = {(ids[a], ids[b]) for a, b in _EDGE_RE.findall(dot) if a in ids and b in ids}
    return set(ids.values()), edges


def _walltimes(ctx: dict) -> dict:
    """Per-rule walltimes {rule: {'candidate': s, 'anchor': s}}, best effort.

    The ``benchmarks`` module is being written concurrently, so both the
    import and the call are guarded; fallback is no time annotation.
    """
    try:
        from .benchmarks import rule_walltimes
    except Exception:
        return {}
    for args in ((ctx,), ()):
        try:
            times = rule_walltimes(*args)
        except TypeError:
            continue
        except Exception:
            return {}
        return times if isinstance(times, dict) else {}
    return {}


def _node_label(rule: str, times: dict, labels: dict) -> str:
    r"""Rule name, plus a second line of per-side walltimes when known.

    Uses the two-character DOT escape ``\n`` (passed through verbatim by
    python-graphviz) for the line break.
    """
    t = times.get(rule)
    if not isinstance(t, dict):
        return rule
    parts = []
    for side in ("candidate", "anchor"):
        v = t.get(side)
        if isinstance(v, int | float):
            parts.append(f"{labels[side]} {v:.0f}s")
    if not parts:
        return rule
    return rule + "\\n" + " | ".join(parts)


def _union_svg(
    cand: tuple[set[str], set[tuple[str, str]]],
    anch: tuple[set[str], set[tuple[str, str]]],
    times: dict,
    labels: dict,
) -> str:
    import graphviz

    cand_nodes, cand_edges = cand
    anch_nodes, anch_edges = anch
    g = graphviz.Digraph("diff_dag")
    g.attr(rankdir="LR", bgcolor="white", margin="0")
    g.attr("node", shape="box", style="rounded,filled", fontname="sans-serif", fontsize="10", color="#666666")
    g.attr("edge", penwidth="1.4")
    for rule in sorted(cand_nodes | anch_nodes):
        if rule in cand_nodes and rule in anch_nodes:
            fill = _FILL_BOTH
        elif rule in cand_nodes:
            fill = _FILL_CAND
        else:
            fill = _FILL_ANCH
        g.node(rule, label=_node_label(rule, times, labels), fillcolor=fill)
    for a, b in sorted(cand_edges | anch_edges):
        if (a, b) in cand_edges and (a, b) in anch_edges:
            color = _EDGE_BOTH
        elif (a, b) in cand_edges:
            color = _EDGE_CAND
        else:
            color = _EDGE_ANCH
        g.edge(a, b, color=color)
    svg = g.pipe(format="svg").decode()
    return svg[svg.find("<svg") :]


def _diff_paragraph(cand_nodes: set[str], anch_nodes: set[str], labels: dict) -> str:
    """Plain-language summary auto-derived from the node diff."""
    cand_only = sorted(cand_nodes - anch_nodes)
    anch_only = sorted(anch_nodes - cand_nodes)
    shared = len(cand_nodes & anch_nodes)

    def _names(rules: list[str]) -> str:
        return ", ".join(f"<code>{html.escape(r)}</code>" for r in rules)

    bits = [f"The two pipelines share {shared} rules."]
    if cand_only:
        bits.append(
            f"Rules that exist only in {html.escape(labels['candidate'])}: {_names(cand_only)}.",
        )
    else:
        bits.append(f"No rules are unique to {html.escape(labels['candidate'])}.")
    if anch_only:
        bits.append(
            f"Rules that exist only in the {html.escape(labels['anchor'])}: {_names(anch_only)}.",
        )
    else:
        bits.append(f"No rules are unique to the {html.escape(labels['anchor'])}.")
    return "<p>" + " ".join(bits) + "</p>"


def render(ctx: dict) -> str:
    labels = ctx["labels"]
    cand_label = html.escape(labels["candidate"])
    anch_label = html.escape(labels["anchor"])
    head = "<h2>Pipeline structure: unified diff-DAG</h2>"
    lead = (
        "<p>Both sides build the same product — a solved western-interconnect "
        "network — but they get there through differently shaped pipelines. "
        "This diagram overlays the two Snakemake rule graphs (prong 1, left "
        "to right in build order) so the structural difference is visible at "
        "a glance: it is the <em>simplify-early reorder</em>, where "
        f"{cand_label} aggregates and pre-clusters the network topology "
        "before the per-bus heavy steps (demand, renewable profiles, "
        f"electricity assembly), while the {anch_label} runs those steps at "
        "substation granularity and simplifies afterwards.</p>"
        "<p><strong>Legend:</strong> "
        f'<span class="badge na">gray</span> rules and gray arrows appear on '
        "both sides; "
        f'<span class="badge pass">green</span> rules and green arrows exist '
        f"only in {cand_label}; "
        f'<span class="badge fail">red</span> rules and red arrows exist only '
        f"in the {anch_label}. Where available, each rule shows its measured "
        "walltime per side on a second line.</p>"
    )
    try:
        cand_dot = _rulegraph_dot(ctx["cand_root"], final_target(1))
        anch_dot = _rulegraph_dot(ctx["anch_root"], anchor_final_target(1))
    except Exception as exc:  # rulegraph extraction failed — nothing to draw
        return (
            head + lead + "<p>Sorry — the diff-DAG could not be generated because the "
            "rulegraph extraction failed: "
            f"<code>{html.escape(str(exc))}</code></p>"
        )
    cand = _parse(cand_dot)
    anch = _parse(anch_dot)
    try:
        body = '<div style="overflow-x:auto">' + _union_svg(cand, anch, _walltimes(ctx), labels) + "</div>"
    except Exception:  # graphviz rendering failed — fall back to raw dot
        body = (
            "<p>Sorry — graphviz could not render the combined graph, so the "
            "raw rule graphs of each side are shown instead.</p>"
            f"<h3>{cand_label} rulegraph (dot)</h3><pre>{html.escape(cand_dot)}</pre>"
            f"<h3>{anch_label} rulegraph (dot)</h3><pre>{html.escape(anch_dot)}</pre>"
        )
    return head + lead + body + _diff_paragraph(cand[0], anch[0], labels)
