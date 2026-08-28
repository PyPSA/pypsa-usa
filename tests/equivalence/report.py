"""Explanatory equivalence report (revamped per 2026-08-07 grilling session).

One self-contained ``equivalence_report.html`` telling the whole story:
executive summary (verdict, provenance fingerprint, delta index, stage
timeline, benchmark headline), unified diff-DAG with per-rule walltimes,
stage-ordered narrative per prong, 3-panel maps (V1-epic | anchor | diff),
and a rule-grouped benchmark table.

User-facing naming: "candidate" renders as "V1-epic"; "anchor" stays.
Internal keys remain 'candidate' (manifests, findings, waivers untouched).

Section modules live in ``report_sections/`` and each expose
``render(ctx) -> str`` (see report_sections/__init__.py for the contract).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .compare import load_network, load_waivers
from .paths import CLUSTERS, INTERCONNECT, UNTIL, prong_pairs

REPO = Path(__file__).resolve().parents[2]
ANCHOR_ROOT = REPO / ".worktrees" / "anchor-e7f8bd70" / "workflow"
CAND_ROOT = REPO / "workflow"
RESULTS = CAND_ROOT / "results" / "equivalence"
LEDGER_MD = REPO / "docs" / "superpowers" / "specs" / "2026-08-07-deltas-ledger.md"

LABELS = {"candidate": "V1-epic", "anchor": "anchor"}
_SUF = "" if INTERCONNECT == "western" else f"_{INTERCONNECT}"

# Build-order stage spine (prong-1 stage keys from paths.prong_pairs).
STAGE_ORDER = [
    "demand",
    "profile_onwind",
    "profile_solar",
    "assembled_substation_network",
    "clustered_network",
    "extra_components",
    "prepared_network",
    "sectored_network",
    "solved_network",
]


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _img(fig, title: str) -> str:
    return f'<figure><img src="{_png(fig)}" style="max-width:100%"><figcaption>{title}</figcaption></figure>'


def norm_label(x) -> str:
    s = str(x)
    return s[:-2] if re.fullmatch(r"\d+\.0", s) else s


def load_regions(side: str, simpl: str, clusters: str | None = None):
    """Regions GeoDataFrame indexed by normalized bus/zone name."""
    import geopandas as gpd

    suffix = f"s{simpl}" + (f"_{clusters}" if clusters else "")
    if side == "candidate":
        p = CAND_ROOT / f"resources/equivalence/geospatial/{INTERCONNECT}/regions_onshore_{suffix}.geojson"
    else:
        p = ANCHOR_ROOT / f"resources/equivalence/{INTERCONNECT}/Geospatial/regions_onshore_{suffix}.geojson"
    gdf = gpd.read_file(p)
    gdf["name"] = gdf["name"].map(norm_label)
    return gdf.set_index("name")


def _git_sha(cwd: Path) -> str:
    cp = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return cp.stdout.strip() or "unknown"


def parse_ledger_rows() -> list[dict]:
    rows: list[dict] = []
    if not LEDGER_MD.exists():
        return rows
    for line in LEDGER_MD.read_text().splitlines():
        if not line.startswith("| DL-"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 6:
            rows.append(
                {
                    "id": cells[0],
                    "stages": cells[1],
                    "delta": cells[2],
                    "cause": cells[3],
                    "why": cells[4],
                    "signoff": cells[5],
                },
            )
    return rows


def build_ctx() -> dict:
    findings = {}
    for prong in (1, 2):
        p = RESULTS / f"findings_{prong}{_SUF}.json"
        findings[prong] = json.loads(p.read_text()) if p.exists() else {"findings": []}
    manifests = {}
    for side in ("candidate", "anchor"):
        p = RESULTS / f"manifest_{side}{_SUF}.json"
        manifests[side] = json.loads(p.read_text()) if p.exists() else {}
    cfg = CAND_ROOT / "config" / "config.equivalence.yaml"
    fingerprint = {
        "V1-epic sha": _git_sha(REPO),
        "anchor sha": _git_sha(ANCHOR_ROOT),
        "config sha256": hashlib.sha256(cfg.read_bytes()).hexdigest()[:12] if cfg.exists() else "n/a",
        "clusters": CLUSTERS,
        "interconnect": INTERCONNECT,
        "until": UNTIL,
        "suffix": _SUF,
        "generated": time.strftime("%Y-%m-%d %H:%M"),
    }
    return {
        "repo": REPO,
        "cand_root": CAND_ROOT,
        "anch_root": ANCHOR_ROOT,
        "results": RESULTS,
        "prongs": [1, 2],
        "findings": findings,
        "manifests": manifests,
        "waivers": load_waivers(),
        "ledger_rows": parse_ledger_rows(),
        "labels": LABELS,
        "fingerprint": fingerprint,
        "stage_order": STAGE_ORDER,
        "clusters": CLUSTERS,
        "interconnect": INTERCONNECT,
        "until": UNTIL,
        "suffix": _SUF,
        "prong_pairs": prong_pairs,
        "png": _png,
        "img": _img,
        "load_network": load_network,
        "norm_label": norm_label,
        "load_regions": load_regions,
        "np": np,
        "pd": pd,
        "plt": plt,
    }


CSS = """
body{background:#ffffff;color:#111111;font-family:system-ui;margin:2rem;max-width:1200px}
a{color:#0b5cad}h1,h2,h3{color:#111111}
table{border-collapse:collapse;font-size:12px;background:#ffffff;color:#111111}
td,th{padding:3px 8px;border-bottom:1px solid #ddd;text-align:left;color:#111111}
figure{margin:1rem 0;background:#ffffff}figcaption{color:#555555;font-size:12px}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;margin-right:4px}
.badge.pass{background:#d7f0d7;color:#135e13}.badge.delta{background:#fde8c8;color:#8a5300}
.badge.fail{background:#f8d2d2;color:#8f1111}.badge.na{background:#eeeeee;color:#555555}
.verdict{font-size:15px;padding:10px 14px;border-left:4px solid #ccc;background:#f7f7f7;margin:0.6rem 0}
.stage-strip span{margin-right:6px}
svg{max-width:100%;height:auto;background:#ffffff}
"""


def build_report() -> Path:
    ctx = build_ctx()
    from .report_sections import benchmarks, dag, maps, stages, summary

    parts = [
        summary.render(ctx),
        dag.render(ctx),
        stages.render(ctx),
        maps.render(ctx),
        benchmarks.render(ctx),
    ]
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Equivalence report ({INTERCONNECT}) - V1-epic vs anchor</title>"
        f"<style>{CSS}</style></head><body>" + "".join(parts) + "</body></html>"
    )
    out = RESULTS / f"equivalence_report{_SUF}.html"
    out.write_text(html)
    return out
