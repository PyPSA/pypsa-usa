"""Loader for the hot-fix registry (``hotfixes.yaml``).

The registry is the machine-readable extract of
``memory/plans/hotfix-ledger.md`` in the project brain: one row per change on
``develop`` that is not part of the core restructuring and that could by itself
move a benchmark number. It answers exactly one question — *is this hot-fix a
legitimate explanation for a master-vs-develop difference?*

The load-bearing rule is :func:`explains`. A hot-fix that has been **ported to
``master-benchmark``** runs on BOTH sides, so it cannot produce a difference
between them. Citing a ported id as the explanation for a residual difference
hides a broken port behind a plausible story, so the registry rejects it. The
same goes for a row that is a no-op under the standing whole-USA config: it is
not wrong, it just does nothing there.

This module deliberately knows nothing about tables, figures or findings; it is
imported by the comparison table, the waiver checks and the branch-policy tests
alike.
"""

from __future__ import annotations

from pathlib import Path

import yaml

HOTFIXES_PATH = Path(__file__).parent / "hotfixes.yaml"

CONFIDENCES = ("high", "medium", "low", "none")
REQUIRED_FIELDS = ("id", "commit", "pr", "title", "confidence", "ported", "usa_noop")


def load_hotfixes(path: Path | None = None) -> dict[str, dict]:
    """``{'HF-8': {...}, ...}``; a missing file yields ``{}``.

    Missing is not an error: a checkout that has not landed the registry yet
    should degrade to "no id resolves", not crash the comparison.
    """
    p = HOTFIXES_PATH if path is None else Path(path)
    if not p.exists():
        return {}
    rows = yaml.safe_load(p.read_text()) or []
    return {str(row["id"]): dict(row) for row in rows}


def ported_ids(registry: dict[str, dict] | None = None) -> set[str]:
    """Ids already carried by the baseline, i.e. running on both sides."""
    reg = load_hotfixes() if registry is None else registry
    return {hid for hid, row in reg.items() if row.get("ported")}


def explains(hotfix_id: str, registry: dict[str, dict] | None = None) -> tuple[bool, str]:
    """Say whether ``hotfix_id`` is an admissible explanation for a difference.

    Returns ``(ok, reason)``. ``reason`` is empty when ``ok`` is true and
    otherwise says why the id was rejected, in words that belong in a table
    cell.
    """
    reg = load_hotfixes() if registry is None else registry
    row = reg.get(hotfix_id)
    if row is None:
        return False, f"{hotfix_id} has no row in {HOTFIXES_PATH.name}"
    if row.get("ported"):
        sha = row.get("ported_sha") or "master-benchmark"
        return False, (
            f"{hotfix_id} is ported to master-benchmark ({sha}), so BOTH sides run it; "
            "a difference tracing to it means the port is broken, not that the "
            "difference is explained"
        )
    if row.get("usa_noop"):
        return False, (
            f"{hotfix_id} is a no-op under the standing whole-USA config, so it cannot "
            "have moved anything here; the ledger says not to chase it whatever its "
            "confidence"
        )
    return True, ""


def live_code_differences(registry: dict[str, dict] | None = None) -> list[dict]:
    """Rows that are a genuine code difference on the standing USA case.

    Not ported (so develop-only) and not a no-op under ``interconnect: usa``.
    This is the candidate-explanation set recorded in ``run_meta.json`` under
    ``known_code_differences``, so a reader of one run's metadata sees the
    suspects without opening the ledger.
    """
    reg = load_hotfixes() if registry is None else registry
    out = []
    for hid, row in sorted(reg.items(), key=lambda kv: int(kv[0].split("-")[1])):
        if row.get("ported") or row.get("usa_noop"):
            continue
        out.append(
            {
                "key": row.get("title", ""),
                "master": row.get("master", ""),
                "develop": row.get("develop", ""),
                "hotfix": hid,
            },
        )
    return out
