"""Explanatory equivalence report — section modules.

Contract: every section module exposes ``render(ctx) -> str`` returning an
HTML fragment. ``ctx`` is the dict built in ``tests/equivalence/report.py``:

- ctx["repo"], ctx["cand_root"], ctx["anch_root"]: Paths (candidate root is
  the main checkout workflow/, anchor root the worktree workflow/)
- ctx["prongs"]: [1, 2]; ctx["findings"][prong]: parsed findings json
- ctx["manifests"]["candidate"|"anchor"]: parsed manifest json
- ctx["waivers"]: parsed waivers.yaml list
- ctx["ledger_rows"]: parsed deltas-ledger table rows (dicts with id, stages,
  delta, cause, why, signoff)
- ctx["labels"]: {"candidate": "V1-epic", "anchor": "anchor"} — ALL
  user-facing text must go through these (internal keys stay 'candidate').
- helpers: ctx["png"](fig) -> data URI; ctx["img"](fig, caption) -> html;
  ctx["load_network"](path); ctx["norm_label"](x); ctx["load_regions"](side,
  stage_key) -> GeoDataFrame indexed by normalized bus name.

Sections are assembled in report.py build_report() in this order:
summary, dag, stages (per prong), maps, benchmarks.
"""
