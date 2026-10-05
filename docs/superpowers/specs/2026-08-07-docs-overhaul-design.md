# Documentation Overhaul (Power Sector) — Design

**Date:** 2026-08-07
**Status:** Approved (grilling complete) — executing on branch `docs-overhaul` (stacked on `v1-epic`)
**Author:** ktehranchi (with Claude)

## Motivation

Two audits (structure/completeness + staleness-vs-code) of `docs/source/` found:

1. **Actively wrong content.** The `literalinclude` marker convention in
   `config-configuration.md` mis-scopes most sections — the `costs` section renders
   the *imports* config block and the real `costs:` block never renders anywhere.
   `workflow/repo_data/config/` and `workflow/config/` have diverged (~200 lines), so
   `init_pypsa_usa.sh` hands fresh clones a different default config than the repo
   shows. `about-usage.md` points at a results filename that is wrong four ways.
   `configtables/electricity.csv` documents PyPSA-Eur carrier names that don't exist
   here. Both DAG images (2025-01) show `add_electricity → simplify_network →
   cluster_network` — the ordering the simplify-early refactor inverted.
2. **No explanatory model documentation.** No page explains the workflow DAG, PyPSA
   component usage, or the custom constraints (~15 `extra_functionality` families;
   five documented nowhere). `data-policies.md` (38 lines of bullets) stands in for
   ~49 KB of constraint code. The front door (`about-introduction.md`) ends in a
   cookiecutter placeholder folder tree. A per-rule "Rules" docs section existed
   historically and was deleted.
3. **Images are absent exactly where users start.** 11 of 21 pages have zero images —
   all power-sector. Sector pages average 7 figures. Orphans exist (`pop_layout/*`,
   CCTS SVGs linked as the word "this"), and ~45 raw-HTML `<img>` figures are
   silently dropped from the PDF build RTD requests.

Full audit findings are recorded in the session transcript; the actionable subset is
inlined in the phase plans below.

## Decision ledger

| ID | Decision | Choice |
|----|----------|--------|
| D1 | Codebase state documented | Post-v1-epic world; docs land on stacked branch behind PR #22 |
| D2 | Audience priority | (1) researchers/analysts who run it, (2) methodology readers/citers, (3) contributors |
| D3 | Explanatory priority | Conceptual architecture → PyPSA-USA-specific formulation → data provenance → results interpretation |
| D4 | Image strategy | Hand-authored diagrams (graphviz/mermaid, versionable) + workflow-generated figures via a dedicated snakemake rule; committed PNGs regenerated on demand; no one-off screenshots |
| D5 | Structure | Light restructure: keep existing pages/URLs; add new "Model Description" toctree section; restore workflow/rules documentation |
| D6 | Effort shape | Phased: P0 correctness → P1 explanation+images → P2 formulation+generated figures+guards |
| D7 | Plan artifact | This spec; execution on `docs-overhaul` |
| D8 | Sector docs | **Frozen**: `data-sectors.md`, `data-services.md`, `data-industrial.md`, `data-transportation.md`, `data-naturalgas.md`, `config-sectors.md`; CCTS prose in `config-configuration.md` also frozen (mechanical SVG embedding allowed) |
| D9 | Frozen-page edit policy | Tier (iii): mechanical fixes (typos, figure-directive syntax, broken paths) + factual corrections (dead rule names, stale claims). No prose/structure rewrites |
| D10 | Unpublished references | Publish `network-schema.md` into Model Description; add user-facing release-notes page seeded (edited) from `CHANGELOG-v1-epic.md` |
| D11 | Constraint formulations | New `model-constraints.md`: every power-sector `extra_functionality` constraint with opts/config trigger, LaTeX, source link. Core LP formulation **points to upstream PyPSA docs** — not re-derived here. `data-policies.md` stays slim provenance, links over |
| D12 | Branch mechanics | `docs-overhaul` stacked on `v1-epic`; PR to `develop` after PR #22 merges |
| D13 | EGS | Reference-only: config keys/carrier documented in tables; methodology deferred to forthcoming paper |
| D14 | Formulation scope | PyPSA-USA additions only; upstream PyPSA is the source of truth for core math |

## Target information architecture

```
Getting Started        about-introduction (rewritten), about-install, about-usage
Model Description  ←NEW  model-workflow        (DAG narrative, rule→script→output table, rendered rulegraph)
                         model-components      (PyPSA component primer as used here; spatial/temporal levels)
                         model-constraints     (custom constraints & opts reference; P2)
                         model-network-schema  (moved from docs/network-schema.md)
Model Data             data-* (power pages refreshed; sector pages frozen)
Model Configuration    config-* (reference fixed; sectors frozen)
Reference              release-notes ←NEW, license, contributing, publications
```

## Phase P0 — Correctness (no published statement is false)

Code-side:
- **Marker convention fix**: in `workflow/repo_data/config/*.yaml`, every documented
  top-level key gets a `# docs : <name>` marker directly above it; every
  `literalinclude` switches to `:start-after: # docs : <name>` + `:end-before: # docs :`.
  New test `tests/docs/` parses each directive, slices the YAML, and asserts the
  rendered top-level keys match the section — includes can never silently mis-scope again.
- **Config-tree sync**: `workflow/config/` regenerated from `workflow/repo_data/config/`;
  test asserts equality so drift fails CI.
- **DAG regeneration**: `snakemake --rulegraph` → refresh `workflow/repo_data/dag.jpg`
  (+ committed `.dot`); swap stale `_static/dag_sector.jpg` (image-file replacement only —
  frozen page untouched).
- `config.cluster.yaml`: replace dead `simplify_network` walltime key with
  `aggregate_to_substations`/`cluster_simpl`/`aggregate_egs` entries.

Docs-side (all findings from the staleness audit):
- `config-configuration.md`: re-scope every include; add missing sections (`enable`,
  `model_topology`, `ucap`, `offshore_*`, `pudl_path`, `renewable.EGS` per D13,
  `walltime`, `custom_files`, `renewable_weather_years/_scenarios/_snapshots`);
  delete dead `load` section + `configtables/load.csv`; `sector` section points to
  config-sectors page.
- `config-wildcards.md`: `simplify_network` → `aggregate_to_substations`+`cluster_simpl`;
  remove phantom `{cutout}` wildcard; document `m`/`a`/`c` suffixes (align with
  config-spatial) and `{ll}` `c` variant; `{sector}` empty-string normalization.
- `configtables/electricity.csv`: real carrier names (`offwind`, `offwind_floating`),
  `StorageUnit` key + real storage carriers, `Link` singular, powerplants path, EGS row,
  missing policy/limit rows. `costs.csv`: drop nonexistent PyPSA-Eur keys, add real ones.
  `clustering.csv`: temporal block, `exclude_carriers`, the `clustering.simplify_network`
  naming-trap note. `solving.csv`, `solar.csv`, `onwind.csv`, `atlite.csv`: reconcile
  with live config. `opts.csv`: complete triggers (`Co2L`, `Ept`, carrier`+p/e/c/m`),
  fix pypsa-eur permalinks (with model-constraints, P2 content allowed to land early).
- `about-usage.md`: correct results filename (add `{simpl}`, `c`, `{sector}`, RDIR) +
  decoder; add `--configfile` to the troubleshooting command; drop the phantom
  tutorial-notebook promise.
- `about-install.md`: state Python `>=3.11,<3.12`; add `uv sync`; solver/config
  alignment notes; load-bearing pins mentioned.
- `data-generators.md`: godeeep is the default; describe the simpl-resolution profile
  pipeline (not "41,564 zones then cluster"); `profile_{tech}_s{simpl}.nc`; fix `# Data`
  H1 bug. `data-demand.md`: add `eer` source + correct horizons. `data-transmission.md`:
  clustering-algorithm claim matches kmeans/modularity reality.
- `config-spatial.md`: mechanism prose updated for simplify-early; `topological_boundaries`
  full option list. `README.md`: minimal correct quickstart (`cd workflow`, `--configfile`).
- Typo sweep (power pages fully; frozen pages word-level only per D9).
- Orphan cleanup: `configtables/emissions.csv`, `configtables/load.csv`,
  `datatables/sector_natural_gas.csv`, `_static/WECC.jpg`; move `pypsa-usa.drawio`
  out of `_static/`.

## Phase P1 — Explain the model (+ images)

- **`model-workflow.md`** (new): stage-by-stage narrative of
  `build_shapes → build_base_network → build_bus_regions → aggregate_to_substations →
  cluster_simpl → profiles/demand → add_demand → add_electricity → cluster_network →
  add_extra_components → prepare_network → (add_sectors) → solve_network`; a
  rule → script → key inputs → outputs table; rendered rulegraph via graphviz; a
  hand-authored simplified pipeline schematic.
- **`model-components.md`** (new): PyPSA component primer (Bus/Carrier/Generator/Link/
  Line/Store/StorageUnit/Load *as used in PyPSA-USA*), pointing to upstream PyPSA docs
  for semantics (D14); spatial hierarchy (interconnect → ReEDS/county → `{simpl}` →
  `{clusters}`) and temporal structure (snapshots, horizons, foresight).
- **`model-network-schema.md`**: moved from `docs/network-schema.md`, toctree'd.
- **`release-notes.md`** (new): user-facing v1-epic summary (DAG reorder, category-first
  resources/, HAC removal, harness-caught fixes) — edited, not verbatim.
- **`about-introduction.md` rewrite**: what PyPSA-USA is/does (CEM vs PCM, spatial and
  temporal scope), real repo tree replacing the cookiecutter placeholder, local DAG
  image (no more `master` hotlink).
- **Image work**: embed the 11 CCTS SVGs as real figures (mechanical); wire the six
  orphaned `pop_layout` figures into a new population-disaggregation subsection of
  `data-demand.md`; authored diagrams for the pipeline schematic and spatial hierarchy;
  convert raw `<img>` → `{figure}` everywhere (PDF-safe; frozen pages mechanical-only);
  downsample >500 KB PNGs.

## Phase P2 — Formulation, generated figures, guards

- **`model-constraints.md`** (new): one entry per power-sector custom constraint —
  RPS/CES, regional CO2 limits, ERM/SAFE reserve margins, TCT, land-use,
  interchange/imports, operational reserves, demand response (power side), fossil
  limits, `Co2L`/`Ep`/`Ept`, carrier`+p/e/c/m` adjustments — each with trigger
  (opts wildcard / config key), LaTeX formulation, and `file:line` source link.
  Sector-coupling constraints listed but deferred to their (frozen) pages.
  Core objective/power-balance/KVL → link to PyPSA optimization docs.
- **`data-policies.md`**: slim to provenance; link each policy dataset to its
  constraint entry.
- **Generated-figure pipeline**: `docs_figures` snakemake rule + script rendering
  canonical figures (base vs clustered network map, ReEDS zones, capacity/dispatch
  example) from the tutorial config into `docs/source/_static/generated/`; committed,
  regenerated on demand.
- **Build hygiene**: prune `docs/requirements.txt` (drop pypsa/atlite/cartopy/dask/
  snakemake/sklearn/plotly + unused pydata theme; pin the rest; add sphinx-copybutton);
  `conf.py` — remove vestigial autodoc/napoleon/sys.path, set `intersphinx_mapping`
  (pypsa, atlite), keep graphviz; `.readthedocs.yaml` — `build.apt_packages:
  [graphviz, imagemagick]`; lightweight docs-build CI job.

## Acceptance

- P0: every BREAKS-USER/MISLEADING audit finding resolved or explicitly waived here;
  marker test + config-sync test green.
- P1: Model Description section live; zero-image power pages ≤ 4; all figures render
  in HTML and PDF builds.
- P2: every power-sector `extra_functionality` constraint has an entry; `opts.csv`
  complete against `_helpers.py`; `docs_figures` runs from the tutorial config;
  `sphinx-build` completes warning-clean enough to enable CI.

## Risks / sequencing

- Docs describe the post-refactor world (D1): they are correct for `v1-epic`+`develop`
  after PR #22, and intentionally ahead of upstream `master` until the epic propagates.
  Official RTD reflects them only when upstream syncs.
- Frozen sector pages may retain known cosmetic issues (empty Validation stubs) —
  flagged, not fixed (D8/D9). Sector DAG image is swapped as a file (allowed).
- `literalinclude` markers remain convention-based; the new test is the guard.
