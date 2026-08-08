# CONTEXT.md — Ubiquitous language for the v1-epic pipeline-evolution work

Glossary of terms used across the specs, plans, tests, and change-log for the
pipeline speed/memory work on `v1-epic`. Terms only — no implementation detail.

## Equivalence testing

- **Anchor** — the pinned `upstream/develop` commit that model results are
  measured against. There is exactly one current anchor at any time; moving it
  is an explicit, recorded decision.
- **Baseline build** — the set of pipeline artifacts produced by running the
  anchor commit.
- **Candidate build** — the artifacts produced by the tip of `v1-epic` (or a PR
  branch) under the same configuration and data as the baseline build.
- **Equivalence run** — one baseline build plus one candidate build plus the
  comparison between them, producing a pass/fail result and a visual report.
- **Config-only determinism** — the principle that both sides of an
  equivalence run produce identical clustering from the same basic
  configuration options alone (shared, seeded clustering code), with no
  injected fixtures and no patches to the anchor.
- **Delta** — any difference between baseline and candidate artifacts that
  exceeds tolerance.
- **Waiver** — a machine-readable annotation that tells the comparison to
  accept one specific, already-signed-off delta.
- **Deltas ledger** — the human-readable, signed record of every accepted
  delta: what differs, why, and who approved it. Every waiver must have a
  ledger entry; every ledger entry must have a waiver.
- **Aggregate invariant** — a quantity that must not change regardless of how
  buses are grouped (e.g. annual demand per state, capacity per carrier per
  zone). Used to validate clustering changes that cannot be pinned.

## Harnesses and tests

- **CA harness** — the small, frequently-run equivalence harness: a
  California-only slice, two weeks of January data. Runs locally in minutes.
- **USA harness** — the full-CONUS equivalence harness, run infrequently on
  HPC at milestones.
- **Tier A / Tier B / Tier C** — the test pyramid: static checks (no data),
  small integration build with artifact-shape assertions, and equivalence runs
  against the anchor, respectively.

## Pipeline

- **Simplify-early refactor** — the v1-epic restructuring that moved substation
  aggregation and `{simpl}` clustering ahead of the per-bus heavy rules
  (renewable profiles, demand, electricity assembly).
- **Pass-through (`simpl=""`)** — running the pipeline with no `{simpl}`
  reduction, so the network entering final clustering is at substation
  granularity. Both the anchor and `v1-epic` support this.
- **Change-log** — the running, human-readable record of every code/behavior
  change on `v1-epic` relative to `develop` (distinct from the deltas ledger,
  which records accepted *result* differences).
