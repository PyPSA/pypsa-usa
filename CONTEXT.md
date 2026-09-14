# CONTEXT.md — Ubiquitous language for the pipeline-evolution work

Glossary of terms used across the specs, plans, tests, and change-log for the
pipeline speed/memory work on `develop`. Terms only — no implementation detail.

## Equivalence testing

The four branch/change terms below are copied **verbatim** from §2 of
`memory/PROJECT.md` in the project brain (`Refactor_PyPSA_USA`), which is their
one home. Change them there first; the copy here exists so the repo is
self-describing, and the two must not drift.

- **`master`** — the released pypsa-usa branch; the baseline ("the old
  version") every comparison is made against.
- **`master-benchmark`** — a branch off `master` in `code/pypsa-usa` that is
  the actual baseline run. It carries the fixes `master` needs to run and
  compare (formerly patched in at build time), plus resource tweaks (memory,
  walltime) so benchmarking is seamless. Every commit on it is listed in the
  hot-fix ledger as "ported to master-benchmark", so a difference it removes is
  never mistaken for a `develop` effect. Decided 2026-09-14.
- **`develop`** — the refactored branch under test. It moves: hot-fixes land
  on it while benchmarking is in progress, so a benchmark always records the
  exact `develop` commit it ran.
- **Hot-fix** — a change on `develop` relative to `master` that is *not* part
  of the core restructuring of the snakemake workflow (a data fix, a bug fix,
  an improved default). Every hot-fix is listed in the hot-fix ledger (§3.3)
  so it can be cited when explaining a `master`/`develop` difference.

Harness terms, local to this repository:

- **Baseline build** — the set of pipeline artifacts produced by building
  `master-benchmark` in `.worktrees/master-benchmark`, detached at the sha
  `EQ_BASELINE_REF` resolves to.
- **Develop build** — the artifacts produced by the tip of `develop` (or a PR
  branch) under the same configuration and data as the baseline build.
- **Equivalence run** — one baseline build plus one develop build plus the
  comparison between them, producing a pass/fail result and PNG figures.
- **Config-only determinism** — the principle that both sides of an
  equivalence run produce identical clustering from the same basic
  configuration options alone (shared, seeded clustering code), with no
  injected fixtures and no build-time patching of either side.
- **Delta** — any difference between the baseline and develop artifacts that
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
  HPC at milestones. The standing benchmark case.
- **Tier A / Tier B / Tier C** — the test pyramid: static checks (no data),
  small integration build with artifact-shape assertions, and equivalence runs
  against the baseline, respectively.

## Pipeline

- **Simplify-early refactor** — the `develop` restructuring that moved
  substation aggregation and `{simpl}` clustering ahead of the per-bus heavy
  rules (renewable profiles, demand, electricity assembly).
- **Pass-through (`simpl=""`)** — running the pipeline with no `{simpl}`
  reduction, so the network entering final clustering is at substation
  granularity. Both `master` and `develop` support this.
- **Change-log** — the running, human-readable record of every code/behavior
  change on `develop` relative to `master` (distinct from the deltas ledger,
  which records accepted *result* differences).
