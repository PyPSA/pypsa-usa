# PyPSA Network Schema Tracking — Design

**Date:** 2026-05-21
**Status:** Approved (brainstorming complete)
**Author:** ktehranchi (with Claude)

## Motivation

The PyPSA-USA workflow transforms a `pypsa.Network` through ~10 sequential
scripts (`build_base_network` → `aggregate_to_substations` → `cluster_simpl`
→ `cluster_network` → `add_electricity` → ...). Each script may add, drop,
or transform columns on PyPSA components (Bus, Generator, Line, Link,
Load, StorageUnit, Transformer).

There is no single place that documents which custom columns exist, who
adds them, who consumes them, or how they should be reduced during
aggregation/clustering. The recurring failure mode is:

> `AssertionError: In Bus cluster X, the values of attribute Y do not
> agree` — raised by `pypsa.clustering.spatial.consense` when a custom
> column that the author of the aggregation script didn't know about
> contains differing or partially-NaN values within a cluster.

The most recent instance: `LAF_state` (added by `build_base_network`,
populated only on buses with `Pd`, NaN elsewhere) is not registered in
the `bus_strategies` dict of `aggregate_to_substations`, so PyPSA falls
back to `consense` and the rule crashes.

The class of bugs is wider than this one column — any new custom column
added upstream of an aggregation step is a latent crash waiting for the
first cluster where its values aren't all-equal-or-all-NaN.

## Goals

1. Give a human reader a single document listing every custom column on
   every PyPSA component, who creates it, who reads it, and how it
   should be aggregated.
2. Give a runtime log showing what columns are present on entry and
   what changed by exit of each script that touches a `.nc` network,
   so future bugs of this class are diagnosable from logs alone.
3. Keep the cost trivial — no new dependencies, no per-script schema
   classes to maintain, no risk of breaking the working pipeline.

## Non-Goals

- **No runtime validation / asserts.** Logging only. An opt-in
  `strict=True` mode may be added later, but is out of scope here.
- **No pandera or other schema library.** The catalog is markdown
  maintained by hand.
- **No auto-generation of the catalog from logs.** Bootstrapping the
  initial version is a one-time manual pass; thereafter it is
  curated in PRs.
- **No Snakefile-level wrappers or hooks.** Just helper calls inside
  each script.
- **No fix for the `LAF_state` bug here.** That is a separate
  follow-up that will validate this work (the catalog should make
  the right aggregation strategy obvious; the diff log should show
  the column being silently dropped today).

## Architecture

Two artifacts, decoupled and independently useful:

### 1. `docs/network-schema.md` — the curated catalog

One markdown section per PyPSA component. Each section is a single
table listing every PyPSA-USA-added column. PyPSA built-in attributes
(e.g. `v_nom`, `bus0`, `s_nom`) are *not* listed — the header links
to PyPSA's own component documentation.

Table columns:

| Field         | Meaning |
|---------------|---------|
| Column        | Attribute name as it appears on `n.<component>` |
| dtype         | Pandas dtype (e.g. `float64`, `str`, `int64`) |
| Added by      | Script that first assigns this column |
| Consumed by   | Scripts that read this column downstream |
| Aggregation   | Strategy to register in `bus_strategies` / `generator_strategies` when clustering (`sum`, `mean`, `max`, `first`, `consense`). Default `consense` only when values are guaranteed identical within any cluster. |
| NaN policy    | Whether NaN is allowed at this stage and what it means semantically (e.g. "NaN = offshore bus, no load data") |
| Description   | One-line semantic description |

The catalog is the source of truth that authors of aggregation /
clustering scripts should consult when registering strategies. PR
reviewers can cross-check the catalog against changes that add or drop
columns.

### 2. `_helpers.log_network_schema(n, stage, baseline=None)`

A small helper added to `workflow/scripts/_helpers.py`. Called twice
per script that touches a `.nc` network:

- **Entry** (after `pypsa.Network(...)`): snapshot the column set of
  every non-empty component, log it, and return the snapshot dict so
  the caller can hold onto it for later diffing.
- **Exit** (before `export_to_netcdf`, passing the entry snapshot
  as `baseline`): log the delta — row count change per component and
  added/removed columns versus entry.

### How the two interact

The catalog answers *"what is supposed to be there and what to do with
it?"*. The log answers *"what is actually there right now, and what
just changed?"*. They are deliberately decoupled — logs still work if
the catalog drifts; the catalog is still useful even with logging
disabled.

## Helper API

```python
# In workflow/scripts/_helpers.py


def log_network_schema(
    n: pypsa.Network,
    stage: str,
    baseline: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Log column schema of each PyPSA component on a network.

    Parameters
    ----------
    n : pypsa.Network
        The network to introspect.
    stage : str
        Tag for the log line, e.g. "entry" or "exit". Appears as
        ``[schema <stage>]`` in log lines.
    baseline : dict, optional
        When passed, the function logs the column-set and row-count
        delta versus this baseline rather than the full column list.
        Typically the return value of the entry call.

    Returns
    -------
    dict[str, list[str]]
        Mapping of component name → column list at this moment.
        Pass this back as ``baseline=`` to a later call to get a diff.
    """
```

### Wiring pattern in each script

```python
n = pypsa.Network(snakemake.input.network)
schema_entry = log_network_schema(n, stage="entry")
...  # existing logic
log_network_schema(n, stage="exit", baseline=schema_entry)
n.export_to_netcdf(snakemake.output.network)
```

### Log output

**Entry** — one INFO line per non-empty component:

```
[schema entry] Bus: 12453 rows, 14 cols: ['v_nom','x','y','sub_id','LAF_state',...]
[schema entry] Line: 8901 rows, 9 cols: [...]
```

**Exit** — only logs components whose row count or column set
changed; quiet for unchanged components:

```
[schema exit] Bus: 12453 -> 487 rows
[schema exit] Bus: +cols=['country_agg'], -cols=['sub_id','balancing_area','LAF_state']
[schema exit] Line: 8901 -> 612 rows (no column change)
```

The `[schema <stage>]` prefix is greppable across `logs/`.

### Implementation notes

- Use `n.iterate_components()` to walk every PyPSA component generically
  — no hard-coded list of components, future-proof against PyPSA additions.
- Skip empty components (e.g. `Transformer` after `remove_transformers`).
- Sort column lists before logging so diffs are deterministic.
- No `n.copy()` needed — only the column names and row count are
  snapshotted, which is a tiny dict of lists.

## Scripts to instrument

Every script that calls `pypsa.Network(...)` *and* `export_to_netcdf`.
The candidate list, to be confirmed by `grep` during implementation:

- `build_base_network.py`
- `aggregate_to_substations.py`
- `cluster_simpl.py`
- `cluster_network.py`
- `add_electricity.py`
- `add_extra_components.py`
- `add_demand.py`
- `prepare_network.py`
- `add_sectors.py` (and any sector-specific writers found on sweep)

Scripts that only *read* a network for plotting / reporting are
out of scope — they don't mutate state, so a schema diff is
uninteresting.

## Initial catalog population

One-time manual pass after the helper is in place:

1. Run the `test_small` config (`western` interconnect) end-to-end
   with logging on.
2. Read each `[schema entry]` and `[schema exit]` line to enumerate
   columns added/removed at each stage.
3. For each custom column observed, fill in the catalog row by
   reading the script that adds it (dtype from the assignment,
   description from the surrounding logic, aggregation strategy
   from how downstream clustering currently handles it — or, where
   missing, from first principles).
4. Commit the seeded catalog.

Subsequent maintenance is per-PR: any PR that adds a custom column
must also add a row to the catalog; any PR that touches an aggregation
script should reference the catalog when adding `bus_strategies`
entries.

## Testing

- **Smoke test:** run the `test_small` config end-to-end (the same
  config that currently exposes the `LAF_state` bug). Confirm:
  - Every `.nc`-touching rule emits at least one `[schema entry]`
    and one `[schema exit]` line in its log file under `logs/`.
  - The `aggregate_to_substations` log diff explicitly shows
    `LAF_state` being dropped — this is the canary that proves the
    system would have surfaced the original bug at review time.
- No regression test required; behavior of the pipeline is
  unchanged (logging is additive).

## Rollout

1. Add `log_network_schema` to `_helpers.py`.
2. Wire entry/exit calls into each script on the instrumented list.
   Default: a single PR covering all instrumented scripts plus the
   seeded catalog. Split into per-chunk PRs (topology / add-* /
   sectors) only if the diff is too large to review in one pass.
3. Seed `docs/network-schema.md` from observed logs (step 2 of the
   "Initial catalog population" section above) before merging the
   instrumentation PR — that way the catalog lands together with
   the helper that justifies it.
4. Open a separate PR to fix `LAF_state` using the new catalog as
   the authority on the correct aggregation strategy.

## Risks & open questions

- **Catalog drift.** If contributors add custom columns without
  updating the catalog, the catalog grows stale. Mitigation: the
  helper logs the actual columns at every run, so drift is visible
  to anyone reading a log. A future enhancement could automatically
  diff observed columns against the catalog and warn on mismatch.
- **Initial catalog completeness.** First pass will likely miss a
  few rarely-touched sector columns. Acceptable — the catalog can
  be extended incrementally as those code paths are exercised.
