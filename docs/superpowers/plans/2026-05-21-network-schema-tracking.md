# Network Schema Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `log_network_schema` helper and a curated `docs/network-schema.md` catalog so PyPSA column changes across the workflow are diagnosable from logs and reviewable from a single document.

**Architecture:** Two artifacts. (1) A pure helper in `workflow/scripts/_helpers.py` that snapshots component column sets and logs entry/exit diffs. (2) A markdown catalog seeded by one smoke-test pass and maintained by hand in PRs. Helper calls are wired into the 10 scripts that read or write a `.nc` network. No runtime asserts; logging only.

**Tech Stack:** Python 3.11, PyPSA, pandas, pytest (existing test infra under `workflow/scripts/test/`), Snakemake.

**Spec:** `docs/superpowers/specs/2026-05-21-network-schema-tracking-design.md` (commit `dfbea9bc`).

**Scripts to instrument (definitive list, from `grep export_to_netcdf`):**
- `build_base_network.py` (write-only → exit log only)
- `aggregate_to_substations.py` (read + write)
- `cluster_simpl.py` (read + write)
- `cluster_network.py` (read + write)
- `add_electricity.py` (read + write)
- `add_extra_components.py` (read + write)
- `add_demand.py` (read + write)
- `prepare_network.py` (read + write)
- `add_sectors.py` (read + write)
- `solve_network.py` (read + write)

---

## File Structure

**Create:**
- `workflow/scripts/test/test_helpers_schema.py` — pytest unit tests for the helper
- `docs/network-schema.md` — the curated catalog (seeded in the final task)

**Modify:**
- `workflow/scripts/_helpers.py` — add `log_network_schema` function near the top, after the existing logging helpers
- 10 script files (above) — add 1-2 helper calls each around the `pypsa.Network(...)` / `export_to_netcdf` lines

Each script change is tiny (≤4 inserted lines) so the diff is dominated by the helper itself.

---

## Task 1: Add `log_network_schema` helper with unit tests

**Files:**
- Create: `workflow/scripts/test/test_helpers_schema.py`
- Modify: `workflow/scripts/_helpers.py` (insert after `setup_custom_logger` ~line 76)

- [ ] **Step 1: Write the failing tests**

Create `workflow/scripts/test/test_helpers_schema.py`:

```python
"""Tests for log_network_schema in _helpers."""

import logging

import pypsa
import pytest

from _helpers import log_network_schema


@pytest.fixture
def small_network():
    """Tiny network with custom columns on Bus and Line."""
    n = pypsa.Network()
    n.add("Bus", "b0", v_nom=230.0)
    n.add("Bus", "b1", v_nom=230.0)
    n.add("Line", "l0", bus0="b0", bus1="b1", x=0.1, r=0.01, s_nom=100)
    n.buses["custom_col"] = [1.0, 2.0]
    return n


def test_entry_returns_snapshot_with_cols_and_rows(small_network):
    snapshot = log_network_schema(small_network, stage="entry")
    assert "Bus" in snapshot
    assert "Line" in snapshot
    assert "custom_col" in snapshot["Bus"]["cols"]
    assert snapshot["Bus"]["rows"] == 2
    assert snapshot["Line"]["rows"] == 1
    # Column lists should be sorted
    assert snapshot["Bus"]["cols"] == sorted(snapshot["Bus"]["cols"])


def test_entry_logs_one_line_per_nonempty_component(small_network, caplog):
    with caplog.at_level(logging.INFO, logger="_helpers"):
        log_network_schema(small_network, stage="entry")
    messages = [r.message for r in caplog.records if "[schema entry]" in r.message]
    assert any("Bus: 2 rows" in m for m in messages)
    assert any("Line: 1 rows" in m for m in messages)
    # Empty components (Generator, Load, etc.) should NOT log
    assert not any("Generator" in m for m in messages)


def test_exit_with_baseline_logs_column_diff(small_network, caplog):
    baseline = log_network_schema(small_network, stage="entry")
    small_network.buses["new_col"] = [9.0, 9.0]
    small_network.buses = small_network.buses.drop(columns=["custom_col"])
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="_helpers"):
        log_network_schema(small_network, stage="exit", baseline=baseline)
    diff_lines = [r.message for r in caplog.records if "[schema exit]" in r.message]
    bus_diff = next(m for m in diff_lines if "Bus" in m and "cols" in m)
    assert "+cols=['new_col']" in bus_diff
    assert "-cols=['custom_col']" in bus_diff


def test_exit_logs_row_count_change(small_network, caplog):
    baseline = log_network_schema(small_network, stage="entry")
    small_network.remove("Bus", "b1")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="_helpers"):
        log_network_schema(small_network, stage="exit", baseline=baseline)
    diff_lines = [r.message for r in caplog.records if "[schema exit]" in r.message]
    assert any("Bus: 2 -> 1 rows" in m for m in diff_lines)


def test_exit_quiet_for_unchanged_components(small_network, caplog):
    baseline = log_network_schema(small_network, stage="entry")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="_helpers"):
        log_network_schema(small_network, stage="exit", baseline=baseline)
    diff_lines = [r.message for r in caplog.records if "[schema exit]" in r.message]
    assert diff_lines == []
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `cd /Users/kamrantehranchi/Local_Documents/pypsa-usa && uv run pytest workflow/scripts/test/test_helpers_schema.py -v`

Expected: ImportError / `cannot import name 'log_network_schema' from '_helpers'` on all 5 tests.

- [ ] **Step 3: Implement the helper**

In `workflow/scripts/_helpers.py`, find the line `def setup_custom_logger(name):` and **after** that function ends (look for the end of its body, around line 76), insert the new function. Verify it lands after `setup_custom_logger` and before `def load_network`:

```python
def log_network_schema(
    n: "pypsa.Network",
    stage: str,
    baseline: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Log column schema of each PyPSA component on a network.

    Call at script entry (stage="entry") right after pypsa.Network(...).
    Capture the return value and pass it as baseline= to a later call
    at script exit (stage="exit") right before export_to_netcdf — this
    emits row-count and column-set deltas instead of full column lists.

    Empty components are skipped. Column lists are sorted for stable
    output. Logging only — no asserts, no behavior change.

    Returns
    -------
    dict[str, dict]
        Mapping of component name -> {"cols": [...], "rows": int}.
        Pass this back as baseline= on the matching exit call.
    """
    snapshot: dict[str, dict] = {}
    for component in n.iterate_components():
        df = component.df
        if df.empty:
            continue
        snapshot[component.name] = {
            "cols": sorted(df.columns.tolist()),
            "rows": len(df),
        }

    if baseline is None:
        for name, info in snapshot.items():
            logger.info(
                "[schema %s] %s: %d rows, %d cols: %s",
                stage,
                name,
                info["rows"],
                len(info["cols"]),
                info["cols"],
            )
        return snapshot

    for name, info in snapshot.items():
        base = baseline.get(name, {"cols": [], "rows": 0})
        added = sorted(set(info["cols"]) - set(base["cols"]))
        removed = sorted(set(base["cols"]) - set(info["cols"]))
        if info["rows"] != base["rows"]:
            logger.info(
                "[schema %s] %s: %d -> %d rows",
                stage,
                name,
                base["rows"],
                info["rows"],
            )
        if added or removed:
            logger.info(
                "[schema %s] %s: +cols=%s, -cols=%s",
                stage,
                name,
                added,
                removed,
            )
    return snapshot
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/kamrantehranchi/Local_Documents/pypsa-usa && uv run pytest workflow/scripts/test/test_helpers_schema.py -v`

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/kamrantehranchi/Local_Documents/pypsa-usa
git add workflow/scripts/_helpers.py workflow/scripts/test/test_helpers_schema.py
git commit -m "$(cat <<'EOF'
Add log_network_schema helper for per-script column tracking

Logs component row count and column set on entry; on exit emits row
and column diffs vs. the entry snapshot. Logging only — no asserts.
Wires into scripts in follow-up tasks; tested in isolation here.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Wire helper into the topology chain

**Files (Modify):**
- `workflow/scripts/build_base_network.py` (exit-only — no input network)
- `workflow/scripts/aggregate_to_substations.py`
- `workflow/scripts/cluster_simpl.py`
- `workflow/scripts/cluster_network.py`

Pattern (read + write scripts):

```python
n = pypsa.Network(snakemake.input.network)
schema_entry = log_network_schema(n, stage="entry")
...  # existing logic unchanged
log_network_schema(n, stage="exit", baseline=schema_entry)
n.export_to_netcdf(snakemake.output.network)
```

Pattern (write-only — `build_base_network.py`):

```python
...  # existing build logic
log_network_schema(n, stage="exit")
n.export_to_netcdf(snakemake.output.network)
```

And add `log_network_schema` to the import line:
```python
from _helpers import configure_logging, log_network_schema
```

- [ ] **Step 1: Wire `build_base_network.py`**

Locate the existing `from _helpers import configure_logging` line. Add `log_network_schema` to it. Find the `n.export_to_netcdf(...)` call near the end of the script and insert `log_network_schema(n, stage="exit")` on the line immediately above it.

Verify the change with:
```bash
grep -n "log_network_schema\|export_to_netcdf" workflow/scripts/build_base_network.py
```
Expected: 3 matches — one import, one exit call, one export call (in that order).

- [ ] **Step 2: Wire `aggregate_to_substations.py`**

Add `log_network_schema` to the `from _helpers import configure_logging` import. In the `if __name__ == "__main__":` block:

Find: `n = pypsa.Network(snakemake.input.network)`
Insert after it: `schema_entry = log_network_schema(n, stage="entry")`

Find: `n.export_to_netcdf(snakemake.output.network)`
Insert before it: `log_network_schema(n, stage="exit", baseline=schema_entry)`

Verify:
```bash
grep -n "log_network_schema\|pypsa.Network\|export_to_netcdf" workflow/scripts/aggregate_to_substations.py
```
Expected: 5 matches in order — import, network read, entry call, exit call, export.

- [ ] **Step 3: Wire `cluster_simpl.py`**

Same pattern as Step 2. Add import, entry call after `pypsa.Network(snakemake.input...)`, exit call before `export_to_netcdf`.

Verify:
```bash
grep -n "log_network_schema\|pypsa.Network\|export_to_netcdf" workflow/scripts/cluster_simpl.py
```
Expected: import + entry/exit pair around each read/write boundary. If the script reads or writes more than once, every entry must pair with the matching exit.

- [ ] **Step 4: Wire `cluster_network.py`**

Same pattern. Same verification grep.

- [ ] **Step 5: Run helper unit tests to confirm no regressions**

```bash
cd /Users/kamrantehranchi/Local_Documents/pypsa-usa
uv run pytest workflow/scripts/test/test_helpers_schema.py -v
```
Expected: all 5 tests still PASS.

- [ ] **Step 6: Smoke-check that wired scripts still parse**

```bash
cd /Users/kamrantehranchi/Local_Documents/pypsa-usa
uv run python -c "import ast; [ast.parse(open(f).read()) for f in ['workflow/scripts/build_base_network.py', 'workflow/scripts/aggregate_to_substations.py', 'workflow/scripts/cluster_simpl.py', 'workflow/scripts/cluster_network.py']]"
```
Expected: no output (success). Any SyntaxError means a wiring step inserted code in the wrong place.

- [ ] **Step 7: Commit**

```bash
git add workflow/scripts/build_base_network.py workflow/scripts/aggregate_to_substations.py workflow/scripts/cluster_simpl.py workflow/scripts/cluster_network.py
git commit -m "$(cat <<'EOF'
Wire log_network_schema into topology chain scripts

Adds entry/exit schema logging to build_base_network,
aggregate_to_substations, cluster_simpl, cluster_network. Logging
only; no behavior change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Wire helper into the add-* chain

**Files (Modify):**
- `workflow/scripts/add_electricity.py`
- `workflow/scripts/add_extra_components.py`
- `workflow/scripts/add_demand.py`

- [ ] **Step 1: Wire `add_electricity.py`**

Same pattern as Task 2 Step 2: add `log_network_schema` to the `_helpers` import; insert entry call after `pypsa.Network(snakemake.input...)`; insert exit call before `export_to_netcdf`.

Verify:
```bash
grep -n "log_network_schema\|pypsa.Network\|export_to_netcdf" workflow/scripts/add_electricity.py
```
Expected: import + entry/exit pair around each read/write boundary.

- [ ] **Step 2: Wire `add_extra_components.py`**

Same pattern. Same verification grep on `workflow/scripts/add_extra_components.py`.

- [ ] **Step 3: Wire `add_demand.py`**

Same pattern. Same verification grep on `workflow/scripts/add_demand.py`.

- [ ] **Step 4: Syntax-check all three**

```bash
cd /Users/kamrantehranchi/Local_Documents/pypsa-usa
uv run python -c "import ast; [ast.parse(open(f).read()) for f in ['workflow/scripts/add_electricity.py', 'workflow/scripts/add_extra_components.py', 'workflow/scripts/add_demand.py']]"
```
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add workflow/scripts/add_electricity.py workflow/scripts/add_extra_components.py workflow/scripts/add_demand.py
git commit -m "$(cat <<'EOF'
Wire log_network_schema into add_electricity / add_extra_components / add_demand

Logging only; no behavior change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Wire helper into prepare / sectors / solve

**Files (Modify):**
- `workflow/scripts/prepare_network.py`
- `workflow/scripts/add_sectors.py`
- `workflow/scripts/solve_network.py`

- [ ] **Step 1: Wire `prepare_network.py`**

Same pattern. Verify:
```bash
grep -n "log_network_schema\|pypsa.Network\|export_to_netcdf" workflow/scripts/prepare_network.py
```

- [ ] **Step 2: Wire `add_sectors.py`**

Same pattern. Same verification grep on `workflow/scripts/add_sectors.py`.

- [ ] **Step 3: Wire `solve_network.py`**

Same pattern. Note: `solve_network` may read and write more than once (e.g. iterating); make sure each `pypsa.Network(...)` read gets an entry call and each `export_to_netcdf` write gets an exit call with the matching baseline. Read the file end-to-end before editing to spot multiple boundaries.

Verify:
```bash
grep -n "log_network_schema\|pypsa.Network\|export_to_netcdf" workflow/scripts/solve_network.py
```
Expected: balanced entry/exit pairs.

- [ ] **Step 4: Syntax-check all three**

```bash
cd /Users/kamrantehranchi/Local_Documents/pypsa-usa
uv run python -c "import ast; [ast.parse(open(f).read()) for f in ['workflow/scripts/prepare_network.py', 'workflow/scripts/add_sectors.py', 'workflow/scripts/solve_network.py']]"
```
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add workflow/scripts/prepare_network.py workflow/scripts/add_sectors.py workflow/scripts/solve_network.py
git commit -m "$(cat <<'EOF'
Wire log_network_schema into prepare_network / add_sectors / solve_network

Logging only; no behavior change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Smoke test on `test_small` config and capture logs

This task validates the wiring end-to-end and produces the raw material for seeding the catalog in Task 6.

The user reported the `LAF_state` crash on `test_small` / western (the same config we'll use here). The crash is *expected* during this smoke test — we want to confirm `[schema entry]` and `[schema exit]` lines appear in the log of every rule that ran successfully *and* in `aggregate_to_substations`'s log up to the point of crash (the entry call comes before the failure).

- [ ] **Step 1: Run the pipeline through `aggregate_to_substations`**

```bash
cd /Users/kamrantehranchi/Local_Documents/pypsa-usa/workflow
uv run snakemake --configfile config/tests/config.test_small.yaml --until aggregate_to_substations -c1 2>&1 | tail -100
```

Expected: same `LAF_state` AssertionError as the user originally hit. This is the canary — we are not fixing it here, we're confirming the schema log fires before the crash.

- [ ] **Step 2: Confirm schema lines appear in logs**

```bash
cd /Users/kamrantehranchi/Local_Documents/pypsa-usa
grep -l "\[schema entry\]\|\[schema exit\]" workflow/logs/**/*.log 2>/dev/null | sort -u
```

Expected: at minimum `logs/build_base_network/*.log` and `logs/aggregate_to_substations/*.log` should appear. The aggregate log should show `[schema entry]` but no `[schema exit]` (because it crashed before the exit call).

- [ ] **Step 3: Capture the column inventory from the build_base_network log**

```bash
cd /Users/kamrantehranchi/Local_Documents/pypsa-usa
grep "\[schema exit\]" workflow/logs/build_base_network/*.log
```

Expected: one line per non-empty component on the freshly-built base network. Save this output — it's the seed list for the Bus / Line / Transformer rows of the catalog in Task 6.

- [ ] **Step 4: Capture the column inventory from aggregate_to_substations entry**

```bash
cd /Users/kamrantehranchi/Local_Documents/pypsa-usa
grep "\[schema entry\]" workflow/logs/aggregate_to_substations/*.log
```

Expected: same columns as the previous step's `[schema exit]` lines (since aggregate's input is build_base's output).

- [ ] **Step 5: Save the captured output to a scratch file**

Write the combined grep output to `/tmp/schema_smoke_observed.txt` so Task 6 can read from it without re-running the pipeline:

```bash
cd /Users/kamrantehranchi/Local_Documents/pypsa-usa
{ grep "\[schema" workflow/logs/build_base_network/*.log;
  grep "\[schema" workflow/logs/aggregate_to_substations/*.log; } > /tmp/schema_smoke_observed.txt
cat /tmp/schema_smoke_observed.txt
```

Expected: a printout of every `[schema ...]` line from those two rules.

- [ ] **Step 6: No commit yet — this task produced runtime artifacts only**

The smoke logs live under `workflow/logs/` which is gitignored. Move to Task 6 with the captured `/tmp/schema_smoke_observed.txt` in hand.

---

## Task 6: Seed `docs/network-schema.md` catalog

**Files (Create):** `docs/network-schema.md`

This task is *manual* in the sense that the engineer must read the source scripts to fill in dtype / aggregation / NaN policy for each observed column. Use `/tmp/schema_smoke_observed.txt` from Task 5 as the inventory of which columns to document.

- [ ] **Step 1: For each custom (non-PyPSA-builtin) column observed in Task 5, find its origin**

For every column in the `build_base_network.py` `[schema exit]` line that is *not* a PyPSA built-in (e.g. `v_nom`, `x`, `y`, `bus0`, `bus1`, `s_nom`, `r`, `x`, `b`, `type`, `length`), find where the column is created:

```bash
cd /Users/kamrantehranchi/Local_Documents/pypsa-usa
# For each suspect column name, grep:
grep -rn "\"COLUMN_NAME\"\|'COLUMN_NAME'\|\.COLUMN_NAME\s*=" workflow/scripts/
```

PyPSA built-in attributes per component are listed at https://pypsa.readthedocs.io/en/latest/user-guide/components.html — keep that page open as a reference. Anything not listed there is a candidate for the catalog.

- [ ] **Step 2: Write `docs/network-schema.md`**

Create the file with this structure. Fill the **Bus** table from observed columns + script inspection; leave Generator/Line/Link/Load/StorageUnit/Transformer as section headers with a `(no custom columns yet — extend as discovered)` placeholder if no custom columns appear in Task 5 output (sector/electricity scripts add their own — those rows get added when those scripts get exercised in a later smoke test).

```markdown
# PyPSA-USA Network Schema

Custom columns added to PyPSA components by PyPSA-USA scripts. PyPSA
built-in attributes are documented at
https://pypsa.readthedocs.io/en/latest/user-guide/components.html and
are not repeated here.

## Conventions

- **Aggregation strategy** is what to register in `bus_strategies` /
  `generator_strategies` when clustering. The default `consense` fails
  if values disagree within a cluster — use it only for columns
  guaranteed identical across any cluster.
- **NaN policy** records whether NaN is allowed at this stage and what
  it means semantically.
- A column appears here only after it has been observed in a
  `[schema ...]` log line during pipeline execution.

## Bus

| Column     | dtype  | Added by             | Consumed by               | Aggregation | NaN policy                          | Description                                       |
|------------|--------|----------------------|---------------------------|-------------|-------------------------------------|---------------------------------------------------|
| sub_id     | int    | build_base_network   | aggregate_to_substations  | first       | never                               | Substation id from Breakthrough Energy topology   |
| LAF_state  | float  | build_base_network   | build_demand              | sum         | NaN = bus has no Pd / offshore      | Load allocation factor within state               |
| (...)      |        |                      |                           |             |                                     | (fill from observed columns)                      |

## Generator

(no custom columns observed yet — extend as smoke tests cover later stages)

## Line

| Column | dtype | Added by | Consumed by | Aggregation | NaN policy | Description |
|--------|-------|----------|-------------|-------------|------------|-------------|
| (...)  |       |          |             |             |            |             |

## Link

(no custom columns observed yet)

## Load

(no custom columns observed yet)

## StorageUnit

(no custom columns observed yet)

## Transformer

(no custom columns observed yet)
```

Replace the `(...)` rows with one row per custom column you found in Step 1. For each row:

- **dtype**: read the assignment line, infer from the right-hand side (`int(...)`, `float`, string literal, etc.)
- **Added by**: the script name from the grep result
- **Consumed by**: grep for read sites — `grep -rn "\.COLUMN_NAME\|\"COLUMN_NAME\"" workflow/scripts/` and filter out the writer
- **Aggregation**: read existing `bus_strategies` / `generator_strategies` in `aggregate_to_substations.py` and `cluster_network.py`. If the column is not in any strategy dict, write `consense` (current behavior) and note in NaN policy whether that's safe
- **NaN policy**: from the assignment context — if it's `df["col"] = X / Y`, NaN propagates from NaN inputs; if it's a constant or string, never NaN
- **Description**: one short line from surrounding comments / variable name

- [ ] **Step 3: Cross-reference: every column in `bus_strategies` in any script must appear in the catalog**

```bash
cd /Users/kamrantehranchi/Local_Documents/pypsa-usa
grep -n "bus_strategies\|generator_strategies" workflow/scripts/aggregate_to_substations.py workflow/scripts/cluster_network.py workflow/scripts/cluster_simpl.py
```

For every column name used as a key in a `*_strategies` dict, verify it has a row in the catalog. If not, add one. This is the consistency check that the catalog covers known aggregation handling.

- [ ] **Step 4: Commit**

```bash
cd /Users/kamrantehranchi/Local_Documents/pypsa-usa
git add docs/network-schema.md
git commit -m "$(cat <<'EOF'
Seed docs/network-schema.md from observed pipeline logs

Initial catalog of custom columns on PyPSA components, populated from
[schema ...] log output of build_base_network and
aggregate_to_substations entry on the test_small western config.
Sector- and electricity-stage columns will be added as those rules
get exercised under the schema logger.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** Tasks 1-4 implement the helper and instrumentation; Task 5 is the smoke test required by the spec; Task 6 seeds the catalog. The deliberate non-goal "no fix for `LAF_state`" is honored — Task 5 acknowledges the crash and proves the canary works, but the fix is a separate PR.
- **Placeholder scan:** No TBDs. The catalog has `(...)` placeholder rows in the template only; Task 6 Step 2 explicitly instructs the engineer to replace them with rows derived from Step 1.
- **Type consistency:** Helper signature `dict[str, dict]` with `{"cols": [...], "rows": int}` is used uniformly across the implementation (Task 1 Step 3), the tests (Task 1 Step 1), and the wiring patterns (Tasks 2-4).
