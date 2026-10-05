# cluster_simpl county fast-path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `simpl="county"` wildcard value to `cluster_simpl` that bypasses k-means and uses the substation network's county FIPS as a direct busmap, dramatically shrinking topology-aggregation wall time when the target is county resolution.

**Architecture:** Extract two pure helpers into `cluster_simpl.py` — `resolve_simpl_mode(value)` (returns one of `"identity" | "county" | "kmeans"`, raises on unknown) and `build_county_busmap(n)` (builds a `sub_id → "<reeds_zone>_<county>"` Series with a clear error when `county` is missing). Wire them into the script's `__main__` dispatch. The county branch calls `get_clustering_from_busmap` directly with that busmap — no k-means, no solver. All downstream rules consume `{simpl}` opaquely, so no rule-graph changes.

**Tech Stack:** Python 3.11, pypsa 0.30.2, snakemake, pytest, pandas. Tests live under `workflow/scripts/test/` and are run from `workflow/scripts/` via `pytest test/`.

---

## File Structure

- **Modify** `workflow/scripts/cluster_simpl.py` — add two helpers (`resolve_simpl_mode`, `build_county_busmap`), refactor `__main__` to dispatch through them, and add a county branch that calls `get_clustering_from_busmap` directly.
- **Create** `workflow/scripts/test/test_cluster_simpl.py` — unit tests for the two helpers plus a small fixture network.
- **Modify** `docs/source/config-wildcards.md:29-34` — extend the `{simpl}` wildcard section to document the `"county"` value.
- **Modify** `workflow/config/config.default.yaml:13` — comment near `scenario.simpl` listing the recognized values.

No rule files (`workflow/rules/*.smk`) change. No Snakefile change.

---

## Task 1: Extract `resolve_simpl_mode` helper with TDD

**Goal:** A pure function `resolve_simpl_mode(value: str) -> str` returning `"identity"` for `""`, `"county"` for `"county"`, `"kmeans"` for an all-digit string, and raising `ValueError` for anything else.

**Files:**
- Create: `workflow/scripts/test/test_cluster_simpl.py`
- Modify: `workflow/scripts/cluster_simpl.py` (add helper near the top, below imports)

- [ ] **Step 1: Write failing test**

Create `workflow/scripts/test/test_cluster_simpl.py` with:

```python
"""Unit tests for cluster_simpl helpers."""

import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cluster_simpl import resolve_simpl_mode


def test_resolve_simpl_mode_identity():
    assert resolve_simpl_mode("") == "identity"


def test_resolve_simpl_mode_county():
    assert resolve_simpl_mode("county") == "county"


def test_resolve_simpl_mode_kmeans_digits():
    assert resolve_simpl_mode("50") == "kmeans"


def test_resolve_simpl_mode_kmeans_large_digits():
    assert resolve_simpl_mode("2000") == "kmeans"


def test_resolve_simpl_mode_unknown_raises():
    with pytest.raises(ValueError, match="Unknown simpl wildcard"):
        resolve_simpl_mode("foo")


def test_resolve_simpl_mode_unknown_lists_sentinels():
    """Error message must list the recognized values so users can self-correct."""
    with pytest.raises(ValueError) as exc:
        resolve_simpl_mode("bar")
    msg = str(exc.value)
    assert '""' in msg
    assert '"county"' in msg
    assert "digits" in msg or "integer" in msg.lower()
```

- [ ] **Step 2: Run test and verify it fails**

Run: `cd workflow/scripts && pytest test/test_cluster_simpl.py -v`

Expected: All six tests fail with `ImportError: cannot import name 'resolve_simpl_mode' from 'cluster_simpl'`.

- [ ] **Step 3: Implement `resolve_simpl_mode` in `cluster_simpl.py`**

Open `workflow/scripts/cluster_simpl.py`. Find the imports block (lines 22-28). Below the `logger = logging.getLogger(__name__)` line (line 30), insert:

```python
def resolve_simpl_mode(value: str) -> str:
    """Map a `{simpl}` wildcard value to its dispatch branch.

    Returns one of:
      - "identity": pass-through (empty string)
      - "county":   fast-path using county FIPS as busmap
      - "kmeans":   numeric value -> N-cluster k-means

    Raises ValueError for anything else, listing the recognized values.
    """
    if value == "":
        return "identity"
    if value == "county":
        return "county"
    if value.isdigit():
        return "kmeans"
    raise ValueError(
        f"Unknown simpl wildcard value {value!r}. Recognized values are: "
        f'"" (identity pass-through), "county" (county FIPS fast-path), '
        f"or a positive integer (k-means)."
    )
```

- [ ] **Step 4: Run test and verify it passes**

Run: `cd workflow/scripts && pytest test/test_cluster_simpl.py -v`

Expected: All six tests pass.

- [ ] **Step 5: Commit**

```bash
git add workflow/scripts/cluster_simpl.py workflow/scripts/test/test_cluster_simpl.py
git commit -m "Add resolve_simpl_mode dispatch helper to cluster_simpl"
```

If the pre-commit hooks modify either file, re-stage them and re-commit with the same message.

---

## Task 2: Extract `build_county_busmap` helper with TDD

**Goal:** A pure function `build_county_busmap(n) -> pd.Series` that returns a Series indexed by bus IDs (sub_ids) with values `"<reeds_zone>_<county_fips>"`, raising `ValueError` with an actionable message when the `county` column is missing or contains NaN.

**Files:**
- Modify: `workflow/scripts/test/test_cluster_simpl.py` (add fixture + tests)
- Modify: `workflow/scripts/cluster_simpl.py` (add second helper)

- [ ] **Step 1: Write failing tests with a fixture**

Append to `workflow/scripts/test/test_cluster_simpl.py`:

```python
import pandas as pd
import pypsa

from cluster_simpl import build_county_busmap


@pytest.fixture
def substation_network():
    """Tiny 4-bus substation-level network with the columns cluster_simpl expects.

    Mirrors what aggregate_to_substations produces under topological_boundaries='county':
    every bus has reeds_zone, county (FIPS), Pd, LAF_state. No loads or generators
    (cluster_simpl runs before add_electricity).
    """
    n = pypsa.Network()
    n.add("Bus", "s1", x=-122.0, y=37.0, carrier="AC")
    n.add("Bus", "s2", x=-122.1, y=37.1, carrier="AC")
    n.add("Bus", "s3", x=-118.0, y=34.0, carrier="AC")
    n.add("Bus", "s4", x=-118.1, y=34.1, carrier="AC")

    n.buses["country"] = ["06001", "06001", "06037", "06037"]
    n.buses["county"] = ["06001", "06001", "06037", "06037"]
    n.buses["reeds_zone"] = ["p9", "p9", "p10", "p10"]
    n.buses["reeds_state"] = ["CA", "CA", "CA", "CA"]
    n.buses["interconnect"] = "western"
    n.buses["Pd"] = [100.0, 200.0, 150.0, 250.0]
    n.buses["LAF_state"] = [0.25, 0.25, 0.25, 0.25]
    n.buses["substation_lv"] = True

    n.add("Line", "l1", bus0="s1", bus1="s2", x=0.01, r=0.001, s_nom=500)
    n.add("Line", "l2", bus0="s3", bus1="s4", x=0.01, r=0.001, s_nom=500)
    n.add("Line", "l3", bus0="s2", bus1="s3", x=0.05, r=0.005, s_nom=300)

    n.add("Carrier", "AC", co2_emissions=0)
    return n


def test_build_county_busmap_happy_path(substation_network):
    busmap = build_county_busmap(substation_network)
    assert list(busmap.index) == ["s1", "s2", "s3", "s4"]
    assert busmap.tolist() == ["p9_06001", "p9_06001", "p10_06037", "p10_06037"]


def test_build_county_busmap_unique_cluster_count(substation_network):
    busmap = build_county_busmap(substation_network)
    assert busmap.nunique() == 2


def test_build_county_busmap_missing_county_column_raises(substation_network):
    substation_network.buses = substation_network.buses.drop(columns=["county"])
    with pytest.raises(ValueError, match="county"):
        build_county_busmap(substation_network)


def test_build_county_busmap_missing_county_error_mentions_topological_boundaries(
    substation_network,
):
    """Error must steer users toward fixing model_topology.topological_boundaries."""
    substation_network.buses = substation_network.buses.drop(columns=["county"])
    with pytest.raises(ValueError, match="topological_boundaries"):
        build_county_busmap(substation_network)


def test_build_county_busmap_nan_county_raises(substation_network):
    substation_network.buses.loc["s2", "county"] = None
    with pytest.raises(ValueError, match="county"):
        build_county_busmap(substation_network)
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `cd workflow/scripts && pytest test/test_cluster_simpl.py -v`

Expected: The 5 new tests fail with `ImportError: cannot import name 'build_county_busmap' from 'cluster_simpl'`. The 6 tests from Task 1 still pass.

- [ ] **Step 3: Implement `build_county_busmap` in `cluster_simpl.py`**

In `workflow/scripts/cluster_simpl.py`, immediately below the `resolve_simpl_mode` function you added in Task 1, append:

```python
def build_county_busmap(n: "pypsa.Network") -> "pd.Series":
    """Construct a sub_id -> '<reeds_zone>_<county_fips>' busmap.

    Used by the simpl='county' fast-path. The county field is the 5-digit FIPS
    GEOID assigned in build_base_network from county_shapes.GEOID, which is
    nationally unique; the reeds_zone prefix is added for human readability
    when inspecting clustered networks.
    """
    if "county" not in n.buses.columns or n.buses.county.isna().any():
        raise ValueError(
            "simpl='county' requires every substation bus to carry a non-null "
            "'county' attribute. This attribute is dropped by "
            "aggregate_to_substations when topological_boundaries='state'. "
            "Set model_topology.topological_boundaries to 'county' (or "
            "'reeds_zone') in your config, or use a numeric {simpl} wildcard."
        )
    return (n.buses.reeds_zone.astype(str) + "_" + n.buses.county.astype(str)).rename(
        "busmap"
    )
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `cd workflow/scripts && pytest test/test_cluster_simpl.py -v`

Expected: All 11 tests pass (6 from Task 1 plus 5 new).

- [ ] **Step 5: Commit**

```bash
git add workflow/scripts/cluster_simpl.py workflow/scripts/test/test_cluster_simpl.py
git commit -m "Add build_county_busmap helper with missing-county guard"
```

---

## Task 3: Wire the helpers into `cluster_simpl.py` `__main__`

**Goal:** Replace the existing `if snakemake.wildcards.simpl: … else: …` block at lines 49-87 of `cluster_simpl.py` with a three-way dispatch via `resolve_simpl_mode`. The new county branch calls `get_clustering_from_busmap` directly with the output of `build_county_busmap`.

**Files:**
- Modify: `workflow/scripts/cluster_simpl.py:22-92` (imports + main block)

This task does not add unit tests — the dispatch lives in `__main__` and is exercised end-to-end by running the workflow. The helpers it calls are already covered by Tasks 1-2.

- [ ] **Step 1: Add the `get_clustering_from_busmap` import**

In `workflow/scripts/cluster_simpl.py`, find the existing import block (around lines 22-28):

```python
import logging

import geopandas as gpd
import pandas as pd
import pypsa
from _helpers import (
    configure_logging,
    log_network_schema,
    plot_geojson,
    update_p_nom_max,
)
from cluster_network import cluster_regions, clustering_for_n_clusters
```

Append one line after the `cluster_network` import:

```python
from pypsa.clustering.spatial import get_clustering_from_busmap
```

- [ ] **Step 2: Replace the dispatch block in `__main__`**

Find the block from line 49 ("if snakemake.wildcards.simpl:") through line 87 (`busmap.to_csv(snakemake.output.busmap)`). Replace the **entire if/else dispatch block (lines 49-81 in the current file)** with the dispatch below. Leave the post-dispatch tail (`busmap.index = busmap.index.astype(str)` and following) unchanged.

Replace:

```python
if snakemake.wildcards.simpl:
    configured_strategy = params.simplify_network.get(
        "weighting_strategy",
        "population",
    )
    if configured_strategy != "population":
        logger.info(
            "cluster_simpl runs before loads/generators are attached; using "
            "weighting_strategy='population' (n.buses.Pd) regardless of "
            "configured '%s'.",
            configured_strategy,
        )

    clustering = clustering_for_n_clusters(
        n,
        int(snakemake.wildcards.simpl),
        focus_weights=params.focus_weights,
        solver_name=solver_name,
        algorithm=params.simplify_network["algorithm"],
        aggregation_strategies=params.aggregation_strategies,
        weighting_strategy="population",
    )
    busmap = clustering.busmap
    n = clustering.network

    cluster_regions((busmap,), snakemake.input, snakemake.output)
else:
    for which in ("regions_onshore", "regions_offshore"):
        regions = gpd.read_file(getattr(snakemake.input, which))
        out_path = getattr(snakemake.output, which)
        regions.to_file(out_path)
        plot_geojson(out_path)
    busmap = pd.Series(n.buses.index, index=n.buses.index, name="cluster_bus")
```

with:

```python
mode = resolve_simpl_mode(snakemake.wildcards.simpl)

if mode == "kmeans":
    configured_strategy = params.simplify_network.get(
        "weighting_strategy",
        "population",
    )
    if configured_strategy != "population":
        logger.info(
            "cluster_simpl runs before loads/generators are attached; using "
            "weighting_strategy='population' (n.buses.Pd) regardless of "
            "configured '%s'.",
            configured_strategy,
        )

    clustering = clustering_for_n_clusters(
        n,
        int(snakemake.wildcards.simpl),
        focus_weights=params.focus_weights,
        solver_name=solver_name,
        algorithm=params.simplify_network["algorithm"],
        aggregation_strategies=params.aggregation_strategies,
        weighting_strategy="population",
    )
    busmap = clustering.busmap
    n = clustering.network
    cluster_regions((busmap,), snakemake.input, snakemake.output)

elif mode == "county":
    logger.info(
        "cluster_simpl fast-path: using county FIPS as busmap (no k-means).",
    )
    busmap = build_county_busmap(n)
    clustering = get_clustering_from_busmap(
        n,
        busmap,
        aggregate_generators_weighted=True,
        aggregate_one_ports=["Load", "StorageUnit"],
        line_length_factor=1.25,
        bus_strategies={"Pd": "sum", "LAF_state": "sum"},
        line_strategies=params.aggregation_strategies.get("lines", {}),
        generator_strategies=params.aggregation_strategies.get("generators", {}),
        one_port_strategies=params.aggregation_strategies.get("one_ports", {}),
        scale_link_capital_costs=False,
    )
    busmap = clustering.busmap
    n = clustering.network
    cluster_regions((busmap,), snakemake.input, snakemake.output)

else:  # mode == "identity"
    for which in ("regions_onshore", "regions_offshore"):
        regions = gpd.read_file(getattr(snakemake.input, which))
        out_path = getattr(snakemake.output, which)
        regions.to_file(out_path)
        plot_geojson(out_path)
    busmap = pd.Series(n.buses.index, index=n.buses.index, name="cluster_bus")
```

- [ ] **Step 3: Verify the file parses and existing tests still pass**

Run: `cd workflow/scripts && python -c "import cluster_simpl; print('ok')"`

Expected: `ok` (no syntax errors or import failures).

Then: `cd workflow/scripts && pytest test/test_cluster_simpl.py -v`

Expected: All 11 tests from Tasks 1-2 still pass.

- [ ] **Step 4: Smoke-test the dispatch via mock_snakemake (optional, skip if pypsa-USA env not available)**

If your environment has pypsa-USA installed and the tutorial outputs cached:

```bash
cd workflow
uv run snakemake -j1 --configfile config/config.tutorial.yaml \
  --until cluster_simpl \
  resources/networks/western/elec_scounty.nc \
  --dry-run
```

Expected: dry-run prints a plan that includes `cluster_simpl` with `simpl=county`, producing `elec_scounty.nc`. If no environment is available, skip this step — Tasks 1-2's unit tests plus the import check in Step 3 cover the helpers.

- [ ] **Step 5: Commit**

```bash
git add workflow/scripts/cluster_simpl.py
git commit -m "Dispatch cluster_simpl through resolve_simpl_mode; add county fast-path"
```

---

## Task 4: Document the new wildcard value

**Goal:** Tell users the `"county"` wildcard value exists.

**Files:**
- Modify: `docs/source/config-wildcards.md:29-34`
- Modify: `workflow/config/config.default.yaml:13`

- [ ] **Step 1: Update `config-wildcards.md`**

Open `docs/source/config-wildcards.md`. Find lines 29-34:

```markdown
(simpl)=
## The ``{simpl}`` wildcard

The ``{simpl}`` wildcard specifies number of buses a detailed
network model should be pre-clustered to in the rule
:mod:`simplify_network` (before :mod:`cluster_network`).
```

Replace with:

```markdown
(simpl)=
## The ``{simpl}`` wildcard

The ``{simpl}`` wildcard specifies number of buses a detailed
network model should be pre-clustered to in the rule
:mod:`simplify_network` (before :mod:`cluster_network`).

Recognized values:

- *empty* (``""``): identity pass-through; the substation-level network is
  forwarded to :mod:`cluster_network` without pre-clustering.
- *positive integer* (e.g. ``50``): pre-cluster with k-means to that many
  buses, weighted by ``n.buses.Pd``.
- ``"county"``: fast-path. Uses each substation's county FIPS as a direct
  busmap and skips k-means entirely. Produces one cluster bus per county,
  named ``<reeds_zone>_<county_fips>`` (e.g. ``p9_06001``). Requires
  ``model_topology.topological_boundaries`` to be ``"county"`` or
  ``"reeds_zone"`` (it is dropped by ``aggregate_to_substations`` when set
  to ``"state"``).
```

- [ ] **Step 2: Update `config.default.yaml`**

Open `workflow/config/config.default.yaml`. Find line 13:

```yaml
  simpl: [75]
```

Replace with:

```yaml
  simpl: [75]  # int -> k-means; "" -> identity; "county" -> fast-path county FIPS busmap
```

- [ ] **Step 3: Commit**

```bash
git add docs/source/config-wildcards.md workflow/config/config.default.yaml
git commit -m "Document simpl='county' fast-path wildcard value"
```

---

## Final verification

After Task 4 commits, run the full test file once more to make sure nothing regressed:

```bash
cd workflow/scripts && pytest test/test_cluster_simpl.py -v
```

Expected: 11 passing tests.

Confirm the git log shows four well-scoped commits:

```bash
git log --oneline -4
```

Expected output (commit hashes will vary):

```
<hash> Document simpl='county' fast-path wildcard value
<hash> Dispatch cluster_simpl through resolve_simpl_mode; add county fast-path
<hash> Add build_county_busmap helper with missing-county guard
<hash> Add resolve_simpl_mode dispatch helper to cluster_simpl
```

---

## Self-review

**Spec coverage:**
- `simpl="county"` wildcard value → Task 3 (dispatch) + Task 2 (busmap).
- Fast-path branch in `cluster_simpl.py` → Task 3.
- Clear error when `county` column missing → Task 2.
- Unit test covering fast path → Task 2 (busmap correctness + missing-county errors); Task 1 (mode dispatch + unknown sentinel).
- Bus IDs `<reeds_zone>_<county_FIPS>` → Task 2.
- Catch-all error for unknown sentinels → Task 1 (`resolve_simpl_mode`).
- Doc updates (`config-wildcards.md`, `config.default.yaml`) → Task 4.
- No rule-graph changes, no other downstream changes → confirmed in plan (only `cluster_simpl.py` + tests + docs).

**Placeholder scan:** No TBDs, no "implement later", no skipped code blocks. Every step shows the actual code or command.

**Type consistency:** `resolve_simpl_mode` returns `str` with the three sentinel values `"identity"`, `"county"`, `"kmeans"`; Task 3's dispatch checks against exactly those strings. `build_county_busmap` returns a `pd.Series` named `"busmap"`; Task 3 reads `clustering.busmap` (a different Series produced by `get_clustering_from_busmap`) for the downstream tail of the script. The names don't collide because the local `busmap` variable is reassigned after the clustering call.

**Out-of-scope items that intentionally don't appear:**
- No fast-path for `simpl="state"` / `simpl="reeds_zone"` (spec explicitly defers).
- No new config knobs (spec rules out).
- No changes to `cluster_network.py`, `add_electricity.py`, or any rule file.

The plan is complete.
