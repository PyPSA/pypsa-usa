# Config Cleanup — Formatting-Only Pass on `repo_data/config/`

**Date:** 2026-05-21
**Scope:** `workflow/repo_data/config/{config.default.yaml, config.tutorial.yaml, config.common.yaml}`
**Risk:** Low. No key renames, no hierarchy changes, no behavior changes.

## Goal

Make the three user-facing YAML configs in `repo_data/config/` self-documenting so a new user can read top-to-bottom and understand every option without cross-referencing the docs or the scripts. Surface every silent-default key as a commented-out example so users see the full surface area.

## Non-Goals

- **No key renames or hierarchy changes** (would cascade into `workflow/scripts/`, `workflow/rules/`, and break existing user configs).
- **No sync between `repo_data/config/` and `workflow/config/`.** The two have drifted; per user direction they may continue to drift. Reconciling is a separate content change.
- **No CSV table edits** in `docs/source/configtables/`. Those remain the canonical option descriptions; YAML comments only summarize and point at units/enums.
- **No content/value changes** (knob values stay the same). The only value-bearing edit is normalizing `False` → `false` for YAML conformance.

## Load-Bearing Constraints

1. **`# docs :` markers are sentinels.** `docs/source/config-configuration.md` slices each section out with `:start-at: <key>:` / `:end-before: # docs :`. The line `# docs :` must remain between every documented section, and section order must be preserved. The marker may be **extended** (e.g. `# docs : SCENARIO`) but the prefix must stay.
2. **Top-level key order documented in `config-configuration.md`:** `run → scenario → snapshots → atlite → electricity → renewable → lines → links → load → co2 → dac → costs → clustering → solving → sector → custom_files`. Maintain this order in `config.default.yaml`. `config.common.yaml` follows the same convention for the sections it owns (`renewable`, `atlite`).
3. **Snakemake config merge.** `workflow/Snakefile` merges `config.common.yaml`, `config.api.yaml`, `config.cluster.yaml`, `config.plotting.yaml`, `config.sector.yaml` underneath the user-specified `--configfile`. Keys in `common` are overridable by the main config; this is unchanged.

## Style Guide

### Section headers

Replace bare `# docs :` with a labeled banner:

```yaml
# ====================================================================
# SCENARIO — wildcards and planning horizons that fan out the workflow
# ====================================================================
# docs : SCENARIO
scenario:
  ...
```

The `# docs : LABEL` line is what the Sphinx slicer matches; the `====` banner is purely cosmetic and goes **above** the slicer sentinel so it doesn't appear in the rendered docs.

### Option comments

One short inline comment per option. Convention:

```yaml
key: value                          # what it does; allowed: A|B|C; unit
```

Skip the comment if the option key is already self-descriptive AND the value is unambiguous (e.g. `name: "Default"`). Always include:

- **Units** (MW, $/MWh, fraction, percent, hours)
- **Enum'd choices** (`A|B|C`)
- **Acronym expansions** on first use per section (efs, eulp, aeo, atb, ucap, corine, cec, boem, EGS, PHS, OCGT, CCGT, REM, ERM, PTC, ITC, PRM, SAFE, ATB)

### File header

Each file opens with a 5-10 line block:

```yaml
# ====================================================================
# PyPSA-USA — <FILE PURPOSE>
# ====================================================================
# <one paragraph: who edits this, when, what gets loaded with it>
#
# Load order: Snakemake auto-loads config.common.yaml, config.api.yaml,
# config.cluster.yaml, config.plotting.yaml, config.sector.yaml beneath
# whatever main config you pass with `--configfile`. Keys here override
# those.
#
# See: docs/source/config-configuration.md
```

### Whitespace

- Exactly one blank line between top-level sections.
- No trailing whitespace.
- Normalize boolean casing: `false`/`true`, never `False`/`True`.
- Two-space indent (existing convention, preserved).

## Sample: `scenario` section, before → after

**Before** (`repo_data/config/config.default.yaml`):

```yaml
# docs :
scenario:
  interconnect: [western] #"usa|texas|western|eastern"
  clusters: [33]
  simpl: [75]
  opts: [REM-3h]
  ll: [v1.0]
  sector: "" # G
  planning_horizons: [2030, 2040, 2050]    #(2018-2023, 2030, 2040, 2050)
foresight:  'perfect' # myopic, perfect
```

**After**:

```yaml
# ====================================================================
# SCENARIO — workflow wildcards and planning horizons
# ====================================================================
# Each list entry becomes a separate run in the Snakemake DAG. Most
# users set a single value per wildcard. See docs: config-wildcards.md
# docs : SCENARIO
scenario:
  interconnect: [western]              # geographic scope; one of: usa | texas | western | eastern
  planning_horizons: [2030, 2040, 2050] # investment-period years (2018-2023 | 2030 | 2040 | 2050)
  clusters: [33]                       # final cluster count; integer, optionally suffixed m/a/c, or "all"
  simpl: [75]                          # pre-clustering kmeans granularity; alphanumeric or "all"
  ll: [v1.0]                           # line-limit scenario; v|c + number/opt/all (e.g. v1.0, copt)
  opts: [REM-3h]                       # dash/plus options string (REM = renewable-energy mix, 3h = 3-hour resolution)
  sector: ""                           # sector-coupling carriers; "" (electricity-only) | E | G | E-G

# docs : FORESIGHT
foresight: 'perfect'                   # multi-horizon solve mode; perfect | myopic
```

Note: `foresight` is left at top level (not nested under `scenario`) because that is the existing schema — the docs slicer addresses it as its own region.

## Per-File Plan

### `config.default.yaml` (the canonical user-facing example)

1. Rewrite file header to describe its purpose as the canonical full-feature example.
2. Add labeled section banners above each `# docs :` sentinel.
3. Reorder keys within each section to surface the most-touched knobs first; do not move keys between sections.
4. Inline-comment every option per the style guide.
5. **Add as commented-out examples**, every option that scripts read but the file currently omits. Verbose mode — completeness over brevity. Methodology in the "Missing-keys discovery" section below.
6. Normalize `False`→`false`.

### `config.tutorial.yaml` (the minimal smoke-test config)

1. Rewrite file header to clarify this is a minimal config: lists only the keys it actually changes vs. default; everything else inherits.
2. Same section banners and inline comments as default.
3. **Do not bloat with all missing keys** — the tutorial's value is brevity. Instead, add a single comment block at top listing which top-level sections from default are intentionally omitted (`co2`, `dac`, `imports/exports`, `demand_response`, `erm`, `walltime`, `renewable_scenarios`, `renewable_snapshots`) and pointing the user to default.
4. Keep all existing keys; just reformat and comment.

### `config.common.yaml` (the always-loaded layered base)

1. Add file header explaining it's auto-loaded under every main config and that user-config keys override anything here.
2. Add the missing `# docs :` sentinels between `lines`, `offshore_shape`, `offshore_network`, `ucap`, and `pudl_path` so the Sphinx slicer can address them. (Pure addition, no break.)
3. Inline-comment every option, including the `corine`-typo workaround (acknowledge it stays misnamed for pypsa-eur parity).
4. Add commented-out examples for any common-section keys read by scripts but absent here.
5. Normalize `False`→`false`.

## Missing-Keys Discovery (methodology)

For each script in `workflow/scripts/` and rule in `workflow/rules/`, extract every `snakemake.config[...]` and `config["..."]` access path. Compare against keys present in the union of `repo_data/config/config.{default,common,tutorial}.yaml` after layered merge. Any path read but not defined is a silent default — add it (commented out, with default value as written in the script and a one-line comment) to whichever file owns its section.

Implementation will produce an audit list inline with the implementation work; the spec does not enumerate keys ahead of time.

## Verification

After applying changes:

1. **YAML parses.** `python -c "import yaml; [yaml.safe_load(open(f)) for f in ['workflow/repo_data/config/config.default.yaml', 'workflow/repo_data/config/config.tutorial.yaml', 'workflow/repo_data/config/config.common.yaml']]"`
2. **Docs slicer still matches.** For each `# docs :` reference in `docs/source/config-configuration.md`, confirm the corresponding section still resolves between `start-at` and the next `# docs :` marker. Spot-check by rendering the docs locally or by running the slice manually:
   ```bash
   awk '/^scenario:/,/^# docs :/' workflow/repo_data/config/config.default.yaml
   ```
3. **Snakemake dry-run.** `cd workflow && uv run snakemake -n --configfile config/config.test_small.yaml rule data_model` (the loaded config is `workflow/config/`, not `repo_data/`, so this catches no regression — but it confirms common-config keys we added or normalized still merge cleanly).
4. **Pre-commit hooks pass.** snakefmt, ruff, pretty-format-yaml will reformat on commit; spec compliance means they do nothing on the staged files.

## Out of Scope (Follow-Ups)

- Syncing `repo_data/config/` ↔ `workflow/config/`. Drift is acknowledged and left alone per user direction.
- Renaming `corine` to `copernicus` (would touch every renewable script).
- Splitting `electricity:` into a clearer sub-taxonomy.
- Updating the CSV tables in `docs/source/configtables/` to cover newly-surfaced keys. Worth doing later but not part of this pass.
