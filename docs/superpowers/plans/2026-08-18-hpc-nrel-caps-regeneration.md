# HPC handoff: regenerate NREL caps artifacts with per-bus coordinates

> Self-contained brief for an agent running on the HPC (Sherlock,
> `/home/groups/iazevedo/asia/pypsa-usa`). No other context is required.
> Written 2026-08-18 on branch `v1-epic` (ktehranchi/pypsa-usa).

## Why

The derived NREL supply-curve capacity files
(`workflow/data/nrel_exclusion/derived/caps_{tech}_{access}{suffix}.nc`)
are rolled up against the **national** substation tessellation (17,890
entries for onwind/reference). Footprint-scoped runs (e.g. CA-only) used to
silently drop every out-of-footprint entry — for CA that was 17,340 entries
carrying 97.3% of national onwind `p_nom_max`, including border regions
holding 13.4% of the West's developable wind (full evidence:
`docs/superpowers/specs/2026-08-07-deltas-ledger.md`, Amendments section).

Commit `3a0fbd2c` added an opt-in recovery
(`nrel_caps_reassign: {enable, max_km}` in `config.common.yaml`) that folds
unmapped entries onto the geographically nearest in-footprint bus. **It
needs per-entry coordinates, and the published caps files have none** —
only `bus` IDs plus `p_nom_max/potential/weight/average_distance/avg_cf`.
The rollup script `workflow/scripts/nrel_exclusion/build_nrel_bus_capacities.py`
now writes per-bus `x`/`y` (capacity-weighted site centroid; bus-polygon
centroid fallback for zero-capacity buses). Your job: regenerate the caps
files with that script so they carry coordinates, **changing nothing else**.

## Hard invariant (the whole point of the harness)

The regenerated files must be **byte-identical to the current ones in every
pre-existing variable and attribute** — the ONLY additions are the `x` and
`y` data variables (and any attrs the new code documents). If any capacity
number shifts, STOP and report; do not publish.

## Step 0 — repo state

```bash
cd /home/groups/iazevedo/asia/pypsa-usa
git fetch origin   # or whatever remote tracks ktehranchi/pypsa-usa
git checkout v1-epic && git pull
git log --oneline -1   # must be 3a0fbd2c or later
```

The conda env is `/home/groups/iazevedo/asia/miniforge3/envs/pypsa-usa`
(hardcoded as `$PY` in the build script).

## Step 1 — establish the generation geometry (verify, don't assume)

The build script `workflow/scripts/nrel_exclusion/build_nrel_artifacts.sh`
defaults to `INTERCONNECT=western` and
`BUS_SUBDIR=nrel_smoke_rcp45cooler_2030_reference`, but the PUBLISHED caps
files are keyed to a **17,890-bus national tessellation**. Before building,
identify which `resources/godeeep/geospatial/<BUS_SUBDIR>/<INTERCONNECT>/regions_onshore.geojson`
on the HPC has exactly the bus set of the current artifacts:

```bash
python - <<'EOF'
import xarray as xr
ds = xr.open_dataset("workflow/data/nrel_exclusion/derived/caps_onwind_reference_cec.nc")
print(len(ds.bus), sorted(ds.bus.values.tolist())[:5])
EOF
# then for candidate shape files:
python - <<'EOF'
import geopandas as gpd
g = gpd.read_file("workflow/resources/godeeep/geospatial/<CANDIDATE>/usa/regions_onshore.geojson")
print(len(g), g.name.head().tolist())
EOF
```

Pick the `BUS_SUBDIR`/`INTERCONNECT` whose region `name` set matches the
caps `bus` set 1:1 (same count, same IDs). If no candidate matches, STOP
and report what exists — regenerating against different geometry would
change results, violating the invariant. NOTE the exact filename suffixes
of the published files too (`_cec`, `_boem`, bare) — they encode
`APPLY_CEC`/`APPLY_BOEM` at generation time; regenerate each file with the
matching flags.

## Step 2 — back up, then force regeneration (caps only)

The build script **skips files that already exist**, so move the current
caps aside first (keep them — they are the identity baseline). Do NOT
touch the `avail_*` files (`build_nrel_availability.py` is unchanged).

```bash
cd workflow/data/nrel_exclusion/derived
mkdir -p pre_xy_backup
for f in caps_*.nc; do cp -p "$f" pre_xy_backup/"$f"; done
rm caps_*.nc
```

Then regenerate every (tech, access) combination that existed before
(inventory `pre_xy_backup/` to enumerate; expected: onwind/solar/offwind/
offwind_floating x reference/limited, with the suffixes found in Step 1):

```bash
cd /home/groups/iazevedo/asia/pypsa-usa/workflow
INTERCONNECT=<from step 1> BUS_SUBDIR=<from step 1> \
  bash scripts/nrel_exclusion/build_nrel_artifacts.sh reference
INTERCONNECT=<from step 1> BUS_SUBDIR=<from step 1> \
  bash scripts/nrel_exclusion/build_nrel_artifacts.sh limited
```

(Use `TECHS=...` / `APPLY_CEC=` / `APPLY_BOEM=` env overrides to reproduce
each published suffix exactly. Raw NREL supply-curve CSVs live under the
paths hardcoded in the two python scripts; if one is missing, STOP and
report rather than substituting.)

## Step 3 — verify the invariant, per file

For EVERY regenerated file vs its backup:

```bash
python - <<'EOF'
import sys, xarray as xr
new = xr.open_dataset(sys.argv[1]); old = xr.open_dataset(sys.argv[2])
assert {"x", "y"} <= set(new.data_vars), "x/y missing"
xr.testing.assert_identical(new.drop_vars(["x", "y"]), old)  # nothing else changed
b = new.sel(bus=new.p_nom_max > 0)
assert float(b.x.min()) > -130 and float(b.x.max()) < -60, "x out of CONUS"
assert float(b.y.min()) > 20 and float(b.y.max()) < 55, "y out of CONUS"
assert float(b.x.isnull().sum()) == 0 and float(b.y.isnull().sum()) == 0
print("OK", sys.argv[1])
EOF
```

(If `assert_identical` fails only on new attrs added by the updated script,
compare `data_vars` identically and report the attr diff instead of
failing.) Also spot-check 3 buses: their `x`/`y` should fall inside or near
their region polygon from the Step 1 geojson.

## Step 4 — quantify the recovery (the number we actually want)

On the HPC (or report back for the laptop side): with the new caps in
place, run the CA-footprint onwind profile build with the flag ON in a
scratch config copy (do NOT edit tracked configs):

- copy `workflow/repo_data/config/config.equivalence.yaml` to a scratch
  name, set `nrel_caps_reassign: {enable: true, max_km: 100}` under the
  NREL block, and run `build_renewable_profiles` for
  `interconnect=western`/CA as that config defines.
- Report: recovered entry count, recovered `p_nom_max` MW, new profile bus
  count vs the baseline 550 caps entries / 544 profile buses, and the same
  with `max_km: 50` and `200` (sensitivity).

## Step 5 — hand-off

1. Do NOT overwrite anything on Zenodo. Leave the new files in
   `derived/`, backups in `derived/pre_xy_backup/`.
2. Report back: the Step 1 geometry identification, per-file verification
   results, Step 4 recovery numbers, and total runtime.
3. Publishing (new Zenodo version + bumping the ID in
   `workflow/rules/retrieve.smk`, precedent: PR #748) is a separate,
   human-approved step — the laptop-side equivalence harness must re-run
   first with the new files and flag OFF to confirm byte-identical
   pipeline behavior.

## Guardrails

- Never modify repo code on the HPC; if a script errors, report the
  traceback instead of patching.
- Never delete the backups.
- If SLURM is needed for the heavy steps, follow the submission pattern in
  `workflow/run_slurm.sh`; the rollup itself is single-node python.
