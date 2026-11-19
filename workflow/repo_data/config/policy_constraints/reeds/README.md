# ReEDS state policy inputs

Copies of NREL ReEDS input tables used for the state portfolio standards (`RPS`/`CES` opts)
and as the source of the ReEDS-derived rows in `../technology_capacity_targets.csv` (`TCT` opt).

Source: <https://github.com/ReEDS-Model/ReEDS>, release
[`2026.09.21`](https://github.com/ReEDS-Model/ReEDS/releases/tag/2026.09.21)
(commit `8a1572331da89b15ad3c0f74db448911f715266c`), directory `inputs/state_policies/`.
The files are byte-identical to the upstream ones; the hash column is the git blob hash
(`git hash-object <file>`), which should equal the `sha` of the file in the ReEDS tree.

| File | ReEDS path | Git blob hash | Used by |
|------|------------|---------------|---------|
| `rps_fraction.csv` | `inputs/state_policies/rps_fraction.csv` | `2b52adb0c0eddf7313a30e60fdc661959bac5ab6` | `RPS` opt (`rps_all` only) |
| `ces_fraction.csv` | `inputs/state_policies/ces_fraction.csv` | `6945f79cb3ea661b4152af3ccc92359c02c69db3` | `RPS` opt (CES carriers) |
| `storage_mandates.csv` | `inputs/state_policies/storage_mandates.csv` | `9f139c60449f2a499ffcf4e1f4f3c6aa55989337` | TCT `reeds_storage_mandate` rows |
| `offshore_req_default.csv` | `inputs/state_policies/offshore_req_default.csv` | `f942c546351d95921f79c3fae4ed4587edf7924a` | TCT `reeds_offshore_wind_mandate` rows |
| — | `inputs/state_policies/techs_banned.yaml` | `2241bfe1e22cd5566224bcc0a47252fe5179f0f6` | TCT `reeds_nuclear_no_build` rows (not copied) |

`ReEDS_Constraints/state_policies/` holds the same four files with the same hashes; the
workflow reads the copies in this directory.

## Formats

- `rps_fraction.csv`: long, `t,st,rps_all,rps_solar,rps_wind`. Only `rps_all` is applied; the
  solar/wind carve-outs are ignored, as in the ReEDS default (`GSw_StateRPS_Carveouts = 0`). The
  `voluntary` pseudo-state is dropped because it matches no `reeds_state`.
- `ces_fraction.csv`: long, `*t,st,Value`. `opts.policy._process_reeds_data` reads only the long
  layout; the wide layout (one column per year) shipped before this update raises an error.
- `storage_mandates.csv`: long, `*st,t,MW` (cumulative MW).
- `offshore_req_default.csv`: wide, one column per year (cumulative MW). This is the ReEDS default
  scenario (`GSw_OfsWindForceScen = default`).

## Derived TCT rows

The ReEDS rows in `../technology_capacity_targets.csv` are regenerated from these files:

- `reeds_nuclear_no_build`: `max = existing` for carriers `nuclear, SMR` in the states banned in
  `techs_banned.yaml` (OR, CA, MN, VT, MA, RI, ME) and in ReEDS zone `p128`, which is exactly the
  two banned New York counties (Nassau 36059, Suffolk 36103) in ReEDS `inputs/zones/z134/county2zone.csv`.
- `reeds_storage_mandate`: `min = MW` of battery power capacity for every state and year in
  `storage_mandates.csv`. The carrier list holds every battery carrier: `battery` (existing EIA
  and remote-contract units, non-extendable) and `2/4/6/8/10hr_battery_storage` (new builds).
  The TCT constraint subtracts the active non-extendable capacity from the target, so existing
  batteries count toward the mandate. This follows ReEDS `eq_batterymandate`
  (`reeds/core/setup/c_model.gms`), which sums `CAP` over every battery technology and every
  vintage (plus the battery part of PV-battery hybrids, which PyPSA-USA does not model).
- `reeds_offshore_wind_mandate`: `min = MW` of `offwind, offwind_floating` for every state and
  year with a positive requirement in `offshore_req_default.csv`.
- `reeds_forced_retirements`: unchanged; it matches ReEDS `forced_retirements.csv` of this
  release (VA, IL, NY), which now lists states instead of ReEDS zones.

## Not updated

`prm_annual.csv` (2024-07-22), `rggicon.csv`, `rggi_states.csv` and `offshore_req_30by30.csv`
(2024-07-29) are older copies. `prm_annual.csv` is passed to the solve rule but no script reads
it; the ReEDS release replaced its `none/static/ramp*` columns with `static/nerc` from 2026.
`offshore_req_30by30.csv` no longer exists in ReEDS.
