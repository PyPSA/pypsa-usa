# Cost input files

## ReEDS inputs

These files are copied unchanged from NREL's ReEDS (<https://github.com/ReEDS-Model/ReEDS>,
earlier published as `NREL/ReEDS-2.0`). Each is identical in release 2026.09.21 (the latest,
checked on 2026-11-19) and on `main` at that date.

| File | ReEDS path | Unchanged since | Hash | Bytes |
| --- | --- | --- | --- | --- |
| `reg_cap_cost_diff_default.csv` | `inputs/financials/` | commit `52c6e8e8` (2026-08-17, "Add h2 fuel cell technology") | git blob `9ecb94d2320f8e1bdafb1f0ea2a529ba49efd77a` | 156422 |
| `interconnection_land.h5` | `inputs/supply_curve/` (Git LFS) | the repository's first commit (version 2026.04.15; also ReEDS-2.0 v2025.7.0) | SHA-256 `cca210228dc0bef8b09abe97ab019152da7bd3bc104abd0ab42fee92bae89f32` | 1932387 |
| `interconnection_offshore.h5` | `inputs/supply_curve/` (Git LFS) | the repository's first commit (version 2026.04.15; also ReEDS-2.0 v2025.7.0) | SHA-256 `f3a8e9c97a97e65e86b3f8c58558b698c6afae46456230fd8dfe343b087b0391` | 765446 |

- `reg_cap_cost_diff_default.csv`: relative capital-cost difference of each county (`p` +
  5-digit FIPS) per ReEDS technology group. `costs: regional_multipliers` applies it as `1 + x`
  to the overnight capex, and `column_to_tech` maps its columns to carriers.
- `interconnection_{land,offshore}.h5`: one row per reV supply-curve site (`sc_point_gid`) with
  its spur-line, point-of-interconnection and network-reinforcement costs in USD/MW and their
  distances; the offshore table adds the export cable, radial and meshed. Their dollar year
  (2023) is the `dollaryear` attribute of the `/data` group. `scripts/reeds_vre.py` reads
  `cost_total_trans_usd_per_mw` (land) and `cost_total_trans_usd_per_mw|radial` (offshore) and
  converts them to `costs: dollar_year`.

To update them, take the files from a newer ReEDS release (the `.h5` files with
`git lfs pull --include "inputs/supply_curve/interconnection_*.h5"`), replace them here, update
the column names in `costs: regional_multipliers: column_to_tech` if ReEDS renamed a group, and
update the release and hashes above.
