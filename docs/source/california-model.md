(california-model)=
# California Model (CPUC SERVM)

PyPSA-USA ships a maintained, runnable California-only configuration at
`workflow/repo_data/config/config.california.yaml` (copied to `workflow/config/` by
`init_pypsa_usa.sh`). This page is the reference for that configuration: what data goes into
it, which weather years are available for each ingredient, and which simplifications the
results carry.

The model is a **California carve-out of the Western Interconnection**, not a standalone
network. The footprint is the four California ReEDS zones (`p8`, `p9`, `p10`, `p11`); the rest
of WECC is represented as capped import/export links rather than as buses. Load is the
California Public Utilities Commission's 2026 Integrated Resource Planning hourly forecast,
produced with the SERVM production-cost model.

## Running it

```bash
cd workflow

# full pipeline (build + solve + figures)
uv run snakemake -j1 --configfile config/config.california.yaml

# data model only, no solver
uv run snakemake data_model -j1 --configfile config/config.california.yaml
```

The first build downloads one ~118 MB SERVM load CSV per planning horizon from
`files.cpuc.ca.gov` (`retrieve_cpuc_servm_load`), the CPUC Baseline Generator List workbook
(`retrieve_cpuc_baseline_generators`), and the GODEEEP / NREL land-access artifacts from
Zenodo. `imports: costs: wholesale` calls the EIA API, so a key must be present in
`config/config.api.yaml`.

Scenario settings as shipped: `interconnect: western`, `simpl: 75`, `clusters: 4`,
`ll: v1.0`, `opts: REM-3h`, `planning_horizons: [2030, 2035, 2040, 2045]`, `foresight: perfect`,
`sector: ""` (electricity only).

## Data inventory

| Ingredient | Source | Where it enters | Location |
| --- | --- | --- | --- |
| **Demand** | CPUC SERVM 2026 IRP hourly load — 6 California load regions, 9 forecast years, 25 stacked weather years, `Net Load` plus every published component | `demand: profile: servm` | `data/cpuc/servm/HourlyLoad_CA_Regions_V2025E_2224_Mon_{year}.csv`, retrieved from [files.cpuc.ca.gov](https://files.cpuc.ca.gov/energy/modeling/2026_servm_updates/) |
| **Demand → bus mapping** | 2020 Decennial Census county population, routed through the SERVM region map | `demand: bus_allocation: population`; `build_servm_load_weights` | `workflow/repo_data/CPUC/servm_region_map.csv` |
| **Existing fleet** | [PUDL](https://catalystcoop-pudl.readthedocs.io) `v2026.8.0` (EIA-860/923, FERC), announced retirements honored | `build_powerplants` → `resources/powerplants/powerplants.csv` | `pudl_path: s3://pudl.catalyst.coop/v2026.8.0` (`config.common.yaml`) |
| **Unit-commitment parameters** | WECC Anchor Data Set thermal tables, merged onto the EIA fleet and clamped to per-carrier bounds | `merge_ads_data` → `sanitize_uc_parameters` | `workflow/repo_data/plants/`, `UC_BOUNDS` in `build_powerplants.py` |
| **Renewable profiles** | [GODEEEP](https://www.pnnl.gov/projects/godeeep) capacity factors, weighted by NREL reV land-access availability | `renewable: dataset: godeeep`, `renewable_land_access: reference` | Zenodo (see [Weather years](ca-weather-years)) |
| **Transmission backbone** | ReEDS/NARIS zonal network | `model_topology: transmission_network: reeds` | `workflow/repo_data/ReEDS_Constraints/transmission/` |
| **Trade capacity** | NARIS AC flowgate ratings on the footprint boundary | `imports/exports: capacity_limit: true` | `transmission_capacity_init_AC_ba_NARIS2024.csv` (or `..._county_...` at county resolution) |
| **Trade interface caps** | CPUC RESOLVE aggregate CAISO interface limits | `model_topology: interface_transmission_limits: true` | `config/policy_constraints/transmission_interface_limits.csv` |
| **Trade prices** | EIA wholesale electricity prices for the weather year | `imports/exports: costs: wholesale` | EIA API (`config/config.api.yaml`) |
| **Out-of-state contracts** | CPUC ledger of physically out-of-state units serving California load (74 rows, 10,873 MW) | `electricity: remote_contracted_resources: enable: true` | `workflow/repo_data/CPUC/servm_out_of_state_units.csv` |
| **Emissions limit** | CARB 2022 Scoping Plan AB 32 trajectory, annual 2025-2050, import emissions factor 0.428 tCO2/MWh | `REM` token in `{opts}`; `electricity: regional_Co2_limits` | `config/policy_constraints/regional_Co2_limits.csv` (`CA_AB32` rows) |
| **Benchmark reference** | CPUC Baseline Generator List (`BaselineGeneratorList_CAISO.xlsx`), plus the out-of-state exclusion ledger | `run: benchmark_cpuc: true` | `data/cpuc/BaselineGeneratorList_CAISO.xlsx`, `workflow/repo_data/CPUC/servm_benchmark_regions.csv`, `servm_tech_map.csv` |
| **Costs** | NREL ATB (`Market` / `Moderate`) with IRA ITC/PTC modifiers; AEO reference fuel outlook | `costs:` block | see [Costs](data-costs.md) |

### Demand regions

SERVM reports six California load regions, mapped onto PyPSA-USA balancing areas by
`workflow/repo_data/CPUC/servm_region_map.csv`: `PGE`→`CISO-PGAE`, `SCE`→`CISO-SCE`,
`SDGE`→`CISO-SDGE`, `IID`→`IID`, `LADWP`→`LDWP`, and `NCNC`→`BANC` + `TIDC`. Four balancing
areas carry a *blank* region and are deliberately dropped with a log message rather than
hard-failing: `CISO-VEA` (a Nevada footprint) and the California slivers served by `PACW`
(Siskiyou/Del Norte/Modoc), `WALC` (Colorado River) and `NEVP` (Tahoe/CalNeva), none of which
the CPUC California-region files cover. An *unknown* balancing area still raises.

Only the `Net Load` component is dispatched against. The full component split is preserved on
the `subsector` index level and written to
`resources/<run>/demand/{interconnect}/power_zonal_components_s{simpl}.parquet`. See the
[SERVM section](servm-demand) of the demand page for the full treatment.

(ca-weather-years)=
## Weather years

Load and renewable profiles are selected by **two independent keys**, and they must be kept
in step. Setting them to different years decorrelates load from wind and solar, which
understates both peak net load and the flexibility requirement.

### Demand

`electricity: demand: scenario: servm_weather_years` selects one of the **25 weather years
(2000-2024)** stacked inside each SERVM forecast-year file. It takes a list with **exactly one**
entry; multiple entries are reserved for stochastic scenarios and currently raise
`NotImplementedError`, because the demand output path is not weather-year specific. A mismatch
against the top-level `renewable_weather_years` is permitted but logs a warning in
`build_electrical_demand`.

The *forecast* year is separate and comes from `scenario: planning_horizons`, which must be
drawn from the nine published SERVM years — 2026, 2028, 2030, 2032, 2035, 2037, 2040, 2042,
2045. SERVM demand is not interpolated or AEO-scaled between them.

### Renewables

Which renewable years are available depends on `renewable_scenarios`:

- **Climate scenarios** (`rcp45hotter`, `rcp45cooler`, `rcp85hotter`, `rcp85cooler`) are indexed
  by the **planning horizon**, not by a weather year. One Zenodo record per `(tech, scenario)`
  publishes exactly three horizons: **2030, 2040 and 2050**. `renewable_weather_years` is not
  consulted for the profile.
- **`historical`** is indexed by `renewable_weather_years[0]`. Every historical year flows
  through the same screened NREL land-access path (`renewable_land_access`, plus optional
  `_cec` / `_boem` overlays); availability depends on which registry source holds the
  compressed per-cell file:

| Source (first match wins) | Solar | Wind 100 m | Wind 125 m |
| --- | --- | --- | --- |
| Local (Oak) mirror, SHA256-verified | **1980-2022** | **1980-2022** | **1980-2022** |
| Zenodo records | 2012 | — | 2012 |

A `(dataset, year)` combination no configured source declares fails at snakemake parse time
with the available years listed — no fallback, no default hub height, no nearest-year
substitution ([issue #803](https://github.com/PyPSA/pypsa-usa/issues/803) is resolved by the
mirror; the interim unscreened bus-aggregated fallback is retired). This makes every SERVM
demand weather year 2000–2022 pairable with a screened renewable profile of the same year;
SERVM years 2023–2024 currently have no matching GODEEEP profile.

```{note}
`config.california.yaml` ships with `renewable_scenarios: ["rcp85cooler"]` and
`planning_horizons: [2030, 2035, 2040, 2045]`. Only 2030 and 2040 have a published GODEEEP
climate record; 2035 and 2045 have none. Use `planning_horizons: [2030, 2040]` under a climate
scenario, or switch to `renewable_scenarios: ["historical"]` (with the weather-year rules
above) for the other SERVM horizons.
```

## Spatial resolution

The footprint is set once, by `model_topology: include: reeds_state: ['CA']` (equivalently
`reeds_zone: ['p8','p9','p10','p11']`). Two transmission resolutions are supported, and
`config.california.yaml` carries the second as a commented alternative block.

| `topological_boundaries` | `clusters` | `simpl` | NARIS flowgate file |
| --- | --- | --- | --- |
| `reeds_zone` (shipped default) | `4` — the four California ReEDS zones | `75` | `transmission_capacity_init_AC_ba_NARIS2024.csv` |
| `county` | `58` — the 58 California counties | `'county'` (county-FIPS fast path) or any number ≥ 58 | `transmission_capacity_init_AC_county_NARIS2024.csv` |

`clusters` cannot go below the number of zones in the footprint, which is why the zonal case is
pinned to 4. The county case is pinned to 58 because the county NARIS interface table carries
exactly 58 `p06xxx` nodes. `add_extra_components` swaps the flowgate file automatically from
`topological_boundaries`; no other key changes. See [Spatial Configuration](spatial) for the
`simpl`/`clusters` split.

## Unit commitment

`conventional: unit_commitment: true` is on by default in this configuration, and
`solving: options: linearized_unit_commitment: true` relaxes the binary commitment variables —
the model is an LP, not a MILP. With `opts: REM-3h` the problem also runs at 3-hour resolution.

Commitment parameters (`min_up_time`, `min_down_time`, `ramp_limit_up`, `ramp_limit_down`,
`start_up_cost`, `minimum_load_mw`) come from the WECC Anchor Data Set thermal tables merged
onto the EIA fleet in `build_powerplants`. `sanitize_uc_parameters` then clamps every
committable row to per-carrier bounds (`UC_BOUNDS`, one band per carrier drawn from
NREL/EPRI/Intertek cycling literature and CAISO/WECC ADS typicals), filling missing values with
the carrier default and logging clamp/fill counts per carrier and parameter. The binding
invariant is `minimum_load_mw / p_nom <= min(summer_derate, winter_derate)`: a larger stable
minimum than the seasonal derate would leave the unit no feasible output above zero.
`add_electricity` re-clips the same fields as a second line of defence, and only committable
units are allowed to carry a non-zero `p_min_pu` (on a non-committable generator `p_min_pu` is
an unconditional must-run).

Clustering aggregates commitment attributes deliberately: `start_up_cost` sums (the aggregate
starts as one unit, so a capacity-weighted average would understate cycling cost by roughly the
member count), ramp limits and stable minima take capacity-weighted averages, and `committable`
takes `any`.

## Trade with the rest of WECC

California is not modelled as an island. `electricity: imports` and `electricity: exports` are
both enabled, adding trade links at boundary buses; three bounds apply.

1. **Per-path capacity** — `capacity_limit: true` rates each link from the NARIS AC flowgate
   table for the active `topological_boundaries`.
2. **Aggregate interface caps** — `interface_transmission_limits: true` adds one per-snapshot
   constraint per interface and direction from the RESOLVE table:

   | Interface | inside (`region_1`) | outside (`region_2`) | `flow_12` MW (export) | `flow_21` MW (import) |
   | --- | --- | --- | --- | --- |
   | `CA_NW` | p9, p10, p11 | p2, p5, p6, p7, p8 | 3,592 | 9,269 |
   | `CA_SW` | p9, p10, p11 | p12, p13, p25, p27, p28, p30 | 10,901 | 10,463 |
   | `CAISO_Imports` | p9, p10, p11 | all of the above | 9,728 | 10,208 |

3. **Annual volume** — `volume_limit: 25` with `balancing_period: year` caps imported (and
   separately exported) energy at 25 % of total demand, roughly CAISO's historical net-import
   share.

Imported energy is priced at EIA monthly wholesale prices and charged
`co2_emissions: 0.428` tCO2/MWh, the same import emissions factor the `CA_AB32` rows carry.
Exports earn the same wholesale price and are assigned zero emissions.

Details of both constraint formulations are in
[Interface transmission limits](interface-transmission-limits) and
[Import and export volume limits](model-constraints.md#import-and-export-volume-limits).

## Policy inputs

`opts: [REM-3h]` activates **regional emissions limits** and 3-hour temporal resolution. The
`CA_AB32` rows of `config/policy_constraints/regional_Co2_limits.csv` give an annual CO2 budget
for California from the 2022 CARB Scoping Plan, from 46.6 MtCO2 in 2025 down to 8.68 MtCO2 in
2045, with imported energy charged at 0.428 tCO2/MWh.

```{note}
`config.california.yaml` also populates `SAFE_reservemargin`, `SAFE_regional_reservemargins`
and `erm`. Neither constraint is active as shipped: the SAFE planning-reserve constraint is
switched on by a `SAFE` token in `{opts}` and the energy reserve margin by an `ERM` token, and
`opts` is `REM-3h`. Add the tokens (e.g. `REM-ERM-3h`) to bind them.
```

`technology_capacity_targets.csv` and `portfolio_standards.csv` are wired in but carry mostly
ReEDS-derived and example rows; see [Policies](data-policies.md).

## CPUC baseline benchmark

`run: benchmark_cpuc: true` adds `benchmark_cpuc_baseline` to the workflow targets, writing
`results/<run>/cpuc_benchmark/cpuc_capacity_benchmark.csv` and a deviation heatmap. It compares
installed capacity in `resources/powerplants/powerplants.csv` against the CPUC Baseline
Generator List, by region and technology, per horizon.

The rule is deliberately **network-free**, so a fleet benchmark never drags in a network build.
`run: benchmark_cpuc_horizons: [2026]` is the shipped default: 2026 is the only pure
fleet-vs-fleet comparison, since the CPUC list is a baseline that stays roughly static at later
years while the model expands. An empty list falls back to `scenario: planning_horizons`.

Two reconciliations happen before the sides are comparable.

- **Region.** EIA reports every CAISO plant under the single code `CISO` with no sub-BA split,
  so the benchmark runs at the coarsest resolution both sides support: `CAISO` (CPUC PGE + SCE
  + SDGE vs. model `CISO`), `LADWP` (vs. `LDWP`), `IID`, and `NCNC` (vs. `BANC` + `TIDC`). The
  model side is additionally restricted to `state == "CA"`, because EIA's `CISO` also covers
  the Nevada `CISO-VEA` footprint. The collapse lives in `servm_benchmark_regions.csv`.
- **Technology.** Both sides map into a shared `compare_category` via `servm_tech_map.csv`,
  which carries a `side` column so one file documents both directions. Anything unmapped
  becomes its own `UNMAPPED:<name>` row — a category is never silently dropped.

Both sides are filtered by the same vintage rule: in service by December 31 of the horizon and
not retired by then.

### The `EXCLUDED` row

Rows of the CPUC list whose physical resource sits outside California are split off before
scoring and reported on a pseudo-region row, `EXCLUDED: out-of-state contracted`, broken down
by technology. Its `model_mw` and delta columns are blank: these are contractual ledger
entries that a physically located model cannot carry, so reporting them as model shortfall
would be misleading.

Setting `electricity: remote_contracted_resources: enable: true` adds the EIA-identifiable
subset of exactly those units back into the model — 74 rows totalling 10,873 MW in
`servm_out_of_state_units.csv`, of which 8 rows / 1,910 MW have no `eia_plant_id` and are
skipped with a summary warning. **The benchmark keeps scoring them on the `EXCLUDED` row
regardless**, because it reads `powerplants.csv` rather than the network. When the option is
on, the MW on that row is what has been added back. See
[Out-of-State Contracted Resources](remote_contracted_resources) for the attachment rules and
their four deliberate simplifications.

## Known caveats

**Demand calendar and timezone**

- SERVM strips are in **fixed Pacific Standard Time (UTC−8) with no DST transition**. This is
  not stated in the source files; it was established empirically from the behind-the-meter PV
  solar-noon centroid. The strips are rolled forward 8 hours to UTC.
- SERVM lays each year's 8760 hours on a **synthetic calendar that starts on a Monday**, so
  weekday-versus-weekend hours do not line up with the real weekdays of the planning horizon.
- For a **leap weather year** the SERVM strip contains February 29 and omits December 31, while
  PyPSA-USA's snapshots do the opposite. Every hour after February therefore lands one calendar
  day earlier than in the source file. The strip is mapped positionally onto the network's own
  snapshots, so each planning horizon must carry exactly 8760 snapshots — a truncated snapshot
  window cannot be used with `profile: servm`.

**Footprint**

- The California slivers served by `PACW`, `WALC` and `NEVP`, and the Nevada `CISO-VEA`
  footprint, carry no SERVM load and are dropped.
- `p8` appears in the `region_2` list of every RESOLVE interface row but is itself a California
  zone. In a California-only model it is *inside* the network, so the internal `p8`-`p9` AC
  corridor (~300 MW in the ReEDS/NARIS balancing-area table) carries no trade links and escapes
  the `CAISO_Imports` cap. Simultaneous CAISO imports are understated by roughly that amount.
  This is documented rather than corrected.

**Fleet and benchmark comparability**

- `Gas Cogen/CHP` is a CPUC-only category. EIA technology descriptions have no CHP concept, so
  California gas cogeneration lands in `CCGT`/`OCGT` by prime mover. Expect the model to be
  short in `Gas Cogen/CHP` and long in the two gas buckets by roughly the same amount.
- `Demand Response` and `Pumping Load` are CPUC-only resources with no PyPSA-USA counterpart;
  they are kept as rows with `model_mw == 0`.
- The CPUC `Capmax MW` column is **nameplate** capacity as SERVM sees it, while the model side
  totals `p_nom`. Seasonal derates are applied downstream in `add_electricity`, not in
  `powerplants.csv`, so the benchmark compares nameplate to nameplate — but any summer-rating
  comparison against a third source will not line up with either.
- `honor_planned_retirements: true` drops units at their EIA planned retirement date as of the
  first investment period, so the fleet is smaller than an all-existing-units inventory.

**Not implemented**

- `conventional: ambient_derate` (CPUC SERVM unit-specific hourly thermal derates) is a
  reserved phase-2 option. `enable: true` raises `NotImplementedError`. When it lands it
  *replaces* the EIA-860 seasonal derate rather than stacking on it — stacking an ambient
  derate on a seasonal derate, or on a UCAP-derated capacity credit, double-counts the same
  thermal deficiency.
- `servm_weather_years` with more than one entry (stochastic weather scenarios) raises
  `NotImplementedError`.
