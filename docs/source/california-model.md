(california-model)=
# California Model (CPUC SERVM)

PyPSA-USA ships a maintained, runnable California-only configuration at
`workflow/repo_data/config/config.california.yaml`. It is a sparse overlay on the layered base
config and is run in place, with `--configfile repo_data/config/config.california.yaml`; it is
*not* copied into `workflow/config/` by `init_pypsa_usa.sh`, which seeds only the three
per-user files (`config.default.yaml`, `config.api.yaml`, `config.slurm.yaml`). This page is
the reference for that configuration: what data goes into it, which weather years are
available for each ingredient, and which simplifications the results carry.

The model is a **California carve-out of the Western Interconnection**, not a standalone
network. The footprint is the four California ReEDS zones (`p8`, `p9`, `p10`, `p11`); the rest
of WECC is not modelled as buses but as external import/export regions at the footprint
boundary — as shipped, an import *generator* behind an unpriced interface link (see
[Trade with the rest of WECC](#trade-with-the-rest-of-wecc)). Load is the
California Public Utilities Commission's 2026 Integrated Resource Planning hourly forecast,
produced with the SERVM production-cost model.

## Running it

```bash
cd workflow

# full pipeline (build + solve + figures)
uv run snakemake -j1 --configfile repo_data/config/config.california.yaml

# data model only, no solver
uv run snakemake data_model -j1 --configfile repo_data/config/config.california.yaml
```

The first build downloads one ~118 MB SERVM load CSV per planning horizon from
`files.cpuc.ca.gov` (`retrieve_cpuc_servm_load`), the CPUC Baseline Generator List workbook
(`retrieve_cpuc_baseline_generators`), and the GODEEEP / NREL land-access artifacts from
Zenodo. Imports are priced at a **flat `imports: costs: 60` \$/MWh**, so no EIA key is needed
for import prices; the `wholesale` setting is available but the EIA series it pulls is
retail-priced ([issue #807](https://github.com/PyPSA/pypsa-usa/issues/807)). Exports are left
at the base default `exports: costs: wholesale`, which does call the EIA API, so a key in
`workflow/config/config.api.yaml` (or `$EIA_API_KEY`) is still required unless exports are
given a flat price too.

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
| **Trade prices** | Imports: flat \$60/MWh. Exports: EIA wholesale electricity prices for the weather year (`wholesale` is available for imports too, but is retail-priced — [#807](https://github.com/PyPSA/pypsa-usa/issues/807)) | `imports: costs: 60`; `exports: costs: wholesale` | flat value in the config; EIA API (`config/config.api.yaml`) for exports |
| **Out-of-state contracts** | CPUC ledger of physically out-of-state units serving California load (75 rows, 11,173 MW) | `electricity: remote_contracted_resources: enable: true` | `workflow/repo_data/CPUC/servm_out_of_state_units.csv` |
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
`config.california.yaml` ships with `renewable_scenarios: ['historical']` and
`planning_horizons: [2030, 2035, 2040, 2045]`, precisely so that all four SERVM horizons can
be modelled: historical profiles are indexed by the weather year, not by the horizon. If you
switch to a climate scenario (`rcp45hotter`, `rcp45cooler`, `rcp85hotter`, `rcp85cooler`) you
must also restrict `planning_horizons` to `[2030, 2040]` — only 2030, 2040 and 2050 are
published per `(tech, scenario)`, and the registry fails at parse time for 2035 and 2045.
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
both enabled, adding an external bus per neighbouring flowgate zone (`{zone}_imports`,
`{zone}_exports`) and trade links from those buses into the footprint.

**The shipped representation is `imports: representation: generator`.** Each external
`{zone}_imports` bus carries a generic import **generator** (carrier `unspecified_imports`)
rated at that zone's total inbound interface capacity and priced by `imports: costs`
(the flat \$60/MWh above); the `imports`-carrier links from the external bus into the
footprint are *unpriced* and rated at the NARIS flowgate capacity, so the price sits on the
generator and the link is pure transfer capacity. The CPUC-contracted out-of-state units are
attached at that same external bus, i.e. **behind** the model boundary, so their deliveries
traverse an `imports` link and are metered against the interface and volume caps below.
Import CO2 (`imports: co2_emissions: 0.428` tCO2/MWh) is carried by the
`unspecified_imports` carrier, and the `imports` carrier is set to zero — PyPSA attributes
primary-energy emissions to generators and stores, never to links, so nothing is
double-counted.

The alternative is `representation: store` (the base default): the external bus carries a
bottomless `Store` on the `imports` carrier instead of a generator, the links into the
footprint are the priced element, and the contracted units are attached at California buses,
where they look like in-state generation and bypass both trade caps. Both modes keep the link
carriers `imports`/`exports` untouched, and the export half is identical in both: the negative
export price stays on the export link, the absorbing `Store` behind it is free, and exported
energy is assigned zero emissions. Imports and exports deliberately use separate external
buses, so the priced import generator cannot sell straight into the export sink.

Three bounds apply to trade.

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

The import emissions factor `co2_emissions: 0.428` tCO2/MWh is the same one the `CA_AB32`
rows carry.

Details of both constraint formulations are in
[Interface transmission limits](interface-transmission-limits) and
[Import and export volume limits](model-constraints.md#import-and-export-volume-limits).

## Policy inputs

`opts: [REM-3h]` activates **regional emissions limits** and 3-hour temporal resolution. The
`CA_AB32` rows of `config/policy_constraints/regional_Co2_limits.csv` give an annual CO2 budget
for California from the 2022 CARB Scoping Plan, from 46.6 MtCO2 in 2025 down to 8.68 MtCO2 in
2045, with imported energy charged at 0.428 tCO2/MWh.

```{note}
The layered base config (`config.default.yaml`) carries `SAFE_reservemargin`,
`SAFE_regional_reservemargins` and `erm`; `config.california.yaml` does not override them and
neither is active as shipped, because `opts` is `REM-3h`.

The **energy reserve margin** is real: an `ERM` token in `{opts}` dispatches
`add_ERM_constraints` in `solve_network`, so `opts: REM-ERM-3h` binds it against the
`electricity: erm` settings.

There is **no SAFE constraint on develop.** A `SAFE` token only sets
`config["solving"]["constraints"]["SAFE"] = True` in `_helpers.py`, and nothing reads that
flag — no solve-time constraint exists. The `SAFE_*` keys are therefore currently dead;
treat them as reserved until a planning-reserve constraint is implemented.
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
subset of exactly those units back into the model. `servm_out_of_state_units.csv` holds 75
rows totalling 11,172.8 MW, of which 9 rows / 2,210.0 MW have no `eia_plant_id` and are
skipped with a summary warning (leaving 66 rows / 8,962.8 MW eligible; rows whose EIA plants
have no live generator in `powerplants.csv` are skipped too, and each unit's `p_nom` is
`min(capmax_mw, live plant capacity)`).

Where those units land depends on `electricity: imports: representation`. Under the shipped
`generator` mode they are *not* attached at California buses: `add_electricity` serializes
them into a bundle and `add_extra_components` attaches them at the external `{zone}_imports`
buses — behind the model boundary — so their output crosses an `imports` link and is metered
against the interface and volume caps. Under `store` mode they are attached directly at the
California bus derived from the contracting SERVM region, inside the footprint and outside
both caps. **The benchmark keeps scoring them on the `EXCLUDED` row
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
