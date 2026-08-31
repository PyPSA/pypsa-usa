(data-generators)=
# Generators & Storage Units

PyPSA-USA utilizes the [Public Utility Data Liberation (PUDL)](https://catalystcoop-pudl.readthedocs.io/en/latest/index.html) project database as the core source for generator and storage device data. The PUDL database aggregates and cleans data from various agencies, including the Energy Information Administration (EIA), Federal Energy Regulatory Commission (FERC), and the National Renewable Energy Laboratory (NREL). This integration supports reproducibility and ensures continuity as new reports are released. The PUDL data is supplemented with data from the WECC Anchor Data Set (ADS) as well as the EIA API.

## Modeling Generators and Energy Storage

PyPSA-USA provides unit-level generator data on heat rates, plant fuel costs, seasonal derating, power and energy capacities, fuel types, and more. While generator data is input at an EIA unit level, the model clusters generators by their technology type (named `carrier`) to reduce the computational cost of optimization models. PyPSA-USA generators are clustered to Combined-Cycle Gas Turbines (CCGT), Open-Cycle Gas Turbines (OCGT), coal, CCGTs with Carbon Capture and Storage (CCS), coal with CCS, oil, Hydrogen Combustion Turbines, Nuclear (large-scale AP1000), Small Modular Reactor Nuclear, biomass, traditional geothermal, waste, hydro, utility-scale solar, onshore wind, fixed-bottom offshore wind, floating offshore wind, and Enhanced Geothermal Systems (EGS). Storage Units include Li-ion battery energy storage systems (2-10 hour storage capacity) and Pumped-Hydro Storage (8-12 hour storage capacity). Users have control over the clustering settings using the configuration settings described in the [configuration section](./config-configuration.md).

## Fuel Costs and Heat-rates

In production cost-minimizing optimization models, a generator’s marginal cost to produce electricity is a primary driver of dispatch decisions and electricity prices. However, generator fuel prices and efficiencies are not uniformly available across the United States, and generators often enter into bilateral contracts that are not directly correlated with wholesale fuel prices. To address these challenges, PyPSA-USA provides a few options for the source of generator fuel prices. Generator heat-rates are also assimilated from multiple data sources by selecting the highest-quality available data source for a given generation unit before falling back to coarser data.

- **Fuel Price Integration**:
    - Fuel prices are collected and overlaid to select the highest resolution available, defaulting to coarser data if necessary.
    - Single-point unit-level generator fuel efficiencies are sourced from a CEMS-based dataset [D. Suri et. al.](https://arxiv.org/pdf/2408.05209).
    - Monthly unit-level fuel prices and additional plant efficiencies are collected via PUDL EIA-923.

- **Data Imputation**:
    - Missing data is imputed using capacity-weighted averages calculated by NERC region and unit technology type.
    - Wholesale daily natural gas prices for fuel regions across the WECC are imputed using CAISO OASIS data.
    - Monthly fuel prices for coal and natural gas, spatially resolved by state, are supplemented by data from the EIA.
    - For technologies like biomass and nuclear, where fuel prices are not available from other sources, projected fuel costs from the NREL ATB are used.

- **Future Fuel Costs**:
    - Forecasted annual fuel prices are imported from the EIA's Annual Energy Outlook (AEO).

(remote_contracted_resources)=
## Out-of-State Contracted Resources (California)

A California-scoped run (`model_topology: include: reeds_state: ['CA']`) attaches only the
plants that fall inside the model footprint, so every resource that is *physically* outside
California is dropped by `filter_plants_by_region` — including roughly 10.9 GW that the CPUC
ledger attributes to California load regions and that genuinely serves California load: SCE's
635 MW share of Palo Verde, LADWP's 1,185 MW of Intermountain and 566 MW of Apex, the Hoover
entitlements, and about 2 GW of Arizona and Nevada solar and battery contracts.

Setting `electricity: remote_contracted_resources: enable: true` adds them back. For each row
of `workflow/repo_data/CPUC/servm_out_of_state_units.csv`, `add_electricity` looks the unit's
`eia_plant_id` up in the same PUDL fleet build that supplies the rest of the model (captured
*before* the regional filter runs, so the plant is still present), and attaches a single
generator — or a StorageUnit for battery contracts — at a California bus:

- **Capacity** is the *contracted* MW (`capmax_mw`), capped at the live capacity the physical
  plant actually has. Retirement and vintage filtering is respected: a contract whose plant has
  no live units at the first investment period is skipped.
- **Bus** is the max-LAF bus of the row's SERVM region, taken from the same
  `build_servm_load_weights` table the demand path uses. This is why the option requires
  `demand: profile: servm`.
- **Techno-economics** (heat rate, efficiency, marginal cost, ramp rates, seasonal derates,
  unit-commitment parameters) are capacity-weighted over the plant's constituent units, so a
  contract inherits the same data the physical fleet carries. Generators are named
  `R <cpuc_unit_name>`, mirroring the `C` prefix on the conventional fleet, and are attached
  **non-extendable**: they are existing contracts, not expansion candidates.

Four simplifications are deliberate and worth knowing before reading results:

1. **Remote wind and solar borrow the capacity-factor profile of their California attachment
   bus**, not of their physical location. Desert-southwest solar is in reality better
   correlated with CAISO's own solar than this implies. If the attachment bus has no profile
   for that carrier, the network mean profile for the carrier is used; if the network has no
   such profile at all the contract is dropped rather than given an implicit 100 % capacity
   factor.
2. **Remote hydro (the Hoover entitlements) is attached as a firm, energy-unlimited
   generator** carrying only the EIA-860 seasonal derate. CRSP hydrology, Lake Mead elevation
   and the monthly energy schedules that actually bound those entitlements are not modelled.
3. **Rows with no `eia_plant_id` are skipped** — Mexicali TDM/LR2, Powerex/BC, the ESJ Baja
   wind contracts and a few pure entitlement rows (about 1.9 GW). They have no PUDL
   techno-economics to inherit and remain in import-machinery territory; the run logs a single
   summary warning naming them and their MW.
4. **The CPUC baseline benchmark keeps scoring these units on its `EXCLUDED` row**, because it
   reads `powerplants.csv` rather than the network. When this option is on, the MW on that row
   is what has been added back into the model.

## Renewable Resources

(renewable_cfs)=
### Renewable Capacity Factors

PyPSA-USA provides two sources for solar and wind capacity-factor time series, selected via
`renewable.dataset` in the configuration (`godeeep` by default). They are interchangeable from
the perspective of downstream rules (both produce the same `profile_{technology}_s{simpl}.nc`
output schema), but they originate from different upstream datasets and embody different
exclusion assumptions.

#### GODEEEP (default)

The default capacity-factor source (`renewable.dataset: godeeep`) is the [GODEEEP](https://www.pnnl.gov/projects/godeeep) dataset — regional-climate-model capacity factors developed at Pacific Northwest National Laboratory under the Grid Operations, Decarbonization, Environmental and Energy Equity Platform. Designed for multi-year climate-change scenario studies, GODEEEP provides hourly solar PV and 125 m hub-height wind capacity factors on a 12 km Lambert Conformal grid for:

- **One historical year** (2012) calibrated against observed weather.
- **Four future climate scenarios** — `rcp45hotter`, `rcp45cooler`, `rcp85hotter`, `rcp85cooler` — under the RCP4.5 and RCP8.5 emissions pathways, downscaled with two GCM ensemble members per pathway.
- **Three planning horizons** (2030, 2040, 2050) per future scenario, drawn from contiguous 20-year (wind) or 40-year (solar) windows.

The raw GODEEEP files are large (~4.4 GB per `(tech, scenario, year)` triple). PyPSA-USA consumes a uint8-quantized + zlib-compressed variant (~350 MB for solar, ~800 MB for wind) published as 10 Zenodo records keyed by `(tech, scenario)`. The compressed files are pulled automatically at runtime by `scripts/zenodo_downloader.py`.

GODEEEP capacity factors are re-aggregated to PyPSA-USA bus polygons using a runtime weighting step:

1. **Per-cell availability raster** — a fraction in [0, 1] for each 12 km GODEEEP cell, derived from NREL reV supply-curve availability scenarios. Three access scenarios are supported: `reference` (least restrictive, ~5× more permissive than Atlite+CORINE), `limited` (closest to the Atlite+CORINE baseline), and `open` (most permissive, for sensitivity studies). Optional overlays apply the California Energy Commission Wind/Solar BaseScreen (`_cec`, CA-only) and BOEM offshore wind planning areas (`_boem`, offshore).
2. **Cell→bus mapping** computed once per bus layout from a county-level shapefile (cached on disk; ~14 min per interconnect at county resolution).
3. **Per-bus rollup** of `weight`, `p_nom_max`, `potential`, `average_distance`, and (for offshore) `underwater_fraction` from NREL supply-curve site locations within each bus polygon.

The availability rasters and per-bus capacity rollups are published as a separate Zenodo record ([10.5281/zenodo.20127899](https://doi.org/10.5281/zenodo.20127899)) and downloaded on first run.

See [`renewable: godeeep`](godeeep_cf) under Model Configuration for the full set of config knobs.

#### Atlite (legacy alternative)

As an alternative (`renewable.dataset: atlite`), PyPSA-USA leverages the [Atlite](https://atlite.readthedocs.io) tool to compute capacity factors at runtime from decades of weather data. Atlite estimates hourly renewable resource availability across the United States from ERA5 reanalysis data, typically at a spatial resolution of 30 km² cells. Within PyPSA-USA, users can configure:

- **Weather Year**
- **Turbine Type**
- **Solar Array Type**
- **Land-Use Parameters**
- **Availability Simulation Parameters**

The hourly renewable capacity factors calculated by Atlite are weighted based on land-use
availability factors. This ensures that areas unsuitable for specific technology types do not
disproportionately affect the renewable resource capacity assigned to each node. Profiles are
computed directly at the model's `{simpl}` cluster resolution: `build_renewable_profiles`
consumes the clustered bus regions (`regions_onshore_s{simpl}.geojson` /
`regions_offshore_s{simpl}.geojson`) and the `busmap_s{simpl}.csv` mapping, and writes one
profile per technology to `resources/profiles/{interconnect}/profile_{technology}_s{simpl}.nc`.

**Enhanced Geothermal (EGS) and Pumped Hydro Storage (PHS)**: These resources require more complex modeling due to subsurface and surface characteristics. Regional supply curves for these resources, including capital costs and technical capacity, are incorporated from specialized datasets.

- **PHS**: Uses data from the [NREL Closed-Loop PHS dataset](https://www2.nrel.gov/gis/psh-supply-curves).
- **EGS**: Availability data is sourced from [FGEM](https://fgem.readthedocs.io/en/latest/), with further details to be provided in a forthcoming paper.


## Data
```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,22,22,22
   :file: datatables/generators.csv
```
Renewables Data:
```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,22,22,22
   :file: datatables/renewables.csv
```
