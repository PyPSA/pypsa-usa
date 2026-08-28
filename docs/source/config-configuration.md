(config)=
# Configuration

(run_cf)=
## `run`

It is common conduct to analyse energy system optimisation models for **multiple scenarios** for a variety of reasons,
e.g. assessing their sensitivity towards changing the temporal and/or geographical resolution or investigating how
investment changes as more ambitious greenhouse-gas emission reduction targets are applied.

The `run` section is used for running and storing scenarios with different configurations which are not covered by [wildcards](#wildcards). It determines the path at which resources, networks and results are stored. Therefore the user can run different configurations within the same directory.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-after: # docs : RUN
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/run.csv
```


(scenario_cf)=
## `scenario`

The `scenario` section is used for setting the wildcards and defining planning horizon settings. All configurations within this section are described in [wildcards](#wildcards) with the exception of planning_horizons and foresight.

Planning horizons determines which year(s) of future demand forecast to use for your planning model. To build a multi-investment period model set multiple `planning_horizons:` years. The `foresight:` option specifies whether perfect foresight or myopic foresight optimization model is developed. In perfect foresight, a monolithic model is developed where all `planning_horizons` specified are optimized at once, e.g. future horizon values of costs and demand are incorporated into decisions made in earlier planning horizons. Myopic optimization solves each planning horizon sequentially, and passes the results forward.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-after: # docs : SCENARIO
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/scenario.csv
```

(model_topology_cf)=
## `model_topology`

The `model_topology` section selects the transmission backbone and the spatial zones the final
network is aggregated to. `transmission_network` chooses between the ReEDS zonal backbone and the
TAMU synthetic nodal network; `topological_boundaries` sets the zone type used after clustering
(county, REeDS zone, state, or balancing area). Use `include` to subset the modeled footprint to
specific zones, states, or balancing authorities (mixed zone types are not supported), and
`aggregate` to pre-aggregate buses into larger regions. `interface_transmission_limits` applies
NARIS2024 inter-regional transfer capacity limits and requires the ReEDS backbone.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-after: # docs : MODEL_TOPOLOGY
   :end-before: # docs :
```

(enable_cf)=
## `enable`

Top-level feature flags. `build_cutout` switches between consuming the prebuilt atlite cutouts
downloaded from Zenodo (the default) and building a fresh cutout from raw ERA5 data, which is slow
and requires a CDS API key. The optional `custom_busmap` flag makes `cluster_network` read a
user-provided busmap from `data/{interconnect}/custom_busmap_{clusters}.csv` instead of computing
one; custom busmaps must key on the bus IDs produced by the `cluster_simpl` stage.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-after: # docs : ENABLE
   :end-before: # docs :
```

(pudl_cf)=
## `pudl_path`

Points the build scripts at a versioned release of the [PUDL](https://catalyst.coop/pudl/)
(Public Utility Data Liberation) parquet outputs, which supply the EIA-860, EIA-923, and CEMS
tables used to build powerplants and fuel prices. Bump the version tag to pull fresher data, or
point it at a local `file://` mirror for offline runs. This key lives in `config.common.yaml`
since it rarely changes per scenario.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.common.yaml
   :language: yaml
   :start-after: # docs : PUDL
   :end-before: # docs :
```

(snapshots_cf)=
## `snapshots`

Specifies the temporal range to build an energy system model for as arguments to [`pandas.date_range`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html)

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-after: # docs : SNAPSHOTS
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/snapshots.csv
```

(renewable_weather_years_cf)=
## `renewable_weather_years`

Sets the weather year(s) the renewable capacity-factor time series are built for. With
`renewable.dataset: atlite` any year with a matching cutout works; with `godeeep` this key is only
used when `renewable_scenarios` is `historical` — for future climate scenarios the year is taken
from `planning_horizons` instead. The optional commented `renewable_weather_years_by_horizon`
mapping assigns a different weather year to each planning horizon in multi-horizon atlite runs.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-after: # docs : RENEWABLE_WEATHER_YEARS
   :end-before: # docs :
```

(renewable_scenarios_cf)=
## `renewable_scenarios`

Selects the GODEEEP climate scenario used for renewable capacity factors — one historical record
or four future climate projections. Only consumed when `renewable.dataset: godeeep`; see
[`renewable: godeeep`](#godeeep_cf) for how scenarios, years, and snapshots interact.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-after: # docs : RENEWABLE_SCENARIOS
   :end-before: # docs :
```

(renewable_snapshots_cf)=
## `renewable_snapshots`

Sets the month/day window sampled within each weather year for GODEEEP-based runs. In
multi-horizon godeeep models the year itself comes from each planning horizon automatically, so
these knobs only control how much of the 8760-hour capacity-factor series is used;
`end_inclusive: true` keeps the full end day rather than stopping at midnight.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-after: # docs : RENEWABLE_SNAPSHOTS
   :end-before: # docs :
```

(atlite_cf)=
## `atlite`

Define and specify the `atlite.Cutout` used for calculating renewable potentials and time-series. All options except for `features` are directly used as [`cutout parameters`](https://atlite.readthedocs.io/en/latest/ref_api.html#cutout)

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.common.yaml
   :language: yaml
   :start-after: # docs : ATLITE
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/atlite.csv
```

(electricity_cf)=
## `electricity`

Specifies the types of generators that are included in the network, which are extendable, and the CO2 base for which the optimized reduction is relative to.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-after: # docs : ELECTRICITY
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/electricity.csv
```

(renewable_cf)=
## `renewable`

Per-technology resource and land-availability settings live in `config.common.yaml` and feed
`build_renewable_profiles`. Note that when `renewable.dataset: godeeep` is selected, the
`corine`, `natura`, and `cec` land screens below are bypassed in favor of the NREL reV
`renewable_land_access` exclusions (see [`renewable: godeeep`](#godeeep_cf)).

### `solar`
```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.common.yaml
   :language: yaml
   :start-after: # docs : SOLAR
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/solar.csv
```

### `onwind`
```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.common.yaml
   :language: yaml
   :start-after: # docs : ONWIND
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/onwind.csv
```

### `offwind`

Fixed-bottom offshore wind, screened to water depths up to `max_depth` (60 m by default) within
the configured shore-distance band.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.common.yaml
   :language: yaml
   :start-after: # docs : OFFWIND
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/offwind.csv
```

### `offwind_floating`

Floating offshore wind, screened to the depth band between `min_depth` and `max_depth`
(60–1300 m by default, following NREL fy22osti/83650) within the configured shore-distance band.
The BOEM offshore-wind planning-area screen (`boem_screen`) is on by default for floating sites.
The same option keys as fixed-bottom `offwind` apply.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.common.yaml
   :language: yaml
   :start-after: # docs : OFFWIND_FLOATING
   :end-before: # docs :
```

### `hydro`

Configures the hydro fleet attached from EIA-860 capacity and atlite inflow time series.
`carriers` selects which hydro types are modeled: `ror` (run-of-river), `PHS` (pumped hydro
storage), and `hydro` (reservoir hydro). The optional commented `PHS_max_hours` overrides the
pumped-storage duration that otherwise comes from the cost tables.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.common.yaml
   :language: yaml
   :start-after: # docs : HYDRO
   :end-before: # docs :
```

### `EGS`

Options for Enhanced Geothermal Systems supply curves (used when `EGS` is listed under
`electricity: extendable_carriers: Generator`). `dispatch` selects baseload (constant output) or
flexible (dispatchable) operation, `drilling_cost` picks the `base` or `advanced` drilling-cost
column of the EGS supply curves, and `seismic_exclusion` applies the seismic-risk mask to
candidate sites. The underlying supply-curve and profile methodology is to be detailed in a
forthcoming publication.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.common.yaml
   :language: yaml
   :start-after: # docs : EGS
   :end-before: # docs :
```

(godeeep_cf)=
### `renewable: godeeep`

PyPSA-USA supports two sources for renewable capacity-factor time series, selected via `renewable.dataset` in `config.common.yaml`:

- `atlite` — runtime computation of capacity factors from ERA5 cutouts, weighted by Copernicus / CORINE land-use exclusions (the legacy default; see [Renewable Capacity Factors](renewable_cfs) under Model Data).
- `godeeep` — pre-computed regional climate-model capacity factors from the [GODEEEP](https://www.pnnl.gov/projects/godeeep) (Grid Operations, Decarbonization, Environmental and Energy Equity Platform) dataset, weighted at runtime by NREL reV land-access exclusions.

The `godeeep` path consumes two file families published on Zenodo, downloaded automatically by `scripts/zenodo_downloader.py` on first run:

- **Compressed GODEEEP capacity factors** — per-cell hourly capacity factors on the GODEEEP Lambert Conformal grid, uint8-quantized and zlib-compressed (~12× smaller than the raw aggregated files). One Zenodo record per `(tech, scenario)`:
   - solar: [historical](https://doi.org/10.5281/zenodo.20127513), [rcp45hotter](https://doi.org/10.5281/zenodo.20127523), [rcp45cooler](https://doi.org/10.5281/zenodo.20127562), [rcp85hotter](https://doi.org/10.5281/zenodo.20127589), [rcp85cooler](https://doi.org/10.5281/zenodo.20127633)
   - wind (125 m): [historical](https://doi.org/10.5281/zenodo.20127520), [rcp45hotter](https://doi.org/10.5281/zenodo.20127545), [rcp45cooler](https://doi.org/10.5281/zenodo.20127572), [rcp85hotter](https://doi.org/10.5281/zenodo.20127604), [rcp85cooler](https://doi.org/10.5281/zenodo.20127645)
- **NREL land-access artifacts** ([10.5281/zenodo.20127899](https://doi.org/10.5281/zenodo.20127899)) — `avail_{tech}_{access}[_cec|_boem].nc` per-cell availability rasters and `caps_{tech}_{access}[_cec|_boem].nc` per-bus rollups (`weight`, `p_nom_max`, `potential`, `average_distance`, and `underwater_fraction` for offshore).

#### Configuring a godeeep run

A complete godeeep configuration requires four config blocks beyond the standard `electricity:` / `clustering:` settings:

1. **Dataset selection** (`config.common.yaml`):

   ```yaml
   renewable:
     dataset: godeeep    # set to atlite for the ERA5 + CORINE workflow
   ```

2. **Scenario and year selection**. The GODEEEP dataset has one historical record (2012) and four future climate scenarios (`rcp45hotter`, `rcp45cooler`, `rcp85hotter`, `rcp85cooler`) at planning horizons 2030 / 2040 / 2050.

   ```yaml
   renewable_scenarios: ["rcp45cooler"]   # one of: historical | rcp45hotter | rcp45cooler | rcp85hotter | rcp85cooler
   renewable_weather_years: [2012]        # used only when scenario == historical
   ```

   Future-scenario years come from the `planning_horizons` wildcard (under `scenario:`); the historical year comes from `renewable_weather_years`.

3. **Snapshots** — the temporal slice within the chosen year. For godeeep this controls how much of the 8760-hour GODEEEP CF is sampled:

   ```yaml
   renewable_snapshots:
     start_month: 1
     start_day: 1
     end_month: 12
     end_day: 31
     end_inclusive: true    # include the full end day, not just up to 00:00
   ```

4. **NREL land-access exclusions** (required when `renewable.dataset: godeeep`; the workflow raises if `renewable_land_access` is unset):

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.common.yaml
   :language: yaml
   :start-after: # docs : NREL_EXCLUSION
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/nrel_exclusion.csv
```

The `_cec` and `_boem` variants overlay additional regulatory screens on top of the base NREL availability raster: California Energy Commission Wind/Solar BaseScreen for onshore solar/wind in CA, and BOEM offshore wind planning areas for offshore. Outside their applicable region the variant equals the base.

(offshore_shape_cf)=
## `offshore_shape`

Selects the offshore region polygons used to delineate offshore wind resource areas: `eez` uses
the federal Exclusive Economic Zone shapes (the default, covering all coasts), while `ca_osw`
restricts offshore development to the California offshore wind areas.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.common.yaml
   :language: yaml
   :start-after: # docs : OFFSHORE_SHAPE
   :end-before: # docs :
```

(offshore_network_cf)=
## `offshore_network`

Controls the density of the synthetic offshore network: `bus_spacing` sets the distance in meters
between adjacent offshore buses, which determines how many offshore wind connection points are
created along the offshore shapes.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.common.yaml
   :language: yaml
   :start-after: # docs : OFFSHORE_NETWORK
   :end-before: # docs :
```

(lines_cf)=
## `lines`
```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-after: # docs : LINES
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/lines.csv
```

(links_cf)=
## `links`

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-after: # docs : LINKS
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/links.csv
```

(co2_cf)=
## `co2`

The `co2` section specifies whether the model may use underground storage to sequester captured CO2 or not. In case underground storage is specified, each node (composing the network) has a specific storage potential and a cost associated with it. The storage potential (in tonnes) is calculated by aggregating all the underlying storage potentials of the U.S. counties encompassed in the node's geographical area. Counties that are only partially covered by the node's geographical area have their potential fractionated accordingly. The storage cost (in $/tonne) is calculated by weighting the potential with the cost of each county encompassed. The dataset containing information about underground CO2 storage potentials and costs at a county level (and used in PyPSA-USA) was provided by Edna Calzado at The University of Texas (Austin), which was derived from the Roads to Removal project (https://roads2removal.org). To get an illustration, enabling underground co2 storage for a sector-less network will render its topography similar to <a href = "_static/CCTS/pypsa-usa_sector-less_with_underground_co2_storage.svg" target = "_blank">this</a>, while for a sector-based network will render its topography similar to <a href = "_static/CCTS/pypsa-usa_sector-based_with_underground_co2_storage.svg" target = "_blank">this</a>. As a reference, disabling underground co2 storage (i.e. no CCTS), a sector-less network has a topography similar to <a href = "_static/CCTS/pypsa-usa_sector-less_without_CCTS.svg" target = "_blank">this</a>, while a sector-based network has a topography similar to <a href = "_static/CCTS/pypsa-usa_sector-based_without_CCTS.svg" target = "_blank">this</a>.

:::{figure} _static/CCTS/pypsa-usa_sector-less_with_underground_co2_storage.svg
:width: 90%
:alt: Sector-less network topology with underground CO2 storage

Sector-less network with underground CO2 storage enabled.
:::

:::{figure} _static/CCTS/pypsa-usa_sector-based_with_underground_co2_storage.svg
:width: 90%
:alt: Sector-based network topology with underground CO2 storage

Sector-based network with underground CO2 storage enabled.
:::

:::{figure} _static/CCTS/pypsa-usa_sector-less_without_CCTS.svg
:width: 90%
:alt: Sector-less network topology without CCTS

Reference: sector-less network without CCTS.
:::

:::{figure} _static/CCTS/pypsa-usa_sector-based_without_CCTS.svg
:width: 90%
:alt: Sector-based network topology without CCTS

Reference: sector-based network without CCTS.
:::

In addition, the section specifies whether the model may transport captured CO2 between nodes or not. In case transportation is specified, a network of CO2 pipelines is built based on the electricity grid layout represented in PyPSA-USA to determine where/how to build pipelines to connect nodes. To get an illustration, enabling co2 transport for a sector-less network will render its topography similar to <a href = "_static/CCTS/pypsa-usa_sector-less_with_underground_co2_storage_and_co2_transport.svg" target = "_blank">this</a>, while for a sector-based network will render its topography similar to <a href = "_static/CCTS/pypsa-usa_sector-based_with_underground_co2_storage_and_co2_transport.svg" target = "_blank">this</a>.

:::{figure} _static/CCTS/pypsa-usa_sector-less_with_underground_co2_storage_and_co2_transport.svg
:width: 90%
:alt: Sector-less network topology with CO2 storage and transport

Sector-less network with underground CO2 storage and CO2 transport.
:::

:::{figure} _static/CCTS/pypsa-usa_sector-based_with_underground_co2_storage_and_co2_transport.svg
:width: 90%
:alt: Sector-based network topology with CO2 storage and transport

Sector-based network with underground CO2 storage and CO2 transport.
:::

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-after: # docs : CO2
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/co2.csv
```

(dac_cf)=
## `dac`

The `dac` section specifies whether the model should use Direct Air Capture (DAC) or not to capture emitted CO2. DAC may operate in a multitude of scenarios depending on different granularities (or scopes) and whether the model is based on sectors or not.

For a sector-less model:

   - When `granularity` is set to `node`, each node (composing the network) has a specific "air atmosphere" into which all the processes belonging to the node emit CO2. For each node, DAC is built to capture CO2 from its "air atmosphere". To get an illustration, setting `granularity` to `node` will render the topography of a sector-less network similar to <a href = "_static/CCTS/pypsa-usa_sector-less_with_underground_co2_storage_and_co2_transport_and_dac_node-based.svg" target = "_blank">this</a>.

   - When `granularity` is set to `state`, each U.S. state (represented in the network) has a specific "air atmosphere" into which all the processes of the nodes belonging to the state emit CO2. For each node, DAC is built to capture CO2 from the state "air atmosphere" it belongs to. To get an illustration, setting `granularity` to `state` will render the topography of a sector-less network similar to <a href = "_static/CCTS/pypsa-usa_sector-less_with_underground_co2_storage_and_co2_transport_and_dac_state-based.svg" target = "_blank">this</a>.

   - When `granularity` is set to `nation`, the model only has one single "air atmosphere" into which all the processes of all the nodes emit CO2. For each node, DAC is built to capture CO2 from this "air atmosphere". To get an illustration, setting `granularity` to `nation` will render the topography of a sector-less network similar to <a href = "_static/CCTS/pypsa-usa_sector-less_with_underground_co2_storage_and_co2_transport_and_dac_nation-based.svg" target = "_blank">this</a>.

:::{figure} _static/CCTS/pypsa-usa_sector-less_with_underground_co2_storage_and_co2_transport_and_dac_node-based.svg
:width: 90%
:alt: Sector-less network with node-granularity DAC

Sector-less network with DAC at `node` granularity.
:::

:::{figure} _static/CCTS/pypsa-usa_sector-less_with_underground_co2_storage_and_co2_transport_and_dac_state-based.svg
:width: 90%
:alt: Sector-less network with state-granularity DAC

Sector-less network with DAC at `state` granularity.
:::

:::{figure} _static/CCTS/pypsa-usa_sector-less_with_underground_co2_storage_and_co2_transport_and_dac_nation-based.svg
:width: 90%
:alt: Sector-less network with nation-granularity DAC

Sector-less network with DAC at `nation` granularity.
:::

For a sector-based model:

   - When `granularity` is set to `node`, each sector/node pair has a specific "air atmosphere" into which all the processes belonging to the sector/node pair emit CO2. For each sector/node pair, DAC is built to capture CO2 from its "air atmosphere". To get an illustration, setting `granularity` to `node` will render the topography of a sector-based network similar to <a href = "_static/CCTS/pypsa-usa_sector-based_with_underground_co2_storage_and_co2_transport_and_dac_node-based.svg" target = "_blank">this</a>.

   - When `granularity` is set to `state`, each sector/U.S. state pair has a specific "air atmosphere" into which all the processes of the nodes belonging to the state emit CO2. For each sector/node pair, DAC is built to capture CO2 from the sector/state pair's "air atmosphere" it belongs to. To get an illustration, setting `granularity` to `state` will render the topography of a sector-based network similar to <a href = "_static/CCTS/pypsa-usa_sector-based_with_underground_co2_storage_and_co2_transport_and_dac_state-based.svg" target = "_blank">this</a>.

   - Given that a `granularity` set to `nation` does not make sense in a sector-based model, it defaults to `node` in this case.

:::{figure} _static/CCTS/pypsa-usa_sector-based_with_underground_co2_storage_and_co2_transport_and_dac_node-based.svg
:width: 90%
:alt: Sector-based network with node-granularity DAC

Sector-based network with DAC at `node` granularity.
:::

:::{figure} _static/CCTS/pypsa-usa_sector-based_with_underground_co2_storage_and_co2_transport_and_dac_state-based.svg
:width: 90%
:alt: Sector-based network with state-granularity DAC

Sector-based network with DAC at `state` granularity.
:::

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-after: # docs : DAC
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/dac.csv
```

(costs_cf)=
## `costs`

Selects the capital- and operating-cost assumptions: the NREL Annual Technology Baseline (ATB)
scenario used for capex/FOM, the EIA Annual Energy Outlook (AEO) case used for fuel-price
escalation, and policy incentives (PTC/ITC modifiers, emission prices, per-carrier availability
years and build-rate caps). Processed cost tables are written to `resources/costs/costs_{year}.csv`.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-after: # docs : COSTS
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/costs.csv
```

(ucap_cf)=
## `ucap`

The `ucap` section applies an Unforced Capacity (UCAP) derate to conventional generators for
resource-adequacy accounting: each carrier's capacity is derated by its Forced Outage Rate (FOR),
implemented as `p_max_pu = 1 - FOR`. Enable it when reserve-margin constraints should be met with
outage-derated rather than installed capacity. Rates are given in percent per carrier.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.common.yaml
   :language: yaml
   :start-after: # docs : UCAP
   :end-before: # docs :
```

(sector_cf)=
## `sector`

Sector-coupling options (natural gas, heating, service, transport, and industrial demand) are
documented on the dedicated [sector configuration page](#sectors).

(clustering_cf)=
## `clustering`


Each clustering and interconnection option will have a different number of minimum nodes which can be clustered to, an error will be thrown in `cluster_network` notifying you of that number if you have selected a value too low.

Cleaned and labeled REeDs Shapes are pulled from this github repository: https://github.com/pandaanson/NYU-law-work

Note the naming trap: the `simplify_network:` config block feeds the rule named `cluster_simpl`
(the historical rule name `simplify_network` no longer exists), while `cluster_network:` feeds the
rule of the same name. The `temporal:` block is the primary temporal-resolution knob for solved
networks.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-after: # docs : CLUSTERING
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/clustering.csv
```

(solving_cf)=
## `solving`

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-after: # docs : SOLVING
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/solving.csv
```

(walltime_cf)=
## `walltime`

Per-rule wall-time overrides consumed as Snakemake `walltime` resources, used by the SLURM
profile when submitting jobs to an HPC scheduler (see `config.cluster.yaml` and
`workflow/run_slurm.sh`). Rules not listed here fall back to per-rule defaults defined in the
workflow. Local runs ignore these values.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-after: # docs : WALLTIME
   :end-before: # docs :
```

(custom_files_cf)=
## `custom_files`

Bring-your-own network or cost inputs. When `activate: true`, `prepare_network` loads
`network_name` from `files_path` in place of the `cluster_network` output (and expects a
`costs_2030.csv` alongside it), letting you solve an externally-modified network with the standard
solve pipeline.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-after: # docs : CUSTOM_FILES
   :end-before: # docs :
```

(plotting_cf)=
## `plotting`

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.plotting.yaml
   :language: yaml

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/plotting.csv
```
