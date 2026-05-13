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
   :start-at: run:
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
   :start-at: scenario:
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/scenario.csv
```

(snapshots_cf)=
## `snapshots`

Specifies the temporal range to build an energy system model for as arguments to [`pandas.date_range`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html)

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-at: snapshots:
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/snapshots.csv
```

(atlite_cf)=
## `atlite`

Define and specify the `atlite.Cutout` used for calculating renewable potentials and time-series. All options except for `features` are directly used as [`cutout parameters`](https://atlite.readthedocs.io/en/latest/ref_api.html#cutout)

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.common.yaml
   :language: yaml
   :start-at: atlite:
   :end-before: # docs

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
   :start-at: electricity:
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/electricity.csv
```

(renewable_cf)=
## `renewable`

### `solar`
```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.common.yaml
   :language: yaml
   :start-at: solar:
   :end-before: hydro:

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/solar.csv
```

### `onwind`
```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.common.yaml
   :language: yaml
   :start-at: onwind:
   :end-before: offwind:

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/onwind.csv
```

### `Offshore wind`
```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.common.yaml
   :language: yaml
   :start-at: offwind:
   :end-before: solar:
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
   :start-at: renewable_land_access:
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/nrel_exclusion.csv
```

The `_cec` and `_boem` variants overlay additional regulatory screens on top of the base NREL availability raster: California Energy Commission Wind/Solar BaseScreen for onshore solar/wind in CA, and BOEM offshore wind planning areas for offshore. Outside their applicable region the variant equals the base.

(lines_cf)=
## `lines`
```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-at: lines:
   :end-before: # docs

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
   :start-at: links:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/links.csv
```

<!-- (load_cf)=
## `load`

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-after: # p_nom_max:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/load.csv
``` -->

(co2_cf)=
## `co2`

The `co2` section specifies whether the model may use underground storage to sequester captured CO2 or not. In case underground storage is specified, each node (composing the network) has a specific storage potential and a cost associated with it. The storage potential (in tonnes) is calculated by aggregating all the underlying storage potentials of the U.S. counties encompassed in the node's geographical area. Counties that are only partially covered by the node's geographical area have their potential fractionated accordingly. The storage cost (in $/tonne) is calculated by weighting the potential with the cost of each county encompassed. The dataset containing information about underground CO2 storage potentials and costs at a county level (and used in PyPSA-USA) was provided by Edna Calzado at The University of Texas (Austin), which was derived from the Roads to Removal project (https://roads2removal.org). To get an illustration, enabling underground co2 storage for a sector-less network will render its topography similar to <a href = "_static/CCTS/pypsa-usa_sector-less_with_underground_co2_storage.svg" target = "_blank">this</a>, while for a sector-based network will render its topography similar to <a href = "_static/CCTS/pypsa-usa_sector-based_with_underground_co2_storage.svg" target = "_blank">this</a>. As a reference, disabling underground co2 storage (i.e. no CCTS), a sector-less network has a topography similar to <a href = "_static/CCTS/pypsa-usa_sector-less_without_CCTS.svg" target = "_blank">this</a>, while a sector-based network has a topography similar to <a href = "_static/CCTS/pypsa-usa_sector-based_without_CCTS.svg" target = "_blank">this</a>.

In addition, the section specifies whether the model may transport captured CO2 between nodes or not. In case transportation is specified, a network of CO2 pipelines is built based on the electricity grid layout represented in PyPSA-USA to determine where/how to build pipelines to connect nodes. To get an illustration, enabling co2 transport for a sector-less network will render its topography similar to <a href = "_static/CCTS/pypsa-usa_sector-less_with_underground_co2_storage_and_co2_transport.svg" target = "_blank">this</a>, while for a sector-based network will render its topography similar to <a href = "_static/CCTS/pypsa-usa_sector-based_with_underground_co2_storage_and_co2_transport.svg" target = "_blank">this</a>.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-at: co2:
   :end-before: # docs

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

For a sector-based model:

   - When `granularity` is set to `node`, each sector/node pair has a specific "air atmosphere" into which all the processes belonging to the sector/node pair emit CO2. For each sector/node pair, DAC is built to capture CO2 from its "air atmosphere". To get an illustration, setting `granularity` to `node` will render the topography of a sector-based network similar to <a href = "_static/CCTS/pypsa-usa_sector-based_with_underground_co2_storage_and_co2_transport_and_dac_node-based.svg" target = "_blank">this</a>.

   - When `granularity` is set to `state`, each sector/U.S. state pair has a specific "air atmosphere" into which all the processes of the nodes belonging to the state emit CO2. For each sector/node pair, DAC is built to capture CO2 from the sector/state pair's "air atmosphere" it belongs to. To get an illustration, setting `granularity` to `state` will render the topography of a sector-based network similar to <a href = "_static/CCTS/pypsa-usa_sector-based_with_underground_co2_storage_and_co2_transport_and_dac_state-based.svg" target = "_blank">this</a>.

   - Given that a `granularity` set to `nation` does not make sense in a sector-based model, it defaults to `node` in this case.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-at: dac:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/dac.csv
```

(costs_cf)=
## `costs`

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-at: costs:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/costs.csv
```

(sector_cf)=
## `sector`
<!-- ```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-at: sector:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/sector.csv
``` -->

```{warning}
Sector coupling studies are all under active development. More info to come!
```


(clustering_cf)=
## `clustering`


Each clustering and interconnection option will have a different number of minimum nodes which can be clustered to, an error will be thrown in `cluster_network` notifying you of that number if you have selected a value too low.

Cleaned and labeled REeDs Shapes are pulled from this github repository: https://github.com/pandaanson/NYU-law-work

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-at: clustering:
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/clustering.csv
```


```{tip}
use `min` in `p_nom_max:` for more conservative assumptions.
```

(solving_cf)=
## `solving`

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-at: solving:

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/solving.csv
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
