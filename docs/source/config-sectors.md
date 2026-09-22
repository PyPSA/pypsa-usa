(sectors)=
# Sectors

To run sector coupled studies, the `sector` wildcard must be assigned to `G`. After doing so,
the following configuration options are exposed to the user.

```{note}
Only single-period studies are currently supported when running sector studies.
```

## Carbon Limits
```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.sector.yaml
   :language: yaml
   :start-at: # docs-co2
   :end-before: # docs-ng

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/sector_carbon.csv
```

## Natural Gas Options
```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.sector.yaml
   :language: yaml
   :start-at: # docs-ng
   :end-before: # docs-heating

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/sector_natural_gas.csv
```

## Heating Sector
```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.sector.yaml
   :language: yaml
   :start-at: # docs-heating
   :end-before: # docs-service

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/sector_heating.csv
```

## Service Sector
```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.sector.yaml
   :language: yaml
   :start-at: # docs-service
   :end-before: # docs-transport

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/sector_service.csv
```

:::{note}
`by_carrier` is optional (default `False`) and applies to the service and industrial sectors.
Set it to `True` to give each end-use carrier its own demand-response settings, replacing the
sector-wide `shift`/`marginal_cost` pair. Recognised carrier keys are `elec`, `heat`,
`space-heat`, `water-heat` and `cool`; a sector only uses the ones it actually models
(`elec`, `space-heat`/`heat` and `cool` for the service sector; `elec` and `heat` for the
industrial sector). For example:
```yaml
demand_response:
  by_carrier: True
  space-heat:
    shift: 20
    marginal_cost: 16
  elec:
    shift: 30
    marginal_cost: 25
  cool:
    shift: 10
    marginal_cost: 30
```
:::

## Transport Sector
```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.sector.yaml
   :language: yaml
   :start-at: # docs-transport
   :end-before: # docs-industrial

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/sector_transport.csv
```

## Industrial Sector
```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.sector.yaml
   :language: yaml
   :start-at: # docs-industrial

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/sector_industrial.csv
```
