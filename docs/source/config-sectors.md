(sectors)=
# Sectors

To run sector coupled studies, the `sector` wildcard must be assigned to `G`. After doing so,
the following configuration options are exposed to the user.

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
