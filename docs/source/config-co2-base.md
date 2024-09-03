(base_emission_values)=
# Base Emission Values

If using the `{opts}` wildcard to reduce emissions, the user must put in a `co2base` value. Provided below are historical yearly CO2 emission values for both the power sector and all sectors at an interconnect level. This data can be used as a starting point for users. **Note the units in this table are Million Metric Tons (MMT).** This data originates from the [EIA State Level CO2 database](https://www.eia.gov/opendata/browser/co2-emissions/co2-emissions-aggregates?frequency=annual&data=value;&sortColumn=period;&sortDirection=desc;), and is compiled by the script `workflow/notebooks/historical_emissions.ipynb`

```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/emissions.csv
```