# Hydrogen sector data

- `adp2024_h2_demand.csv`: annual hydrogen demand by scenario, state and year in MWh (higher heating value).
  Built offline from the "To" table (petajoules) of the "Hydrogen source and use" sheet of
  `ADP2024_detailed_output_data.xlsx`, Evolved Energy Research, Annual Decarbonization Perspective 2024
  (https://www.evolved.energy/annual-decarbonization-perspectives). Every end use but "Power Generation"
  (modelled endogenously), summed by scenario, state and year; the baseline, current policy and central
  scenarios; District of Columbia added to Maryland, Alaska and Hawaii dropped.
- `h2_storage_sites.gpkg`: geological hydrogen storage sites, layers `salt` and `hardrock`, from ReEDS
  (https://github.com/ReEDS-Model/ReEDS, `inputs/shapefiles/h2_storage_sites.gpkg`, BSD-3-Clause), based on
  Lord, Kobos and Borns (2014), "Geologic storage of hydrogen: Scaling up to meet city transportation demands",
  International Journal of Hydrogen Energy 39(28), https://doi.org/10.1016/j.ijhydene.2014.07.121.
