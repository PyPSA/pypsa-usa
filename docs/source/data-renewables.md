(data-renewables)=
# Renewables

## Weather Data and Renewable Resource Availability

### Integration with Atlite

PyPSA-USA leverages the Atlite tool to provide access to decades of weather data with varying spatial resolutions. Atlite is used to estimate hourly renewable resource availability across the United States, typically at a spatial resolution of 30 km² cells. Within PyPSA-USA, users can configure:

- **Weather Year**
- **Turbine Type**
- **Solar Array Type**
- **Land-Use Parameters**
- **Simulation Parameters**

The hourly renewable capacity factors calculated by Atlite are weighted based on land-use availability factors. This ensures that areas unsuitable for specific technology types do not disproportionately affect the renewable resource capacity assigned to each node. These weighted capacity factors are aggregated into 41,564 distinct zones across the United States. These zones are then clustered using one of the clustering algorithms developed for PyPSA-Eur.

### Land-Use Data and Renewable Integration

Land-use data is a critical factor in determining the technical potential for renewable energy integration. PyPSA-USA provides users with data on renewable resource availability, which is informed by layers of flexibly assigned land-use classifications, including:

- **Urban Areas**
- **Forested Regions**
- **Scrub-Land**
- **Satellite Imagery**
- **Federally Protected Lands**
- **Bathymetry**
- **State-Level Land Exclusions**

These land exclusion layers are combined to create estimates of land available for renewable energy development, which can be customized for different technologies. This approach allows users to accurately assess the technical potential for renewable integration based on realistic land-use constraints.

Additional details on the configurations available in the Atlite weather-energy simulation tool can be found in the configurations section.


### Data
```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,22,22,22
   :file: datatables/renewables.csv
```
