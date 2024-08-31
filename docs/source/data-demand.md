(data-demand)=
## Demand

PyPSA-USA offers access to both historical and future forecasted demand data, which are defined exogenously.

### Historical Demand

Historical demand data is imported from the EIA930 via the GridEmissions tool, covering the years 2018-2024. This data is defined at the balancing area region level.

### Forecasted Demand

Forecasted demand is sourced from the NREL Electrification Futures Study (EFS), providing hourly demand forecasts for the years 2030, 2040, and 2050. The EFS data includes forecasts for varying levels and speeds of electrification across sectorally specified residential, commercial, and industrial end-uses, such as:

- **Lighting**
- **Heating/Cooling**
- **Vehicle Electrification**

This segmentation of end-use demand allows users to define custom constraints and devices that modify only specific loads, enabling the modeling of Distributed Energy Resources (DERs).

### Demand Scaling and Interpolation

For the years between 2030, 2040, and 2050, PyPSA-USA implements a scaling factor that interpolates between future years or scales historical demand using forecasts from the Annual Energy Outlook (AEO). 

### Demand Distribution

The user-selected demand (whether historical or future) is distributed to nodes in the full network according to the population density defined by BE-TAMU.
