# Retrieve Data

Numerous datasets used in PyPSA USA are large and are not stored on GitHub. Insted, data is stored on Zenodo or supplier websites, and the workflow will automatically download these datasets via the `retrieve` rules

```{note}
If you recieve the follwing error while running a retrieve rule on Linux

    FileNotFoundError: [Errno 2] No such file or directory: 'unzip'

Run the command `sudo apt install zip`
```

(databundle)=
## Rule `retrieve_zenodo_databundles`

Data used to create the base electrical network is pulled from [Breakthrough Energy](https://breakthroughenergy.org/) (~4.3GB). This includes geolocated data on substations, power lines, generators, electrical demand, and resource potentials.

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.4538590.svg)](https://zenodo.org/record/4538590)

Protected land area data for the USA is retrieved from [Protected Planet](https://www.protectedplanet.net/en) via the [PyPSA Meets-Earth](https://pypsa-meets-earth.github.io/) data deposit (`natura_global`) (~100MB).

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.10067222.svg)](https://zenodo.org/records/10067222)

Baythymetry data via [GEBCO](https://www.gebco.net/) and a cutout of USA [Copernicus Global Land Service](https://land.copernicus.eu/global/products/lc) data are downloaded from a PyPSA USA Zenodo depost (~2GB).

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.10067222.svg)](https://zenodo.org/records/10067222)

(databundle-sector)=
## Rule `retrieve_sector_databundle`
Retrives data for sector coupling

[![DOI](https://sandbox.zenodo.org/badge/DOI/10.5072/zenodo.10019422.svg)](https://zenodo.org/records/10019422)

**Geographic Data**

Geographic boundaries of the United States counties are taken from the
United States Census Bureau. Note, these follow 2020 boundaries to match
census numbers

[![URL](https://img.shields.io/badge/URL-Cartographic_Boundaries-blue)](<https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.2020.html#list-tab-1883739534>)

County level populations are taken from the United States Census Bureau. Filters applied:
 - Geography: All Counties within United States and Puerto Rico
 - Year: 2020
 - Surveys: Decennial Census, Demographic and Housing Characteristics

Sheet Name: Decennial Census - P1 | Total Population - 2020: DEC Demographic and Housing Characteristics

[![URL](https://img.shields.io/badge/URL-United_States_Census_Bureau-blue)](<https://data.census.gov/>)

County level urbanization rates are taken from the United States Census Bureau. Filters applied:
 - Geography: All Counties within United States and Puerto Rico
 - Year: 2020
 - Surveys: Decennial Census, Demographic and Housing Characteristics

Sheet Name: Decennial Census - H1 | Housing Units - 2020: DEC Demographic and Housing Characteristics

[![URL](https://img.shields.io/badge/URL-United_States_Census_Bureau-blue)](<https://data.census.gov/>)

**Natural Gas Data**

Natural Gas infrastructure includes:
- State to State pipeline capacity
- State level tranmsission pipeline volume
- Natural gas processing facility locations
- Natural gas processing facility locations (via EIA API)
- Natural gas underground storage (via EIA API)
- Natural Gas imports/exports by point of entry (via EIA API)

[![URL](https://img.shields.io/badge/URL-Pipeline_Capacity-blue)](<https://www.eia.gov/naturalgas/data.php>)
[![URL](https://img.shields.io/badge/URL-Pipeline_Shape-blue)](<https://hifld-geoplatform.opendata.arcgis.com/datasets/f44e00fce8b943f69a40a2324cf49dfd_0/explore>)
[![URL](https://img.shields.io/badge/URL-Processing_Capacity-blue)](<https://www.eia.gov/naturalgas/ngqs/#?report=RP9&year1=2017&year2=2017&company=Name>)

(retrieve-eia)=
## Rule `retrieve_eia_data`
```{eval-rst}
.. automodule:: retrieve_eia_data
```

(retrieve-wecc)=
## Rule `retrieve_WECC_forcast_data`

Forecasted electricity demand data and generator operational charasteristics for the [Western Electricity Coordinating Council](https://www.wecc.org/Pages/home.aspx) (WECC) region are retrieved directly from WECC. Projected data for both 2030 and 2032 are retrieved (~300MB each).

[![URL](https://img.shields.io/badge/URL-WECC_Data-blue)](<https://www.wecc.org/Reliability/Forms/Default%20View.aspx>)

(retrieve-efs)=
## Rule `retrieve_nrel_efs_data`

The [Electrification Futures Study](https://www.nrel.gov/analysis/electrification-futures.html) (EFS) are a series of publications from the NREL that explore the impacts of electrification in all USA economic sectors. As part of this, study are the EFS hourly load profiles. These load profiles represent projected end-use electricity demand for various scenarios. Load profiles are provided for a subset of years (2018, 2020, 2024, 2030, 2040, 2050) and are aggregated to the state, sector, and select subsector level. See the [EFS Load Profile Data Catalog](https://data.nrel.gov/submissions/126) for full details.

[![URL](https://img.shields.io/badge/URL-EFS_Load_Profiles-blue)](<https://data.nrel.gov/submissions/126>)

(retrieve-cutout)=
## Rule `retrieve_cutout`

Cutouts are spatio-temporal subsets of the USA weather data from the [ERA5 dataset](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview). They have been prepared by and are for use with the [atlite](https://github.com/PyPSA/atlite) tool. You can either generate them yourself using the build_cutouts rule or retrieve them directly from zenodo through the rule `retrieve_cutout`.

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.10067222.svg)](https://zenodo.org/records/10067222)

```{note}
Only the 2019 interconnects based on ERA5 have been prepared and saved to Zenodo for download
```

(costs)=
## Rule `retrieve_cost_data`

This rule downloads economic assumptions from various sources.

The [NREL](https://www.nrel.gov/) [Annual Technology Baseline](https://atb.nrel.gov/) provides economic parameters on capital costs, fixed operation costs, variable operating costs, fuel costs, technology specific discount rates, average capacity factors, and efficiencies.

[![URL](https://img.shields.io/badge/URL-NREL_ATB-blue)](<https://atb.nrel.gov/x>)

[![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=ATB%2F)

State level capital cost supply side generator cost multipliers are pulled from the "Capital Cost and Performance
Characteristic Estimates for Utility Scale Electric Power Generating Technologies" by the [EIA](https://www.eia.gov/). Note, these have been saved as CSV's and come with the repository download

[![URL](https://img.shields.io/badge/URL-CAPEX_Multipliers-blue)](<https://www.eia.gov/analysis/studies/powerplants/capitalcost/pdf/capital_cost_AEO2020.pdf>)

State level historial monthly **natural gas** fuel prices are taken from the [EIA](https://www.eia.gov/). This includes seperate prices for electrical power producers, industrial customers, commercial customers, and residential customers.

[![URL](https://img.shields.io/badge/URL-EIA_Natural_Gas_Prices-blue)](<https://www.eia.gov/dnav/ng/ng_pri_sum_dcu_nus_m.htm>)

State level historical **coal** fuel prices are taken from the [EIA](https://www.eia.gov/).

[![URL](https://img.shields.io/badge/URL-EIA_Coal_Prices-blue)](<https://www.eia.gov/coal/data/browser/#/topic/45?agg=1,0&geo=vvvvvvvvvvvvo&rank=g&freq=Q&start=200801&end=202303&ctype=columnchart&ltype=pin&rtype=s&maptype=0&rse=0&pin=>)

The [Annual Technology Baseline](https://atb.nrel.gov/) also provides data on the [transportation sector](https://atb.nrel.gov/transportation/2020/index), including fuel usage and capital costs.

[![URL](https://img.shields.io/badge/URL-NREL_ATB_Transportation-blue)](<https://atb.nrel.gov/transportation/2020/index>)

To populate any missing data, the [PyPSA/technology-data](https://github.com/PyPSA/technology-data) project is used. Data from here is only used when no other sources can be found, as it is mostly European focused.

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PyPSA/technology-data)

**Relevant Settings**

```yaml
enable:
    retrieve_cost_data:

costs:
    year:
    version:
```

```{seealso}
Documentation of the configuration file ``config/config.yaml`` at
:ref:`costs_cf`
```

**Outputs**

- ``resources/costs.csv``

(retrieve-caiso-data)=
## Rule `retrieve_caiso_data`
```{eval-rst}
.. automodule:: retrieve_caiso_data
```
