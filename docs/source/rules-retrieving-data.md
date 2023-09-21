# Retrieve Data

Numerous datasets used in PyPSA USA are large and are not stored on GitHub. Insted, data is stored on Zenodo or supplier websites, and the workflow will automatically download these datasets via the `retrieve` rules

(databundle)=
## Rule `retrieve_zenodo_databundles`

Data used to create the base electrical network is pulled from [Breakthrough Energy](https://breakthroughenergy.org/) (~4.3GB). This includes geolocated data on substations, power lines, generators, electrical demand, and resource potentials. 

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.4538590.svg)](https://zenodo.org/record/4538590)

Protected land area data for the USA is retrieved from [Protected Planet](https://www.protectedplanet.net/en) via the [PyPSA Meets-Earth](https://pypsa-meets-earth.github.io/) data deposit (`natura_global`) (~100MB). 

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.1223907.svg)](https://zenodo.org/record/1223907)

Baythymetry data via [GEBCO](https://www.gebco.net/) and a cutout of USA [Copernicus Global Land Service](https://land.copernicus.eu/global/products/lc) data are downloaded from a PyPSA USA Zenodo depost (~2GB). 

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.8175051.svg)](https://zenodo.org/record/8175051)

## Rule `retrieve_eia_data`

Historical electrical load data from 2015 till the last present month are retrieved from the [US Energy Information Agency](https://www.eia.gov/) (EIA). Data is downloaded at hourly temporal resolution and at a spatial resolution of balancing authority region. 

## Rule `retrieve_WECC_forcast_data`

Forecasted electricity demand data and generator operational charasteristics for the [Western Electricity Coordinating Council](https://www.wecc.org/Pages/home.aspx) (WECC) region are retrieved directly from WECC. Projected data for both 2030 and 2032 are retrieved (~300MB each). 

[![URL](https://img.shields.io/badge/URL-WECC_Data-blue)](<https://www.wecc.org/Reliability/Forms/Default%20View.aspx>)

(cutout)=
## Rule `retrieve_cutout`

Cutouts are spatio-temporal subsets of the USA weather data from the [ERA5 dataset](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview). They have been prepared by and are for use with the [atlite](https://github.com/PyPSA/atlite) tool. You can either generate them yourself using the build_cutouts rule or retrieve them directly from zenodo through the rule `retrieve_cutout`.

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.1225941.svg)](https://zenodo.org/record/1225941)

```{note}
Only the western region for 2019 has been prepared and saved to Zenodo for download. Any other region needs to be created by the user.
```

(costs)=
## Rule `retrieve_cost_data`

This rule downloads generator economic assumptions from the [NREL](https://www.nrel.gov/) [Annual Technology Baseline](https://atb.nrel.gov/). 

[![URL](https://img.shields.io/badge/URL-NREL_ATB-blue)](<https://atb.nrel.gov/x>)

[![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=ATB%2F)

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
