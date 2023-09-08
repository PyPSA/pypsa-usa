# PyPSA-USA

**NOTE: This model is under active development. We welcome you to file issues on github as we continue to develop the model. You can email ktehranchi@stanford.edu for questions/support**

Please see [Model Documentation](https://pypsa-usa.readthedocs.io/en/latest/) for install & usage instructions.

PyPSA-USA is an open-source power systems model of the bulk transmission systems in the United States. This workflow draws from the work of [pypsa-eur](https://pypsa-eur.readthedocs.io/en/latest/index.html) and [pypsa-meets-earth](https://pypsa-earth.readthedocs.io/en/latest/how_to_contribute.html) to build a highly configurable power systems model that can be used for capacity expansion modeling, production cost simulation, and power flow analysis. This model is currently under development, and is only stable under certain configurations detailed below.

The model draws data from:

- The [TAMU/BreakthroughEnergy](https://www.breakthroughenergy.org/) transmission network model. This model has 82,071 bus network, 41,083 substations, and 104,192 lines across the three interconnections.
- Powerplant Data can be drawn from three options: the Breakthrough Network, the public version of the WECC Anchor Data Set Production Cost Model, or the EIA860
- Historical load data from the EIA via the EIA930.
- Forecasted load data from the [public WECC ADS PCM](https://www.wecc.org/ReliabilityModeling/Pages/AnchorDataSet.aspx).
- Renewable time series based on ERA5, assembled using the atlite tool.
- Geographical potentials for wind and solar generators based on [land use](https://land.copernicus.eu/global/products/lc) and excluding [protected lands](https://www.protectedplanet.net/country/USA) are computed with the atlite library.

Example 500 Node Western Interconnection Network:
![pypsa-usa Base Network](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/repo_data/network_500.jpg)

# Contributing

We welcome your contributions to this project. Please see the [contributions](https://pypsa-usa.readthedocs.io/en/latest/contributing.html) guide in our readthedocs page for more information. Please do not hesitate to reachout to ktehranchi@stanford.edu with specific questions, requests, or feature ideas.


# License

The project is licensed under MIT License.
