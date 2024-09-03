% pypsa-usa documentation master file, created by
% sphinx-quickstart on Thu Aug 17 14:02:31 2023.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.

# PyPSA-USA

PyPSA-USA is an open-source energy system dataset and modeling tool designed for expansion planning and operational simulations across the United States.

% update to be a url
![PyPSA-USA_Network](_static/PyPSA-USA_network.png)

PyPSA-USA provides a versatile toolkit that allows you to customize both the **data** and **policy constraints** of your energy system planning model with ease. Through a straightforward configuration file, you can control the spatial, temporal, and operational resolution of your model, using access to cleaned and prepared historical and forecasted data.

You can create and export the power system data model to use in your own homebrewed optimization model OR use the built-in PyPSA-USA optimization features to layer on additional policy and operational constraints. For planning studies, we've integrated data on regional Renewable Portfolio Standards (RPS), emissions constraints, and other state-level policy constraints. We're actively building this model so more features are on the way!

PyPSA-USA builds on and leverages the work of [PyPSA-EUR](https://pypsa-eur.readthedocs.io/en/latest/index.html) developed by TU Berlin. PyPSA-USA is actively developed by the [INES Research Group](https://ines.stanford.edu) at Stanford University and the [ΔE+ Research Group](https://www.sfu.ca/see/research/delta-e.html) at Simon Fraser University.

```{warning}
**This model is under active development. If you would like to chat about using the model please don't hesitate to reach out to ktehranchi@stanford.edu and trevor_barnes@sfu.ca for support**
```

<!-- ```{include} ../../README.md
:relative-images:
``` -->

<!-- # Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search` -->

<!-- ```{toctree}
:caption: 'Contents:'
:maxdepth: 2
``` -->

```{toctree}
:caption: 'Getting Started:'
:maxdepth: 1
:hidden:

about-introduction
about-install
about-usage
```

```{toctree}
:caption: 'Model Data:'
:maxdepth: 1
:hidden:

data-transmission
data-demand
data-generators
data-renewables
data-costs
data-policies
data-naturalgas
data-services
data-industrial
data-transportation
```

```{toctree}
:caption: 'Model Configuration:'
:maxdepth: 1
:hidden:

config-wildcards
config-configuration
config-sectors
```

```{toctree}
:caption: 'Reference:'
:maxdepth: 1
:hidden:

license
changelog
contributing
```

```{toctree}
:hidden:
rules-retrieving-data
config-co2-base
```
