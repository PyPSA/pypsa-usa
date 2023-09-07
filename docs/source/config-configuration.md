# Configuration 

**This workflow is currently only being tested for the `western` interconnection wildcard.**

## Pre-set Configuration Options

The `network_configuration` option in the `config.yaml` file accepts 3 values: `pypsa-usa` , `ads2032`, and `breakthrough`. Each cooresponds to a different combiation of input datasources for the generators, demand data, and generation timeseries for renewable generators. The public version of the WECC ADS PCM does not include data on the transmission network, but does provide detailed information on generators. For this reason the WECC ADS generators are superimposed on the TAMU/BE network.

| Configuration Options: | PyPSA-USA | ADS2032(lite) |
|:----------:|:----------:|:----------:|
| Transmission | TAMU/BE | TAMU/BE |
| Thermal Generators | EIA860 | WECC-ADS |
| Renewable Time-Series | Atlite | WECC-ADS |
| Demand | EIA930 | WECC-ADS |
| Years Supported | 2019 (soon 2017-2023) | 2032 |
| Interconnections Supported | WECC (soon entire US) | WECC |

## Clustering

There have been issues in running operations-only simulations with clusters >50 for the WECC. Issue is currently being addressed.

Minimum Number of clusters:
```
Eastern: TBD
Western: 30
Texas: TBD
```

Maximum Number of clusters:
```
Eastern: 35047
Western: 4786
Texas: 1250
```