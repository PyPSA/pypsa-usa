(wildcards)=
# Wildcards

It is easy to run PyPSA-USA for multiple scenarios using the wildcards feature of `snakemake`.
Wildcards generalise a rule to produce all files that follow a regular expression pattern
which e.g. defines one particular scenario. One can think of a wildcard as a parameter that shows
up in the input/output file names of the `Snakefile` and thereby determines which rules to run,
what data to retrieve and what files to produce.

```{note}
Detailed explanations of how wildcards work in ``snakemake`` can be found in the
`relevant section of the [documentation](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#wildcards).
```

(interconnect)=
## The `{interconnect}` wildcard

The `{interconnect}` wildcard sets the geographc scope of the model run. Models
can be run for the `western`, `eastern`, `texas`, or `usa` grid. The interconnects
follow the representation described by [Breakthrough Energy](https://breakthroughenergy.org/).

A visual representation of each `{interconnect}` is shown below:

```{eval-rst}
.. image:: _static/cutouts/cutouts.png
    :scale: 100 %
```

<!-- (simpl)=
## The ``{simpl}`` wildcard -->

<!-- The ``{simpl}`` wildcard specifies number of buses a detailed
network model should be pre-clustered to in the rule
:mod:`simplify_network` (before :mod:`cluster_network`). -->

(clusters)=
## The `{clusters}` wildcard

The `{clusters}` wildcard specifies the number of buses a detailed network model should be reduced to in the rule :mod:`cluster_network`.
The number of clusters must be lower than the total number of nodes and higher than the number of balancing authoritites.

If an `m` is placed behind the number of clusters (e.g. `100m`), generators are only moved to the clustered buses but not aggregated by carrier; i.e. the clustered bus may have more than one e.g. wind generator.

(ll)=
## The `{ll}` wildcard

The `{ll}` wildcard specifies what limits on
line expansion are set for the optimisation model.
It is handled in the rule :mod:`prepare_network`.

The wildcard, in general, consists of two parts:

    1. The first part can be
       ``v`` (for setting a limit on line volume) or
       ``c`` (for setting a limit on line cost)

    2. The second part can be
       ``opt`` or a float bigger than one (e.g. 1.25).

       (a) If ``opt`` is chosen line expansion is optimised
           according to its capital cost
           (where the choice ``v`` only considers overhead costs for HVDC transmission lines, while
           ``c`` uses more accurate costs distinguishing between
           overhead and underwater sections and including inverter pairs).

       (b) ``v1.25`` will limit the total volume of line expansion
           to 25 % of currently installed capacities weighted by
           individual line lengths; investment costs are neglected.

       (c) ``c1.25`` will allow to build a transmission network that
           costs no more than 25 % more than the current system.

(opts)=
## The `{opts}` wildcard

The `{opts}` wildcard is used for electricity-only studies. It triggers
optional constraints, which are activated in either :mod:`prepare_network` or
the :mod:`solve_network` step. It may hold multiple triggers separated by `-`,
i.e. `Co2L-3H` contains the `Co2L` trigger and the `3H` switch. There are
currently:

```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 10,20,10,10
   :file: configtables/opts.csv
```


(sector)=
## The `{sector}` wildcard

The `{sector}` wildcard is used to specify what sectors to include. If `None`
is provided, an electrical only study is completed.

| Sector      | Code | Description                                                  |
|-------------|------|--------------------------------------------------------------|
| Electricity | E    | Electrical sector. Will always be run.                       |
| Natural Gas | G    | Natural gas sector                                           |
| Heating     | H    | Residential and commercial heating and cooling demand        |
<!-- | Transport   | T    | Residential and light duty commercial transportation demand  |
| Methane     | M    | Methane tracking. Requires natural gas sector.               | -->

(scope)=
## The `{scope}` wildcard
Takes values `residential`, `urban`, `total`. Used in sector coupling
studies to define population breakdown.

Used in the following rules:
- `build_heat_demands`
- `build_temperature_profiles`
- `build_solar_thermal_profiles`


(cutout_wc)=
## The `{cutout}` wildcard

The `{cutout}` wildcard facilitates running the rule :mod:`build_cutout`
for all cutout configurations specified under `atlite: cutouts:`. Each cutout
is descibed in the form `{dataset}_{year}`. These cutouts will be stored in a
folder specified by `{cutout}`.

Valid dataset names include: `era5`
Valid years can be from `1940` to `2022`

```{note}
Data for `era5_2019` has been pre-pared for the user and will be automatically downloaded
during the workflow. If other years are needed, the user will need to prepaer the
cutout themself.
```
