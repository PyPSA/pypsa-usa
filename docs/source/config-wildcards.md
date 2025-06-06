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

(simpl)=
## The ``{simpl}`` wildcard

The ``{simpl}`` wildcard specifies number of buses a detailed
network model should be pre-clustered to in the rule
:mod:`simplify_network` (before :mod:`cluster_network`).

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

We reccomend using the line volume limit for constraining
transission expansion. Use ``lv`` (for setting a limit on line volume)

After ``lv`` you can specify two type of limits:

       ``opt`` or a float bigger than one (e.g. 1.25).

       (a) If ``opt`` is chosen line expansion is optimised
           according to its capital cost.

       (b) ``v1.25`` will limit the total volume of line expansion
           to 25 % of currently installed capacities weighted by
           individual line lengths; investment costs are neglected.


(opts)=
## The `{opts}` wildcard

The `{opts}` wildcard is used for electricity-only studies. It triggers
optional constraints, which are activated in either :mod:`prepare_network` or
the :mod:`solve_network` step. It may hold multiple triggers separated by `-`,
i.e. `REM-3H` contains the `REM` regional emissions limit trigger and the `3H` switch.

The REM, SAFER, RPS can be defined using either the reeds zone name 'p##"
the state code (eg, TX, CA, MT), pypsa-usa interconnect name (western, eastern, texas, usa),
or nerc region name.

```{warning}
TCT Targets can only be used with renewable generators and utility scale batteries in sector studies.
```

There are currently:

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

| Sector      | Code | Description                                    | Status      |
|-------------|------|------------------------------------------------|-------------|
| Electricity | E    | Electrical sector. Will always be run.         | Runs        |
| Natural Gas | G    | All sectors added                              | Development |


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
