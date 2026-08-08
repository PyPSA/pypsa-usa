(wildcards)=
# Wildcards

It is easy to run PyPSA-USA for multiple scenarios using the wildcards feature of `snakemake`.
Wildcards generalise a rule to produce all files that follow a regular expression pattern
which e.g. defines one particular scenario. One can think of a wildcard as a parameter that shows
up in the input/output file names of the `Snakefile` and thereby determines which rules to run,
what data to retrieve and what files to produce.

```{note}
Detailed explanations of how wildcards work in `snakemake` can be found in the
relevant section of the [snakemake documentation](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#wildcards).
```

(interconnect)=
## The `{interconnect}` wildcard

The `{interconnect}` wildcard sets the geographic scope of the model run. Models
can be run for the `western`, `eastern`, `texas`, or `usa` grid. The interconnects
follow the representation described by [Breakthrough Energy](https://breakthroughenergy.org/).

A visual representation of each `{interconnect}` is shown below:

```{eval-rst}
.. image:: _static/cutouts/cutouts.png
    :scale: 100 %
```

(simpl)=
## The `{simpl}` wildcard

The `{simpl}` wildcard specifies the number of buses the substation-level network is
pre-clustered to in the rule `cluster_simpl` (which runs directly after
`aggregate_to_substations`, before any per-bus data is built).

Under the simplify-early architecture, `{simpl}` is the resolution at which the model's
data layers are **built**: demand disaggregation, renewable resource profiles and their
land-use potentials, and generator placement all operate on the `{simpl}`-bus network
(`elec_s{simpl}.nc`) produced by `cluster_simpl`. The rule `cluster_network` then reduces
the network further to the final `{clusters}` transmission resolution. `{simpl}` therefore
controls the *resource* resolution, while `{clusters}` controls the *transmission*
resolution — see the {ref}`spatial configuration page <spatial>` for how the two interact.

(clusters)=
## The `{clusters}` wildcard

The `{clusters}` wildcard specifies the number of buses the `{simpl}`-level network is
reduced to in the rule `cluster_network`. The number of clusters must be lower than the
number of `{simpl}` buses and at least the number of balancing authorities (or, for the
ReEDS networks, the zone counts listed on the
{ref}`spatial configuration page <spatial>` — `cluster_network` reports the correct
minimum if the value is infeasible).

A plain integer (e.g. `33`) aggregates buses *and* generators: generators at the merged
buses are combined per carrier. A letter suffix controls which carriers are aggregated,
letting resource zones keep their `{simpl}`-level detail on a coarser transmission grid:

| Value  | Behaviour |
|--------|-----------|
| `33`   | Aggregate all carriers to the clustered buses (one generator per carrier and bus). |
| `33m`  | Aggregate only conventional carriers; renewable generators are moved to the clustered buses but keep their distinct `{simpl}`-level resource zones (a clustered bus may host several wind generators). |
| `33c`  | Aggregate all *except* conventional carriers; conventional plants keep `{simpl}`-level detail. |
| `33a`  | Aggregate no carriers — all generators keep their `{simpl}`-level resolution. |
| `all`  | Skip spatial reduction entirely (one cluster per bus). |

Carriers listed in `clustering: cluster_network: exclude_carriers` are never aggregated.
For non-aggregated carriers, land-use limits remain enforced at the `{simpl}`-level
`land_region`, so resource potentials are not artificially merged.

(ll)=
## The `{ll}` wildcard

The `{ll}` wildcard specifies what limits on transmission expansion are set for the
optimisation model. It is handled in the rule `prepare_network`.

The wildcard consists of a type letter followed by a factor, e.g. `v1.25`, `copt`:

- **`v` (volume):** limits the total *volume* of line expansion — capacity increases
  weighted by line length (MW·km) — across AC lines and AC/DC links.
- **`c` (cost):** limits the total *cost* of line expansion — capacity increases weighted
  by capital cost — across AC lines and AC/DC links.

After the type letter you can specify:

- a float, e.g. `v1.25`: branches become extendable and total expansion is limited to
  25 % above today's length- (or cost-) weighted capacity. A factor of exactly `1.0`
  (e.g. `v1.0`) keeps branches non-extendable, i.e. fixed at today's capacities.
- `opt`, e.g. `vopt` or `copt`: branches become extendable and expansion is optimised
  purely on capital cost, with no global cap (only per-branch `s_nom_max`/`p_nom_max`
  bounds apply).

We recommend using the line-volume limit (`v...`) for constraining transmission expansion.

```{note}
In result filenames the wildcard is preceded by the letter `l`
(e.g. `elec_s75_c33_ec_lv1.0_...`): the `l` is part of the filename pattern, while the
wildcard value itself starts with `v` or `c`. A bare `all` is accepted by the wildcard
pattern but is not currently handled by `prepare_network`.
```

(opts)=
## The `{opts}` wildcard

The `{opts}` wildcard triggers optional constraints and temporal settings, which are
activated in either the `prepare_network` or the `solve_network` step. It may hold
multiple triggers separated by `-`, i.e. `REM-3h` contains the `REM` regional emissions
limit trigger and the `3h` temporal averaging switch. In sector-coupled runs the same
tokens apply, with `RPS` and `REM` dispatching their sector-specific variants.

The mathematical formulation of every constraint token, its configuration keys, and its
source code location are documented on the {ref}`custom constraints page <model-constraints>`.

The regions for `REM`, `ERM`, and `RPS` can be defined using either the ReEDS zone name
(`p##`), the state code (e.g. TX, CA, MT), the PyPSA-USA interconnect name
(`western`, `eastern`, `texas`, `usa`), or the NERC region name.

```{warning}
TCT targets can only be used with renewable generators and utility scale batteries in
sector studies.
```

There are currently:

```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 10,20,10,10
   :file: configtables/opts.csv
```

### Energy Reserve Margin (ERM) Configuration

The ERM constraint ensures that each region has sufficient firm capacity to meet demand plus
a reserve margin at every timestep. Unlike traditional planning reserve margins that only
consider peak demand, ERM enforces the constraint across all snapshots.

**Key Features:**
- Resources must be "energy-backed" - storage devices must have sufficient state of charge to contribute to the reserve
- Supports multiple non-overlapping regions with different reserve margins
- Defaults to 15% reserve margin for all regions if not specified

**Configuration:**

To enable ERM, add `ERM` to the `opts` wildcard in your scenario configuration:

```yaml
scenario:
  opts: [ERM-3h]  # or [REM-ERM-3h] to combine with other opts
```

To customize the ERM values per region, add an `erm` section under `electricity` in your config file:

```yaml
electricity:
  erm:
    all: 0.15        # 15% reserve margin for all regions (default)
    # Or specify per region:
    # western: 0.15
    # SPP: 0.12
    # CISO: 0.17
```

If no `erm` configuration is provided, a default of `{'all': 0.15}` (15% reserve margin for all regions) is used.

**Valid region identifiers:**
- `all` - applies to all buses in the network
- State codes: `TX`, `CA`, `MT`, etc.
- Interconnect names: `western`, `eastern`, `texas`
- NERC region names
- ReEDS zone names: `p1`, `p2`, etc.

(sector)=
## The `{sector}` wildcard

The `{sector}` wildcard specifies which sectors to include. In the configuration it is set
via `scenario: sector:`; an empty string (`sector: ""`, the default) runs an
electricity-only study. The `Snakefile` normalises the value automatically: electricity is
always included, so `""` becomes `E` and `G` becomes `E-G`. The wildcard itself must match
`([EG]-)*[EG]`, i.e. dash-separated sector codes.

| Sector      | Code | Description                                    | Status      |
|-------------|------|------------------------------------------------|-------------|
| Electricity | E    | Electrical sector. Will always be run.         | Runs        |
| Natural Gas | G    | All sectors added                              | Development |


(cutout_wc)=
## Atlite cutout files

There is no `{cutout}` wildcard among the workflow's global wildcard constraints. Weather
cutouts enter the workflow as **files** named `cutouts/{interconnect}_{cutout}.nc`, where
the cutout identifier takes the form `{dataset}_{year}` (e.g. `era5_2019`) as configured
under `atlite: cutouts:`. They are built by the rule `build_cutout` only when
`enable: build_cutout: true` is set; otherwise a prepared cutout is expected on disk.

Valid dataset names include: `era5`. Valid years range from `1940` to `2022`.

```{note}
Under the default renewable dataset (`renewable: dataset: godeeep`), renewable profiles
are taken from the GODEEEP dataset and atlite cutouts are not used. Cutouts are only
required when `renewable: dataset: atlite` is selected. Data for `era5_2019` has been
prepared and is downloaded automatically during the workflow; other years require the
user to prepare the cutout themselves.
```
