(model-constraints)=
# Custom Constraints

PyPSA-USA formulates capacity-expansion and dispatch problems as linear (or mixed-integer)
programs using [PyPSA](https://pypsa.readthedocs.io/) and [linopy](https://linopy.readthedocs.io/).
The core formulation — the objective function, nodal energy balances, linearized power flow
(KVL), storage consistency equations, and investment bounds — is inherited unchanged from
PyPSA. It is documented in the PyPSA user guide on
[optimal power flow](https://pypsa.readthedocs.io/en/latest/user-guide/optimal-power-flow.html)
and is not re-derived here. This page documents only the constraints that PyPSA-USA adds on
top of that formulation.

Custom constraints attach to the optimization model in two ways:

1. **At solve time**, through the `extra_functionality` hook of `n.optimize()` implemented in
   [solve_network.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/solve_network.py).
   A registry maps `{opts}` wildcard tokens (`RPS`, `REM`, `ERM`, `TCT`) to constraint
   functions in `workflow/scripts/opts/`; further constraints in the same hook are activated
   by configuration keys alone, or are always active.
2. **At network-preparation time**, in
   [prepare_network.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/prepare_network.py),
   where `{opts}` tokens (`Co2L`, `CH4L`, `Ep`) and the `{ll}` wildcard are translated into
   PyPSA `GlobalConstraint` components or cost adjustments before the model is built.

Wildcard tokens are parsed from the dash-separated `{opts}` string by
[`update_config_from_wildcards`](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/_helpers.py)
and written into the run's configuration, so every token has an equivalent config-file
setting. See {ref}`the opts wildcard <opts>` for the token reference table.

## Notation

Throughout this page, symbols follow PyPSA naming: {math}`p_{g,t}` is the dispatch of
generator {math}`g` in snapshot {math}`t`, {math}`P_g^{nom}` the (extendable) nominal
capacity variable, {math}`p_g^{nom}` a fixed nominal capacity, {math}`\bar{p}_{g,t}` the
per-unit availability (`p_max_pu`), {math}`\eta` a conversion efficiency, {math}`w_t` the
snapshot weighting in hours, and {math}`d_{n,t}` the exogenous load at bus {math}`n`.
Regions used by the policy constraints may be specified as state codes, ReEDS zones
(`p1`, `p2`, ...), interconnect names, NERC region names, individual bus names, or `all`
(see `get_region_buses` in
[opts/_helpers.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/opts/_helpers.py)).

## Overview

| Constraint | What it enforces | Trigger | Source |
|---|---|---|---|
| [Portfolio standards (RPS/CES)](#portfolio-standards-rps) | Minimum share of eligible generation relative to demand per region and horizon | `RPS` opts token; `electricity: portfolio_standards` + ReEDS RPS/CES data | [policy.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/opts/policy.py) |
| [Regional emission limits](#regional-emission-limits-rem) | Cap on annual power-sector CO2 per region and horizon | `REM` opts token; `electricity: regional_Co2_limits` | [policy.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/opts/policy.py) |
| [Energy reserve margin](#energy-reserve-margin-erm) | Energy-backed firm capacity above demand in every snapshot, per region | `ERM` opts token; `electricity: erm` | [reserves.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/opts/reserves.py) |
| [Technology capacity targets](#technology-capacity-targets-tct) | Minimum/maximum nominal capacity per carrier group, region, and horizon | `TCT` opts token; `electricity: technology_capacity_targets` | [policy.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/opts/policy.py) |
| [Operational reserves](#operational-reserves) | System-wide spinning-reserve requirement (GenX formulation) | `electricity: operational_reserve: activate` | [reserves.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/opts/reserves.py) |
| [Land-use limits](#land-use-limits) | Renewable capacity per carrier and land region bounded by developable potential | always active | [land.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/opts/land.py) |
| [Bidirectional link coupling](#bidirectional-link-coupling) | Equal capacity expansion of paired forward/reverse links | always active | [bidirectional_link.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/opts/bidirectional_link.py) |
| [Demand-response capacity](#demand-response-capacity) | Shifted load bounded by a fixed share of nominal load per bus and snapshot | `electricity: demand_response: shift` | [sector.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/opts/sector.py) |
| [Import/export volume limits](#import-and-export-volume-limits) | Traded energy bounded by a share of demand per balancing period | `electricity: imports/exports: volume_limit` | [interchange.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/opts/interchange.py) |
| [National emission cap](#national-emission-cap-co2l) | System-wide CO2 cap via PyPSA `GlobalConstraint` | `Co2L` opts token; `electricity: co2limit` | [prepare_network.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/prepare_network.py) |
| [Natural gas limit](#natural-gas-limit-ch4l) | Cap on annual gas-fired primary energy | `CH4L` opts token; `electricity: gaslimit` | [prepare_network.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/prepare_network.py) |
| [Emission pricing](#emission-pricing-ep) | CO2 price added to marginal costs (objective, not a constraint) | `Ep` opts token; `costs: emission_prices` | [prepare_network.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/prepare_network.py) |
| [Transmission expansion limit](#transmission-expansion-limits-ll) | Bound on total line-volume or line-cost expansion | `{ll}` wildcard | [prepare_network.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/prepare_network.py) |
| [Sector-coupling constraints](#sector-coupling-constraints) | Heat pumps, gas trade, water heating, EVs, sector CO2, sector DR | sector studies (`{sector}` ≠ `E`) | [sector.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/opts/sector.py) |

(portfolio-standards-rps)=
## Portfolio standards (RPS)

Renewable Portfolio Standards (RPS) and Clean Energy Standards (CES) require a minimum share
of electricity to come from eligible carriers. Standards are enforced per Renewable Energy
Credit (REC) trading zone — states are mapped to their `rec_trading_zone` so that
in-zone REC trading is implicit — and per planning horizon. State-level targets are read
from `electricity: portfolio_standards` (CSV) and merged with ReEDS RPS and CES trajectories
supplied by the workflow. Eligible carriers default to
`onwind, offwind, offwind_floating, solar, hydro, geothermal, biomass, EGS` for RPS; CES
additionally includes `nuclear, SMR, hydrogen_ct, CCGT-95CCS, CCGT-99CCS, Coal-95CCS`.
Custom rows may specify any carrier group.

**Trigger:** `RPS` token in `{opts}`.

For each REC trading zone {math}`Z` (a set of states {math}`r`), planning horizon
{math}`y`, and eligible carrier group {math}`C`:

\begin{align*}
    &\ \text{let:} \\
    &\ \hspace{1cm} T_y \hspace{1cm} \text{Set of snapshots in planning horizon } y \\
    &\ \hspace{1cm} G_{Z,C} \hspace{0.72cm} \text{Generators with carrier in } C \text{ at buses in zone } Z \\
    &\ \hspace{1cm} d_{r,t} = \text{Exogenous load in state } r \text{ at snapshot } t \\
    &\ \hspace{1cm} \gamma_{r,y} = \text{Required generation share for state } r \text{ in horizon } y \\
    &\ s.t. \\
    &\ \hspace{1cm} \sum_{g \in G_{Z,C}} \sum_{t \in T_y} p_{g,t} \;\geq\; \sum_{r \in Z} \gamma_{r,y} \sum_{t \in T_y} d_{r,t}
\end{align*}

In electricity-only studies the right-hand side is computed from the exogenous load time
series. In sector-coupled studies final electricity demand is endogenous, so the requirement
is instead applied against total power-sector supply: eligible generation (generators plus
efficiency-weighted links feeding AC buses) must exceed {math}`\gamma_{r,y}` times total
generation in each state of the zone (`add_RPS_constraints_sector`).

Data sources and coverage are described on the {ref}`policies page <data-policies>`.

(regional-emission-limits-rem)=
## Regional emission limits (REM)

Regional emission limits cap annual power-sector CO2 emissions for arbitrary bus regions
(states, ReEDS zones, interconnects, NERC regions, or `all`) and planning horizons. Limits
are read from the CSV at `electricity: regional_Co2_limits` with columns
`regions`, `planning_horizon`, `limit` (tonnes CO2).

**Trigger:** `REM` token in `{opts}`.

For each limit row with region set {math}`R`, horizon {math}`y`, and cap {math}`E_{R,y}`:

\begin{align*}
    &\ \text{let:} \\
    &\ \hspace{1cm} G_R^{em} \hspace{0.9cm} \text{Emitting generators at buses in } R \text{ (carrier CO2 intensity} > 0) \\
    &\ \hspace{1cm} \epsilon_{c} = \text{CO2 intensity of carrier } c \text{ [t/MWh}_{th}] \\
    &\ \hspace{1cm} \eta_{g,t} = \text{Generator efficiency [MWh}_{el}\text{/MWh}_{th}] \\
    &\ \hspace{1cm} e^{atm}_{y} = \text{End-of-horizon level of CO2 atmosphere stores, if present} \\
    &\ s.t. \\
    &\ \hspace{1cm} \sum_{g \in G_R^{em}} \sum_{t \in T_y} w_t \, \frac{\epsilon_{c(g)}}{\eta_{g,t}} \, p_{g,t} + e^{atm}_{y} \;\leq\; E_{R,y}
\end{align*}

The atmosphere-store term applies only when the network tracks CO2 explicitly with a `co2`
carrier (sector networks). In sector-coupled studies the `REM` token instead dispatches
sector-specific CO2 constraints that allow different end-use sectors to decarbonize at
different rates; see {ref}`sector emission targets <data-sector-coupling>`.

(energy-reserve-margin-erm)=
## Energy reserve margin (ERM)

The ERM constraint requires each region to hold firm, deliverable capacity above its demand
in **every snapshot**, not just at annual peak. Its distinguishing feature is that
contributions must be *energy-backed*: storage and transmission contribute through a shadow
("reserve") dispatch that must itself satisfy the full set of operational constraints —
storage energy balance, dispatch bounds, and line/link ratings. This prevents storage from
being credited for power it could not sustain and transmission from wheeling reserve that
lines could not carry.

**Trigger:** `ERM` token in `{opts}`; margins configured under `electricity: erm` as
`{region: margin}` (default `{'all': 0.15}`).

For every region {math}`R` with margin {math}`m_R`, auxiliary reserve variables
{math}`\tilde{p}^{dis}_{s,t}, \tilde{p}^{sto}_{s,t}, \widetilde{soc}_{s,t}` (storage
units), {math}`\tilde{s}_{\ell,t}` (lines), and {math}`\tilde{p}_{\ell,t}` (links) are
added, mirroring the real dispatch variables and subject to the same bounds and storage
energy balances (suffixed `_RESERVES` in the model). The nodal reserve adequacy constraint
is, for every bus {math}`n \in R` and snapshot {math}`t`:

\begin{align*}
    &\ \hspace{1cm}
    \sum_{g \in G^{ext}_n} \bar{p}_{g,t} P^{nom}_g
    + \sum_{s \in S_n} \left( \tilde{p}^{dis}_{s,t} - \tilde{p}^{sto}_{s,t} \right)
    + \sum_{\ell \in L_R} K_{n\ell} \, \eta_\ell \, \tilde{f}_{\ell,t} \\
    &\ \hspace{2cm} \geq\; (1 + m_R) \, d_{n,t}
    - \sum_{g \in G^{fix}_n} \bar{p}_{g,t} \, p^{nom}_g
    \hspace{0.5cm} \forall_{n \in R,\; t}
\end{align*}

where {math}`G^{ext}_n`/{math}`G^{fix}_n` are the extendable and fixed generators at bus
{math}`n` (credited at their availability {math}`\bar{p}_{g,t}`, i.e. the capacity factor
of that snapshot), {math}`S_n` the storage units at {math}`n`, {math}`\tilde{f}_{\ell,t}`
the reserve flow of lines and links, and {math}`K_{n\ell}` the signed network incidence
({math}`-1` at the sending bus, {math}`+1` at the receiving bus, where {math}`\eta_\ell`
applies only to link deliveries). Only branches {math}`L_R` with **both** endpoints in
{math}`R` contribute — reserve is shared within a region but not imported across its
boundary. All terms are activity-masked by build year and lifetime in multi-horizon
models. In planning horizons where a national CO2 limit of zero is active, emitting
carriers receive no capacity credit.

The dual of this constraint is stored per bus and snapshot as `n.buses_t["erm_price"]`
($/MW per snapshot) when solving with `ERM`, enabling capacity-price analysis.

Configuration details and valid region identifiers are documented in
{ref}`the opts wildcard section <opts>`.

## Operational reserves

An optional system-wide spinning-reserve requirement following the
[GenX formulation](https://genxproject.github.io/GenX/dev/core/#Reserves). Non-negative
reserve variables {math}`r_{g,t}` are added for every generator.

**Trigger:** `electricity: operational_reserve: activate: true` with parameters
`epsilon_load` ({math}`\varepsilon^{L}`), `epsilon_vres` ({math}`\varepsilon^{V}`), and
`contingency` ({math}`c`, MW).

Writing {math}`\kappa_g` for installed capacity ({math}`P^{nom}_g` if generator {math}`g`
is extendable, the parameter {math}`p^{nom}_g` otherwise):

\begin{align*}
    &\ \hspace{1cm} \sum_{g} r_{g,t}
    \;\geq\; \varepsilon^{L} \sum_{n} d_{n,t}
    + \varepsilon^{V} \sum_{g \in VRES} \bar{p}_{g,t} \, \kappa_g
    + c \hspace{0.5cm} \forall_{t} \\
    &\ \hspace{1cm} p_{g,t} + r_{g,t} \;\leq\; \bar{p}_{g,t} \, \kappa_g \hspace{0.5cm} \forall_{g,t}
\end{align*}

The first constraint sizes the reserve requirement as a share of load plus a share of
variable-renewable potential plus a fixed contingency; for extendable renewables the
potential term is a linear expression in {math}`P^{nom}_g`. The second couples reserve
provision to headroom below available capacity.

(technology-capacity-targets-tct)=
## Technology capacity targets (TCT)

TCT constraints impose minimum and/or maximum total nominal capacity for a carrier group in
a region and planning horizon — e.g. offshore-wind procurement mandates, nuclear retention,
or coal phase-outs. Targets are read from the CSV at
`electricity: technology_capacity_targets` with columns
`name, planning_horizon, region, carrier, min, max` (MW); `min`/`max` may be the keyword
`existing` to lock in currently installed capacity.

**Trigger:** `TCT` token in `{opts}`.

For each target row with region {math}`R`, carrier group {math}`C`, and horizon
{math}`y`, let {math}`A^{ext}` be the extendable generators, storage units, and links of
those carriers active in {math}`y` at buses in {math}`R`, and {math}`p^{exist}_{R,C,y}`
the summed nominal capacity of their non-extendable counterparts. Then:

\begin{align*}
    &\ \hspace{1cm} \underline{P}_{R,C,y} - p^{exist}_{R,C,y}
    \;\leq\; \sum_{a \in A^{ext}} P^{nom}_a
    \;\leq\; \overline{P}_{R,C,y} - p^{exist}_{R,C,y}
\end{align*}

with whichever bound is present in the data. Because existing capacity enters the
right-hand side as a constant, a `max` below existing capacity cannot retire
non-extendable assets through the LP; in myopic runs, targets with `max = 0` are instead
enforced by zeroing the affected non-extendable capacities before each horizon's solve
(`apply_forced_retirements`).

```{warning}
TCT targets can only be used with renewable generators and utility-scale batteries in
sector-coupled studies.
```

## Land-use limits

Renewable expansion is limited by developable land. Each extendable generator carries a
`land_region` attribute (assigned during clustering) and a `p_nom_max` potential derived
from the land-eligibility screens of the renewable-profile build. Because several
generator vintages or classes can share one resource area, capacity is constrained
jointly per carrier and land region rather than per generator. This constraint is always
active.

\begin{align*}
    &\ \hspace{1cm} \sum_{g \in G^{ext}_{c,z}} P^{nom}_g
    \;\leq\; \max_{g \in G^{ext}_{c,z}} p^{nom,max}_{g}
    \hspace{0.5cm} \forall \; \text{carrier } c, \text{ land region } z
\end{align*}

The maximum (rather than sum) on the right-hand side reflects that all members of a group
share the same land-region potential. With the `{clusters}` suffixes `m`/`a`/`c`,
non-aggregated carriers keep their pre-clustering bus as `land_region`, so land limits are
enforced at `{simpl}` resolution even when the transmission network is coarser.

## Bidirectional link coupling

Links that represent a single physical corridor in two directions (transport-model
transmission, H2 pipelines) are modeled as paired `_fwd`/`_rev` links. For each extendable
pair, capacity **expansion** must be equal so both directions describe the same asset:

\begin{align*}
    &\ \hspace{1cm} P^{nom}_{fwd} - p^{nom}_{fwd} \;=\; P^{nom}_{rev} - p^{nom}_{rev}
\end{align*}

This constraint is always active (it is a no-op when no paired extendable links exist).
Note that ReEDS interface transfer limits (ITLs) are not a custom constraint: they enter
the transport-model topology as link capacity ratings during clustering.

## Demand-response capacity

Price-responsive load shifting is modeled with paired storage buses and charger/discharger
links added at every load bus (see the
{ref}`sector-coupling demand-response description <data-sector-coupling>` for the
storage-based implementation and its cost accounting — the power-sector version uses the
same structure). The solve-time constraint caps how much load can be served from the
demand-response buffer in any snapshot:

**Trigger:** `electricity: demand_response: shift` (per-unit; `inf` leaves shifting
unconstrained).

\begin{align*}
    &\ \text{let:} \\
    &\ \hspace{1cm} D_n \hspace{1cm} \text{Demand-response discharger links delivering to bus } n \\
    &\ \hspace{1cm} s = \text{Allowable shiftable share of load [per unit]} \\
    &\ s.t. \\
    &\ \hspace{1cm} \sum_{\ell \in D_n} p_{\ell,t} \;\leq\; s \cdot d_{n,t} \hspace{0.5cm} \forall_{n,t}
\end{align*}

In sector-coupled studies the same bound is applied to flows out of the demand-response
bus (final demand being endogenous), and additional per-sector variants exist — see the
sector list below.

## Import and export volume limits

When imports/exports to regions outside the model scope are enabled
(`electricity: imports: enable` / `electricity: exports: enable`), dedicated `imports` /
`exports` links are added at boundary buses. A volume constraint bounds traded energy per
balancing period (default monthly; `day`, `week`, `month`, or `year`):

**Trigger:** `electricity: imports/exports: volume_limit` (percent of demand; default 10).

\begin{align*}
    &\ \hspace{1cm} \sum_{\ell \in M} \sum_{t \in \tau} w_t \, p_{\ell,t}
    \;\leq\; \frac{v}{100} \sum_{t \in \tau} w_t \, d_{t}
    \hspace{0.5cm} \forall \; \text{balancing periods } \tau
\end{align*}

where {math}`M` is the set of import (or export) links, {math}`v` the volume limit in
percent, and {math}`d_t` total AC load in period {math}`\tau`. In sector studies demand is
measured as the flow into the end-use sectors and the bound becomes a linear constraint in
both trade and demand variables.

(national-emission-cap-co2l)=
## National emission cap (Co2L)

`Co2L` in `{opts}` sets `electricity: co2limit_enable: true` and adds a standard PyPSA
`GlobalConstraint` (`CO2Limit`) on the `co2_emissions` carrier attribute:

\begin{align*}
    &\ \hspace{1cm} \sum_{g} \sum_{t} w_t \, \frac{\epsilon_{c(g)}}{\eta_{g,t}} \, p_{g,t}
    \;\leq\; \Omega \cdot n_{yr}
\end{align*}

with {math}`\Omega` = `electricity: co2limit` (tCO2/yr) and {math}`n_{yr}` the number of weather years.
The token should always carry a numeric factor scaling a reference budget: `Co2L0.05` sets
{math}`\Omega = 0.05 \times` `electricity: co2base` (which must be defined in the config).
Alternatively, set `electricity: co2limit_enable: true` and `electricity: co2limit`
directly in the config without any token. The constraint mechanics (including the emission
shadow price) are PyPSA's; see the
[PyPSA global-constraints documentation](https://pypsa.readthedocs.io/en/latest/user-guide/optimal-power-flow.html#global-constraints).

```{warning}
The token parser reads the *last* number in each token, so a bare `Co2L` (no factor) picks
up the `2` from the token name itself and silently sets {math}`\Omega = 2 \times` `co2base`.
Always append an explicit factor, or configure the cap via `electricity: co2limit`.
The same applies to `CH4L` below (a bare `CH4L` parses as 4 TWh).
```

(natural-gas-limit-ch4l)=
## Natural gas limit (CH4L)

`CH4L` sets `electricity: gaslimit_enable: true` and adds a `GlobalConstraint` (`GasLimit`)
on a `gas_usage` attribute assigned to the `OCGT`, `CCGT`, and `CHP` carriers, capping their
combined primary (thermal) energy use at `electricity: gaslimit` (MWh thermal per year,
scaled by the number of weather years). The appended number gives the limit in TWh thermal:
`CH4L200` caps gas use at 200 TWh thermal per year (always append a value — see the warning
above).
Equivalently, set `electricity: gaslimit_enable` and `electricity: gaslimit` in the config.

(emission-pricing-ep)=
## Emission pricing (Ep)

`Ep` enables `costs: emission_prices` and is a cost adjustment rather than a constraint:
the CO2 price {math}`\pi` (`costs: emission_prices: co2`, optionally given inline as e.g.
`Ep50`) is added to marginal costs of generators and storage units in proportion to their
emission intensity,

\begin{align*}
    &\ \hspace{1cm} c^{marg}_{g} \;\mathrel{+}=\; \pi \, \frac{\epsilon_{c(g)}}{\eta_g}
\end{align*}

```{note}
The `Ept` token (time-varying monthly CO2 prices) sets
`costs: emission_prices: co2_monthly_prices`, but no workflow step currently consumes that
setting. Because `Ept` also matches the `Ep` parser, its net effect today is identical to
plain `Ep`: static emission pricing at the configured `costs: emission_prices: co2` value.
```

(transmission-expansion-limits-ll)=
## Transmission expansion limits ({ll})

The `{ll}` wildcard (handled in `prepare_network`, not `{opts}`) bounds total transmission
expansion with a PyPSA `GlobalConstraint` of type `transmission_volume_expansion_limit`
(`v`, MW·km) or `transmission_expansion_cost_limit` (`c`, $): expansion across AC lines and
AC/DC links is limited to `factor` times today's length- or cost-weighted capacity, or left
to the optimizer with `opt`. See {ref}`the ll wildcard <ll>` for the syntax and the
[PyPSA documentation](https://pypsa.readthedocs.io/en/latest/user-guide/optimal-power-flow.html#global-constraints)
for the formulation.

## Temporal resolution (nH, nSEG)

Temporal aggregation happens in `prepare_network` before the model is built and only
changes the snapshot set, not the formulation. `nH` (e.g. `3h`) resamples all time series
by averaging over every `n` hours and scales snapshot weightings accordingly. `nSEG`
(e.g. `4380SEG`) applies [tsam](https://tsam.readthedocs.io/en/latest/index.html)
time-series segmentation, choosing `n` variable-length segments per investment period based
on load, renewable availability, and inflow profiles. Both write
`clustering: temporal: resolution_elec`, which can equivalently be set directly in the
config file.

## Per-carrier adjustments

The `{opts}` grammar also accepts `<carrier>+{p,e,c,m}<factor>` tokens (e.g.
`solar+c0.5`), parsed into `adjustments: electricity: {p_nom_max | e_nom_max |
capital_cost | marginal_cost}: {carrier: factor}`.

```{warning}
These adjustment tokens are parsed into the config, but no script in the current workflow
applies `adjustments:` to the network — the mechanism is inherited from PyPSA-Eur and is
presently inactive in PyPSA-USA.
```

(sector-coupling-constraints)=
## Sector-coupling constraints

Sector-coupled studies (`{sector}` other than `E`) activate additional constraints in the
same `extra_functionality` hook. They are documented (with their data and schematics) on
the sector pages and are only listed here:

- **Sector CO2 targets** — replaces the power-sector `REM` constraint with per-sector
  emission limits; see {ref}`sector coupling <data-sector-coupling>` and the
  [sector configuration](./config-sectors.md#carbon-limits).
- **Sector RPS** — the endogenous-demand RPS variant described
  [above](#portfolio-standards-rps).
- **Cooling heat-pump coupling** — ties heat-pump cooling output to installed heating
  capacity; see the [service-sector heat-pump docs](./data-services.md#heat-pumps).
- **GSHP capacity limit** — bounds ground-source relative to air-source heat pumps by the
  rural/urban population ratio; see the
  [heat-pump documentation](./data-services.md#heat-pumps).
- **Natural-gas import/export limits** — bounds gas trade with out-of-scope regions to
  ranges around historical volumes; see the
  [natural-gas sector page](./data-naturalgas.md).
- **Water-heater storage** — forces water-heating demand to be served through the storage
  buffer; see the [service sector page](./data-services.md).
- **EV generation policy** — caps electric-vehicle-served transport demand per mode and
  horizon; see the [transportation sector page](./data-transportation.md).
- **Sector demand response** — per-sector (and optionally per-carrier) shiftable-load
  bounds; see {ref}`demand response <data-sector-coupling>`.
- **Industrial fossil minimum** — a floor on fossil-served industrial heat while
  industrial end uses are under development; see the
  [industrial sector page](./data-industrial.md).
