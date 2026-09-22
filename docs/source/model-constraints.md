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
[opts/_helpers.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/opts/_helpers.py));
`country` and, where present, `region` bus attributes also match, and interconnect names
are compared lower-cased.

## Overview

| Constraint | What it enforces | Trigger | Source |
|---|---|---|---|
| [Portfolio standards (RPS/CES)](#portfolio-standards-rps) | Minimum share of eligible generation relative to demand per region and horizon | `RPS` opts token (skipped when no generator is extendable); `electricity: portfolio_standards` + ReEDS RPS/CES data | [policy.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/opts/policy.py) |
| [Regional emission limits](#regional-emission-limits-rem) | Cap on annual power-sector CO2 per region and horizon | `REM` opts token (skipped when no generator is extendable; in sector runs dispatches the sector CO2 constraints instead); `electricity: regional_Co2_limits` | [policy.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/opts/policy.py) |
| [Energy reserve margin](#energy-reserve-margin-erm) | Energy-backed firm capacity above demand in every snapshot, per region | `ERM` opts token (skipped when no generator is extendable); `electricity: erm` | [reserves.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/opts/reserves.py) |
| [Technology capacity targets](#technology-capacity-targets-tct) | Minimum/maximum nominal capacity per carrier group, region, and horizon | `TCT` opts token (skipped when no generator is extendable; under `foresight: myopic` also applies forced retirements before each horizon); `electricity: technology_capacity_targets` | [policy.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/opts/policy.py) |
| [Land-use limits](#land-use-limits) | Renewable capacity per carrier and land region bounded by developable potential | always active | [land.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/opts/land.py) |
| [Bidirectional link coupling](#bidirectional-link-coupling) | Equal capacity expansion of paired forward/reverse links | always active | [bidirectional_link.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/opts/bidirectional_link.py) |
| [Demand-response capacity](#demand-response-capacity) | Shifted load bounded by a fixed share of nominal load per bus and snapshot | `electricity: demand_response: shift` ≥ 0.001 | [sector.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/opts/sector.py) |
| [Import/export volume limits](#import-and-export-volume-limits) | Traded energy bounded by a share of demand per balancing period | `electricity: imports/exports: enable: true` and a finite `volume_limit` > 0 | [interchange.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/opts/interchange.py) |
| [Interface transmission limits](#interface-transmission-limits) | Aggregate MW cap on the total flow across a bundle of transmission paths | `model_topology: interface_transmission_limits`; `electricity: transmission_interface_limits` | [interfaces.py](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/opts/interfaces.py) |
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

:::{figure} _static/generated/rec_trading_zones.png
:width: 100%
:alt: CONUS states coloured by REC trading zone

The REC trading zones over which portfolio standards are pooled. States in the same
tracking system (WREGIS, M-RETS, PJM-GATS, NEPOOL, ...) share one constraint; states not
mapped to a system form a zone of their own. The mapping is
`REC_TRADING_ZONE_MAPPER` in `workflow/scripts/constants.py`, applied to each bus's
`reeds_state` in `build_base_network`. Regenerate with `snakemake docs_rec_trading_zones`.
:::

**Trigger:** `RPS` token in `{opts}`. Like the other policy tokens it is skipped
when the network has no extendable generator (dispatch-only runs).

Targets are grouped by REC trading zone, planning horizon and the *exact* carrier string of
the row, so a custom row whose carrier list is spelled differently from the ReEDS default
forms its own constraint. The `region` column must be a single `reeds_state` code; rows
with any other region identifier are dropped. Only the states of a zone that carry a
positive target for that horizon and carrier group enter the constraint, on both sides:
for each zone {math}`Z`, horizon {math}`y` and carrier group {math}`C`, let
{math}`Z' \subseteq Z` be those states. Then:

\begin{align*}
    &\ \text{let:} \\
    &\ \hspace{1cm} T_y \hspace{1cm} \text{Set of snapshots in planning horizon } y \\
    &\ \hspace{1cm} G_{Z',C} \hspace{0.72cm} \text{Generators with carrier in } C \text{ at buses in the states } Z' \\
    &\ \hspace{1cm} d_{r,t} = \text{Exogenous load in state } r \text{ at snapshot } t \\
    &\ \hspace{1cm} \gamma_{r,y} = \text{Required generation share for state } r \text{ in horizon } y \\
    &\ s.t. \\
    &\ \hspace{1cm} \sum_{g \in G_{Z',C}} \sum_{t \in T_y} p_{g,t} \;\geq\; \sum_{r \in Z'} \gamma_{r,y} \sum_{t \in T_y} d_{r,t}
\end{align*}

Neither side carries snapshot weightings, so the constraint is a share of energy only
under uniform weightings. A zone/horizon/carrier group with no eligible generator is skipped
with a warning. In electricity-only studies the right-hand side is computed from the
exogenous load time series. In sector-coupled studies final electricity demand is
endogenous, so the requirement is instead applied against power-sector supply, pooled over
the zone: eligible generation (generators plus links feeding AC buses, weighted by their
static efficiency) must exceed {math}`\sum_{r \in Z'} \gamma_{r,y}` times the supply from
carriers listed under `conventional_carriers`, `renewable_carriers` and
`extendable_carriers: Generator` in each state (`add_RPS_constraints_sector`).

Data sources and coverage are described on the {ref}`policies page <data-policies>`.

(regional-emission-limits-rem)=
## Regional emission limits (REM)

Regional emission limits cap annual power-sector CO2 emissions for arbitrary bus regions
(states, ReEDS zones, interconnects, NERC regions, or `all`) and planning horizons. Limits
are read from the CSV at `electricity: regional_Co2_limits` with columns
`name`, `regions`, `planning_horizon`, `limit` (tonnes CO2); `name` becomes the constraint
name. Rows whose horizon is not an investment period, whose region matches no bus, or
whose region has no emitting generator are skipped silently.

**Trigger:** `REM` token in `{opts}`.

For each limit row with region set {math}`R`, horizon {math}`y`, and cap {math}`E_{R,y}`:

\begin{align*}
    &\ \text{let:} \\
    &\ \hspace{1cm} G_R^{em} \hspace{0.9cm} \text{Emitting generators at buses in } R \text{ (carrier CO2 intensity} \neq 0) \\
    &\ \hspace{1cm} \epsilon_{c} = \text{CO2 intensity of carrier } c \text{ [t/MWh}_{th}] \\
    &\ \hspace{1cm} \eta_{g,t} = \text{Generator efficiency [MWh}_{el}\text{/MWh}_{th}] \\
    &\ \hspace{1cm} e^{atm}_{y} = \text{End-of-horizon level of all CO2 atmosphere stores in the network, if present} \\
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

For every region {math}`R` with margin {math}`m_R`, auxiliary reserve variables are
added for storage units, lines and links, mirroring the real dispatch variables and subject
to the same bounds and storage energy balances (suffixed `_RESERVES` in the model). The
nodal reserve adequacy constraint is:

\begin{align*}
    &\ \text{let:} \\
    &\ \hspace{1cm} G^{ext}_n, G^{fix}_n \hspace{0.5cm} \text{Extendable / fixed generators at bus } n \\
    &\ \hspace{1cm} S_n \hspace{1.55cm} \text{Storage units at bus } n \\
    &\ \hspace{1cm} L_R \hspace{1.5cm} \text{Lines and links with both endpoints in } R \\
    &\ \hspace{1cm} \tilde{p}^{dis}_{s,t}, \tilde{p}^{sto}_{s,t} = \text{Reserve discharge / charge of storage unit } s \\
    &\ \hspace{1cm} \tilde{f}_{\ell,t} = \text{Reserve flow on branch } \ell \\
    &\ \hspace{1cm} K_{n\ell} = \text{Signed incidence: } -1 \text{ at the sending bus, } +1 \text{ at the receiving bus} \\
    &\ \hspace{1cm} \eta_\ell = \text{Link efficiency, applied to deliveries only (1 for lines)} \\
    &\ \hspace{1cm} m_R = \text{Reserve margin of region } R \\
    &\ s.t. \\
    &\ \hspace{1cm}
    \sum_{g \in G^{ext}_n} \bar{p}_{g,t} P^{nom}_g
    + \sum_{s \in S_n} \left( \tilde{p}^{dis}_{s,t} - \tilde{p}^{sto}_{s,t} \right)
    + \sum_{\ell \in L_R} K_{n\ell} \, \eta_\ell \, \tilde{f}_{\ell,t} \\
    &\ \hspace{2cm} \geq\; (1 + m_R) \, d_{n,t}
    - \sum_{g \in G^{fix}_n} \bar{p}_{g,t} \, p^{nom}_g
    \hspace{0.5cm} \forall_{n \in R,\; t}
\end{align*}

Generators are credited at their availability {math}`\bar{p}_{g,t}`, i.e. the capacity
factor of that snapshot. Only branches with **both** endpoints in {math}`R` contribute —
reserve is shared within a region but not imported across its boundary. All terms are
activity-masked by build year and lifetime in multi-horizon
models. In planning horizons for which the `electricity: regional_Co2_limits` table has a
row with `regions: all` and `limit: 0` (read whether or not `REM` is in `{opts}`),
carriers with positive CO2 intensity receive no capacity credit. A bus in {math}`R` with
no extendable generator, storage unit or in-region branch — an empty left-hand side — and
a non-zero right-hand side raises an error rather than an infeasible model.

The dual of this constraint is stored per bus and snapshot as `n.buses_t["erm_price"]`
($/MW per snapshot) when solving with `ERM`, enabling capacity-price analysis.

Configuration details and valid region identifiers are documented in
{ref}`the opts wildcard section <opts>`.

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
{math}`y`:

\begin{align*}
    &\ \text{let:} \\
    &\ \hspace{1cm} A^{ext}_{R,C,y} \hspace{0.6cm} \text{Extendable generators, storage units and links of carriers } C \text{ at buses in } R \text{, active in } y \\
    &\ \hspace{1cm} p^{exist}_{R,C,y} = \text{Summed nominal capacity of their non-extendable counterparts} \\
    &\ \hspace{1cm} \underline{P}_{R,C,y}, \overline{P}_{R,C,y} = \text{Target minimum / maximum capacity [MW]} \\
    &\ s.t. \\
    &\ \hspace{1cm} \underline{P}_{R,C,y} - p^{exist}_{R,C,y}
    \;\leq\; \sum_{a \in A^{ext}_{R,C,y}} P^{nom}_a
    \;\leq\; \overline{P}_{R,C,y} - p^{exist}_{R,C,y}
\end{align*}

with whichever bound is present in the data. Because existing capacity enters the
right-hand side as a constant, a `max` below existing capacity cannot retire
non-extendable assets through the LP; in myopic runs, targets with `max = 0` are instead
enforced by zeroing the affected non-extendable capacities before each horizon's solve
(`apply_forced_retirements`).

:::{figure} _static/generated/tct_targets.png
:width: 100%
:alt: Three CONUS maps of the default technology capacity targets: nuclear no-build regions, forced retirements and storage mandates

The technology capacity targets shipped with the repo, by policy family: ReEDS nuclear
no-build (`max = existing` for `nuclear, SMR`, over both state and ReEDS-zone regions),
ReEDS forced retirements (`max = 0` for the fossil and biomass carriers listed on each
state) and ReEDS storage mandates (an annual `min` MW trajectory of
`4hr_battery_storage`, shaded by the 2030 minimum). The map shows the shipped defaults
only — a study can point `electricity: technology_capacity_targets` at any other CSV with
the same columns. Regenerate with `snakemake docs_tct_targets`.
:::

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
    &\ \text{let:} \\
    &\ \hspace{1cm} G^{ext}_{c,z} \hspace{0.8cm} \text{Extendable generators of carrier } c \text{ with land region } z \\
    &\ \hspace{1cm} G^{fix}_{c,z} \hspace{0.85cm} \text{Non-extendable generators of carrier } c \text{ with land region } z \text{, active in the horizon} \\
    &\ \hspace{1cm} p^{nom,max}_{g} = \text{Developable potential of generator } g \text{ [MW]} \\
    &\ \hspace{1cm} p^{nom}_{g} = \text{Installed capacity of generator } g \text{ [MW]} \\
    &\ s.t. \\
    &\ \hspace{1cm} \sum_{g \in G^{ext}_{c,z}} P^{nom}_g
    \;\leq\; \max_{g \in G^{ext}_{c,z}} p^{nom,max}_{g}
    \;-\; \sum_{g \in G^{fix}_{c,z}} p^{nom}_{g}
    \hspace{0.5cm} \forall_{c,\, z}
\end{align*}

The maximum (rather than sum) on the first right-hand term reflects that all members of a
group share the same land-region potential. That potential is *gross*: it comes from the
land-eligibility screens of `build_renewable_profiles` and is never reduced by the plants
already standing on the site, so capacity that is not represented by a decision variable —
today's brownfield plants, and under `foresight: myopic` every build frozen by
`freeze_prior_periods` — is subtracted from it. Only capacity that is active in the
horizon being solved counts (`build_year`/`lifetime`); a retired unit releases its land.
Extendable assets stay on the left-hand side, so no MW is charged to the land twice, and
a group whose existing capacity already exceeds its potential gets a right-hand side of
zero rather than a negative one.

With the `{clusters}` suffixes `m`/`a`/`c`, non-aggregated carriers keep their
pre-clustering bus as `land_region`, so land limits are enforced at `{simpl}` resolution
even when the transmission network is coarser.

## Bidirectional link coupling

Links that represent a single physical corridor in two directions (transport-model
transmission, H2 pipelines) are modeled as paired `_fwd`/`_rev` links. For each extendable
pair, capacity **expansion** must be equal so both directions describe the same asset:

\begin{align*}
    &\ \text{let:} \\
    &\ \hspace{1cm} (fwd, rev) \hspace{0.6cm} \text{A pair of extendable links } \texttt{<name>\_fwd} \text{, } \texttt{<name>\_rev} \\
    &\ \hspace{1cm} p^{nom}_{fwd}, p^{nom}_{rev} = \text{Existing capacity in each direction [MW]} \\
    &\ s.t. \\
    &\ \hspace{1cm} P^{nom}_{fwd} - p^{nom}_{fwd} \;=\; P^{nom}_{rev} - p^{nom}_{rev}
    \hspace{0.5cm} \forall \text{ pairs}
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

**Trigger:** `electricity: demand_response: shift`, a per-unit number. `0` (the
default) adds no demand-response components at all; values below `0.001` add the
components but no constraint. There is no "unbounded" setting — the schema declares
`shift` as a number, so the string `inf` is rejected at parse time.

\begin{align*}
    &\ \text{let:} \\
    &\ \hspace{1cm} D_n \hspace{1cm} \text{Demand-response discharger links delivering to bus } n \\
    &\ \hspace{1cm} s = \text{Allowable shiftable share of load [per unit]} \\
    &\ s.t. \\
    &\ \hspace{1cm} \sum_{\ell \in D_n} p_{\ell,t} \;\leq\; s \cdot d_{n,t} \hspace{0.5cm} \forall_{n,t}
\end{align*}

In sector-coupled studies final demand is endogenous, so the bound is instead
{math}`\sum_{\ell \in D_n} p_{\ell,t} \leq s \sum_{\ell:\, bus0 = n} p_{\ell,t}`
over the links leaving AC bus {math}`n` toward the end-use sectors (carriers not ending in
`-dr`): demand-response delivery into a bus relative to all link flow leaving it. The
additional per-sector variants take `shift` in **percent**, not per unit — see the sector
list below.

## Import and export volume limits

When imports/exports to regions outside the model scope are enabled
(`electricity: imports: enable` / `electricity: exports: enable`), dedicated `imports` /
`exports` links are added at boundary buses. A volume constraint bounds traded energy per
balancing period (default monthly; `day`, `week`, `month`, or `year`):

**Trigger:** `electricity: imports: volume_limit` / `electricity: exports: volume_limit`,
a number in (0, 100] read as percent of demand, together with `enable: true` for that
direction; the balancing period is `electricity: imports/exports: balancing_period`. The
shipped default is `inf`, which adds no constraint. A value of `0` or an absent key is
skipped before the constraint function is reached and also adds no constraint, so a hard
zero-trade cap has to be expressed through `capacity_limit` or by disabling trade.

\begin{align*}
    &\ \text{let:} \\
    &\ \hspace{1cm} M \hspace{1.2cm} \text{Import (or export) links} \\
    &\ \hspace{1cm} \tau \hspace{1.25cm} \text{A balancing period (day, week, month or year)} \\
    &\ \hspace{1cm} v = \text{Volume limit [\% of demand]} \\
    &\ \hspace{1cm} d_t = \text{Time-varying } p\_set \text{ of loads with carrier AC, summed} \\
    &\ \hspace{1cm} w_t = \text{Objective snapshot weighting} \\
    &\ s.t. \\
    &\ \hspace{1cm} \sum_{\ell \in M} \sum_{t \in \tau} w_t \, p_{\ell,t}
    \;\leq\; \frac{v}{100} \sum_{t \in \tau} w_t \, d_{t}
    \hspace{0.5cm} \forall_{\tau}
\end{align*}

In sector studies demand is
measured as the flow into the end-use sectors, the bound becomes a linear constraint in
both trade and demand variables, and the share is rounded to two decimals
(`12.5` % becomes `12` %).

(interface-transmission-limits)=
## Interface transmission limits

A transmission *interface* is a bundle of paths between two groups of regions that is rated in
aggregate rather than path-by-path — CAISO's simultaneous import capability being the canonical
example. Setting `model_topology: interface_transmission_limits: true` reads the interface table
at `electricity: transmission_interface_limits` (columns
`interface, region_1, region_2, flow_12, flow_21`) and adds one per-snapshot constraint per
interface and direction:

**Trigger:** `model_topology: interface_transmission_limits: true`.

\begin{align*}
    &\ \text{let:} \\
    &\ \hspace{1cm} I^{\rightarrow} \hspace{0.9cm} \text{Export links from buses in region\_1 to region\_2} \\
    &\ \hspace{1cm} I^{\leftarrow} \hspace{0.9cm} \text{Import links from region\_2 to buses in region\_1} \\
    &\ \hspace{1cm} F_{12}, F_{21} = \text{Interface ratings out of / into region\_1 [MW] (} flow\_12 \text{, } flow\_21 \text{)} \\
    &\ s.t. \\
    &\ \hspace{1cm} \sum_{\ell \in I^{\rightarrow}} p_{\ell,t} \;\leq\; F_{12}
    \hspace{0.5cm} \forall_t \\
    &\ \hspace{1cm} \sum_{\ell \in I^{\leftarrow}} p_{\ell,t} \;\leq\; F_{21}
    \hspace{0.5cm} \forall_t
\end{align*}

{math}`I^{\rightarrow}` and {math}`I^{\leftarrow}` are selected by bus membership and
carrier, never by link name.

```{important}
Only the `imports` / `exports` links added by `add_extra_components` are constrained, so the
limits are a **no-op when `electricity: imports` and `electricity: exports` are both disabled**.
A `region_2` entry that is itself inside the modeled footprint contributes no trade links, so
internal AC lines between it and `region_1` escape the cap. In the shipped
`CAISO_Imports` row this applies to `p8`, which is a California zone: in a California-only run
the internal `p8`-`p9` corridor (~300 MW in the ReEDS/NARIS balancing-area table) is not
counted against the CAISO import cap, understating simultaneous imports by roughly that
corridor's rating. This is documented, not corrected.
```

(national-emission-cap-co2l)=
## National emission cap (Co2L)

`Co2L` in `{opts}` sets `electricity: co2limit_enable: true` and adds a standard PyPSA
`GlobalConstraint` (`CO2Limit`) on the `co2_emissions` carrier attribute:

\begin{align*}
    &\ \text{let:} \\
    &\ \hspace{1cm} T_y \hspace{1.1cm} \text{Snapshots of investment period } y \\
    &\ \hspace{1cm} k_y = \text{Period weighting (years to the next horizon; } investment\_period\_weightings.years \text{)} \\
    &\ \hspace{1cm} \epsilon_{c(g)} = \text{CO2 intensity of the carrier of } g \text{ [t/MWh}_{th}] \\
    &\ \hspace{1cm} \Omega = \text{Annual cap, } electricity{:}\ co2limit \text{ [tCO2/yr]} \\
    &\ \hspace{1cm} n_{yr} = \text{Weather years in the first investment period} \\
    &\ s.t. \\
    &\ \hspace{1cm} \sum_{y} k_y \sum_{g} \sum_{t \in T_y} w_t \, \frac{\epsilon_{c(g)}}{\eta_{g,t}} \, p_{g,t}
    \;\leq\; \Omega \cdot n_{yr}
\end{align*}

It is a single PyPSA
`primary_energy` constraint over **all** snapshots of the model, not one per year: in a
perfect-foresight run with horizons 2030/2040/2050 it caps
{math}`10\,E_{2030} + 10\,E_{2040} + 1\,E_{2050} \leq \Omega \cdot n_{yr}`, and in a
myopic run each horizon {math}`y` solved alone is capped at {math}`\Omega \cdot n_{yr} / k_y`.
Non-cyclic stores and storage units whose carrier has non-zero `co2_emissions` also enter
through their final energy level.
The token should always carry a numeric factor scaling a reference budget: `Co2L0.05` sets
{math}`\Omega = 0.05 \times` `electricity: co2base`. The constraint mechanics (including
the emission shadow price) are PyPSA's; see the
[PyPSA global-constraints documentation](https://pypsa.readthedocs.io/en/latest/user-guide/optimal-power-flow.html#global-constraints).

```{warning}
`Co2L` is currently not usable on `develop`
([#813](https://github.com/PyPSA/pypsa-usa/issues/813)): `co2base` is defined in no config
layer, so any `Co2L<x>` token raises `KeyError: 'co2base'`, and `co2base`, `co2limit`,
`co2limit_enable`, `gaslimit` and `gaslimit_enable` are absent from the closed
`electricity:` schema block, so they cannot be set in a config file either. When fixed:
the token parser reads the *last* number in each token, so a bare `Co2L` (no factor) picks
up the `2` from the token name itself and silently sets {math}`\Omega = 2 \times` `co2base`.
Always append an explicit factor. The same applies to `CH4L` below (a bare `CH4L` parses as
4 TWh).
```

(natural-gas-limit-ch4l)=
## Natural gas limit (CH4L)

`CH4L` sets `electricity: gaslimit_enable: true` and adds a `GlobalConstraint` (`GasLimit`)
on a `gas_usage` attribute assigned to the `OCGT`, `CCGT`, and `CHP` carriers, capping their
combined primary (thermal) energy use at `electricity: gaslimit` × {math}`n_{yr}`, with
the same cumulative, period-weighted form as `Co2L` above (one constraint over all
snapshots, weighted by {math}`k_y`). The appended number gives the limit in TWh thermal:
`CH4L200` sets 200 TWh thermal (always append a value — see the warning above). Only the
token route works today: `gaslimit` is written into the in-memory config after schema
validation, whereas setting it in a config file fails validation
([#813](https://github.com/PyPSA/pypsa-usa/issues/813)).

(emission-pricing-ep)=
## Emission pricing (Ep)

`Ep` enables `costs: emission_prices` and is a cost adjustment rather than a constraint:
the CO2 price {math}`\pi` (`costs: emission_prices: co2`, optionally given inline as e.g.
`Ep50`) is added to marginal costs of generators and storage units in proportion to their
emission intensity,

\begin{align*}
    &\ \text{let:} \\
    &\ \hspace{1cm} \pi = \text{CO2 price, } costs{:}\ emission\_prices{:}\ co2 \text{ [\$/t]} \\
    &\ \hspace{1cm} c^{marg}_{g} = \text{Marginal cost of generator or storage unit } g \text{ [\$/MWh]} \\
    &\ \hspace{1cm} \eta_g = \text{Static efficiency of } g \\
    &\ \text{then:} \\
    &\ \hspace{1cm} c^{marg}_{g} \;\mathrel{+}=\; \pi \, \frac{\epsilon_{c(g)}}{\eta_g}
    \hspace{0.5cm} \forall_g
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
(`v`, MW·km) or `transmission_expansion_cost_limit` (`c`, $). The constraint bounds the
**total** length- or cost-weighted nominal capacity of all *extendable* AC lines and AC/DC
links, summed over every vintage, to `factor` times today's value: `v1.25` allows 25 %
more MW·km, not 125 %. Factors ≤ 1 (including the default `v1.0`) make nothing extendable
and add no binding constraint, so `v0.8` is identical to `v1.0`; `opt` makes everything
extendable with no bound. In the ReEDS transport model each interface contributes its
forward and reverse links (each carrying half the interface capital cost) to both sides,
so the constant is twice the per-interface volume while the ratio is unaffected.

```{warning}
`c<factor>` is only consistent with a single investment period
([#815](https://github.com/PyPSA/pypsa-usa/issues/815)): PyPSA weights each extendable
asset's cost by the sum of its active periods' objective weightings, while the reference is
unweighted, so with several horizons the constraint is infeasible for any realistic
factor. Use `v<factor>` or `opt` in multi-period runs.
```

See {ref}`the ll wildcard <ll>` for the syntax and the
[PyPSA documentation](https://pypsa.readthedocs.io/en/latest/user-guide/optimal-power-flow.html#global-constraints)
for the formulation.

## Temporal resolution (nH, nSEG)

Temporal aggregation happens in `prepare_network` before the model is built. `nH`
(e.g. `3h`) resamples all time series by averaging over every `n` hours within each
investment period, sums snapshot weightings accordingly, and rescales unit-commitment
parameters of committable generators (`min_up_time`/`min_down_time` from hours to
snapshots, ramp limits multiplied by `n` and clipped at 1). `nSEG` (e.g. `4380SEG`)
applies [tsam](https://tsam.readthedocs.io/en/latest/index.html) time-series segmentation
(an optional dependency, solved with the configured solver), choosing `n` variable-length
segments per investment period from *all* time-varying component attributes, each
normalised by its annual maximum. Each segment is represented by the **first hour** of the
segment, not the segment mean, weighted by the segment duration. Both write
`clustering: temporal: resolution_elec`, which can equivalently be set directly in the
config file.

## Per-carrier adjustments

The `{opts}` grammar also accepts `<carrier>+{p,e,c,m}<factor>` tokens (e.g.
`solar+c0.5`), parsed into `adjustments: electricity: {p_nom_max | e_nom_max |
capital_cost | marginal_cost}: {carrier: factor}`.

```{warning}
These tokens are not usable ([#814](https://github.com/PyPSA/pypsa-usa/issues/814)): no
config layer defines a top-level `adjustments:` key, so any `+` token raises
`KeyError: 'adjustments'` in every rule that carries `{opts}`; and even with the key
present no script applies `adjustments:` to the network. The mechanism is inherited from
PyPSA-Eur.
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
  rural/urban population ratio; only when `sector: service_sector: split_urban_rural` is
  false; see the [heat-pump documentation](./data-services.md#heat-pumps).
- **Natural-gas import/export limits** — bounds gas trade with out-of-scope regions to
  ranges around historical volumes, per link and investment period; only when
  `sector: natural_gas: imports` is set; upper bounds given as `inf` are skipped; see the
  [natural-gas sector page](./data-naturalgas.md).
- **Water-heater storage** — forces water-heating demand to be served through the storage
  buffer; only when `sector: service_sector: water_heating: simple_storage` is **false**
  (the default is true, so it is off by default); see the
  [service sector page](./data-services.md).
- **EV generation policy** — caps electric-vehicle-served transport demand per mode and
  horizon as an unweighted sum over snapshots; only when
  `sector: transport_sector: ev_policy` is non-empty; see the
  [transportation sector page](./data-transportation.md).
- **Sector demand response** — per-sector (and optionally per-carrier) shiftable-load
  bounds, with `shift` in **percent** (the power-sector `shift` is per unit); see
  {ref}`demand response <data-sector-coupling>`.
- **Industrial fossil minimum** — a floor on fossil-served industrial heat per state,
  unweighted; only when `sector: industrial_sector: min_fossil_generation` (percent)
  exceeds 0.1; see the [industrial sector page](./data-industrial.md).
