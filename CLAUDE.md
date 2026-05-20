# pypsa-usa — Oregon WWS Project

## Project Overview

This repo is being used for a Stanford WWS (Wind, Water, and Sunlight) class project modeling a 100% WWS transition for Oregon by 2050. The work lives on the `hdk-oregon_wws` branch and uses the existing pypsa-usa pipeline with an Oregon-specific config and scenario.

**Goal:** Follow the Jacobson et al. WWS roadmap methodology for Oregon — project energy demand to 2050, electrify all sectors, size WWS generators, and perform a grid stability analysis using pypsa-usa.

## Project Structure

- **Config:** `workflow/config/config.oregon_wws.yaml` — Oregon-specific scenario config (to be created)
- **Analysis notebooks:** `workflow/notebooks/` — demand conversion spreadsheet validation, resource analysis, results visualization
- **Scenario:** Western interconnect, Oregon state scope, 2050 planning horizon, 100% WWS

## WWS Roadmap Steps → pypsa-usa Mapping

| Step | Description | Tool |
|------|-------------|------|
| 1–2 | Project BAU demand → electrify sectors | EIA data + spreadsheet |
| 3 | Compute demand reduction from transition | Spreadsheet / notebook |
| 4 | WWS resource analysis (wind, solar, hydro, EGS) | pypsa-usa resource output + notebook |
| 5 | Size generators to meet demand | pypsa-usa optimization output |
| 6 | Avoided energy/air/climate costs | Post-processing notebook |
| 7 | Grid stability (dispatch, storage, balancing) | **pypsa-usa core** |
| 8–11 | Jobs, footprint, timeline, policy | Spreadsheet / written analysis |

## Oregon Energy Data

**Source:** EIA State Energy Consumption Estimates (most recent year: 2023)  
**Tables:** Residential, Commercial, Industrial, Transportation (separate tables, bottom half = trillion BTU/year)  
**Columns needed:** Coal, Natural gas, Petroleum (total), Biomass, Geothermal, Solar (heat), Electricity  
**Note:** Do NOT include "Electrical system energy losses" — instead add 4–6% grid losses after WWS conversion (except rooftop PV, which has no grid losses).

## WWS Resources for Oregon

Key resources to include in the model:
- **Onshore wind** — Columbia River Gorge, eastern Oregon (high capacity factor)
- **Offshore wind** — Oregon coast (floating offshore; significant undeveloped potential)
- **Solar PV** — utility-scale in eastern/southern Oregon; rooftop in population centers
- **Hydropower** — existing firm capacity (Columbia River system); treated as dispatchable storage
- **EGS (Enhanced Geothermal Systems)** — Cascade Range; already modeled in this repo
- **Wave energy** — Oregon coast; nascent technology, minor role

## Config Setup

Create `workflow/config/config.oregon_wws.yaml` based on `config.tutorial.yaml` with:

```yaml
run:
  name: "Oregon_WWS_2050"

scenario:
  interconnect: [western]
  clusters: [...]       # tune for Oregon resolution
  planning_horizons: [2050]
  sector: "G"           # include sector coupling for full electrification

model_topology:
  topological_boundaries: 'reeds_zone'
  include:
    reeds_state: ['OR']

electricity:
  renewable_carriers: [onwind, offwind_floating, solar, hydro]
  # EGS handled separately via EGS supply curve pipeline
```

## Key Conversion Factors (from class)

After collecting EIA end-use energy by fuel type, apply WWS conversion factors provided in class to convert each fuel/sector combination to WWS electricity demand. Then add 4–6% grid losses to get total generator output required.

Sectors: Residential, Commercial, Industrial, Transportation  
Special case: Agriculture/forestry/fishing rows use residential conversion factors, except petroleum → use transportation factor.

## Workflow Commands

```bash
# Activate environment
conda activate pypsa-usa  # or: source .venv/bin/activate

# Run the Oregon WWS scenario
snakemake -j4 --configfile workflow/config/config.oregon_wws.yaml

# Run on Sherlock (Stanford HPC)
bash workflow/run_slurm.sh
```

## Notes

- The EGS pipeline is already built into this repo — see `config.egs_western.yaml` and `workflow/notebooks/egs_supply_curves.ipynb` for reference.
- Hydro is treated as existing dispatchable capacity, not a new build option.
- Offshore wind uses `offwind_floating` carrier (relevant for Oregon's deep-water coast).
- For demand projection to 2050, use EIA 2023 baseline and apply growth/efficiency assumptions separately before passing to pypsa-usa as exogenous load.
