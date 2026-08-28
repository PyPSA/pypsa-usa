# Seam-plant leak: empirical quantification (2026-08-23)

Scratch analysis backing the `SEAM_PLANT_MAX_KM` bound added to
`workflow/scripts/add_electricity.py::filter_plants_by_region`
(branch `proto/seam-plant-bound`). Pending ledger countersignature.

## The defect

`filter_plants_by_region` filters plants by `sjoin` against `regions_onshore` /
`regions_offshore`. Since DL-11 (`88bede47`, "Scope empty-county sweep to
model_topology.include footprint") those layers tile only the model footprint in
scoped runs — a `model_topology.include: {reeds_state: [CA]}` run tiles 409.8k km²
instead of the pre-fix 2.93M km².

The `plants_must_add` fallback, however, is unconditional. It selects plants that

1. intersect **no** ReEDS shape of the run's interconnect (`reeds_shapes`), **and**
2. do intersect some **national** ReEDS shape (`repo_data/geospatial/Reeds_Shapes/rb_and_ba_areas.shp`), **and**
3. whose ReEDS-membership `interconnect` disagrees with their EIA-derived
   `interconnection` (`nerc_region` → `const.NERC_REGION_MAPPER`),

and concatenates them into `plants_filt` with no test against the (footprint-shrunk)
regions. `match_plant_to_bus`'s second pass then assigns any plant lacking a zone
match to the nearest network bus with **no distance bound**. Net effect in a CA-only
run: an out-of-footprint fleet attaches to California buses.

## Method

Replication script: [`quantify_seam.py`](#appendix-quantify_seampy). It ports
`load_powerplants` and the `plants_must_add` branch of `filter_plants_by_region`
verbatim, then measures each entry's distance to the union of the run's
`regions_onshore_s` + `regions_offshore_s` in **EPSG:5070**.

Inputs (read-only from the main checkout, `workflow/`):

| role | path |
|---|---|
| plants | `resources/powerplants/powerplants.csv` |
| interconnect ReEDS shapes | `resources/equivalence/geospatial/western/reeds_shapes.geojson` (33 WECC zones) |
| national ReEDS shapes | `repo_data/geospatial/Reeds_Shapes/rb_and_ba_areas.shp` (134 zones, EPSG:3857) |
| membership | `repo_data/ReEDS_Constraints/membership.csv` |
| CA footprint (DL-11-fixed) | `resources/equivalence/geospatial/western/regions_onshore_s.geojson` (1,972 regions, 409,842 km²) + `regions_offshore_s.geojson` (42) |
| full-WECC proxy A | `resources/equivalence/geospatial/western/reeds_shapes.geojson` union |
| full-WECC proxy B | `resources/test/geospatial/western/regions_onshore.geojson` (4,762 regions, 2,934,751 km² — the *pre*-DL-11 CA build, which tessellates ~the whole WECC footprint) |

Run parameters match `config/config.equivalence.yaml`: `interconnect: western`,
`planning_horizons: [2030]`, `simpl: ''`, `model_topology.include.reeds_state: [CA]`.

Note the rule input is `regions_onshore_s{simpl}.geojson`, not `regions_onshore.geojson`.
With `simpl: ''` these are the `_s` files; their unions are equivalent
(both 1,972 regions / 409,842 km²).

## Population funnel

```
plants (western, 2030 investment period):                     6050
plants intersecting no western ReEDS shape:                     80
  of those, landing in a NATIONAL ReEDS shape:                  34
    -> plants_must_add (interconnect != interconnection):       23   1887.4 MW
    -> remaining_plants (nearshore sjoin_nearest path):         11    230.0 MW
  not in any national shape (dropped by the sjoin):             46   3326.3 MW  (CA 41 / WA 5 — legit coastal/offshore imprecision)
duplicated generator_name rows in plants_must_add:               0
```

The `plants_must_add` population is **23 plants / 1,887.4 MW**, spread over exactly the
four states named in the defect report (NM, MT, SD, IN):

| state | plants | MW |
|---|---:|---:|
| NM | 16 | 1112.1 |
| MT | 5 | 370.3 |
| SD | 1 | 210.0 |
| IN | 1 | 195.0 |

| carrier | plants | MW |
|---|---:|---:|
| onwind | 8 | 1416.5 |
| solar | 10 | 281.5 |
| hydro | 4 | 162.4 |
| oil | 1 | 27.0 |

## Keep/drop against the CA footprint

Distance to `union(regions_onshore_s, regions_offshore_s)`, EPSG:5070:

| max_km | KEPT (n) | KEPT (MW) | DROPPED (n) | DROPPED (MW) |
|---:|---:|---:|---:|---:|
| 50 | 0 | 0.0 | 23 | 1887.4 |
| 100 | 0 | 0.0 | 23 | 1887.4 |
| 200 | 0 | 0.0 | 23 | 1887.4 |

The population is **bimodally far**: the nearest entry is 890 km from the CA footprint.
The threshold choice is therefore not sensitive between 50 and 200 km for this run —
100 km simply preserves headroom for genuine near-seam plants that a different scoped
footprint (e.g. a two-state include) would produce.

### Per-plant table (100 km case)

Sorted by distance to the CA footprint. `dist to WECC` is proxy A (the western
`reeds_shapes` union), shown to make the gate argument below concrete.

| generator_name | plant_name_eia | state | carrier | MW | dist to CA footprint (km) | dist to WECC (km) | 100 km verdict |
|---|---|---|---|---:|---:|---:|---|
| SEV NM Phase 2_58479_NMP2 | SEV NM Phase 2 | NM | solar | 2.0 | 890.1 | 35.0 | DROP |
| Chaves County Solar II_66405_CCS2 | Chaves County Solar II | NM | solar | 30.0 | 894.1 | 40.6 | DROP |
| GSE NM1_58576_NMP1 | GSE NM1 | NM | solar | 2.3 | 900.8 | 44.9 | DROP |
| New Mexico Wind Energy Center_56097_GE15 | New Mexico Wind Energy Center | NM | onwind | 204.0 | 919.8 | 3.2 | DROP |
| SPS5 Hopi_57740_1 | SPS5 Hopi | NM | solar | 10.1 | 939.7 | 57.3 | DROP |
| Quay County_58125_1 | Quay County | NM | oil | 27.0 | 949.8 | 6.8 | DROP |
| Oso Grande Wind Farm_63502_OGW24 | Oso Grande Wind Farm | NM | onwind | 33.8 | 975.4 | 117.5 | DROP |
| Oso Grande Wind Farm_63502_OGW45 | Oso Grande Wind Farm | NM | onwind | 216.0 | 975.4 | 117.5 | DROP |
| SoCore Clovis 1_64524_79100 | SoCore Clovis 1 | NM | solar | 2.0 | 992.6 | 63.2 | DROP |
| Broadview Energy KW, LLC_60152_1 | Broadview Energy KW, LLC | NM | onwind | 142.6 | 999.8 | 72.0 | DROP |
| Broadview Energy JN, LLC_60145_1 | Broadview Energy JN, LLC | NM | onwind | 181.7 | 1005.1 | 74.6 | DROP |
| Grady Wind Energy Center, LLC_60317_1 | Grady Wind Energy Center, LLC | NM | onwind | 220.5 | 1006.5 | 75.5 | DROP |
| SPS4 Monument_57739_1 | SPS4 Monument | NM | solar | 10.1 | 1014.4 | 145.8 | DROP |
| SPS3 Lea_57738_1 | SPS3 Lea | NM | solar | 10.0 | 1018.3 | 142.8 | DROP |
| SPS2 Jal_57737_1 | SPS2 Jal | NM | solar | 10.0 | 1039.4 | 155.7 | DROP |
| SPS1 Dollarhide_57736_1 | SPS1 Dollarhide | NM | solar | 10.0 | 1045.3 | 159.9 | DROP |
| Clearwater Wind East_66183_CWE | Clearwater Wind East | MT | onwind | 207.9 | 1232.3 | 8.8 | DROP |
| Fort Peck_6623_1 | Fort Peck | MT | hydro | 41.2 | 1260.3 | 72.1 | DROP |
| Fort Peck_6623_3 | Fort Peck | MT | hydro | 41.2 | 1260.3 | 72.1 | DROP |
| Fort Peck_6623_5 | Fort Peck | MT | hydro | 40.0 | 1260.3 | 72.1 | DROP |
| Fort Peck_6623_4 | Fort Peck | MT | hydro | 40.0 | 1260.3 | 72.1 | DROP |
| Buffalo Ridge II LLC_57424_1 | Buffalo Ridge II LLC | SD | onwind | 210.0 | 1859.3 | 588.9 | DROP |
| Hardy Hills Solar Energy LLC_67852_HHSPV | Hardy Hills Solar Energy LLC | IN | solar | 195.0 | 2508.3 | 1304.8 | DROP |

## Why the gate is mandatory

Measured against a **full-western** footprint the same `plants_must_add` population is
mostly legitimate — an unconditional bound would therefore change unfiltered western
results, which the equivalence protocol forbids:

Proxy A — western `reeds_shapes` union:

| max_km | KEPT (n) | KEPT (MW) | DROPPED (n) | DROPPED (MW) |
|---:|---:|---:|---:|---:|
| 50 | 6 | 473.2 | 17 | 1414.2 |
| 100 | 15 | 1192.5 | 8 | 694.9 |
| 200 | 21 | 1482.4 | 2 | 405.0 |

Proxy B — the 2.93M km² pre-DL-11 tessellation, identical splits:

| max_km | KEPT (n) | KEPT (MW) | DROPPED (n) | DROPPED (MW) |
|---:|---:|---:|---:|---:|
| 50 | 6 | 473.2 | 17 | 1414.2 |
| 100 | 15 | 1192.5 | 8 | 694.9 |
| 200 | 21 | 1482.4 | 2 | 405.0 |

So a 100 km bound applied **unconditionally** would delete 8 plants / 694.9 MW —
including Hardy Hills (IN, 195 MW) and Buffalo Ridge II (SD, 210 MW) — from an
unfiltered western run. Gating on `model_topology.include` being truthy makes
unfiltered interconnect/usa runs byte-identical **by construction**: with the gate off
not a single statement changes.

## Caveats / findings that differ from the defect brief

1. **Population is 23 plants / 1,887.4 MW, not 27 / 1,890.6 MW.** Same four states,
   and every named exemplar reproduces exactly: Buffalo Ridge II SD 210.0 MW,
   Hardy Hills Solar IN 195.0 MW (2,508 km from the CA footprint), Fort Peck MT hydro
   162.4 MW across 4 units, NM 1,112.1 MW of wind/solar. The residual is 4 rows /
   3.2 MW. Alternative capacity columns do not explain it either
   (`summer_capacity_mw` = 1,911.8 MW, `winter_capacity_mw` = 1,917.1 MW). Most likely
   the brief counted at a slightly different stage (e.g. generator rows after
   `add_electricity` attach, or a different `powerplants.csv` snapshot). The
   discrepancy is immaterial: the bound removes whatever is in this population.
2. **Zero out-of-state plants leak through the primary `sjoin`.** Against the
   DL-11-fixed CA regions, `sjoin(plants, regions_onshore/offshore)` matches 2,995
   plants / 105,384.3 MW, of which **0** have `state != CA`. So the out-of-state
   slivers noted in the DL-11 commit are not themselves a plant-leak path;
   `plants_must_add` is the whole leak.
3. **`plants_nearshore` is dead code in practice.** `remaining_clean` is EPSG:4326
   while `regions_onshore.to_crs(epsg=3857)` is metres, so `gpd.sjoin_nearest(...,
   max_distance=2000)` compares degrees against metres and matches nothing
   (geopandas emits a CRS-mismatch warning). All 11 `remaining_plants`
   (230.0 MW, all SD: Ben French, Fall River Solar, Lange Gas Turbines) are silently
   dropped today. **Not touched by this fix** — flagged for a separate ticket, since
   repairing the CRS would be results-changing for unfiltered runs too.
4. **46 plants / 3,326.3 MW (CA 41, WA 5) are dropped before `plants_must_add`**
   because they intersect no national ReEDS shape at all (coastal/offshore
   imprecision, e.g. Diablo Canyon-style cases). Pre-existing, unchanged by this fix,
   and the intended target of the `plants_nearshore` path in (3).
5. The threshold is insensitive on this run (nearest entry 890 km). 100 km is a
   headroom choice, matching `nrel_caps_reassign.max_km`'s default, not a
   data-fitted one.

## Appendix: `quantify_seam.py`

```python
"""Quantify the seam-plant leak in filter_plants_by_region (add_electricity.py).

Replicates the `plants_must_add` population exactly as add_electricity.py builds
it for interconnect=western, then measures each entry's distance to the union of
the run's regions_onshore + regions_offshore (the DL-11-fixed CA footprint) in
EPSG:5070, and reports keep/drop splits at several max_km thresholds.

Read-only against the MAIN checkout's artifacts.
"""

import sys

import geopandas as gpd
import pandas as pd

MAIN = "/Users/kamrantehranchi/Local_Documents/pypsa-usa/workflow/"
EQ = MAIN + "resources/equivalence/geospatial/western/"

NERC_REGION_MAPPER = {
    "WECC": "western",
    "TRE": "texas",
    "SERC": "eastern",
    "RFC": "eastern",
    "NPCC": "eastern",
    "MRO": "eastern",
}

INTERCONNECT = "western"
INVESTMENT_PERIODS = [2030]


def load_powerplants(plants_fn, investment_periods, interconnect):
    """Verbatim port of add_electricity.load_powerplants."""
    plants = pd.read_csv(plants_fn)
    plants = plants.set_index("generator_name")
    plants["current_planned_generator_operating_date"] = pd.to_datetime(
        plants["current_planned_generator_operating_date"],
    )
    plants["generator_retirement_date"] = pd.to_datetime(
        plants["generator_retirement_date"]
    )
    plants.loc[plants.operational_status == "proposed", "build_year"] = plants.loc[
        plants.operational_status == "proposed",
        "current_planned_generator_operating_date",
    ].dt.year
    retirement_date = pd.to_datetime("2100-01-01")
    plants.loc[
        plants.operational_status.isin(["existing", "proposed"]),
        "generator_retirement_date",
    ] = retirement_date
    plants.loc[plants.generator_retirement_date.isna(), "generator_retirement_date"] = (
        pd.to_datetime("1900-01-01")
    )
    plants = plants[plants.build_year <= investment_periods[0]]
    plants = plants[plants.generator_retirement_date.dt.year > investment_periods[0]]
    plants = plants[plants.nerc_region != "non-conus"]
    if (interconnect is not None) & (interconnect != "usa"):
        plants["interconnection"] = plants["nerc_region"].map(NERC_REGION_MAPPER)
        plants = plants[plants.interconnection == interconnect]
    return plants


def build_plants_must_add(plants, reeds_shapes, all_reeds_shapes, reeds_memberships):
    """Verbatim port of the plants_must_add branch of filter_plants_by_region."""
    plants = plants.copy()
    plants["geometry"] = gpd.points_from_xy(
        plants.longitude, plants.latitude, crs="EPSG:4326"
    )
    gdf_plants = gpd.GeoDataFrame(plants, geometry="geometry")

    plants_in_regions = gpd.sjoin(
        gdf_plants, reeds_shapes, how="inner", predicate="intersects"
    )
    plants_no_region = gdf_plants[~gdf_plants.index.isin(plants_in_regions.index)]
    print(f"plants (western, 2030): {len(gdf_plants)}")
    print(f"plants with no western ReEDS region: {len(plants_no_region)}")
    if plants_no_region.empty:
        return gpd.GeoDataFrame()

    plants_no_region = plants_no_region.to_crs(epsg=3857)
    pnr_all = gpd.sjoin(
        plants_no_region.reset_index(),
        all_reeds_shapes,
        how="inner",
        predicate="intersects",
    )
    pnr_all = pnr_all.to_crs(epsg=4326)
    reeds_memberships = reeds_memberships.copy()
    reeds_memberships.loc[reeds_memberships.interconnect == "ercot", "interconnect"] = (
        "texas"
    )
    pnr_all = pnr_all.merge(
        reeds_memberships[["ba", "interconnect"]],
        left_on="rb",
        right_on="ba",
        how="left",
    )
    print(f"of those, landing in a NATIONAL ReEDS shape: {len(pnr_all)}")
    plants_must_add = pnr_all[pnr_all.interconnect != pnr_all.interconnection]
    remaining = pnr_all[pnr_all.interconnect == pnr_all.interconnection]
    print(
        f"  -> plants_must_add (interconnect != EIA interconnection): {len(plants_must_add)}"
    )
    print(
        f"  -> remaining_plants (sjoin_nearest, 2000 m bound):        {len(remaining)}"
    )
    plants_must_add = plants_must_add.set_index("generator_name")
    dupes = int(plants_must_add.index.duplicated().sum())
    print(f"  -> duplicated generator_name rows in plants_must_add: {dupes}")
    plants_must_add = plants_must_add[~plants_must_add.index.duplicated()]
    return gpd.GeoDataFrame(plants_must_add, geometry="geometry", crs="EPSG:4326")


def distance_km_to_regions(gdf, regions_list):
    """Distance in km from each plant to the union of the given region layers (EPSG:5070)."""
    geoms = [
        r.to_crs(epsg=5070).geometry
        for r in regions_list
        if r is not None and not r.empty
    ]
    union = pd.concat(geoms).union_all()
    return gdf.to_crs(epsg=5070).distance(union) / 1000.0


def main():
    plants = load_powerplants(
        MAIN + "resources/powerplants/powerplants.csv",
        INVESTMENT_PERIODS,
        INTERCONNECT,
    )
    reeds_shapes = gpd.read_file(EQ + "reeds_shapes.geojson")
    all_reeds_shapes = gpd.read_file(
        MAIN + "repo_data/geospatial/Reeds_Shapes/rb_and_ba_areas.shp"
    )
    memberships = pd.read_csv(MAIN + "repo_data/ReEDS_Constraints/membership.csv")

    must_add = build_plants_must_add(
        plants, reeds_shapes, all_reeds_shapes, memberships
    )
    cap_col = "p_nom"
    print(
        f"\nplants_must_add total: {len(must_add)} plants / {must_add[cap_col].sum():.1f} MW"
    )

    # --- CA footprint (DL-11-fixed regions actually consumed by add_electricity) ---
    on_ca = gpd.read_file(EQ + "regions_onshore_s.geojson")
    off_ca = gpd.read_file(EQ + "regions_offshore_s.geojson")
    must_add = must_add.copy()
    must_add["dist_km_ca"] = distance_km_to_regions(must_add, [on_ca, off_ca])

    # --- Full-western proxy: union of the interconnect's ReEDS shapes ---
    must_add["dist_km_western"] = distance_km_to_regions(must_add, [reeds_shapes])

    for label, col in (
        ("CA footprint (regions_onshore_s + regions_offshore_s)", "dist_km_ca"),
        (
            "full western reeds_shapes union (proxy for unfiltered run)",
            "dist_km_western",
        ),
    ):
        print(f"\n=== {label} ===")
        for km in (50.0, 100.0, 200.0):
            keep = must_add[must_add[col] <= km]
            drop = must_add[must_add[col] > km]
            print(
                f"  max_km={km:6.1f}: KEPT {len(keep):3d} / {keep[cap_col].sum():9.1f} MW"
                f"   DROPPED {len(drop):3d} / {drop[cap_col].sum():9.1f} MW",
            )

    cols = [
        "plant_name_eia",
        "state",
        "carrier",
        cap_col,
        "dist_km_ca",
        "dist_km_western",
    ]
    tbl = must_add.reset_index()[["generator_name", *cols]].sort_values("dist_km_ca")

    print("\n=== markdown (100 km case, CA footprint) ===")
    tbl["verdict_100km"] = ["KEEP" if d <= 100.0 else "DROP" for d in tbl["dist_km_ca"]]
    print(
        "| generator_name | plant_name_eia | state | carrier | MW | "
        "dist to CA footprint (km) | dist to WECC (km) | 100 km verdict |",
    )
    print("|---|---|---|---|---:|---:|---:|---|")
    for _, r in tbl.iterrows():
        print(
            f"| {r['generator_name']} | {r['plant_name_eia']} | {r['state']} | {r['carrier']} | "
            f"{r[cap_col]:.1f} | {r['dist_km_ca']:.1f} | {r['dist_km_western']:.1f} | {r['verdict_100km']} |",
        )

    by_state = (
        must_add.groupby("state")[cap_col]
        .agg(["count", "sum"])
        .sort_values("sum", ascending=False)
    )
    print("\n=== by state ===")
    print(by_state.to_string())
    by_carrier = (
        must_add.groupby("carrier")[cap_col]
        .agg(["count", "sum"])
        .sort_values("sum", ascending=False)
    )
    print("\n=== by carrier ===")
    print(by_carrier.to_string())


if __name__ == "__main__":
    sys.exit(main())
```
