"""Assimilates data on existing generator and storage resources from PUDL, CEMS, ADS, and other sources."""

import logging
import re

import constants as const
import duckdb
import numpy as np
import pandas as pd
from _helpers import configure_logging, weighted_avg

logger = logging.getLogger(__name__)


# ======================================================================
# Unit-commitment parameter bounds
# ======================================================================
# The UC columns written to powerplants.csv come from the WECC ADS merge and,
# where ADS has no match, from `impute_missing_plant_data` group means. Both
# paths produce values that are physically impossible for the unit they land
# on:
#   * absolute ($, not $/MW) start-up costs imputed onto sub-MW units, giving
#     start costs up to 1e6 $/MW,
#   * baseload min-up/min-down times imputed onto aircraft-derivative peakers,
#   * EIA-860 `minimum_load_mw` reported at plant level but joined to a single
#     sub-unit, giving minimum_load_mw > p_nom (p_min_pu > 1),
#   * NaNs, which `add_electricity` otherwise fills with the PyPSA defaults
#     (min_up_time=0, min_down_time=0, start_up_cost=0) — i.e. a unit that is
#     infinitely flexible and free to cycle.
# With `conventional.unit_commitment: true` every one of those either distorts
# the dispatch or makes the MILP infeasible, so they are clamped here, at the
# source, rather than in each consumer.
#
# Bounds are per carrier and deliberately wide: the intent is to remove the
# impossible, not to overwrite plausible unit-specific data.
#
# Keys, all applied only to the committable (conventional thermal) carriers:
#   up_h / down_h : (min, max, default) minimum up / down time, in HOURS.
#                   `add_electricity` hands these to PyPSA's `min_up_time` /
#                   `min_down_time`, whose unit is SNAPSHOTS — the hours->
#                   snapshots rescale happens in prepare_network when the
#                   `{opts}` string reduces the temporal resolution.
#   ramp          : (min, max, default) ramp limit, per-unit of p_nom per HOUR.
#   start_usd_mw  : (min, max, default) start-up cost in $ per MW of p_nom.
#   p_min_pu      : (max, default) minimum stable level as a fraction of p_nom.
UC_CARRIERS = ("nuclear", "coal", "CCGT", "OCGT", "oil", "biomass", "geothermal", "waste")

UC_BOUNDS = {
    # Nuclear cycles on refueling timescales; NRC/EPRI load-follow studies put
    # min up/down at ~1 day, ramp at ~5-25 %/h, start cost ~100 $/MW, and a
    # minimum stable level of 40-60 % (French load-follow practice).
    "nuclear": {
        "up_h": (8.0, 168.0, 24.0),
        "down_h": (8.0, 168.0, 24.0),
        "ramp": (0.05, 1.0, 0.25),
        "start_usd_mw": (20.0, 500.0, 100.0),
        "p_min_pu": (0.90, 0.50),
    },
    # Coal steam: NREL/Intertek cycling study — warm start 4-12 h, min load
    # 25-50 % of rating, ramp 20-60 %/h, warm-start cost 100-300 $/MW.
    "coal": {
        "up_h": (2.0, 48.0, 8.0),
        "down_h": (2.0, 48.0, 8.0),
        "ramp": (0.10, 1.0, 0.40),
        "start_usd_mw": (30.0, 600.0, 200.0),
        "p_min_pu": (0.80, 0.40),
    },
    # Combined cycle: CAISO/WECC ADS typical min up 2-6 h, min down 4-8 h,
    # min load 30-50 % (1x1 mode), ramp 40-100 %/h, warm start 50-100 $/MW.
    "CCGT": {
        "up_h": (1.0, 24.0, 4.0),
        "down_h": (1.0, 24.0, 6.0),
        "ramp": (0.20, 1.0, 0.60),
        "start_usd_mw": (20.0, 300.0, 75.0),
        "p_min_pu": (0.80, 0.40),
    },
    # Simple-cycle CT: designed to start in <1 h; min up/down 1 h, full-range
    # ramp, min load 20-50 %, start cost 80-120 $/MW (NREL cycling study).
    "OCGT": {
        "up_h": (1.0, 8.0, 1.0),
        "down_h": (1.0, 8.0, 1.0),
        "ramp": (0.20, 1.0, 1.0),
        "start_usd_mw": (20.0, 300.0, 100.0),
        "p_min_pu": (0.70, 0.30),
    },
    # Oil-fired units in the fleet are overwhelmingly small recip/CT peakers;
    # treat them like OCGT but allow a slightly wider start-cost band because
    # distillate start fuel is expensive.
    "oil": {
        "up_h": (1.0, 8.0, 1.0),
        "down_h": (1.0, 8.0, 1.0),
        "ramp": (0.20, 1.0, 1.0),
        "start_usd_mw": (20.0, 400.0, 100.0),
        "p_min_pu": (0.70, 0.30),
    },
    # Biomass steam: small stoker/BFB boilers, cycled rarely; min up/down of a
    # few hours, slow ramp, min load ~40 %.
    "biomass": {
        "up_h": (1.0, 24.0, 4.0),
        "down_h": (1.0, 24.0, 4.0),
        "ramp": (0.05, 1.0, 0.30),
        "start_usd_mw": (10.0, 300.0, 60.0),
        "p_min_pu": (0.80, 0.40),
    },
    # Geothermal binary/flash plants run baseload; ADS reports zero start cost
    # for every unit, which under UC makes cycling free. Floor it so the model
    # does not use geothermal as a zero-cost switching resource.
    "geothermal": {
        "up_h": (1.0, 24.0, 8.0),
        "down_h": (1.0, 24.0, 6.0),
        "ramp": (0.05, 1.0, 0.20),
        "start_usd_mw": (5.0, 200.0, 30.0),
        "p_min_pu": (0.90, 0.50),
    },
    # MSW/landfill-gas steam: baseload-ish, must keep burning feedstock; same
    # envelope as biomass.
    "waste": {
        "up_h": (1.0, 24.0, 6.0),
        "down_h": (1.0, 24.0, 4.0),
        "ramp": (0.05, 1.0, 0.30),
        "start_usd_mw": (5.0, 200.0, 30.0),
        "p_min_pu": (0.80, 0.40),
    },
}


def _clamp_series(
    values: pd.Series,
    lower: float,
    upper: float,
    default: float,
) -> tuple[pd.Series, int, int]:
    """
    Clamp `values` into [lower, upper] and fill NaN with `default`.

    Returns the cleaned series plus the number of values clamped and the number
    filled, so the caller can report both.
    """
    s = pd.to_numeric(values, errors="coerce")
    n_filled = int(s.isna().sum())
    clamped = s.clip(lower=lower, upper=upper)
    n_clamped = int((clamped != s).sum())  # NaN != NaN is False, so fills are not counted here
    return clamped.fillna(default), n_clamped, n_filled


def sanitize_uc_parameters(plants: pd.DataFrame) -> pd.DataFrame:
    """
    Clamp and fill the unit-commitment columns of the committable carriers.

    Operates per carrier using `UC_BOUNDS`. Rows of non-committable carriers
    (wind, solar, hydro, battery, ...) are left untouched — PyPSA never marks
    them committable, so their UC columns are inert.

    Guarantees, for every committable row:
      * `min_up_time` / `min_down_time` finite, positive, within the carrier band;
      * `ramp_limit_up` / `ramp_limit_down` finite, in (0, 1];
      * `start_up_cost` finite, non-negative, within the carrier's $/MW band, and
        still equal to `startup_cost_fixed + start_fuel_cost` (both components are
        rescaled by the same factor, as is `start_fuel_mmbtu`);
      * `minimum_load_mw / p_nom <= min(summer_derate, winter_derate)` — the
        feasibility invariant. `p_max_pu` for a conventional unit is the seasonal
        derate (see `add_electricity.apply_seasonal_capacity_derates`), so a
        larger `p_min_pu` leaves the unit no feasible output above zero. PyPSA
        can then only satisfy both bounds by pinning `status = 0`, silently
        deleting that capacity for the whole tighter season — and on a
        non-committable generator (an ADS must-run under
        `conventional.must_run`) it makes the LP outright infeasible.

    Counts of clamped and filled values are logged per carrier and parameter.
    """
    required = {
        "carrier",
        "p_nom",
        "min_up_time",
        "min_down_time",
        "ramp_limit_up",
        "ramp_limit_down",
        "start_up_cost",
        "startup_cost_fixed",
        "start_fuel_cost",
        "start_fuel_mmbtu",
        "minimum_load_mw",
        "summer_derate",
        "winter_derate",
    }
    missing = required - set(plants.columns)
    if missing:
        raise KeyError(f"sanitize_uc_parameters is missing required columns: {sorted(missing)}")

    report: list[dict] = []

    for carrier, bounds in UC_BOUNDS.items():
        mask = plants.carrier == carrier
        if not mask.any():
            continue
        sub = plants.loc[mask]
        p_nom = pd.to_numeric(sub.p_nom, errors="coerce").clip(lower=1e-3)

        # ---- min up / down time (hours) -----------------------------------
        for col, key in (("min_up_time", "up_h"), ("min_down_time", "down_h")):
            lo, hi, default = bounds[key]
            cleaned, n_clamped, n_filled = _clamp_series(sub[col], lo, hi, default)
            plants.loc[mask, col] = cleaned
            report.append(dict(carrier=carrier, param=col, clamped=n_clamped, filled=n_filled, n=int(mask.sum())))

        # ---- ramp limits (per-unit of p_nom per hour) ----------------------
        lo, hi, default = bounds["ramp"]
        for col in ("ramp_limit_up", "ramp_limit_down"):
            cleaned, n_clamped, n_filled = _clamp_series(sub[col], lo, hi, default)
            plants.loc[mask, col] = cleaned
            report.append(dict(carrier=carrier, param=col, clamped=n_clamped, filled=n_filled, n=int(mask.sum())))

        # ---- start-up cost ($, bounded in $/MW) ----------------------------
        lo, hi, default = bounds["start_usd_mw"]
        specific = pd.to_numeric(sub.start_up_cost, errors="coerce") / p_nom
        cleaned_specific, n_clamped, n_filled = _clamp_series(specific, lo, hi, default)
        new_total = cleaned_specific * p_nom
        old_total = pd.to_numeric(sub.start_up_cost, errors="coerce")
        old_fixed = pd.to_numeric(sub.startup_cost_fixed, errors="coerce")
        old_fuel = pd.to_numeric(sub.start_fuel_cost, errors="coerce")
        # Keep startup_cost_fixed + start_fuel_cost == start_up_cost. Where the
        # old total is missing or zero, or the two components do not reproduce it
        # (one of them is NaN), the split is undefined: the whole (clamped or
        # defaulted) cost becomes a fixed cost and the start fuel goes to zero.
        # The columns as written are only consistent to the 4 decimals they were
        # rounded to, so rebuild the fixed part as the residual rather than
        # scaling both parts and inheriting an amplified rounding error.
        splittable = (old_total > 0) & np.isclose(old_fixed + old_fuel, old_total, rtol=1e-4, atol=1e-3)
        ratio = (new_total / old_total.where(splittable)).astype(float)
        new_fuel = pd.Series(np.where(splittable, old_fuel.fillna(0) * ratio.fillna(0), 0.0), index=sub.index)
        plants.loc[mask, "start_up_cost"] = new_total
        plants.loc[mask, "start_fuel_cost"] = new_fuel
        plants.loc[mask, "startup_cost_fixed"] = new_total - new_fuel
        plants.loc[mask, "start_fuel_mmbtu"] = np.where(
            splittable,
            pd.to_numeric(sub.start_fuel_mmbtu, errors="coerce").fillna(0) * ratio.fillna(0),
            0.0,
        )
        report.append(
            dict(carrier=carrier, param="start_up_cost", clamped=n_clamped, filled=n_filled, n=int(mask.sum())),
        )

        # ---- minimum load / the p_min_pu <= p_max_pu feasibility invariant --
        p_min_max, p_min_default = bounds["p_min_pu"]
        derate = np.minimum(
            pd.to_numeric(sub.summer_derate, errors="coerce").fillna(1.0),
            pd.to_numeric(sub.winter_derate, errors="coerce").fillna(1.0),
        ).clip(lower=0.0, upper=1.0)
        # The unit may not be asked to run above what its worst season allows.
        # Compare against the derate truncated to the 4 decimals `set_parameters`
        # writes, so the invariant holds against the columns as they land in
        # powerplants.csv and not only against their full-precision values.
        ceiling = np.floor(np.minimum(derate, p_min_max) * 1e4) / 1e4
        raw_pu = pd.to_numeric(sub.minimum_load_mw, errors="coerce") / p_nom
        n_filled = int(raw_pu.isna().sum())
        clipped_pu = raw_pu.clip(lower=0.0, upper=ceiling)
        n_clamped = int((clipped_pu != raw_pu).sum())
        filled_pu = clipped_pu.fillna(np.minimum(ceiling, p_min_default))
        # `set_parameters` rounds every numeric column to 4 decimals on the way
        # out. On a sub-MW unit that rounding is a large relative step, and
        # rounding *up* would put p_min_pu back above the derate. Truncate to the
        # same 4 decimals here so the later round() cannot move the value at all.
        plants.loc[mask, "minimum_load_mw"] = np.floor((filled_pu * p_nom).astype(float) * 1e4) / 1e4
        report.append(
            dict(carrier=carrier, param="minimum_load_mw", clamped=n_clamped, filled=n_filled, n=int(mask.sum())),
        )

    if report:
        summary = pd.DataFrame(report)
        touched = summary[(summary.clamped > 0) | (summary.filled > 0)]
        logger.warning(
            "Unit-commitment bounds enforcement clamped %d and filled %d values across %d committable rows.",
            int(summary.clamped.sum()),
            int(summary.filled.sum()),
            int(plants.carrier.isin(UC_CARRIERS).sum()),
        )
        if not touched.empty:
            logger.warning(
                "Per-carrier UC clamp/fill counts:\n%s",
                touched.to_string(index=False),
            )
    return plants


def initialize_duckdb():
    duckdb.connect(database=":memory:", read_only=False)
    duckdb.query("INSTALL httpfs;")


def load_eia_operable_data(parquet_path: str):
    """Queries the parquet files directly for operable plant data."""
    # ges and p are pre-aggregated to one row per generator/plant before joining.
    # Without this, the LEFT JOINs multiply rows by ~24 years of EIA-860 history
    # before the GROUP BY collapses them, producing a large intermediate table.
    return duckdb.query(
        f"""
        WITH monthly_generators AS (
            SELECT
                plant_id_eia,
                generator_id,
                array_agg(unit_heat_rate_mmbtu_per_mwh ORDER BY report_date DESC) FILTER (WHERE unit_heat_rate_mmbtu_per_mwh IS NOT NULL)[1] AS unit_heat_rate_mmbtu_per_mwh
            FROM read_parquet('{parquet_path}/out_eia__monthly_generators.parquet')
            WHERE report_date >= '2023-01-01'
            GROUP BY plant_id_eia, generator_id
        ),
        ges_latest AS (
            SELECT
                plant_id_eia,
                generator_id,
                array_agg(max_charge_rate_mw ORDER BY report_date DESC) FILTER (WHERE max_charge_rate_mw IS NOT NULL)[1] AS max_charge_rate_mw,
                array_agg(max_discharge_rate_mw ORDER BY report_date DESC) FILTER (WHERE max_discharge_rate_mw IS NOT NULL)[1] AS max_discharge_rate_mw,
                array_agg(storage_technology_code_1 ORDER BY report_date DESC) FILTER (WHERE storage_technology_code_1 IS NOT NULL)[1] AS storage_technology_code_1
            FROM read_parquet('{parquet_path}/core_eia860__scd_generators_energy_storage.parquet')
            GROUP BY plant_id_eia, generator_id
        ),
        plants_latest AS (
            SELECT
                plant_id_eia,
                array_agg(nerc_region ORDER BY report_date DESC) FILTER (WHERE nerc_region IS NOT NULL)[1] AS nerc_region,
                array_agg(balancing_authority_code_eia ORDER BY report_date DESC) FILTER (WHERE balancing_authority_code_eia IS NOT NULL)[1] AS balancing_authority_code_eia
            FROM read_parquet('{parquet_path}/core_eia860__scd_plants.parquet')
            GROUP BY plant_id_eia
        )
        SELECT
            yg.plant_id_eia,
            yg.generator_id,
            array_agg(yg.plant_name_eia ORDER BY yg.report_date DESC) FILTER (WHERE yg.plant_name_eia IS NOT NULL)[1] AS plant_name_eia,
            array_agg(yg.capacity_mw ORDER BY yg.report_date DESC) FILTER (WHERE yg.capacity_mw IS NOT NULL)[1] AS capacity_mw,
            array_agg(yg.summer_capacity_mw ORDER BY yg.report_date DESC) FILTER (WHERE yg.summer_capacity_mw IS NOT NULL)[1] AS summer_capacity_mw,
            array_agg(yg.winter_capacity_mw ORDER BY yg.report_date DESC) FILTER (WHERE yg.winter_capacity_mw IS NOT NULL)[1] AS winter_capacity_mw,
            array_agg(yg.minimum_load_mw ORDER BY yg.report_date DESC) FILTER (WHERE yg.minimum_load_mw IS NOT NULL)[1] AS minimum_load_mw,
            array_agg(yg.energy_source_code_1 ORDER BY yg.report_date DESC) FILTER (WHERE yg.energy_source_code_1 IS NOT NULL)[1] AS energy_source_code_1,
            array_agg(yg.technology_description ORDER BY yg.report_date DESC) FILTER (WHERE yg.technology_description IS NOT NULL)[1] AS technology_description,
            array_agg(yg.operational_status ORDER BY yg.report_date DESC) FILTER (WHERE yg.operational_status IS NOT NULL)[1] AS operational_status,
            array_agg(yg.prime_mover_code ORDER BY yg.report_date DESC) FILTER (WHERE yg.prime_mover_code IS NOT NULL)[1] AS prime_mover_code,
            -- Deliberately NOT most-recent-non-null: a newer filing reporting NULL
            -- means the retirement announcement was withdrawn (e.g. Diablo Canyon
            -- post-SB846), and resurrecting an older filing's date would retire a
            -- unit its owner no longer plans to retire.
            array_agg(yg.planned_generator_retirement_date ORDER BY yg.report_date DESC)[1] AS planned_generator_retirement_date,
            array_agg(yg.energy_storage_capacity_mwh ORDER BY yg.report_date DESC) FILTER (WHERE yg.energy_storage_capacity_mwh IS NOT NULL)[1] AS energy_storage_capacity_mwh,
            array_agg(yg.generator_operating_date ORDER BY yg.report_date DESC) FILTER (WHERE yg.generator_operating_date IS NOT NULL)[1] AS generator_operating_date,
            array_agg(yg.state ORDER BY yg.report_date DESC) FILTER (WHERE yg.state IS NOT NULL)[1] AS state,
            array_agg(yg.latitude ORDER BY yg.report_date DESC) FILTER (WHERE yg.latitude IS NOT NULL)[1] AS latitude,
            array_agg(yg.longitude ORDER BY yg.report_date DESC) FILTER (WHERE yg.longitude IS NOT NULL)[1] AS longitude,
            first(ges.max_charge_rate_mw) AS max_charge_rate_mw,
            first(ges.max_discharge_rate_mw) AS max_discharge_rate_mw,
            first(ges.storage_technology_code_1) AS storage_technology_code_1,
            first(p.nerc_region) AS nerc_region,
            first(p.balancing_authority_code_eia) AS balancing_authority_code_eia,
            array_agg(yg.current_planned_generator_operating_date ORDER BY yg.report_date DESC) FILTER (WHERE yg.current_planned_generator_operating_date IS NOT NULL)[1] AS current_planned_generator_operating_date,
            array_agg(yg.operational_status_code ORDER BY yg.report_date DESC) FILTER (WHERE yg.operational_status_code IS NOT NULL)[1] AS operational_status_code,
            array_agg(yg.generator_retirement_date ORDER BY yg.report_date DESC) FILTER (WHERE yg.generator_retirement_date IS NOT NULL)[1] AS generator_retirement_date,
            array_agg(yg.fuel_type_code_pudl ORDER BY yg.report_date DESC) FILTER (WHERE yg.fuel_type_code_pudl IS NOT NULL)[1] AS fuel_type_code_pudl,
            first(mg.unit_heat_rate_mmbtu_per_mwh) AS unit_heat_rate_mmbtu_per_mwh
        FROM read_parquet('{parquet_path}/out_eia__yearly_generators.parquet') yg
        LEFT JOIN ges_latest ges
            ON yg.plant_id_eia = ges.plant_id_eia AND yg.generator_id = ges.generator_id
        LEFT JOIN plants_latest p
            ON yg.plant_id_eia = p.plant_id_eia
        LEFT JOIN monthly_generators mg
            ON yg.plant_id_eia = mg.plant_id_eia AND yg.generator_id = mg.generator_id
        WHERE
            yg.operational_status_code IN ('RE','OP', 'SC', 'SB', 'CO' ,'U', 'V', 'TS', 'T')
            AND yg.report_date >= '2023-01-01'
        GROUP BY yg.plant_id_eia, yg.generator_id
    """,
    ).to_df()


def load_heat_rates_data(parquet_path: str, start_date: str, end_date: str):
    """Queries the parquet files for heat rate and fuel cost data within the specified date range."""
    # yg and p are pre-aggregated to one row per generator/plant before joining.
    # Without this, monthly rows in the date range get cross-joined with all
    # ~24 years of EIA-860 history per generator, producing a multi-GB
    # intermediate table that downstream code only collapses with a mean().
    query = f"""
    WITH monthly_generators AS (
        SELECT
            plant_id_eia,
            generator_id,
            report_date,
            unit_heat_rate_mmbtu_per_mwh,
            fuel_cost_per_mwh,
            fuel_cost_per_mmbtu
        FROM read_parquet('{parquet_path}/out_eia__monthly_generators.parquet')
        WHERE operational_status = 'existing'
        AND report_date BETWEEN '{start_date}' AND '{end_date}'
        AND unit_heat_rate_mmbtu_per_mwh IS NOT NULL
    ),
    yg_latest AS (
        SELECT
            plant_id_eia,
            generator_id,
            array_agg(plant_name_eia ORDER BY report_date DESC) FILTER (WHERE plant_name_eia IS NOT NULL)[1] AS plant_name_eia,
            array_agg(capacity_mw ORDER BY report_date DESC) FILTER (WHERE capacity_mw IS NOT NULL)[1] AS capacity_mw,
            array_agg(energy_source_code_1 ORDER BY report_date DESC) FILTER (WHERE energy_source_code_1 IS NOT NULL)[1] AS energy_source_code_1,
            array_agg(technology_description ORDER BY report_date DESC) FILTER (WHERE technology_description IS NOT NULL)[1] AS technology_description,
            array_agg(operational_status ORDER BY report_date DESC) FILTER (WHERE operational_status IS NOT NULL)[1] AS operational_status,
            array_agg(prime_mover_code ORDER BY report_date DESC) FILTER (WHERE prime_mover_code IS NOT NULL)[1] AS prime_mover_code,
            array_agg(state ORDER BY report_date DESC) FILTER (WHERE state IS NOT NULL)[1] AS state
        FROM read_parquet('{parquet_path}/out_eia__yearly_generators.parquet')
        GROUP BY plant_id_eia, generator_id
    ),
    plants_latest AS (
        SELECT
            plant_id_eia,
            array_agg(nerc_region ORDER BY report_date DESC) FILTER (WHERE nerc_region IS NOT NULL)[1] AS nerc_region,
            array_agg(balancing_authority_code_eia ORDER BY report_date DESC) FILTER (WHERE balancing_authority_code_eia IS NOT NULL)[1] AS balancing_authority_code_eia
        FROM read_parquet('{parquet_path}/core_eia860__scd_plants.parquet')
        GROUP BY plant_id_eia
    )
    SELECT
        mg.plant_id_eia,
        mg.generator_id,
        mg.report_date,
        mg.unit_heat_rate_mmbtu_per_mwh,
        mg.fuel_cost_per_mwh,
        mg.fuel_cost_per_mmbtu,
        yg.plant_name_eia,
        yg.capacity_mw,
        yg.energy_source_code_1,
        yg.technology_description,
        yg.operational_status,
        yg.prime_mover_code,
        yg.state,
        p.nerc_region,
        p.balancing_authority_code_eia
    FROM monthly_generators mg
    LEFT JOIN yg_latest yg
        ON mg.plant_id_eia = yg.plant_id_eia AND mg.generator_id = yg.generator_id
    LEFT JOIN plants_latest p
        ON mg.plant_id_eia = p.plant_id_eia
    WHERE yg.operational_status = 'existing'
    ORDER BY mg.report_date DESC
    """
    return duckdb.query(query).to_df()


def set_non_conus(eia_data_operable):
    """Set NERC region and balancing authority code for non-CONUS plants."""
    eia_data_operable.loc[eia_data_operable.state.isin(["AK", "HI"]), "nerc_region"] = "non-conus"
    eia_data_operable.loc[
        eia_data_operable.state.isin(["AK", "HI"]),
        "balancing_authority_code",
    ] = "non-conus"


def set_derates(plants):
    plants["derate_summer_capacity"] = np.minimum(
        plants.summer_capacity_mw,
        plants.ads_maxcapmw.fillna(np.inf),
    )
    plants["derate_winter_capacity"] = np.minimum(
        plants.winter_capacity_mw,
        plants.ads_maxcapmw.fillna(np.inf),
    )

    plants["summer_derate"] = 1 - ((plants.p_nom - plants.derate_summer_capacity) / plants.p_nom)
    plants["winter_derate"] = 1 - ((plants.p_nom - plants.derate_winter_capacity) / plants.p_nom)
    plants.summer_derate = plants.summer_derate.clip(
        upper=1,
    ).clip(lower=0)
    plants.winter_derate = plants.winter_derate.clip(
        upper=1,
    ).clip(lower=0)
    # EIA-860 reports summer/winter capacity only on a representative generator
    # for multi-unit combined-cycle plants, leaving sub-units (e.g. CCGT LMB/LMC/STA)
    # with NaN derates. Treat missing derate info as "no derate" so downstream
    # p_max_pu construction does not propagate NaN across every snapshot.
    plants.summer_derate = plants.summer_derate.fillna(1.0)
    plants.winter_derate = plants.winter_derate.fillna(1.0)


# Create DataFrames from constants for mapping
eia_tech_map = pd.DataFrame(const.EIA_TECH_MAP).set_index("Technology")
eia_fuel_map = pd.DataFrame(const.EIA_FUEL_MAP).set_index("Energy Source 1")
eia_primemover_map = pd.DataFrame(const.EIA_PRIMEMOVER_MAP).set_index("Prime Mover")


def set_tech_fuels_primer_movers(eia_data_operable):
    """
    Maps technologies, fuels, and prime movers from EIA data to PyPSA carrier
    names.
    """
    maps = {
        "carrier": (
            eia_data_operable["technology_description"],
            eia_tech_map["tech_type"],
        ),
        "fuel_type": (
            eia_data_operable["energy_source_code_1"],
            eia_fuel_map["fuel_type"],
        ),
        "fuel_name": (
            eia_data_operable["energy_source_code_1"],
            eia_fuel_map["fuel_name"],
        ),
        "prime_mover_name": (
            eia_data_operable["prime_mover_code"],
            eia_primemover_map["prime_mover"],
        ),
    }
    for col, (data_col, map_df) in maps.items():
        eia_data_operable[col] = data_col.map(dict(zip(map_df.index, map_df.values)))


def standardize_col_names(columns, prefix="", suffix=""):
    """
    Standardize column names by removing spaces, converting to lowercase,
    removing parentheses, and adding prefix and suffix.
    """
    return [prefix + col.lower().replace(" ", "_").replace("(", "").replace(")", "") + suffix for col in columns]


def merge_ads_data(eia_data_operable):
    """Merges WECC ADS Data into the prepared EIA Data."""
    path_ads = snakemake.input.wecc_ads
    ads_thermal = pd.read_csv(
        path_ads + "/Thermal_General_Info.csv",
        skiprows=1,
    )
    ads_thermal = ads_thermal[
        [
            "GeneratorName",
            " Turbine Type",
            "MustRun",
            "MinimumDownTime(hr)",
            "MinimumUpTime(hr)",
            "MaxUpTime(hr)",
            "RampUp Rate(MW/minute)",
            "RampDn Rate(MW/minute)",
            "Startup Cost Fixed($)",
            "StartFuel(MMBTu)",
            "Startup Time",
            "VOM Cost",
        ]
    ]
    ads_thermal.columns = standardize_col_names(ads_thermal.columns)

    ads_ioc = pd.read_csv(
        path_ads + "/Thermal_IOCurve_Info.csv",
        skiprows=1,
    ).rename(columns={"Generator Name": "GeneratorName"})
    ads_ioc = ads_ioc[
        [
            "GeneratorName",
            "IOMaxCap(MW)",
            "IOMinCap(MW)",
            "MinInput(MMBTu)",
        ]
    ]
    ads_ioc.columns = standardize_col_names(ads_ioc.columns)

    # Merge ADS plant data with thermal IOC data
    ads_thermal_ioc = pd.merge(ads_thermal, ads_ioc, on="generatorname", how="left")

    # loading ads to match ads_name with generator key in order to link with ads thermal file
    ads = pd.read_csv(
        path_ads + "/GeneratorList.csv",
        skiprows=2,
        encoding="unicode_escape",
    )
    # pandas 3 str dtype: astype(str) preserves NaN instead of stringifying
    # to "nan"; keep the pandas-2 behavior these name-match keys relied on
    ads["Long Name"] = ads["Long Name"].fillna("nan").astype(str)
    ads["Name"] = ads["Name"].str.replace(" ", "")
    ads["Name"] = ads["Name"].apply(lambda x: re.sub(r"[^a-zA-Z0-9]", "", x).lower())
    ads["Long Name"] = ads["Long Name"].str.replace(" ", "")
    ads["Long Name"] = ads["Long Name"].apply(
        lambda x: re.sub(r"[^a-zA-Z0-9]", "", x).lower(),
    )
    ads["SubType"] = ads["SubType"].apply(
        lambda x: re.sub(r"[^a-zA-Z0-9]", "", x).lower(),
    )
    ads = ads.rename(
        {
            "Name": "ads_name",
            "Long Name": "ads_long_name",
            "SubType": "subtype",
            "Commission Date": "commission_date",
            "Retirement Date": "retirement_date",
            "Area Name": "balancing_area",
        },
        axis=1,
    )
    ads = ads.rename(str.lower, axis="columns")
    ads["long id"] = ads["long id"].astype(str)
    ads = ads.loc[
        :,
        ~ads.columns.isin(
            ["save to binary", "county", "city", "zipcode", "internalid"],
        ),
    ]
    ads_name_key_dict = dict(zip(ads["ads_name"], ads["generatorkey"]))
    ads.columns

    ads_thermal_ioc["generator_name_alt"] = (
        ads_thermal_ioc["generatorname"].str.replace(" ", "").str.lower().str.replace("_", "").str.replace("-", "")
    )
    ads_thermal_ioc["generator_key"] = ads_thermal_ioc["generator_name_alt"].map(
        ads_name_key_dict,
    )

    # Identify Generators not in ads generator list that are in the IOC curve.
    # This could potentially be matched with manual work.
    ads_thermal_ioc[ads_thermal_ioc.generator_key.isna()]

    # Merge ads thermal_IOC data with ads generator data
    # Only keeping thermal plants for their heat rate and ramping data
    ads_complete = ads_thermal_ioc.merge(
        ads,
        left_on="generator_key",
        right_on="generatorkey",
        how="left",
    )
    ads_complete.columns = standardize_col_names(ads_complete.columns, prefix="ads_")
    ads_complete = ads_complete.loc[~ads_complete.ads_state.isin(["MX"])]

    # load mapping file to match the ads thermal to the eia_plants_locs file
    eia_ads_mapper = pd.read_csv(snakemake.input.eia_ads_generator_mapping)
    eia_ads_mapper = eia_ads_mapper.loc[
        :,
        [
            "generatorkey",
            "ads_name",
            "plant_id_ads",
            "plant_id_eia",
            "generator_id_ads",
        ],
    ]
    eia_ads_mapper.columns = standardize_col_names(
        eia_ads_mapper.columns,
        prefix="mapper_",
    )
    eia_ads_mapper = eia_ads_mapper.dropna(subset=["mapper_plant_id_eia"])
    eia_ads_mapper.mapper_plant_id_eia = eia_ads_mapper.mapper_plant_id_eia.astype(int)
    eia_ads_mapper.mapper_ads_name = eia_ads_mapper.mapper_ads_name.astype(str)
    eia_ads_mapper.mapper_generatorkey = eia_ads_mapper.mapper_generatorkey.astype(int)

    ads_complete = ads_complete.dropna(subset=["ads_generator_key"])
    ads_complete.ads_generator_key = ads_complete.ads_generator_key.astype(int)
    eia_ads_mapper.mapper_generatorkey = eia_ads_mapper.mapper_generatorkey.astype(int)

    eia_ads_mapping = pd.merge(
        ads_complete,
        eia_ads_mapper,
        left_on="ads_generator_key",
        right_on="mapper_generatorkey",
        how="inner",
    )

    # Merge EIA and ADS Data
    eia_ads_merged = pd.merge(
        left=eia_data_operable,
        right=eia_ads_mapping,
        left_on=["plant_id_eia", "generator_id"],
        right_on=["mapper_plant_id_eia", "mapper_generator_id_ads"],
        how="left",
    )
    eia_ads_merged = eia_ads_merged.drop(columns=eia_ads_mapper.columns)
    eia_ads_merged = eia_ads_merged.drop(
        columns=[
            "ads_generator_name_alt",
            "ads_generator_key",
            "ads_generatorkey",
            "ads_ads_name",
            "ads_bus_id",
            "ads_bus_name",
            "ads_bus_kv",
            "ads_unit_id",
            "ads_generator_typeid",
            "ads_subtype",
            "ads_long_id",
            "ads_ads_long_name",
            "ads_state",
            "ads_btm",
            "ads_devstatus",
            "ads_retirement_date",
            "ads_commission_date",
            "ads_servicestatus",
        ],
    )
    eia_ads_merged = eia_ads_merged.drop_duplicates(
        subset=["plant_id_eia", "generator_id"],
        keep="first",
    )

    return eia_ads_merged


def impute_missing_plant_data(
    plants: pd.DataFrame,
    aggregation_fields: list[str],
    data_fields: list[str],
) -> pd.DataFrame:
    """
    Imputes missing data for the`data_fields` in the plants dataframe based on
    the average values of the  `aggregation_fields`.
    """
    # Calculate the weighted averages excluding NaNs
    weighted_averages = (
        plants.groupby(aggregation_fields)[plants.columns]
        .apply(
            lambda x: pd.Series(
                {field: weighted_avg(x, field, "p_nom") for field in data_fields},
            ),
        )
        .reset_index()
    )

    # Merge weighted averages back into the original DataFrame
    plants_merged = pd.merge(
        plants.reset_index(),
        weighted_averages,
        on=aggregation_fields,
        suffixes=("", "_weighted"),
    )

    # Fill NaN values using the weighted averages
    for field in data_fields:
        plants_merged[field] = plants_merged[field].fillna(
            plants_merged[f"{field}_weighted"],
        )
        if field in ["fuel_cost", "heat_rate"]:
            # need to properly assign weighted average to the entries which took their values
            # if the field has values equal to the _weighted column, then the source is the weighted average
            plants_merged[f"{field}_source"] = np.where(
                plants_merged[field] == plants_merged[f"{field}_weighted"],
                "weighted_average",
                plants_merged[f"{field}_source"],
            )
    # Drop the weighted average columns after filling NaNs
    plants_merged = plants_merged.drop(
        columns=[f"{field}_weighted" for field in data_fields],
    )
    return plants_merged.set_index("generator_name")


def set_parameters(plants: pd.DataFrame):
    """
    Sets generator naming schemes, updates parameter names, and imputes missing
    data.
    """
    # EIA leaves nerc_region NULL for plants that first appear in a recent 860
    # vintage, and downstream add_electricity maps nerc_region -> interconnect,
    # so a plain isin() silently deletes new-build (e.g. 2.2 GW of CA renewables
    # under PUDL v2025.5.0). Impute a representative NERC region from the
    # plant's state before filtering; the representative choice round-trips
    # through const.NERC_REGION_MAPPER for interconnect scoping.
    interconnect_to_nerc = {"western": "WECC", "texas": "TRE", "eastern": "SERC"}
    null_nerc = plants.nerc_region.isna()
    if null_nerc.any():
        imputed = plants.loc[null_nerc, "state"].map(const.STATES_INTERCONNECT_MAPPER).map(interconnect_to_nerc)
        plants.loc[null_nerc, "nerc_region"] = imputed
        logger.info(
            "Imputed nerc_region from state for %d plants (%.0f MW) with NULL nerc_region "
            "(recent EIA filings not yet backfilled).",
            imputed.notna().sum(),
            plants.loc[null_nerc & plants.nerc_region.notna(), "capacity_mw"].sum(),
        )
    plants = plants[plants.nerc_region.isin(["WECC", "TRE", "MRO", "SERC", "RFC", "NPCC"])]
    plants = plants.rename(
        {
            "fuel_cost_per_mwh_source": "fuel_cost_source",
            "unit_heat_rate_mmbtu_per_mwh_source": "heat_rate_source",
        },
        axis=1,
    )

    plants["generator_name"] = (
        plants.plant_name_eia.astype(str)
        + "_"
        + plants.plant_id_eia.astype(str)
        + "_"
        + plants.generator_id.astype(str)
    )
    plants = plants.set_index("generator_name")
    plants["p_nom"] = plants.pop("capacity_mw")
    plants["build_year"] = plants.pop("generator_operating_date").dt.year
    # pandas 3 str dtype: astype(str) preserves NaN instead of stringifying to
    # "nan"; keep the pandas-2 "nan0s" bucket so plants with no operating date
    # (proposed units) still match a group in impute_missing_plant_data's inner merge
    plants["build_decade"] = plants.build_year.astype(str).fillna("nan").str[:3] + "0s"
    plants["heat_rate"] = plants.pop("unit_heat_rate_mmbtu_per_mwh")
    plants["vom"] = plants.pop("ads_vom_cost")
    plants["fuel_cost"] = plants.pop("fuel_cost_per_mmbtu")

    zero_mc_fuel_types = ["solar", "wind", "hydro", "geothermal", "battery"]
    plants.loc[plants.fuel_type.isin(zero_mc_fuel_types), "fuel_cost"] = 0
    plants = impute_missing_plant_data(
        plants,
        ["state", "fuel_name"],
        ["fuel_cost"],
    )
    plants = impute_missing_plant_data(
        plants,
        ["balancing_authority_code_eia", "fuel_name"],
        ["fuel_cost"],
    )
    plants = impute_missing_plant_data(
        plants,
        ["nerc_region", "fuel_name"],
        ["fuel_cost"],
    )

    plants = impute_missing_plant_data(plants, ["fuel_name"], ["fuel_cost"])
    plants = impute_missing_plant_data(plants, ["prime_mover_code"], ["fuel_cost"])
    plants.loc[plants.carrier.isin(["nuclear"]), "fuel_cost"] = np.float32(0.71)  # 2023 AEO

    # Unit Commitment Parameters
    plants["start_fuel_mmbtu"] = plants.pop("ads_startfuelmmbtu")
    plants["startup_cost_fixed"] = plants.pop("ads_startup_cost_fixed$")
    plants["min_down_time"] = plants.pop("ads_minimumdowntimehr")
    plants["min_up_time"] = plants.pop("ads_minimumuptimehr")
    plants.loc[plants.fuel_type.isin(["solar", "wind", "hydro", "battery"]), "start_fuel_mmbtu"] = 0
    plants.loc[plants.fuel_type.isin(["solar", "wind", "hydro", "battery"]), "startup_cost_fixed"] = 0

    # Ramp Limit Parameters
    plants["ramp_limit_up"] = (plants.pop("ads_rampup_ratemw/minute") / plants.p_nom * 60).clip(
        lower=0,
        upper=1,
    )  # MW/min to p.u./hour
    plants["ramp_limit_down"] = (plants.pop("ads_rampdn_ratemw/minute") / plants.p_nom * 60).clip(
        lower=0,
        upper=1,
    )  # MW/min to p.u./hour

    # Impute parameters for UC and infrastructure characteristics
    data_fields = [
        "startup_cost_fixed",
        "start_fuel_mmbtu",
        "min_down_time",
        "min_up_time",
        "ramp_limit_up",
        "ramp_limit_down",
        "vom",
    ]

    plants = impute_missing_plant_data(plants, ["technology_description", "build_decade"], data_fields)
    plants = impute_missing_plant_data(plants, ["prime_mover_code", "build_decade"], data_fields)
    plants = impute_missing_plant_data(plants, ["carrier", "build_decade"], data_fields)

    plants["start_fuel_cost"] = plants.start_fuel_mmbtu * plants.fuel_cost
    plants["start_up_cost"] = plants.startup_cost_fixed + plants.start_fuel_cost

    # replace heat-rate above theoretical minimum with nan
    plants.loc[plants.heat_rate < 3.412, "heat_rate"] = np.nan
    plants.loc[
        plants.fuel_type.isin(["solar", "wind", "hydro", "battery"]),
        "heat_rate",
    ] = 3.412

    plants = impute_missing_plant_data(
        plants,
        ["nerc_region", "prime_mover_code"],
        ["heat_rate"],
    )
    plants = impute_missing_plant_data(
        plants,
        ["nerc_region", "technology_description"],
        ["heat_rate"],
    )
    plants = impute_missing_plant_data(
        plants,
        ["nerc_region", "prime_mover_code"],
        ["heat_rate"],
    )
    plants = impute_missing_plant_data(plants, ["prime_mover_code"], ["heat_rate"])
    plants = impute_missing_plant_data(
        plants,
        ["technology_description"],
        ["heat_rate"],
    )
    plants = impute_missing_plant_data(plants, ["carrier"], ["heat_rate"])

    plants["marginal_cost"] = plants.vom + (plants.fuel_cost * plants.heat_rate)  # (MMBTu/MW) * (USD/MMBTu) = USD/MW
    plants["efficiency"] = 1 / (plants["heat_rate"] / 3.412)  # MMBTu/MWh to MWh_electric/MWh_thermal

    set_derates(plants)

    # Must run after set_derates: the p_min_pu <= p_max_pu feasibility invariant
    # is expressed against the seasonal derates.
    plants = sanitize_uc_parameters(plants)

    plants["heat_rate_source"] = plants["heat_rate_source"].fillna("NA")
    plants["fuel_cost_source"] = plants["fuel_cost_source"].fillna("NA")

    # Check for missing heat rate data
    if plants["heat_rate"].isna().sum() > 0:
        logger.warning(
            "Missing {} heat rate records.".format(plants["heat_rate"].isna().sum()),
        )

    # Check for missing fuel cost data
    if plants["fuel_cost"].isna().sum() > 0:
        logger.warning(
            "Missing {} fuel cost records.".format(plants["fuel_cost"].isna().sum()),
        )

    # Remove all column names that start with "ads_" except ads_mustrun
    plants = plants.loc[:, ~plants.columns.str.startswith("ads_") | (plants.columns == "ads_mustrun")]

    # Round all numeric columns to 4 decimal places
    plants = plants.round(4)

    return plants.reset_index()


def filter_outliers_iqr_grouped(df, group_column, value_column):
    """Filter outliers using IQR for each generator group."""

    def filter_outliers(group):
        q1 = group[value_column].quantile(0.25)
        q3 = group[value_column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return group[(group[value_column] >= lower_bound) & (group[value_column] <= upper_bound)]

    return df.groupby(group_column)[df.columns].apply(filter_outliers).reset_index(drop=True)


def filter_outliers_zscore(temporal_data, target_field_name):
    """Filter outliers using Z-score."""
    # Calculate mean and standard deviation for each generator
    stats = temporal_data.groupby(["generator_name"])[target_field_name].agg(["mean", "std"]).reset_index()
    stats["mean"] = stats["mean"].replace(np.inf, np.nan)
    stats = stats.dropna()

    # Merge mean and std back to the original dataframe
    temporal_stats = temporal_data.merge(
        stats,
        on=["generator_name"],
        how="left",
        suffixes=("", "_stats"),
    )

    # Calculate the Z-score for each month's entry
    temporal_stats["z_score"] = (temporal_stats[target_field_name] - temporal_stats["mean"]) / temporal_stats["std"]

    # Filter out the outliers using Z-score
    threshold = 3
    filtered_temporal = temporal_stats[np.abs(temporal_stats["z_score"]) <= threshold]
    filtered_temporal = filtered_temporal.drop(columns=["mean", "std", "z_score"])
    return filtered_temporal


def merge_fc_hr_data(
    plants: pd.DataFrame,
    temporal_data: pd.DataFrame,
    target_field_name: str,
):
    temporal_data["generator_name"] = (
        temporal_data["plant_name_eia"].astype(str)
        + "_"
        + temporal_data["plant_id_eia"].astype(str)
        + "_"
        + temporal_data["generator_id"].astype(str)
    )

    # Apply Z-score filtering to each generator
    filtered_temporal = filter_outliers_zscore(temporal_data, target_field_name)

    # Apply IQR filtering to each generator group
    filtered_temporal = filter_outliers_iqr_grouped(
        filtered_temporal,
        "technology_description",
        target_field_name,
    )

    # Apply temporal average heat rates to plants dataframe
    temporal_average = (
        filtered_temporal.groupby(["plant_id_eia", "generator_id"])[target_field_name].mean().reset_index()
    )

    if target_field_name in plants.columns:
        plants = plants.drop(columns=[target_field_name])

    temporal_average[f"{target_field_name}_source"] = "pudl_reciepts"

    plants = pd.merge(
        left=plants,
        right=temporal_average,
        on=["plant_id_eia", "generator_id"],
        how="left",
    )
    return plants


def apply_cems_heat_rates(plants, crosswalk_fn, cems_fn):
    # Apply CEMS calculated heat rates
    cems_hr = pd.read_excel(cems_fn)[["Facility ID", "Unit ID", "Heat Input (mmBtu/MWh)"]]
    crosswalk = pd.read_csv(crosswalk_fn)[["CAMD_PLANT_ID", "CAMD_UNIT_ID", "EIA_PLANT_ID", "EIA_GENERATOR_ID"]]
    cems_hr = pd.merge(
        cems_hr,
        crosswalk,
        left_on=["Facility ID", "Unit ID"],
        right_on=["CAMD_PLANT_ID", "CAMD_UNIT_ID"],
        how="inner",
    )
    cems_hr["hr_source_cems"] = "cems"
    plants = pd.merge(
        cems_hr,
        plants,
        left_on=["EIA_PLANT_ID", "EIA_GENERATOR_ID"],
        right_on=["plant_id_eia", "generator_id"],
        how="right",
    )

    plants = plants.rename(columns={"Heat Input (mmBtu/MWh)": "heat_rate_"})
    plants.heat_rate_ = plants.heat_rate_.fillna(
        plants.unit_heat_rate_mmbtu_per_mwh,
    )  # First take CEMS, then use PUDL
    plants.unit_heat_rate_mmbtu_per_mwh = plants.pop("heat_rate_")

    plants.hr_source_cems = plants.hr_source_cems.fillna(
        "unit_heat_rate_mmbtu_per_mwh_source",
    )
    plants.unit_heat_rate_mmbtu_per_mwh_source = plants.pop("hr_source_cems")

    plants = plants.drop(
        columns=[
            "Facility ID",
            "Unit ID",
            "CAMD_PLANT_ID",
            "CAMD_UNIT_ID",
            "EIA_PLANT_ID",
            "EIA_GENERATOR_ID",
        ],
    )

    return plants


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_powerplants")
        rootpath = ".."
    else:
        rootpath = "."
    configure_logging(snakemake)

    data_year = 2025  # latest complete EIA-923 year in the pinned PUDL release (v2026.8.0)
    start_date = f"{data_year}-01-01"
    end_date = f"{data_year + 1}-01-01"

    initialize_duckdb()
    eia_data_operable = load_eia_operable_data(snakemake.params.pudl_path)
    heat_rates = load_heat_rates_data(snakemake.params.pudl_path, start_date, end_date)

    eia_data_operable = merge_fc_hr_data(
        eia_data_operable,
        heat_rates,
        "unit_heat_rate_mmbtu_per_mwh",
    )
    eia_data_operable = merge_fc_hr_data(
        eia_data_operable,
        heat_rates,
        "fuel_cost_per_mwh",
    )
    eia_data_operable = merge_fc_hr_data(
        eia_data_operable,
        heat_rates,
        "fuel_cost_per_mmbtu",
    )
    eia_data_operable = apply_cems_heat_rates(
        eia_data_operable,
        snakemake.input.epa_crosswalk,
        snakemake.input.cems,
    )
    set_non_conus(eia_data_operable)
    set_tech_fuels_primer_movers(eia_data_operable)
    eia_ads_merged = merge_ads_data(eia_data_operable)
    plants = set_parameters(eia_ads_merged)

    # Throwing out plants without GPS data
    missing_locations = plants[plants.longitude.isna() | plants.latitude.isna()]
    logger.warning(
        f"Tossing out plants without locations: {missing_locations.shape[0]}",
    )
    # plants[plants.index.isin(missing_locations.index)].to_csv('missing_gps_pudl.csv')
    plants = plants[~plants.index.isin(missing_locations.index)]

    logger.info(f"Exporting Powerplants, with {plants.shape[0]} entries.")

    # Sort columns alphabetically for consistent diffing
    plants = plants.reindex(sorted(plants.columns), axis=1)
    # Sort rows by generator_name for consistent diffing
    plants = plants.sort_values("generator_name")

    plants.to_csv(snakemake.output.powerplants, index=False)
