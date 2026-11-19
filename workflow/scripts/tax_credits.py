# By PyPSA-USA Authors
"""
Investment and production tax credits per carrier and build year.

By default a carrier gets the credits of the ATB model case ``load_costs`` selects
for it: a ``+ TC`` case annualizes capex at a fixed charge rate that holds the ITC
and carries the levelized PTC in ``ptc_per_mwh`` (both from ``build_cost_data``);
every other case has none.

``costs: itc_modifier`` and ``costs: ptc_modifier`` override that per carrier, with
one number for every build year or a mapping ``{build year: number}``. ``null``, an
unlisted carrier and an unlisted year keep the default.

* **ITC**: the share of the capital charge the credit pays back (effective, not
  statutory). The ITC a ``+ TC`` case holds, ``itc_effective``, is read off ATB's
  project finance factor against the case without tax credits
  (``build_cost_data.implied_itc``). The carrier is annualized at
  ``capital_charge_rate * (1 - itc) / (1 - itc_effective)``: the WACC of the
  selected case is kept and only its credit replaced. Both values are logged. The
  effective ITC holds the depreciation basis reduction and the tax gross-up of the
  credit, so it differs from the statutory rate.
* **PTC**: the first-year credit in USD/MWh of ``costs: dollar_year``, paid for
  :data:`PTC_YEARS` years and levelized over the cost recovery period at the
  carrier's WACC, the basis of the ATB-derived ``ptc_per_mwh``.

A project claims one of the two credits. A positive ITC or PTC set here is that
election and drops the ATB value of the other credit for that carrier and year,
unless both are set.
"""

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

#: Years a production tax credit is paid for (IRC 45 / 45Y).
PTC_YEARS = 10

#: Suffix ATB gives the model cases with tax credits.
TAX_CREDIT_CASE_SUFFIX = " + TC"

#: Cost parameters proportional to the capital charge rate.
ANNUALIZED_CAPEX = ["annualized_capex_per_mw", "annualized_capex_per_mw_km", "annualized_connection_capex_per_mw_km"]

_COST_FILE_YEAR = re.compile(r"costs_(\d{4})\.csv$")


def base_model_case(model_case: str) -> str:
    """The ATB model case without tax credits that ``model_case`` builds on."""
    return model_case.removesuffix(TAX_CREDIT_CASE_SUFFIX)


def year_from_cost_file(path) -> int | None:
    """Build year of a ``costs_{year}.csv`` written by ``build_cost_data``, else None."""
    match = _COST_FILE_YEAR.search(str(path))
    return int(match.group(1)) if match else None


def has_tax_credit_overrides(costs_config: dict | None) -> bool:
    """Whether ``costs: itc_modifier`` or ``ptc_modifier`` sets anything."""
    costs_config = costs_config or {}
    return bool(costs_config.get("itc_modifier") or costs_config.get("ptc_modifier"))


def credit_for_year(spec, year: int | None, key: str, carrier: str) -> float | None:
    """The value ``spec`` (a number, ``{year: number}`` or None) sets for ``year``.

    None means not set, so the ATB default applies.
    """
    if spec is None:
        return None
    if isinstance(spec, dict):
        if year is None:
            raise ValueError(f"costs.{key}.{carrier} is set per year, but the build year of the cost table is unknown.")
        by_year = {int(y): v for y, v in spec.items()}
        spec = by_year.get(int(year))
        if spec is None:
            return None
    value = float(spec)
    if key == "itc_modifier" and not 0 <= value < 1:
        raise ValueError(f"costs.{key}.{carrier} must lie in [0, 1); received {value}.")
    if key == "ptc_modifier" and value < 0:
        raise ValueError(f"costs.{key}.{carrier} must not be negative; received {value}.")
    return value


def levelize_ptc(ptc: float, wacc: float, cost_recovery_period: float, years: int = PTC_YEARS) -> float:
    """First-year PTC paid for ``years`` years, levelized over the cost recovery period."""
    from _helpers import calculate_annuity

    if not cost_recovery_period > 0:
        return ptc
    return ptc * calculate_annuity(cost_recovery_period, wacc) / calculate_annuity(years, wacc)


def _reannualize(costs: pd.DataFrame, carrier: str, factor: float) -> None:
    """Scale the annualized capex of ``carrier`` and its capital charge rate by ``factor``; FOM is left untouched.

    A carrier whose annualized cost is given directly (no ``capital_charge_rate``) keeps none.
    """
    if factor == 1.0:
        return
    old = costs.at[carrier, "annualized_capex_per_mw"] if "annualized_capex_per_mw" in costs.columns else 0.0
    for col in ANNUALIZED_CAPEX:
        if col in costs.columns:
            costs.at[carrier, col] *= factor
    if "annualized_capex_fom" in costs.columns:
        costs.at[carrier, "annualized_capex_fom"] += old * (factor - 1.0)
    if "capital_charge_rate" in costs.columns and costs.at[carrier, "capital_charge_rate"] > 0:
        costs.at[carrier, "capital_charge_rate"] *= factor


def resolve_tax_credits(costs: pd.DataFrame, costs_config: dict | None, year: int | None) -> pd.DataFrame:
    """Apply ``costs: itc_modifier`` / ``ptc_modifier`` to the cost table of the selected cases.

    ``year`` is the build year of the table. An explicit ITC replaces the ATB one,
    ``itc_effective`` (``build_cost_data.implied_itc``): the rate without it is
    ``capital_charge_rate / (1 - itc_effective)``, which keeps the WACC of the case.
    """
    costs_config = costs_config or {}
    itc_cfg = costs_config.get("itc_modifier") or {}
    ptc_cfg = costs_config.get("ptc_modifier") or {}
    carriers = set(itc_cfg) | set(ptc_cfg)
    if not carriers:
        return costs

    unknown = sorted(carriers - set(costs.index))
    if unknown:
        logger.warning(f"Tax credits set for {unknown}, which the cost table does not carry; they are ignored.")
    if itc_cfg and "itc_effective" not in costs.columns:
        logger.warning(
            "The cost table has no itc_effective (built before it was added): an ITC set in costs.itc_modifier "
            "comes on top of any ATB ITC. Rebuild the cost tables (build_cost_data).",
        )

    costs = costs.copy()
    for col in ("ptc_per_mwh", "itc_effective"):
        if col not in costs.columns:
            costs[col] = 0.0
    for carrier in sorted(carriers & set(costs.index)):
        itc = credit_for_year(itc_cfg.get(carrier), year, "itc_modifier", carrier)
        ptc = credit_for_year(ptc_cfg.get(carrier), year, "ptc_modifier", carrier)
        if itc is None and ptc is None:
            continue
        if itc is None and ptc > 0:
            itc = 0.0  # the PTC is claimed: no ITC from the ATB case

        atb_itc = float(costs.at[carrier, "itc_effective"])
        if itc is not None:
            _reannualize(costs, carrier, (1 - itc) / (1 - atb_itc))
            costs.at[carrier, "itc_effective"] = itc

        atb_ptc = float(costs.at[carrier, "ptc_per_mwh"])
        if ptc is not None:
            costs.at[carrier, "ptc_per_mwh"] = levelize_ptc(
                ptc,
                float(costs.at[carrier, "wacc_real"]) if "wacc_real" in costs.columns else 0.0,
                float(costs.at[carrier, "cost_recovery_period_years"])
                if "cost_recovery_period_years" in costs.columns
                else 0.0,
            )
        elif itc > 0:
            costs.at[carrier, "ptc_per_mwh"] = 0.0  # the ITC is claimed

        logger.info(
            f"{carrier} {year}: effective ITC {'ATB' if itc is None else f'{itc:.3f}'} (ATB case: {atb_itc:.3f}), "
            f"levelized PTC {costs.at[carrier, 'ptc_per_mwh']:.2f} USD/MWh (ATB case: {atb_ptc:.2f}).",
        )
    return costs


def check_tax_credit_years(costs_config: dict | None, investment_periods) -> None:
    """Warn about per-year tax credits set for years that are no investment period."""
    periods = {int(p) for p in investment_periods}
    for key in ("itc_modifier", "ptc_modifier"):
        for carrier, spec in ((costs_config or {}).get(key) or {}).items():
            if isinstance(spec, dict):
                unused = sorted({int(y) for y in spec} - periods)
                if unused:
                    logger.warning(
                        f"costs.{key}.{carrier} sets {unused}, which are no investment period {sorted(periods)}.",
                    )
