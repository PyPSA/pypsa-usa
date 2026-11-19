# By PyPSA-USA Authors
"""
Investment and production tax credits per carrier and build year.

``costs: itc_modifier`` and ``costs: ptc_modifier`` give each carrier one value for
every build year, or a mapping ``{build year: value}`` so that each vintage gets the
credit of the year it is built in. ``null`` and a build year the mapping does not
list grant none.
"""

import logging

logger = logging.getLogger(__name__)


def credit_for_year(spec, year: int | None, key: str, carrier: str) -> float | None:
    """The value ``spec`` (a number, ``{year: number}`` or None) sets for build ``year``.

    None means not set.
    """
    if spec is None:
        return None
    if isinstance(spec, dict):
        if year is None:
            raise ValueError(f"costs.{key}.{carrier} is set per year, but the build year is unknown.")
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
