"""
Dollar-year conversions for cost inputs.

Every cost in the model is in real US dollars of one base year, ``costs:
dollar_year`` in the config. Inputs arrive in other dollar years (the 2025 ATB
in 2023$, AEO2026 in 2025$, EIA equipment costs in 2022$, ...) and sometimes
in other currencies. This module is the one place they are brought to the base
year.

Convention
----------
- A value in a foreign currency is converted to US dollars at the annual
  average exchange rate of its own dollar year, then inflated in US dollars.
- Inflation uses the CPI-U annual average (FRED CPIAUCNS, all items, not
  seasonally adjusted: the published BLS annual averages), the single price
  index of the model. EIA builds its real-dollar AEO series on the GDP price
  index, so AEO values converted here differ from EIA's own conversion by a
  fraction of a percent.
- A nominal time series is deflated by the year of each observation, never by
  the investment period it is used in: the model is in real dollars of the
  base year in every period.

Data
----
Both series come from packages pinned in the environment, so no table in the
repository has to be kept current:

- ``cpi`` (https://github.com/palewire/cpi) ships the BLS CPI-U series
  CUUR0000SA0, whose annual values are the published BLS annual averages
  (1913 onward). The package data is as of its release; ``cpi.update()`` is
  never called, so a run depends only on the pinned package version.
- ``currencyconverter`` (https://github.com/alexprengere/currencyconverter)
  ships the ECB daily reference rates (1999 onward, EUR base). The annual
  average exchange rate of a currency is the mean of its daily USD cross rates
  over the calendar year, which agrees with the FRED annual averages to about
  0.05 %.
"""

from __future__ import annotations

import datetime
import logging
import re
import warnings
from functools import cache, lru_cache

import pandas as pd

with warnings.catch_warnings():
    # the package warns at import whenever its bundled data is more than 90
    # days old; the data is as of the pinned release and is never updated
    warnings.simplefilter("ignore")
    _cpi_logger = logging.getLogger("cpi")
    _cpi_level = _cpi_logger.level
    _cpi_logger.setLevel(logging.ERROR)
    import cpi

    _cpi_logger.setLevel(_cpi_level)

from currency_converter import CurrencyConverter, RateNotFoundError

# dollar year of each NREL ATB report year. PUDL's ATB tables carry no dollar
# year, so it is pinned here; see https://atb.nrel.gov/electricity/<year>/changes_in_<year>
ATB_DOLLAR_YEAR = {2023: 2021, 2024: 2022, 2025: 2023}

# a 4-digit year next to a dollar marker: "2025 $/MMBtu", "2016$/kBtu/hr",
# "USD2004perMW", "(2020 US Dollars per Installed Kilowatt)"
_UNIT_YEAR = re.compile(
    r"(?P<pre>\$|USD|US\s*Dollars?)\s*(?P<year1>(?:19|20)\d{2})|(?P<year2>(?:19|20)\d{2})\s*(?P<post>\$|USD|US\s*Dollars?)",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def cpi_index() -> pd.Series:
    """CPI-U annual averages by year (BLS series CUUR0000SA0, from the ``cpi`` package)."""
    df = cpi.series.get().to_dataframe()
    annual = df[df["period_type"] == "annual"]
    return annual.set_index(annual["year"].astype(int))["value"].astype(float).sort_index().rename("index")


@lru_cache(maxsize=1)
def _converter() -> CurrencyConverter:
    return CurrencyConverter()


@cache
def fx_to_usd(currency: str, year: int) -> float:
    """
    US dollars per unit of ``currency`` at the annual-average rate of ``year``.

    The mean of the ECB daily reference rates of the calendar year, expressed
    in US dollars per unit of ``currency``. Raises when the ECB publishes no
    rate for the currency in that year.
    """
    currency = currency.upper()
    if currency == "USD":
        return 1.0
    converter = _converter()
    if currency not in converter.currencies:
        raise ValueError(
            f"No exchange rate for {currency}; the ECB reference rates cover {sorted(converter.currencies)}"
        )
    rates = []
    day = datetime.date(year, 1, 1)
    while day.year == year:
        try:
            rates.append(converter.convert(1.0, currency, "USD", date=day))
        except RateNotFoundError:
            pass
        day += datetime.timedelta(days=1)
    if not rates:
        first, last = converter.bounds[currency]
        raise ValueError(
            f"No {currency} exchange rate for {year}; the ECB reference rates cover {first}-{last} for {currency}"
        )
    return float(sum(rates) / len(rates))


def deflator(from_year: int, to_year: int) -> float:
    """Factor that takes a value in US dollars of ``from_year`` to dollars of ``to_year``."""
    index = cpi_index()
    for year in (from_year, to_year):
        if year not in index.index:
            raise ValueError(
                f"No CPI-U value for {year}; the annual averages cover {index.index.min()}-{index.index.max()}"
            )
    return float(index.at[to_year] / index.at[from_year])


def conversion_factor(year: int, to_year: int, currency: str = "USD") -> float:
    """Factor that takes a value in ``currency`` of ``year`` to US dollars of ``to_year``."""
    return fx_to_usd(currency, year) * deflator(year, to_year)


def to_dollars(value, *, year: int, to_year: int, currency: str = "USD"):
    """
    Convert ``value`` (scalar, array, Series or DataFrame) from ``currency`` of ``year`` to US dollars of ``to_year``.

    Foreign currency is exchanged at the annual-average rate of ``year``, then
    inflated in US dollars.
    """
    return value * conversion_factor(year, to_year, currency)


def convert_frame(
    df: pd.DataFrame,
    to_year: int,
    *,
    value_col: str = "value",
    year_col: str = "dollar_year",
    currency_col: str = "currency",
) -> pd.DataFrame:
    """
    Convert every row of a cost table to US dollars of ``to_year``.

    Each row carries its own dollar year in ``year_col`` and, optionally, its
    currency in ``currency_col`` (missing column or NaN means USD). Rows without
    a dollar year raise. The returned copy has ``value_col`` converted and
    ``year_col`` / ``currency_col`` set to ``to_year`` / ``USD``.
    """
    if year_col not in df.columns:
        raise ValueError(f"Cost table has no '{year_col}' column; every cost row needs a dollar year")
    years = pd.to_numeric(df[year_col], errors="coerce")
    if years.isna().any():
        missing = df.index[years.isna()].tolist()
        raise ValueError(f"{len(missing)} cost rows have no dollar year: {missing[:10]}")
    currencies = df[currency_col].fillna("USD") if currency_col in df.columns else pd.Series("USD", index=df.index)
    currencies = currencies.astype(str).str.upper()

    factors = pd.Series(1.0, index=df.index)
    for (year, currency), idx in df.groupby([years.astype(int), currencies]).groups.items():
        factors.loc[idx] = conversion_factor(int(year), to_year, currency)

    out = df.copy()
    out[value_col] = out[value_col] * factors
    out[year_col] = to_year
    out[currency_col] = "USD"
    return out


def deflate_timeseries(data: pd.Series | pd.DataFrame, to_year: int) -> pd.Series | pd.DataFrame:
    """
    Deflate a nominal time series to US dollars of ``to_year``.

    ``data`` is indexed by a ``DatetimeIndex``; each observation is deflated by
    the CPI-U of its own calendar year.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("deflate_timeseries needs a DatetimeIndex, so each observation has a year")
    index = cpi_index()
    years = pd.Index(data.index.year)
    missing = sorted(set(years) - set(index.index))
    if missing:
        raise ValueError(
            f"No CPI-U value for {missing}; the annual averages cover {index.index.min()}-{index.index.max()}"
        )
    factors = pd.Series((index.at[to_year] / index.reindex(years)).to_numpy(), index=data.index)
    return data.mul(factors, axis=0) if isinstance(data, pd.DataFrame) else data * factors


def parse_unit_year(unit: str) -> tuple[int, str]:
    """
    Read the dollar year out of a unit string, such as ``"2025 $/MMBtu"``.

    Returns the year and the unit without it (``"$/MMBtu"``). Raises if the
    string names no dollar year, so a source whose dollar year is unknown is
    never silently treated as the base year.
    """
    match = _UNIT_YEAR.search(unit)
    if not match:
        raise ValueError(f"No dollar year in unit '{unit}'")
    year = int(match.group("year1") or match.group("year2"))
    start, end = match.span("year1") if match.group("year1") else match.span("year2")
    remainder = re.sub(r"\(\s+", "(", re.sub(r"\s+", " ", unit[:start] + unit[end:])).strip()
    return year, remainder


def atb_dollar_year(report_year: int) -> int:
    """Dollar year of the NREL ATB of ``report_year``."""
    if report_year not in ATB_DOLLAR_YEAR:
        raise ValueError(f"Dollar year of the {report_year} ATB is not pinned in money.ATB_DOLLAR_YEAR")
    return ATB_DOLLAR_YEAR[report_year]


def dollar_year_from_config(config: dict) -> int:
    """The model's base dollar year, ``costs: dollar_year``, checked against the CPI-U table."""
    try:
        year = int(config["costs"]["dollar_year"])
    except (KeyError, TypeError) as e:
        raise KeyError("config needs costs: dollar_year, the base dollar year of every cost in the model") from e
    if year not in cpi_index().index:
        raise ValueError(f"costs: dollar_year {year} has no CPI-U annual average yet")
    return year
