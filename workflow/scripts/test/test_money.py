"""
Unit tests for the dollar-year conversions in ``money``.

The CPI-U and exchange-rate series come from the pinned ``cpi`` and
``currencyconverter`` packages and are checked against published BLS and FRED
values, so a package whose data disagrees with the official series fails here.
The base dollar year of the default config must be covered.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import money

pytestmark = pytest.mark.fast

WORKFLOW = Path(__file__).resolve().parents[2]

# BLS CPI-U annual averages, all items, U.S. city average, 1982-84 = 100
BLS_CPI_U = {2004: 188.9, 2015: 237.017, 2019: 255.657, 2020: 258.811, 2022: 292.655, 2023: 304.702, 2024: 313.689}
# FRED annual-average exchange rates, US dollars per unit. The ECB daily
# reference rates average to within 0.05 % of these.
FRED_USD_PER_UNIT = {("EUR", 2019): 1.1194, ("EUR", 2023): 1.0817, ("EUR", 2025): 1.1306, ("GBP", 2023): 1.2440}
FX_TOL = 1e-3


# the series


@pytest.mark.parametrize(("year", "index"), BLS_CPI_U.items())
def test_cpi_series_matches_bls_annual_averages(year, index):
    assert money.cpi_index().at[year] == pytest.approx(index, rel=1e-4)  # BLS rounds older years to one decimal


def test_cpi_series_is_annual_and_covers_recent_years():
    index = money.cpi_index()

    assert index.index.is_unique
    assert index.index.min() <= 1990
    assert index.index.max() >= 2024
    assert (index.diff().dropna().abs() < 30).all()  # annual steps, no monthly values mixed in


@pytest.mark.parametrize(("key", "rate"), FRED_USD_PER_UNIT.items())
def test_exchange_rates_match_fred_annual_averages(key, rate):
    assert money.fx_to_usd(*key) == pytest.approx(rate, abs=FX_TOL)


def test_cad_rate_is_usd_per_cad():
    # FRED AEXCAUS is CAD per USD (1.3494 in 2023)
    assert money.fx_to_usd("CAD", 2023) == pytest.approx(1 / 1.3494, abs=FX_TOL)


def test_importing_money_is_quiet():
    # the cpi package warns and logs at import when its bundled data is older
    # than 90 days; money silences that, as the data is pinned by version
    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", "import logging; logging.basicConfig(); import money"],
        cwd=WORKFLOW / "scripts",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "out of date" not in result.stderr


def test_default_config_dollar_year_is_covered():
    config = yaml.safe_load((WORKFLOW / "repo_data/config/config.default.yaml").read_text())

    assert money.dollar_year_from_config(config) in money.cpi_index().index


def test_schema_requires_dollar_year():
    schema = yaml.safe_load((WORKFLOW / "schemas/config.schema.yaml").read_text())

    assert "dollar_year" in schema["properties"]["costs"]["required"]


# deflator and exchange rates


def test_deflator_is_the_cpi_ratio():
    assert money.deflator(2022, 2023) == pytest.approx(304.702 / 292.655)
    assert money.deflator(2025, 2023) == pytest.approx(1 / 1.0566, abs=1e-4)
    assert money.deflator(2023, 2023) == 1.0
    assert money.deflator(2004, 2023) * money.deflator(2023, 2004) == pytest.approx(1.0)


def test_deflator_rejects_uncovered_years():
    with pytest.raises(ValueError, match="No CPI-U value for 1900"):
        money.deflator(1900, 2023)
    with pytest.raises(ValueError, match="2040"):
        money.deflator(2023, 2040)


def test_usd_needs_no_exchange_rate():
    assert money.fx_to_usd("USD", 1850) == 1.0
    assert money.fx_to_usd("usd", 2023) == 1.0


def test_unknown_currency_or_year_is_rejected():
    with pytest.raises(ValueError, match="No exchange rate for XXX"):
        money.fx_to_usd("XXX", 2023)
    with pytest.raises(ValueError, match="EUR exchange rate for 1990"):
        money.fx_to_usd("EUR", 1990)  # ECB reference rates start in 1999


# conversions


def test_to_dollars_exchanges_in_the_source_year_then_inflates():
    expected = 100 * 1.1194 * 304.702 / 255.657  # 2019 EUR -> 2019 USD -> 2023 USD

    assert money.to_dollars(100, year=2019, to_year=2023, currency="EUR") == pytest.approx(expected, rel=1e-3)
    assert money.to_dollars(100, year=2023, to_year=2023) == 100


def test_to_dollars_works_on_series_and_arrays():
    factor = money.deflator(2022, 2023)
    s = pd.Series([1.0, 2.0], index=["a", "b"])

    pd.testing.assert_series_equal(money.to_dollars(s, year=2022, to_year=2023), s * factor)
    np.testing.assert_allclose(money.to_dollars(np.array([1.0, 2.0]), year=2022, to_year=2023), [factor, 2 * factor])


def test_convert_frame_converts_each_row_by_its_own_year_and_currency():
    df = pd.DataFrame(
        {
            "technology": ["a", "b", "c", "d"],
            "value": [1.0, 1.0, 1.0, 1.0],
            "dollar_year": [2023, 2022, 2019, 2015],
            "currency": ["USD", None, "EUR", "eur"],
        },
    )

    out = money.convert_frame(df, 2023)

    assert out.value.tolist() == pytest.approx(
        [1.0, money.deflator(2022, 2023), 1.1194 * money.deflator(2019, 2023), 1.1096 * money.deflator(2015, 2023)],
        rel=1e-3,
    )
    assert (out.dollar_year == 2023).all()
    assert (out.currency == "USD").all()
    # the input is untouched
    assert df.value.tolist() == [1.0] * 4


def test_convert_frame_without_currency_column_assumes_usd():
    df = pd.DataFrame({"value": [2.0], "dollar_year": [2022]})

    out = money.convert_frame(df, 2023)

    assert out.value.iat[0] == pytest.approx(2 * money.deflator(2022, 2023))
    assert out.currency.iat[0] == "USD"


def test_convert_frame_rejects_rows_without_a_dollar_year():
    with pytest.raises(ValueError, match="no 'dollar_year' column"):
        money.convert_frame(pd.DataFrame({"value": [1.0]}), 2023)

    df = pd.DataFrame({"value": [1.0, 1.0], "dollar_year": [2023, None]}, index=["ok", "undated"])
    with pytest.raises(ValueError, match="1 cost rows have no dollar year: \\['undated'\\]"):
        money.convert_frame(df, 2023)


def test_deflate_timeseries_uses_the_year_of_each_observation():
    idx = pd.date_range("2019-11-01", "2020-02-01", freq="MS")
    s = pd.Series(1.0, index=idx)

    out = money.deflate_timeseries(s, 2023)

    assert out.loc["2019"].tolist() == pytest.approx([money.deflator(2019, 2023)] * 2)
    assert out.loc["2020"].tolist() == pytest.approx([money.deflator(2020, 2023)] * 2)

    df = pd.DataFrame({"x": 1.0, "y": 2.0}, index=idx)
    out = money.deflate_timeseries(df, 2023)
    assert out.loc["2020-01-01", "y"] == pytest.approx(2 * money.deflator(2020, 2023))


def test_deflate_timeseries_needs_dates():
    with pytest.raises(TypeError, match="DatetimeIndex"):
        money.deflate_timeseries(pd.Series([1.0], index=[2019]), 2023)


# unit strings


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("2025 $/MMBtu", (2025, "$/MMBtu")),
        ("2016$/kBtu/hr", (2016, "$/kBtu/hr")),
        ("USD2004perMW", (2004, "USDperMW")),
        (
            "System Cost (2020 US Dollars per Installed Kilowatt)",
            (2020, "System Cost (US Dollars per Installed Kilowatt)"),
        ),
        ("Capital Cost (2016$)", (2016, "Capital Cost ($)")),
    ],
)
def test_parse_unit_year(unit, expected):
    assert money.parse_unit_year(unit) == expected


@pytest.mark.parametrize("unit", ["$/MMBtu", "USD/MW/year", "per unit", "2030"])
def test_parse_unit_year_rejects_units_without_a_dollar_year(unit):
    with pytest.raises(ValueError, match="No dollar year"):
        money.parse_unit_year(unit)


# ATB and config


def test_atb_dollar_year_is_pinned():
    assert money.atb_dollar_year(2025) == 2023
    assert money.atb_dollar_year(2024) == 2022
    with pytest.raises(ValueError, match="2026 ATB"):
        money.atb_dollar_year(2026)


def test_dollar_year_from_config_requires_the_key_and_coverage():
    assert money.dollar_year_from_config({"costs": {"dollar_year": 2023}}) == 2023
    with pytest.raises(KeyError, match="costs: dollar_year"):
        money.dollar_year_from_config({"costs": {}})
    with pytest.raises(ValueError, match="2040"):
        money.dollar_year_from_config({"costs": {"dollar_year": 2040}})
