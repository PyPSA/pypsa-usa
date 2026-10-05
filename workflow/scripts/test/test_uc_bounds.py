"""Unit tests for the unit-commitment parameter bounds.

`conventional.unit_commitment: true` (the California configuration) hands the UC
columns of `powerplants.csv` straight to PyPSA's committable formulation. Those
columns come from the WECC ADS merge and from group-mean imputation, and both
paths emit values that are physically impossible for the unit they land on —
absolute ($, not $/MW) start-up costs on sub-MW units, baseload minimum up times
on aircraft-derivative peakers, `minimum_load_mw` reported at plant level but
joined to a single sub-unit (so `p_min_pu > 1`), and NaNs that would otherwise
be filled with the PyPSA defaults of "no constraint, free start".

These tests pin:
  * out-of-band values are clamped into the carrier band,
  * in-band values are left untouched (the clamp is a filter, not an overwrite),
  * NaNs are filled with the conservative carrier default, never with zero,
  * the feasibility invariant `p_min_pu <= min(summer_derate, winter_derate)`
    holds for every committable row, which is what keeps the MILP from going
    infeasible in the tighter season,
  * `start_up_cost` stays equal to `startup_cost_fixed + start_fuel_cost`,
  * non-committable carriers are not touched,
  * `prepare_network` converts the hourly UC attributes into snapshot units when
    the `{opts}` string reduces the temporal resolution, and is a no-op when
    there are no committable generators.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from build_powerplants import UC_BOUNDS, UC_CARRIERS, sanitize_uc_parameters
from prepare_network import _hours_per_snapshot, rescale_uc_attrs_to_resolution

pytestmark = pytest.mark.fast


UC_COLUMNS = [
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
]


def make_plant(carrier, p_nom=100.0, **overrides):
    """A single in-band plant row; overrides push individual columns out of band."""
    bounds = UC_BOUNDS.get(carrier)
    if bounds is None:  # non-committable carrier: no band, use benign values
        row = dict(
            carrier=carrier,
            p_nom=p_nom,
            min_up_time=np.nan,
            min_down_time=np.nan,
            ramp_limit_up=np.nan,
            ramp_limit_down=np.nan,
            start_up_cost=np.nan,
            startup_cost_fixed=np.nan,
            start_fuel_cost=np.nan,
            start_fuel_mmbtu=np.nan,
            minimum_load_mw=np.nan,
            summer_derate=1.0,
            winter_derate=1.0,
        )
        row.update(overrides)
        return row

    up_default = bounds["up_h"][2]
    down_default = bounds["down_h"][2]
    ramp_default = bounds["ramp"][2]
    start_default = bounds["start_usd_mw"][2]
    p_min_default = bounds["p_min_pu"][1]
    fixed = 0.6 * start_default * p_nom
    fuel = 0.4 * start_default * p_nom
    row = dict(
        carrier=carrier,
        p_nom=p_nom,
        min_up_time=up_default,
        min_down_time=down_default,
        ramp_limit_up=ramp_default,
        ramp_limit_down=ramp_default,
        start_up_cost=fixed + fuel,
        startup_cost_fixed=fixed,
        start_fuel_cost=fuel,
        start_fuel_mmbtu=fuel / 4.0,
        minimum_load_mw=p_min_default * p_nom,
        summer_derate=1.0,
        winter_derate=1.0,
    )
    row.update(overrides)
    return row


def frame(rows):
    return pd.DataFrame(rows, columns=UC_COLUMNS)


# ---------------------------------------------------------------------------
# clamping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("carrier", UC_CARRIERS)
def test_in_band_values_are_untouched(carrier):
    """A plausible unit must survive sanitization byte-for-byte."""
    before = frame([make_plant(carrier)])
    after = sanitize_uc_parameters(before.copy())
    pd.testing.assert_frame_equal(
        after[UC_COLUMNS].astype({c: float for c in UC_COLUMNS if c != "carrier"}),
        before[UC_COLUMNS].astype({c: float for c in UC_COLUMNS if c != "carrier"}),
        check_dtype=False,
    )


def test_absurd_min_up_and_down_times_are_clamped():
    """A 108 h minimum up time on a peaker is imputation leakage, not data."""
    hi_up = UC_BOUNDS["OCGT"]["up_h"][1]
    hi_down = UC_BOUNDS["OCGT"]["down_h"][1]
    df = sanitize_uc_parameters(
        frame([make_plant("OCGT", min_up_time=108.0, min_down_time=48.0)]),
    )
    assert df.min_up_time.iloc[0] == hi_up
    assert df.min_down_time.iloc[0] == hi_down


def test_negative_and_zero_times_are_raised_to_the_floor():
    lo_up = UC_BOUNDS["coal"]["up_h"][0]
    df = sanitize_uc_parameters(frame([make_plant("coal", min_up_time=-5.0, min_down_time=0.0)]))
    assert df.min_up_time.iloc[0] == lo_up
    assert df.min_down_time.iloc[0] == UC_BOUNDS["coal"]["down_h"][0]


def test_ramp_limits_are_bounded_to_a_positive_fraction():
    """Ramp must stay in (0, 1]: 0 freezes the unit, >1 is faster than hourly."""
    df = sanitize_uc_parameters(
        frame(
            [
                make_plant("CCGT", ramp_limit_up=0.0, ramp_limit_down=5.0),
                make_plant("biomass", ramp_limit_up=-0.2, ramp_limit_down=1.0),
            ],
        ),
    )
    assert df.ramp_limit_up.iloc[0] == UC_BOUNDS["CCGT"]["ramp"][0]
    assert df.ramp_limit_down.iloc[0] == 1.0
    assert df.ramp_limit_up.iloc[1] == UC_BOUNDS["biomass"]["ramp"][0]
    assert (df.ramp_limit_up > 0).all() and (df.ramp_limit_up <= 1).all()


def test_start_up_cost_is_bounded_per_mw_not_per_unit():
    """The 0.1 MW biomass unit that inherited a $71.5k fixed cost is the real case."""
    lo, hi, _ = UC_BOUNDS["biomass"]["start_usd_mw"]
    p_nom = 0.1
    df = sanitize_uc_parameters(
        frame(
            [
                make_plant(
                    "biomass",
                    p_nom=p_nom,
                    start_up_cost=103_348.0,
                    startup_cost_fixed=71_549.0,
                    start_fuel_cost=31_799.0,
                ),
                make_plant("biomass", p_nom=50.0, start_up_cost=0.0, startup_cost_fixed=0.0, start_fuel_cost=0.0),
            ],
        ),
    )
    assert df.start_up_cost.iloc[0] == pytest.approx(hi * p_nom)
    assert df.start_up_cost.iloc[1] == pytest.approx(lo * 50.0)


def test_geothermal_zero_start_cost_gets_a_floor():
    """ADS reports 0 for every geothermal unit; free cycling is not a real option."""
    lo = UC_BOUNDS["geothermal"]["start_usd_mw"][0]
    df = sanitize_uc_parameters(
        frame(
            [
                make_plant(
                    "geothermal",
                    p_nom=20.0,
                    start_up_cost=0.0,
                    startup_cost_fixed=0.0,
                    start_fuel_cost=0.0,
                    start_fuel_mmbtu=0.0,
                ),
            ],
        ),
    )
    assert df.start_up_cost.iloc[0] == pytest.approx(lo * 20.0)


def test_start_up_cost_components_stay_consistent():
    """start_up_cost must remain the sum of its two reported components."""
    df = sanitize_uc_parameters(
        frame(
            [
                make_plant("coal", p_nom=500.0),
                make_plant(
                    "OCGT",
                    p_nom=0.3,
                    start_up_cost=110_562.0,
                    startup_cost_fixed=71_549.0,
                    start_fuel_cost=39_013.0,
                ),
                make_plant(
                    "CCGT",
                    p_nom=200.0,
                    start_up_cost=np.nan,
                    startup_cost_fixed=np.nan,
                    start_fuel_cost=np.nan,
                ),
            ],
        ),
    )
    assert np.allclose(df.startup_cost_fixed + df.start_fuel_cost, df.start_up_cost)
    assert (df.start_up_cost >= 0).all()
    assert (df.start_fuel_mmbtu >= 0).all()


def test_start_up_cost_is_never_negative():
    df = sanitize_uc_parameters(
        frame([make_plant("oil", p_nom=10.0, start_up_cost=-500.0, startup_cost_fixed=-500.0, start_fuel_cost=0.0)]),
    )
    assert df.start_up_cost.iloc[0] == pytest.approx(UC_BOUNDS["oil"]["start_usd_mw"][0] * 10.0)


# ---------------------------------------------------------------------------
# NaN fills
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("carrier", UC_CARRIERS)
def test_nans_are_filled_with_carrier_defaults_not_zero(carrier):
    """PyPSA's own defaults (min_up=0, min_down=0, start cost=0) make a unit free
    to cycle; the fill must be the conservative carrier value instead.
    """
    bounds = UC_BOUNDS[carrier]
    row = make_plant(
        carrier,
        p_nom=100.0,
        min_up_time=np.nan,
        min_down_time=np.nan,
        ramp_limit_up=np.nan,
        ramp_limit_down=np.nan,
        start_up_cost=np.nan,
        startup_cost_fixed=np.nan,
        start_fuel_cost=np.nan,
        start_fuel_mmbtu=np.nan,
        minimum_load_mw=np.nan,
    )
    df = sanitize_uc_parameters(frame([row]))
    assert df.min_up_time.iloc[0] == bounds["up_h"][2]
    assert df.min_down_time.iloc[0] == bounds["down_h"][2]
    assert df.ramp_limit_up.iloc[0] == bounds["ramp"][2]
    assert df.ramp_limit_down.iloc[0] == bounds["ramp"][2]
    assert df.start_up_cost.iloc[0] == pytest.approx(bounds["start_usd_mw"][2] * 100.0)
    assert df.minimum_load_mw.iloc[0] == pytest.approx(bounds["p_min_pu"][1] * 100.0)
    assert not df[[c for c in UC_COLUMNS if c != "carrier"]].isna().to_numpy().any()


def test_nan_minimum_load_fill_respects_a_tight_derate():
    """A unit derated to 0.2 must not be filled with a 0.4 default minimum load."""
    df = sanitize_uc_parameters(
        frame([make_plant("coal", p_nom=100.0, minimum_load_mw=np.nan, summer_derate=0.2, winter_derate=0.9)]),
    )
    assert df.minimum_load_mw.iloc[0] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# the feasibility invariant
# ---------------------------------------------------------------------------


def test_minimum_load_above_nameplate_is_clamped():
    """`minimum_load_mw` is reported at plant level and joined per sub-unit, so
    p_min_pu > 1 appears in the raw data (e.g. YKK USA Chestney at 31x).
    """
    df = sanitize_uc_parameters(frame([make_plant("oil", p_nom=1.6, minimum_load_mw=50.0)]))
    assert df.minimum_load_mw.iloc[0] / 1.6 <= UC_BOUNDS["oil"]["p_min_pu"][0]


def test_minimum_load_never_exceeds_the_tighter_seasonal_derate():
    """The feasibility invariant: p_max_pu of a conventional unit is the seasonal
    derate, so p_min_pu above it empties the dispatch box in that season.
    """
    rows = []
    for carrier in UC_CARRIERS:
        rows.append(make_plant(carrier, p_nom=100.0, minimum_load_mw=95.0, summer_derate=0.35, winter_derate=0.8))
        rows.append(make_plant(carrier, p_nom=100.0, minimum_load_mw=np.nan, summer_derate=0.9, winter_derate=0.12))
    df = sanitize_uc_parameters(frame(rows))
    p_min_pu = df.minimum_load_mw / df.p_nom
    ceiling = np.minimum(df.summer_derate, df.winter_derate)
    assert (p_min_pu <= ceiling + 1e-9).all()
    assert (p_min_pu >= 0).all()


def test_carrier_p_min_pu_cap_is_also_respected():
    """Even with no derate, a peaker must not be pinned above its carrier cap."""
    df = sanitize_uc_parameters(frame([make_plant("OCGT", p_nom=100.0, minimum_load_mw=99.0)]))
    assert df.minimum_load_mw.iloc[0] / 100.0 == pytest.approx(UC_BOUNDS["OCGT"]["p_min_pu"][0])


def test_invariant_survives_the_four_decimal_rounding_of_set_parameters():
    """`set_parameters` rounds every numeric column to 4 decimals before writing.

    On sub-MW units that step is large in relative terms, so a minimum load that
    rounds *up* would put p_min_pu back above the derate. The clamp truncates
    instead, which the round() then cannot move.
    """
    rows = [
        make_plant("oil", p_nom=0.1, minimum_load_mw=5.0, summer_derate=d, winter_derate=1.0)
        for d in (0.6667, 0.3333, 0.1234, 0.9999)
    ]
    rows += [make_plant("OCGT", p_nom=0.3, minimum_load_mw=np.nan, summer_derate=0.6667, winter_derate=1.0)]
    df = sanitize_uc_parameters(frame(rows)).round(4)
    p_min_pu = df.minimum_load_mw / df.p_nom
    ceiling = np.minimum(df.summer_derate, df.winter_derate)
    assert (p_min_pu <= ceiling).all()


# ---------------------------------------------------------------------------
# scope
# ---------------------------------------------------------------------------


def test_non_committable_carriers_are_untouched():
    """Wind/solar/battery are never committable, so their UC columns stay as-is."""
    before = frame([make_plant("solar"), make_plant("onwind"), make_plant("battery")])
    after = sanitize_uc_parameters(before.copy())
    pd.testing.assert_frame_equal(after, before)


def test_missing_columns_raise():
    with pytest.raises(KeyError):
        sanitize_uc_parameters(pd.DataFrame({"carrier": ["coal"], "p_nom": [100.0]}))


def test_every_committable_carrier_has_a_band():
    assert set(UC_CARRIERS) == set(UC_BOUNDS)
    for carrier, bounds in UC_BOUNDS.items():
        for key in ("up_h", "down_h", "ramp", "start_usd_mw"):
            lo, hi, default = bounds[key]
            assert lo <= default <= hi, f"{carrier}.{key} default outside its own band"
        cap, default = bounds["p_min_pu"]
        assert 0 < default <= cap <= 1, f"{carrier}.p_min_pu default/cap out of range"


# ---------------------------------------------------------------------------
# hours -> snapshots rescale in prepare_network
# ---------------------------------------------------------------------------


def _uc_network():
    import pypsa

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=8, freq="h"))
    n.add("Bus", "bus")
    n.add(
        "Generator",
        ["ccgt", "coal"],
        bus="bus",
        p_nom=100.0,
        committable=True,
        min_up_time=[4, 9],
        min_down_time=[6, 8],
        ramp_limit_up=[0.6, 0.2],
        ramp_limit_down=[0.6, 0.2],
    )
    n.add("Generator", "solar", bus="bus", p_nom=10.0, committable=False)
    return n


@pytest.mark.parametrize("offset,expected", [("h", 1), ("3h", 3), ("24h", 24)])
def test_hours_per_snapshot(offset, expected):
    assert _hours_per_snapshot(offset) == expected


def test_hours_per_snapshot_rejects_non_hourly():
    with pytest.raises(ValueError):
        _hours_per_snapshot("4380seg")


def test_uc_attrs_rescaled_to_three_hour_snapshots():
    n = _uc_network()
    rescale_uc_attrs_to_resolution(n, "3h")
    # min up/down time round UP: a 4 h minimum still blocks two 3 h snapshots.
    assert n.generators.at["ccgt", "min_up_time"] == 2
    assert n.generators.at["coal", "min_up_time"] == 3
    assert n.generators.at["ccgt", "min_down_time"] == 2
    # ramp scales with the snapshot length and saturates at 1.
    assert n.generators.at["ccgt", "ramp_limit_up"] == pytest.approx(1.0)
    assert n.generators.at["coal", "ramp_limit_up"] == pytest.approx(0.6)


def test_uc_rescale_is_a_noop_at_hourly_resolution():
    n = _uc_network()
    before = n.generators.copy()
    rescale_uc_attrs_to_resolution(n, "h")
    pd.testing.assert_frame_equal(n.generators, before)


def test_uc_rescale_is_a_noop_without_committable_generators():
    """Runs with unit_commitment: false must be bit-for-bit unchanged."""
    n = _uc_network()
    n.generators["committable"] = False
    before = n.generators.copy()
    rescale_uc_attrs_to_resolution(n, "3h")
    pd.testing.assert_frame_equal(n.generators, before)


# ---------------------------------------------------------------------------
# up_time_before: the horizon must not start with a forced commitment
# ---------------------------------------------------------------------------


def _committable_uc_model(up_time_before):
    """One committable CCGT plus expensive backup, on a low/high/low load shape."""
    import pypsa

    pypsa.options.api.legacy_string_dtype = True
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=24, freq="3h"))
    n.add("Bus", "bus")
    n.add("Carrier", ["CCGT", "backup"])
    n.add(
        "Generator",
        "ccgt",
        bus="bus",
        carrier="CCGT",
        p_nom=500.0,
        committable=True,
        p_min_pu=0.4,
        min_up_time=2,
        min_down_time=2,
        ramp_limit_up=1.0,
        ramp_limit_down=1.0,
        start_up_cost=37_500.0,
        marginal_cost=50.0,
        up_time_before=up_time_before,
    )
    n.generators_t.p_max_pu = pd.DataFrame(0.91, index=n.snapshots, columns=["ccgt"])
    n.add("Generator", "backup", bus="bus", carrier="backup", p_nom=1e4, marginal_cost=5000.0)
    n.add(
        "Load",
        "load",
        bus="bus",
        p_set=pd.Series(np.r_[np.full(8, 50.0), np.full(8, 450.0), np.full(8, 50.0)], index=n.snapshots),
    )
    return n


def test_pypsa_default_up_time_before_forces_a_start_of_horizon_commitment():
    """Why add_electricity sets up_time_before = 0 on every committable unit.

    PyPSA defaults it to 1 — "the unit was online just before the horizon" — so a
    unit with min_up_time > 1 must stay online through the first snapshots and
    push its whole minimum stable level into them. At the annual load minimum,
    with load shedding off, that is an infeasibility rather than a cost.
    """
    pytest.importorskip("highspy")
    n = _committable_uc_model(up_time_before=1)
    status, _ = n.optimize(solver_name="highs", linearized_unit_commitment=True)
    assert status != "ok"


def test_zero_up_time_before_leaves_the_first_snapshot_free():
    pytest.importorskip("highspy")
    n = _committable_uc_model(up_time_before=0)
    status, condition = n.optimize(solver_name="highs", linearized_unit_commitment=True)
    assert (status, condition) == ("ok", "optimal")

    p = n.generators_t.p["ccgt"]
    st = n.model.solution["Generator-status"].to_pandas()["ccgt"]
    # The linearized relaxation keeps status continuous in [0, 1] ...
    assert 0.0 - 1e-6 <= st.min() and st.max() <= 1.0 + 1e-6
    # ... and both commitment bounds scale with it, which is what makes p_min_pu
    # a minimum *stable level* rather than an unconditional must-run.
    assert (p <= 0.91 * 500.0 * st + 1e-6).all()
    assert (p >= 0.40 * 500.0 * st - 1e-6).all()
