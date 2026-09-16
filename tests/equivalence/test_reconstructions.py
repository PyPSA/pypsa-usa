"""Fast unit tests for ``tests.equivalence.reconstructions``.

Tier A (``fast``): a handful of synthetic plants, substations and table rows.
No ``data/``, no ``resources/``, no snakemake, no network on disk.

What is under test is a CLAIM, not a number: that ``develop - master`` on the
existing-capacity rows equals the MW at substations master's profile file does
not cover. The claim is checkable because the reconstruction predicts BOTH sides
separately -- ``profiled_mw`` against master's column (gate G1) and ``fleet_mw``
against develop's (gate G2) -- so the tests below drive each failure mode on its
own: a wrong population, a wrong coverage set, and a relocation, which fails both
gates with equal and opposite residuals.

Deltas ledger DL-20; hot-fix registry HF-26.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.equivalence import metrics, reconstructions, tables

pytestmark = pytest.mark.fast

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "workflow" / "scripts"

CAPACITY_TOL = tables.tolerance_for("p_nom_existing_by_zone_carrier")


# --- helpers ----------------------------------------------------------------


def _fleet(rows) -> pd.DataFrame:
    """``(sub_id, carrier, p_nom, zone)`` rows as the frame the core consumes."""
    return pd.DataFrame(list(rows), columns=["sub_id", "carrier", "p_nom", "zone"])


def _zone_frame(values: dict[tuple[str, str], tuple[float, float]]) -> pd.DataFrame:
    """A ``p_nom_existing_by_zone_carrier`` metrics frame from (master, develop)."""
    master = pd.Series({k: v[0] for k, v in values.items()})
    develop = pd.Series({k: v[1] for k, v in values.items()})
    out = metrics.frame(master, develop, name="zone_carrier")
    out.index = pd.MultiIndex.from_tuples(list(out.index), names=["zone", "carrier"])
    return out


def _carrier_frame(values: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """A ``capacity_existing_by_carrier`` metrics frame from (master, develop)."""
    master = pd.Series({k: v[0] for k, v in values.items()})
    develop = pd.Series({k: v[1] for k, v in values.items()})
    return metrics.frame(master, develop, name="carrier")


#: Five plants over three substations, two zones, one carrier. ``s3`` is the
#: substation master's profile file does not cover.
FIVE_PLANTS = _fleet(
    [
        ("s1", "onwind", 100.0, "pA"),
        ("s1", "onwind", 50.0, "pA"),
        ("s2", "onwind", 200.0, "pA"),
        ("s3", "onwind", 300.0, "pB"),
        ("s3", "onwind", 25.0, "pB"),
    ],
)
ALL_SUBS = {"onwind": {"s1", "s2", "s3"}}
PROFILED_S1_S2 = {"onwind": {"s1", "s2"}}
NO_SUBS: dict[str, set[str]] = {"onwind": set()}


# --- core: the drop arithmetic ----------------------------------------------


def test_dropped_is_the_fleet_at_unprofiled_substations():
    out = reconstructions.reconstruct_drop(FIVE_PLANTS, PROFILED_S1_S2)
    assert out.loc[("pA", "onwind"), "fleet_mw"] == pytest.approx(350.0)
    assert out.loc[("pA", "onwind"), "profiled_mw"] == pytest.approx(350.0)
    assert out.loc[("pA", "onwind"), "dropped_mw"] == pytest.approx(0.0)
    assert out.loc[("pB", "onwind"), "fleet_mw"] == pytest.approx(325.0)
    assert out.loc[("pB", "onwind"), "profiled_mw"] == pytest.approx(0.0)
    assert out.loc[("pB", "onwind"), "dropped_mw"] == pytest.approx(325.0)
    assert int(out.loc[("pA", "onwind"), "n_subs"]) == 2
    assert int(out.loc[("pB", "onwind"), "n_subs_dropped"]) == 1


def test_nothing_is_dropped_when_every_substation_is_profiled():
    out = reconstructions.reconstruct_drop(FIVE_PLANTS, ALL_SUBS)
    assert out["dropped_mw"].abs().max() == pytest.approx(0.0)
    assert out["fleet_mw"].sum() == pytest.approx(675.0)
    assert out["profiled_mw"].sum() == pytest.approx(675.0)
    assert int(out["n_subs_dropped"].sum()) == 0


def test_everything_is_dropped_when_no_substation_is_profiled():
    out = reconstructions.reconstruct_drop(FIVE_PLANTS, NO_SUBS)
    assert out["profiled_mw"].sum() == pytest.approx(0.0)
    assert out["dropped_mw"].sum() == pytest.approx(675.0)


def test_a_carrier_with_no_profile_entry_at_all_is_fully_dropped():
    """A missing carrier key is not "everything covered"; it is nothing covered."""
    out = reconstructions.reconstruct_drop(FIVE_PLANTS, {"solar": {"s1"}})
    assert out["dropped_mw"].sum() == pytest.approx(675.0)


def test_substation_ids_reconcile_across_formats():
    """``'35827'``, ``35827.0`` and ``'35827.0'`` are one substation.

    They arrive from three artifacts in three spellings. A silent mismatch makes
    the covered set empty and every row read as 100 % dropped -- which still
    satisfies the develop gate, so only the master gate would catch it. Cheaper
    to catch here.
    """
    assert reconstructions.normalize_sub_id("35827") == "35827"
    assert reconstructions.normalize_sub_id(35827.0) == "35827"
    assert reconstructions.normalize_sub_id("35827.0") == "35827"
    assert reconstructions.normalize_sub_id(35827) == "35827"
    assert reconstructions.normalize_sub_id(np.float64(35827.0)) == "35827"
    # A foreign labelling convention is NOT guessed at.
    assert reconstructions.normalize_sub_id("035827") is None
    assert reconstructions.normalize_sub_id("35827.5") is None
    assert reconstructions.normalize_sub_id(float("nan")) is None
    assert reconstructions.normalize_sub_id(None) is None

    # The JOIN key falls back to the literal, so an id with no integer form
    # matches itself instead of matching nothing. Both sides go through it, so
    # '035827' still never equals '35827'.
    assert reconstructions.canonical_sub_id("35827.0") == "35827"
    assert reconstructions.canonical_sub_id("035827") == "035827"
    assert reconstructions.canonical_sub_id("035827") != reconstructions.canonical_sub_id("35827")
    # Missing stays missing: never the string 'nan'.
    assert reconstructions.canonical_sub_id(float("nan")) is None
    assert reconstructions.canonical_sub_id(None) is None

    fleet = _fleet(
        [
            ("35827", "onwind", 100.0, "pA"),  # busmap spelling
            (35828.0, "onwind", 200.0, "pA"),  # bus2sub spelling (a float)
        ],
    )
    # ...against the profile file's spelling.
    out = reconstructions.reconstruct_drop(fleet, {"onwind": {"35827.0", "35828.0"}})
    assert out.loc[("pA", "onwind"), "dropped_mw"] == pytest.approx(0.0)


# --- core: the gates --------------------------------------------------------


def _rows(
    fleet: pd.DataFrame,
    profiled: dict[str, set[str]],
    zone_values: dict[tuple[str, str], tuple[float, float]],
    carrier_values: dict[str, tuple[float, float]] | None = None,
):
    recon = reconstructions.reconstruct_drop(fleet, profiled)
    return reconstructions.build_rows(
        recon,
        _zone_frame(zone_values),
        _carrier_frame(carrier_values or {}),
        CAPACITY_TOL,
    )


def test_residual_is_zero_and_row_is_ok_when_both_gates_hold():
    rows = _rows(
        FIVE_PLANTS,
        PROFILED_S1_S2,
        {("pA", "onwind"): (350.0, 350.0), ("pB", "onwind"): (0.0, 325.0)},
    )
    pb = rows[("p_nom_existing_by_zone_carrier", "pB | onwind")]
    assert pb.ok, pb.note
    assert pb.dropped_mw == pytest.approx(325.0)
    assert pb.residual_mw == pytest.approx(0.0)
    assert pb.master_recon_err == pytest.approx(0.0)
    assert pb.develop_recon_err == pytest.approx(0.0)


def test_gate_g1_fails_when_the_master_column_disagrees():
    """``profiled_mw`` 10 MW off master: the coverage set is wrong."""
    rows = _rows(
        FIVE_PLANTS,
        PROFILED_S1_S2,
        {("pA", "onwind"): (340.0, 350.0), ("pB", "onwind"): (0.0, 325.0)},
    )
    pa = rows[("p_nom_existing_by_zone_carrier", "pA | onwind")]
    assert not pa.ok
    assert "master gate" in pa.note
    assert pa.master_recon_err == pytest.approx(10.0)


def test_gate_g2_fails_when_the_develop_column_disagrees():
    """``fleet_mw`` off develop: the plant population is wrong."""
    rows = _rows(
        FIVE_PLANTS,
        PROFILED_S1_S2,
        {("pA", "onwind"): (350.0, 350.0), ("pB", "onwind"): (0.0, 225.0)},
    )
    pb = rows[("p_nom_existing_by_zone_carrier", "pB | onwind")]
    assert not pb.ok
    assert "develop gate" in pb.note
    assert pb.develop_recon_err == pytest.approx(100.0)


def test_relocation_shows_as_equal_and_opposite_residuals():
    """The HF-27 shape: 100 MW that moved zone is explained by neither zone.

    The fleet says both zones' plants are where master put them; the table says
    develop moved 100 MW from A to B. Nothing in HF-26 can do that, so both rows
    must fail and their residuals must cancel -- which is what identifies the
    difference as a relocation rather than a drop.
    """
    fleet = _fleet([("s1", "onwind", 500.0, "pA"), ("s2", "onwind", 200.0, "pB")])
    rows = _rows(
        fleet,
        {"onwind": {"s1", "s2"}},
        {("pA", "onwind"): (500.0, 400.0), ("pB", "onwind"): (200.0, 300.0)},
    )
    pa = rows[("p_nom_existing_by_zone_carrier", "pA | onwind")]
    pb = rows[("p_nom_existing_by_zone_carrier", "pB | onwind")]
    assert not pa.ok and not pb.ok
    assert pa.residual_mw == pytest.approx(-100.0)
    assert pb.residual_mw == pytest.approx(100.0)
    assert pa.residual_mw + pb.residual_mw == pytest.approx(0.0)


def test_a_zone_the_reconstruction_never_saw_still_gets_a_row():
    """Appear-from-nothing must not fall through as "no prediction, so fine"."""
    fleet = _fleet([("s1", "onwind", 500.0, "pA")])
    rows = _rows(
        fleet,
        {"onwind": {"s1"}},
        {("pA", "onwind"): (500.0, 500.0), ("pZ", "onwind"): (0.0, 101.2)},
    )
    pz = rows[("p_nom_existing_by_zone_carrier", "pZ | onwind")]
    assert pz.dropped_mw == pytest.approx(0.0)
    assert pz.residual_mw == pytest.approx(101.2)
    assert not pz.ok


def test_gate_tolerance_is_the_rows_own_tolerance():
    """0.5 % of MASTER, with a 1 MW floor -- byte for byte the table's own rule.

    ``_row_verdict`` calls a row equivalent when ``|delta| <= atol`` or
    ``|delta_pct| <= rtol * 100``, and ``delta_pct`` is ``100 * delta / master``.
    So the row's tolerance in MW is master-denominated, and the gate must be too.
    Scaling by ``max(|master|, |develop|)`` -- which this did until the verifier
    caught it -- made the gate systematically looser than the verdict it guards,
    because develop exceeds master throughout the HF-26 regime.
    """
    tol = CAPACITY_TOL
    assert tol.rtol == pytest.approx(5e-3)
    assert tol.atol == pytest.approx(1.0)
    # Below the floor, the floor wins.
    assert reconstructions.gate_tolerance(100.0, 100.0, tol) == pytest.approx(1.0)
    # Above it, 0.5 % of MASTER -- not of develop, and not of the larger side.
    assert reconstructions.gate_tolerance(10_000.0, 12_000.0, tol) == pytest.approx(50.0)
    assert reconstructions.gate_tolerance(10_000.0, 10_000.0, tol) == pytest.approx(50.0)
    assert reconstructions.gate_tolerance(0.0, 65_000.0, tol) == pytest.approx(1.0)
    assert reconstructions.gate_tolerance(0.0, 0.0, tol) == pytest.approx(1.0)


def test_gate_tolerance_equals_what_the_comparison_table_calls_equivalent():
    """The drift alarm: the gate and ``_row_verdict`` must agree on the boundary.

    A delta one part in a thousand inside the row's tolerance is ``equivalent``
    in the table; the same number must be inside the gate. One part outside, and
    both must refuse it. Asserting the property rather than the formula is what
    keeps the two from drifting the next time a tolerance moves.
    """
    tol = CAPACITY_TOL
    for master in (200.0, 10_000.0, 17_770.9):
        gate = reconstructions.gate_tolerance(master, master * 1.5, tol)
        for delta, want in ((gate * 0.999, "equivalent"), (gate * 1.001, "UNEXPLAINED")):
            frames = {"p_nom_existing_by_zone_carrier": _zone_frame({("pA", "onwind"): (master, master + delta)})}
            out = tables.comparison_table(frames, {}, [])
            assert out.iloc[0]["verdict"] == want, (master, delta, gate, out.iloc[0].to_dict())


def test_a_gate_error_inside_the_rows_tolerance_still_passes():
    """A 9 MW G1 error on a row whose own tolerance is 50 MW is not a failure."""
    rows = _rows(
        _fleet([("s1", "onwind", 12_000.0, "pA")]),
        {"onwind": {"s1"}},
        {("pA", "onwind"): (10_000.0, 12_000.0)},
    )
    # master 10,000 -> gate 50 MW; reconstruction profiles the whole 12,000, so
    # G1 is 2,000 MW off and the row must FAIL.
    row = rows[("p_nom_existing_by_zone_carrier", "pA | onwind")]
    assert row.gate_tol_mw == pytest.approx(50.0)
    assert not row.ok and "master gate" in row.note

    # Now a 9 MW G1 error on a row whose own tolerance is ~60 MW: inside, so ok.
    rows = _rows(
        _fleet([("s1", "onwind", 12_000.0, "pA"), ("s2", "onwind", 9.0, "pA")]),
        {"onwind": {"s1"}},
        {("pA", "onwind"): (11_991.0, 12_009.0)},
    )
    row = rows[("p_nom_existing_by_zone_carrier", "pA | onwind")]
    assert row.gate_tol_mw == pytest.approx(59.955)
    assert row.master_recon_err == pytest.approx(9.0)
    assert row.ok, row.note


# --- core: the keys ---------------------------------------------------------


def test_national_row_is_the_zone_sum():
    rows = _rows(
        FIVE_PLANTS,
        PROFILED_S1_S2,
        {("pA", "onwind"): (350.0, 350.0), ("pB", "onwind"): (0.0, 325.0)},
        {"onwind": (350.0, 675.0)},
    )
    national = rows[("capacity_existing_by_carrier", "onwind")]
    zone_rows = [r for (m, _k), r in rows.items() if m == "p_nom_existing_by_zone_carrier"]
    assert national.zone is None
    assert national.fleet_mw == pytest.approx(sum(r.fleet_mw for r in zone_rows))
    assert national.dropped_mw == pytest.approx(sum(r.dropped_mw for r in zone_rows))
    assert national.ok, national.note
    assert national.residual_mw == pytest.approx(0.0)


def test_a_carrier_outside_the_reconstruction_has_no_row():
    """``CCGT`` moves for other reasons; the waiver must not reach it."""
    rows = _rows(
        FIVE_PLANTS,
        PROFILED_S1_S2,
        {("pA", "onwind"): (350.0, 350.0), ("pA", "CCGT"): (100.0, 900.0)},
        {"onwind": (350.0, 675.0), "CCGT": (100.0, 900.0)},
    )
    assert ("p_nom_existing_by_zone_carrier", "pA | CCGT") not in rows
    assert ("capacity_existing_by_carrier", "CCGT") not in rows
    recon = reconstructions.Reconstruction(
        name="x",
        frame=reconstructions.rows_frame(rows),
        rows=rows,
    )
    assert recon.lookup("capacity_existing_by_carrier", "CCGT") is None
    assert recon.lookup("capacity_existing_by_carrier", "onwind") is not None


def test_row_key_matches_the_comparison_tables_own_label():
    """``p10 | onwind``, exactly. A near-miss explains nothing while looking right."""
    frames = {"p_nom_existing_by_zone_carrier": _zone_frame({("p10", "onwind"): (1.0, 2.0)})}
    table = tables.comparison_table(frames, {}, [])
    assert table.iloc[0]["key"] == reconstructions.row_key("p10", "onwind")
    assert reconstructions.row_key(None, "onwind") == "onwind"


def test_a_failed_reconstruction_looks_up_nothing():
    """Partial results must not explain: the gates were never shown to hold."""
    rows = _rows(FIVE_PLANTS, PROFILED_S1_S2, {("pA", "onwind"): (350.0, 350.0)})
    broken = reconstructions.Reconstruction(
        name="x",
        frame=reconstructions.rows_frame(rows),
        rows=rows,
        error="OSError: master network missing",
    )
    assert not broken.ok
    assert broken.lookup("p_nom_existing_by_zone_carrier", "pA | onwind") is None


# --- tables.py integration --------------------------------------------------

HOTFIX_REGISTRY = {"HF-26": {"id": "HF-26", "ported": False, "usa_noop": False}}
RECON_WAIVER = {
    "metric": "p_nom_existing_by_zone_carrier",
    "reconstruction": "hf26_existing_renewable_drop",
    "hotfix": "HF-26",
    "expect_sign": "+",
}


def _registry(rows) -> dict:
    name = "hf26_existing_renewable_drop"
    return {name: reconstructions.Reconstruction(name=name, frame=reconstructions.rows_frame(rows), rows=rows)}


_DEFAULT = object()


def _table(zone_values, rows, waivers=None, reconstructions_arg=_DEFAULT):
    frames = {"p_nom_existing_by_zone_carrier": _zone_frame(zone_values)}
    reg = _registry(rows) if reconstructions_arg is _DEFAULT else reconstructions_arg
    return tables.comparison_table(
        frames,
        HOTFIX_REGISTRY,
        [RECON_WAIVER] if waivers is None else waivers,
        reconstructions=reg,
    )


def test_reconstruction_waiver_explains_a_matching_row():
    zone_values = {("pB", "onwind"): (0.0, 325.0)}
    rows = _rows(FIVE_PLANTS, PROFILED_S1_S2, {("pA", "onwind"): (350.0, 350.0), **zone_values})
    out = _table(zone_values, rows)
    row = out[out["key"] == "pB | onwind"].iloc[0]
    assert row["verdict"] == "explained"
    assert row["hotfix"] == "HF-26"


def test_reconstruction_waiver_refuses_a_row_with_a_residual():
    """The relocation shape stays UNEXPLAINED, with the MW in the cell."""
    fleet = _fleet([("s1", "onwind", 500.0, "pA"), ("s2", "onwind", 200.0, "pB")])
    zone_values = {("pA", "onwind"): (500.0, 400.0), ("pB", "onwind"): (200.0, 300.0)}
    rows = _rows(fleet, {"onwind": {"s1", "s2"}}, zone_values)
    out = _table(zone_values, rows)
    row = out[out["key"] == "pB | onwind"].iloc[0]
    assert row["verdict"] == "UNEXPLAINED"
    assert "reconstruction" in row["hotfix"]


def test_reconstruction_waiver_refuses_when_no_reconstruction_is_supplied():
    """``reconstructions=None`` means the waiver has NOT been shown to hold."""
    zone_values = {("pB", "onwind"): (0.0, 325.0)}
    out = _table(zone_values, {}, reconstructions_arg=None)
    row = out[out["key"] == "pB | onwind"].iloc[0]
    assert row["verdict"] == "UNEXPLAINED"
    assert row["hotfix"] == "waiver HF-26 bounds violated (reconstruction unavailable)"


def test_reconstruction_waiver_refuses_a_failed_gate():
    """A mechanism that cannot reproduce master's own column explains nothing."""
    zone_values = {("pA", "onwind"): (340.0, 350.0), ("pB", "onwind"): (0.0, 325.0)}
    rows = _rows(FIVE_PLANTS, PROFILED_S1_S2, zone_values)
    out = _table(zone_values, rows)
    row = out[out["key"] == "pA | onwind"].iloc[0]
    assert row["verdict"] == "UNEXPLAINED"
    assert "reconstruction gate" in row["hotfix"]
    assert "master gate" in row["hotfix"]


def test_reconstruction_waiver_still_honours_expect_sign():
    """A negative delta of the right magnitude is still the wrong difference.

    The gates hold and the residual is 0 -- master profiled 400 MW, develop's
    fleet is 100 MW -- so only ``expect_sign`` stands between this row and an
    ``explained`` verdict it has no business getting: HF-26 can only ADD capacity
    on develop.
    """
    zone_values = {("pA", "onwind"): (400.0, 100.0)}
    rows = {
        ("p_nom_existing_by_zone_carrier", "pA | onwind"): reconstructions._make_row(
            "p_nom_existing_by_zone_carrier",
            "pA",
            "onwind",
            fleet_mw=100.0,
            profiled_mw=400.0,
            master_mw=400.0,
            develop_mw=100.0,
            tol=CAPACITY_TOL,
        ),
    }
    out = _table(zone_values, rows)
    row = out[out["key"] == "pA | onwind"].iloc[0]
    assert row["verdict"] == "UNEXPLAINED"
    assert row["hotfix"] == "waiver HF-26 bounds violated (sign)"


def test_reconstruction_waiver_refuses_a_key_it_has_no_row_for():
    """The metric-wide waiver does not reach ``CCGT``."""
    zone_values = {("pA", "CCGT"): (100.0, 900.0)}
    rows = _rows(FIVE_PLANTS, PROFILED_S1_S2, {("pA", "onwind"): (350.0, 350.0)})
    out = _table(zone_values, rows)
    row = out[out["key"] == "pA | CCGT"].iloc[0]
    assert row["verdict"] == "UNEXPLAINED"
    assert "reconstruction unavailable" in row["hotfix"]


def test_comparison_md_has_a_reconstructions_section():
    zone_values = {("pB", "onwind"): (0.0, 325.0)}
    rows = _rows(FIVE_PLANTS, PROFILED_S1_S2, {("pA", "onwind"): (350.0, 350.0), **zone_values})
    table = _table(zone_values, rows)
    md = tables.to_markdown(table, reconstructions=_registry(rows))
    assert "## Reconstructions" in md
    assert "hf26_existing_renewable_drop" in md
    assert "pB \\| onwind" in md

    # Present and explicit when none ran.
    empty = tables.to_markdown(table, reconstructions={})
    assert "## Reconstructions" in empty
    assert "No reconstructions were consulted" in empty


def test_reconstruction_csv_and_json_are_written(tmp_path):
    zone_values = {("pB", "onwind"): (0.0, 325.0)}
    rows = _rows(FIVE_PLANTS, PROFILED_S1_S2, {("pA", "onwind"): (350.0, 350.0), **zone_values})
    table = _table(zone_values, rows)
    written = tables.write_tables({"comparison": table}, tmp_path, reconstructions=_registry(rows))
    csv = tmp_path / "tables" / "reconstructions.csv"
    assert csv in written
    frame = pd.read_csv(csv)
    assert set(frame["zone"]) == {"pA", "pB"}
    assert "residual_mw" in frame.columns
    summary = tables.reconstructions_summary(_registry(rows))
    assert summary[0]["name"] == "hf26_existing_renewable_drop"
    assert summary[0]["ok"] is True
    assert summary[0]["n_rows"] == len(rows)


def test_reconstruction_summary_records_a_failure():
    """A reconstruction that could not run must survive the run in writing."""
    broken = {
        "hf26_existing_renewable_drop": reconstructions.Reconstruction(
            name="hf26_existing_renewable_drop",
            frame=reconstructions.empty_recon_frame(),
            error="FileNotFoundError: elec_base_network.nc",
        ),
    }
    summary = tables.reconstructions_summary(broken)
    assert summary[0]["ok"] is False
    assert "FileNotFoundError" in summary[0]["error"]
    assert "ERROR" in tables.to_markdown(pd.DataFrame(), reconstructions=broken)


def test_reconstruction_is_a_table_bound_the_ledger_test_can_see():
    assert "reconstruction" in tables.WAIVER_BOUND_KEYS
    assert "reconstruction" in tables.WAIVER_COMPUTED_KEYS
    assert tables.reconstruction_names([RECON_WAIVER]) == ["hf26_existing_renewable_drop"]


# --- the lazy registry ------------------------------------------------------


def test_the_registry_is_lazy_and_memoised():
    """Nothing runs until a row asks, and then only once."""
    calls = []

    def builder(art, frames):
        calls.append(1)
        return reconstructions.Reconstruction(name="probe", frame=reconstructions.empty_recon_frame())

    reg = reconstructions.LazyRegistry(object(), {}, {"probe": builder})
    assert reg.resolved() == {}
    assert calls == []
    assert reg["probe"].name == "probe"
    assert reg["probe"].name == "probe"
    assert calls == [1]
    assert list(reg.resolved()) == ["probe"]


def test_reconstructions_can_be_switched_off(monkeypatch):
    """``EQ_RECONSTRUCTIONS=0`` is fail-safe: rows go UNEXPLAINED, never explained."""
    monkeypatch.setenv("EQ_RECONSTRUCTIONS", "0")
    assert not reconstructions.enabled()
    assert reconstructions.registry(object(), {}) == {}
    monkeypatch.setenv("EQ_RECONSTRUCTIONS", "1")
    assert reconstructions.enabled()


def test_a_reconstruction_that_raises_becomes_an_error_not_an_exception():
    """The entry point catches broadly; a missing artifact must not abort a run."""

    class NoArtifacts:
        prong = 2
        master_root = Path("/nonexistent/master/workflow")
        develop_root = Path("/nonexistent/develop/workflow")

    recon = reconstructions.hf26_existing_renewable_drop(
        NoArtifacts(),
        {"capacity_existing_by_carrier": _carrier_frame({"onwind": (1.0, 2.0)})},
    )
    assert recon.error
    assert recon.lookup("capacity_existing_by_carrier", "onwind") is None


def test_investment_year_prefers_the_already_loaded_network(tmp_path):
    """Never re-open a 100 s file for one integer the caller is already holding.

    ``plots.load_artifacts`` has loaded master's assembled network as
    ``art.assembled_master`` before any reconstruction runs; reading
    ``investment_periods`` off it costs nothing, while ``pypsa.Network(path)``
    on the USA ``elec_s300.nc`` costs about 100 s of every run.
    """
    # A path that would raise if it were opened; the loaded object wins first.
    loaded = SimpleNamespace(investment_periods=[2040])
    assert reconstructions.investment_year(tmp_path, 2, {}, loaded) == 2040

    # A real pypsa network carries a pandas INDEX here, not a list, and
    # `Index or []` raises. That bug once made the file branch silently useless.
    index_like = SimpleNamespace(investment_periods=pd.Index([2030, 2040]))
    assert reconstructions.investment_year(tmp_path, 2, {}, index_like) == 2030
    assert reconstructions._first_investment_period(index_like) == 2030
    assert reconstructions._first_investment_period(object()) is None

    # No loaded object, no file: the harness config answers.
    cfg = {"scenario": {"planning_horizons": [2035]}}
    assert reconstructions.investment_year(tmp_path, 2, cfg) == 2035

    # An empty investment_periods list is not an answer; fall through.
    assert reconstructions.investment_year(tmp_path, 2, cfg, SimpleNamespace(investment_periods=[])) == 2035

    with pytest.raises(RuntimeError, match="no investment year"):
        reconstructions.investment_year(tmp_path, 2, {})


def test_no_metric_frames_is_an_error_not_a_blank_cheque():
    class NoArtifacts:
        prong = 2
        master_root = Path("/nonexistent/master/workflow")
        develop_root = Path("/nonexistent/develop/workflow")

    recon = reconstructions.hf26_existing_renewable_drop(NoArtifacts(), {})
    assert recon.error and "metric frames" in recon.error


# --- population parity ------------------------------------------------------


SYNTHETIC_PLANTS = [
    # name, status, build_year, planned_op, retire, planned_retire, nerc, pm, carrier, MW
    ("keep_existing_wind", "existing", 2010, None, None, None, "WECC", "WT", "onwind", 100.0),
    ("keep_existing_solar", "existing", 2015, None, None, None, "WECC", "PV", "solar", 200.0),
    ("keep_proposed_wind", "proposed", None, "2028-06-01", None, None, "WECC", "WT", "onwind", 300.0),
    ("drop_proposed_late", "proposed", None, "2045-06-01", None, None, "WECC", "WT", "onwind", 400.0),
    ("drop_built_after", "existing", 2044, None, None, None, "WECC", "WT", "onwind", 500.0),
    ("drop_retired_before", "retired", 2000, None, "2020-01-01", None, "WECC", "WT", "onwind", 600.0),
    ("drop_announced_retirement", "existing", 2000, None, None, "2027-01-01", "WECC", "WT", "onwind", 700.0),
    ("keep_late_announced", "existing", 2000, None, None, "2045-01-01", "WECC", "WT", "onwind", 800.0),
    ("drop_non_conus", "existing", 2000, None, None, None, "non-conus", "WT", "onwind", 900.0),
    ("drop_other_interconnect", "existing", 2000, None, None, None, "TRE", "WT", "onwind", 1000.0),
    ("drop_pumped_storage", "existing", 2000, None, None, None, "WECC", "PS", "onwind", 1100.0),
    ("drop_other_carrier", "existing", 2000, None, None, None, "WECC", "CT", "CCGT", 1200.0),
]


def _synthetic_plant_csv(path: Path) -> Path:
    rows = []
    for name, status, build, planned_op, retire, planned_retire, nerc, pm, carrier, mw in SYNTHETIC_PLANTS:
        rows.append(
            {
                "generator_name": name,
                "operational_status": status,
                "build_year": build,
                "current_planned_generator_operating_date": planned_op,
                "generator_retirement_date": retire,
                "planned_generator_retirement_date": planned_retire,
                "nerc_region": nerc,
                "prime_mover_code": pm,
                "carrier": carrier,
                "p_nom": mw,
                "latitude": 37.0,
                "longitude": -120.0,
            },
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_filter_fleet_matches_the_workflow_load_powerplants(tmp_path):
    """The drift alarm for the hand-mirrored plant filter.

    ``filter_fleet`` mirrors ``add_electricity.load_powerplants`` rather than
    calling it, because importing master-benchmark's ``add_electricity``
    alongside develop's under the same module name is a ``sys.path`` collision.
    Mirrored code drifts, so the two are run side by side on a synthetic table
    covering every branch: proposed, built-too-late, already-retired,
    announced-retirement, non-conus, another interconnect, pumped storage and
    another carrier.

    Assumption, stated because the test depends on it: ``honor_planned_retirements``
    and the PUDL year are **ported** to ``master-benchmark`` (harness decision
    2026-09-14), so develop's ``load_powerplants`` is master's too.
    """
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    add_electricity = pytest.importorskip("add_electricity")

    csv = _synthetic_plant_csv(tmp_path / "powerplants.csv")
    workflow = add_electricity.load_powerplants(
        csv,
        investment_periods=[2030],
        interconnect="western",
        honor_planned_retirements=True,
    )
    workflow = workflow[workflow["prime_mover_code"] != "PS"]
    workflow = workflow[workflow["carrier"].isin(["onwind", "solar"])]

    mine = reconstructions.filter_fleet(
        pd.read_csv(csv),
        investment_year=2030,
        interconnect="western",
        honor_planned_retirements=True,
    )
    assert sorted(mine.index) == sorted(workflow.index)
    assert sorted(mine.index) == [
        "keep_existing_solar",
        "keep_existing_wind",
        "keep_late_announced",
        "keep_proposed_wind",
    ]
    assert mine["p_nom"].sum() == pytest.approx(1400.0)


def test_filter_fleet_honours_the_retirement_switch(tmp_path):
    """``honor_planned_retirements: false`` keeps the announced-retirement unit."""
    csv = _synthetic_plant_csv(tmp_path / "powerplants.csv")
    kept = reconstructions.filter_fleet(
        pd.read_csv(csv),
        investment_year=2030,
        interconnect="western",
        honor_planned_retirements=False,
    )
    assert "drop_announced_retirement" in kept.index


def test_filter_fleet_usa_keeps_every_conus_interconnect(tmp_path):
    """``interconnect: usa`` must not go through the NERC mapper at all."""
    csv = _synthetic_plant_csv(tmp_path / "powerplants.csv")
    kept = reconstructions.filter_fleet(
        pd.read_csv(csv),
        investment_year=2030,
        interconnect="usa",
        honor_planned_retirements=True,
    )
    assert "drop_other_interconnect" in kept.index
    assert "drop_non_conus" not in kept.index


def test_nearest_bus_match_is_the_workflows_own_metric():
    """A BallTree over RAW DEGREES, the same one ``match_nearest_bus`` builds.

    Equivalence with master depends on reproducing master's metric including the
    degree distortion, so the assignment is checked against the workflow function
    rather than against a haversine distance that would be more correct and less
    equal.
    """
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    add_electricity = pytest.importorskip("add_electricity")

    buses = pd.DataFrame(
        {
            "x": [-120.0, -119.0, -118.0],
            "y": [37.0, 37.0, 37.0],
            "sub_id": ["1", "2", "3"],
            "reeds_zone": ["pA", "pA", "pB"],
        },
        index=pd.Index(["b0", "b1", "b2"], name="Bus"),
    )
    plants = pd.DataFrame(
        {"longitude": [-119.9, -118.2], "latitude": [37.0, 37.1], "p_nom": [10.0, 20.0]},
        index=["p1", "p2"],
    )
    mine = reconstructions.assign_to_master_substations(plants, buses)
    theirs = add_electricity.match_nearest_bus(plants.copy(), buses)
    assert list(mine["bus_assignment"]) == list(theirs["bus_assignment"])
    assert list(mine["sub_id"]) == ["1", "3"]
    assert list(mine["zone"]) == ["pA", "pB"]


def test_assign_to_master_substations_tolerates_an_empty_fleet():
    buses = pd.DataFrame(
        {"x": [-120.0], "y": [37.0], "sub_id": ["1"], "reeds_zone": ["pA"]},
        index=pd.Index(["b0"], name="Bus"),
    )
    empty = pd.DataFrame(columns=["longitude", "latitude", "p_nom", "carrier"])
    out = reconstructions.assign_to_master_substations(empty, buses)
    assert out.empty
    assert {"sub_id", "zone"} <= set(out.columns)
    assert reconstructions.reconstruct_drop(out, {}).empty


# ---------------------------------------------------------------------------
# HF-24: the NREL caps p_nom_max master drops with a zero-availability bus.
#
# Same shape as HF-26 -- master drops, develop keeps, delta >= 0 -- but the
# mechanism is one level earlier: `build_renewable_profiles` filters on
# `profile.mean("time") > min_p_max_pu`, a substation with zero summed GODEEEP
# availability has a NaN CF, `NaN > 0` is False, and the bus leaves the file
# taking its caps `p_nom_max` with it. Develop sums the caps per cluster BEFORE
# that filter, so the MW survives.


#: Five substations over two clusters in two zones. ``s5`` is outside the
#: busmap (out of footprint); ``s3``/``s4`` are the ones master drops.
CAPS = pd.Series(
    {"1.0": 100.0, "2.0": 50.0, "3.0": 200.0, "4.0": 25.0, "5.0": 999.0},
    name="p_nom_max",
)
BUSMAP = pd.Series({"1": "cA", "2": "cA", "3": "cB", "4": "cB"})
ZONE_BY_CLUSTER = pd.Series({"cA": "pA", "cB": "pB", "cC": "pC"})
COMMON = ["cA", "cB"]


def test_caps_drop_is_the_caps_at_unprofiled_substations():
    out = reconstructions.reconstruct_caps_drop(CAPS, BUSMAP, {"1.0", "2.0"}, ZONE_BY_CLUSTER, COMMON)
    assert out.loc["pA", "caps_mw"] == pytest.approx(150.0)
    assert out.loc["pA", "covered_mw"] == pytest.approx(150.0)
    assert out.loc["pA", "dropped_mw"] == pytest.approx(0.0)
    assert out.loc["pB", "caps_mw"] == pytest.approx(225.0)
    assert out.loc["pB", "covered_mw"] == pytest.approx(0.0)
    assert out.loc["pB", "dropped_mw"] == pytest.approx(225.0)
    assert int(out.loc["pB", "n_subs_dropped"]) == 2


def test_nothing_is_dropped_when_master_profiles_every_substation():
    out = reconstructions.reconstruct_caps_drop(
        CAPS,
        BUSMAP,
        {"1.0", "2.0", "3.0", "4.0"},
        ZONE_BY_CLUSTER,
        COMMON,
    )
    assert out["dropped_mw"].abs().max() == pytest.approx(0.0)
    assert out["caps_mw"].sum() == pytest.approx(375.0)


def test_substations_outside_the_busmap_enter_neither_term():
    """The caps file is national; a footprint-scoped run must not count ``s5``.

    Counting it in ``caps_mw`` would fail gate G2 against a develop column that
    never contained it -- a harness artefact reported as a model difference.
    """
    out = reconstructions.reconstruct_caps_drop(CAPS, BUSMAP, {"1.0", "2.0"}, ZONE_BY_CLUSTER, COMMON)
    assert out["caps_mw"].sum() == pytest.approx(375.0)  # 999 MW of s5 excluded
    assert "pC" not in out.index


def test_clusters_outside_the_common_set_are_excluded():
    """A develop-only cluster must not inflate ``caps_mw``.

    ``metrics.common_cluster_subset`` removes it from BOTH sides of the metric
    before either is summed, so a reconstruction that kept it would be predicting
    a develop column that does not exist.
    """
    out = reconstructions.reconstruct_caps_drop(CAPS, BUSMAP, {"1.0", "2.0"}, ZONE_BY_CLUSTER, ["cA"])
    assert list(out.index) == ["pA"]
    assert out["caps_mw"].sum() == pytest.approx(150.0)


def test_caps_substation_ids_reconcile_across_formats():
    """Caps keys are ``'35827.0'``, busmap keys ``'35827'``, profile ``'35827.0'``."""
    caps = pd.Series({"35827.0": 10.0, "35828.0": 20.0})
    busmap = pd.Series({"35827": "cA", "35828": "cA"})
    zone = pd.Series({"cA": "pA"})
    out = reconstructions.reconstruct_caps_drop(caps, busmap, {"35827.0"}, zone, ["cA"])
    assert out.loc["pA", "caps_mw"] == pytest.approx(30.0)
    assert out.loc["pA", "covered_mw"] == pytest.approx(10.0)
    assert out.loc["pA", "dropped_mw"] == pytest.approx(20.0)


POTENTIAL_TOL = tables.tolerance_for("p_nom_max_by_zone_onwind")


def _caps_rows(zone_values, profiled=("1.0", "2.0"), common=COMMON):
    """``{(metric, zone): ReconRow}`` for the onwind potential metric."""
    recon = reconstructions.reconstruct_caps_drop(CAPS, BUSMAP, set(profiled), ZONE_BY_CLUSTER, common)
    rows = {}
    for zone, (master, develop) in zone_values.items():
        caps = float(recon["caps_mw"].get(zone, 0.0))
        covered = float(recon["covered_mw"].get(zone, 0.0))
        row = reconstructions._make_row(
            "p_nom_max_by_zone_onwind",
            zone,
            "onwind",
            caps,
            covered,
            master,
            develop,
            POTENTIAL_TOL,
            key=zone,
        )
        rows[("p_nom_max_by_zone_onwind", row.key)] = row
    return rows


def test_caps_residual_is_zero_and_row_is_ok_when_both_gates_hold():
    rows = _caps_rows({"pB": (0.0, 225.0)})
    row = rows[("p_nom_max_by_zone_onwind", "pB")]
    assert row.key == "pB", "the potential metric is keyed by the bare zone"
    assert row.ok, row.note
    assert row.dropped_mw == pytest.approx(225.0)
    assert row.residual_mw == pytest.approx(0.0)


def test_caps_gate_g1_fails_when_the_master_column_disagrees():
    rows = _caps_rows({"pA": (140.0, 150.0)})
    row = rows[("p_nom_max_by_zone_onwind", "pA")]
    assert not row.ok and "master gate" in row.note
    assert row.master_recon_err == pytest.approx(10.0)


def test_caps_gate_g2_fails_when_the_develop_column_disagrees():
    rows = _caps_rows({"pB": (0.0, 200.0)})
    row = rows[("p_nom_max_by_zone_onwind", "pB")]
    assert not row.ok and "develop gate" in row.note
    assert row.develop_recon_err == pytest.approx(25.0)


CAPS_WAIVER = {
    "metric": "p_nom_max_by_zone_onwind",
    "reconstruction": "hf24_nrel_caps_drop",
    "hotfix": "HF-24",
    "expect_sign": "+",
}
CAPS_REGISTRY_HOTFIX = {"HF-24": {"id": "HF-24", "ported": False, "usa_noop": False}}


def _caps_table(zone_values, rows):
    master = pd.Series({z: v[0] for z, v in zone_values.items()})
    develop = pd.Series({z: v[1] for z, v in zone_values.items()})
    frames = {"p_nom_max_by_zone_onwind": metrics.frame(master, develop, name="zone")}
    registry = {
        "hf24_nrel_caps_drop": reconstructions.Reconstruction(
            name="hf24_nrel_caps_drop",
            frame=reconstructions.rows_frame(rows),
            rows=rows,
        ),
    }
    return tables.comparison_table(frames, CAPS_REGISTRY_HOTFIX, [CAPS_WAIVER], reconstructions=registry)


def test_reconstruction_waiver_explains_a_matching_p_nom_max_row():
    zone_values = {"pB": (0.0, 225.0)}
    out = _caps_table(zone_values, _caps_rows(zone_values))
    row = out[out["key"] == "pB"].iloc[0]
    assert row["verdict"] == "explained"
    assert row["hotfix"] == "HF-24"


def test_reconstruction_waiver_refuses_a_p_nom_max_row_with_a_residual():
    zone_values = {"pB": (0.0, 200.0)}
    out = _caps_table(zone_values, _caps_rows(zone_values))
    row = out[out["key"] == "pB"].iloc[0]
    assert row["verdict"] == "UNEXPLAINED"
    assert "reconstruction" in row["hotfix"]


@pytest.mark.parametrize(
    ("config", "want"),
    [
        ({"renewable_land_access": "reference"}, "caps_onwind_reference.nc"),
        ({"renewable_land_access": "open"}, "caps_onwind_open.nc"),
        (
            {"renewable_land_access": "reference", "apply_cec_basescreen": True},
            "caps_onwind_reference_cec.nc",
        ),
        (
            {"renewable_land_access": "reference", "apply_boem_osw": True},
            "caps_onwind_reference.nc",  # boem is offwind-only
        ),
    ],
)
def test_nrel_caps_path_matches_the_snakemake_input_function(tmp_path, config, want):
    """Mirrors ``build_electricity.smk::nrel_exclusion_artifact(kind='caps')``.

    A path this function gets wrong does not raise: it either misses (error, so
    nothing is explained) or, worse, finds a DIFFERENT access scenario's file and
    reconstructs rows from land rules the run never used.
    """
    path = reconstructions.nrel_caps_path(tmp_path, "onwind", config)
    assert path.name == want
    assert path.parent == tmp_path / "data" / "nrel_exclusion" / "derived"


def test_nrel_caps_path_offwind_honours_boem_and_not_cec(tmp_path):
    cfg = {"renewable_land_access": "reference", "apply_cec_basescreen": True, "apply_boem_osw": True}
    assert reconstructions.nrel_caps_path(tmp_path, "offwind_floating", cfg).name == (
        "caps_offwind_floating_reference_boem.nc"
    )


def test_nrel_caps_path_raises_when_land_access_is_unset(tmp_path):
    """The atlite path has no caps file, so the mechanism must not hold there."""
    with pytest.raises(RuntimeError, match="renewable_land_access"):
        reconstructions.nrel_caps_path(tmp_path, "onwind", {})


def test_hf24_without_the_common_cluster_set_is_an_error_not_a_guess():
    """``art.profiles`` is only filled by ``collect_metrics`` at prong 2."""
    art = SimpleNamespace(
        prong=2,
        develop_root=Path("/nonexistent/develop/workflow"),
        master_root=Path("/nonexistent/master/workflow"),
        profiles={},
        zone_develop=ZONE_BY_CLUSTER,
    )
    recon = reconstructions.hf24_nrel_caps_drop(art, {"p_nom_max_by_zone_onwind": pd.DataFrame()})
    assert recon.error
    assert recon.lookup("p_nom_max_by_zone_onwind", "pA") is None


def test_both_providers_are_registered():
    assert set(reconstructions.RECONSTRUCTIONS) == {
        "hf26_existing_renewable_drop",
        "hf24_nrel_caps_drop",
    }


def test_rows_frame_keeps_every_zoned_row_whatever_the_metric():
    """One frame, one CSV, one set of figures, for both providers."""
    caps_rows = _caps_rows({"pA": (150.0, 150.0), "pB": (0.0, 225.0)})
    frame = reconstructions.rows_frame(caps_rows)
    assert set(frame["zone"]) == {"pA", "pB"}
    assert set(frame["metric"]) == {"p_nom_max_by_zone_onwind"}
    # HF-26's national rows carry no zone and stay out of the map data.
    mixed = _rows(FIVE_PLANTS, PROFILED_S1_S2, {("pA", "onwind"): (350.0, 350.0)}, {"onwind": (350.0, 675.0)})
    assert all(z is not None for z in reconstructions.rows_frame(mixed)["zone"])
    assert ("capacity_existing_by_carrier", "onwind") in mixed


def test_reconstruction_totals_sum_the_zone_rows_for_either_provider():
    """The summary line must not read "0 MW" for a provider with no national row.

    HF-26 carries national rows off ``capacity_existing_by_carrier``; HF-24's
    metric is already per tech and has none. Summing the national rows made the
    HF-24 summary read 0 MW beside 50 explained rows.
    """
    hf26 = _rows(
        FIVE_PLANTS,
        PROFILED_S1_S2,
        {("pA", "onwind"): (350.0, 350.0), ("pB", "onwind"): (0.0, 325.0)},
        {"onwind": (350.0, 675.0)},
    )
    r26 = reconstructions.Reconstruction("hf26", reconstructions.rows_frame(hf26), hf26)
    assert r26.totals()["fleet_mw"] == pytest.approx(675.0)
    assert r26.totals()["dropped_mw"] == pytest.approx(325.0)
    # ...and it agrees with the national row it also has.
    assert r26.totals()["fleet_mw"] == pytest.approx(r26.national()["onwind"]["fleet_mw"])

    hf24 = _caps_rows({"pA": (150.0, 150.0), "pB": (0.0, 225.0)})
    r24 = reconstructions.Reconstruction("hf24", reconstructions.rows_frame(hf24), hf24)
    assert r24.national() == {}, "the potential metric has no national row"
    assert r24.totals()["fleet_mw"] == pytest.approx(375.0)
    assert r24.totals()["dropped_mw"] == pytest.approx(225.0)
    assert r24.max_abs_residual() == pytest.approx(0.0)


def test_the_reconstructions_summary_reports_both_providers(tmp_path):
    """One `reconstructions.csv` and one markdown section cover every provider."""
    hf26 = _rows(FIVE_PLANTS, PROFILED_S1_S2, {("pB", "onwind"): (0.0, 325.0)})
    hf24 = _caps_rows({"pB": (0.0, 225.0)})
    registry = {
        "hf26_existing_renewable_drop": reconstructions.Reconstruction(
            "hf26_existing_renewable_drop",
            reconstructions.rows_frame(hf26),
            hf26,
        ),
        "hf24_nrel_caps_drop": reconstructions.Reconstruction(
            "hf24_nrel_caps_drop",
            reconstructions.rows_frame(hf24),
            hf24,
        ),
    }
    summary = tables.reconstructions_summary(registry)
    assert {s["name"] for s in summary} == set(registry)
    assert all(s["ok"] for s in summary)
    by_name = {s["name"]: s for s in summary}
    assert by_name["hf24_nrel_caps_drop"]["totals"]["dropped_mw"] == pytest.approx(225.0)

    frame = tables.reconstruction_frame(registry)
    assert set(frame["reconstruction"]) == set(registry)
    assert set(frame["metric"]) == {"p_nom_existing_by_zone_carrier", "p_nom_max_by_zone_onwind"}

    tables.write_tables({"comparison": pd.DataFrame()}, tmp_path, reconstructions=registry)
    csv = pd.read_csv(tmp_path / "tables" / "reconstructions.csv")
    assert set(csv["reconstruction"]) == set(registry)
