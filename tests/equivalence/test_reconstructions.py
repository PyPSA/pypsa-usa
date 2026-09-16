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
    """0.5 % of the larger side, with a 1 MW floor -- the table's own rule."""
    tol = CAPACITY_TOL
    assert tol.rtol == pytest.approx(5e-3)
    assert tol.atol == pytest.approx(1.0)
    # Below the floor, the floor wins.
    assert reconstructions.gate_tolerance(100.0, 100.0, tol) == pytest.approx(1.0)
    # Above it, 0.5 % of the larger side.
    assert reconstructions.gate_tolerance(10_000.0, 12_000.0, tol) == pytest.approx(60.0)
    assert reconstructions.gate_tolerance(0.0, 0.0, tol) == pytest.approx(1.0)
    # A 9 MW gate error on a 10,000/12,000 row is inside the row's tolerance.
    rows = _rows(
        _fleet([("s1", "onwind", 12_000.0, "pA")]),
        {"onwind": {"s1"}},
        {("pA", "onwind"): (12_000.0 - 9.0, 12_000.0)},
    )
    assert rows[("p_nom_existing_by_zone_carrier", "pA | onwind")].ok


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
    csv = tmp_path / "tables" / "hf26_reconstruction.csv"
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
