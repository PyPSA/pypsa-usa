"""Fast unit tests for ``tests.equivalence.tables``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.equivalence import compare, metrics, tables

from .conftest import make_network

pytestmark = pytest.mark.fast


def _metric_frame(master: dict[str, float], develop: dict[str, float]) -> pd.DataFrame:
    return metrics.frame(pd.Series(master), pd.Series(develop), name="carrier")


def test_tolerances_are_the_single_source_of_truth():
    """``compare.py`` must import every tolerance, not restate any of them."""
    assert compare.OBJECTIVE_RTOL == tables.TOLERANCES["objective"].rtol
    assert compare.CAPACITY_RTOL == tables.TOLERANCES["capacity"].rtol
    assert compare.CAPACITY_ATOL == tables.TOLERANCES["capacity"].atol
    assert compare.RTOL == tables.TOLERANCES["p_max_pu"].rtol
    src = (__import__("pathlib").Path(compare.__file__)).read_text()
    for literal in ("OBJECTIVE_RTOL = 1e-3", "CAPACITY_RTOL = 5e-3", "RTOL = 1e-3", "atol=1.0"):
        assert literal not in src, f"{literal!r} is a second home for a tolerance"


def test_every_family_has_a_relative_and_an_absolute_tolerance():
    for family, tol in tables.TOLERANCES.items():
        assert tol.rtol > 0, family
        assert tol.atol >= 0, family


def test_tolerance_family_resolution():
    assert tables.tolerance_family("objective") == "objective"
    assert tables.tolerance_family("capacity_existing_by_carrier") == "capacity"
    assert tables.tolerance_family("p_nom_existing_by_zone_carrier") == "capacity"
    assert tables.tolerance_family("p_nom_max_by_zone_solar") == "p_nom_max"
    assert tables.tolerance_family("p_max_pu_quantiles_solar") == "p_max_pu"
    assert tables.tolerance_family("mean_cf_by_zone_solar") == "p_max_pu"
    assert tables.tolerance_family("dispatch_by_carrier") == "dispatch"
    assert tables.tolerance_family("capacity_factor_by_carrier") == "dispatch"
    assert tables.tolerance_family("demand_by_zone") == "demand"


def test_tolerance_family_unknown_metric_raises():
    with pytest.raises(ValueError, match="no tolerance family"):
        tables.tolerance_family("something_nobody_assigned")


def test_comparison_table_verdicts():
    frames = {
        "capacity_existing_by_carrier": _metric_frame(
            {"solar": 100.0, "onwind": 100.0, "CCGT": 100.0},
            {"solar": 100.0, "onwind": 150.0, "CCGT": 180.0},
        ),
    }
    hotfixes = {
        "HF-8": {"id": "HF-8", "ported": False, "expect": ["capacity_existing_by_carrier/onwind"]},
    }
    out = tables.comparison_table(frames, hotfixes)
    assert list(out.columns) == tables.COMPARISON_COLUMNS
    verdicts = dict(zip(out["key"], out["verdict"]))
    assert verdicts["solar"] == "equivalent"
    assert verdicts["onwind"] == "explained"
    assert verdicts["CCGT"] == "UNEXPLAINED"
    # UNEXPLAINED first, then |delta_pct| descending.
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"
    assert out.iloc[0]["key"] == "CCGT"
    assert dict(zip(out["key"], out["hotfix"]))["onwind"] == "HF-8"
    assert (out["tolerance_pct"] == tables.TOLERANCES["capacity"].rtol * 100.0).all()
    assert (out["tolerance_abs"] == tables.TOLERANCES["capacity"].atol).all()


def test_ported_hotfix_is_not_an_explanation():
    """A fix on ``master-benchmark`` is on BOTH sides, so it explains nothing."""
    frames = {"capacity_existing_by_carrier": _metric_frame({"solar": 100.0}, {"solar": 180.0})}
    hotfixes = {"HF-8": {"id": "HF-8", "ported": True, "expect": ["capacity_existing_by_carrier/*"]}}
    out = tables.comparison_table(frames, hotfixes)
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"
    assert "ported" in out.iloc[0]["hotfix"]
    assert "HF-8" in out.iloc[0]["hotfix"]


def test_unported_hotfix_wins_over_a_ported_one():
    frames = {"capacity_existing_by_carrier": _metric_frame({"solar": 100.0}, {"solar": 180.0})}
    hotfixes = {
        "HF-8": {"id": "HF-8", "ported": True, "expect": ["capacity_existing_by_carrier/*"]},
        "HF-16": {"id": "HF-16", "ported": False, "expect": ["capacity_existing_by_carrier/*"]},
    }
    out = tables.comparison_table(frames, hotfixes)
    assert out.iloc[0]["verdict"] == "explained"
    assert out.iloc[0]["hotfix"] == "HF-16"


def test_waiver_hotfix_explains_a_row():
    """``explained`` may also come from a waiver carrying a ``hotfix:`` tag."""
    frames = {"dispatch_by_carrier": _metric_frame({"CCGT": 100.0}, {"CCGT": 140.0})}
    hotfixes = {"HF-14": {"id": "HF-14", "ported": False}}
    waivers = [{"metric": "dispatch_by_carrier", "key": "CCGT", "ledger": "DL-15", "hotfix": "HF-14"}]
    out = tables.comparison_table(frames, hotfixes, waivers)
    assert out.iloc[0]["verdict"] == "explained"
    assert out.iloc[0]["hotfix"] == "HF-14"


def test_waiver_without_a_hotfix_does_not_explain():
    frames = {"dispatch_by_carrier": _metric_frame({"CCGT": 100.0}, {"CCGT": 140.0})}
    waivers = [{"metric": "dispatch_by_carrier", "key": "CCGT", "ledger": "DL-15"}]
    out = tables.comparison_table(frames, {}, waivers)
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"


def test_comparison_table_accepts_the_objective_series():
    row = metrics.objective_row(
        type("S", (), {"objective": 100.0, "objective_constant": 10.0})(),
        type("S", (), {"objective": 110.0, "objective_constant": 0.0})(),
    )
    out = tables.comparison_table({"objective": row}, {})
    assert len(out) == 1
    assert out.iloc[0]["key"] == "total_system_cost"
    assert out.iloc[0]["verdict"] == "equivalent"
    assert out.iloc[0]["tolerance_pct"] == pytest.approx(0.1)


def test_appear_from_nothing_is_over_tolerance():
    frames = {"capacity_existing_by_carrier": _metric_frame({"solar": 0.0}, {"solar": 5.0})}
    out = tables.comparison_table(frames, {})
    assert np.isnan(out.iloc[0]["delta_pct"])
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"


def test_comparison_table_multiindex_key_is_flattened():
    idx = pd.MultiIndex.from_tuples([("p1", "solar")], names=["zone", "carrier"])
    df = pd.DataFrame(
        {"master": [100.0], "develop": [100.0], "delta": [0.0], "delta_pct": [0.0]},
        index=idx,
    )
    out = tables.comparison_table({"p_nom_existing_by_zone_carrier": df}, {})
    assert out.iloc[0]["key"] == "p1 | solar"


def test_comparison_table_empty_input():
    out = tables.comparison_table({}, {})
    assert out.empty
    assert list(out.columns) == tables.COMPARISON_COLUMNS
    assert tables.verdict_counts(out) == dict.fromkeys(tables.VERDICT_ORDER, 0)
    assert tables.n_failing(out) == 0


def test_write_tables_roundtrip(tmp_path):
    frames = {
        "capacity_existing_by_carrier": _metric_frame(
            {"solar": 100.0, "CCGT": 100.0},
            {"solar": 100.0, "CCGT": 180.0},
        ),
    }
    comparison = tables.comparison_table(frames, {})
    written = tables.write_tables({**frames, "comparison": comparison}, tmp_path)
    names = {p.name for p in written}
    assert names == {"capacity_existing_by_carrier.csv", "comparison.csv", "comparison.md"}
    # An empty ``hotfix`` cell round-trips through CSV as NaN; that is the
    # only representational difference, and it is restored here so the rest of
    # the frame is compared strictly.
    reloaded = pd.read_csv(tmp_path / "tables" / "comparison.csv", dtype={"hotfix": str})
    reloaded["hotfix"] = reloaded["hotfix"].fillna("")
    pd.testing.assert_frame_equal(reloaded, comparison, check_dtype=False)
    md = (tmp_path / "tables" / "comparison.md").read_text()
    assert "UNEXPLAINED" in md
    assert "| metric | key |" in md


def test_write_tables_markdown_is_capped(tmp_path):
    n = tables.MD_ROW_CAP + 7
    frames = {
        "capacity_existing_by_carrier": _metric_frame(
            {f"c{i}": 100.0 for i in range(n)},
            {f"c{i}": 100.0 + i for i in range(n)},
        ),
    }
    comparison = tables.comparison_table(frames, {})
    tables.write_tables({"comparison": comparison}, tmp_path)
    md = (tmp_path / "tables" / "comparison.md").read_text()
    assert "7 more rows in comparison.csv" in md
    assert md.count("\n| ") <= tables.MD_ROW_CAP + 2


def test_load_hotfixes_missing_file(tmp_path):
    assert tables.load_hotfixes(tmp_path / "nope.yaml") == {}


def test_load_hotfixes_reads_the_list_form(tmp_path):
    p = tmp_path / "hotfixes.yaml"
    p.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "HF-1",
                    "commit": "34b643f",
                    "pr": 801,
                    "title": "PUDL release pin",
                    "confidence": "high",
                    "ported": False,
                },
            ],
        ),
    )
    out = tables.load_hotfixes(p)
    assert set(out) == {"HF-1"}
    assert out["HF-1"]["pr"] == 801
    assert out["HF-1"]["ported"] is False


def test_repo_hotfixes_file_loads_if_present():
    """T4 owns the file; this only checks the loader agrees with it once it lands."""
    from pathlib import Path

    p = Path(tables.__file__).parent / "hotfixes.yaml"
    out = tables.load_hotfixes(p)
    if not out:
        pytest.skip("hotfixes.yaml not written yet (T4)")
    assert all(k.startswith("HF-") for k in out)
    assert all("ported" in e for e in out.values())


# ---------------------------------------------------------------------------
# Verifier defect 1: a cell waiver must not explain every row in the table.
# ---------------------------------------------------------------------------


def _real_waivers():
    """The shape every entry in waivers.yaml actually has."""
    from pathlib import Path

    import yaml as _yaml

    return _yaml.safe_load((Path(tables.__file__).parent / "waivers.yaml").read_text()) or []


def test_cell_waiver_shape_never_explains_a_table_row():
    """waivers.yaml entries key on stage/component/column/kind, never metric/key.

    Treating their absent metric/key/family as wildcards let a single
    hotfix:-tagged cell waiver explain every over-tolerance row in every family.
    """
    frames = {"capacity_existing_by_carrier": _metric_frame({"solar": 100.0}, {"solar": 180.0})}
    waiver = {"stage": "*", "component": "Bus", "column": "control", "kind": "value", "hotfix": "HF-8"}
    hotfixes = {"HF-8": {"id": "HF-8", "ported": False}}
    out = tables.comparison_table(frames, hotfixes, [waiver])
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"
    assert out.iloc[0]["hotfix"] == ""


def test_real_waivers_file_explains_nothing_in_the_table():
    frames = {
        "capacity_existing_by_carrier": _metric_frame({"solar": 100.0}, {"solar": 180.0}),
        "dispatch_by_carrier": _metric_frame({"CCGT": 100.0}, {"CCGT": 180.0}),
    }
    hotfixes = {f"HF-{i}": {"id": f"HF-{i}", "ported": False} for i in range(1, 22)}
    out = tables.comparison_table(frames, hotfixes, _real_waivers())
    assert set(out["verdict"]) == {"UNEXPLAINED"}


def test_waiver_naming_only_the_family_explains_that_family():
    frames = {"dispatch_by_carrier": _metric_frame({"CCGT": 100.0}, {"CCGT": 180.0})}
    hotfixes = {"HF-14": {"id": "HF-14", "ported": False}}
    out = tables.comparison_table(frames, hotfixes, [{"family": "dispatch", "hotfix": "HF-14"}])
    assert out.iloc[0]["verdict"] == "explained"


def test_waiver_naming_a_different_metric_does_not_explain():
    frames = {"dispatch_by_carrier": _metric_frame({"CCGT": 100.0}, {"CCGT": 180.0})}
    hotfixes = {"HF-14": {"id": "HF-14", "ported": False}}
    out = tables.comparison_table(frames, hotfixes, [{"metric": "demand_by_zone", "hotfix": "HF-14"}])
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"


# ---------------------------------------------------------------------------
# Verifier defect 2: an id must resolve in the registry; no bare-family globs.
# ---------------------------------------------------------------------------


def test_unknown_hotfix_id_does_not_explain():
    frames = {"capacity_existing_by_carrier": _metric_frame({"solar": 100.0}, {"solar": 180.0})}
    waivers = [{"metric": "capacity_existing_by_carrier", "key": "solar", "hotfix": "HF-99"}]
    out = tables.comparison_table(frames, {}, waivers)
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"
    # The wording is hotfixes.explains'; assert the substance, not the string.
    assert "HF-99" in out.iloc[0]["hotfix"]
    assert "no row" in out.iloc[0]["hotfix"]


def test_empty_registry_explains_nothing():
    frames = {"capacity_existing_by_carrier": _metric_frame({"solar": 100.0}, {"solar": 180.0})}
    waivers = [{"metric": "capacity_existing_by_carrier", "key": "solar", "hotfix": "HF-8"}]
    out = tables.comparison_table(frames, {}, waivers)
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"


def test_bare_family_expect_glob_does_not_match():
    """``expect: [capacity]`` would claim every capacity metric at once."""
    frames = {"capacity_existing_by_carrier": _metric_frame({"solar": 100.0}, {"solar": 180.0})}
    hotfixes = {"HF-8": {"id": "HF-8", "ported": False, "expect": ["capacity"]}}
    out = tables.comparison_table(frames, hotfixes)
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"


def test_metric_name_expect_glob_matches():
    frames = {"capacity_existing_by_carrier": _metric_frame({"solar": 100.0}, {"solar": 180.0})}
    hotfixes = {"HF-8": {"id": "HF-8", "ported": False, "expect": ["capacity_*"]}}
    out = tables.comparison_table(frames, hotfixes)
    assert out.iloc[0]["verdict"] == "explained"
    assert out.iloc[0]["hotfix"] == "HF-8"


# ---------------------------------------------------------------------------
# Verifier defect 3: an absolute floor, so solver noise is not a difference.
# ---------------------------------------------------------------------------


def test_solver_noise_on_an_unbuilt_carrier_is_equivalent():
    """1e-7 MW on a carrier neither side built is noise, not a -100 % delta."""
    frames = {"capacity_opt_by_carrier": _metric_frame({"coal": 1e-7}, {"coal": 0.0})}
    out = tables.comparison_table(frames, {})
    assert out.iloc[0]["delta_pct"] == pytest.approx(-100.0)
    assert out.iloc[0]["verdict"] == "equivalent"


def test_appear_from_nothing_below_the_floor_is_equivalent():
    frames = {"capacity_opt_by_carrier": _metric_frame({"coal": 0.0}, {"coal": 1e-4})}
    out = tables.comparison_table(frames, {})
    assert np.isnan(out.iloc[0]["delta_pct"])
    assert out.iloc[0]["verdict"] == "equivalent"


def test_appear_from_nothing_above_the_floor_is_unexplained():
    frames = {"capacity_opt_by_carrier": _metric_frame({"coal": 0.0}, {"coal": 500.0})}
    out = tables.comparison_table(frames, {})
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"


def test_absolute_floor_does_not_swallow_a_real_difference():
    frames = {"capacity_opt_by_carrier": _metric_frame({"coal": 100.0}, {"coal": 180.0})}
    out = tables.comparison_table(frames, {})
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"


def test_objective_has_no_absolute_floor():
    """A dollar of objective must not be excused by a MW-sized floor."""
    assert tables.TOLERANCES["objective"].atol == 0.0
    frames = {"objective": _metric_frame({"total": 0.0}, {"total": 0.5})}
    out = tables.comparison_table(frames, {})
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"


def test_tolerance_abs_is_reported():
    frames = {"capacity_opt_by_carrier": _metric_frame({"coal": 100.0}, {"coal": 100.0})}
    out = tables.comparison_table(frames, {})
    assert out.iloc[0]["tolerance_abs"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Verifier defect 4: ratio metrics keep NaN; NaN is not "equivalent, 0 vs 0".
# ---------------------------------------------------------------------------


def test_nan_on_both_sides_is_undefined_not_equivalent():
    df = metrics.frame(
        pd.Series({"coal": np.nan}), pd.Series({"coal": np.nan}), name="carrier", fill=None,
    )
    out = tables.comparison_table({"capacity_factor_by_carrier": df}, {})
    assert out.iloc[0]["verdict"] == "undefined"


def test_one_sided_nan_is_flagged_not_minus_one_hundred_percent():
    df = metrics.frame(
        pd.Series({"coal": np.nan}), pd.Series({"coal": 0.4}), name="carrier", fill=None,
    )
    out = tables.comparison_table({"capacity_factor_by_carrier": df}, {})
    assert out.iloc[0]["verdict"] == "one-sided"
    assert out.iloc[0]["hotfix"] == "develop only"


def test_one_sided_and_undefined_do_not_fail_the_run():
    df = metrics.frame(
        pd.Series({"a": np.nan, "b": np.nan}), pd.Series({"a": 0.4, "b": np.nan}),
        name="carrier", fill=None,
    )
    out = tables.comparison_table({"capacity_factor_by_carrier": df}, {})
    assert tables.n_failing(out) == 0
    assert tables.verdict_counts(out)["one-sided"] == 1
    assert tables.verdict_counts(out)["undefined"] == 1


# ---------------------------------------------------------------------------
# Verifier defect 5: a metric that raises becomes a MISSING row, not a skip.
# ---------------------------------------------------------------------------


def test_missing_metric_becomes_a_missing_row():
    missing = [{"metric": "p_nom_existing_by_zone_carrier", "reason": "ValueError: no reeds_zone"}]
    out = tables.comparison_table({}, {}, None, missing)
    assert len(out) == 1
    assert out.iloc[0]["verdict"] == "MISSING"
    assert out.iloc[0]["metric"] == "p_nom_existing_by_zone_carrier"
    assert "reeds_zone" in out.iloc[0]["hotfix"]


def test_missing_rows_fail_the_run_and_sort_first():
    frames = {"capacity_existing_by_carrier": _metric_frame({"solar": 100.0}, {"solar": 100.0})}
    missing = [{"metric": "demand_by_zone", "reason": "ValueError: boom"}]
    out = tables.comparison_table(frames, {}, None, missing)
    assert out.iloc[0]["verdict"] == "MISSING"
    assert tables.n_failing(out) == 1
    assert tables.verdict_counts(out)["MISSING"] == 1


def test_missing_metric_with_an_unknown_name_still_lands():
    out = tables.comparison_table({}, {}, None, [{"metric": "nobody_assigned_this", "reason": "x"}])
    assert out.iloc[0]["verdict"] == "MISSING"
    assert np.isnan(out.iloc[0]["tolerance_pct"])


def test_verdict_counts_covers_every_verdict():
    assert set(tables.verdict_counts(pd.DataFrame(columns=tables.COMPARISON_COLUMNS))) == set(
        tables.VERDICT_ORDER,
    )


# ---------------------------------------------------------------------------
# Integration with T4's hot-fix registry and run-directory layout.
# ---------------------------------------------------------------------------


def test_load_hotfixes_delegates_to_the_registry_module():
    """One loader, not two: the table layer must not re-read hotfixes.yaml."""
    from tests.equivalence import hotfixes as hf

    assert tables.load_hotfixes() == hf.load_hotfixes()


def test_registry_ships_and_is_well_formed():
    reg = tables.load_hotfixes()
    assert reg, "hotfixes.yaml should have landed with T4"
    assert all(k.startswith("HF-") for k in reg)
    assert all("ported" in e for e in reg.values())


def test_ported_registry_row_cannot_explain_a_real_difference():
    """Against the shipped registry rows, not a hand-made one.

    Each ported row is put in play alone, so nothing else in the registry can
    explain the difference and the verdict is attributable to that row only.
    """
    from tests.equivalence import hotfixes as hf

    reg = tables.load_hotfixes()
    ported = sorted(hf.ported_ids(reg))
    assert ported, "some rows should be marked ported after T1"
    frames = {"capacity_existing_by_carrier": _metric_frame({"solar": 100.0}, {"solar": 180.0})}
    for hid in ported:
        waivers = [{"metric": "capacity_existing_by_carrier", "key": "solar", "hotfix": hid}]
        out = tables.comparison_table(frames, {hid: reg[hid]}, waivers)
        assert out.iloc[0]["verdict"] == "UNEXPLAINED", hid
        assert "ported" in out.iloc[0]["hotfix"], hid


def test_no_ported_id_is_ever_credited_as_an_explanation():
    """Across the whole shipped registry and every metric it claims."""
    from tests.equivalence import hotfixes as hf

    reg = tables.load_hotfixes()
    ported = set(hf.ported_ids(reg))
    frames = {
        m: _metric_frame({"solar": 100.0}, {"solar": 180.0})
        for m in tables.KNOWN_METRICS
        if m != "objective"
    }
    out = tables.comparison_table(frames, reg)
    credited = {
        hid
        for cell, verdict in zip(out["hotfix"], out["verdict"])
        if verdict == "explained"
        for hid in str(cell).split(",")
        if hid
    }
    assert not (credited & ported), sorted(credited & ported)


def test_real_waivers_hotfix_tags_stay_out_of_the_table():
    """waivers.yaml carries `hotfix: HF-12` on cell waivers (T4).

    Those entries key on stage/component/column/kind. If the table treated their
    absent metric/key as wildcards, HF-12 — the pypsa stack migration, which is
    not ported — would mark rows 'explained' that nothing in the registry claims.
    The property asserted is exact: passing the real waivers changes no verdict.
    """
    tagged = [w for w in _real_waivers() if w.get("hotfix")]
    assert tagged, "T4 added hotfix: tags to waivers.yaml"
    assert all(
        all(w.get(k) is None for k in ("metric", "key", "family")) for w in tagged
    ), "a waiver that names a metric/key/family is a table waiver and must be reviewed here"
    frames = {
        "capacity_existing_by_carrier": _metric_frame({"solar": 100.0}, {"solar": 180.0}),
        "dispatch_by_carrier": _metric_frame({"CCGT": 100.0}, {"CCGT": 180.0}),
        "demand_by_zone": _metric_frame({"p1": 100.0}, {"p1": 180.0}),
    }
    reg = tables.load_hotfixes()
    without = tables.comparison_table(frames, reg, [])
    with_waivers = tables.comparison_table(frames, reg, _real_waivers())
    pd.testing.assert_frame_equal(without, with_waivers)


def test_registry_expect_patterns_are_all_satisfiable():
    """A pattern that can never match any metric is dead configuration.

    T4's registry was written in the plan's ``capacity/*`` family vocabulary,
    which the matcher does not speak (a bare family must not claim every
    difference in it). Translating those to metric-name globs is only safe if
    something checks that each one still matches something.
    """
    dead = tables.unmatched_expect_patterns(tables.load_hotfixes())
    assert dead == {}, f"expect patterns that can never match: {dead}"


def test_known_metrics_covers_what_collect_metrics_produces():
    """KNOWN_METRICS must stay in step with the metric names plots emits."""
    from tests.equivalence import plots

    master, develop = make_network(), make_network()
    art = plots.Artifacts(
        prong=2, develop_root=Path("/nonexistent"), master_root=Path("/nonexistent"),
        n_master=master, n_develop=develop, solved_master=master, solved_develop=develop,
    )
    produced = set(plots.collect_metrics(art, []))
    assert produced <= set(tables.KNOWN_METRICS), sorted(produced - set(tables.KNOWN_METRICS))


def test_family_glob_in_the_old_vocabulary_is_rejected():
    assert not tables.pattern_is_satisfiable("capacity/*")
    assert not tables.pattern_is_satisfiable("capacity")
    assert tables.pattern_is_satisfiable("capacity_existing_by_carrier/*")
    assert tables.pattern_is_satisfiable("objective")


def test_export_all_writes_the_run_directory_tables(tmp_path, monkeypatch):
    """``tables.export_all(run_dir, ctx, result)`` is the hook run.py calls."""
    from tests.equivalence import plots

    frames = {"capacity_existing_by_carrier": _metric_frame({"solar": 100.0}, {"solar": 100.0})}
    monkeypatch.setattr(plots, "run_metrics", lambda prong: (None, frames, []))
    ctx = type("Ctx", (), {"prong": 2})()
    comparison = tables.export_all(tmp_path, ctx, {"findings": []})
    assert (tmp_path / "tables" / "comparison.csv").exists()
    assert (tmp_path / "tables" / "comparison.md").exists()
    assert (tmp_path / "tables" / "capacity_existing_by_carrier.csv").exists()
    assert tables.verdict_counts(comparison)["equivalent"] == 1
