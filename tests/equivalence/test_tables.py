"""Fast unit tests for ``tests.equivalence.tables``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.equivalence import compare, metrics, tables

pytestmark = pytest.mark.fast


def _metric_frame(master: dict[str, float], develop: dict[str, float]) -> pd.DataFrame:
    return metrics.frame(pd.Series(master), pd.Series(develop), name="carrier")


def test_tolerances_are_the_single_source_of_truth():
    """``compare.py`` must import them, not restate them."""
    assert compare.OBJECTIVE_RTOL is tables.TOLERANCES["objective"]
    assert compare.CAPACITY_RTOL is tables.TOLERANCES["capacity"]
    src = (__import__("pathlib").Path(compare.__file__)).read_text()
    assert "OBJECTIVE_RTOL = 1e-3" not in src
    assert "CAPACITY_RTOL = 5e-3" not in src


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
        "HF-8": {"id": "HF-8", "ported": False, "expect": ["capacity/onwind"]},
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
    assert (out["tolerance_pct"] == tables.TOLERANCES["capacity"] * 100.0).all()


def test_ported_hotfix_is_not_an_explanation():
    """A fix on ``master-benchmark`` is on BOTH sides, so it explains nothing."""
    frames = {"capacity_existing_by_carrier": _metric_frame({"solar": 100.0}, {"solar": 180.0})}
    hotfixes = {"HF-8": {"id": "HF-8", "ported": True, "expect": ["capacity/*"]}}
    out = tables.comparison_table(frames, hotfixes)
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"
    assert "ported" in out.iloc[0]["hotfix"]
    assert "HF-8" in out.iloc[0]["hotfix"]


def test_unported_hotfix_wins_over_a_ported_one():
    frames = {"capacity_existing_by_carrier": _metric_frame({"solar": 100.0}, {"solar": 180.0})}
    hotfixes = {
        "HF-8": {"id": "HF-8", "ported": True, "expect": ["capacity/*"]},
        "HF-16": {"id": "HF-16", "ported": False, "expect": ["capacity/*"]},
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
    assert tables.verdict_counts(out) == {"equivalent": 0, "explained": 0, "UNEXPLAINED": 0}


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
