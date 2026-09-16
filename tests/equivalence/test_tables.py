"""Fast unit tests for ``tests.equivalence.tables``."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.equivalence import compare, hotfixes, metrics, tables

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
    waivers = [{"metric": "capacity_existing_by_carrier", "key": "onwind", "hotfix": "HF-8"}]
    out = tables.comparison_table(frames, hotfixes, waivers)
    assert list(out.columns) == tables.COMPARISON_COLUMNS
    verdicts = dict(zip(out["key"], out["verdict"]))
    assert verdicts["solar"] == "equivalent"
    assert verdicts["onwind"] == "explained"
    assert verdicts["CCGT"] == "UNEXPLAINED"
    # UNEXPLAINED first, then |delta_pct| descending.
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"
    assert out.iloc[0]["key"] == "CCGT"
    assert dict(zip(out["key"], out["hotfix"]))["onwind"] == "HF-8"
    # expect is advisory: it names candidates, it does not grant the verdict.
    assert dict(zip(out["key"], out["candidates"]))["onwind"] == "HF-8"
    assert (out["tolerance_pct"] == tables.TOLERANCES["capacity"].rtol * 100.0).all()
    assert (out["tolerance_abs"] == tables.TOLERANCES["capacity"].atol).all()


def test_ported_hotfix_is_not_an_explanation():
    """A fix on ``master-benchmark`` is on BOTH sides, so it explains nothing."""
    frames = {"capacity_existing_by_carrier": _metric_frame({"solar": 100.0}, {"solar": 180.0})}
    hotfixes = {"HF-8": {"id": "HF-8", "ported": True, "expect": ["capacity_existing_by_carrier/*"]}}
    waivers = [{"metric": "capacity_existing_by_carrier", "key": "solar", "hotfix": "HF-8"}]
    out = tables.comparison_table(frames, hotfixes, waivers)
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"
    assert "ported" in out.iloc[0]["hotfix"]
    assert "HF-8" in out.iloc[0]["hotfix"]


def test_usa_noop_hotfix_is_not_an_explanation():
    """HF-12 is inert on the standing USA config, so it explains nothing there."""
    frames = {"dispatch_by_carrier": _metric_frame({"CCGT": 100.0}, {"CCGT": 180.0})}
    hotfixes = {"HF-12": {"id": "HF-12", "ported": False, "usa_noop": True}}
    waivers = [{"metric": "dispatch_by_carrier", "key": "CCGT", "hotfix": "HF-12"}]
    out = tables.comparison_table(frames, hotfixes, waivers)
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"
    assert "no-op" in out.iloc[0]["hotfix"]


def test_the_waiver_decides_which_hotfix_explains(monkeypatch):
    """Two candidates, one waiver: the waiver's id is the one that counts."""
    frames = {"capacity_existing_by_carrier": _metric_frame({"solar": 100.0}, {"solar": 180.0})}
    hotfixes = {
        "HF-8": {"id": "HF-8", "ported": True, "expect": ["capacity_existing_by_carrier/*"]},
        "HF-16": {"id": "HF-16", "ported": False, "expect": ["capacity_existing_by_carrier/*"]},
    }
    waivers = [{"metric": "capacity_existing_by_carrier", "key": "solar", "hotfix": "HF-16"}]
    out = tables.comparison_table(frames, hotfixes, waivers)
    assert out.iloc[0]["verdict"] == "explained"
    assert out.iloc[0]["hotfix"] == "HF-16"
    # Both still show up as candidates, ported one included: if the difference
    # really is HF-8's, the port is what to check.
    assert out.iloc[0]["candidates"] == "HF-8,HF-16"


def test_waiver_hotfix_explains_a_row():
    """``explained`` may also come from a waiver carrying a ``hotfix:`` tag."""
    frames = {"dispatch_by_carrier": _metric_frame({"CCGT": 100.0}, {"CCGT": 140.0})}
    hotfixes = {"HF-14": {"id": "HF-14", "ported": False}}
    waivers = [{"metric": "dispatch_by_carrier", "key": "CCGT", "ledger": "DL-15", "hotfix": "HF-14"}]
    out = tables.comparison_table(frames, hotfixes, waivers)
    assert out.iloc[0]["verdict"] == "explained"
    assert out.iloc[0]["hotfix"] == "HF-14"


def test_waiver_scoped_to_another_run_does_not_explain():
    """A western prong-1 waiver must not sign off a USA prong-2 difference.

    ``compare.is_waived`` has always scoped cell waivers by ``interconnect`` and
    ``prong``; the table did not, so a waiver written for the deferred western
    leg reached forward and explained a whole-USA row it never saw.
    """
    frames = {"dispatch_by_carrier": _metric_frame({"CCGT": 100.0}, {"CCGT": 140.0})}
    hotfixes = {"HF-5": {"id": "HF-5", "ported": False}}
    waivers = [
        {
            "interconnect": "western",
            "prong": 1,
            "metric": "dispatch_by_carrier",
            "key": "CCGT",
            "ledger": "DL-1",
            "hotfix": "HF-5",
        },
    ]
    out = tables.comparison_table(frames, hotfixes, waivers, interconnect="usa", prong=2)
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"

    # The run it WAS written for still gets it.
    same = tables.comparison_table(frames, hotfixes, waivers, interconnect="western", prong=1)
    assert same.iloc[0]["verdict"] == "explained"
    assert same.iloc[0]["hotfix"] == "HF-5"


@pytest.mark.parametrize(
    ("scope", "explains"),
    [
        ({}, True),  # unscoped: any run
        ({"interconnect": "*", "prong": "*"}, True),  # explicit wildcards
        ({"interconnect": "usa"}, True),  # matches
        ({"prong": 2}, True),  # matches
        ({"interconnect": "western"}, False),  # wrong footprint
        ({"prong": 1}, False),  # wrong prong
        ({"interconnect": "usa", "prong": 1}, False),  # one field wrong is enough
    ],
)
def test_waiver_scope_fields(scope, explains):
    frames = {"dispatch_by_carrier": _metric_frame({"CCGT": 100.0}, {"CCGT": 140.0})}
    hotfixes = {"HF-5": {"id": "HF-5", "ported": False}}
    waivers = [{**scope, "metric": "dispatch_by_carrier", "key": "CCGT", "hotfix": "HF-5"}]
    out = tables.comparison_table(frames, hotfixes, waivers, interconnect="usa", prong=2)
    assert (out.iloc[0]["verdict"] == "explained") is explains


def test_scoped_waiver_is_refused_when_the_run_is_unknown():
    """A scope that cannot be checked has not been shown to apply.

    ``comparison_table`` without ``interconnect``/``prong`` does not know what
    run it is describing, so a waiver that demands a particular one is refused
    rather than assumed to match. An UNSCOPED waiver still applies.
    """
    frames = {"dispatch_by_carrier": _metric_frame({"CCGT": 100.0}, {"CCGT": 140.0})}
    hotfixes = {"HF-5": {"id": "HF-5", "ported": False}}
    scoped = [
        {"interconnect": "western", "prong": 1, "metric": "dispatch_by_carrier", "hotfix": "HF-5"},
    ]
    assert tables.comparison_table(frames, hotfixes, scoped).iloc[0]["verdict"] == "UNEXPLAINED"

    unscoped = [{"metric": "dispatch_by_carrier", "hotfix": "HF-5"}]
    assert tables.comparison_table(frames, hotfixes, unscoped).iloc[0]["verdict"] == "explained"


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
    # An empty ``hotfix`` or ``candidates`` cell round-trips through CSV as NaN;
    # that is the only representational difference, and it is restored here so
    # the rest of the frame is compared strictly.
    reloaded = pd.read_csv(
        tmp_path / "tables" / "comparison.csv",
        dtype={"hotfix": str, "candidates": str},
    )
    for col in ("hotfix", "candidates"):
        reloaded[col] = reloaded[col].fillna("")
    pd.testing.assert_frame_equal(reloaded, comparison, check_dtype=False)
    md = (tmp_path / "tables" / "comparison.md").read_text()
    assert "UNEXPLAINED" in md
    assert "| metric | key |" in md


#: One prong-2 run whose only difference is a develop-only cluster, plus a
#: live available-power finding whose waiver's bounds broke. The cluster is the
#: case this whole section exists for: master is rolled up to the common set, so
#: `p_nom_max_by_zone_solar p8` reads EQUIVALENT while 2,158 MW sits on one side.
_DISCLOSURE_RESULT = {
    "prong": 2,
    "profile_cluster_sets": [
        {
            "kind": "profile_rollup",
            "stage": "profile_solar",
            "rolled_up": True,
            "n_master": 19,
            "n_develop": 20,
            "n_common": 19,
            "only_master": {},
            "only_develop": {"p87 0": 2158.0},
            "only_master_mw": 0.0,
            "only_develop_mw": 2158.0,
            "common_total_mw": 1_800_000.0,
            "equal": False,
        },
    ],
    "findings": [
        {
            "stage": "profile_solar",
            "component": "cluster_set",
            "column": "<index>",
            "kind": "row_set",
            "waived": True,
            "waiver": "HF-24",
            "detail": {
                "n_master": 19,
                "n_develop": 20,
                "n_common": 19,
                "only_develop": {"p87 0": 2158.0},
                "only_master": {},
                "common_total_mw": 1_800_000.0,
            },
        },
        {
            "stage": "profile_solar",
            "component": "system_potential_mw",
            "column": "sum(p_nom_max)",
            "kind": "value",
            "waived": True,
            "waiver": "HF-24",
            "detail": {"develop": 1_802_566.0, "master": 1_800_000.0, "rel_pct": 0.1426},
        },
        {
            "stage": "profile_onwind",
            "component": "system_available_mw",
            "column": "sum_bus(profile*p_nom_max)",
            "kind": "value",
            "waived": False,
            "waiver_note": "waiver HF-24 bounds violated (max_total_pct)",
            "waiver_bound_failed": "max_total_pct",
            "detail": {
                "develop_total_mwh": 310_000_000.0,
                "master_total_mwh": 620_823_223.0,
                "total_rel_pct": -50.07,
                "energy_weighted_mean_rel_pct": 50.07,
                "hours_mismatched": 8371,
                "hours_compared": 8760,
            },
        },
    ],
}


def _disclosure_md(tmp_path, result) -> str:
    frames = {"capacity_existing_by_carrier": _metric_frame({"solar": 100.0}, {"solar": 100.0})}
    comparison = tables.comparison_table(frames, {})
    written = tables.write_tables({**frames, "comparison": comparison}, tmp_path, result=result)
    assert "findings.csv" in {p.name for p in written}
    return (tmp_path / "tables" / "comparison.md").read_text()


def test_comparison_md_discloses_a_develop_only_cluster(tmp_path):
    """The 2,158 MW cluster is named, with its MW, in comparison.md itself.

    It moves no comparison-table row — master was rolled up to the common
    clusters — so before the Findings section it existed only in
    findings_2.json and run_meta.json, neither of which a human reads.
    """
    md = _disclosure_md(tmp_path, _DISCLOSURE_RESULT)
    assert "## Findings" in md
    assert "p87 0 (2,158 MW)" in md
    assert (
        "Profile metrics use the 19 common clusters (profile_solar); "
        "one-sided clusters: develop-only p87 0 (2,158 MW) (see Findings)." in md
    )
    # The verdict per finding, including WHICH waiver covered it.
    assert "waived HF-24" in md
    assert "LIVE (bounds violated: max_total_pct)" in md
    # The potential finding reports both totals and the relative move.
    assert "develop 1,802,566 MW vs master 1,800,000 MW (+0.1426 %)" in md


def test_findings_csv_is_the_markdown_sections_twin(tmp_path):
    _disclosure_md(tmp_path, _DISCLOSURE_RESULT)
    fcsv = pd.read_csv(tmp_path / "tables" / "findings.csv")
    assert list(fcsv.columns) == list(tables.FINDINGS_COLUMNS)
    assert len(fcsv) == len(_DISCLOSURE_RESULT["findings"])
    assert sorted(fcsv["verdict"]) == sorted(
        ["LIVE (bounds violated: max_total_pct)", "waived HF-24", "waived HF-24"],
    )
    assert "p87 0 (2,158 MW)" in fcsv.loc[fcsv["component"] == "cluster_set", "detail"].iloc[0]


def test_findings_section_is_present_even_when_there_are_none(tmp_path):
    """An empty Findings section is a statement the run makes, not a section that vanished."""
    md = _disclosure_md(tmp_path, {"prong": 2, "findings": [], "profile_cluster_sets": []})
    assert "## Findings" in md
    assert "No stage-by-stage findings were recorded" in md
    assert "one-sided clusters" not in md
    assert pd.read_csv(tmp_path / "tables" / "findings.csv").empty


def test_a_matching_cluster_set_says_so(tmp_path):
    """A rollup that found no one-sided cluster still says which population it used."""
    result = {
        "prong": 2,
        "findings": [],
        "profile_cluster_sets": [
            {
                "kind": "profile_rollup",
                "stage": "profile_onwind",
                "rolled_up": True,
                "n_master": 18,
                "n_develop": 18,
                "n_common": 18,
                "only_master": {},
                "only_develop": {},
                "equal": True,
            },
        ],
    }
    md = _disclosure_md(tmp_path, result)
    assert "Profile metrics use the 18 common clusters (profile_onwind); one-sided clusters: none" in md


def test_the_note_falls_back_to_the_cluster_set_findings():
    """An older findings JSON has no ``profile_cluster_sets``; the note still renders."""
    result = {"findings": [_DISCLOSURE_RESULT["findings"][0]]}
    assert tables.cluster_note_lines(result) == [
        "Profile metrics use the 19 common clusters (profile_solar); "
        "one-sided clusters: develop-only p87 0 (2,158 MW) (see Findings).",
    ]


def test_findings_markdown_cannot_be_broken_by_a_pipe_or_swallowed_as_html():
    """``|`` must not split the row, and ``<index>`` must not render as nothing."""
    rows = tables.findings_markdown(
        {
            "findings": [
                {"stage": "s|1", "component": "Bus", "column": "<index>", "kind": "value", "detail": "a|b"},
            ],
        },
    )
    row = next(line for line in rows if line.startswith("| s\\|1 "))
    assert "\\|" in row
    assert row.replace("\\|", "").count("|") == len(tables.FINDINGS_COLUMNS) + 1
    assert "&lt;index&gt;" in row
    # The CSV twin keeps the raw value: it is data, not markdown.
    assert tables.findings_rows({"findings": [{"column": "<index>"}]})[0]["column"] == "<index>"


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


def _stub_reconstruction(
    rows: dict[tuple[str, str], dict],
    name: str = "hf26_existing_renewable_drop",
    tol=None,
):
    """A ``{name: Reconstruction}`` mapping built from four numbers per row.

    What ``tables.py`` is being tested on is the CONTRACT it holds a
    reconstruction to, not whether artifacts can be opened. The gates and the
    residual are still computed by ``reconstructions._make_row``, the same
    function the real loader uses, so a stub cannot pass a row the real thing
    would fail.
    """
    from tests.equivalence import reconstructions as recon_mod

    made: dict[tuple[str, str], object] = {}
    for (metric, key), v in rows.items():
        # HF-26 keys a zone row "p10 | onwind" and a national row "onwind";
        # HF-24's potential metric is already per tech, so its key is the bare
        # zone. A row may say which it is rather than be guessed at.
        if "zone" in v or "carrier" in v:
            zone, carrier = v.get("zone"), v.get("carrier", "")
        elif " | " in key:
            zone, carrier = key.split(" | ", 1)
        else:
            zone, carrier = None, key
        row = recon_mod._make_row(
            metric,
            zone,
            carrier,
            v["fleet_mw"],
            v["profiled_mw"],
            v["master_mw"],
            v["develop_mw"],
            tol or tables.tolerance_for(metric),
            key=key,
        )
        made[(metric, row.key)] = row
    return {name: recon_mod.Reconstruction(name=name, frame=recon_mod.rows_frame(made), rows=made)}


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
    assert out.iloc[0]["candidates"] == ""


def test_metric_name_expect_glob_only_names_a_candidate():
    """A matching expect glob fills `candidates`, and changes nothing else."""
    frames = {"capacity_existing_by_carrier": _metric_frame({"solar": 100.0}, {"solar": 180.0})}
    hotfixes = {"HF-8": {"id": "HF-8", "ported": False, "expect": ["capacity_*"]}}
    out = tables.comparison_table(frames, hotfixes)
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"
    assert out.iloc[0]["hotfix"] == ""
    assert out.iloc[0]["candidates"] == "HF-8"


def test_candidates_are_sorted_numerically():
    frames = {"dispatch_by_carrier": _metric_frame({"CCGT": 100.0}, {"CCGT": 180.0})}
    hotfixes = {f"HF-{i}": {"id": f"HF-{i}", "ported": False, "expect": ["dispatch_by_carrier/*"]} for i in (2, 13, 7)}
    out = tables.comparison_table(frames, hotfixes)
    assert out.iloc[0]["candidates"] == "HF-2,HF-7,HF-13"


def test_the_shipped_registry_cannot_explain_anything_on_its_own():
    """THE regression guard for the expect-is-advisory decision.

    In the metric-name vocabulary the shipped `expect` globs cover every one of
    KNOWN_METRICS between them. While `expect` granted verdicts, that made
    UNEXPLAINED unreachable and the failing verdict could never fire — a safety
    net that catches everything is not a safety net. A +50 % row in every family
    must come back UNEXPLAINED against the real registry and the real waivers.
    """
    from tests.equivalence.compare import load_waivers

    registry = tables.load_hotfixes()
    waivers = load_waivers()
    frames = {
        metric: _metric_frame({"probe": 100.0}, {"probe": 150.0})
        for metric in tables.KNOWN_METRICS
        if metric != "objective"
    }
    out = tables.comparison_table(frames, registry, waivers)
    assert not out.empty
    assert set(out["verdict"]) == {"UNEXPLAINED"}, dict(zip(out["metric"], out["verdict"]))
    # ...and the advisory column is doing its job, so the information is not lost.
    assert out["candidates"].str.len().gt(0).any()


def test_a_waiver_with_a_hotfix_flips_exactly_one_row():
    """The other half of the same guard: attribution still works, per row."""
    from tests.equivalence.compare import load_waivers

    registry = tables.load_hotfixes()
    frames = {
        metric: _metric_frame({"probe": 100.0}, {"probe": 150.0})
        for metric in ("capacity_existing_by_carrier", "dispatch_by_carrier")
    }
    # HF-5 is unported and not a USA no-op in the shipped registry.
    waivers = [
        *load_waivers(),
        {"metric": "capacity_existing_by_carrier", "key": "probe", "ledger": "DL-1", "hotfix": "HF-5"},
    ]
    out = tables.comparison_table(frames, registry, waivers).set_index("metric")
    assert out.loc["capacity_existing_by_carrier", "verdict"] == "explained"
    assert out.loc["capacity_existing_by_carrier", "hotfix"] == "HF-5"
    assert out.loc["dispatch_by_carrier", "verdict"] == "UNEXPLAINED"


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
        pd.Series({"coal": np.nan}),
        pd.Series({"coal": np.nan}),
        name="carrier",
        fill=None,
    )
    out = tables.comparison_table({"capacity_factor_by_carrier": df}, {})
    assert out.iloc[0]["verdict"] == "undefined"


def test_one_sided_nan_is_flagged_not_minus_one_hundred_percent():
    df = metrics.frame(
        pd.Series({"coal": np.nan}),
        pd.Series({"coal": 0.4}),
        name="carrier",
        fill=None,
    )
    out = tables.comparison_table({"capacity_factor_by_carrier": df}, {})
    assert out.iloc[0]["verdict"] == "one-sided"
    assert out.iloc[0]["hotfix"] == "develop only"


def test_one_sided_and_undefined_do_not_fail_the_run():
    df = metrics.frame(
        pd.Series({"a": np.nan, "b": np.nan}),
        pd.Series({"a": 0.4, "b": np.nan}),
        name="carrier",
        fill=None,
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
    frames = {m: _metric_frame({"solar": 100.0}, {"solar": 180.0}) for m in tables.KNOWN_METRICS if m != "objective"}
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
    The property asserted is exact: passing the real waivers changes no verdict
    on metrics they do not name.

    ``waivers.yaml`` also holds genuine TABLE waivers (HF-24's
    ``p_nom_max_by_zone_*`` rows). Those are legitimate and are checked here for
    the two things that make them safe: they name a metric the table can
    actually produce, and their reach is bounded to what someone measured —
    either by ``interconnect``/``prong`` scoping, or by a ``reconstruction:``
    that recomputes the bound per row and is therefore its own scope.

    The property asserted on the numbers is that the real waivers change no
    VERDICT on metrics they do not explain. It is verdicts, not whole frames:
    the unscoped HF-26 reconstruction waivers do name
    ``capacity_existing_by_carrier``, so with no reconstruction supplied the
    ``hotfix`` cell now records "considered and refused" instead of staying
    blank. That is a disclosure, not an explanation, and the verdict is what
    decides whether a run passes.
    """
    tagged = [w for w in _real_waivers() if w.get("hotfix")]
    assert tagged, "T4 added hotfix: tags to waivers.yaml"
    cell = [w for w in tagged if all(w.get(k) is None for k in ("metric", "key", "family"))]
    table_waivers = [w for w in tagged if w not in cell]
    assert cell, "the HF-12 cell waivers must still be cell waivers"
    for w in table_waivers:
        assert w.get("metric") in tables.KNOWN_METRICS, w
        scoped = bool(w.get("interconnect")) and bool(w.get("prong"))
        assert scoped or w.get("reconstruction"), f"unscoped, uncomputed table waiver: {w}"
    frames = {
        "capacity_existing_by_carrier": _metric_frame({"solar": 100.0}, {"solar": 180.0}),
        "dispatch_by_carrier": _metric_frame({"CCGT": 100.0}, {"CCGT": 180.0}),
        "demand_by_zone": _metric_frame({"p1": 100.0}, {"p1": 180.0}),
    }
    reg = tables.load_hotfixes()
    without = tables.comparison_table(frames, reg, [])
    with_waivers = tables.comparison_table(frames, reg, _real_waivers())
    assert list(with_waivers["verdict"]) == list(without["verdict"])
    assert set(with_waivers["verdict"]) == {"UNEXPLAINED"}
    named = with_waivers[with_waivers["hotfix"] != ""]
    assert set(named["metric"]) <= {"capacity_existing_by_carrier"}
    assert all("bounds violated" in c for c in named["hotfix"]), list(named["hotfix"])


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
        prong=2,
        develop_root=Path("/nonexistent"),
        master_root=Path("/nonexistent"),
        n_master=master,
        n_develop=develop,
        solved_master=master,
        solved_develop=develop,
    )
    produced = set(plots.collect_metrics(art, []))
    assert produced <= set(tables.KNOWN_METRICS), sorted(produced - set(tables.KNOWN_METRICS))


def test_family_glob_in_the_old_vocabulary_is_rejected():
    assert not tables.pattern_is_satisfiable("capacity/*")
    assert not tables.pattern_is_satisfiable("capacity")
    assert tables.pattern_is_satisfiable("capacity_existing_by_carrier/*")
    assert tables.pattern_is_satisfiable("objective")


@pytest.mark.parametrize(
    "pattern",
    [
        "objective/total_system_cost",
        "dispatch_by_carrier/CCGT",
        "p_max_pu_quantiles_solar/0.5",
        "mean_cf_by_zone_*",
        "*_by_carrier",
    ],
)
def test_literal_metric_slash_key_patterns_are_satisfiable(pattern):
    """A row is addressed as ``metric`` or ``metric/key``; keys are open-ended.

    ``objective/total_system_cost`` is not a glob at all, and neither is
    ``dispatch_by_carrier/CCGT`` — both name rows this harness really produces,
    so the lint must not call them dead.
    """
    assert tables.pattern_is_satisfiable(pattern)


@pytest.mark.parametrize("pattern", ["*", "**", "*/*", "*/**", "**/*", "", "   "])
def test_vacuous_patterns_are_rejected(pattern):
    """A pattern that matches everything makes the emptiest claim there is.

    Accepting "this hot-fix can move any metric at all" would let bare-family
    looseness back in through the wildcard door, with the lint reporting
    nothing.
    """
    assert not tables.pattern_is_satisfiable(pattern)


@pytest.mark.parametrize("metric", tables.KNOWN_METRICS)
def test_every_known_metric_is_addressable(metric):
    """Each metric is nameable both bare and with a key."""
    assert tables.pattern_is_satisfiable(metric)
    assert tables.pattern_is_satisfiable(f"{metric}/some_key")


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


def _hf24_recon(master: float, develop: float, zone: str = "p8", tech: str = "onwind"):
    """A reconstruction that accounts for exactly ``develop - master`` in ``zone``."""
    return _stub_reconstruction(
        {
            (f"p_nom_max_by_zone_{tech}", zone): dict(
                fleet_mw=develop,
                profiled_mw=master,
                master_mw=master,
                develop_mw=develop,
                zone=zone,
                carrier=tech,
            ),
        },
        name="hf24_nrel_caps_drop",
    )


def test_hf24_table_waiver_explains_any_leg_the_reconstruction_covers():
    """The HF-24 waivers are unscoped now: the RECONSTRUCTION is the scope.

    They used to be ``interconnect: western, prong: 2`` with ``max_abs_pct: 5``,
    so the identical row on the usa leg stayed UNEXPLAINED — which is what left
    the whole-USA run with 46 onwind + 4 solar unexplained potential rows. The
    bound is now recomputed per row from the caps artifact, so the same waiver
    holds on any leg where the arithmetic holds, and on none where it does not.
    """
    frames = {"p_nom_max_by_zone_onwind": _metric_frame({"p8": 33348.0}, {"p8": 33444.0})}
    reg = tables.load_hotfixes()
    waivers = _real_waivers()
    recon = _hf24_recon(33348.0, 33444.0)

    for ic, prong in (("western", 2), ("usa", 2), ("western", 1)):
        out = tables.comparison_table(
            frames,
            reg,
            waivers,
            interconnect=ic,
            prong=prong,
            reconstructions=recon,
        )
        assert out.iloc[0]["verdict"] == "explained", (ic, prong)
        assert out.iloc[0]["hotfix"] == "HF-24"

    # And on no leg at all without the reconstruction that bounds it.
    for ic, prong in (("western", 2), ("usa", 2)):
        out = tables.comparison_table(frames, reg, waivers, interconnect=ic, prong=prong)
        assert out.iloc[0]["verdict"] == "UNEXPLAINED", (ic, prong)
        assert "reconstruction unavailable" in out.iloc[0]["hotfix"]


# ---------------------------------------------------------------------------
# A table waiver is bounded in sign and magnitude, or it is a blank cheque.
# ---------------------------------------------------------------------------


def _bounded_waiver(**extra) -> list[dict]:
    return [
        {
            "metric": "p_nom_max_by_zone_solar",
            "key": "p8",
            "ledger": "DL-18",
            "hotfix": "HF-24",
            "expect_sign": "+",
            "max_abs_pct": 5,
            **extra,
        },
    ]


def _solar_p8(master: float, develop: float) -> dict[str, pd.DataFrame]:
    return {"p_nom_max_by_zone_solar": _metric_frame({"p8": master}, {"p8": develop})}


def test_bounded_waiver_explains_a_row_inside_its_bounds():
    reg = {"HF-24": {"id": "HF-24", "ported": False, "usa_noop": False}}
    out = tables.comparison_table(_solar_p8(188425.0, 190583.0), reg, _bounded_waiver())
    assert out.iloc[0]["verdict"] == "explained"
    assert out.iloc[0]["hotfix"] == "HF-24"


def test_bounded_waiver_refuses_the_wrong_sign():
    """HF-24 RAISES develop's potential; a -50 % row is not that difference."""
    reg = {"HF-24": {"id": "HF-24", "ported": False, "usa_noop": False}}
    out = tables.comparison_table(_solar_p8(188425.0, 94212.5), reg, _bounded_waiver())
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"
    assert out.iloc[0]["hotfix"] == "waiver HF-24 bounds violated (sign)"


def test_bounded_waiver_refuses_an_out_of_scale_magnitude():
    reg = {"HF-24": {"id": "HF-24", "ported": False, "usa_noop": False}}
    out = tables.comparison_table(_solar_p8(188425.0, 1.88425e9), reg, _bounded_waiver())
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"
    assert out.iloc[0]["hotfix"] == "waiver HF-24 bounds violated (magnitude)"


def test_bounded_waiver_refuses_an_undefined_delta_pct():
    """Master 0, develop non-zero: |delta %| is undefined, so the bound fails."""
    reg = {"HF-24": {"id": "HF-24", "ported": False, "usa_noop": False}}
    out = tables.comparison_table(_solar_p8(0.0, 2158.0), reg, _bounded_waiver())
    assert np.isnan(out.iloc[0]["delta_pct"])
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"
    assert out.iloc[0]["hotfix"] == "waiver HF-24 bounds violated (magnitude)"


def test_an_unbounded_waiver_still_explains_whatever_it_names():
    """Bounds are optional; the pre-existing unbounded form is unchanged."""
    reg = {"HF-24": {"id": "HF-24", "ported": False, "usa_noop": False}}
    waivers = [{"metric": "p_nom_max_by_zone_solar", "key": "p8", "ledger": "DL-18", "hotfix": "HF-24"}]
    out = tables.comparison_table(_solar_p8(188425.0, 94212.5), reg, waivers)
    assert out.iloc[0]["verdict"] == "explained"


def test_each_bound_can_be_given_on_its_own():
    reg = {"HF-24": {"id": "HF-24", "ported": False, "usa_noop": False}}
    sign_only = _bounded_waiver()
    sign_only[0].pop("max_abs_pct")
    out = tables.comparison_table(_solar_p8(188425.0, 1.88425e9), reg, sign_only)
    assert out.iloc[0]["verdict"] == "explained", "magnitude is unbounded here"

    cap_only = _bounded_waiver()
    cap_only[0].pop("expect_sign")
    out = tables.comparison_table(_solar_p8(188425.0, 94212.5), reg, cap_only)
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"
    assert out.iloc[0]["hotfix"] == "waiver HF-24 bounds violated (magnitude)"


def test_the_shipped_hf24_waivers_are_bounded():
    """Every shipped HF-24 table waiver says which direction it explains.

    Two rows now, one per tech, bounded by a COMPUTED reconstruction instead of
    the four hand-measured ``max_abs_pct: 5`` rows that were scoped to western.
    The HF-24 CELL waivers (``system_potential_mw``, ``system_available_mw``,
    ``cluster_set``) are a different thing entirely and are not counted here.
    """
    bounded = [w for w in _real_waivers() if w.get("hotfix") == "HF-24" and w.get("metric")]
    assert bounded, "the HF-24 table waivers are gone"
    assert len(bounded) == 2, f"expected one waiver per tech, got {len(bounded)}"
    assert {w["metric"] for w in bounded} == {"p_nom_max_by_zone_onwind", "p_nom_max_by_zone_solar"}
    for w in bounded:
        assert w["expect_sign"] == "+", w
        assert w["reconstruction"] == "hf24_nrel_caps_drop", w
        assert w["ledger"] == "DL-18", w
        assert w.get("interconnect") is None and w.get("prong") is None, f"scoped by hand: {w}"
        assert w.get("key") is None, f"a key would defeat the point of a computed bound: {w}"
        assert "max_abs_pct" not in w, f"a hand bound beside the reconstruction downgrades it: {w}"


def test_the_shipped_hf24_cell_waivers_are_untouched():
    """The three bounded HF-24 COMPONENT waivers are not table rows.

    ``system_potential_mw``, ``system_available_mw`` and ``cluster_set`` are
    ``compare.py`` findings; the reconstruction has nothing to say about them and
    must not have taken them with it.
    """
    cells = [w for w in _real_waivers() if w.get("hotfix") == "HF-24" and not w.get("metric")]
    assert {w.get("component") for w in cells} == {
        "system_potential_mw",
        "system_available_mw",
        "cluster_set",
    }


def test_the_shipped_hf24_waiver_refuses_a_sign_flip():
    """End to end on the real files: a drop in develop's potential is not HF-24.

    The reconstruction here reproduces both sides exactly, so the gates hold and
    the residual is 0 — only ``expect_sign`` stands between this row and a
    verdict it must not get.
    """
    reg = tables.load_hotfixes()
    waivers = _real_waivers()
    flipped = tables.comparison_table(
        _solar_p8(188425.0, 94212.5),
        reg,
        waivers,
        interconnect="western",
        prong=2,
        reconstructions=_hf24_recon(188425.0, 94212.5, zone="p8", tech="solar"),
    )
    assert flipped.iloc[0]["verdict"] == "UNEXPLAINED"
    assert "bounds violated (sign)" in flipped.iloc[0]["hotfix"]


# ---------------------------------------------------------------------------
# max_abs_delta_mw: the bound for a row whose master side is 0.
# ---------------------------------------------------------------------------


def _zone_carrier_frame(master: float, develop: float, key: str = "p8 | onwind") -> dict[str, pd.DataFrame]:
    return {"p_nom_existing_by_zone_carrier": _metric_frame({key: master}, {key: develop})}


def _abs_bounded_waiver(**extra) -> list[dict]:
    return [
        {
            "metric": "p_nom_existing_by_zone_carrier",
            "key": "p8 | onwind",
            "ledger": "DL-21",
            "hotfix": "HF-27",
            "max_abs_delta_mw": 200,
            **extra,
        },
    ]


def test_max_abs_delta_mw_explains_an_appear_from_nothing_row():
    """0 -> 101 MW: |delta %| is undefined, but 101 MW is inside the bound.

    This is the row ``max_abs_pct`` cannot express. Before the absolute bound,
    HF-27's p8 onwind delta could only be waived unbounded (a blank cheque) or
    not at all.
    """
    reg = {"HF-27": {"id": "HF-27", "ported": False, "usa_noop": False}}
    out = tables.comparison_table(_zone_carrier_frame(0.0, 101.2), reg, _abs_bounded_waiver())
    assert np.isnan(out.iloc[0]["delta_pct"])
    assert out.iloc[0]["verdict"] == "explained"
    assert out.iloc[0]["hotfix"] == "HF-27"


def test_max_abs_delta_mw_refuses_an_out_of_scale_delta():
    reg = {"HF-27": {"id": "HF-27", "ported": False, "usa_noop": False}}
    out = tables.comparison_table(_zone_carrier_frame(0.0, 5_000.0), reg, _abs_bounded_waiver())
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"
    assert out.iloc[0]["hotfix"] == "waiver HF-27 bounds violated (magnitude)"


def test_max_abs_delta_mw_bounds_the_magnitude_in_both_directions():
    """It bounds ``|delta|``; the direction is ``expect_sign``'s job, if given."""
    reg = {"HF-27": {"id": "HF-27", "ported": False, "usa_noop": False}}
    assert (
        tables.comparison_table(_zone_carrier_frame(500.0, 400.0), reg, _abs_bounded_waiver()).iloc[0]["verdict"]
        == "explained"
    )
    signed = _abs_bounded_waiver(expect_sign="+")
    out = tables.comparison_table(_zone_carrier_frame(500.0, 400.0), reg, signed)
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"
    assert out.iloc[0]["hotfix"] == "waiver HF-27 bounds violated (sign)"


def test_both_magnitude_bounds_are_checked_when_both_are_given():
    """A percent bound that holds does not excuse an MW bound that does not."""
    reg = {"HF-27": {"id": "HF-27", "ported": False, "usa_noop": False}}
    both = _abs_bounded_waiver(max_abs_pct=100)
    # +50 % is inside max_abs_pct, but +5,000 MW is ten times max_abs_delta_mw.
    out = tables.comparison_table(_zone_carrier_frame(10_000.0, 15_000.0), reg, both)
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"
    assert out.iloc[0]["hotfix"] == "waiver HF-27 bounds violated (magnitude)"


def test_max_abs_delta_mw_is_a_table_bound_the_ledger_test_can_see():
    """The bound registry is what makes the key legal in waivers.yaml."""
    assert "max_abs_delta_mw" in tables.WAIVER_BOUND_KEYS
    assert "max_abs_delta_mw" not in compare.CELL_BOUND_KEYS


def test_the_shipped_hf26_waivers_are_bounded():
    """Every shipped HF-26 table waiver states what it measured.

    Six hand-measured ``max_abs_pct`` rows (western prong 2) were replaced by
    TWO ``reconstruction:`` rows, one per metric, whose bound is recomputed per
    row by ``reconstructions.hf26_existing_renewable_drop``. The six are deleted
    rather than kept beside them: a row that the reconstruction refuses must not
    fall back on a weaker percentage that happens to cover it.

    They are deliberately UNSCOPED — no ``interconnect``, no ``prong`` — because
    the reconstruction is the scope. On the USA leg the same difference reaches
    +28,620 % on rows where master is ~0 MW, which no percentage bound states.

    HF-27 used to contribute three more rows here (``p11 | solar``,
    ``p8 | onwind``, ``p8 | solar``). They were removed when the zone
    misassignment turned out to be a DEVELOP regression and was fixed — see
    :func:`test_no_shipped_waiver_still_cites_hf27`.
    """
    rows = [w for w in _real_waivers() if w.get("hotfix") in ("HF-26", "HF-27") and w.get("metric")]
    assert len(rows) == 2, f"expected the 2 shipped existing-capacity waivers, got {len(rows)}"
    assert {w["metric"] for w in rows} == {
        "capacity_existing_by_carrier",
        "p_nom_existing_by_zone_carrier",
    }
    for w in rows:
        assert w["ledger"] == "DL-20", w
        assert w.get("interconnect") is None and w.get("prong") is None, f"scoped by hand: {w}"
        assert w.get("key") is None, f"a key would defeat the point of a computed bound: {w}"
        assert w["reconstruction"] == "hf26_existing_renewable_drop", w
        assert w["expect_sign"] == "+", w
        assert "max_abs_pct" not in w, f"a hand bound beside the reconstruction downgrades it: {w}"


def test_no_shipped_waiver_still_cites_hf27():
    """The fix removes the need for them; a leftover would sign off a real bug.

    HF-27's per-zone deltas were develop matching plants to ``{simpl}`` cluster
    centroids instead of to substations. ``add_electricity`` now matches
    substations and maps through ``busmap_s{simpl}``, which reproduces master's
    zone for 841 of 841 western plants, so those rows must come back
    ``equivalent``. A waiver left behind would instead explain them away.
    """
    assert [w for w in _real_waivers() if w.get("hotfix") == "HF-26"], "HF-26 waivers vanished with HF-27's"
    assert not [w for w in _real_waivers() if w.get("hotfix") == "HF-27"]


def test_the_shipped_hf26_waiver_holds_at_the_measured_western_delta():
    """End to end on the real files: onwind 3,097.6 -> 6,554.5 MW is explained.

    The waiver now carries a reconstruction, so the table needs one: the
    western 2127 numbers are master 3,097.6 MW profiled out of a 6,554.5 MW
    reconstructed fleet, i.e. 3,456.9 MW dropped, which is exactly the delta.
    """
    reg = tables.load_hotfixes()
    frames = {"capacity_existing_by_carrier": _metric_frame({"onwind": 3097.6}, {"onwind": 6554.5})}
    recon = _stub_reconstruction(
        {
            ("capacity_existing_by_carrier", "onwind"): dict(
                fleet_mw=6554.5,
                profiled_mw=3097.6,
                master_mw=3097.6,
                develop_mw=6554.5,
            ),
        },
    )
    out = tables.comparison_table(
        frames,
        reg,
        _real_waivers(),
        interconnect="western",
        prong=2,
        reconstructions=recon,
    )
    assert out.iloc[0]["verdict"] == "explained"
    assert out.iloc[0]["hotfix"] == "HF-26"

    # HF-26 only ever ADDS capacity on develop; a loss is a different animal.
    flipped = {"capacity_existing_by_carrier": _metric_frame({"onwind": 6554.5}, {"onwind": 3097.6})}
    flipped_recon = _stub_reconstruction(
        {
            ("capacity_existing_by_carrier", "onwind"): dict(
                fleet_mw=3097.6,
                profiled_mw=6554.5,
                master_mw=6554.5,
                develop_mw=3097.6,
            ),
        },
    )
    out = tables.comparison_table(
        flipped,
        reg,
        _real_waivers(),
        interconnect="western",
        prong=2,
        reconstructions=flipped_recon,
    )
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"
    assert "bounds violated (sign)" in out.iloc[0]["hotfix"]

    # And with no reconstruction at all the row is not explained either.
    out = tables.comparison_table(frames, reg, _real_waivers(), interconnect="western", prong=2)
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"
    assert "reconstruction unavailable" in out.iloc[0]["hotfix"]


def test_the_western_p8_zone_row_is_no_longer_explained_by_hf27():
    """The row HF-27 used to waive now has to come back equivalent.

    0 -> 101.2 MW of onwind in p8 was develop matching those plants to a cluster
    centroid in the next zone. With the substation-first match it should not
    reappear at all; if it does, the shipped files must call it UNEXPLAINED
    rather than hand it HF-27's old signature. Both halves of that are checked:
    nothing explains the row, and HF-27 would be refused even if something tried
    (``usa_noop: true``).

    HF-26's reconstruction waiver does NAME this metric, so the cell is no longer
    blank: with no reconstruction supplied it reads "bounds violated
    (reconstruction unavailable)", which is the waiver being refused. That is the
    point of a computed bound. What must not appear is an ``explained`` verdict,
    or HF-27.
    """
    reg = tables.load_hotfixes()
    out = tables.comparison_table(
        _zone_carrier_frame(0.0, 101.2),
        reg,
        _real_waivers(),
        interconnect="western",
        prong=2,
    )
    assert out.iloc[0]["verdict"] == "UNEXPLAINED"
    assert "reconstruction unavailable" in out.iloc[0]["hotfix"]
    assert "HF-27" not in out.iloc[0]["hotfix"]

    ok, reason = hotfixes.explains("HF-27", reg)
    assert not ok and "no-op" in reason, reason


def test_a_zone_carrier_key_does_not_break_the_markdown_row():
    """``p8 | oil`` must stay ONE cell.

    ``_key_label`` flattens a (zone, carrier) MultiIndex with " | ", so every
    ``p_nom_existing_by_zone_carrier`` row carries a literal pipe. Unescaped, it
    ends the cell and shifts every number one column right of its heading — the
    table then reads as though 15 MW of oil were a tolerance.
    """
    frames = {"p_nom_existing_by_zone_carrier": _metric_frame({"p8 | oil": 0.0}, {"p8 | oil": 15.0})}
    table = tables.comparison_table(frames, {}, [])
    md = tables.to_markdown(table)
    row = next(line for line in md.splitlines() if line.startswith("| p_nom_existing_by_zone_carrier"))
    assert "p8 \\| oil" in row
    # header, metric, key, master, develop, delta, delta %, tol %, tol abs,
    # verdict, hot-fix, trailing -> 11 unescaped separators.
    assert len(re.findall(r"(?<!\\)\|", row)) == 11, row
