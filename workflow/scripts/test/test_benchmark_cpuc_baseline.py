"""Tests for the CPUC Baseline Generator List capacity benchmark."""

import os
import sys

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from benchmark_cpuc_baseline import (
    CPUC_HEADER_ROW,
    CPUC_SHEET,
    build_comparison,
    cpuc_capacity_by_region_tech,
    filter_active,
    load_benchmark_regions,
    load_tech_map,
    model_capacity_by_region_tech,
    plot_deviation_heatmap,
    read_cpuc_baseline,
    reconcile_totals,
)

REPO_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "repo_data", "CPUC")
TECH_MAP_CSV = os.path.join(REPO_DATA, "servm_tech_map.csv")
REGION_MAP_CSV = os.path.join(REPO_DATA, "servm_benchmark_regions.csv")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tech_map():
    return load_tech_map(TECH_MAP_CSV)


@pytest.fixture(scope="module")
def region_maps():
    return load_benchmark_regions(REGION_MAP_CSV)


def cpuc_frame(rows):
    """Synthetic CPUC baseline frame with the workbook's column names."""
    return pd.DataFrame(
        rows,
        columns=["SERVM Region", "SERVM Tech Category", "Capmax MW", "Insvdt", "RetireDate"],
    ).astype({"Insvdt": "datetime64[ns]", "RetireDate": "datetime64[ns]"})


def plants_frame(rows):
    """Synthetic ``powerplants.csv`` frame with the columns the benchmark reads."""
    return pd.DataFrame(
        rows,
        columns=[
            "state",
            "balancing_authority_code_eia",
            "carrier",
            "prime_mover_code",
            "p_nom",
            "build_year",
            "operational_status",
            "generator_retirement_date",
            "current_planned_generator_operating_date",
        ],
    )


# --------------------------------------------------------------------------- #
# workbook layout
# --------------------------------------------------------------------------- #


def test_header_layout_parses(tmp_path):
    """The real workbook keeps row 1 blank and the header on row 2."""
    openpyxl = pytest.importorskip("openpyxl")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = CPUC_SHEET
    ws.append([])  # the blank first row that forces header=1
    ws.append(["Unit Name", "Capmax MW", "Insvdt", "RetireDate", "SERVM Region", "SERVM Tech Category"])
    ws.append(["CA_CC_1", 500.0, "2015-01-01", "2050-12-01", "PGE", "CC"])
    ws.append(["CA_BATT_1", 100.0, "2023-06-01", "2050-12-01", "SCE", "Battery_4h"])
    ws.append(["WY_WIND_1", 300.0, "2015-01-01", "2050-12-01", "PACE", "Wind"])
    path = tmp_path / "baseline.xlsx"
    wb.save(path)

    assert CPUC_HEADER_ROW == 1

    df = read_cpuc_baseline(str(path))
    # the non-California row is filtered out, the two CA rows survive
    assert len(df) == 2
    assert set(df["SERVM Region"]) == {"PGE", "SCE"}
    assert df["Capmax MW"].sum() == pytest.approx(600.0)
    assert pd.api.types.is_datetime64_any_dtype(df["Insvdt"])


def test_read_cpuc_baseline_rejects_a_layout_change(tmp_path):
    """A missing expected column must fail loudly rather than silently empty out."""
    openpyxl = pytest.importorskip("openpyxl")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = CPUC_SHEET
    ws.append([])
    ws.append(["Unit Name", "Capmax MW", "SERVM Region"])
    ws.append(["CA_CC_1", 500.0, "PGE"])
    path = tmp_path / "broken.xlsx"
    wb.save(path)

    with pytest.raises(ValueError, match="missing"):
        read_cpuc_baseline(str(path))


# --------------------------------------------------------------------------- #
# vintage / retirement filtering
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "horizon,expected",
    [(2030, 1), (2035, 1), (2040, 0)],
)
def test_insvdt_retiredate_filter_at_horizon(horizon, expected):
    """In service 2028, retiring 2038: present through 2035, gone by 2040."""
    df = cpuc_frame([["PGE", "CC", 500.0, "2028-01-01", "2038-06-01"]])
    assert len(filter_active(df, horizon, "Insvdt", "RetireDate")) == expected


def test_unit_not_yet_in_service_is_excluded():
    df = cpuc_frame([["PGE", "CC", 500.0, "2032-01-01", "2050-12-01"]])
    assert filter_active(df, 2030, "Insvdt", "RetireDate").empty
    assert len(filter_active(df, 2032, "Insvdt", "RetireDate")) == 1


def test_blank_retiredate_never_retires():
    df = cpuc_frame([["PGE", "CC", 500.0, "2010-01-01", None]])
    for horizon in (2030, 2040, 2050):
        assert len(filter_active(df, horizon, "Insvdt", "RetireDate")) == 1


def test_2050_sentinel_is_not_special_cased():
    """The workbook's 2050-12-01 'no retirement' placeholder retires in 2050."""
    df = cpuc_frame([["PGE", "CC", 500.0, "2010-01-01", "2050-12-01"]])
    assert len(filter_active(df, 2045, "Insvdt", "RetireDate")) == 1
    assert filter_active(df, 2050, "Insvdt", "RetireDate").empty


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #


def test_region_tech_pivot_shape(tech_map):
    df = cpuc_frame(
        [
            ["PGE", "CC", 500.0, "2010-01-01", None],
            ["PGE", "Solar_1Axis", 100.0, "2010-01-01", None],
            ["PGE", "Solar_Fixed", 50.0, "2010-01-01", None],
            ["SCE", "CC", 200.0, "2010-01-01", None],
            ["SCE", "Battery_4h", 300.0, "2010-01-01", None],
        ],
    )
    out = cpuc_capacity_by_region_tech(df, tech_map["cpuc"], 2030)

    assert set(out.columns) == {"region", "compare_category", "cpuc_mw"}
    # PGE: Gas CC + Solar; SCE: Gas CC + Battery
    assert len(out) == 4
    pivot = out.pivot(index="region", columns="compare_category", values="cpuc_mw")
    # the two solar buckets collapse into one comparison category
    assert pivot.loc["PGE", "Solar"] == pytest.approx(150.0)
    assert pivot.loc["SCE", "Battery"] == pytest.approx(300.0)


def test_caiso_regions_aggregate(tech_map, region_maps):
    """PGE + SCE + SDGE roll up to a single CAISO row; the rest stay separate."""
    servm_regions, _ = region_maps
    df = cpuc_frame(
        [
            ["PGE", "CC", 500.0, "2010-01-01", None],
            ["SCE", "CC", 200.0, "2010-01-01", None],
            ["SDGE", "CC", 100.0, "2010-01-01", None],
            ["LADWP", "CC", 400.0, "2010-01-01", None],
            ["NCNC", "CC", 30.0, "2010-01-01", None],
            ["IID", "CC", 20.0, "2010-01-01", None],
        ],
    )
    out = cpuc_capacity_by_region_tech(df, tech_map["cpuc"], 2030, region_map=servm_regions)
    by_region = out.set_index("region").cpuc_mw

    assert set(by_region.index) == {"CAISO", "LADWP", "NCNC", "IID"}
    assert by_region["CAISO"] == pytest.approx(800.0)
    assert by_region["LADWP"] == pytest.approx(400.0)


def test_unknown_servm_region_raises(tech_map, region_maps):
    servm_regions, _ = region_maps
    df = cpuc_frame([["PACE", "CC", 500.0, "2010-01-01", None]])
    with pytest.raises(ValueError, match="absent from the benchmark region map"):
        cpuc_capacity_by_region_tech(df, tech_map["cpuc"], 2030, region_map=servm_regions)


# --------------------------------------------------------------------------- #
# model side
# --------------------------------------------------------------------------- #


def test_model_side_maps_ba_codes_and_splits_phs(tech_map, region_maps):
    _, ba_regions = region_maps
    plants = plants_frame(
        [
            ["CA", "CISO", "CCGT", "CA", 500.0, 2010, "existing", None, None],
            ["CA", "LDWP", "solar", "PV", 200.0, 2015, "existing", None, None],
            ["CA", "BANC", "hydro", "HY", 90.0, 1980, "existing", None, None],
            ["CA", "BANC", "hydro", "PS", 60.0, 1980, "existing", None, None],
            ["CA", "TIDC", "OCGT", "GT", 40.0, 2000, "existing", None, None],
            # Nevada CISO-VEA plant: CISO code, but not California
            ["NV", "CISO", "solar", "PV", 999.0, 2015, "existing", None, None],
        ],
    )
    out = model_capacity_by_region_tech(plants, ba_regions, tech_map["pypsa"], 2030)
    keyed = out.set_index(["region", "compare_category"]).model_mw

    assert keyed[("CAISO", "Gas CC")] == pytest.approx(500.0)
    assert keyed[("LADWP", "Solar")] == pytest.approx(200.0)
    # BANC and TIDC both roll up to NCNC
    assert keyed[("NCNC", "Hydro")] == pytest.approx(90.0)
    assert keyed[("NCNC", "PSH")] == pytest.approx(60.0)
    assert keyed[("NCNC", "Gas CT/ICE/Steam")] == pytest.approx(40.0)
    # the Nevada plant never reaches a CAISO row
    assert ("CAISO", "Solar") not in keyed.index


def test_model_side_mirrors_load_powerplants_retirement(tech_map, region_maps):
    """Existing units honor their announced retirement date, as add_electricity does."""
    _, ba_regions = region_maps
    plants = plants_frame(
        [
            # announced (planned) 2035 retirement, operational_status == existing
            ["CA", "CISO", "CCGT", "CA", 500.0, 2010, "existing", None, None],
            # already retired, with the date kept
            ["CA", "CISO", "coal", "ST", 300.0, 1980, "retired", "2025-01-01", None],
            # proposed: build year comes from the planned operating date
            ["CA", "CISO", "solar", "PV", 100.0, None, "proposed", None, "2032-01-01"],
            # existing with no announcement: survives indefinitely
            ["CA", "CISO", "OCGT", "GT", 50.0, 2001, "existing", None, None],
        ],
    )
    plants["planned_generator_retirement_date"] = ["2035-01-01", None, None, None]

    at_2040 = model_capacity_by_region_tech(plants, ba_regions, tech_map["pypsa"], 2040)
    keyed = at_2040.set_index("compare_category").model_mw

    assert "Gas CC" not in keyed.index  # honors its announced 2035 retirement
    assert "Coal" not in keyed.index  # retired 2025
    assert keyed["Solar"] == pytest.approx(100.0)  # proposed unit online 2032
    assert keyed["Gas CT/ICE/Steam"] == pytest.approx(50.0)  # no announcement -> survives

    at_2030 = model_capacity_by_region_tech(plants, ba_regions, tech_map["pypsa"], 2030)
    keyed_2030 = at_2030.set_index("compare_category").model_mw
    assert keyed_2030["Gas CC"] == pytest.approx(500.0)  # still online before the announced date
    assert "Solar" not in keyed_2030.index


def test_model_side_reports_out_of_scope_balancing_areas(tech_map, region_maps):
    """A California plant in a non-benchmark BA is surfaced, not dropped."""
    _, ba_regions = region_maps
    plants = plants_frame([["CA", "WALC", "solar", "PV", 75.0, 2015, "existing", None, None]])
    out = model_capacity_by_region_tech(plants, ba_regions, tech_map["pypsa"], 2030)

    assert list(out.region) == ["UNMAPPED:WALC"]
    assert out.model_mw.sum() == pytest.approx(75.0)


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #


def test_unmapped_tech_reported_not_dropped(tech_map, region_maps):
    """CPUC-only DR and an unknown model carrier both survive into the output."""
    servm_regions, ba_regions = region_maps

    cpuc = cpuc_capacity_by_region_tech(
        cpuc_frame(
            [
                ["PGE", "CC", 500.0, "2010-01-01", None],
                ["PGE", "DR", 250.0, "2010-01-01", None],
                ["PGE", "Fusion_Reactor", 42.0, "2010-01-01", None],
            ],
        ),
        tech_map["cpuc"],
        2030,
        region_map=servm_regions,
    )
    model = model_capacity_by_region_tech(
        plants_frame(
            [
                ["CA", "CISO", "CCGT", "CA", 500.0, 2010, "existing", None, None],
                ["CA", "CISO", "unobtainium", "XX", 7.0, 2010, "existing", None, None],
            ],
        ),
        ba_regions,
        tech_map["pypsa"],
        2030,
    )
    out = build_comparison(cpuc, model, 2030)
    keyed = out.set_index("compare_category")

    # CPUC-only demand response is kept with a zero model side
    assert keyed.loc["Demand Response", "cpuc_mw"] == pytest.approx(250.0)
    assert keyed.loc["Demand Response", "model_mw"] == pytest.approx(0.0)
    # an unrecognised label on either side gets its own explicit row
    assert keyed.loc["UNMAPPED:Fusion_Reactor", "cpuc_mw"] == pytest.approx(42.0)
    assert keyed.loc["UNMAPPED:unobtainium", "model_mw"] == pytest.approx(7.0)
    # nothing vanished
    assert out.cpuc_mw.sum() == pytest.approx(792.0)
    assert out.model_mw.sum() == pytest.approx(507.0)


def test_delta_pct_zero_when_identical(tech_map, region_maps):
    servm_regions, ba_regions = region_maps
    cpuc = cpuc_capacity_by_region_tech(
        cpuc_frame(
            [
                ["PGE", "CC", 500.0, "2010-01-01", None],
                ["LADWP", "Solar_1Axis", 200.0, "2010-01-01", None],
            ],
        ),
        tech_map["cpuc"],
        2030,
        region_map=servm_regions,
    )
    model = model_capacity_by_region_tech(
        plants_frame(
            [
                ["CA", "CISO", "CCGT", "CA", 500.0, 2010, "existing", None, None],
                ["CA", "LDWP", "solar", "PV", 200.0, 2010, "existing", None, None],
            ],
        ),
        ba_regions,
        tech_map["pypsa"],
        2030,
    )
    out = build_comparison(cpuc, model, 2030)

    assert len(out) == 2
    assert (out.delta_mw.abs() < 1e-9).all()
    assert (out.delta_pct.abs() < 1e-9).all()
    assert (out.horizon == 2030).all()


def test_delta_pct_is_nan_for_model_only_categories(tech_map, region_maps):
    servm_regions, ba_regions = region_maps
    cpuc = cpuc_capacity_by_region_tech(
        cpuc_frame([["PGE", "CC", 500.0, "2010-01-01", None]]),
        tech_map["cpuc"],
        2030,
        region_map=servm_regions,
    )
    model = model_capacity_by_region_tech(
        plants_frame([["CA", "CISO", "oil", "IC", 25.0, 2010, "existing", None, None]]),
        ba_regions,
        tech_map["pypsa"],
        2030,
    )
    out = build_comparison(cpuc, model, 2030).set_index("compare_category")

    assert pd.isna(out.loc["Oil", "delta_pct"])
    assert out.loc["Oil", "delta_mw"] == pytest.approx(25.0)
    assert out.loc["Gas CC", "delta_pct"] == pytest.approx(-100.0)


def test_reconcile_totals_labels_both_sides_without_borrowing_cpuc_mw(tech_map, region_maps):
    """The network/fleet cross-check must not masquerade as a CPUC comparison."""
    import pypsa

    _, ba_regions = region_maps
    n = pypsa.Network()
    n.add("Bus", "b1")
    n.add("Generator", "g1", bus="b1", carrier="CCGT", p_nom=400.0)
    n.add("StorageUnit", "s1", bus="b1", carrier="battery", p_nom=120.0)

    fleet = model_capacity_by_region_tech(
        plants_frame(
            [
                ["CA", "CISO", "CCGT", "CA", 500.0, 2010, "existing", None, None],
                ["CA", "CISO", "battery", "BA", 100.0, 2020, "existing", None, None],
            ],
        ),
        ba_regions,
        tech_map["pypsa"],
        2030,
    )
    out = reconcile_totals(n, fleet, tech_map["pypsa"])
    keyed = out.set_index(["region", "compare_category"]).model_mw

    assert keyed[("RECONCILE: fleet table (CA)", "Gas CC")] == pytest.approx(500.0)
    assert keyed[("RECONCILE: network total", "Gas CC")] == pytest.approx(400.0)
    assert keyed[("RECONCILE: network total", "Battery")] == pytest.approx(120.0)
    # the fleet table's Battery entry is 100 MW; both sides carry every category
    assert keyed[("RECONCILE: fleet table (CA)", "Battery")] == pytest.approx(100.0)
    # the CPUC is not one of the two things compared here
    assert (out.cpuc_mw == 0.0).all()


def test_reconcile_totals_zero_fills_one_sided_categories(tech_map, region_maps):
    """A category present in only one of the two totals still gets both rows."""
    import pypsa

    _, ba_regions = region_maps
    n = pypsa.Network()
    n.add("Bus", "b1")
    n.add("Generator", "g1", bus="b1", carrier="CCGT", p_nom=400.0)

    fleet = model_capacity_by_region_tech(
        plants_frame([["CA", "CISO", "solar", "PV", 300.0, 2010, "existing", None, None]]),
        ba_regions,
        tech_map["pypsa"],
        2030,
    )
    keyed = reconcile_totals(n, fleet, tech_map["pypsa"]).set_index(["region", "compare_category"]).model_mw

    assert keyed[("RECONCILE: network total", "Solar")] == pytest.approx(0.0)
    assert keyed[("RECONCILE: fleet table (CA)", "Gas CC")] == pytest.approx(0.0)


def test_plot_deviation_heatmap_excludes_reconciliation_rows(tmp_path):
    """Reconciliation pseudo-regions must not appear on the CPUC-deviation scale."""
    df = pd.DataFrame(
        {
            "horizon": [2030, 2030],
            "region": ["CAISO", "RECONCILE: network total"],
            "compare_category": ["Gas CC", "Gas CC"],
            "cpuc_mw": [500.0, 0.0],
            "model_mw": [450.0, 400.0],
            "delta_mw": [-50.0, float("nan")],
            "delta_pct": [-10.0, float("nan")],
        },
    )
    path = tmp_path / "deviation.pdf"
    plot_deviation_heatmap(df, str(path))
    assert path.exists() and path.stat().st_size > 0


def test_plot_deviation_heatmap_writes_one_panel_per_horizon(tmp_path):
    df = pd.DataFrame(
        {
            "horizon": [2030, 2030, 2040, 2040],
            "region": ["CAISO", "LADWP", "CAISO", "LADWP"],
            "compare_category": ["Gas CC", "Solar", "Gas CC", "Solar"],
            "cpuc_mw": [500.0, 200.0, 500.0, 200.0],
            "model_mw": [450.0, 260.0, 400.0, 300.0],
            "delta_mw": [-50.0, 60.0, -100.0, 100.0],
            "delta_pct": [-10.0, 30.0, -20.0, 50.0],
        },
    )
    path = tmp_path / "deviation.pdf"
    plot_deviation_heatmap(df, str(path))
    assert path.exists() and path.stat().st_size > 0


# --------------------------------------------------------------------------- #
# committed mapping files
# --------------------------------------------------------------------------- #


def test_repo_tech_map_covers_every_pypsa_carrier(tech_map):
    """Every carrier build_powerplants can emit has a comparison category."""
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    import constants as const

    carriers = set(pd.DataFrame(const.EIA_TECH_MAP).tech_type.unique())
    missing = sorted(carriers - set(tech_map["pypsa"]))
    assert not missing, f"powerplants.csv carriers with no compare_category: {missing}"


# --------------------------------------------------------------------------- #
# out-of-state exclusion
# --------------------------------------------------------------------------- #


def test_out_of_state_split_and_stale_warning(tmp_path, caplog):
    import logging

    from benchmark_cpuc_baseline import load_out_of_state_units, split_out_of_state

    cpuc = pd.DataFrame(
        {
            "Unit Name": ["PVERDE_5_SCEDYN", "DIABLO_7_UNIT_1", "Apex_CC1a"],
            "Capmax MW": [635.0, 1150.0, 398.0],
        },
    )
    csv = tmp_path / "oos.csv"
    csv.write_text(
        "cpuc_unit_name,servm_region,compare_category,capmax_mw,physical_location,eia_plant_id,evidence\n"
        'PVERDE_5_SCEDYN,SCE,Nuclear,635,Arizona,6008,"share; (AZ, BA SRP)"\n'
        "Apex_CC1a,LADWP,Gas CC,398,Nevada,,apex\n"
        "RENAMED_UNIT,PGE,Solar,10,Arizona,,gone from workbook\n",
    )
    with caplog.at_level(logging.WARNING):
        names = load_out_of_state_units(str(csv), cpuc)
    assert names == {"PVERDE_5_SCEDYN", "Apex_CC1a", "RENAMED_UNIT"}
    assert "not found in the CPUC workbook" in caplog.text

    in_scope, excluded = split_out_of_state(cpuc, names)
    assert list(in_scope["Unit Name"]) == ["DIABLO_7_UNIT_1"]
    assert set(excluded["Unit Name"]) == {"PVERDE_5_SCEDYN", "Apex_CC1a"}
