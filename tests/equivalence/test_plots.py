"""Fast unit tests for ``tests.equivalence.plots``.

Figures are rendered to an Agg backend under ``tmp_path``; nothing here reads
``data/`` or ``resources/`` or touches the network.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tests.equivalence import metrics, plots

from .conftest import make_network, make_objective, make_profile

pytestmark = pytest.mark.fast


def _metric_frame(master: dict[str, float], develop: dict[str, float]) -> pd.DataFrame:
    return metrics.frame(pd.Series(master), pd.Series(develop), name="carrier")


def _stems(outdir, suffix):
    return {p.stem for p in (outdir / "figures").glob(f"*{suffix}")}


def test_save_figure_writes_png_and_csv(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    data = _metric_frame({"solar": 1.0}, {"solar": 2.0})
    png, csv = plots.save_figure(fig, data, "demo", tmp_path)
    assert png.exists() and png.stat().st_size > 0
    assert csv.exists()
    assert png.stem == csv.stem
    reloaded = pd.read_csv(csv, index_col=0)
    assert reloaded.loc["solar", "develop"] == pytest.approx(2.0)


def test_paired_bar_produces_png_and_csv(tmp_path):
    df = _metric_frame({"solar": 100.0, "CCGT": 300.0}, {"solar": 110.0, "CCGT": 300.0})
    png, csv = plots.paired_bar(df, "existing capacity", "MW", "capacity_existing_by_carrier", tmp_path)
    assert png.exists() and csv.exists()
    assert pd.read_csv(csv, index_col=0).loc["solar", "delta_pct"] == pytest.approx(10.0)


def test_paired_bar_empty_frame(tmp_path):
    """A missing solve must produce a labelled placeholder, not an exception."""
    png, csv = plots.paired_bar(metrics.empty_frame(), "annual dispatch", "MWh", "dispatch_by_carrier", tmp_path)
    assert png.exists()
    assert csv.exists()
    assert png.stem == csv.stem


def test_paired_bar_none_frame(tmp_path):
    png, _ = plots.paired_bar(None, "annual dispatch", "MWh", "dispatch_by_carrier", tmp_path)
    assert png.exists()


def test_paired_bar_handles_appear_from_nothing(tmp_path):
    df = _metric_frame({"solar": 0.0}, {"solar": 5.0})
    png, csv = plots.paired_bar(df, "existing capacity", "MW", "cap", tmp_path)
    assert png.exists()
    assert np.isnan(pd.read_csv(csv, index_col=0).loc["solar", "delta_pct"])


def test_objective_figure_shows_raw_and_normalised(tmp_path):
    row = metrics.objective_row(make_objective(100.0, 10.0), make_objective(110.0, 0.0))
    png, csv = plots.objective_figure(row, "objective", tmp_path)
    assert png.exists()
    data = pd.read_csv(csv, index_col=0)["value"]
    assert data["master_raw"] == pytest.approx(100.0)
    assert data["develop_raw"] == pytest.approx(110.0)
    assert data["delta_pct"] == pytest.approx(0.0)


def test_objective_figure_without_a_solve(tmp_path):
    png, _ = plots.objective_figure(None, "objective", tmp_path)
    assert png.exists()


def _point_zones(names=("p1", "p2", "p3")):
    """Zone coordinates in the network's own x/y space (the no-shapes fallback)."""
    names = list(names)
    return pd.DataFrame(
        {"x": np.arange(len(names), dtype=float), "y": np.arange(len(names), dtype=float) * 0.5},
        index=names,
    )


def test_choropleth_triptych_point_map(tmp_path):
    zones = _point_zones()
    vm = pd.Series({"p1": 10.0, "p2": 20.0, "p3": 30.0})
    vd = pd.Series({"p1": 11.0, "p2": 20.0, "p3": 27.0})
    png, csv = plots.choropleth_triptych(zones, vm, vd, "potential", "GW", "potential_zones", tmp_path)
    assert png.exists()
    data = pd.read_csv(csv, index_col=0)
    assert data.loc["p1", "delta"] == pytest.approx(1.0)


def test_choropleth_handles_missing_zone(tmp_path):
    """A zone with no value is grey, not a raise."""
    zones = _point_zones()
    vm = pd.Series({"p1": 10.0, "p2": 20.0})  # p3 absent on both sides
    vd = pd.Series({"p1": 11.0, "p2": 20.0})
    png, csv = plots.choropleth_triptych(zones, vm, vd, "potential", "GW", "potential_zones", tmp_path)
    assert png.exists()
    assert "p3" not in pd.read_csv(csv, index_col=0).index


def test_choropleth_empty_input(tmp_path):
    png, _ = plots.choropleth_triptych(
        _point_zones(), pd.Series(dtype=float), pd.Series(dtype=float), "potential", "GW", "z", tmp_path,
    )
    assert png.exists()


def test_choropleth_geodataframe(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    zones = gpd.GeoDataFrame(
        {"geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]},
        index=["p1", "p2", "p3"],
        crs="EPSG:4326",
    )
    vm = pd.Series({"p1": 10.0, "p2": 20.0})
    vd = pd.Series({"p1": 12.0, "p2": 20.0})
    png, _ = plots.choropleth_triptych(zones, vm, vd, "potential", "GW", "geo_zones", tmp_path)
    assert png.exists()


def test_duration_curve(tmp_path):
    ds_m = make_profile(n_bus=3, n_time=24, seed=10)
    ds_d = make_profile(n_bus=5, n_time=24, seed=11, bus_prefix="d")
    png, csv = plots.duration_curve(
        pd.Series(metrics.profile_values(ds_m)),
        pd.Series(metrics.profile_values(ds_d)),
        "solar capacity factor", "-", "p_max_pu_duration_solar", tmp_path,
    )
    assert png.exists()
    data = pd.read_csv(csv, index_col=0)
    assert len(data) == 201
    assert data["master"].is_monotonic_decreasing


def test_duration_curve_empty(tmp_path):
    png, _ = plots.duration_curve(pd.Series(dtype=float), pd.Series(dtype=float), "cf", "-", "dc", tmp_path)
    assert png.exists()


def test_timeseries_pair(tmp_path):
    ds = make_profile(n_bus=3, n_time=48, seed=12)
    s = metrics.available_power(ds)
    png, csv = plots.timeseries_pair(s, s * 1.1, "available power", "MW", "solar_national_available_power", tmp_path)
    assert png.exists()
    data = pd.read_csv(csv, index_col=0)
    assert data["delta_pct"].round(3).eq(10.0).all()


def test_findings_by_stage(tmp_path):
    findings = [
        {"stage": "demand", "waived": False},
        {"stage": "demand", "waived": True},
        {"stage": "profile_solar", "waived": True},
    ]
    png, csv = plots.findings_by_stage(findings, "findings_by_stage", tmp_path)
    assert png.exists()
    counts = pd.read_csv(csv, index_col=0)
    assert counts.loc["demand", "live"] == 1
    assert counts.loc["demand", "waived"] == 1


def test_findings_by_stage_empty(tmp_path):
    png, _ = plots.findings_by_stage([], "findings_by_stage", tmp_path)
    assert png.exists()


def test_carrier_color_is_stable_and_entity_keyed():
    assert plots.carrier_color("solar") == plots.carrier_color("solar")
    assert plots.carrier_color("solar") == plots.carrier_color("SOLAR")
    assert plots.carrier_color("solar") != plots.carrier_color("coal")
    assert plots.carrier_color("a carrier nobody named").startswith("#")


def test_side_colors_are_fixed():
    assert plots.SIDE_COLORS["master"] != plots.SIDE_COLORS["develop"]
    assert set(plots.SIDE_COLORS) == {"master", "develop"}


def test_every_figure_has_a_csv(tmp_path):
    """The stem sets of the PNGs and the CSVs must be equal — no orphan figures."""
    master, develop = make_network(), make_network(scale=1.05)
    zones = _point_zones(names=("p1", "p2"))
    frames = {
        "capacity_existing_by_carrier": metrics.capacity_by_carrier(master, develop, attr="p_nom"),
        "capacity_opt_by_carrier": metrics.capacity_by_carrier(master, develop, attr="p_nom_opt"),
        "dispatch_by_carrier": metrics.dispatch_by_carrier(master, develop),
        "capacity_factor_by_carrier": metrics.capacity_factor_by_carrier(master, develop),
        "demand_by_zone": metrics.demand_by_zone(master, develop),
    }
    plots.objective_figure(
        metrics.objective_row(make_objective(100.0, 10.0), make_objective(110.0, 0.0)), "objective", tmp_path,
    )
    for name, title, unit in (
        ("capacity_existing_by_carrier", "existing capacity", "MW"),
        ("capacity_opt_by_carrier", "optimised capacity", "MW"),
        ("dispatch_by_carrier", "annual dispatch", "MWh"),
        ("capacity_factor_by_carrier", "realised capacity factor", "-"),
        ("demand_by_zone", "demand by zone", "MW"),
    ):
        plots.paired_bar(frames[name], title, unit, name if name != "demand_by_zone" else "demand_zones", tmp_path)
    zc = metrics.capacity_by_zone_carrier(master, develop, attr="p_nom")
    for carrier in ("solar", "CCGT"):
        sub = zc.xs(carrier, level="carrier")
        plots.choropleth_triptych(
            zones, sub["master"], sub["develop"], f"{carrier} existing capacity", "MW",
            f"p_nom_existing_zones_{carrier}", tmp_path,
        )
    ds_m, ds_d = make_profile(seed=20), make_profile(seed=21)
    for tech in ("solar", "onwind"):
        pot = metrics.p_nom_max_by_zone(
            ds_m, ds_d, pd.Series({"b0": "p1", "b1": "p1", "b2": "p2"}), pd.Series({"b0": "p1", "b1": "p1", "b2": "p2"}),
        )
        plots.choropleth_triptych(
            zones, pot["master"], pot["develop"], f"{tech} potential", "GW", f"{tech}_potential_zones", tmp_path,
        )
        cf = metrics.mean_cf_by_zone(
            ds_m, ds_d, pd.Series({"b0": "p1", "b1": "p1", "b2": "p2"}), pd.Series({"b0": "p1", "b1": "p1", "b2": "p2"}),
        )
        plots.choropleth_triptych(
            zones, cf["master"], cf["develop"], f"{tech} mean CF", "-", f"{tech}_meancf_zones", tmp_path,
        )
        plots.duration_curve(
            pd.Series(metrics.profile_values(ds_m)), pd.Series(metrics.profile_values(ds_d)),
            f"{tech} capacity factor", "-", f"p_max_pu_duration_{tech}", tmp_path,
        )
        plots.timeseries_pair(
            metrics.available_power(ds_m), metrics.available_power(ds_d),
            f"{tech} available power", "MW", f"{tech}_national_available_power", tmp_path,
        )
    plots.findings_by_stage([{"stage": "demand", "waived": False}], "findings_by_stage", tmp_path)

    pngs, csvs = _stems(tmp_path, ".png"), _stems(tmp_path, ".csv")
    assert pngs == csvs, f"PNG-only: {sorted(pngs - csvs)}; CSV-only: {sorted(csvs - pngs)}"
    required = {
        "objective",
        "capacity_existing_by_carrier",
        "capacity_opt_by_carrier",
        "dispatch_by_carrier",
        "p_max_pu_duration_solar",
        "p_nom_existing_zones_solar",
        "solar_potential_zones",
        "solar_meancf_zones",
        "solar_national_available_power",
        "findings_by_stage",
    }
    assert required <= pngs, f"missing required figures: {sorted(required - pngs)}"


def test_export_all_degrades_gracefully(tmp_path):
    """With nothing built, the export still produces a complete figure set.

    This is the orchestration path ``run.py`` takes; a half-built run must yield
    labelled placeholders rather than an exception, and the PNG/CSV stem sets
    must still match.
    """
    art = plots.Artifacts(prong=2, develop_root=tmp_path / "dev", master_root=tmp_path / "mas")
    figdir = plots.export_all(tmp_path, artifacts=art, metric_frames={}, findings=[])
    assert figdir == tmp_path / "figures"
    pngs, csvs = _stems(tmp_path, ".png"), _stems(tmp_path, ".csv")
    assert pngs == csvs
    assert {"objective", "dispatch_by_carrier", "findings_by_stage"} <= pngs
