"""Fast unit tests for ``tests.equivalence.plots``.

Figures are rendered to an Agg backend under ``tmp_path``; nothing here reads
``data/`` or ``resources/`` or touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

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
        _point_zones(),
        pd.Series(dtype=float),
        pd.Series(dtype=float),
        "potential",
        "GW",
        "z",
        tmp_path,
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
        "solar capacity factor",
        "-",
        "p_max_pu_duration_solar",
        tmp_path,
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
        metrics.objective_row(make_objective(100.0, 10.0), make_objective(110.0, 0.0)),
        "objective",
        tmp_path,
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
            zones,
            sub["master"],
            sub["develop"],
            f"{carrier} existing capacity",
            "MW",
            f"p_nom_existing_zones_{carrier}",
            tmp_path,
        )
    ds_m, ds_d = make_profile(seed=20), make_profile(seed=21)
    for tech in ("solar", "onwind"):
        pot = metrics.p_nom_max_by_zone(
            ds_m,
            ds_d,
            pd.Series({"b0": "p1", "b1": "p1", "b2": "p2"}),
            pd.Series({"b0": "p1", "b1": "p1", "b2": "p2"}),
        )
        plots.choropleth_triptych(
            zones,
            pot["master"],
            pot["develop"],
            f"{tech} potential",
            "GW",
            f"{tech}_potential_zones",
            tmp_path,
        )
        cf = metrics.mean_cf_by_zone(
            ds_m,
            ds_d,
            pd.Series({"b0": "p1", "b1": "p1", "b2": "p2"}),
            pd.Series({"b0": "p1", "b1": "p1", "b2": "p2"}),
        )
        plots.choropleth_triptych(
            zones,
            cf["master"],
            cf["develop"],
            f"{tech} mean CF",
            "-",
            f"{tech}_meancf_zones",
            tmp_path,
        )
        plots.duration_curve(
            pd.Series(metrics.profile_values(ds_m)),
            pd.Series(metrics.profile_values(ds_d)),
            f"{tech} capacity factor",
            "-",
            f"p_max_pu_duration_{tech}",
            tmp_path,
        )
        plots.timeseries_pair(
            metrics.available_power(ds_m),
            metrics.available_power(ds_d),
            f"{tech} available power",
            "MW",
            f"{tech}_national_available_power",
            tmp_path,
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


def test_timeseries_pair_short_series_keeps_native_resolution(tmp_path):
    """A 24-hour fixture resamples to one daily point, which plots nothing."""
    ds = make_profile(n_bus=2, n_time=24, seed=30)
    s = metrics.available_power(ds)
    png, csv = plots.timeseries_pair(s, s * 1.05, "available power", "MW", "short_series", tmp_path)
    assert png.exists()
    assert len(pd.read_csv(csv, index_col=0)) == 24


# ---------------------------------------------------------------------------
# Verifier defect 5: a metric that raises is recorded, never silently skipped.
# ---------------------------------------------------------------------------


def test_missing_reeds_zone_is_recorded_not_skipped():
    """capacity_by_zone_carrier raising must not delete the criterion.

    Before this, ``_safe`` swallowed the ValueError, the geographic-assignment
    criterion vanished from comparison.csv and the run still exited 0.
    """
    from tests.equivalence import tables

    master, develop = make_network(), make_network()
    for n in (master, develop):
        n.buses.drop(columns=["reeds_zone"], inplace=True)
    art = plots.Artifacts(
        prong=2,
        develop_root=Path("/nonexistent/dev"),
        master_root=Path("/nonexistent/mas"),
        n_master=master,
        n_develop=develop,
    )
    missing = []
    frames = plots.collect_metrics(art, missing)
    names = {e["metric"] for e in missing}
    assert "p_nom_existing_by_zone_carrier" in names
    assert "demand_by_zone" in names
    assert "p_nom_existing_by_zone_carrier" not in frames
    assert any("reeds_zone" in e["reason"] for e in missing)

    table = tables.comparison_table(frames, {}, None, missing)
    verdicts = dict(zip(table["metric"], table["verdict"]))
    assert verdicts["p_nom_existing_by_zone_carrier"] == "MISSING"
    assert tables.n_failing(table) >= 1


def test_collect_metrics_records_nothing_when_all_is_well():
    art = plots.Artifacts(
        prong=2,
        develop_root=Path("/nonexistent/dev"),
        master_root=Path("/nonexistent/mas"),
        n_master=make_network(),
        n_develop=make_network(),
    )
    missing = []
    frames = plots.collect_metrics(art, missing)
    assert missing == []
    assert "p_nom_existing_by_zone_carrier" in frames


def test_export_all_writes_missing_metrics_json(tmp_path):
    art = plots.Artifacts(prong=2, develop_root=tmp_path / "d", master_root=tmp_path / "m")
    missing = [{"metric": "demand_by_zone", "reason": "ValueError: boom"}]
    plots.export_all(tmp_path, artifacts=art, metric_frames={}, findings=[], missing=missing)
    written = json.loads((tmp_path / "missing_metrics.json").read_text())
    assert written == missing


def test_run_metrics_is_memoised_so_the_networks_load_once(monkeypatch, tmp_path):
    """tables.export_all and plots.export_all must not each read the networks."""
    calls = []

    def fake_load(prong, develop_root, master_root):
        calls.append(prong)
        return plots.Artifacts(prong=prong, develop_root=Path(develop_root), master_root=Path(master_root))

    monkeypatch.setattr(plots, "load_artifacts", fake_load)
    monkeypatch.setattr(plots, "_RUN_METRICS", {})
    a1, m1, x1 = plots.run_metrics(2, tmp_path / "d", tmp_path / "m")
    a2, m2, x2 = plots.run_metrics(2, tmp_path / "d", tmp_path / "m")
    assert calls == [2]
    assert a1 is a2 and m1 is m2 and x1 is x2


def test_read_findings_uses_the_run_directory(tmp_path):
    """Plan D5: findings sit in the run dir beside run_meta.json."""
    (tmp_path / "findings_2.json").write_text(json.dumps({"findings": [{"stage": "demand", "waived": False}]}))
    assert plots._read_findings(tmp_path, 2) == [{"stage": "demand", "waived": False}]
    assert plots._read_findings(tmp_path, 1) == []


def test_export_all_reads_findings_from_the_run_dir(tmp_path):
    art = plots.Artifacts(prong=2, develop_root=tmp_path / "d", master_root=tmp_path / "m")
    (tmp_path / "findings_2.json").write_text(
        json.dumps({"findings": [{"stage": "demand", "waived": False}, {"stage": "demand", "waived": True}]}),
    )
    plots.export_all(tmp_path, artifacts=art, metric_frames={}, missing=[])
    counts = pd.read_csv(tmp_path / "figures" / "findings_by_stage.csv", index_col=0)
    assert counts.loc["demand", "live"] == 1
    assert counts.loc["demand", "waived"] == 1


def test_export_all_accepts_the_two_arg_form_run_py_uses():
    """run.py calls export_all(ctx.run_dir, ctx); the signature must allow it."""
    import inspect

    params = list(inspect.signature(plots.export_all).parameters)
    assert params[:2] == ["run_dir", "ctx"]
    for name, param in list(inspect.signature(plots.export_all).parameters.items())[2:]:
        assert param.default is not inspect.Parameter.empty, name


# ---------------------------------------------------------------------------
# Prong 2: master's nodal profile is rolled up before any profile metric.
# ---------------------------------------------------------------------------


def test_collect_metrics_aggregates_master_at_prong_2(tmp_path, monkeypatch):
    """Without the rollup the quantile rows differ by construction.

    Master here is the 6-bus nodal file; develop is exactly its 2-cluster
    aggregate, i.e. the two pipelines agree perfectly. The comparison must then
    read as zero delta — which it can only do if master is aggregated first.
    """
    busmap = pd.Series({f"b{i}": ("c0" if i < 3 else "c1") for i in range(6)}, dtype=object)
    master = make_profile(n_bus=6, n_time=24, seed=70)
    develop = metrics.aggregate_profile_to_clusters(master, busmap)

    dev_dir, mas_dir = tmp_path / "dev", tmp_path / "mas"
    art = plots.Artifacts(prong=2, develop_root=dev_dir, master_root=mas_dir, busmap=busmap)
    zone = pd.Series({"c0": "p1", "c1": "p2"}, dtype=object)
    art.zone_develop, art.zone_master = zone, pd.Series({f"b{i}": "p1" for i in range(6)}, dtype=object)

    pairs = art.profile_pairs
    assert pairs, "prong 2 has no profile pairs to exercise"
    pair = pairs[0]
    tech = pair.stage.replace("profile_", "")
    dp, mp = dev_dir / pair.develop, mas_dir / pair.master
    dp.parent.mkdir(parents=True, exist_ok=True)
    mp.parent.mkdir(parents=True, exist_ok=True)
    develop.to_netcdf(dp)
    master.to_netcdf(mp)
    monkeypatch.setattr(plots.Artifacts, "profile_pairs", property(lambda self: [pair]))

    missing: list[dict] = []
    frames = plots.collect_metrics(art, missing)
    assert missing == []
    q = frames[f"p_max_pu_quantiles_{tech}"]
    assert np.allclose(q["delta"].dropna().to_numpy(), 0.0, atol=1e-6)
    pot = frames[f"p_nom_max_by_zone_{tech}"]
    # Master now carries cluster ids, so it must have been joined through the
    # DEVELOP zone map; the substation map would have put everything in p1.
    assert set(pot.index) == {"p1", "p2"}
    assert np.allclose(pot["delta"].to_numpy(), 0.0, atol=1e-3)
    # Same cluster set on both sides: nothing to report and nothing removed.
    assert art.cluster_sets[tech]["equal"] is True


def test_a_develop_only_cluster_is_kept_out_of_the_pooled_metrics(tmp_path, monkeypatch):
    """The HF-24 case: develop has a cluster master never built.

    Pooled in, its 24 hours of low capacity factor move every quantile row and
    its 2,158 MW move its zone's potential, so a row-set difference reads as a
    distributional one. The metrics must be computed over the common clusters,
    with the one-sided cluster recorded instead.
    """
    busmap = pd.Series({f"b{i}": ("c0" if i < 3 else "c1") for i in range(6)}, dtype=object)
    master = make_profile(n_bus=6, n_time=24, seed=71)
    develop = metrics.aggregate_profile_to_clusters(master, busmap)
    extra = make_profile(n_bus=1, n_time=24, seed=72, cf_scale=0.2).assign_coords(bus=["p87 0"])
    extra["p_nom_max"][:] = np.array([2158.0])
    develop = xr.concat([develop, extra], dim="bus")

    dev_dir, mas_dir = tmp_path / "dev", tmp_path / "mas"
    art = plots.Artifacts(prong=2, develop_root=dev_dir, master_root=mas_dir, busmap=busmap)
    art.zone_develop = pd.Series({"c0": "p1", "c1": "p2", "p87 0": "p8"}, dtype=object)
    art.zone_master = pd.Series({f"b{i}": "p1" for i in range(6)}, dtype=object)

    pair = art.profile_pairs[0]
    tech = pair.stage.replace("profile_", "")
    dp, mp = dev_dir / pair.develop, mas_dir / pair.master
    dp.parent.mkdir(parents=True, exist_ok=True)
    mp.parent.mkdir(parents=True, exist_ok=True)
    develop.to_netcdf(dp)
    master.to_netcdf(mp)
    monkeypatch.setattr(plots.Artifacts, "profile_pairs", property(lambda self: [pair]))

    missing: list[dict] = []
    frames = plots.collect_metrics(art, missing)
    assert missing == []

    info = art.cluster_sets[tech]
    assert info["only_develop"] == {"p87 0": pytest.approx(2158.0)}
    assert info["only_develop_mw"] == pytest.approx(2158.0)
    assert info["n_common"] == 2

    # The two pipelines agree on the clusters they share, so every pooled row
    # is zero — the one-sided cluster is reported, never averaged in.
    q = frames[f"p_max_pu_quantiles_{tech}"]
    assert np.allclose(q["delta"].dropna().to_numpy(), 0.0, atol=1e-6)
    pot = frames[f"p_nom_max_by_zone_{tech}"]
    assert set(pot.index) == {"p1", "p2"}, "zone p8 is the develop-only cluster's zone"
    assert np.allclose(pot["delta"].to_numpy(), 0.0, atol=1e-3)
    cf = frames[f"mean_cf_by_zone_{tech}"]
    assert np.allclose(cf["delta"].dropna().to_numpy(), 0.0, atol=1e-9)


def _prong2_profile_fixture(tmp_path, monkeypatch, seed: int = 73):
    """A prong-2 Artifacts whose two sides AGREE once master is rolled up.

    Master is a 6-bus nodal file; develop is exactly its 2-cluster aggregate, so
    every number the harness reports must be zero-delta. Returns
    ``(art, tech, master, develop)``.
    """
    busmap = pd.Series({f"b{i}": ("c0" if i < 3 else "c1") for i in range(6)}, dtype=object)
    master = make_profile(n_bus=6, n_time=24, seed=seed)
    develop = metrics.aggregate_profile_to_clusters(master, busmap)

    dev_dir, mas_dir = tmp_path / "dev", tmp_path / "mas"
    art = plots.Artifacts(prong=2, develop_root=dev_dir, master_root=mas_dir, busmap=busmap)
    art.zone_develop = pd.Series({"c0": "p1", "c1": "p2"}, dtype=object)
    art.zone_master = pd.Series({f"b{i}": "p1" for i in range(6)}, dtype=object)

    pair = art.profile_pairs[0]
    tech = pair.stage.replace("profile_", "")
    dp, mp = dev_dir / pair.develop, mas_dir / pair.master
    dp.parent.mkdir(parents=True, exist_ok=True)
    mp.parent.mkdir(parents=True, exist_ok=True)
    develop.to_netcdf(dp)
    master.to_netcdf(mp)
    monkeypatch.setattr(plots.Artifacts, "profile_pairs", property(lambda self: [pair]))
    return art, tech, master, develop


def test_the_duration_curve_plots_the_rolled_up_master_at_prong_2(tmp_path, monkeypatch):
    """The figure and its table row must be the same object.

    ``p_max_pu_duration_onwind`` used to reopen master's RAW nodal file while
    ``p_max_pu_quantiles_onwind`` beside it used the rolled-up one, so the PNG
    showed a flat-versus-steep pair of curves for a run whose quantile rows were
    equivalent. Here master rolls up to exactly develop, so every percentile of
    the figure's CSV twin must match — and the old code path provably would not.
    """
    art, tech, master, _develop = _prong2_profile_fixture(tmp_path, monkeypatch)
    missing: list[dict] = []
    frames = plots.collect_metrics(art, missing)
    assert missing == []
    plots.export_all(tmp_path, artifacts=art, metric_frames=frames, findings=[], missing=missing)

    data = pd.read_csv(tmp_path / "figures" / f"p_max_pu_duration_{tech}.csv", index_col=0)
    assert np.allclose(data["master"].to_numpy(), data["develop"].to_numpy(), atol=1e-12)
    assert np.allclose(data["delta"].to_numpy(), 0.0, atol=1e-12)
    # The figure agrees with the comparison-table row it belongs to.
    q = frames[f"p_max_pu_quantiles_{tech}"]
    assert np.allclose(q["delta"].dropna().to_numpy(), 0.0, atol=1e-12)

    # The old path: master's RAW nodal values against develop's clusters.
    # Cluster averaging cuts the tails, so this is a different distribution --
    # which is what the figure was showing.
    pct = data.index.to_numpy(dtype=float)
    raw = np.percentile(metrics.profile_values(master), 100.0 - pct)
    stale = "the nodal and clustered distributions coincide; the fixture no longer separates the two code paths"
    assert not np.allclose(raw, data["develop"].to_numpy(), atol=1e-3), stale


def test_the_available_power_figure_uses_the_prepared_pair(tmp_path, monkeypatch):
    """The national available-power figure takes the same prepared datasets."""
    art, tech, _master, _develop = _prong2_profile_fixture(tmp_path, monkeypatch, seed=74)
    missing: list[dict] = []
    frames = plots.collect_metrics(art, missing)
    plots.export_all(tmp_path, artifacts=art, metric_frames=frames, findings=[], missing=missing)

    data = pd.read_csv(tmp_path / "figures" / f"{tech}_national_available_power.csv", index_col=0)
    assert np.allclose(data["master"].to_numpy(), data["develop"].to_numpy(), rtol=1e-9)


def test_the_rolled_up_master_is_labelled_on_the_figure(tmp_path, monkeypatch):
    """A reader must be told the blue curve is not the file master wrote."""
    art, tech, _master, _develop = _prong2_profile_fixture(tmp_path, monkeypatch, seed=75)
    missing: list[dict] = []
    plots.collect_metrics(art, missing)
    prep = art.profiles[tech]
    assert prep.rolled_up is True
    assert f"rolled up to s{plots.SIMPL2}" in prep.master_label
    assert f"master rolled up to s{plots.SIMPL2}" in prep.subtitle
    assert f"{prep.info['n_common']} common clusters" in prep.subtitle

    plain = plots.PreparedProfile(master=None, develop=None)
    assert plain.master_label == plots.LABELS["master"]
    assert plain.subtitle == ""


def test_prong_1_passes_both_sides_through_unaggregated(tmp_path):
    """No busmap, no rollup — and nothing cached, so a nodal file is not held."""
    art = plots.Artifacts(prong=1, develop_root=tmp_path / "d", master_root=tmp_path / "m")
    master, develop = make_profile(n_bus=4, seed=76), make_profile(n_bus=4, seed=77)
    prep = plots.prepare_profiles(art, "onwind", develop, master, [])
    assert prep.rolled_up is False
    assert prep.master is master and prep.develop is develop
    assert art.profiles == {}


# ---------------------------------------------------------------------------
# Existing capacity at the assembled stage (HF-26 / HF-27).
#
# `EQ_UNTIL=assembled` stops the run before cluster_network, so there is no
# clustered network and, until these metrics fell back to the assembled pair,
# no existing-capacity comparison at all: the western leg's 3.5 GW onwind
# difference was not a row in the table, it was nothing.
# ---------------------------------------------------------------------------


def _assembled_only(master, develop) -> plots.Artifacts:
    return plots.Artifacts(
        prong=2,
        develop_root=Path("/nonexistent/dev"),
        master_root=Path("/nonexistent/mas"),
        assembled_master=master,
        assembled_develop=develop,
    )


def test_existing_capacity_falls_back_to_the_assembled_networks():
    art = _assembled_only(make_network(), make_network(scale=2.0))
    missing: list[dict] = []
    frames = plots.collect_metrics(art, missing)

    assert missing == []
    assert "capacity_existing_by_carrier" in frames
    assert "p_nom_existing_by_zone_carrier" in frames
    # No clustered network, so the demand metric stays absent rather than
    # being computed off a different stage.
    assert "demand_by_zone" not in frames

    solar = frames["capacity_existing_by_carrier"].loc["solar"]
    assert solar["master"] == pytest.approx(100.0)
    assert solar["develop"] == pytest.approx(200.0)
    assert solar["delta_pct"] == pytest.approx(100.0)


def test_the_clustered_network_still_wins_when_both_stages_are_loaded():
    """A full run must keep comparing existing capacity at the clustered stage.

    Existing capacity is conserved by clustering, so either stage answers the
    question — but silently swapping which one a shipped number comes from is
    exactly the kind of change that makes two runs incomparable.
    """
    art = plots.Artifacts(
        prong=2,
        develop_root=Path("/nonexistent/dev"),
        master_root=Path("/nonexistent/mas"),
        n_master=make_network(),
        n_develop=make_network(scale=3.0),
        assembled_master=make_network(),
        assembled_develop=make_network(scale=2.0),
    )
    frames = plots.collect_metrics(art, [])
    solar = frames["capacity_existing_by_carrier"].loc["solar"]
    assert solar["develop"] == pytest.approx(300.0), "the clustered pair should have been used"


def test_no_assembled_and_no_clustered_network_means_no_capacity_rows():
    """Absent artifacts are absent, not MISSING: a profile-only run is legitimate."""
    art = plots.Artifacts(
        prong=2,
        develop_root=Path("/nonexistent/dev"),
        master_root=Path("/nonexistent/mas"),
    )
    missing: list[dict] = []
    frames = plots.collect_metrics(art, missing)
    assert "capacity_existing_by_carrier" not in frames
    assert missing == []


def test_a_broken_zone_map_on_the_assembled_pair_is_recorded_not_skipped():
    """The geographic-assignment criterion must not vanish at the assembled stage either."""
    master, develop = make_network(), make_network()
    for n in (master, develop):
        n.buses.drop(columns=["reeds_zone"], inplace=True)
    missing: list[dict] = []
    frames = plots.collect_metrics(_assembled_only(master, develop), missing)
    assert "p_nom_existing_by_zone_carrier" not in frames
    assert any(e["metric"] == "p_nom_existing_by_zone_carrier" for e in missing)


def test_load_artifacts_reads_the_assembled_pair(tmp_path, monkeypatch):
    """``load_artifacts`` pairs develop's assembled pkl with master's .nc."""
    from tests.equivalence import paths

    dev_root, mas_root = tmp_path / "dev", tmp_path / "mas"
    dp = dev_root / paths.assembled_target(2)
    mp = mas_root / paths.baseline_assembled_target(2)
    for p in (dp, mp):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")

    loaded: list[Path] = []

    def fake_load_network(path):
        loaded.append(Path(path))
        return make_network()

    monkeypatch.setattr("tests.equivalence.compare.load_network", fake_load_network)
    monkeypatch.setattr(plots, "_zone_maps", lambda root: (pd.Series(dtype=object), pd.Series(dtype=object), None))
    monkeypatch.setattr("tests.equivalence.compare.load_busmap", lambda root, *a, **k: None)

    art = plots.load_artifacts(2, dev_root, mas_root)
    assert art.assembled_develop is not None
    assert art.assembled_master is not None
    assert set(loaded) == {dp, mp}
    assert dp.suffix == ".pkl" and mp.suffix == ".nc"


# ---------------------------------------------------------------------------
# HF-26 reconstruction figures.


def _recon(rows: dict[tuple[str, str], tuple[float, float, float, float]], error: str | None = None):
    """A :class:`reconstructions.Reconstruction` from (fleet, profiled, master, develop)."""
    from tests.equivalence import reconstructions, tables

    tol = tables.tolerance_for("p_nom_existing_by_zone_carrier")
    made = {}
    for (zone, carrier), (fleet, profiled, master, develop) in rows.items():
        row = reconstructions._make_row(
            "p_nom_existing_by_zone_carrier",
            zone,
            carrier,
            fleet,
            profiled,
            master,
            develop,
            tol,
        )
        made[("p_nom_existing_by_zone_carrier", row.key)] = row
    return reconstructions.Reconstruction(
        name="hf26_existing_renewable_drop",
        frame=reconstructions.rows_frame(made),
        rows=made,
        error=error,
    )


def test_hf26_figure_writes_png_and_full_csv(tmp_path):
    """The plot caps at ``cap`` zones; the CSV twin keeps every one of them.

    The cap exists because the USA leg has 134 ReEDS zones. A capped CSV would
    make the figure's own data table disagree with the national totals in its
    title, which is the one thing the CSV twin is for.
    """
    rows = {(f"p{i}", "onwind"): (100.0 + i, 50.0, 50.0, 100.0 + i) for i in range(8)}
    rows[("p0", "solar")] = (900.0, 400.0, 400.0, 900.0)
    png, csv = plots.hf26_dropped_zone_figure(_recon(rows), "hf26_dropped_mw_by_zone", tmp_path, cap=3)
    assert png.exists() and csv.exists()
    frame = pd.read_csv(csv)
    # Every zone, uncapped, both carriers.
    assert len(frame) == len(rows)
    assert set(frame["zone"]) == {f"p{i}" for i in range(8)}
    assert set(frame["carrier"]) == {"onwind", "solar"}
    assert {"dropped_mw", "residual_mw", "gate_tol_mw"} <= set(frame.columns)


def test_hf26_figure_is_a_placeholder_when_the_reconstruction_failed(tmp_path):
    png, csv = plots.hf26_dropped_zone_figure(
        _recon({}, error="FileNotFoundError: elec_base_network.nc"),
        "hf26_dropped_mw_by_zone",
        tmp_path,
    )
    assert png.exists() and csv.exists()
    png, _ = plots.hf26_dropped_zone_figure(None, "hf26_none", tmp_path)
    assert png.exists()


def test_hf26_map_figure_writes_png_and_csv(tmp_path):
    """The CSV twin is per zone AND carrier, and carries the gate as well as the residual.

    Summing the residual over carriers, which an earlier version did, lets a
    +X onwind / -X solar zone cancel to exactly zero and draw as perfect.
    """
    zones = _point_zones(names=("p0", "p1", "p2"))
    rows = {
        ("p0", "onwind"): (300.0, 100.0, 100.0, 300.0),
        ("p1", "onwind"): (50.0, 50.0, 50.0, 50.0),
        ("p0", "solar"): (900.0, 400.0, 400.0, 900.0),
    }
    png, csv = plots.hf26_dropped_map_figure(zones, _recon(rows), "hf26_dropped_mw_map", tmp_path)
    assert png.exists() and csv.exists()
    frame = pd.read_csv(csv)
    assert {"zone", "carrier", "dropped_mw", "residual_mw", "gate_tol_mw", "residual_over_gate", "ok"} <= set(
        frame.columns,
    )
    assert len(frame) == len(rows)
    onwind_p0 = frame[(frame["zone"] == "p0") & (frame["carrier"] == "onwind")].iloc[0]
    assert onwind_p0["dropped_mw"] == pytest.approx(200.0)
    assert onwind_p0["residual_over_gate"] == pytest.approx(0.0)


def test_hf26_map_residual_is_not_summed_across_carriers(tmp_path):
    """+100 onwind and -100 solar in one zone must NOT cancel to zero."""
    zones = _point_zones(names=("p0", "p1"))
    rows = {
        # fleet, profiled, master, develop -> residual = (dev-mas) - (fleet-profiled)
        ("p0", "onwind"): (500.0, 500.0, 500.0, 600.0),  # residual +100
        ("p0", "solar"): (500.0, 500.0, 500.0, 400.0),  # residual -100
    }
    _png, csv = plots.hf26_dropped_map_figure(zones, _recon(rows), "hf26_map_signs", tmp_path)
    frame = pd.read_csv(csv).set_index(["zone", "carrier"])
    assert frame.loc[("p0", "onwind"), "residual_mw"] == pytest.approx(100.0)
    assert frame.loc[("p0", "solar"), "residual_mw"] == pytest.approx(-100.0)
    # Both fail their own gate, and the figure says so per carrier.
    assert not bool(frame.loc[("p0", "onwind"), "ok"])
    assert not bool(frame.loc[("p0", "solar"), "ok"])


def test_hf26_map_residual_ratio_makes_a_small_failure_visible(tmp_path):
    """A 2 MW residual against a 1 MW gate must read as a failure, not as noise.

    This is the USA ``p129 | solar`` shape. Beside a zone holding 20,000 MW, a
    raw-MW scale drew it at under 10 % of the ramp; in units of its own gate it
    is 2.0, i.e. twice the band edge.
    """
    zones = _point_zones(names=("p129", "p10"))
    rows = {
        ("p129", "solar"): (100.0, 100.0, 100.0, 102.0),  # residual +2, gate 1.0 (atol)
        ("p10", "solar"): (21000.0, 17700.0, 17700.0, 21000.0),  # exact, gate 88.5
    }
    _png, csv = plots.hf26_dropped_map_figure(zones, _recon(rows), "hf26_map_ratio", tmp_path)
    frame = pd.read_csv(csv).set_index(["zone", "carrier"])
    assert frame.loc[("p129", "solar"), "gate_tol_mw"] == pytest.approx(1.0)
    assert frame.loc[("p129", "solar"), "residual_over_gate"] == pytest.approx(2.0)
    assert not bool(frame.loc[("p129", "solar"), "ok"])
    # The big zone is exact, so it sits at the neutral middle of the same scale.
    assert frame.loc[("p10", "solar"), "residual_over_gate"] == pytest.approx(0.0)
    assert bool(frame.loc[("p10", "solar"), "ok"])
    assert abs(2.0) < plots.RECON_RESIDUAL_RATIO_MAX, "the failure must be inside the drawn range"


def test_hf26_map_figure_is_skipped_without_zones(tmp_path):
    rows = {("p0", "onwind"): (300.0, 100.0, 100.0, 300.0)}
    png, csv = plots.hf26_dropped_map_figure(None, _recon(rows), "hf26_map_none", tmp_path)
    assert png.exists() and csv.exists()
    assert pd.read_csv(csv).empty
