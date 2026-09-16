"""Computed explanations: reconstruct a difference from the artifacts.

A ``reconstruction:`` waiver in ``waivers.yaml`` names a function here instead of
a hand-measured percentage. The function returns, per comparison-table row, the
MW the named mechanism predicts, and ``tables.py`` explains the row only when the
prediction matches the row's delta within the metric's own tolerance.

The mechanism this module was written for is **HF-26**. Both branches run
``add_electricity.attach_renewable_capacities_to_atlite``, which is
byte-identical between them; its last step is::

    mapped_values = generators_tech.sub_assignment.map(caps_per_bus).dropna()
    n.generators.loc[mapped_values.index, "p_nom"] = mapped_values

so an existing plant keeps its MW only if its bus already carries a
**profile-derived** generator of the same carrier. Master runs this on the NODAL
network, whose profile files cover a minority of substations, so the rest is
dropped. Develop runs it after simplify-early at ``s{simpl}`` cluster
granularity, where a cluster is covered if any of its substations is, and loses
nothing (``test_attach_conservation.py`` pins exactly this on a toy network).

The prediction is therefore, per (ReEDS zone, carrier)::

    fleet_mw    = MW of existing c-plants whose nearest master nodal bus is in z
    profiled_mw = the part of fleet_mw at substations master's profile_{c}.nc covers
    dropped_mw  = fleet_mw - profiled_mw

    gate G1 (master):  profiled_mw ~= master   (the table's master column)
    gate G2 (develop): fleet_mw    ~= develop  (develop drops nothing)
    residual        = (develop - master) - dropped_mw

Both gates predict a side **separately**, which is what makes this a
reconstruction rather than a restatement of the delta: a wrong plant population
fails G2, a wrong coverage set fails G1, and a zone relocation (the HF-27 shape)
fails both with equal and opposite residuals. ``residual`` is identically
``master_recon_err - develop_recon_err``, so it cannot be small by accident when
either gate is large.

A reconstruction that cannot be computed sets :attr:`Reconstruction.error` and
every lookup returns ``None``, so the rows stay ``UNEXPLAINED``. Failing open
would be the one outcome worse than an unexplained row.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics, tables
from .paths import CONFIGFILE, EQ, INTERCONNECT, REPO, baseline_assembled_target, prong_pairs

#: The two comparison-table metrics this module has anything to say about.
ZONE_METRIC = "p_nom_existing_by_zone_carrier"
NATIONAL_METRIC = "capacity_existing_by_carrier"

#: The carriers HF-26 moves. A carrier outside this set gets NO row, so a waiver
#: that names the metric without a key still cannot explain ``CCGT``.
DEFAULT_CARRIERS: tuple[str, ...] = ("onwind", "solar")

#: Columns of the per-(zone, carrier) reconstruction frame, in output order.
RECON_COLUMNS: tuple[str, ...] = (
    "metric",
    "key",
    "zone",
    "carrier",
    "fleet_mw",
    "profiled_mw",
    "dropped_mw",
    "master_mw",
    "develop_mw",
    "master_recon_err",
    "develop_recon_err",
    "residual_mw",
    "gate_tol_mw",
    "ok",
    "note",
)

#: Columns of :func:`reconstruct_drop`'s output.
DROP_COLUMNS: tuple[str, ...] = ("fleet_mw", "profiled_mw", "dropped_mw", "n_subs", "n_subs_dropped")


# ---- pure core -------------------------------------------------------------


@dataclass(frozen=True)
class ReconRow:
    """What the mechanism predicts for ONE comparison-table row."""

    metric: str
    key: str  # the comparison-table key, e.g. "p10 | onwind" or "onwind"
    zone: str | None  # None for a national row
    carrier: str
    fleet_mw: float
    profiled_mw: float
    dropped_mw: float  # fleet_mw - profiled_mw; the prediction
    master_mw: float  # the table's master value
    develop_mw: float  # the table's develop value
    master_recon_err: float  # profiled_mw - master_mw   (gate G1)
    develop_recon_err: float  # fleet_mw    - develop_mw  (gate G2)
    residual_mw: float  # (develop_mw - master_mw) - dropped_mw
    gate_tol_mw: float  # the MW the gates and the residual are judged against
    ok: bool
    note: str  # "" when ok, else why not


@dataclass
class Reconstruction:
    """One named reconstruction's rows, plus the frame behind them."""

    name: str
    frame: pd.DataFrame  # index (zone, carrier); every column of ReconRow
    rows: dict[tuple[str, str], ReconRow] = field(default_factory=dict)
    error: str | None = None  # set when it could not be computed at all

    def lookup(self, metric: str, key: str) -> ReconRow | None:
        """The row for ``(metric, key)``, or ``None``.

        ``None`` whenever the reconstruction failed, whatever rows survived: a
        partial reconstruction has not shown that it reproduces either side, so
        it must not explain anything.
        """
        if self.error:
            return None
        return self.rows.get((str(metric), str(key)))

    @property
    def ok(self) -> bool:
        """Was this reconstruction computed at all (no :attr:`error`)."""
        return self.error is None

    def national(self) -> dict[str, dict[str, float]]:
        """``{carrier: {fleet_mw, profiled_mw, dropped_mw, master_mw, ...}}``."""
        out: dict[str, dict[str, float]] = {}
        for (metric, key), row in self.rows.items():
            if metric != NATIONAL_METRIC:
                continue
            out[str(key)] = {
                "fleet_mw": row.fleet_mw,
                "profiled_mw": row.profiled_mw,
                "dropped_mw": row.dropped_mw,
                "master_mw": row.master_mw,
                "develop_mw": row.develop_mw,
                "residual_mw": row.residual_mw,
                "ok": row.ok,
            }
        return out


def empty_recon_frame() -> pd.DataFrame:
    """An empty (zone, carrier) frame with the full column set."""
    idx = pd.MultiIndex.from_arrays([[], []], names=["zone", "carrier"])
    return pd.DataFrame({c: pd.Series(dtype=object) for c in RECON_COLUMNS}, index=idx)


def empty_drop_frame() -> pd.DataFrame:
    """An empty :func:`reconstruct_drop` result."""
    idx = pd.MultiIndex.from_arrays([[], []], names=["zone", "carrier"])
    return pd.DataFrame({c: pd.Series(dtype=float) for c in DROP_COLUMNS}, index=idx)


def normalize_sub_id(value) -> str | None:
    r"""``'35827'``, ``35827.0`` and ``'35827.0'`` all normalise to ``'35827'``.

    Substation ids reach this module in three formats — plain strings from
    ``busmap_s{simpl}.csv``, floats from ``bus2sub.csv``'s ``sub_id`` column, and
    float-formatted strings from the ``bus`` coordinate of master's
    ``profile_{tech}.nc``. Normalising once is not cosmetic: an id-format
    mismatch makes the covered set empty, every row reads as 100 % dropped, and
    the master gate is the only thing that catches it.

    Delegates to :func:`metrics.float_int_label`, which is the same ``\\.0$``
    strip ``add_electricity._sub_id_strings`` applies, so there is one rule and
    not three. A label that is neither an integer nor its float spelling
    (``'35827.5'``, ``'035827'``) returns ``None`` and is counted as unmatched
    rather than guessed at.
    """
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return metrics.float_int_label(value)


def canonical_sub_id(value) -> str | None:
    """The id substation sets are JOINED on: the integer form, else the literal.

    :func:`normalize_sub_id` is deliberately strict because it is the rule for
    mapping a label onto a value, where a wrong guess silently merges data.
    Matching two id SETS is a different question: the only risk is failing to
    match, so a label with no integer form falls back to itself. Both sides go
    through this same function, so ``'035827'`` still matches only ``'035827'``
    and never ``'35827'`` -- the strictness is preserved, the silent emptying is
    not.

    Missing is still missing: ``None`` and NaN return ``None`` rather than the
    string ``'nan'``, which would join every un-substationed bus to every other.
    """
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    norm = metrics.float_int_label(value)
    if norm is not None:
        return norm
    text = str(value).strip()
    return text or None


def normalize_sub_ids(values) -> pd.Series:
    """:func:`canonical_sub_id` over a Series, keeping the index."""
    s = pd.Series(values)
    return pd.Series([canonical_sub_id(v) for v in s], index=s.index, dtype=object)


def row_key(zone: str | None, carrier: str) -> str:
    """The comparison-table key for a (zone, carrier) or national row.

    Built with ``tables._key_label`` so it cannot drift from the label
    :func:`tables.comparison_table` puts in the ``key`` column; a reconstruction
    keyed ``"p10|onwind"`` against a table keyed ``"p10 | onwind"`` would explain
    nothing while looking like it should.
    """
    if zone is None:
        return tables._key_label(carrier)
    return tables._key_label((zone, carrier))


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def reconstruct_drop(
    fleet: pd.DataFrame,
    profiled_subs: Mapping[str, set[str]],
) -> pd.DataFrame:
    """(zone, carrier) -> fleet_mw, profiled_mw, dropped_mw, n_subs, n_subs_dropped.

    ``fleet`` carries one row per plant with ``sub_id``, ``carrier``, ``p_nom``
    and ``zone``; ``profiled_subs`` maps a carrier to the substation ids master's
    profile file for that carrier covers. Both id spaces go through
    :func:`normalize_sub_id`, so the caller may hand over whichever spelling its
    artifact happened to use.
    """
    if fleet is None or len(fleet) == 0:
        return empty_drop_frame()
    df = pd.DataFrame(
        {
            "zone": fleet["zone"].astype(str).to_numpy(),
            "carrier": fleet["carrier"].astype(str).to_numpy(),
            "sub_id": normalize_sub_ids(fleet["sub_id"]).to_numpy(),
            "p_nom": pd.to_numeric(pd.Series(fleet["p_nom"].to_numpy()), errors="coerce").fillna(0.0).to_numpy(),
        },
    )
    covered = {
        str(carrier): {s for s in (canonical_sub_id(v) for v in subs) if s is not None}
        for carrier, subs in (profiled_subs or {}).items()
    }
    df["profiled"] = [
        sid is not None and sid in covered.get(car, frozenset())
        for sid, car in zip(df["sub_id"], df["carrier"], strict=False)
    ]
    df["_profiled_mw"] = df["p_nom"].where(df["profiled"], 0.0)
    grouped = df.groupby(["zone", "carrier"], sort=True)
    out = grouped.agg(
        fleet_mw=("p_nom", "sum"),
        profiled_mw=("_profiled_mw", "sum"),
        n_subs=("sub_id", "nunique"),
    )
    dropped_subs = df[~df["profiled"]].groupby(["zone", "carrier"], sort=True)["sub_id"].nunique()
    out["n_subs_dropped"] = dropped_subs.reindex(out.index).fillna(0).astype(int)
    out["dropped_mw"] = out["fleet_mw"] - out["profiled_mw"]
    out.index.names = ["zone", "carrier"]
    return out[list(DROP_COLUMNS)]


def gate_tolerance(master: float, develop: float, tol: tables.Tolerance) -> float:
    """``max(tol.atol, tol.rtol * max(|master|, |develop|))`` — the row's own tolerance.

    The same rule ``tables._row_verdict`` uses to call a row equivalent, so a
    reconstruction is never held to a standard the table itself does not apply:
    a prediction that lands inside the row's tolerance explains it, and one that
    does not, does not.
    """
    m, d = abs(_f(master)), abs(_f(develop))
    scale = float(np.nanmax([0.0, 0.0 if np.isnan(m) else m, 0.0 if np.isnan(d) else d]))
    return max(float(tol.atol), float(tol.rtol) * scale)


def _make_row(
    metric: str,
    zone: str | None,
    carrier: str,
    fleet_mw: float,
    profiled_mw: float,
    master_mw: float,
    develop_mw: float,
    tol: tables.Tolerance,
) -> ReconRow:
    """One :class:`ReconRow`, gates evaluated."""
    fleet_mw, profiled_mw = _f(fleet_mw), _f(profiled_mw)
    master_mw, develop_mw = _f(master_mw), _f(develop_mw)
    dropped_mw = fleet_mw - profiled_mw
    g1 = profiled_mw - master_mw
    g2 = fleet_mw - develop_mw
    residual = (develop_mw - master_mw) - dropped_mw
    gate_tol = gate_tolerance(master_mw, develop_mw, tol)
    notes: list[str] = []
    if not np.isfinite(master_mw) or not np.isfinite(develop_mw):
        notes.append(f"table row is not finite (master {master_mw}, develop {develop_mw})")
    else:
        if abs(g1) > gate_tol:
            notes.append(
                f"master gate: reconstructed profiled {profiled_mw:,.1f} MW vs master {master_mw:,.1f} MW "
                f"(off by {g1:,.1f} MW, tolerance {gate_tol:,.1f} MW)",
            )
        if abs(g2) > gate_tol:
            notes.append(
                f"develop gate: reconstructed fleet {fleet_mw:,.1f} MW vs develop {develop_mw:,.1f} MW "
                f"(off by {g2:,.1f} MW, tolerance {gate_tol:,.1f} MW)",
            )
    return ReconRow(
        metric=metric,
        key=row_key(zone, carrier),
        zone=zone,
        carrier=str(carrier),
        fleet_mw=fleet_mw,
        profiled_mw=profiled_mw,
        dropped_mw=dropped_mw,
        master_mw=master_mw,
        develop_mw=develop_mw,
        master_recon_err=g1,
        develop_recon_err=g2,
        residual_mw=residual,
        gate_tol_mw=gate_tol,
        ok=not notes,
        note="; ".join(notes),
    )


def _zone_values(by_zone) -> dict[tuple[str, str], tuple[float, float]]:
    """``{(zone, carrier): (master, develop)}`` off the metrics frame."""
    out: dict[tuple[str, str], tuple[float, float]] = {}
    if by_zone is None or len(by_zone) == 0:
        return out
    for key, row in by_zone.iterrows():
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        out[(str(key[0]), str(key[1]))] = (_f(row.get("master")), _f(row.get("develop")))
    return out


def _carrier_values(by_carrier) -> dict[str, tuple[float, float]]:
    """``{carrier: (master, develop)}`` off the metrics frame."""
    out: dict[str, tuple[float, float]] = {}
    if by_carrier is None or len(by_carrier) == 0:
        return out
    for key, row in by_carrier.iterrows():
        out[str(key)] = (_f(row.get("master")), _f(row.get("develop")))
    return out


def build_rows(
    recon: pd.DataFrame,
    by_zone: pd.DataFrame,
    by_carrier: pd.DataFrame,
    tol: tables.Tolerance,
    carriers: Sequence[str] = DEFAULT_CARRIERS,
) -> dict[tuple[str, str], ReconRow]:
    """Zone rows keyed ``('p_nom_existing_by_zone_carrier', 'p10 | onwind')`` and
    national rows keyed ``('capacity_existing_by_carrier', 'onwind')``, the
    latter summed over zones.

    A carrier outside ``carriers`` gets NO row: the reconstruction has nothing
    to say about it and the waiver must not hold.

    The key set is the UNION of what the reconstruction found and what the
    comparison table reports. A zone the table has but the reconstruction does
    not is the relocation shape (HF-27): it gets a row with zero predicted drop,
    so the whole delta lands in the residual and the row stays unexplained,
    rather than falling through as "no prediction, therefore fine".
    """
    carriers = tuple(str(c) for c in carriers)
    zone_vals = _zone_values(by_zone)
    nat_vals = _carrier_values(by_carrier)

    found: dict[tuple[str, str], tuple[float, float]] = {}
    if recon is not None and len(recon) > 0:
        for key, row in recon.iterrows():
            if not isinstance(key, tuple) or len(key) != 2:
                continue
            zone, carrier = str(key[0]), str(key[1])
            if carrier not in carriers:
                continue
            found[(zone, carrier)] = (_f(row.get("fleet_mw")), _f(row.get("profiled_mw")))

    rows: dict[tuple[str, str], ReconRow] = {}
    keys = sorted(set(found) | {k for k in zone_vals if k[1] in carriers})
    for zone, carrier in keys:
        fleet, profiled = found.get((zone, carrier), (0.0, 0.0))
        master, develop = zone_vals.get((zone, carrier), (0.0, 0.0))
        row = _make_row(ZONE_METRIC, zone, carrier, fleet, profiled, master, develop, tol)
        rows[(ZONE_METRIC, row.key)] = row

    for carrier in carriers:
        parts = [v for (_z, c), v in found.items() if c == carrier]
        if not parts and carrier not in nat_vals:
            continue
        fleet = float(sum(p[0] for p in parts))
        profiled = float(sum(p[1] for p in parts))
        master, develop = nat_vals.get(carrier, (0.0, 0.0))
        row = _make_row(NATIONAL_METRIC, None, carrier, fleet, profiled, master, develop, tol)
        rows[(NATIONAL_METRIC, row.key)] = row
    return rows


def rows_frame(rows: Mapping[tuple[str, str], ReconRow]) -> pd.DataFrame:
    """The per-(zone, carrier) frame behind ``rows``, for the CSV and the figures."""
    records = [asdict(r) for (metric, _k), r in sorted(rows.items()) if metric == ZONE_METRIC]
    if not records:
        return empty_recon_frame()
    out = pd.DataFrame.from_records(records)
    out = out.set_index(["zone", "carrier"], drop=False)
    out.index.names = ["zone", "carrier"]
    return out[list(RECON_COLUMNS)]


# ---- population ------------------------------------------------------------


def nerc_region_mapper() -> dict[str, str]:
    """``workflow/scripts/constants.NERC_REGION_MAPPER``, read from the workflow.

    Imported with ``sys.path`` restored afterwards. The workflow's scripts are
    flat modules that import each other by bare name, so the directory has to go
    on the path for the import to work — but leaving it there for the rest of a
    20-hour harness process is a shadowing hazard nobody asked for.
    """
    scripts = str(REPO / "workflow" / "scripts")
    added = scripts not in sys.path
    if added:
        sys.path.insert(0, scripts)
    try:
        import constants

        return dict(constants.NERC_REGION_MAPPER)
    finally:
        if added and scripts in sys.path:
            sys.path.remove(scripts)


def filter_fleet(
    plants: pd.DataFrame,
    investment_year: int,
    interconnect: str,
    honor_planned_retirements: bool,
    carriers: Sequence[str] = DEFAULT_CARRIERS,
) -> pd.DataFrame:
    """The fleet ``add_electricity`` sees, mirroring ``load_powerplants``.

    Proposed units take their planned operating year as ``build_year``,
    announced retirements are honoured, ``build_year <= period``, retirement
    year ``> period``, ``nerc_region != 'non-conus'``, the interconnect filter
    through ``const.NERC_REGION_MAPPER``, ``prime_mover_code != 'PS'``
    (``attach_renewable_capacities_to_atlite`` drops PS before the map) and
    ``carrier in carriers``.

    This is NOT a call into ``load_powerplants``: importing master-benchmark's
    ``add_electricity`` alongside develop's under the same module name is a
    ``sys.path`` collision. Parity is enforced instead by
    ``test_filter_fleet_matches_the_workflow_load_powerplants``, which runs
    develop's own ``load_powerplants`` on a synthetic table and demands the same
    index. That test assumes ``honor_planned_retirements`` and the PUDL year are
    **ported** to ``master-benchmark`` (harness decision 2026-09-14), so
    develop's ``load_powerplants`` is master's too.
    """
    carriers = tuple(str(c) for c in carriers)
    df = plants.copy()
    if "generator_name" in df.columns:
        df = df.set_index("generator_name")
    for col in ("current_planned_generator_operating_date", "generator_retirement_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    proposed = df["operational_status"].astype(str) == "proposed"
    df.loc[proposed, "build_year"] = df.loc[proposed, "current_planned_generator_operating_date"].dt.year

    far_future = pd.to_datetime("2100-01-01")
    live = df["operational_status"].astype(str).isin(["existing", "proposed"])
    if honor_planned_retirements:
        planned = pd.to_datetime(df["planned_generator_retirement_date"], errors="coerce")
        df.loc[live, "generator_retirement_date"] = planned[live].fillna(far_future)
    else:
        df.loc[live, "generator_retirement_date"] = far_future
    df.loc[df["generator_retirement_date"].isna(), "generator_retirement_date"] = pd.to_datetime("1900-01-01")

    build_year = pd.to_numeric(df["build_year"], errors="coerce")
    df = df[(build_year <= investment_year).fillna(False)]
    df = df[df["generator_retirement_date"].dt.year > investment_year]
    df = df[df["nerc_region"] != "non-conus"]
    if interconnect is not None and str(interconnect) != "usa":
        mapper = nerc_region_mapper()
        df = df[df["nerc_region"].map(mapper) == str(interconnect)]
    if "prime_mover_code" in df.columns:
        df = df[df["prime_mover_code"] != "PS"]
    return df[df["carrier"].astype(str).isin(carriers)].copy()


def filter_to_footprint(plants: pd.DataFrame, regions) -> pd.DataFrame:
    """Plants whose point intersects master's ``regions_onshore`` u ``regions_offshore``.

    Mirrors the two ``gpd.sjoin`` calls at the head of master's
    ``filter_plants_by_region``. Master then RE-ADDS "must add" seam plants that
    fall outside every ReEDS shape; that add-back is deliberately omitted here,
    and its absence is safe rather than free — a plant it would have added shows
    up as a gate-G2 shortfall and leaves the row unexplained, which is the right
    failure. Widening the gate instead of adding the seam pass would not be.
    """
    import geopandas as gpd

    if plants is None or len(plants) == 0 or regions is None or len(regions) == 0:
        return plants.iloc[0:0].copy() if plants is not None else plants
    points = gpd.GeoDataFrame(
        plants,
        geometry=gpd.points_from_xy(plants["longitude"], plants["latitude"]),
        crs="EPSG:4326",
    )
    shapes = regions[["geometry"]]
    if getattr(shapes, "crs", None) is not None and str(shapes.crs) != "EPSG:4326":
        shapes = shapes.to_crs("EPSG:4326")
    joined = gpd.sjoin(points, shapes, how="inner", predicate="intersects")
    keep = joined.index[~joined.index.duplicated()]
    return plants.loc[keep].copy()


def assign_to_master_substations(plants: pd.DataFrame, buses: pd.DataFrame) -> pd.DataFrame:
    """Nearest master nodal bus, then bus -> ``sub_id`` and bus -> ``reeds_zone``.

    A BallTree over RAW DEGREES — master's own metric in
    ``add_electricity.match_nearest_bus``, degree distortion included. Switching
    to haversine would be more correct and less equivalent; the question here is
    which substation MASTER picked, not which one is nearest on the sphere.

    Master's ``match_plant_to_bus`` has a first pass that restricts the search to
    the plant's own ReEDS zone by comparing the plant's ``country`` column to the
    bus ``reeds_zone``. After ``filter_plants_by_region`` that column holds the
    region layer's ``country`` (a county id such as ``p06023``), so the first
    pass matches nothing and the global second pass does all the work — which is
    what this single global tree reproduces.
    """
    from sklearn.neighbors import BallTree

    out = plants.copy()
    if len(out) == 0 or buses is None or len(buses) == 0:
        out["bus_assignment"] = pd.Series(dtype=object)
        out["distance_nearest"] = pd.Series(dtype=float)
        out["sub_id"] = pd.Series(dtype=object)
        out["zone"] = pd.Series(dtype=object)
        return out
    tree = BallTree(buses[["x", "y"]].to_numpy(dtype=float), leaf_size=2)
    dist, idx = tree.query(out[["longitude", "latitude"]].to_numpy(dtype=float), k=1)
    picked = buses.index.to_numpy()[idx.flatten()]
    out["bus_assignment"] = picked
    out["distance_nearest"] = dist.flatten()
    out["sub_id"] = pd.Series(picked, index=out.index).map(buses["sub_id"])
    out["zone"] = pd.Series(picked, index=out.index).map(buses["reeds_zone"]).astype(str)
    return out


# ---- loaders (the only file-touching part) ---------------------------------


def master_bus_frame(master_root: Path, interconnect: str) -> pd.DataFrame:
    """Index Bus -> ``x``, ``y``, ``sub_id`` (normalised), ``reeds_zone``.

    ``x``/``y``/``reeds_zone`` come from master's
    ``resources/equivalence/{ic}/elec_base_network.nc`` (5 MB western, 104 MB
    usa — always built, unlike the GIS csv export); ``sub_id`` from
    ``{ic}/bus2sub.csv``, falling back to the network's own ``sub_id`` column.

    Master's matcher runs on ``elec_base_network_dem.nc``, whose bus set
    ``add_demand`` does not change, so the base network answers the same
    question. The 6.5 GB ``elec_base_network_l_pp.pkl`` is deliberately NOT
    opened.
    """
    import pypsa

    root = Path(master_root)
    net = root / f"{EQ}/{interconnect}/elec_base_network.nc"
    if not net.exists():
        raise FileNotFoundError(f"master nodal network not found: {net}")
    n = pypsa.Network(str(net))
    buses = n.buses
    out = pd.DataFrame(
        {
            "x": pd.to_numeric(buses["x"], errors="coerce"),
            "y": pd.to_numeric(buses["y"], errors="coerce"),
            "reeds_zone": buses["reeds_zone"].astype(str) if "reeds_zone" in buses else "<unmapped>",
        },
    )
    out.index = pd.Index([str(b) for b in buses.index], name="Bus")

    sub: pd.Series | None = None
    b2s = root / f"{EQ}/{interconnect}/bus2sub.csv"
    if b2s.exists():
        raw = pd.read_csv(b2s)
        if {"Bus", "sub_id"} <= set(raw.columns):
            sub = pd.Series(
                normalize_sub_ids(raw["sub_id"]).to_numpy(),
                index=pd.Index([str(b) for b in raw["Bus"]]),
            )
            sub = sub[~sub.index.duplicated()]
    if sub is not None:
        out["sub_id"] = out.index.map(sub)
    elif "sub_id" in buses:
        out["sub_id"] = normalize_sub_ids(pd.Series(buses["sub_id"].to_numpy(), index=out.index))
    else:
        raise KeyError(f"no substation ids: neither {b2s} nor a sub_id column on {net}")
    return out.dropna(subset=["x", "y", "sub_id"])


def master_profiled_subs(master_root: Path, prong: int) -> dict[str, set[str]]:
    """``carrier -> substation ids master's profile file covers``.

    Read from the ``bus`` COORDINATE of the ``profile_{tech}`` master path in
    :func:`paths.prong_pairs`, so the historical-vs-2030 horizon subdir is not
    re-derived here. Coordinates only: the 622 MB USA file is never loaded.

    Verified equal to the covered set derived from master's nodal generators on
    western (``diagnosis/fig2_per_substation_caps_master_drop.csv``: 0 of 506
    rows where ``profiled != in_profile_file``).
    """
    import xarray as xr

    root = Path(master_root)
    out: dict[str, set[str]] = {}
    for pair in prong_pairs(prong):
        if pair.kind != "profile":
            continue
        tech = pair.stage.replace("profile_", "")
        path = root / pair.master
        if not path.exists():
            continue
        with xr.open_dataset(path) as ds:
            if "bus" not in ds.coords:
                continue
            labels = list(ds["bus"].values)
        out[tech] = {s for s in (canonical_sub_id(v) for v in labels) if s is not None}
    return out


def _harness_config() -> dict:
    """The shared harness config, as a dict."""
    import yaml

    with open(REPO / "workflow" / CONFIGFILE) as fh:
        return yaml.safe_load(fh) or {}


def investment_year(master_root: Path, prong: int, config: dict | None = None) -> int:
    """The first investment period both sides built for.

    Master's assembled network first (it is what actually ran), then the harness
    config's ``scenario.planning_horizons[0]``. Neither available is an error,
    not a default: guessing the year silently re-populates the fleet.
    """
    path = Path(master_root) / baseline_assembled_target(prong)
    if path.exists():
        try:
            import pypsa

            periods = list(getattr(pypsa.Network(str(path)), "investment_periods", []) or [])
            if periods:
                return int(periods[0])
        except Exception:  # pragma: no cover - fall through to the config
            pass
    cfg = _harness_config() if config is None else config
    horizons = (cfg.get("scenario") or {}).get("planning_horizons") or []
    if horizons:
        return int(horizons[0])
    raise RuntimeError(
        f"no investment year: {path} has no investment_periods and the harness config "
        "has no scenario.planning_horizons",
    )


def honor_planned_retirements(config: dict | None = None) -> bool:
    """``electricity.honor_planned_retirements`` from the harness config."""
    cfg = _harness_config() if config is None else config
    return bool((cfg.get("electricity") or {}).get("honor_planned_retirements", True))


def master_regions(master_root: Path, interconnect: str):
    """``regions_onshore`` u ``regions_offshore`` as one EPSG:4326 GeoDataFrame."""
    import geopandas as gpd

    root = Path(master_root)
    parts = []
    for name in ("regions_onshore", "regions_offshore"):
        path = root / f"{EQ}/{interconnect}/Geospatial/{name}.geojson"
        if not path.exists():
            continue
        layer = gpd.read_file(path)
        if layer.empty:
            continue
        layer = layer[["geometry"]]
        if layer.crs is not None and str(layer.crs) != "EPSG:4326":
            layer = layer.to_crs("EPSG:4326")
        parts.append(layer)
    if not parts:
        raise FileNotFoundError(f"no master region layers under {root / EQ / interconnect / 'Geospatial'}")
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs="EPSG:4326")


def master_plant_table(master_root: Path) -> pd.DataFrame:
    """Master's ``workflow/resources/powerplants.csv`` (12 MB)."""
    path = Path(master_root) / "resources" / "powerplants.csv"
    if not path.exists():
        raise FileNotFoundError(f"master plant table not found: {path}")
    return pd.read_csv(path, low_memory=False)


def hf26_existing_renewable_drop(art, frames) -> Reconstruction:
    """The entry point a waiver names: HF-26's dropped existing renewable MW.

    ``art`` is a :class:`plots.Artifacts`, ``frames`` the metric frames. Catches
    broadly: any failure sets :attr:`Reconstruction.error` and every lookup
    returns ``None``, so rows stay ``UNEXPLAINED`` — a reconstruction that cannot
    run must never explain anything.
    """
    name = "hf26_existing_renewable_drop"
    try:
        frames = frames or {}
        by_zone = frames.get(ZONE_METRIC)
        by_carrier = frames.get(NATIONAL_METRIC)
        if (by_zone is None or len(by_zone) == 0) and (by_carrier is None or len(by_carrier) == 0):
            return Reconstruction(
                name=name,
                frame=empty_recon_frame(),
                error="no existing-capacity metric frames in this run",
            )
        master_root = Path(art.master_root)
        prong = int(getattr(art, "prong", 2))
        interconnect = str(INTERCONNECT)
        cfg = _harness_config()

        fleet = filter_fleet(
            master_plant_table(master_root),
            investment_year(master_root, prong, cfg),
            interconnect,
            honor_planned_retirements(cfg),
        )
        fleet = filter_to_footprint(fleet, master_regions(master_root, interconnect))
        fleet = assign_to_master_substations(fleet, master_bus_frame(master_root, interconnect))
        recon = reconstruct_drop(fleet, master_profiled_subs(master_root, prong))
        rows = build_rows(recon, by_zone, by_carrier, tables.tolerance_for(ZONE_METRIC))
        return Reconstruction(name=name, frame=rows_frame(rows), rows=rows)
    except Exception as exc:  # deliberately broad: failing closed is the whole point
        return Reconstruction(
            name=name,
            frame=empty_recon_frame(),
            error=f"{type(exc).__name__}: {exc}",
        )


RECONSTRUCTIONS = {"hf26_existing_renewable_drop": hf26_existing_renewable_drop}


class LazyRegistry(Mapping):
    """The reconstructions available to one run, computed on first lookup.

    A ``Mapping`` so ``tables`` can treat it exactly like the plain dict the unit
    tests hand it. Nothing is computed until a waiver actually asks for it, so a
    run whose existing-capacity rows are all equivalent never opens a network;
    :meth:`resolved` is what the markdown section reports on, so writing that
    section cannot itself force a computation.
    """

    def __init__(self, art, frames, builders: Mapping | None = None):
        self._art = art
        self._frames = frames
        self._builders = dict(RECONSTRUCTIONS if builders is None else builders)
        self._cache: dict[str, Reconstruction] = {}

    def __getitem__(self, name: str) -> Reconstruction:
        name = str(name)
        if name not in self._cache:
            self._cache[name] = self._builders[name](self._art, self._frames)
        return self._cache[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._builders)

    def __len__(self) -> int:
        return len(self._builders)

    def resolved(self) -> dict[str, Reconstruction]:
        """Only the reconstructions that were actually computed, in name order."""
        return dict(sorted(self._cache.items()))


_REGISTRY: dict[tuple[str, ...], LazyRegistry] = {}


def enabled() -> bool:
    """``EQ_RECONSTRUCTIONS=0`` turns every reconstruction off (fail-safe)."""
    return str(os.environ.get("EQ_RECONSTRUCTIONS", "1")).strip().lower() not in ("0", "false", "no", "off")


def registry(art, frames) -> Mapping[str, Reconstruction]:
    """The run's reconstructions, memoised and lazy.

    Keyed by ``(prong, develop_root, master_root, interconnect)`` like
    ``plots._RUN_METRICS``, so the table and the figures share one object and no
    artifact is opened twice. ``EQ_RECONSTRUCTIONS=0`` yields an empty mapping,
    which makes every ``reconstruction:`` waiver unavailable and every row it
    would have covered ``UNEXPLAINED``.
    """
    if not enabled():
        return {}
    key = (
        str(getattr(art, "prong", 2)),
        str(getattr(art, "develop_root", "")),
        str(getattr(art, "master_root", "")),
        str(INTERCONNECT),
    )
    if key not in _REGISTRY:
        _REGISTRY[key] = LazyRegistry(art, frames)
    return _REGISTRY[key]


def resolved(reconstructions) -> dict[str, Reconstruction]:
    """The computed reconstructions of any mapping, lazy registry or plain dict."""
    if reconstructions is None:
        return {}
    getter = getattr(reconstructions, "resolved", None)
    if callable(getter):
        return getter()
    return {str(k): v for k, v in dict(reconstructions).items()}


def _main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - a shell entry point
    import argparse
    import json

    from . import plots
    from .build import BASELINE_WORKTREE

    ap = argparse.ArgumentParser(description="Print a reconstruction frame for the built artifacts.")
    ap.add_argument("--prong", type=int, default=2)
    ap.add_argument("--name", default="hf26_existing_renewable_drop")
    ap.add_argument("--develop-root", default=None)
    ap.add_argument("--master-root", default=None)
    ap.add_argument("--json", action="store_true", help="print the national totals as JSON")
    args = ap.parse_args(argv)

    develop_root = Path(args.develop_root or REPO / "workflow")
    master_root = Path(args.master_root or BASELINE_WORKTREE / "workflow")
    # load_artifacts/collect_metrics rather than plots.run_metrics: run_metrics
    # writes run_meta.json, and reading a frozen run must not touch it.
    art = plots.load_artifacts(args.prong, develop_root, master_root)
    frames = plots.collect_metrics(art, [])
    recon = RECONSTRUCTIONS[args.name](art, frames)
    if recon.error:
        print(f"[reconstructions] {args.name}: ERROR {recon.error}")
        return 1
    if args.json:
        print(json.dumps({"name": recon.name, "national": recon.national()}, indent=1, default=float))
    else:
        pd.set_option("display.width", 220)
        print(recon.frame.reset_index(drop=True).to_string(index=False))
        print("\nnational:", recon.national())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
