"""Maps section: 3-panel choropleths (V1-epic | anchor | difference).

Section A maps the prong-1 assembled substation networks at full substation
granularity (~41,000 regions for the whole US, ~2,000 for the western
interconnect): demand, per-carrier generator capacity (p_nom, every nonzero
carrier), maximum installable capacity (p_nom_max — both the pre-network
profile supply curves and the assembled network's finite extendable caps),
capacity-weighted mean availability (p_max_pu) per carrier, and the profile
capacity-factor maps. Section B maps the prong-1 solved zonal networks per
carrier; it renders only when solved artifacts exist.

Reuses ``_plot_choropleth_on_ax`` from ``workflow/scripts/plot_network_maps``
(it does ``regions.set_index("name")`` internally, so regions are passed with
their normalized-name index reset back into a ``name`` column).

Scale awareness: above ``LARGE_REGION_COUNT`` polygons the panels switch to a
PlateCarree axes (cartopy reprojection of ~41k polygons dominates render time
otherwise), sub-pixel simplified geometry, zero region-edge linewidth, and
``MAP_DPI`` (<=100) rasterization to keep the HTML manageable.
"""

from __future__ import annotations

from ..compare import CAPACITY_RTOL, OBJECTIVE_RTOL, RTOL
from ..paths import INTERCONNECT as IC

# Relative (rel) thresholds come from the harness tolerance policy itself
# (compare.RTOL for assembled-stage per-bus vectors (D2-ish), and
# compare.CAPACITY_RTOL / compare.OBJECTIVE_RTOL for the solved stage (D7)),
# so the maps cannot drift from the gate that decides pass/fail.
ABS_FLOOR = 1e-3  # MW — ignore sub-kW noise when flagging a carrier

MAP_DPI = 96  # <= 100 per scale policy: many ~41k-polygon panels must stay small
LARGE_REGION_COUNT = 1000  # above this: no edges, simplified geometry, fast projection
SIMPLIFY_TOL_DEG = 0.02  # sub-pixel at MAP_DPI for a national-extent panel

# Profiled techs whose supply-curve files may exist on both sides.
PROFILE_TECHS = ("onwind", "solar", "offwind_floating")


def render(ctx) -> str:
    import sys

    sys.path.insert(0, str(ctx["repo"] / "workflow" / "scripts"))

    parts: list[str] = ["<h2 id='maps'>Where the differences are: maps</h2>"]

    try:
        import cartopy.crs as ccrs
        import plot_network_maps as pnm
        from matplotlib.colors import Normalize
    except Exception as e:
        parts.append(f"<p>map failed: could not import map plumbing: {e}</p>")
        return "".join(parts)

    plt, np, pd = ctx["plt"], ctx["np"], ctx["pd"]
    labels, norm = ctx["labels"], ctx["norm_label"]

    parts.append(
        "<p>Numbers in tables say <em>how much</em> the two builds differ; maps say "
        "<em>where</em>. Each row below computes the same quantity independently on "
        f"both sides and paints it onto the bus regions: the {labels['candidate']} and "
        f"{labels['anchor']} panels share one color scale, and the third panel shows "
        f"{labels['candidate']} &minus; {labels['anchor']} on a diverging scale centered at "
        "zero &mdash; so any real difference appears as colored structure, and equivalence "
        "appears as a blank (near-white) panel. Gray regions carry no data for that "
        "quantity (no generator of that carrier there, a zero-weight bus, or a bus with "
        "no polygon in the shape file). At national scale (~41,000 regions) region edges "
        "are dropped and figures are rasterized at low dpi so the panels stay legible "
        "and the report stays a manageable size.</p>",
    )

    # ------------------------------------------------------------------ helpers
    import base64
    import io

    def img(fig, caption):
        """Low-dpi variant of ctx['img']: MAP_DPI keeps ~41k-polygon panels small."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=MAP_DPI, bbox_inches="tight")
        plt.close(fig)
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        return f'<figure><img src="{uri}" style="max-width:100%"><figcaption>{caption}</figcaption></figure>'

    _simplified: dict[int, object] = {}

    def plot_regions(regs):
        """Regions handed to the choropleth helper: sub-pixel simplified copy when large."""
        if len(regs) <= LARGE_REGION_COUNT:
            return regs
        key = id(regs)
        if key not in _simplified:
            s = regs.copy()
            s["geometry"] = s.geometry.simplify(SIMPLIFY_TOL_DEG, preserve_topology=True)
            _simplified[key] = s
        return _simplified[key]

    def three_panel(nc, na, regs_c, regs_a, vals_c, vals_a, title, unit, diff_fill=0.0):
        """One figure: candidate abs | anchor abs | difference. Returns (html, dmax)."""
        vals_c = vals_c.astype(float)
        vals_a = vals_a.astype(float)
        idx = vals_c.index.union(vals_a.index)
        if diff_fill is None:  # intensive quantity: NaN where a side is missing
            diff = vals_c.reindex(idx) - vals_a.reindex(idx)
        else:  # extensive quantity: absence means zero
            diff = vals_c.reindex(idx, fill_value=diff_fill) - vals_a.reindex(idx, fill_value=diff_fill)
        finite = diff.values[np.isfinite(diff.values)]
        dmax = float(np.abs(finite).max()) if finite.size else 0.0
        dmax_rel_pct = None
        try:
            _i = diff.abs().idxmax()
            _a = abs(float(vals_a.reindex(diff.index).get(_i, float("nan"))))
            if np.isfinite(_a) and _a > 0:
                dmax_rel_pct = abs(float(diff.get(_i))) / _a * 100.0
        except Exception:
            pass
        vmax = float(
            max(
                vals_c.max() if len(vals_c) else 0.0,
                vals_a.max() if len(vals_a) else 0.0,
                1e-9,
            ),
        )
        dlim = max(dmax, 1e-9)
        large = max(len(regs_c), len(regs_a)) > LARGE_REGION_COUNT
        lon = float(nc.buses.x.mean())
        # PlateCarree keeps the region->axes transform trivial: EqualEarth reprojection
        # of ~41k polygons per panel dominates render time at national scale.
        proj = ccrs.PlateCarree() if large else ccrs.EqualEarth(lon)
        fig, axes = plt.subplots(
            1,
            3,
            subplot_kw={"projection": proj},
            figsize=(15, 4.5),
        )
        panels = [
            (axes[0], nc, regs_c, vals_c, "viridis", 0.0, vmax, labels["candidate"]),
            (axes[1], na, regs_a, vals_a, "viridis", 0.0, vmax, labels["anchor"]),
            (
                axes[2],
                nc,
                regs_c,
                diff,
                "RdBu_r",
                -dlim,
                dlim,
                f"{labels['candidate']} \u2212 {labels['anchor']}",
            ),
        ]
        for ax, n, regs, vals, cmap, vmin_, vmax_, name in panels:
            pnm._plot_choropleth_on_ax(
                n,
                vals,
                plot_regions(regs).reset_index(),  # helper set_index('name')s internally
                ax,
                cmap=cmap,
                vmin=vmin_,
                vmax=vmax_,
                show_lines=False,
            )
            if large:  # ~41k polygons: edge lines would drown the fill colors
                for coll in ax.collections:
                    coll.set_linewidth(0.0)
            ax.set_title(name, fontsize=10)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin_, vmax_))
            fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
        fig.suptitle(title, fontsize=12)
        same = dmax <= max(1e-6, RTOL * vmax)
        note = (
            "identical within tolerance &mdash; the difference panel is blank by construction"
            if same
            else f"largest per-region difference: {dmax:,.4g} {unit}"
            + (f" ({dmax_rel_pct:,.3g}% of the anchor value there)" if dmax_rel_pct is not None else "")
        )
        return img(fig, f"{title} &mdash; {note}"), dmax

    def guarded(fn, what):
        try:
            return fn()
        except Exception as e:
            return f"<p>map failed ({what}): {e}</p>"

    def per_bus_load(n):
        """Mean load (MW) per normalized bus id from loads_t.p_set."""
        m = n.loads_t.p_set.mean()
        bus = n.loads.bus.reindex(m.index)
        # candidate load names ARE bus ids; anchor names are '35827 AC' -> map via loads.bus
        bus = bus.fillna(pd.Series(list(m.index), index=m.index))
        s = m.groupby(bus.map(norm)).sum()
        s.index = s.index.map(str)
        return s

    def pnom_by_carrier_bus(n, attr):
        g = n.generators
        return g.groupby([g.carrier, g.bus.map(norm)])[attr].sum()

    def ext_pnom_max_by_carrier_bus(n):
        """Per-(carrier, bus) sums of FINITE p_nom_max over extendable generators."""
        g = n.generators
        ext = g[g.p_nom_extendable]
        finite = ext.p_nom_max[np.isfinite(ext.p_nom_max.astype(float))]
        if finite.empty:
            return pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []]))
        sub = ext.loc[finite.index]
        return finite.groupby([sub.carrier, sub.bus.map(norm)]).sum()

    def mean_pu_by_carrier_bus(n):
        """Per-(carrier, bus) capacity-weighted mean of the time-mean p_max_pu.

        Weights are p_nom; zero-weight buses are guarded out (dropped -> gray).
        """
        tv = n.generators_t.p_max_pu
        if tv.empty:
            return pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []]))
        m = tv.mean()
        g = n.generators
        cols = m.index.intersection(g.index)
        m = m[cols]
        w = g.p_nom.reindex(cols).astype(float)
        grp = [g.carrier.reindex(cols), g.bus.reindex(cols).map(norm)]
        num = (m * w).groupby(grp).sum()
        den = w.groupby(grp).sum()
        return (num / den.where(den > 0)).dropna()

    def carrier_vec(stacked, carrier):
        if carrier in stacked.index.get_level_values(0):
            return stacked.loc[carrier]
        return pd.Series(dtype=float)

    def carriers_by_total(pc, pa):
        """Union of carriers ordered by summed total (both sides) descending."""
        cars = set(pc.index.get_level_values(0)) | set(pa.index.get_level_values(0))
        totals = {c: float(carrier_vec(pc, c).sum() + carrier_vec(pa, c).sum()) for c in cars}
        return sorted(totals, key=lambda c: (-totals[c], c)), totals

    def differing_carriers(pc, pa, rtol):
        """Split union of carriers into (differing, identical) per-bus-vector-wise."""
        differing, identical = [], []
        for car in sorted(set(pc.index.get_level_values(0)) | set(pa.index.get_level_values(0))):
            a = carrier_vec(pc, car)
            b = carrier_vec(pa, car)
            idx = a.index.union(b.index)
            a = a.reindex(idx, fill_value=0.0).astype(float)
            b = b.reindex(idx, fill_value=0.0).astype(float)
            d = (a - b).abs()
            scale = np.maximum(a.abs(), b.abs())
            if bool(((d > rtol * scale) & (d > ABS_FLOOR)).any()):
                differing.append(car)
            else:
                identical.append(car)
        return differing, identical

    # ---------------------------------------------------------- A. assembled
    parts.append("<h3>Assembled substation network (prong 1, full substation granularity)</h3>")
    parts.append(
        "<p>This is the last stage where both builds exist at full substation "
        "granularity, so it is where a spatial wiring mistake (wrong bus keying, "
        "dropped region, shifted profile) would be most visible.</p>",
    )

    nc = na = regs_c = regs_a = None
    try:
        nc = ctx["load_network"](ctx["cand_root"] / f"resources/equivalence/networks/{IC}/elec_s_l_pp.pkl")
        na = ctx["load_network"](ctx["anch_root"] / f"resources/equivalence/{IC}/elec_s.nc")
        regs_c = ctx["load_regions"]("candidate", "")
        regs_a = ctx["load_regions"]("anchor", "")
    except Exception as e:
        parts.append(f"<p>map failed: could not load assembled-stage inputs: {e}</p>")

    if nc is not None and regs_c is not None:
        parts.append(
            f"<p>Scale of this stage: {len(nc.buses):,} buses painted onto "
            f"{len(regs_c):,} onshore regions ({labels['candidate']} side; "
            f"{labels['anchor']} has {len(na.buses):,} buses / {len(regs_a):,} regions).</p>",
        )

        # A1. demand — always rendered: a blank diff panel PROVES sameness.
        def _demand():
            html, _ = three_panel(
                nc,
                na,
                regs_c,
                regs_a,
                per_bus_load(nc),
                per_bus_load(na),
                "Demand: mean load per substation (MW)",
                "MW",
            )
            return (
                "<h4>Demand</h4><p>Mean electric load at each substation over the "
                "modeled year. The demand pipeline was rebuilt around the clustered "
                "network in V1-epic (and a &minus;6.3% conservation bug was caught and "
                "fixed by this harness), so this map is the visual proof the fixed "
                "pipeline lands the same megawatts on the same buses.</p>" + html
            )

        parts.append(guarded(_demand, "assembled demand"))

        # A2. generator p_nom — EVERY carrier with nonzero capacity gets a map row.
        def _pnom():
            pc = pnom_by_carrier_bus(nc, "p_nom")
            pa = pnom_by_carrier_bus(na, "p_nom")
            order, totals = carriers_by_total(pc, pa)
            order = [c for c in order if totals[c] > 0.0]
            differing, identical = differing_carriers(pc, pa, RTOL)
            out = [
                "<h4>Existing generator capacity (p_nom) by carrier</h4>"
                "<p>Installed capacity attached to each substation, split by carrier. "
                "Every carrier with nonzero capacity on either side gets a map row, "
                "largest fleet first &mdash; at this scale a blank difference panel is "
                "the proof of equivalence, not a reason to skip the map.</p>",
            ]
            if differing:
                out.append(
                    f"<p>Summary: per-bus vectors differ beyond {RTOL:.0e} relative for "
                    "<b>"
                    + ", ".join(differing)
                    + "</b>; identical within tolerance for "
                    + (", ".join(identical) if identical else "none")
                    + ".</p>",
                )
            else:
                out.append(
                    f"<p>Summary: all {len(identical)} carriers are identical within "
                    f"{RTOL:.0e} relative per bus ("
                    + ", ".join(identical)
                    + ") &mdash; every difference panel below should be blank.</p>",
                )
            for car in order:
                a = carrier_vec(pc, car)
                b = carrier_vec(pa, car)
                out.append(
                    guarded(
                        lambda a=a, b=b, car=car: three_panel(
                            nc,
                            na,
                            regs_c,
                            regs_a,
                            a,
                            b,
                            f"Generator p_nom per substation: {car} (MW)",
                            "MW",
                        )[0],
                        f"p_nom map for {car}",
                    ),
                )
            return "".join(out)

        parts.append(guarded(_pnom, "assembled p_nom"))

        # A3. maximum installable capacity (p_nom_max) — profile files + network.
        def _pnom_max():
            import xarray as xr

            out = [
                "<h4>Maximum installable capacity (p_nom_max)</h4>"
                "<p>The caps that bound the expansion optimizer. Two views: the "
                "renewable supply-curve files as produced by build_renewable_profiles "
                "(pre-network), and the same caps as they land on extendable "
                "generators in the assembled network.</p>",
                "<h5>Renewable supply curves (profile files, pre-network)</h5>"
                "<p>Per-substation developable capacity from the profile files "
                "&mdash; the land-availability rollup before any network assembly.</p>",
            ]
            for tech in PROFILE_TECHS:
                pcand = ctx["cand_root"] / f"resources/equivalence/profiles/{IC}/2030/profile_{tech}_s.nc"
                panch = ctx["anch_root"] / f"resources/equivalence/{IC}/2030/profile_{tech}.nc"
                if not (pcand.exists() and panch.exists()):
                    continue

                def _one(pcand=pcand, panch=panch, tech=tech):
                    with xr.open_dataset(pcand) as dc, xr.open_dataset(panch) as da:
                        vc = dc["p_nom_max"].to_pandas()
                        va = da["p_nom_max"].to_pandas()
                    vc.index = vc.index.map(norm)
                    va.index = va.index.map(norm)
                    return three_panel(
                        nc,
                        na,
                        regs_c,
                        regs_a,
                        vc,
                        va,
                        f"Developable capacity p_nom_max per substation: {tech} (MW)",
                        "MW",
                    )[0]

                out.append(guarded(_one, f"profile p_nom_max map for {tech}"))

            out.append(
                "<h5>Extendable generators in the assembled network</h5>"
                "<p>Per-bus sums of finite p_nom_max over extendable generators. "
                "Carriers whose cap is everywhere infinite (unbounded, e.g. gas) "
                "carry no spatial information and are listed instead of mapped.</p>",
            )
            mc = ext_pnom_max_by_carrier_bus(nc)
            ma = ext_pnom_max_by_carrier_bus(na)
            order, _ = carriers_by_total(mc, ma)
            ext_cars = set(nc.generators.carrier[nc.generators.p_nom_extendable]) | set(
                na.generators.carrier[na.generators.p_nom_extendable],
            )
            unbounded = sorted(ext_cars - set(order))
            if unbounded:
                out.append(
                    "<p>Unbounded (p_nom_max infinite everywhere, nothing to map): " + ", ".join(unbounded) + ".</p>",
                )
            if not order:
                out.append("<p>No extendable carrier carries a finite p_nom_max at this stage.</p>")
            for car in order:
                a = carrier_vec(mc, car)
                b = carrier_vec(ma, car)
                out.append(
                    guarded(
                        lambda a=a, b=b, car=car: three_panel(
                            nc,
                            na,
                            regs_c,
                            regs_a,
                            a,
                            b,
                            f"Extendable p_nom_max per substation: {car} (MW)",
                            "MW",
                        )[0],
                        f"network p_nom_max map for {car}",
                    ),
                )
            return "".join(out)

        parts.append(guarded(_pnom_max, "assembled p_nom_max"))

        # A4. capacity-weighted mean availability per carrier (network stage).
        def _mean_pu():
            cc = mean_pu_by_carrier_bus(nc)
            ca = mean_pu_by_carrier_bus(na)
            # Order consistently with the capacity section: installed MW descending.
            order, _ = carriers_by_total(
                pnom_by_carrier_bus(nc, "p_nom"),
                pnom_by_carrier_bus(na, "p_nom"),
            )
            tv_cars = set(cc.index.get_level_values(0)) | set(ca.index.get_level_values(0))
            cars = [c for c in order if c in tv_cars] + sorted(tv_cars - set(order))
            out = [
                "<h4>Average availability: capacity-weighted mean p_max_pu by carrier</h4>"
                "<p>For every carrier with time-varying availability in the assembled "
                "network: the time-mean of each generator's p_max_pu, averaged onto "
                "its bus weighted by p_nom. This is the network-stage complement of "
                "the profile maps below &mdash; it covers thermal derates and hydro "
                "as well as wind/solar. Buses with zero installed capacity in a "
                "carrier carry no weight and stay gray. Intensive quantity: the "
                "difference is only taken where both sides have the bus.</p>",
            ]
            if not cars:
                out.append("<p>No carrier has time-varying availability at this stage.</p>")
            for car in cars:
                a = carrier_vec(cc, car)
                b = carrier_vec(ca, car)
                out.append(
                    guarded(
                        lambda a=a, b=b, car=car: three_panel(
                            nc,
                            na,
                            regs_c,
                            regs_a,
                            a,
                            b,
                            f"Capacity-weighted mean p_max_pu: {car}",
                            "p.u.",
                            diff_fill=None,  # intensive: NaN where a side lacks the bus
                        )[0],
                        f"mean p_max_pu map for {car}",
                    ),
                )
            return "".join(out)

        parts.append(guarded(_mean_pu, "assembled mean p_max_pu"))

        # A5. renewable capacity factors from the profile files — always rendered.
        def _profiles():
            import xarray as xr

            out = [
                "<h4>Renewable capacity factors (profile files, pre-network)</h4>"
                "<p>Time-mean capacity factor per substation from the renewable "
                "profile files &mdash; the direct output of the relocated "
                "build_renewable_profiles stage, before any network assembly. "
                "Identical maps here mean the simplify-early refactor feeds the "
                "same weather to the same places. <b>Blank in-state regions "
                "carry no profile because NREL's reference land-access data "
                "genuinely excludes them</b> (verified against the source "
                "availability raster: median 0.0% developable in missing "
                "regions vs 8.1% in kept ones; urban coast, federal/wilderness "
                "Sierra) &mdash; modeled resource exclusion, not missing data. "
                "<b>Exception:</b> the giant out-of-state border regions are "
                "blank for a different reason &mdash; the NREL caps rollup is "
                "keyed to the national substation tessellation and the "
                "CA-focus busmap silently drops out-of-state entries "
                "(build_renewable_profiles.py:50), stranding ~13.4% of the "
                "West's developable wind area; flagged as a data-model gap "
                "shared identically by both sides (0.0% difference in the "
                "excluded set, so equivalence is unaffected).</p>",
            ]
            for tech in PROFILE_TECHS:
                pcand = ctx["cand_root"] / f"resources/equivalence/profiles/{IC}/2030/profile_{tech}_s.nc"
                panch = ctx["anch_root"] / f"resources/equivalence/{IC}/2030/profile_{tech}.nc"
                if not (pcand.exists() and panch.exists()):
                    continue

                def _one(pcand=pcand, panch=panch, tech=tech):
                    with xr.open_dataset(pcand) as dc, xr.open_dataset(panch) as da:
                        vc = dc["profile"].mean("time").to_pandas()
                        va = da["profile"].mean("time").to_pandas()
                    vc.index = vc.index.map(norm)
                    va.index = va.index.map(norm)
                    return three_panel(
                        nc,
                        na,
                        regs_c,
                        regs_a,
                        vc,
                        va,
                        f"Mean capacity factor: {tech}",
                        "CF",
                        diff_fill=None,  # intensive: NaN where a side lacks the bus
                    )[0]

                out.append(guarded(_one, f"{tech} CF map"))
            return "".join(out)

        parts.append(guarded(_profiles, "profile CF maps"))

    # ------------------------------------------------------------- B. solved
    parts.append(f"<h3>Solved network (prong 1, {ctx['clusters']} zones)</h3>")
    parts.append(
        "<p>After clustering to "
        f"{ctx['clusters']} zones and solving, the question changes from "
        "&lsquo;is the input data identical?&rsquo; to &lsquo;does the optimizer build the same "
        "system?&rsquo;. Zone-level optimal capacity (p_nom_opt) per carrier is mapped "
        f"only where the two sides disagree by more than {CAPACITY_RTOL:.1%} (the "
        "harness's solved-stage tolerance).</p>",
    )

    def _solved():
        pair = ctx["prong_pairs"](1)[-1]  # solved_network pair (when the run solves)
        if pair.stage != "solved_network":
            return (
                "<p>This run stops at the assembled stage (until="
                f"{ctx['until'] or 'assembled'}), so there are no solved artifacts to map.</p>"
            )
        cand_path = ctx["cand_root"] / pair.candidate
        anch_path = ctx["anch_root"] / pair.anchor
        if not (cand_path.exists() and anch_path.exists()):
            return "<p>Solved networks not found on disk &mdash; section skipped.</p>"
        nsc = ctx["load_network"](cand_path)
        nsa = ctx["load_network"](anch_path)
        r4c = ctx["load_regions"]("candidate", "", ctx["clusters"])
        r4a = ctx["load_regions"]("anchor", "", ctx["clusters"])
        out = []
        try:
            oc, oa = float(nsc.objective), float(nsa.objective)
            rel = abs(oc - oa) / max(abs(oa), 1e-9)
            out.append(
                f"<p>Total system cost (solver objective): {labels['candidate']} "
                f"{oc:,.0f} vs {labels['anchor']} {oa:,.0f} &mdash; relative difference "
                f"{rel:.1e}, within the {OBJECTIVE_RTOL:.1%} solved-stage tolerance."
                if rel <= OBJECTIVE_RTOL
                else f"<p>Total system cost (solver objective): {labels['candidate']} "
                f"{oc:,.0f} vs {labels['anchor']} {oa:,.0f} &mdash; relative difference "
                f"{rel:.1e}, OUTSIDE the {OBJECTIVE_RTOL:.1%} tolerance.",
            )
            out[-1] += "</p>"
        except Exception as e:
            out.append(f"<p>map failed (system cost note): {e}</p>")
        pc = pnom_by_carrier_bus(nsc, "p_nom_opt")
        pa = pnom_by_carrier_bus(nsa, "p_nom_opt")
        differing, identical = differing_carriers(pc, pa, CAPACITY_RTOL)
        if identical:
            out.append(
                "<p>Zone-level optimal capacity identical within tolerance for: " + ", ".join(identical) + ".</p>",
            )
        if not differing:
            out.append(
                "<p>No carrier's zonal build-out differs beyond tolerance &mdash; the "
                "optimizer reaches the same expansion plan on both sides.</p>",
            )
        for car in differing:
            a = carrier_vec(pc, car)
            b = carrier_vec(pa, car)
            out.append(
                guarded(
                    lambda a=a, b=b, car=car: three_panel(
                        nsc,
                        nsa,
                        r4c,
                        r4a,
                        a,
                        b,
                        f"Optimal capacity p_nom_opt per zone: {car} (MW)",
                        "MW",
                    )[0],
                    f"solved p_nom_opt map for {car}",
                ),
            )
        return "".join(out)

    parts.append(guarded(_solved, "solved-stage maps"))

    return "".join(parts)
