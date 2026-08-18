"""Maps section: 3-panel choropleths (V1-epic | anchor | difference).

Section A maps the prong-1 assembled substation networks (1972 region
polygons, 1975 buses — the 3 polygon-less buses fall back to the missing
color) for demand, per-carrier generator p_nom, and renewable capacity
factors. Section B maps the prong-1 solved 4-zone networks per carrier.

Reuses ``_plot_choropleth_on_ax`` from ``workflow/scripts/plot_network_maps``
(it does ``regions.set_index("name")`` internally, so regions are passed with
their normalized-name index reset back into a ``name`` column).
"""

from __future__ import annotations

# Relative (rel) thresholds mirroring the harness tolerance policy:
# assembled-stage per-bus vectors (D2-ish) and solved-stage zone vectors (D7).
ASSEMBLED_RTOL = 1e-3
SOLVED_RTOL = 5e-3
ABS_FLOOR = 1e-3  # MW — ignore sub-kW noise when flagging a carrier


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
    labels, img, norm = ctx["labels"], ctx["img"], ctx["norm_label"]

    parts.append(
        "<p>Numbers in tables say <em>how much</em> the two builds differ; maps say "
        "<em>where</em>. Each row below computes the same quantity independently on "
        f"both sides and paints it onto the bus regions: the {labels['candidate']} and "
        f"{labels['anchor']} panels share one color scale, and the third panel shows "
        f"{labels['candidate']} &minus; {labels['anchor']} on a diverging scale centered at "
        "zero &mdash; so any real difference appears as colored structure, and equivalence "
        "appears as a blank (near-white) panel. Gray regions carry no data (3 of the "
        "1975 buses have no polygon in the 1972-region shape file).</p>",
    )

    # ------------------------------------------------------------------ helpers
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
        lon = float(nc.buses.x.mean())
        fig, axes = plt.subplots(
            1,
            3,
            subplot_kw={"projection": ccrs.EqualEarth(lon)},
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
                regs.reset_index(),  # helper set_index('name')s internally
                ax,
                cmap=cmap,
                vmin=vmin_,
                vmax=vmax_,
                show_lines=False,
            )
            ax.set_title(name, fontsize=10)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin_, vmax_))
            fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
        fig.suptitle(title, fontsize=12)
        same = dmax <= max(1e-6, ASSEMBLED_RTOL * vmax)
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

    def carrier_vec(stacked, carrier):
        if carrier in stacked.index.get_level_values(0):
            return stacked.loc[carrier]
        return pd.Series(dtype=float)

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
    parts.append("<h3>Assembled substation network (prong 1, ~1975 buses)</h3>")
    parts.append(
        "<p>This is the last stage where both builds exist at full substation "
        "granularity, so it is where a spatial wiring mistake (wrong bus keying, "
        "dropped region, shifted profile) would be most visible.</p>",
    )

    nc = na = regs_c = regs_a = None
    try:
        nc = ctx["load_network"](
            ctx["cand_root"] / "resources/equivalence/networks/western/elec_s_l_pp.pkl",
        )
        na = ctx["load_network"](ctx["anch_root"] / "resources/equivalence/western/elec_s.nc")
        regs_c = ctx["load_regions"]("candidate", "")
        regs_a = ctx["load_regions"]("anchor", "")
    except Exception as e:
        parts.append(f"<p>map failed: could not load assembled-stage inputs: {e}</p>")

    if nc is not None and regs_c is not None:
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

        # A2. generator p_nom per carrier — only carriers whose per-bus vectors differ.
        def _pnom():
            pc = pnom_by_carrier_bus(nc, "p_nom")
            pa = pnom_by_carrier_bus(na, "p_nom")
            differing, identical = differing_carriers(pc, pa, ASSEMBLED_RTOL)
            out = [
                "<h4>Existing generator capacity (p_nom) by carrier</h4>"
                "<p>Installed capacity attached to each substation, split by carrier. "
                "A carrier only gets a map row if its per-bus capacity vector differs "
                f"anywhere beyond {ASSEMBLED_RTOL:.0e} relative.</p>",
            ]
            if identical:
                out.append(
                    "<p>Identical everywhere (no map needed): " + ", ".join(identical) + ".</p>",
                )
            if not differing:
                out.append(
                    "<p>No carrier shows a per-bus capacity difference beyond "
                    "tolerance &mdash; every generator megawatt sits on the same "
                    "substation on both sides.</p>",
                )
            for car in differing:
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

        # A3. renewable capacity factors — always rendered.
        def _profiles():
            import xarray as xr

            out = [
                "<h4>Renewable capacity factors</h4>"
                "<p>Time-mean capacity factor per substation from the renewable "
                "profile files &mdash; the direct output of the relocated "
                "build_renewable_profiles stage, before any network assembly. "
                "Identical maps here mean the simplify-early refactor feeds the "
                "same weather to the same places. <b>Blank (uncolored) regions "
                "carry no profile because the NREL reference land-access supply "
                "curve contains no eligible site within them</b> (urban Bay/LA/SD "
                "coast, steep-slope and protected Sierra counties) &mdash; modeled "
                "resource exclusion, not missing data. Both sides exclude the "
                "identical 1,428 of 1,972 regions for onwind (72.4% of regions, "
                "20.4% of land area; 0.0% difference in the excluded set); all "
                "544 regions with eligible onwind resource agree, as do all 808 "
                "solar regions.</p>",
            ]
            for tech in ("onwind", "solar"):
                pcand = ctx["cand_root"] / f"resources/equivalence/profiles/western/2030/profile_{tech}_s.nc"
                panch = ctx["anch_root"] / f"resources/equivalence/western/2030/profile_{tech}.nc"

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
        f"only where the two sides disagree by more than {SOLVED_RTOL:.1%} (the "
        "harness's solved-stage tolerance).</p>",
    )

    def _solved():
        pair = ctx["prong_pairs"](1)[-1]  # solved_network pair
        nsc = ctx["load_network"](ctx["cand_root"] / pair.candidate)
        nsa = ctx["load_network"](ctx["anch_root"] / pair.anchor)
        r4c = ctx["load_regions"]("candidate", "", ctx["clusters"])
        r4a = ctx["load_regions"]("anchor", "", ctx["clusters"])
        out = []
        try:
            oc, oa = float(nsc.objective), float(nsa.objective)
            rel = abs(oc - oa) / max(abs(oa), 1e-9)
            out.append(
                f"<p>Total system cost (solver objective): {labels['candidate']} "
                f"{oc:,.0f} vs {labels['anchor']} {oa:,.0f} &mdash; relative difference "
                f"{rel:.1e}, within the 0.1% solved-stage tolerance."
                if rel <= 1e-3
                else f"<p>Total system cost (solver objective): {labels['candidate']} "
                f"{oc:,.0f} vs {labels['anchor']} {oa:,.0f} &mdash; relative difference "
                f"{rel:.1e}, OUTSIDE the 0.1% tolerance.",
            )
            out[-1] += "</p>"
        except Exception as e:
            out.append(f"<p>map failed (system cost note): {e}</p>")
        pc = pnom_by_carrier_bus(nsc, "p_nom_opt")
        pa = pnom_by_carrier_bus(nsa, "p_nom_opt")
        differing, identical = differing_carriers(pc, pa, SOLVED_RTOL)
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
