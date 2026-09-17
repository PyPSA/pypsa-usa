"""Build the "simplify-early" deck: figures, then the .pptx, then verify it.

Run (Sherlock, inside an srun allocation - never on the login node):

    export UV_CACHE_DIR="$GROUP_SCRATCH/kamran/cache/uv"
    export UV_LINK_MODE=copy
    ml load gcc/12.4.0 system/git/2.45.1 devel/uv/0.10.8
    cd <worktree> && uv run --with python-pptx python docs/decks/simplify_early/build_deck.py

Outputs, all next to this file:
    figures/*.png   one per figure
    figures/*.csv   the exact plotted data, one per figure
    simplify_early.pptx

Every number that reaches a slide comes from ``measurements.py``, which records
the artifact each one was read from. Quantities that were never measured are
labelled "NOT MEASURED" on the slide rather than filled in.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

import figures  # noqa: E402
import measurements as M  # noqa: E402

OUT = HERE / "simplify_early.pptx"
FIGDIR = HERE / "figures"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

INK = RGBColor(0x0B, 0x0B, 0x0B)
INK2 = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x8A, 0x89, 0x85)
MASTER_C = RGBColor(0x2A, 0x78, 0xD6)
DEVELOP_C = RGBColor(0xEB, 0x68, 0x34)

# --------------------------------------------------------------------------
# derived numbers (computed, never typed twice)
# --------------------------------------------------------------------------
T = M.heavy_totals(M.USA_RULES)
NEW_S = sum(r.develop_s for r in M.USA_DEVELOP_ONLY)
HEAVY_SPEEDUP = T["master_s"] / T["develop_s"]
SEGMENT_SPEEDUP = T["master_s"] / (T["develop_s"] + NEW_S)
WEST = M.heavy_totals(M.WESTERN_RULES)
WEST_SPEEDUP = WEST["master_s"] / WEST["develop_s"]


def _rule(needle: str):
    """Find a USA rule by a substring of its display label (newlines stripped)."""
    for r in M.USA_RULES:
        if needle in r.stage.replace("\n", " "):
            return r
    raise KeyError(needle)


ADD_ELEC = _rule("add_electricity")
ADD_DEM = _rule("add_demand")
DEMAND = _rule("build_electrical")
ONWIND = _rule("(onwind)")
SOLAR = _rule("(solar)")
FUEL = _rule("build_fuel_prices")
CLUSTER_NET = _rule("cluster_network")
SOLVE = _rule("solve_network")


# --------------------------------------------------------------------------
# slide helpers
# --------------------------------------------------------------------------


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _title(slide, text, size=26, top=0.30, color=INK):
    box = slide.shapes.add_textbox(Inches(0.55), Inches(top), Inches(12.2), Inches(0.85))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = True
    p.font.color.rgb = color
    return box


def _subtitle(slide, text, top=1.08, size=13, color=INK2):
    box = slide.shapes.add_textbox(Inches(0.55), Inches(top), Inches(12.2), Inches(0.6))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    return box


def _notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text.strip()


def _picture(slide, png: Path, top=1.62, max_h=5.45, max_w=12.3):
    """Insert a PNG scaled to fit the content box, horizontally centred."""
    from PIL import Image

    with Image.open(png) as im:
        w_px, h_px = im.size
    aspect = w_px / h_px
    h = max_h
    w = h * aspect
    if w > max_w:
        w = max_w
        h = w / aspect
    left = (13.333 - w) / 2
    return slide.shapes.add_picture(str(png), Inches(left), Inches(top), Inches(w), Inches(h))


def _bullets(slide, items, top=1.75, left=0.7, width=12.0, size=15, gap=10):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(5.2))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if isinstance(item, tuple):
            text, lvl, bold, col = item
        else:
            text, lvl, bold, col = item, 0, False, INK2
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("- " if lvl == 0 else "    - ") + text
        p.font.size = Pt(size if lvl == 0 else size - 2)
        p.font.bold = bold
        p.font.color.rgb = col
        p.space_after = Pt(gap)
    return box


def _footer(slide, text):
    box = slide.shapes.add_textbox(Inches(0.55), Inches(7.02), Inches(12.2), Inches(0.35))
    tf = box.text_frame
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(9)
    p.font.color.rgb = MUTED
    return box


SRC_FOOT = (
    "Source: snakemake benchmark TSVs, USA equivalence case, 2026-09-15 "
    "(develop d773abc / master-benchmark ca73014), Sherlock serc"
)


# --------------------------------------------------------------------------
# the deck
# --------------------------------------------------------------------------


def build(figs: dict) -> Presentation:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    # ---- 1. title -------------------------------------------------------
    s = _blank(prs)
    box = s.shapes.add_textbox(Inches(0.9), Inches(2.25), Inches(11.5), Inches(1.5))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Simplify-early"
    p.font.size = Pt(50)
    p.font.bold = True
    p.font.color.rgb = INK
    p = tf.add_paragraph()
    p.text = "Reordering the PyPSA-USA DAG so the network shrinks before the heavy rules run"
    p.font.size = Pt(21)
    p.font.color.rgb = INK2
    box2 = s.shapes.add_textbox(Inches(0.9), Inches(4.35), Inches(11.5), Inches(2.0))
    tf2 = box2.text_frame
    tf2.word_wrap = True
    for i, line in enumerate(
        [
            f"master-benchmark {M.MASTER_COMMIT[:7]}  vs  develop {M.DEVELOP_COMMIT[:7]}",
            f"Case: {M.CASE}",
            f"Measured on {M.MACHINE}, 2026-09-15",
        ]
    ):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.text = line
        p.font.size = Pt(13)
        p.font.color.rgb = INK2 if i == 0 else MUTED
        p.space_after = Pt(6)
    _notes(
        s,
        f"""
This deck is about one structural change to the workflow: the order of the
simplification/clustering rules relative to the per-bus heavy rules. It is not
about the equivalence bugs; those live in the run cards.

Branches: master-benchmark {M.MASTER_COMMIT} is the baseline that actually
builds; develop {M.DEVELOP_COMMIT} is the refactored branch.

Case: {M.CASE}.

All timing and memory numbers come from snakemake `benchmark:` TSVs written on
2026-09-15 on Sherlock's serc partition. {M.TOTAL_PIPELINE_CAVEAT}
""",
    )

    # ---- 2. the change in one line --------------------------------------
    s = _blank(prs)
    _title(s, "The change in one line")
    _bullets(
        s,
        [
            ("master reduces LAST.", 0, True, MASTER_C),
            (
                "build_base_network -> build_bus_regions -> build_renewable_profiles, "
                "build_electrical_demand, add_demand, add_electricity ALL at nodal resolution "
                "-> simplify_network -> cluster_network.",
                1,
                False,
                INK2,
            ),
            ("develop reduces FIRST.", 0, True, DEVELOP_C),
            (
                "simplify_network is split into aggregate_to_substations (topology) and "
                "cluster_resources ({simpl} kmeans), and both move to the front. The heavy "
                "rules then consume elec_s{simpl}.",
                1,
                False,
                INK2,
            ),
            (
                f"On the USA case the heavy rules go from {M.HEAVY_RULE_RESOLUTION['master']:,} buses "
                f"to {M.HEAVY_RULE_RESOLUTION['develop']:,}, a {M.REDUCTION_FACTOR:,.0f}x reduction in the "
                "dimension every per-bus array is keyed on.",
                0,
                True,
                INK,
            ),
            (
                "Nothing else about the model changes: same config, same data, same final "
                "cluster count. The equivalence harness exists to hold that claim to account.",
                0,
                False,
                INK2,
            ),
        ],
        top=1.85,
        size=16,
        gap=16,
    )
    _notes(
        s,
        """
The load-bearing sentence is in the repo's own CLAUDE.md, section "Architecture:
the DAG and the resources/ layout": "topology aggregation and {simpl} kmeans
clustering happen before the per-bus heavy rules. The clustered network is the
input to demand/RE/electricity assembly, not the substation-level network."

Note the split is two rules, not one moved rule. master's simplify_network did
both the nodal->substation aggregation and the kmeans; develop separates them so
the topology step (aggregate_to_substations) and the resource step
(cluster_resources) can be reasoned about and cached independently.

A second consequence, visible in the rule signatures: the {simpl} wildcard is now
load-bearing on nearly every rule downstream of cluster_resources. Any new
per-bus rule must carry _s{simpl} on its inputs and outputs.
""",
    )

    # ---- 3. master DAG --------------------------------------------------
    s = _blank(prs)
    _title(s, "master: the heavy rules run on the full nodal network")
    _subtitle(
        s,
        "Bus counts under each stage. simplify_network does substation aggregation AND "
        "{simpl} kmeans, in one rule, after all the per-bus work is finished.",
    )
    _picture(s, figs["fig_dag_master"])
    _footer(s, "Rule graph read from .worktrees/master-benchmark/workflow/rules/build_electricity.smk; bus counts from the busmap CSVs")
    _notes(
        s,
        f"""
Read off master's build_electricity.smk:
  build_base_network   -> elec_base_network.nc
  build_bus_regions    -> regions_onshore.geojson (one region per nodal bus)
  build_renewable_profiles  input regions = regions_onshore.geojson  (NODAL)
  add_demand           input  elec_base_network.nc -> elec_base_network_dem.nc (NODAL)
  add_electricity      input  elec_base_network_dem.nc -> elec_base_network_l_pp.pkl (NODAL)
  simplify_network     input  elec_base_network_l_pp.pkl + bus2sub + sub -> elec_s{{simpl}}.nc
  cluster_network      elec_s{{simpl}}.nc -> elec_s{{simpl}}_c{{clusters}}.nc

So on master every per-bus rule is keyed on {M.USA_LADDER[0].buses:,} buses.

Bus counts (source: develop-side busmap CSVs, which both branches' shared
build_base_network feeds):
  nodal {M.USA_LADDER[0].buses:,} = rows in bus2sub.csv
  substations {M.USA_LADDER[1].buses:,} = unique sub_id in busmap_b.csv
  {{simpl}} {M.USA_LADDER[2].buses} = unique cluster_bus in busmap_s300.csv
  final {M.USA_LADDER[3].buses} = unique target in busmap_s300_134.csv

Note for the audience: the brief's working estimate of "about 20k substations"
was wrong; the measured figure is {M.USA_LADDER[1].buses:,}.
""",
    )

    # ---- 4. develop DAG -------------------------------------------------
    s = _blank(prs)
    _title(s, "develop: aggregate and cluster first, then do the heavy work")
    _subtitle(
        s,
        "The same rules, reordered. Two reduction steps now sit ahead of the heavy rules, "
        "which see 300 buses instead of 82,549.",
    )
    _picture(s, figs["fig_dag_develop"])
    _footer(s, "Rule graph read from workflow/rules/build_electricity.smk on feat/equivalence-usa-benchmark; see also the repo's CLAUDE.md")
    _notes(
        s,
        """
Read off develop's build_electricity.smk (file order is not DAG order - this is
the input/output graph):
  aggregate_to_substations  elec_base_network.nc + bus2sub + sub -> elec_b.nc, busmap_b.csv
  cluster_resources         elec_b.nc + regions -> elec_s{simpl}.nc, regions_*_s{simpl}.geojson, busmap_s{simpl}.csv
  build_renewable_profiles  regions = regions_onshore_s{simpl}.geojson -> profile_{tech}_s{simpl}.nc
  add_demand                elec_s{simpl}.nc -> elec_s{simpl}_dem.nc
  add_electricity           elec_s{simpl}_dem.nc -> elec_s{simpl}_l_pp.pkl
  cluster_network           elec_s{simpl}_l_pp.pkl -> elec_s{simpl}_c{clusters}.nc

One honest nuance: build_renewable_profiles on develop still takes the NODAL
regions as a second input (regions_nodal). Land-availability cells are computed
against the nodal regions and then carried to the cluster through
busmap_s{simpl}. That is why this one rule's wall time does not fall even though
its output shrinks ~60x - see the runtime slide.

EGS gets a parallel aggregate_egs rule that remaps NREL substation-keyed supply
curves through busmap_s{simpl} before add_electricity sees them. HAC clustering
was removed; kmeans and modularity remain.
""",
    )

    # ---- 5. resolution ladder -------------------------------------------
    s = _blank(prs)
    _title(s, "Where the network shrinks, and where each branch does its work")
    _picture(s, figs["fig_resolution_ladder"], top=1.35, max_h=5.55)
    _footer(s, "Source: unique-value counts in resources/equivalence/busmaps/usa/*.csv, read 2026-09-17")
    _notes(
        s,
        f"""
Measured bus counts, USA case, all from busmap CSVs (no model run needed):
  nodal          {M.USA_LADDER[0].buses:,}   {M.USA_LADDER[0].source}
  substations    {M.USA_LADDER[1].buses:,}   {M.USA_LADDER[1].source}
  {{simpl}}=300     {M.USA_LADDER[2].buses}       {M.USA_LADDER[2].source}
  clusters=134   {M.USA_LADDER[3].buses}       {M.USA_LADDER[3].source}

The two reduction steps develop inserts are {M.USA_LADDER[0].buses:,} -> {M.USA_LADDER[1].buses:,}
(a 2.0x topology aggregation) and {M.USA_LADDER[1].buses:,} -> {M.USA_LADDER[2].buses}
(a {M.USA_LADDER[1].buses / M.USA_LADDER[2].buses:,.0f}x kmeans reduction).

Net: master's heavy rules see {M.HEAVY_RULE_RESOLUTION['master']:,} buses,
develop's see {M.HEAVY_RULE_RESOLUTION['develop']}. That is the
{M.REDUCTION_FACTOR:,.0f}x factor every later slide traces back to.

The final cluster count (134) is identical on both branches - the refactor does
not change the modelled resolution, only where in the DAG the reduction happens.
""",
    )

    # ---- 6. runtime -----------------------------------------------------
    s = _blank(prs)
    _title(s, "Wall time per rule")
    _subtitle(s, "Log scale on top; develop minus master on its own panel below, with a zero line.")
    _picture(s, figs["fig_runtime_usa"], top=1.55, max_h=5.4)
    _footer(s, SRC_FOOT)
    _notes(
        s,
        f"""
All values are the `s` column of the snakemake benchmark TSV for that rule.

  build_fuel_prices                master {FUEL.master_s:>9,.1f} s   develop {FUEL.develop_s:>9,.1f} s
  build_renewable_profiles onwind  master {ONWIND.master_s:>9,.1f} s   develop {ONWIND.develop_s:>9,.1f} s
  build_renewable_profiles solar   master {SOLAR.master_s:>9,.1f} s   develop {SOLAR.develop_s:>9,.1f} s
  build_electrical_demand          master {DEMAND.master_s:>9,.1f} s   develop {DEMAND.develop_s:>9,.1f} s
  add_demand                       master {ADD_DEM.master_s:>9,.1f} s   develop {ADD_DEM.develop_s:>9,.1f} s
  add_electricity                  master {ADD_ELEC.master_s:>9,.1f} s   develop {ADD_ELEC.develop_s:>9,.1f} s
  cluster_network                  master {CLUSTER_NET.master_s:>9,.1f} s   develop {CLUSTER_NET.develop_s:>9,.1f} s
  solve_network                    master {SOLVE.master_s:>9,.1f} s   develop {SOLVE.develop_s:>9,.1f} s

Headlines: add_demand {ADD_DEM.master_s / ADD_DEM.develop_s:,.0f}x faster,
add_electricity {ADD_ELEC.master_s / ADD_ELEC.develop_s:,.0f}x faster,
build_electrical_demand {DEMAND.master_s / DEMAND.develop_s:,.0f}x faster.

Three rules go the other way and we should say so:
 - build_renewable_profiles (onwind) is SLOWER on develop (107.7 s vs 69.1 s).
   The rule still evaluates land availability against the nodal regions.
 - cluster_network is slower on develop (231.3 s vs 118.9 s) because it now
   consumes the pickled add_electricity output, i.e. it inherited work that
   master had already done before simplify_network.
 - solve_network is close (3157 s develop vs 3437 s master) but the two sides do
   not yet produce the same solution, so do not read it as a speedup.

Caveat to state if asked: these TSVs were not all written by one Slurm job - the
harness reuses finished rules across jobs. Each row's mtime and job are in
figures/fig4_runtime_usa.csv and in measurements.py. They are per-rule cold
measurements of the same config on the same partition.
""",
    )

    # ---- 7. memory ------------------------------------------------------
    s = _blank(prs)
    _title(s, "Peak memory per rule")
    _subtitle(s, "max_rss from the same benchmark TSVs. Linear scale; delta on its own panel.")
    _picture(s, figs["fig_memory_usa"], top=1.55, max_h=5.4)
    _footer(s, SRC_FOOT)
    _notes(
        s,
        f"""
max_rss column, MiB.

  add_demand        master {ADD_DEM.master_rss_mib:>10,.0f}  develop {ADD_DEM.develop_rss_mib:>10,.0f}   ({ADD_DEM.master_rss_mib / ADD_DEM.develop_rss_mib:,.0f}x less)
  add_electricity   master {ADD_ELEC.master_rss_mib:>10,.0f}  develop {ADD_ELEC.develop_rss_mib:>10,.0f}   ({ADD_ELEC.master_rss_mib / ADD_ELEC.develop_rss_mib:,.1f}x less)
  profiles onwind   master {ONWIND.master_rss_mib:>10,.0f}  develop {ONWIND.develop_rss_mib:>10,.0f}
  profiles solar    master {SOLAR.master_rss_mib:>10,.0f}  develop {SOLAR.develop_rss_mib:>10,.0f}
  build_electrical_demand  master {DEMAND.master_rss_mib:>10,.0f}  develop {DEMAND.develop_rss_mib:>10,.0f}  (essentially flat)
  solve_network     master {SOLVE.master_rss_mib:>10,.0f}  develop {SOLVE.develop_rss_mib:>10,.0f}  (develop HIGHER)

The clean win is add_demand: {ADD_DEM.master_rss_mib:,.0f} MiB -> {ADD_DEM.develop_rss_mib:,.0f} MiB.
It no longer materialises a demand array keyed on 82,549 buses.

Two rules do not improve and must not be glossed:
 - build_electrical_demand peak RSS is flat (10,604 -> 11,077 MiB). It still
   reads the same zonal source data; only its output shrinks.
 - solve_network uses MORE memory on develop (17,536 -> 25,973 MiB). That is
   downstream of this refactor and is tangled with the still-open equivalence
   differences, so it is reported, not claimed as a result of the reordering.

Whole-job peak RSS from sacct, for scale: job 43640844 31.6 GiB, job 43671766
33.4 GiB, western smoke 43671762 1.74 GiB.
""",
    )

    # ---- 8. artifact sizes ----------------------------------------------
    s = _blank(prs)
    _title(s, "Why it saves work: the arrays get smaller by the reduction factor")
    _subtitle(s, "Same rule, same config, different keying resolution. Sizes are bytes on disk.")
    _picture(s, figs["fig_artifact_sizes"], top=1.55, max_h=5.4)
    _footer(s, f"Source: {M.ARTIFACT_SOURCE}")
    _notes(
        s,
        f"""
The mechanism is not subtle: a per-bus rule writes an array whose leading
dimension is the bus count. Divide the bus count by {M.REDUCTION_FACTOR:,.0f} and
the array divides with it.

  onwind profile        {M.USA_ARTIFACTS[0].master_bytes:>14,} B -> {M.USA_ARTIFACTS[0].develop_bytes:>12,} B   {M.USA_ARTIFACTS[0].master_bytes / M.USA_ARTIFACTS[0].develop_bytes:,.1f}x
  solar profile         {M.USA_ARTIFACTS[1].master_bytes:>14,} B -> {M.USA_ARTIFACTS[1].develop_bytes:>12,} B   {M.USA_ARTIFACTS[1].master_bytes / M.USA_ARTIFACTS[1].develop_bytes:,.1f}x
  electrical demand CSV {M.USA_ARTIFACTS[2].master_bytes:>14,} B -> {M.USA_ARTIFACTS[2].develop_bytes:>12,} B   {M.USA_ARTIFACTS[2].master_bytes / M.USA_ARTIFACTS[2].develop_bytes:,.1f}x
  network after add_demand {M.USA_ARTIFACTS[3].master_bytes:>11,} B -> {M.USA_ARTIFACTS[3].develop_bytes:>12,} B   {M.USA_ARTIFACTS[3].master_bytes / M.USA_ARTIFACTS[3].develop_bytes:,.1f}x
  network after add_electricity {M.USA_ARTIFACTS[4].master_bytes:>6,} B -> {M.USA_ARTIFACTS[4].develop_bytes:>12,} B   {M.USA_ARTIFACTS[4].master_bytes / M.USA_ARTIFACTS[4].develop_bytes:,.1f}x

The add_electricity pickle is the smallest ratio ({M.USA_ARTIFACTS[4].master_bytes / M.USA_ARTIFACTS[4].develop_bytes:,.1f}x, not
{M.REDUCTION_FACTOR:,.0f}x) and that is the honest limit of the argument: it carries one
entry per attached generator, and the generator count does not fall as steeply
as the per-bus time series do. master's is 6.5 GB; develop's is 1.1 GB.

Caveat on the demand CSV bar: develop also writes
power_zonal_components_s300.parquet (186,302,484 B) which master has no
counterpart for. It is excluded from the bar and noted in the CSV twin.

A nice illustration for the room: elec_s300.nc is 137,502,390 B on master but
597,423 B on develop. Same bus count, different pipeline stage - on master it
already carries everything the heavy rules attached; on develop it is topology
only, because the heavy rules have not run yet.
""",
    )

    # ---- 9. segment totals + unmeasured ---------------------------------
    s = _blank(prs)
    _title(s, "The segment total, and the one number we do not have")
    _picture(s, figs["fig_segment_totals"], top=1.35, max_h=5.5)
    _footer(s, "master's simplify_network carries no benchmark: directive on master-benchmark, so its cost was never recorded")
    _notes(
        s,
        f"""
Summing the five per-bus heavy rules measured on BOTH branches:
  master  {T['master_s']:,.0f} s
  develop {T['develop_s']:,.0f} s      -> {HEAVY_SPEEDUP:,.1f}x faster

develop pays for that with two new rules master does not have:
  aggregate_to_substations {M.USA_DEVELOP_ONLY[0].develop_s:,.0f} s, peak {M.USA_DEVELOP_ONLY[0].develop_rss_mib:,.0f} MiB
  cluster_resources        {M.USA_DEVELOP_ONLY[1].develop_s:,.0f} s, peak {M.USA_DEVELOP_ONLY[1].develop_rss_mib:,.0f} MiB
  total new work           {NEW_S:,.0f} s

Segment as measured: master {T['master_s']:,.0f} s vs develop {T['develop_s'] + NEW_S:,.0f} s
= {SEGMENT_SPEEDUP:,.1f}x.

THE GAP, say it out loud: master's simplify_network does the same two jobs
develop's new rules do, but it has no `benchmark:` directive on
master-benchmark, so no wall time and no peak RSS exist for it. master's bar is
therefore a LOWER BOUND; the true factor is larger than {SEGMENT_SPEEDUP:,.1f}x. The
hatched block on master's bar is drawn at an arbitrary height purely to mark the
gap - it is not an estimate and is labelled NOT MEASURED.

Second honest point, the right-hand panel: peak memory in this segment did not
fall, it MOVED. master's worst rule is add_demand at {T['master_peak_rss']:,.0f} MiB;
develop's worst is its new aggregate_to_substations at {M.USA_DEVELOP_ONLY[0].develop_rss_mib:,.0f} MiB,
which is higher. Whether that is better or worse than master's unmeasured
simplify_network peak is exactly the thing we cannot say. Adding a benchmark:
directive to master's simplify_network is the cheap fix and is worth doing.
""",
    )

    # ---- 10. scale dependence -------------------------------------------
    s = _blank(prs)
    _title(s, "The win scales with the problem")
    _subtitle(s, "The same five heavy rules on the California smoke case and on the USA case.")
    _picture(s, figs["fig_scale_dependence"], top=1.55, max_h=5.35)
    _footer(s, "Source: benchmarks/equivalence/{western,usa}/ on both branches")
    _notes(
        s,
        f"""
western (California only, {M.WESTERN_LADDER[0].buses:,} nodal -> {M.WESTERN_LADDER[2].buses} clusters):
  master {WEST['master_s']:,.0f} s, develop {WEST['develop_s']:,.0f} s -> {WEST_SPEEDUP:,.1f}x

usa ({M.USA_LADDER[0].buses:,} nodal -> {M.USA_LADDER[2].buses} clusters):
  master {T['master_s']:,.0f} s, develop {T['develop_s']:,.0f} s -> {HEAVY_SPEEDUP:,.1f}x

This is the expected shape: the saving is a function of how much the reduction
step removes, so the small diagnostic leg shows a modest win and the acceptance
case shows a large one. On western, add_electricity is actually slower on
develop (10.7 s -> 32.7 s) - at 20 clusters there is nothing to save and the
extra indirection costs a little.

Practical read: do not benchmark this refactor on a toy case and expect the USA
numbers. Equally, do not promise USA-scale savings to someone running a
single-state model.

Note the western leg's name is the interconnect wildcard; the modelled footprint
is California only (model_topology.include.reeds_state: [CA], ReEDS zones
p8-p11).
""",
    )

    # ---- 11. what stays equivalent --------------------------------------
    s = _blank(prs)
    _title(s, "What stays equivalent, and how that is checked")
    _bullets(
        s,
        [
            (
                "A reordering is only worth anything if the model it produces is the same model. "
                "That is enforced by the Tier C equivalence harness, not by inspection.",
                0,
                True,
                INK,
            ),
            (f"Lives in {M.HARNESS['location']}, documented in {M.HARNESS['docs']}.", 0, False, INK2),
            (
                "The baseline is a branch, not a patched checkout: master-benchmark is a sha you "
                "can check out, carrying only the commits master needs to run the same config.",
                0,
                False,
                INK2,
            ),
            ("Both branches build the same config; the harness then compares:", 0, True, INK),
            ("capacity factor profiles (p_max_pu)", 1, False, INK2),
            ("resource availability (p_nom_max) by zone and carrier", 1, False, INK2),
            ("existing capacity (p_nom) by zone and carrier", 1, False, INK2),
            ("demand by zone", 1, False, INK2),
            ("solved objective, capacity and dispatch", 1, False, INK2),
            (
                "Every compared row gets one of three verdicts: equivalent, explained (it maps to a "
                "numbered hot-fix in the ledger), or UNEXPLAINED - and UNEXPLAINED is a test failure.",
                0,
                True,
                INK,
            ),
            (
                f"{M.HARNESS['test_files']} test modules, plus hotfixes.yaml and waivers.yaml as "
                "machine-readable ledgers, so 'we know why that differs' is version-controlled rather than remembered.",
                0,
                False,
                INK2,
            ),
        ],
        top=1.6,
        size=14,
        gap=9,
    )
    _footer(s, "The current verdict table for the USA case is a run-card matter, not a slide in this deck")
    _notes(
        s,
        """
Keep this slide short and do not be drawn into the specific open differences -
they are a separate conversation with their own run cards and GitHub issues.

The point to land: the refactor is not asserted to be equivalent, it is tested
for equivalence, on the same config, against a baseline that is a checkout-able
commit rather than a patched working tree. Differences are not tolerated
silently; each one is either equivalent, mapped to a numbered hot-fix with a
written reason, or a failure.

If someone asks "is it currently passing?": on the USA case, no - there are open
UNEXPLAINED rows under active diagnosis, tracked in experiments/runs/. That is
the harness doing its job. The western smoke leg is much closer. Point them at
the run cards rather than debugging live.

Harness modules: build.py provisions and builds both sides; metrics.py defines
the compared quantities; compare.py/tables.py produce the verdict table;
plots.py emits the PNG+CSV figure set.
""",
    )

    # ---- 12. what we measured / did not ---------------------------------
    s = _blank(prs)
    _title(s, "What is measured here, and what is not")
    _bullets(
        s,
        [
            ("Measured, from artifacts:", 0, True, INK),
            ("per-rule wall time and peak RSS - snakemake benchmark TSVs, both branches, USA and western", 1, False, INK2),
            ("bus counts at every stage - unique-value counts in the busmap CSVs", 1, False, INK2),
            ("artifact sizes - bytes on disk for the outputs of the same rule on each branch", 1, False, INK2),
            ("whole-job elapsed and MaxRSS - sacct for jobs 43640844, 43671766, 43671762", 1, False, INK2),
            ("NOT measured - stated as such on the slides:", 0, True, RGBColor(0xE3, 0x49, 0x48)),
            (
                "master's simplify_network wall time and peak RSS. No benchmark: directive on "
                "master-benchmark, so master's reduction cost is absent and its segment bar is a lower bound.",
                1,
                False,
                INK2,
            ),
            (
                "end-to-end cold pipeline wall time, master vs develop. No single job built both "
                "sides from a cold cache: in job 43671766 develop built in 3,910 s while master "
                "reused finished rules in 117 s; in job 43640844 master built in 8,297 s while "
                "develop was cached at 4 s. Those pairs are not comparable and are not charted.",
                1,
                False,
                INK2,
            ),
            (
                "master-side build_base_network, build_bus_regions, build_shapes, add_extra_components, "
                "prepare_network, add_sectors - instrumented on develop only, so not charted.",
                1,
                False,
                INK2,
            ),
            (
                "Every figure ships a CSV twin with the exact plotted numbers, the source file and the mtime.",
                0,
                True,
                INK,
            ),
        ],
        top=1.55,
        size=13.5,
        gap=8,
    )
    _footer(s, "figures/*.csv carry the provenance of every plotted value; measurements.py is the single source")
    _notes(
        s,
        f"""
This slide exists so nobody has to guess which numbers are real.

{M.TOTAL_PIPELINE_CAVEAT}

The cheap fixes, if we want the missing numbers:
 1. add a `benchmark:` directive to simplify_network on master-benchmark, then
    rebuild the master side once. That closes the single biggest gap.
 2. run one job with both sides cold (delete both branches' resources for the
    case) to get a true end-to-end total. Expensive - master's cold build alone
    was 8,297 s - but it is the number people will ask for.

Also worth stating: the eq-usa jobs show State=FAILED in sacct. That is the
equivalence comparison verdict, not a build crash; the networks were built on
both sides. See experiments/runs/eq-usa-80abb3bc125f/CARD.md.
""",
    )

    # ---- 13. summary ----------------------------------------------------
    s = _blank(prs)
    _title(s, "Summary")
    _bullets(
        s,
        [
            (
                f"The refactor moves the reduction from after the heavy rules to before them, so the "
                f"per-bus rules run at {M.HEAVY_RULE_RESOLUTION['develop']} buses instead of "
                f"{M.HEAVY_RULE_RESOLUTION['master']:,} - a {M.REDUCTION_FACTOR:,.0f}x cut in the dimension that matters.",
                0,
                True,
                INK,
            ),
            (
                f"The five heavy rules go from {T['master_s']:,.0f} s to {T['develop_s']:,.0f} s on the USA case "
                f"({HEAVY_SPEEDUP:,.1f}x); including develop's two new reduction rules the segment is still "
                f"{SEGMENT_SPEEDUP:,.1f}x faster, and that is a lower bound.",
                0,
                False,
                INK2,
            ),
            (
                f"The arrays shrink with the bus count: profiles {M.USA_ARTIFACTS[0].master_bytes / M.USA_ARTIFACTS[0].develop_bytes:,.0f}-"
                f"{M.USA_ARTIFACTS[1].master_bytes / M.USA_ARTIFACTS[1].develop_bytes:,.0f}x, demand {M.USA_ARTIFACTS[2].master_bytes / M.USA_ARTIFACTS[2].develop_bytes:,.0f}x, "
                f"post-add_demand network {M.USA_ARTIFACTS[3].master_bytes / M.USA_ARTIFACTS[3].develop_bytes:,.0f}x.",
                0,
                False,
                INK2,
            ),
            (
                f"Memory did not simply fall - it moved. add_demand drops {ADD_DEM.master_rss_mib / ADD_DEM.develop_rss_mib:,.0f}x, "
                f"but develop's new aggregate_to_substations peaks at {M.USA_DEVELOP_ONLY[0].develop_rss_mib:,.0f} MiB, "
                "the highest rule in the segment.",
                0,
                False,
                INK2,
            ),
            (
                f"The win scales with the problem: {WEST_SPEEDUP:,.1f}x on the California smoke case, "
                f"{HEAVY_SPEEDUP:,.1f}x on the USA case.",
                0,
                False,
                INK2,
            ),
            (
                "Equivalence is tested, not asserted: the Tier C harness compares both branches on the "
                "same config and treats an unexplained difference as a failure.",
                0,
                False,
                INK2,
            ),
            (
                "Next measurement to take: benchmark master's simplify_network, and run one job with "
                "both sides cold for a true end-to-end total.",
                0,
                True,
                INK,
            ),
        ],
        top=1.55,
        size=14,
        gap=11,
    )
    _footer(s, SRC_FOOT)
    _notes(
        s,
        f"""
One-sentence version: the network now shrinks before the expensive rules run,
which is worth about {HEAVY_SPEEDUP:,.0f}x on the heavy segment of the USA build and roughly
two orders of magnitude on the intermediate files, at the cost of two new rules
and a new memory peak we have not yet compared against master's equivalent.

Open asks for the room:
 - is {M.REDUCTION_FACTOR:,.0f}x the right {{simpl}} for USA, or should s300 be revisited now that
   the heavy rules are cheap?
 - is aggregate_to_substations' {M.USA_DEVELOP_ONLY[0].develop_rss_mib:,.0f} MiB peak worth attacking, given it
   now sets the memory requirement for the whole build segment?
""",
    )

    return prs


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

EXPECTED = [
    ("Title", 0),
    ("The change in one line", 0),
    ("master DAG", 1),
    ("develop DAG", 1),
    ("Resolution ladder", 1),
    ("Wall time per rule", 1),
    ("Peak memory per rule", 1),
    ("Artifact sizes", 1),
    ("Segment totals", 1),
    ("Scale dependence", 1),
    ("What stays equivalent", 0),
    ("Measured / not measured", 0),
    ("Summary", 0),
]


def verify(path: Path) -> None:
    """Re-open the built file and check every slide is intact."""
    from pptx import Presentation as P

    prs = P(str(path))
    n = len(prs.slides)
    print(f"\nVerification: re-opened {path.name}")
    print(f"  slide size: {prs.slide_width} x {prs.slide_height} EMU "
          f"({prs.slide_width / 914400:.3f} x {prs.slide_height / 914400:.3f} in)")
    assert prs.slide_width == Emu(int(SLIDE_W)), "slide width is not 13.333in"
    assert prs.slide_height == Emu(int(SLIDE_H)), "slide height is not 7.5in"
    assert n == len(EXPECTED), f"expected {len(EXPECTED)} slides, found {n}"

    problems = []
    for i, (slide, (label, want_pics)) in enumerate(zip(prs.slides, EXPECTED), start=1):
        pics = [sh for sh in slide.shapes if sh.shape_type == 13]
        texts = [sh for sh in slide.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
        notes = ""
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
        ok_pic = len(pics) == want_pics
        ok_txt = len(texts) >= 1
        ok_notes = len(notes) > 40
        flag = "ok " if (ok_pic and ok_txt and ok_notes) else "FAIL"
        if flag == "FAIL":
            problems.append((i, label, len(pics), want_pics, len(texts), len(notes)))
        print(
            f"  [{flag}] slide {i:>2}  {label:<26} pictures={len(pics)} (want {want_pics})  "
            f"textboxes={len(texts)}  notes={len(notes)} chars"
        )
    if problems:
        raise AssertionError(f"slides failed verification: {problems}")
    print(f"\n  ALL {n} SLIDES OK")


def main() -> int:
    print("Building figures...")
    figs = figures.build_all()

    missing = [k for k, v in figs.items() if not Path(v).exists()]
    if missing:
        raise SystemExit(f"figures missing: {missing}")

    print("\nBuilding deck...")
    prs = build(figs)
    prs.save(str(OUT))
    print(f"  wrote {OUT}  ({OUT.stat().st_size:,} bytes)")

    verify(OUT)

    pngs = sorted(FIGDIR.glob("*.png"))
    csvs = sorted(FIGDIR.glob("*.csv"))
    print(f"\n  {len(pngs)} PNGs, {len(csvs)} CSV twins")
    for p in pngs:
        twin = p.with_suffix(".csv")
        assert twin.exists(), f"PNG without a CSV twin: {p.name}"
    print("  every PNG has a CSV twin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
