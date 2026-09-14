"""The config-equivalence gate, on hand-written dicts.

Tier A (``fast``): ``merged_config`` is monkeypatched, so nothing runs
snakemake, nothing touches ``data/`` and nothing needs a worktree. What is under
test is the *decision*: which differences are tolerated, which abort the run,
and whether the abort message tells you enough to act.

The gate exists because the two branches do not load the same set of config
files (hot-fix HF-20): master's Snakefile has ``config.default.yaml`` commented
out, develop layers it in. So an omitted key resolves to the shipped default on
one side and to a script's inline fallback on the other, and "on the same
config" becomes an assumption instead of a fact. Comparing objective, capacity
and dispatch across two different configurations measures the configs.
"""

from __future__ import annotations

import pytest

from tests.equivalence import context

pytestmark = pytest.mark.fast


BASE = {
    "run": {"name": "equivalence"},
    "pudl_path": "s3://pudl.catalyst.coop/v2026.8.0",
    "electricity": {
        "honor_planned_retirements": True,
        "demand": {"bus_allocation": "breakthrough"},
    },
    "renewable_land_access": "reference",
    "scenario": {"interconnect": ["usa"], "clusters": ["134"]},
}


def _patch(monkeypatch, master: dict, develop: dict) -> None:
    monkeypatch.setattr(
        context,
        "merged_config",
        lambda side: master if side == "master" else develop,
    )


def test_identical_configs_pass(monkeypatch):
    _patch(monkeypatch, BASE, BASE)
    assert context.assert_config_equivalent() == []


def test_allowlisted_difference_is_returned_with_its_reason(monkeypatch):
    develop = {**BASE, "scenario": {"interconnect": ["usa"], "clusters": ["134s"]}}
    _patch(monkeypatch, BASE, develop)

    allowed = context.assert_config_equivalent()
    assert len(allowed) == 1
    (diff,) = allowed
    assert diff["key"] == "scenario.clusters"
    assert diff["kind"] == "value"
    assert diff["master"] == ["134"]
    assert diff["develop"] == ["134s"]
    assert diff["reason"], "an allowed difference must carry the reason it is allowed"


def test_pudl_path_difference_raises_with_both_values(monkeypatch):
    """The exact HF-1 failure mode: two PUDL releases, one comparison."""
    develop = {**BASE, "pudl_path": "s3://pudl.catalyst.coop/v2025.2.0"}
    _patch(monkeypatch, BASE, develop)

    with pytest.raises(RuntimeError) as exc:
        context.assert_config_equivalent()
    msg = str(exc.value)
    assert "pudl_path" in msg
    assert "v2026.8.0" in msg and "v2025.2.0" in msg
    assert "CONFIG_DIFF_ALLOWLIST" in msg


def test_falsy_key_present_on_one_side_only_is_recorded_not_fatal(monkeypatch):
    """HF-20's diffuse class: recorded with its reason, not an abort.

    Only when the value is FALSY. ``config.get(key, {})`` followed by
    ``if cfg:`` takes the same branch whether the key is absent or present-and-
    falsy, so the side without it behaves as the side with it does.
    """
    develop = {**BASE, "some_disabled_flag": False}
    _patch(monkeypatch, BASE, develop)

    allowed = context.assert_config_equivalent()
    assert len(allowed) == 1
    assert allowed[0]["key"] == "some_disabled_flag"
    assert allowed[0]["kind"] == "only_develop"
    assert "HF-20" in allowed[0]["reason"]


@pytest.mark.parametrize("empty", [None, {}, [], "", 0, False])
def test_every_falsy_literal_counts_as_absent(monkeypatch, empty):
    develop = {**BASE, "some_unread_key": empty}
    _patch(monkeypatch, BASE, develop)
    allowed = context.assert_config_equivalent()
    assert [d["kind"] for d in allowed] == ["only_develop"]


def test_falsy_presence_only_aborts_under_strict(monkeypatch):
    develop = {**BASE, "some_disabled_flag": False}
    _patch(monkeypatch, BASE, develop)
    monkeypatch.setenv("EQ_STRICT_CONFIG_GATE", "1")

    with pytest.raises(RuntimeError) as exc:
        context.assert_config_equivalent()
    assert "some_disabled_flag" in str(exc.value)
    assert "only_develop" in str(exc.value)


def test_truthy_key_present_on_one_side_only_is_fatal(monkeypatch):
    """The demand_response failure mode, in one test.

    ``electricity.demand_response: {marginal_cost: 999999, shift: 0}`` lives in
    develop's default layer only. It is TRUTHY, so develop calls
    ``add_demand_response``, which adds a ``demand_response`` Carrier before its
    own ``shift == 0`` early return, while the baseline's
    ``.get("demand_response", {})`` is falsy and adds nothing — an unwaived
    Carrier row_set finding at three stages. A gate that waves through every
    present-on-one-side key would have shipped that.
    """
    master = {
        **BASE,
        "electricity": {k: v for k, v in BASE["electricity"].items()},
    }
    develop = {
        **BASE,
        "electricity": {
            **BASE["electricity"],
            "demand_response": {"marginal_cost": 999999, "shift": 0},
        },
    }
    _patch(monkeypatch, master, develop)

    with pytest.raises(RuntimeError) as exc:
        context.assert_config_equivalent()
    msg = str(exc.value)
    assert "electricity.demand_response" in msg
    assert "default_only" in msg


def test_populated_subtree_against_an_empty_one_is_a_value_difference(monkeypatch):
    """Not a scatter of present-on-one-side leaves: one differing mapping."""
    master = {**BASE, "block": {}}
    develop = {**BASE, "block": {"enable": True, "size": 3}}
    _patch(monkeypatch, master, develop)

    with pytest.raises(RuntimeError) as exc:
        context.assert_config_equivalent()
    msg = str(exc.value)
    assert "block [value]" in msg
    assert "block.enable" not in msg


def test_value_difference_is_always_fatal(monkeypatch):
    """A key BOTH sides carry, with different values, is a real disagreement."""
    develop = {**BASE, "renewable_land_access": "unscreened"}
    _patch(monkeypatch, BASE, develop)
    with pytest.raises(RuntimeError) as exc:
        context.assert_config_equivalent()
    assert "renewable_land_access" in str(exc.value)


def test_empty_and_absent_mappings_compare_equal(monkeypatch):
    """``model_topology.include: {}`` vs ``null`` is not a difference.

    Snakemake's merge writes nothing for an empty-mapping update, so the shared
    config's ``include: {}`` arrives as ``None`` on develop and ``{}`` on
    master. Both are falsy, so the HF-9 / HF-10 scoping gates evaluate False on
    both sides. A POPULATED include on one side still differs — that is the case
    that matters.
    """
    master = {**BASE, "model_topology": {"include": {}}}
    develop = {**BASE, "model_topology": {"include": None}}
    _patch(monkeypatch, master, develop)
    assert context.assert_config_equivalent() == []

    scoped = {**BASE, "model_topology": {"include": {"reeds_state": ["CA"]}}}
    _patch(monkeypatch, master, scoped)
    with pytest.raises(RuntimeError) as exc:
        context.assert_config_equivalent()
    assert "model_topology.include" in str(exc.value)


def test_every_blocking_difference_is_listed_at_once(monkeypatch):
    """One run of the gate tells you everything to decide, not the first trip."""
    develop = {
        **BASE,
        "pudl_path": "s3://pudl.catalyst.coop/v2025.2.0",
        "renewable_land_access": None,
        "electricity": {
            "honor_planned_retirements": False,
            "demand": {"bus_allocation": "population"},
        },
    }
    _patch(monkeypatch, BASE, develop)

    with pytest.raises(RuntimeError) as exc:
        context.assert_config_equivalent()
    msg = str(exc.value)
    for key in (
        "pudl_path",
        "renewable_land_access",
        "electricity.honor_planned_retirements",
        "electricity.demand.bus_allocation",
    ):
        assert key in msg, f"{key} missing from the gate's message"
    assert "4 differing key(s)" in msg


def test_subtree_present_on_one_side_is_reported_whole(monkeypatch):
    """One entry for the whole block, not one per leaf inside it.

    ``dac:`` exists on develop and nowhere on master. Six separate
    ``dac.capital_cost``-style entries would be six copies of one fact, and six
    allowlist rows to write.
    """
    develop = {**BASE, "godeeep_cf_registry": {"sources": [{"kind": "zenodo"}]}}
    _patch(monkeypatch, BASE, develop)
    allowed = context.assert_config_equivalent()
    assert [d["key"] for d in allowed] == ["godeeep_cf_registry"]
    assert allowed[0]["develop"] == {"sources": [{"kind": "zenodo"}]}


def test_nested_allowlist_prefix_covers_children(monkeypatch):
    """An allowlisted path covers keys underneath it."""
    master = {**BASE, "godeeep_cf_registry": {"sources": [{"kind": "zenodo"}], "copy_local": True}}
    develop = {**BASE, "godeeep_cf_registry": {"sources": [{"kind": "oak"}], "copy_local": True}}
    _patch(monkeypatch, master, develop)
    allowed = context.assert_config_equivalent()
    assert [d["key"] for d in allowed] == ["godeeep_cf_registry.sources"]


def test_allowlist_matching_is_not_a_bare_prefix(monkeypatch):
    """``scenario.clustersomething`` is not covered by ``scenario.clusters``."""
    assert context.allowlist_reason("scenario.clusters") is not None
    assert context.allowlist_reason("scenario.clusters.0") is not None
    assert context.allowlist_reason("scenario.clustersomething") is None
    assert context.allowlist_reason("scenario.interconnect") is None


def test_skip_env_records_that_it_skipped(monkeypatch):
    """The escape hatch is loud: a skipped gate is visible in run_meta.json."""

    def _boom(side):  # pragma: no cover - must never be called
        raise AssertionError("merged_config called despite EQ_SKIP_CONFIG_GATE=1")

    monkeypatch.setattr(context, "merged_config", _boom)
    monkeypatch.setenv("EQ_SKIP_CONFIG_GATE", "1")

    allowed = context.assert_config_equivalent()
    assert len(allowed) == 1
    assert allowed[0]["kind"] == "skipped"
    assert "EQ_SKIP_CONFIG_GATE" in allowed[0]["reason"]


def test_flatten_treats_the_scoping_keys_as_atomic():
    """``model_topology.include`` is compared whole, never leaf by leaf.

    It is the gate expression HF-9 and HF-10 hinge on. Flattened, a scoped
    develop against an unscoped master would show up only as "present on one
    side" and be recorded rather than refused; compared whole it is the value
    difference it actually is. Everything else still flattens.
    """
    flat = context._flatten(
        {
            "model_topology": {"include": {"reeds_state": ["CA"]}, "aggregate": {"a": 1}},
            "clustering": {"cluster_network": {"algorithm": "kmeans"}},
        },
    )
    assert flat == {
        "model_topology.include": {"reeds_state": ["CA"]},
        "model_topology.aggregate": {"a": 1},
        "clustering.cluster_network.algorithm": "kmeans",
    }


# --- allowlist guards --------------------------------------------------------


def test_allowlist_guard_rejects_a_moved_value():
    """An allowlisted key whose value left its reason is not allowlisted.

    ``costs.atb`` is allowed because develop's default layer supplies exactly
    master's inline fallback. Change the scenario and the reason no longer
    holds, so neither does the allowance.
    """
    ok = {
        "key": "costs.atb",
        "kind": "default_only",
        "master": None,
        "develop": {"scenario": "Moderate", "model_case": "Market", "overrides": None},
    }
    moved = {**ok, "develop": {**ok["develop"], "scenario": "Advanced"}}
    assert context.allowlist_reason("costs.atb", ok) is not None
    assert context.allowlist_reason("costs.atb", moved) is None


def test_weighting_strategy_guard_rejects_population():
    """The only branch either branch takes on this key is == 'population'."""
    ok = {"key": "k", "kind": "default_only", "master": None, "develop": "demand-capacity"}
    bad = {**ok, "develop": "population"}
    key = "clustering.cluster_network.weighting_strategy"
    assert context.allowlist_reason(key, ok) is not None
    assert context.allowlist_reason(key, bad) is None


def test_disabled_block_guard_rejects_an_enabled_block():
    key = "electricity.imports"
    off = {"key": key, "kind": "default_only", "master": None, "develop": {"enable": False}}
    on = {**off, "develop": {"enable": True}}
    assert context.allowlist_reason(key, off) is not None
    assert context.allowlist_reason(key, on) is None


def test_walltime_guard_rejects_a_block_with_anything_else_in_it():
    key = "cluster_network"
    ok = {"key": key, "kind": "default_only", "master": {"walltime": "09:00:00"}, "develop": None}
    bad = {**ok, "master": {"walltime": "09:00:00", "algorithm": "kmeans"}}
    assert context.allowlist_reason(key, ok) is not None
    assert context.allowlist_reason(key, bad) is None


def test_allowlist_without_a_diff_still_resolves():
    """Callers that only have the key (docs, tests) get the reason unguarded."""
    assert context.allowlist_reason("costs.atb") is not None
