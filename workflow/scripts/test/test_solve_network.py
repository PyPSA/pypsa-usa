"""
End-to-end tests for solve_network().

These tests call solve_network() directly, exercising the full extra_functionality
dispatch path that runs on Oak. They depend on the 8-bus, 3-period fixture network.

test_network.nc is gitignored — build it with:
    uv run workflow/scripts/test/fixtures/build_test_network.py
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import solve_network as sn_module

FIXTURE_NETWORK = os.path.join(os.path.dirname(__file__), "fixtures", "test_network.nc")


@pytest.fixture
def e2e_network():
    """Load the realistic 8-bus, 3-period fixture network."""
    if not os.path.exists(FIXTURE_NETWORK):
        pytest.skip(
            f"Fixture network not found: {FIXTURE_NETWORK}. "
            "Run workflow/scripts/test/fixtures/build_test_network.py to generate it."
        )
    return pypsa.Network(FIXTURE_NETWORK)


def _solving():
    """Minimal solving config — skip iterative line expansion for speed."""
    return {
        "solver": {"name": "glpk", "options": None},
        "solver_options": {},
        "options": {"skip_iterations": True},
    }


def _config(foresight="perfect", erm=None):
    cfg = {"foresight": foresight, "electricity": {}}
    if erm:
        cfg["electricity"]["erm"] = erm
    return cfg


def _mock_snakemake(foresight, output=None):
    mock = MagicMock()
    mock.params.foresight = foresight
    if output is not None:
        mock.output = [str(output)]
    return mock


def test_e2e_solve_network_myopic(e2e_network, tmp_path):
    """ERM constraint added in every period of the myopic loop via solve_network().

    prepare_brownfield is patched: it modifies generator DataFrames in ways the
    minimal fixture network doesn't support (custom columns like heat_rate),
    and brownfield correctness is not what we're testing here.
    """
    n = e2e_network.copy()
    output_nc = tmp_path / "result.nc"

    with patch.object(sn_module, "snakemake", _mock_snakemake("myopic", output=output_nc), create=True), \
         patch.object(sn_module, "prepare_brownfield"):
        result = sn_module.solve_network(
            n,
            _config("myopic", erm={"all": 0.15}),
            _solving(),
            opts=["ERM"],
        )

    # solve_network saves a per-period .nc for each investment period
    for period in e2e_network.investment_periods:
        assert (tmp_path / f"result_period_{period}.nc").exists(), \
            f"Missing period file for {period}"

    assert "GlobalConstraint-all_ERM" in result.model.constraints
