"""
Common fixtures for PyPSA-USA API tests.

This module contains shared fixtures used across API test files.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_workflow_path(temp_workspace):
    """Mock the workflow path for testing."""
    mock_path = temp_workspace / "workflow"
    mock_path.mkdir(parents=True)
    (mock_path / "Snakefile").touch()
    
    with patch('pypsa_usa.api.workflow.get_workflow_path') as mock_get_path:
        mock_get_path.return_value = mock_path
        yield mock_path


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary for testing."""
    return {
        "scenario": {
            "interconnect": "western",
            "clusters": 30,
            "simpl": 75,
        },
        "run": {
            "name": "test_run",
        },
        "costs": {
            "ng_fuel_year": 2019,
        },
        "solving": {
            "solver": "gurobi",
        },
    }


@pytest.fixture
def sample_config_file(temp_workspace, sample_config_dict):
    """Create a sample config file for testing."""
    config_path = temp_workspace / "test_config.yaml"
    
    # Create a simple YAML-like config file
    with open(config_path, 'w') as f:
        f.write("scenario:\n")
        f.write("  interconnect: western\n")
        f.write("  clusters: 30\n")
        f.write("  simpl: 75\n")
        f.write("run:\n")
        f.write("  name: test_run\n")
        f.write("costs:\n")
        f.write("  ng_fuel_year: 2019\n")
        f.write("solving:\n")
        f.write("  solver: gurobi\n")
    
    return config_path


@pytest.fixture
def mock_snakemake_success():
    """Mock snakemake.snakemake to return success."""
    with patch('pypsa_usa.api.workflow.snakemake.snakemake') as mock_snakemake:
        mock_snakemake.return_value = True
        yield mock_snakemake


@pytest.fixture
def mock_snakemake_failure():
    """Mock snakemake.snakemake to return failure."""
    with patch('pypsa_usa.api.workflow.snakemake.snakemake') as mock_snakemake:
        mock_snakemake.return_value = False
        yield mock_snakemake


@pytest.fixture
def mock_snakemake_exception():
    """Mock snakemake.snakemake to raise an exception."""
    with patch('pypsa_usa.api.workflow.snakemake.snakemake') as mock_snakemake:
        mock_snakemake.side_effect = Exception("Test exception")
        yield mock_snakemake


@pytest.fixture
def clean_environment():
    """Clean environment variables before and after tests."""
    import os
    
    # Store original environment variables
    original_python = os.environ.get('PYTHON')
    original_pythonpath = os.environ.get('PYTHONPATH')
    
    # Clean environment
    if 'PYTHON' in os.environ:
        del os.environ['PYTHON']
    if 'PYTHONPATH' in os.environ:
        del os.environ['PYTHONPATH']
    
    yield
    
    # Restore original environment variables
    if original_python is not None:
        os.environ['PYTHON'] = original_python
    if original_pythonpath is not None:
        os.environ['PYTHONPATH'] = original_pythonpath


@pytest.fixture
def mock_user_config_path(temp_workspace):
    """Mock the user config path for testing."""
    config_path = temp_workspace / "user_config.json"
    
    with patch('pypsa_usa.api.workflow.get_user_config_path') as mock_get_path:
        mock_get_path.return_value = config_path
        yield config_path
