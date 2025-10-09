"""
Main conftest.py for PyPSA-USA test suite.

This module contains shared fixtures and configuration for all tests.
"""

import os
import sys
from pathlib import Path

import pytest

# Add the src directory to the Python path so we can import pypsa_usa modules
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def project_root_path():
    """Get the project root path."""
    return project_root


@pytest.fixture(scope="session")
def src_path():
    """Get the src directory path."""
    return src_path


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment for all tests."""
    # Store original working directory
    original_cwd = os.getcwd()

    # Change to project root for tests
    os.chdir(project_root)

    yield

    # Restore original working directory
    os.chdir(original_cwd)
