# PyPSA-USA Tests

This directory contains tests for the PyPSA-USA model components and constraints.

## Overview

These tests validate the functionality of key components in PyPSA-USA, ensuring that constraints and features work as expected. The test suite is organized into separate modules for different components:

1. **Energy Reserve Margin (ERM) and Planning Reserve Margin (PRM)** tests in `test_reserves.py`
2. **Land Use Constraints** tests in `test_land.py`
3. **Policy Constraints** tests in `test_content.py` (RPS and TCT)

## Test Cases

### Energy Reserve and Planning Reserve Margins (test_reserves.py)

- Tests for basic functionality of ERM constraints
  - Variables creation
  - Peak hour constraint application
  - Data storage
  - Storage contribution to ERM
  - Transmission line contribution to ERM

- Tests for ERM constraints across multiple regions
  - Constraint application in multiple regions
  - Interconnection contribution to ERM

- Tests for basic functionality of PRM constraints
  - Constraint creation
  - Capacity sizing according to PRM requirements

- Tests for combining ERM and PRM constraints

### Land Use Constraints (test_land.py)

- Tests for land use constraint creation and functionality
  - Constraint creation
  - Limiting capacity expansion
  - Multiple carriers in the same region
  - Edge cases (empty land_region, infinite p_nom_max)
  - Models without extendable generators
  - Full optimization workflow

### Policy Constraints (test_content.py)

- Tests for Renewable Portfolio Standards (RPS)
  - Constraint creation
  - Effect on generation mix
  - Multi-region targets

- Tests for Technology Capacity Targets (TCT)
  - Constraint creation
  - Minimum target enforcement
  - Maximum target enforcement
  - "Existing" value handling

## Running the Tests

To run all tests:

```bash
# Using the run_tests.py script
./run_tests.py

# Directly using pytest
pytest -v
```

To run specific test files:

```bash
# Run a specific test file
pytest test_reserves.py -v
pytest test_land.py -v
pytest test_content.py -v
```

To run specific test functions:

```bash
# Run a specific test function
pytest test_reserves.py::test_erm_peak_demand_hour -v
pytest test_land.py::test_land_use_constraint_limiting -v

# Run tests matching a pattern
pytest -v -k "storage"
```

## Requirements

- **pytest**: The tests use pytest as the testing framework
- **GLPK**: The tests use the GLPK solver by default

Install the required packages:

```bash
pip install pytest
```

## Test Data

The tests create temporary files and data as needed:

- RPS and TCT tests create temporary CSV files with configuration data
- ERM/PRM tests create temporary PRM CSV files in `/tmp/`

## Adding New Tests

When adding new tests:

1. Follow the pytest style: write test functions that start with `test_`
2. Use fixtures to set up and share test data
3. Use assertions to verify expected behavior
4. Document the test purpose with clear docstrings

Example of a new test:

```python
def test_new_feature(fixture_network, fixture_config):
    """Test description."""
    n = fixture_network.copy()

    # Test setup

    # Execute function being tested
    result = function_to_test(n)

    # Assert expected outcomes
    assert result == expected_value, "Error message if test fails"
