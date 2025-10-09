# PyPSA-USA Test Suite

This directory contains the complete test suite for PyPSA-USA, organized by functionality and test type.

## Directory Structure

```
tests/
├── conftest.py                 # Main test configuration and shared fixtures
├── api/                        # API tests
│   ├── conftest.py            # API-specific fixtures
│   ├── test_workflow.py       # Core API functionality tests
│   ├── test_workflow_usage.py # Usage scenario tests
│   ├── test_environment_handling.py # Environment variable handling tests
│   └── README_TESTS.md        # Detailed API test documentation
├── workflow/                   # Workflow and script tests
│   ├── conftest.py            # Workflow-specific fixtures
│   ├── fixtures/              # Test data fixtures
│   ├── test_land.py           # Land use constraint tests
│   ├── test_policy.py         # Policy constraint tests
│   ├── test_reserves.py       # Reserve constraint tests
│   └── README.md              # Workflow test documentation
├── unit/                       # Unit tests (to be added)
└── integration/                # Integration tests (to be added)
```

## Running Tests

### Run All Tests
```bash
# From project root
uv run pytest tests/

# Or using the test runner script
./run_api_tests.py
```

### Run Specific Test Categories
```bash
# API tests only
uv run pytest tests/api/

# Workflow tests only
uv run pytest tests/workflow/

# Unit tests only
uv run pytest tests/unit/

# Integration tests only
uv run pytest tests/integration/
```

### Run Specific Test Files
```bash
# API workflow tests
uv run pytest tests/api/test_workflow.py

# API usage scenario tests
uv run pytest tests/api/test_workflow_usage.py

# Environment handling tests
uv run pytest tests/api/test_environment_handling.py

# Workflow policy tests
uv run pytest tests/workflow/test_policy.py
```

### Run with Coverage
```bash
# All tests with coverage
uv run pytest tests/ --cov=pypsa_usa --cov-report=html --cov-report=term

# API tests with coverage
uv run pytest tests/api/ --cov=pypsa_usa.api --cov-report=html
```

### Run with Specific Options
```bash
# Verbose output
uv run pytest tests/ -v

# Stop on first failure
uv run pytest tests/ -x

# Run only tests matching a pattern
uv run pytest tests/ -k "environment"

# Run tests excluding slow tests
uv run pytest tests/ -m "not slow"
```

## Test Categories

### API Tests (`tests/api/`)
- **Core API functionality**: Workspace management, configuration handling, workflow execution
- **Usage scenarios**: Different ways users can run workflows
- **Environment handling**: Python interpreter and environment variable management
- **Error handling**: Exception scenarios and edge cases

### Workflow Tests (`tests/workflow/`)
- **Land use constraints**: Land availability and usage restrictions
- **Policy constraints**: Renewable portfolio standards, carbon limits, etc.
- **Reserve constraints**: Operating reserves and capacity requirements
- **Script functionality**: Individual workflow script testing

### Unit Tests (`tests/unit/`)
- **Individual function testing**: Testing specific functions in isolation
- **Component testing**: Testing individual components without dependencies
- **Mock-heavy tests**: Tests that heavily use mocking for isolation

### Integration Tests (`tests/integration/`)
- **End-to-end workflows**: Complete workflow execution testing
- **Component interaction**: Testing how different components work together
- **Real data testing**: Tests using actual data files and configurations

## Test Configuration

### pytest.ini
The main pytest configuration is in the project root (`pytest.ini`):
- Test discovery patterns
- Default options
- Markers for test categorization
- Warning filters

### conftest.py Files
- **`tests/conftest.py`**: Main configuration, shared fixtures, path setup
- **`tests/api/conftest.py`**: API-specific fixtures and utilities
- **`tests/workflow/conftest.py`**: Workflow-specific fixtures and test data

## Test Data and Fixtures

### Fixtures Directory (`tests/workflow/fixtures/`)
Contains CSV files with test data for various constraints:
- `ces_reeds.csv`: Clean Energy Standards data
- `portfolio_standards.csv`: Portfolio standard requirements
- `regional_co2_limits.csv`: Regional CO2 emission limits
- `technology_capacity_targets.csv`: Technology capacity targets
- And more...

### Temporary Data
Tests use temporary directories and files that are automatically cleaned up:
- Temporary workspaces for API testing
- Mock configuration files
- Temporary test data

## Writing New Tests

### For API Tests
1. Add test functions to appropriate files in `tests/api/`
2. Use fixtures from `tests/api/conftest.py`
3. Follow naming convention: `test_<functionality>`
4. Use descriptive test names and docstrings

### For Workflow Tests
1. Add test functions to appropriate files in `tests/workflow/`
2. Use fixtures from `tests/workflow/conftest.py`
3. Use test data from `tests/workflow/fixtures/`
4. Test both success and failure scenarios

### For Unit Tests
1. Create new files in `tests/unit/`
2. Focus on testing individual functions in isolation
3. Use extensive mocking to avoid dependencies
4. Test edge cases and error conditions

### For Integration Tests
1. Create new files in `tests/integration/`
2. Test complete workflows and component interactions
3. Use real data when possible
4. Test realistic usage scenarios

## Best Practices

### Test Organization
- Group related tests in the same file
- Use descriptive class and function names
- Keep tests focused and atomic
- Use fixtures for common setup

### Test Data
- Use fixtures for reusable test data
- Create temporary data for tests that need modification
- Use realistic but minimal test data
- Clean up after tests

### Assertions
- Use specific assertions (e.g., `assert x == y` not `assert x`)
- Test both positive and negative cases
- Include meaningful error messages
- Test edge cases and boundary conditions

### Mocking
- Mock external dependencies
- Use mocks for expensive operations
- Verify mock calls when important
- Keep mocks simple and focused

## Continuous Integration

Tests are designed to run in CI/CD environments:
- All tests use dry runs to avoid actual workflow execution
- Tests are isolated and don't depend on external services
- Temporary files and directories are automatically cleaned up
- Tests run quickly and reliably

## Contributing

When adding new functionality:
1. Add corresponding tests to the appropriate directory
2. Update this README if adding new test categories
3. Ensure all tests pass before submitting
4. Add new fixtures to conftest.py files as needed
5. Update test documentation as needed
