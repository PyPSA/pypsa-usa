# PyPSA-USA API Test Suite

This directory contains comprehensive test cases for the PyPSA-USA API workflow functionality. The tests cover different usage scenarios and ensure the API works correctly with various configurations.

## Test Files

### `test_workflow.py`
Core API functionality tests including:
- Workspace management (setting/getting default workspace)
- Configuration handling (file and dictionary configs)
- Workflow execution with different parameters
- Error handling and exception scenarios
- Environment variable management

### `test_workflow_usage.py`
Usage scenario tests covering different ways users can run workflows:
- Basic usage with default workspace
- Explicit workspace specification
- Configuration dictionary vs file usage
- Different target combinations
- Various core counts and flags
- Mixed target types (rules and files)
- Error handling scenarios

### `test_environment_handling.py`
Specific tests for environment variable handling that ensures Snakemake uses the correct Python interpreter:
- PYTHON environment variable setting
- PYTHONPATH environment variable management
- Environment restoration after execution
- Exception handling with environment cleanup
- Multiple consecutive calls
- Complex PYTHONPATH scenarios

### `conftest.py`
Shared fixtures for all test files:
- Temporary workspace creation
- Mock workflow paths
- Sample configuration data
- Snakemake mocking utilities
- Environment cleanup utilities

## Running Tests

### Using the test runner script:
```bash
# Run all API tests
./run_api_tests.py

# Run specific test file
./run_api_tests.py --test test_workflow_usage

# Run with coverage reporting
./run_api_tests.py --coverage

# Run in quiet mode
./run_api_tests.py --quiet
```

### Using pytest directly:
```bash
# Run all API tests
uv run pytest src/pypsa_usa/api/

# Run specific test file
uv run pytest src/pypsa_usa/api/test_workflow_usage.py

# Run with verbose output
uv run pytest src/pypsa_usa/api/ -v

# Run with coverage
uv run pytest src/pypsa_usa/api/ --cov=pypsa_usa.api --cov-report=html
```

### Using pytest with specific test patterns:
```bash
# Run only environment handling tests
uv run pytest src/pypsa_usa/api/ -k "environment"

# Run only usage scenario tests
uv run pytest src/pypsa_usa/api/ -k "usage"

# Run tests excluding slow tests
uv run pytest src/pypsa_usa/api/ -m "not slow"
```

## Test Coverage

The test suite covers:

### Core API Functions
- ✅ `run_workflow()` - Main workflow execution function
- ✅ `set_default_workspace()` - Workspace management
- ✅ `get_default_workspace()` - Workspace retrieval
- ✅ `create_user_config()` - Configuration creation
- ✅ `list_available_configs()` - Config template listing
- ✅ `get_workflow_info()` - Workflow information
- ✅ `setup_user_workspace()` - Workspace setup

### Usage Scenarios
- ✅ Basic usage with default workspace
- ✅ Explicit workspace specification
- ✅ Configuration file vs dictionary
- ✅ Relative vs absolute config paths
- ✅ Single vs multiple targets
- ✅ Different core counts (1, 2, 4, 8, 16)
- ✅ Force flags (forceall, forcetargets)
- ✅ Additional Snakemake kwargs
- ✅ Dry run vs actual execution
- ✅ String vs Path objects
- ✅ Non-existent workspace creation
- ✅ Complex configuration dictionaries
- ✅ Specific rule targets
- ✅ Mixed target types
- ✅ Concurrent workflow handling

### Environment Handling
- ✅ PYTHON environment variable setting
- ✅ PYTHONPATH environment variable management
- ✅ Environment restoration after execution
- ✅ Exception handling with cleanup
- ✅ Multiple consecutive calls
- ✅ Complex PYTHONPATH scenarios
- ✅ UV-managed vs system Python

### Error Handling
- ✅ Snakemake execution failures
- ✅ Exception handling
- ✅ Invalid configuration paths
- ✅ Missing default workspace
- ✅ Environment variable cleanup

## Test Data

Tests use:
- **Temporary workspaces**: Created and cleaned up automatically
- **Mock configurations**: Sample config files and dictionaries
- **Mock Snakemake**: Simulated Snakemake execution for dry runs
- **Mock workflow paths**: Simulated workflow directory structure

## Key Features Tested

### Environment Variable Fix
The tests specifically verify the fix for the "No module named snakemake" issue:
- Environment variables are set correctly during execution
- Snakemake uses the correct Python interpreter
- Environment is restored after execution
- Exception handling preserves environment cleanup

### Dry Run Testing
All tests use `dryrun=True` to avoid actual workflow execution:
- Tests verify Snakemake is called with correct parameters
- No actual data processing or file I/O occurs
- Fast test execution
- Safe for CI/CD environments

### Comprehensive Coverage
The test suite covers all major usage patterns:
- Different ways to specify workspaces and configurations
- Various target combinations and execution options
- Error scenarios and edge cases
- Environment handling and cleanup

## Contributing

When adding new API functionality:
1. Add corresponding test cases to the appropriate test file
2. Update this README with new test coverage
3. Ensure all tests pass with `uv run pytest src/pypsa_usa/api/`
4. Add new fixtures to `conftest.py` if needed

## Dependencies

Tests require:
- `pytest` - Test framework
- `pytest-cov` - Coverage reporting (optional)
- `unittest.mock` - Mocking utilities
- `tempfile` - Temporary file/directory creation
- `pathlib` - Path handling
