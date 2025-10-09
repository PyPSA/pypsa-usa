#!/usr/bin/env python3
"""
Test runner for PyPSA-USA test suite.

This script provides a comprehensive test runner for all PyPSA-USA tests,
organized by category and functionality.
"""

import subprocess
import sys
from pathlib import Path


def run_tests(test_category=None, test_pattern=None, verbose=True, coverage=False, stop_on_failure=False):
    """
    Run PyPSA-USA tests.

    Args:
        test_category: Test category to run (api, workflow, unit, integration, all)
        test_pattern: Specific test pattern to run
        verbose: Whether to run tests in verbose mode
        coverage: Whether to run with coverage reporting
        stop_on_failure: Whether to stop on first failure
    """
    # Get the project root directory
    project_root = Path(__file__).parent

    # Build pytest command
    cmd = ["uv", "run", "pytest"]

    if verbose:
        cmd.append("-v")

    if stop_on_failure:
        cmd.append("-x")

    if coverage:
        if test_category == "api":
            cmd.extend(["--cov=pypsa_usa.api", "--cov-report=html", "--cov-report=term"])
        elif test_category == "workflow":
            cmd.extend(["--cov=pypsa_usa.workflow", "--cov-report=html", "--cov-report=term"])
        else:
            cmd.extend(["--cov=pypsa_usa", "--cov-report=html", "--cov-report=term"])

    # Add test path based on category
    if test_category == "api":
        if test_pattern:
            cmd.append(f"tests/api/{test_pattern}.py")
        else:
            cmd.append("tests/api/")
    elif test_category == "workflow":
        if test_pattern:
            cmd.append(f"tests/workflow/{test_pattern}.py")
        else:
            cmd.append("tests/workflow/")
    elif test_category == "unit":
        if test_pattern:
            cmd.append(f"tests/unit/{test_pattern}.py")
        else:
            cmd.append("tests/unit/")
    elif test_category == "integration":
        if test_pattern:
            cmd.append(f"tests/integration/{test_pattern}.py")
        else:
            cmd.append("tests/integration/")
    elif test_category == "all" or test_category is None:
        if test_pattern:
            cmd.append(f"tests/{test_pattern}")
        else:
            cmd.append("tests/")
    else:
        print(f"❌ Unknown test category: {test_category}")
        print("Available categories: api, workflow, unit, integration, all")
        return False

    # Add additional options
    cmd.extend(
        [
            "--tb=short",
            "--strict-markers",
            "--disable-warnings",
        ]
    )

    print(f"Running command: {' '.join(cmd)}")
    print(f"Working directory: {project_root}")

    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        print("\n✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ Error: 'uv' command not found. Please ensure uv is installed and in PATH.")
        return False


def main():
    """Main function to run tests based on command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run PyPSA-USA tests")
    parser.add_argument(
        "--category",
        choices=["api", "workflow", "unit", "integration", "all"],
        default="all",
        help="Test category to run (default: all)",
    )
    parser.add_argument(
        "--pattern",
        help="Specific test pattern to run (e.g., 'test_workflow_usage')",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run tests with coverage reporting",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Run tests in quiet mode",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop on first failure",
    )

    args = parser.parse_args()

    success = run_tests(
        test_category=args.category,
        test_pattern=args.pattern,
        verbose=not args.quiet,
        coverage=args.coverage,
        stop_on_failure=args.stop_on_failure,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
