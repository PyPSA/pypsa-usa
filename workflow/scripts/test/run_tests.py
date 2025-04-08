#!/usr/bin/env python
"""
Run all tests for the ERM and PRM constraints.

This script discovers and runs all tests for the Energy Reserve Margin (ERM) and
Planning Reserve Margin (PRM) constraints using pytest.
"""
import os
import sys

import pytest

# Add the parent directory to the path so the tests can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    # Run pytest with specific arguments
    args = [
        "-v",  # verbose output
        "--color=yes",  # colored output
        os.path.dirname(__file__),  # test directory
    ]

    # Add any command line arguments
    args.extend(sys.argv[1:])

    # Run pytest and use the return code as exit code
    sys.exit(pytest.main(args))
