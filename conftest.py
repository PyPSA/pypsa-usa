"""Repo-root pytest configuration.

Auto-marks every test collected from ``workflow/scripts/test/`` with the
``fast`` marker so the existing unit tests join Tier A without code changes.
Tests under ``tests/`` declare their marker explicitly.
"""

from pathlib import Path

import pytest

_UNIT_TEST_DIR = (Path(__file__).parent / "workflow" / "scripts" / "test").resolve()


def pytest_collection_modifyitems(config, items):
    for item in items:
        try:
            item_path = Path(item.fspath).resolve()
        except (TypeError, ValueError):
            continue
        if _UNIT_TEST_DIR in item_path.parents:
            item.add_marker(pytest.mark.fast)
