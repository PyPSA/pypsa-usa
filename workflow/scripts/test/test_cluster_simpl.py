"""Unit tests for cluster_simpl helpers."""

import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cluster_simpl import resolve_simpl_mode


def test_resolve_simpl_mode_identity():
    assert resolve_simpl_mode("") == "identity"


def test_resolve_simpl_mode_county():
    assert resolve_simpl_mode("county") == "county"


def test_resolve_simpl_mode_kmeans_digits():
    assert resolve_simpl_mode("50") == "kmeans"


def test_resolve_simpl_mode_kmeans_large_digits():
    assert resolve_simpl_mode("2000") == "kmeans"


def test_resolve_simpl_mode_unknown_raises():
    with pytest.raises(ValueError, match="Unknown simpl wildcard"):
        resolve_simpl_mode("foo")


def test_resolve_simpl_mode_unknown_lists_sentinels():
    """Error message must list the recognized values so users can self-correct."""
    with pytest.raises(ValueError) as exc:
        resolve_simpl_mode("bar")
    msg = str(exc.value)
    assert '""' in msg
    assert '"county"' in msg
    assert "digits" in msg or "integer" in msg.lower()
