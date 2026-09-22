"""Unit tests for the on-disk EIA response cache in ``eia.py``.

The cache sits at the request layer (``DataExtractor._request_eia_data``), so
every EIA consumer benefits without changes. Two properties matter:

1. a repeated request costs no network call, and
2. the API key never reaches disk — the cache key and the stored header are
   built from the request url with ``api_key`` removed, which also makes the
   directory shareable between users.

The HTTP layer is patched throughout; nothing here touches the network.
"""

import json
import os
import sys

import pytest
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import eia

pytestmark = pytest.mark.fast

SECRET = "SECRETKEY123"
URL = f"{eia.API_BASE}natural-gas/pri/sum/data/?api_key={SECRET}&frequency=monthly&start=2019-01"
OTHER_URL = f"{eia.API_BASE}natural-gas/pri/sum/data/?api_key={SECRET}&frequency=monthly&start=2020-01"
PAYLOAD = {"response": {"data": [{"period": "2019-01", "value": 2.74, "state": "U.S."}]}}


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class _Recorder:
    """Replacement for ``requests.Session.get`` that counts calls.

    Patched onto the class as a plain (non-descriptor) attribute, so ``self``
    of the session is not passed through — the first argument is the url.
    """

    def __init__(self, payload=PAYLOAD, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.urls: list[str] = []

    def __call__(self, url, **kwargs):
        self.urls.append(url)
        return _FakeResponse(self.payload, self.status_code)

    @property
    def calls(self) -> int:
        return len(self.urls)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Point the cache at a tmp_path via the documented env override."""
    d = tmp_path / "eia-cache"
    monkeypatch.setenv("EIA_CACHE_DIR", str(d))
    return d


@pytest.fixture
def recorder(monkeypatch):
    r = _Recorder()
    monkeypatch.setattr(requests.Session, "get", r)
    return r


def cache_files(cache_dir):
    return sorted(cache_dir.glob("*.json"))


def test_second_identical_request_is_served_from_cache(cache_dir, recorder):
    first = eia.DataExtractor._request_eia_data(URL)

    assert recorder.calls == 1
    assert first == PAYLOAD

    files = cache_files(cache_dir)
    assert len(files) == 1

    second = eia.DataExtractor._request_eia_data(URL)

    assert recorder.calls == 1  # no second network call
    assert second == PAYLOAD
    assert cache_files(cache_dir) == files


def test_api_key_never_reaches_disk(cache_dir, recorder):
    eia.DataExtractor._request_eia_data(URL)

    # the real request did carry the key
    assert SECRET in recorder.urls[0]

    for path in cache_dir.rglob("*"):
        if path.is_file():
            assert SECRET not in path.name
            assert SECRET not in path.read_text()

    entry = json.loads(cache_files(cache_dir)[0].read_text())
    assert entry["payload"] == PAYLOAD
    assert SECRET not in entry["url"]
    assert entry["url"].startswith(eia.API_BASE)
    assert "frequency=monthly" in entry["url"]
    assert entry["fetched_at"]


def test_different_url_misses(cache_dir, recorder):
    eia.DataExtractor._request_eia_data(URL)
    eia.DataExtractor._request_eia_data(OTHER_URL)

    assert recorder.calls == 2
    assert len(cache_files(cache_dir)) == 2


def test_same_query_different_key_shares_one_entry(cache_dir, recorder):
    """The cache is keyed without the api key, so two users share it."""
    eia.DataExtractor._request_eia_data(URL)
    eia.DataExtractor._request_eia_data(URL.replace(SECRET, "ANOTHERKEY"))

    assert recorder.calls == 1
    assert len(cache_files(cache_dir)) == 1


def test_query_parameter_order_does_not_matter(cache_dir, recorder):
    reordered = f"{eia.API_BASE}natural-gas/pri/sum/data/?start=2019-01&frequency=monthly&api_key={SECRET}"
    eia.DataExtractor._request_eia_data(URL)
    eia.DataExtractor._request_eia_data(reordered)

    assert recorder.calls == 1


def test_non_200_is_not_cached_and_still_raises(cache_dir, monkeypatch):
    failing = _Recorder(payload={}, status_code=503)
    monkeypatch.setattr(requests.Session, "get", failing)

    with pytest.raises(requests.ConnectionError):
        eia.DataExtractor._request_eia_data(URL)

    assert not cache_dir.exists() or cache_files(cache_dir) == []


def test_corrupt_cache_entry_is_refetched(cache_dir, recorder):
    eia.DataExtractor._request_eia_data(URL)
    assert recorder.calls == 1

    cache_files(cache_dir)[0].write_text("not json")

    assert eia.DataExtractor._request_eia_data(URL) == PAYLOAD
    assert recorder.calls == 2


def test_strip_api_key_removes_only_the_key():
    stripped = eia.strip_api_key(URL)

    assert "api_key" not in stripped
    assert SECRET not in stripped
    assert "frequency=monthly" in stripped
    assert "start=2019-01" in stripped
