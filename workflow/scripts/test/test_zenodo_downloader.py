"""Unit tests for the Zenodo downloader's fail-loud contract.

The CF retrieval registry has zero fallback paths: a bad dataset key, a missing
file inside a record, an exhausted retry budget or a checksum mismatch must all
raise ``ZenodoDownloadError`` with an actionable message — never return ``None``
and never leave a truncated artifact on disk for a later rule to open.

Every test monkeypatches ``requests.get``; nothing here touches the network.
"""

import hashlib
import os
import sys

import pytest
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import zenodo_downloader as zd
from zenodo_downloader import ZenodoDownloadError, ZenodoScenarioDownloader

pytestmark = pytest.mark.fast

RECORD_ID = 20316475
FILENAME = "caps_solar_reference.nc"
PAYLOAD = b"netcdf-bytes" * 64


def md5(payload: bytes) -> str:
    return f"md5:{hashlib.md5(payload).hexdigest()}"


def record_metadata(files=None):
    """A Zenodo record manifest shaped like the real API response."""
    files = files if files is not None else [(FILENAME, PAYLOAD), ("avail_solar_reference.nc", b"other")]
    return {
        "files": [
            {
                "key": key,
                "size": len(content),
                "checksum": md5(content),
                "links": {"self": f"https://zenodo.org/api/records/{RECORD_ID}/files/{key}/content"},
            }
            for key, content in files
        ],
    }


class FakeResponse:
    """The slice of `requests.Response` the downloader actually touches."""

    def __init__(self, *, json_data=None, content=b"", status_code=200):
        self._json = json_data
        self.content = content
        self.status_code = status_code
        self.headers = {"content-length": str(len(content))}

    def raise_for_status(self):
        """Raise an HTTPError carrying this response, as requests does."""
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} Server Error", response=self)

    def json(self):
        """Return the JSON body, or raise like requests on a non-JSON body."""
        if self._json is None:
            raise ValueError("no JSON body")
        return self._json

    def iter_content(self, chunk_size=8192):
        """Yield the body in chunks, as a streamed response does."""
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Record backoff delays instead of sleeping through them."""
    delays = []
    monkeypatch.setattr(zd.time, "sleep", delays.append)
    return delays


@pytest.fixture
def no_network(monkeypatch):
    """Fail the test loudly if any code path reaches requests.get."""

    def boom(*args, **kwargs):
        raise AssertionError(f"unexpected network call: {args} {kwargs}")

    monkeypatch.setattr(zd.requests, "get", boom)


def install_fake_get(monkeypatch, *, metadata=None, file_responses=None):
    """Route metadata URLs to `metadata` and file URLs through `file_responses`.

    `file_responses` is a list of per-attempt outcomes: either an exception to
    raise or a FakeResponse to return. Returns the list of requested URLs so
    tests can assert on the attempt count.
    """
    metadata = record_metadata() if metadata is None else metadata
    file_responses = list(file_responses or [])
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if url.endswith(f"/api/records/{RECORD_ID}"):
            return FakeResponse(json_data=metadata)
        outcome = file_responses.pop(0) if file_responses else FakeResponse(content=PAYLOAD)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(zd.requests, "get", fake_get)
    return calls


def file_calls(calls):
    return [url for url in calls if not url.endswith(f"/api/records/{RECORD_ID}")]


def test_unknown_dataset_key_raises_listing_available_keys(tmp_path, no_network):
    downloader = ZenodoScenarioDownloader(download_dir=tmp_path / "data")

    with pytest.raises(ZenodoDownloadError) as excinfo:
        downloader.download_scenario_file("solar_rcp45hotter_compressed", "rcp45hotter", FILENAME)

    message = str(excinfo.value)
    assert "solar_rcp45hotter_compressed" in message
    assert "nrel_exclusion_v1" in message  # available keys are named


def test_compressed_cf_records_are_no_longer_hardcoded(tmp_path):
    downloader = ZenodoScenarioDownloader(download_dir=tmp_path / "data")
    assert not [key for key in downloader.scenario_records if key.endswith("_compressed")]


def test_missing_file_in_record_raises_listing_available_files(tmp_path, monkeypatch):
    install_fake_get(monkeypatch)
    downloader = ZenodoScenarioDownloader(download_dir=tmp_path / "data")

    with pytest.raises(ZenodoDownloadError) as excinfo:
        downloader.download_to_path(RECORD_ID, "caps_wind_open.nc", tmp_path / "out.nc")

    message = str(excinfo.value)
    assert "caps_wind_open.nc" in message
    assert FILENAME in message
    assert "avail_solar_reference.nc" in message
    assert not (tmp_path / "out.nc").exists()


def test_transient_errors_are_retried_then_succeed(tmp_path, monkeypatch, no_sleep):
    calls = install_fake_get(
        monkeypatch,
        file_responses=[
            requests.exceptions.ConnectionError("connection reset"),
            FakeResponse(status_code=503),
            FakeResponse(content=PAYLOAD),
        ],
    )
    downloader = ZenodoScenarioDownloader(download_dir=tmp_path / "data")
    out_path = tmp_path / "out" / "caps.nc"

    result = downloader.download_to_path(RECORD_ID, FILENAME, out_path)

    assert result == str(out_path)
    assert out_path.read_bytes() == PAYLOAD
    assert len(file_calls(calls)) == 3
    assert no_sleep == [2.0, 4.0]  # exponential backoff between the 3 attempts


def test_permanent_failure_raises_after_three_attempts(tmp_path, monkeypatch, no_sleep):
    calls = install_fake_get(
        monkeypatch,
        file_responses=[requests.exceptions.ConnectionError("no route to host")] * 3,
    )
    downloader = ZenodoScenarioDownloader(download_dir=tmp_path / "data")
    out_path = tmp_path / "out" / "caps.nc"

    with pytest.raises(ZenodoDownloadError) as excinfo:
        downloader.download_to_path(RECORD_ID, FILENAME, out_path)

    assert "after 3 attempts" in str(excinfo.value)
    assert FILENAME in str(excinfo.value)
    assert len(file_calls(calls)) == zd.MAX_DOWNLOAD_ATTEMPTS == 3
    assert not out_path.exists()


def test_non_transient_http_error_is_not_retried(tmp_path, monkeypatch):
    calls = install_fake_get(monkeypatch, file_responses=[FakeResponse(status_code=404)])
    downloader = ZenodoScenarioDownloader(download_dir=tmp_path / "data")

    with pytest.raises(ZenodoDownloadError, match="404"):
        downloader.download_to_path(RECORD_ID, FILENAME, tmp_path / "out.nc")

    assert len(file_calls(calls)) == 1


def test_checksum_mismatch_raises_and_removes_partial_file(tmp_path, monkeypatch):
    install_fake_get(monkeypatch, file_responses=[FakeResponse(content=b"truncated")])
    downloader = ZenodoScenarioDownloader(download_dir=tmp_path / "data")
    out_path = tmp_path / "out" / "caps.nc"

    with pytest.raises(ZenodoDownloadError) as excinfo:
        downloader.download_to_path(RECORD_ID, FILENAME, out_path)

    assert "Checksum mismatch" in str(excinfo.value)
    assert md5(PAYLOAD) in str(excinfo.value)
    assert not out_path.exists()  # the truncated body must not survive


def test_record_without_checksum_raises_before_downloading(tmp_path, monkeypatch):
    metadata = record_metadata()
    del metadata["files"][0]["checksum"]
    calls = install_fake_get(monkeypatch, metadata=metadata)
    downloader = ZenodoScenarioDownloader(download_dir=tmp_path / "data")

    with pytest.raises(ZenodoDownloadError, match="no checksum"):
        downloader.download_to_path(RECORD_ID, FILENAME, tmp_path / "out.nc")

    assert file_calls(calls) == []


def test_metadata_fetch_failure_raises(tmp_path, monkeypatch, no_sleep):
    def fake_get(url, **kwargs):
        raise requests.exceptions.Timeout("read timed out")

    monkeypatch.setattr(zd.requests, "get", fake_get)
    downloader = ZenodoScenarioDownloader(download_dir=tmp_path / "data")

    with pytest.raises(ZenodoDownloadError) as excinfo:
        downloader.download_to_path(RECORD_ID, FILENAME, tmp_path / "out.nc")

    assert str(RECORD_ID) in str(excinfo.value)
    assert "after 3 attempts" in str(excinfo.value)


def test_scenario_download_verifies_checksum_and_returns_path(tmp_path, monkeypatch):
    install_fake_get(monkeypatch)
    download_dir = tmp_path / "data"
    downloader = ZenodoScenarioDownloader(download_dir=download_dir)

    result = downloader.download_scenario_file("nrel_exclusion_v1", "historical", FILENAME)

    expected = download_dir / "zenodo" / "historical" / FILENAME
    assert result == str(expected)
    assert expected.read_bytes() == PAYLOAD


def test_existing_file_is_reused_without_network(tmp_path, monkeypatch, no_network):
    download_dir = tmp_path / "data"
    target = download_dir / "zenodo" / "historical" / FILENAME
    target.parent.mkdir(parents=True)
    target.write_bytes(PAYLOAD)
    downloader = ZenodoScenarioDownloader(download_dir=download_dir)

    assert downloader.download_scenario_file("nrel_exclusion_v1", "historical", FILENAME) == str(target)
