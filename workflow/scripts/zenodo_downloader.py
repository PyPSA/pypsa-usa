"""Download scenarios from Zenodo.

Every failure path in this module raises :class:`ZenodoDownloadError`. Nothing
returns ``None`` and nothing silently substitutes a different record, file or
partially-downloaded artifact: a caller that gets a path back is guaranteed a
complete file whose checksum matches the one Zenodo publishes for it.

Record IDs for the GODEEEP ``*_compressed.nc`` capacity factors are no longer
hardcoded here — they are declared in the ``godeeep_cf_registry`` config block
and reach this module through ``download_to_path``.
"""

import time
from pathlib import Path

import requests
from _helpers import validate_checksum

# Transient-failure retry policy for both metadata and file requests.
MAX_DOWNLOAD_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 2.0
METADATA_TIMEOUT_SECONDS = 30
# (connect, read) — the read budget has to cover a slow chunk, not the file.
DOWNLOAD_TIMEOUT_SECONDS = (10, 120)
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class ZenodoDownloadError(Exception):
    """Raised when a Zenodo record, file or download cannot be resolved."""


def _is_transient(exc: Exception) -> bool:
    """True for errors worth retrying (connection resets, timeouts, 5xx/429)."""
    if isinstance(
        exc,
        requests.exceptions.Timeout | requests.exceptions.ConnectionError | requests.exceptions.ChunkedEncodingError,
    ):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        status = getattr(exc.response, "status_code", None)
        return status is not None and (status >= 500 or status in RETRYABLE_STATUS_CODES)
    return False


class ZenodoScenarioDownloader:
    """Download scenarios from Zenodo."""

    def __init__(self, download_dir="./data"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # NREL land-access availability + bus-capacity artifacts; one record
        # holds all 48 avail_*.nc / caps_*.nc files.
        self.scenario_records = {
            "nrel_exclusion_v1": 20316475,
        }

        # Cache for record metadata to avoid repeated API calls
        self._metadata_cache = {}

    def get_record_metadata(self, record_id):
        """
        Get metadata for a record (with caching).

        Raises
        ------
        ZenodoDownloadError
            If the record cannot be fetched or does not parse as JSON.
        """
        if record_id in self._metadata_cache:
            return self._metadata_cache[record_id]

        url = f"https://zenodo.org/api/records/{record_id}"
        what = f"metadata for Zenodo record {record_id} ({url})"

        for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
            try:
                response = requests.get(url, timeout=METADATA_TIMEOUT_SECONDS)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                self._retry_or_raise(e, attempt, what)

        try:
            metadata = response.json()
        except ValueError as e:
            raise ZenodoDownloadError(f"Zenodo returned a non-JSON response for {what}: {e}") from e

        self._metadata_cache[record_id] = metadata
        return metadata

    def download_scenario_file(self, scenario_final, scenario, filename, force_redownload=False):
        """
        Download a specific file from a scenario dataset.

        Parameters
        ----------
        scenario_final : str
            Lookup key into scenario_records (e.g. "nrel_exclusion_v1").
        scenario : str
            Climate-scenario name used to choose the on-disk subdir
            (e.g. "historical", "rcp45hotter").
        filename : str
            Name of the file to download, e.g., "caps_solar_reference.nc".
        force_redownload : bool, optional
            If True, re-download the file even if it exists locally. Default is False.

        Returns
        -------
        str
            Path to the complete, checksum-verified local file.

        Raises
        ------
        ZenodoDownloadError
            If `scenario_final` is not a known key, if `filename` is not in the
            record, or if the download fails or fails checksum validation.
        """
        # Validated up front, even when the file is already on disk: an unknown
        # key is a config bug and must never resolve to a cached artifact.
        try:
            record_id = self.scenario_records[scenario_final]
        except KeyError as e:
            available = ", ".join(sorted(self.scenario_records)) or "<none>"
            raise ZenodoDownloadError(
                f"Unknown Zenodo dataset key '{scenario_final}'. Available keys: {available}.",
            ) from e

        local_filepath = self.download_dir / "zenodo" / scenario / filename
        local_filepath.parent.mkdir(parents=True, exist_ok=True)

        # Check if file already exists locally and skip Zenodo
        if local_filepath.exists() and not force_redownload:
            print(
                f"File already exists locally: {local_filepath}. Skipping download. Use force_redownload=True to re-download.",
            )
            return str(local_filepath)

        return self._download_file(record_id, filename, local_filepath, force_redownload)

    def download_to_path(self, record_id, filename, out_path, force_redownload=False):
        """
        Download a file from a Zenodo record to an explicit on-disk path.

        Useful when called from a snakemake rule whose `output:` already names
        the destination — bypasses the scenario subdir conventions of
        download_scenario_file.

        Parameters
        ----------
        record_id : int or str
            Zenodo record ID.
        filename : str
            File name inside the record (used to find the right file in the
            record's manifest).
        out_path : str or Path
            Destination path. Parent directories are created if missing.

        Returns
        -------
        str
            Path to the complete, checksum-verified local file.

        Raises
        ------
        ZenodoDownloadError
            If `filename` is not in the record, or if the download fails or
            fails checksum validation.
        """
        out_path = Path(out_path)
        if out_path.exists() and not force_redownload:
            print(f"File already exists locally: {out_path}. Skipping download.")
            return str(out_path)
        return self._download_file(record_id, filename, out_path, force_redownload)

    def _download_file(self, record_id, filename, local_filepath, force_redownload=False):
        """
        Internal method to download a file from Zenodo.

        This is only called after confirming the file doesn't exist locally.
        """
        local_filepath = Path(local_filepath)
        local_filepath.parent.mkdir(parents=True, exist_ok=True)

        metadata = self.get_record_metadata(record_id)
        record_files = metadata.get("files", [])

        # Find the specific file
        target_file = next((f for f in record_files if f["key"] == filename), None)
        if target_file is None:
            available = ", ".join(sorted(f["key"] for f in record_files)) or "<none>"
            raise ZenodoDownloadError(
                f"File '{filename}' not found in Zenodo record {record_id}. Available files: {available}.",
            )

        # Checked before spending bandwidth: an unverifiable download is a
        # failure, not something to wave through.
        checksum = target_file.get("checksum")
        if not checksum:
            raise ZenodoDownloadError(
                f"Zenodo record {record_id} publishes no checksum for '{filename}'; "
                "refusing to download a file whose integrity cannot be verified.",
            )

        download_url = target_file["links"]["self"]
        print(f"Downloading {filename} from record {record_id}...")
        if target_file.get("size"):
            print(f"Size: {target_file['size'] / (1024 * 1024):.1f} MB")
        print(f"Saving to: {local_filepath}")

        what = f"'{filename}' from Zenodo record {record_id} ({download_url})"
        for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
            try:
                response = requests.get(download_url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS)
                response.raise_for_status()
                self._write_stream(response, local_filepath)
                break
            except requests.exceptions.RequestException as e:
                local_filepath.unlink(missing_ok=True)  # never keep a torn file
                self._retry_or_raise(e, attempt, what)

        try:
            validate_checksum(local_filepath, checksum=checksum)
        except (AssertionError, ValueError) as e:
            local_filepath.unlink(missing_ok=True)
            raise ZenodoDownloadError(
                f"Checksum mismatch for {what}: expected {checksum}. "
                f"The partial or corrupt file {local_filepath} has been removed; re-run the rule.",
            ) from e

        print(f"Successfully downloaded {filename}")
        return str(local_filepath)

    @staticmethod
    def _write_stream(response, local_filepath):
        """Stream a response body to disk, reporting progress on large files."""
        total_size = int(response.headers.get("content-length", 0))
        downloaded_size = 0

        with open(local_filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)

                    # Show progress for large files
                    if total_size > 10 * 1024 * 1024:  # Show progress for files > 10MB
                        progress = (downloaded_size / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end="", flush=True)

        if total_size > 10 * 1024 * 1024:
            print()  # New line after progress

    @staticmethod
    def _retry_or_raise(exc, attempt, what):
        """Sleep before the next attempt, or raise once retries are exhausted.

        Non-transient failures (404, 403, ...) are raised immediately — retrying
        them would only delay the error.
        """
        if not _is_transient(exc):
            raise ZenodoDownloadError(f"Failed to fetch {what}: {exc}") from exc
        if attempt >= MAX_DOWNLOAD_ATTEMPTS:
            raise ZenodoDownloadError(
                f"Failed to fetch {what} after {MAX_DOWNLOAD_ATTEMPTS} attempts: {exc}",
            ) from exc
        delay = BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
        print(
            f"Transient error fetching {what} (attempt {attempt}/{MAX_DOWNLOAD_ATTEMPTS}): {exc}. Retrying in {delay:.0f}s..."
        )
        time.sleep(delay)
