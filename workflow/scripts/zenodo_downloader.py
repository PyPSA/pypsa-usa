"""Download scenarios from Zenodo."""

from pathlib import Path

import requests


class ZenodoScenarioDownloader:
    """Download scenarios from Zenodo."""

    def __init__(self, download_dir="./data"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)

        # NREL land-access availability + bus-capacity artifacts; one record
        # holds all 48 avail_*.nc / caps_*.nc files.
        # GODEEEP _compressed.nc records used by the NREL land-access path.
        # Keyed by (tech + wind_height + scenario) — no year_range — because
        # one record holds all years per (tech, scenario).
        self.scenario_records = {
            "nrel_exclusion_v1": 20316475,
            "solar_historical_compressed": 20127513,
            "solar_rcp45hotter_compressed": 20127523,
            "solar_rcp45cooler_compressed": 20127562,
            "solar_rcp85hotter_compressed": 20127589,
            "solar_rcp85cooler_compressed": 20127633,
            "wind_125m_historical_compressed": 20127520,
            "wind_125m_rcp45hotter_compressed": 20127545,
            "wind_125m_rcp45cooler_compressed": 20127572,
            "wind_125m_rcp85hotter_compressed": 20127604,
            "wind_125m_rcp85cooler_compressed": 20127645,
        }

        # Cache for record metadata to avoid repeated API calls
        self._metadata_cache = {}

    def get_record_metadata(self, record_id):
        """Get metadata for a record (with caching)."""
        if record_id in self._metadata_cache:
            return self._metadata_cache[record_id]

        url = f"https://zenodo.org/api/records/{record_id}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            metadata = response.json()
            self._metadata_cache[record_id] = metadata
            return metadata

        except requests.exceptions.RequestException as e:
            print(f"Failed to get metadata for record {record_id}: {e}")
            return None

    def download_scenario_file(self, scenario_final, scenario, filename, force_redownload=False):
        """
        Download a specific file from a scenario dataset.

        Parameters
        ----------
        scenario_final : str
            Lookup key into scenario_records (e.g. "solar_rcp45hotter_compressed").
        scenario : str
            Climate-scenario name used to choose the on-disk subdir
            (e.g. "historical", "rcp45hotter").
        filename : str
            Name of the file to download, e.g., "solar_gen_cf_2030_compressed.nc".
        force_redownload : bool, optional
            If True, re-download the file even if it exists locally. Default is False.
        """
        (self.download_dir / "zenodo" / scenario).mkdir(exist_ok=True)
        local_filepath = f"{self.download_dir}/zenodo/{scenario}/{filename}"

        # Check if file already exists locally and skip Zenodo
        if Path(local_filepath).exists() and not force_redownload:
            print(
                f"File already exists locally: {local_filepath}. Skipping download. Use force_redownload=True to re-download.",
            )
            return str(local_filepath)
        # Only check record_id if we need to download
        else:
            record_id = self.scenario_records.get(scenario_final)

            if not record_id:
                print(f"No record ID found for scenario: {scenario_final}")
                print("Available scenarios with record IDs:")
                for scenario, rec_id in self.scenario_records.items():
                    if rec_id is not None:
                        print(f"  - {scenario} (ID: {rec_id})")
                return None

            return self._download_file(record_id, filename, Path(local_filepath), force_redownload)

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
        # Ensure directory exists
        local_filepath.parent.mkdir(parents=True, exist_ok=True)

        # Get record metadata
        metadata = self.get_record_metadata(record_id)
        if not metadata:
            return None

        # Find the specific file
        target_file = None
        for file_info in metadata.get("files", []):
            if file_info["key"] == filename:
                target_file = file_info
                break

        if not target_file:
            print(f"File '{filename}' not found in record {record_id}")
            print("Available files:")
            for file_info in metadata.get("files", []):
                print(f"  - {file_info['key']}")
            return None

        # Download the file
        download_url = target_file["links"]["self"]
        file_size_mb = target_file["size"] / (1024 * 1024)

        print(f"Downloading {filename} from record {record_id}...")
        print(f"Size: {file_size_mb:.1f} MB")
        print(f"Saving to: {local_filepath}")

        try:
            response = requests.get(download_url, stream=True)
            response.raise_for_status()

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

            print(f"Successfully downloaded {filename}")
            return str(local_filepath)

        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}")
            if Path(local_filepath).exists():
                Path(local_filepath).unlink()  # Remove partial file
            return None
