"""Download scenarios from Zenodo."""

from pathlib import Path

import requests


class ZenodoScenarioDownloader:
    """Download scenarios from Zenodo."""

    def __init__(self, download_dir="./data"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)

        # Mapping of scenarios to their Zenodo record IDs
        self.scenario_records = {
            "solar_historical": 17410574,
            "solar_rcp45hotter_2020_2059": None,
            "solar_rcp45hotter_2060_2099": None,
            "solar_rcp45cooler_2020_2059": None,
            "solar_rcp45cooler_2060_2099": None,
            "solar_rcp85hotter_2020_2059": None,
            "solar_rcp85hotter_2060_2099": None,
            "solar_rcp85cooler_2020_2059": None,
            "solar_rcp85cooler_2060_2099": None,
            "wind_100m_historical": 17429560,
            "wind_100m_rcp45hotter_2020_2039": None,
            "wind_100m_rcp45hotter_2040_2059": None,
            "wind_100m_rcp45hotter_2060_2079": None,
            "wind_100m_rcp45hotter_2080_2099": None,
            "wind_100m_rcp45cooler_2020_2039": None,
            "wind_100m_rcp45cooler_2040_2059": None,
            "wind_100m_rcp45cooler_2060_2079": None,
            "wind_100m_rcp45cooler_2080_2099": None,
            "wind_100m_rcp85hotter_2020_2039": None,
            "wind_100m_rcp85hotter_2040_2059": None,
            "wind_100m_rcp85hotter_2060_2079": None,
            "wind_100m_rcp85hotter_2080_2099": None,
            "wind_100m_rcp85cooler_2020_2039": None,
            "wind_100m_rcp85cooler_2040_2059": None,
            "wind_100m_rcp85cooler_2060_2079": None,
            "wind_100m_rcp85cooler_2080_2099": None,
            "capacities": 17576458,
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
        scenario_name : str
            Name of the scenario, e.g., "solar_historical".
        filename : str
            Name of the file to download, e.g., "solar_gen_cf_1980_aggregated.nc".
        force_redownload : bool, optional
            If True, re-download the file even if it exists locally. Default is False.
        """
        # pointing file path to workflow/data/zenodo/{scenario_name}
        if scenario_final == "capacities":
            local_filepath = f"{self.download_dir}/zenodo/{filename}"
        else:
            (self.download_dir / "zenodo" / scenario).mkdir(
                exist_ok=True,
            )  # create the zenodo directory if it doesn't exist
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

    def download_by_record_id(self, record_id, filename, force_redownload=False):
        """
        Download a file directly using a record ID.

        Parameters
        ----------
        record_id : int or str
            Zenodo record ID (e.g. 17059209).
        filename : str
            Name of the file to download.
        force_redownload : bool, optional
            If True, redownload even if file exists locally. Default is False.

        Returns
        -------
        str or None
            Path to the downloaded file, or None if download failed.
        """
        # pointing file path to workflow/data/zenodo
        local_filepath = f"{self.download_dir}/zenodo/{filename}"

        # Check if file already exists
        if Path(local_filepath).exists() and not force_redownload:
            print(f"File {filename} already exists. Use force_redownload=True to redownload.")
            return str(local_filepath)

        # Only proceed with download if needed
        return self._download_file(record_id, filename, Path(local_filepath), force_redownload)

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

    def list_available_files(self, scenario_name):
        """List all available files in a scenario dataset."""
        record_id = self.scenario_records.get(scenario_name)
        if not record_id:
            print(f"No record ID found for scenario: {scenario_name}")
            print("Available scenarios with record IDs:")
            for scenario, rec_id in self.scenario_records.items():
                if rec_id is not None:
                    print(f"  - {scenario} (ID: {rec_id})")
            return []

        return self.list_files_by_record_id(record_id)

    def list_files_by_record_id(self, record_id):
        """List all files in a record by record ID."""
        metadata = self.get_record_metadata(record_id)
        if not metadata:
            return []

        files = []
        record_title = metadata.get("metadata", {}).get("title", "Unknown")
        print(f"Available files in record {record_id} ({record_title}):")

        for file_info in metadata.get("files", []):
            filename = file_info["key"]
            size_mb = file_info["size"] / (1024 * 1024)
            files.append(filename)
            print(f"  - {filename} ({size_mb:.1f} MB)")

        return files

    def get_available_scenarios(self):
        """Get list of available scenarios (ones with record IDs)."""
        available = []
        print("Available scenarios:")
        for scenario, record_id in self.scenario_records.items():
            if record_id is not None:
                available.append(scenario)
                print(f"  - {scenario} (Record ID: {record_id})")
        return available


def download_scenario_file(scenario_final, scenario, filename, download_dir="./data/zenodo"):
    """Quick function to download a single file from a scenario."""
    downloader = ZenodoScenarioDownloader(download_dir)
    return downloader.download_scenario_file(scenario_final, scenario, filename)


def download_by_record_id(record_id, filename, download_dir="./data/zenodo"):
    """Quick function to download a file directly by record ID."""
    downloader = ZenodoScenarioDownloader(download_dir)
    return downloader.download_by_record_id(record_id, filename)


def list_available_scenarios():
    """List all available scenarios."""
    downloader = ZenodoScenarioDownloader()
    return downloader.get_available_scenarios()
