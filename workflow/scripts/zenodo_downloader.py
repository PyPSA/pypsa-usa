from pathlib import Path

import requests


class ZenodoScenarioDownloader:
    def __init__(self, download_dir="./data"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)

        # Mapping of scenarios to their Zenodo record IDs
        self.scenario_records = {
            "solar_historical": 17059209,
            "solar_rcp45hotter_2020_2059": 17069731,
            "solar_rcp45hotter_2060_2099": 17069758,
            "solar_rcp45cooler_2020_2059": 17069655,
            "solar_rcp45cooler_2060_2099": 17069689,
            "solar_rcp85hotter_2020_2059": 17069842,
            "solar_rcp85hotter_2060_2099": 17069858,
            "solar_rcp85cooler_2020_2059": 17069790,
            "solar_rcp85cooler_2060_2099": 17069818,
            "wind_100m_historical": 17073578,
            "wind_100m_rcp45hotter_2020_2039": 17070280,
            "wind_100m_rcp45hotter_2040_2059": 17070299,
            "wind_100m_rcp45hotter_2060_2079": 17070334,
            "wind_100m_rcp45hotter_2080_2099": 17070348,
            "wind_100m_rcp45cooler_2020_2039": 17070385,
            "wind_100m_rcp45cooler_2040_2059": 17070419,
            "wind_100m_rcp45cooler_2060_2079": 17070459,
            "wind_100m_rcp45cooler_2080_2099": 17070496,
            "wind_100m_rcp85hotter_2020_2039": 17070534,
            "wind_100m_rcp85hotter_2040_2059": 17070556,
            "wind_100m_rcp85hotter_2060_2079": 17070591,
            "wind_100m_rcp85hotter_2080_2099": 17070632,
            "wind_100m_rcp85cooler_2020_2039": 17070655,
            "wind_100m_rcp85cooler_2040_2059": 17070676,
            "wind_100m_rcp85cooler_2060_2079": 17070717,
            "wind_100m_rcp85cooler_2080_2099": 17070771,
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

    def download_scenario_file(self, scenario_name, filename, force_redownload=False):
        """
        Download a specific file from a scenario dataset.

        Parameters
        ----------
        - scenario_name: e.g., "solar_historical"
        - filename: e.g., "solar_historical_solar_gen_cf_1980_bus_mean.nc"
        - force_redownload: If True, redownload even if file exists
        """
        # pointing file path to workflow/data/zenodo
        (self.download_dir / "zenodo").mkdir(exist_ok=True)  # create the zenodo directory if it doesn't exist
        local_filepath = f"{self.download_dir}/zenodo/{filename}"

        # Check if file already exists
        if Path(local_filepath).exists() and not force_redownload:
            print(f"File {filename} already exists. Use force_redownload=True to redownload.")
            return str(local_filepath)

        # Get the record ID for this scenario
        record_id = self.scenario_records.get(scenario_name)

        if not record_id:
            print(f"No record ID found for scenario: {scenario_name}")
            print("Available scenarios with record IDs:")
            for scenario, rec_id in self.scenario_records.items():
                if rec_id is not None:
                    print(f"  - {scenario} (ID: {rec_id})")
            return None

        return self.download_by_record_id(record_id, filename, force_redownload)

    def download_by_record_id(self, record_id, filename, force_redownload=False):
        """
        Download a file directly using a record ID.

        Parameters
        ----------
        - record_id: Zenodo record ID (e.g., 17059209)
        - filename: Name of the file to download
        - force_redownload: If True, redownload even if file exists
        """
        # pointing file path to workflow/data/zenodo
        local_filepath = f"{self.download_dir}/zenodo/{filename}"

        # Check if file already exists
        if Path(local_filepath).exists() and not force_redownload:
            print(f"File {filename} already exists. Use force_redownload=True to redownload.")
            return str(local_filepath)

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


def download_scenario_file(scenario_name, filename, download_dir="./data/zenodo"):
    """
    Quick function to download a single file from a scenario.

    Example:
    filepath = download_scenario_file("solar_historical",
                                    "solar_historical_solar_gen_cf_1980_bus_mean.nc")
    """
    downloader = ZenodoScenarioDownloader(download_dir)
    return downloader.download_scenario_file(scenario_name, filename)


def download_by_record_id(record_id, filename, download_dir="./data/zenodo"):
    """
    Quick function to download a file directly by record ID

    Example:
    filepath = download_by_record_id(17059209, "solar_historical_solar_gen_cf_1980_bus_mean.nc")
    """
    downloader = ZenodoScenarioDownloader(download_dir)
    return downloader.download_by_record_id(record_id, filename)


def list_available_scenarios():
    """List all available scenarios."""
    downloader = ZenodoScenarioDownloader()
    return downloader.get_available_scenarios()
