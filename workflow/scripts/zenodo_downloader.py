"""Download scenarios from Zenodo."""

import shutil
from pathlib import Path
from zipfile import BadZipFile, ZipFile

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
        (self.download_dir / "zenodo" / scenario).mkdir(
            parents=True,
            exist_ok=True,
        )
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
                raise ValueError(
                    f"No Zenodo record ID configured for scenario: {scenario_final}",
                )

            return self._download_file(record_id, filename, Path(local_filepath), force_redownload)

    def download_to_path(self, record_id, filename, out_path, force_redownload=False):
        """
        Download a file from a Zenodo record to an explicit on-disk path.

        Useful when called from a snakemake rule whose `output:` already names
        the destination — bypasses the scenario subdir conventions of
        download_scenario_file / download_by_record_id.

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

    def _download_file(
        self,
        record_id,
        filename,
        local_filepath,
        force_redownload=False,
    ):
        """Download a file directly or extract it from a Zenodo ZIP archive."""
        local_filepath.parent.mkdir(parents=True, exist_ok=True)

        metadata = self.get_record_metadata(record_id)
        if not metadata:
            raise RuntimeError(
                f"Could not retrieve metadata for Zenodo record {record_id}",
            )

        files = metadata.get("files", [])

        # Some Zenodo records expose the requested NetCDF file directly.
        target_file = next(
            (file_info for file_info in files if file_info["key"] == filename),
            None,
        )

        if target_file is not None:
            self._download_url(
                target_file["links"]["self"],
                local_filepath,
                target_file["size"],
            )
            return str(local_filepath)

        # Future renewable datasets are published as ZIP archives.
        zip_file = next(
            (file_info for file_info in files if file_info["key"].lower().endswith(".zip")),
            None,
        )

        if zip_file is None:
            available_files = ", ".join(file_info["key"] for file_info in files)
            raise FileNotFoundError(
                f"Neither '{filename}' nor a ZIP archive was found in "
                f"Zenodo record {record_id}. Available files: {available_files}",
            )

        archive_path = local_filepath.parent / zip_file["key"]

        if force_redownload or not archive_path.exists():
            self._download_url(
                zip_file["links"]["self"],
                archive_path,
                zip_file["size"],
            )

        try:
            with ZipFile(archive_path) as archive:
                archive_filename = filename.replace("_aggregated.nc", ".nc")

                matches = [member for member in archive.namelist() if Path(member).name in {filename, archive_filename}]

                if len(matches) != 1:
                    raise FileNotFoundError(
                        f"Expected exactly one of '{filename}' or "
                        f"'{archive_filename}' in '{archive_path.name}', "
                        f"found {len(matches)}",
                    )

                with (
                    archive.open(matches[0]) as source,
                    local_filepath.open("wb") as destination,
                ):
                    shutil.copyfileobj(
                        source,
                        destination,
                        length=1024 * 1024,
                    )

        except (BadZipFile, OSError):
            local_filepath.unlink(missing_ok=True)
            raise

        return str(local_filepath)

    def _download_url(self, download_url, destination, expected_size):
        """Download a URL to a local path using a temporary partial file."""
        partial_path = destination.with_suffix(
            destination.suffix + ".part",
        )

        print(
            f"Downloading {destination.name} ({expected_size / 1024**3:.1f} GiB)...",
        )

        try:
            with requests.get(
                download_url,
                stream=True,
                timeout=(30, 300),
            ) as response:
                response.raise_for_status()

                with partial_path.open("wb") as output:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024,
                    ):
                        if chunk:
                            output.write(chunk)

            actual_size = partial_path.stat().st_size
            if expected_size and actual_size != expected_size:
                raise OSError(
                    f"Incomplete download for {destination.name}: expected {expected_size} bytes, got {actual_size}",
                )

            partial_path.replace(destination)

        except Exception:
            partial_path.unlink(missing_ok=True)
            raise

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
