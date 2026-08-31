"""Place one compressed GODEEEP capacity-factor file where the DAG expects it.

Driven by ``rule retrieve_godeeep_cf`` (rules/retrieve.smk). Retrieval used to
happen inside ``build_renewable_profiles.py``, hidden behind a hardcoded Zenodo
record table that returned ``None`` on a miss. It is now an explicit DAG step
whose source — the local mirror or a Zenodo record — is declared in
``config["godeeep_cf_registry"]`` and resolved by ``godeeep_cf_registry``.

There is NO fallback after resolution: the winning source must serve exactly the
``(scenario, technology, hub height, year)`` the output path names. A missing
file, a checksum mismatch or a resolution that disagrees with the requested
wildcards raises — nothing substitutes another year, hub height or screening
variant (issue #803).
"""

import logging
import os
import re
import shutil
from pathlib import Path

from _helpers import configure_logging, validate_checksum
from godeeep_cf_registry import CfResolution, cf_filename, load_sources, resolve_cf
from zenodo_downloader import ZenodoScenarioDownloader

logger = logging.getLogger(__name__)

#: The ``{cf_file}`` wildcard of rule retrieve_godeeep_cf, e.g.
#: ``wind_gen_cf_2019_125m_compressed`` / ``solar_gen_cf_2019_compressed``.
CF_FILE_RE = re.compile(
    r"^(?P<technology>solar|wind)_gen_cf_(?P<year>\d{4})(?P<wind_height>_\d+m)?_compressed$",
)

#: Optional checksum manifest at the root of a local source: one
#: ``<sha256>  <path relative to root>`` line per published file.
CHECKSUM_MANIFEST = "SHA256SUMS"


def parse_cf_file(cf_file: str) -> tuple[str, str, int]:
    """Split the ``{cf_file}`` wildcard into ``(technology, hub height, year)``.

    Raises
    ------
    ValueError
        If ``cf_file`` is not a published GODEEEP CF file name. The rule's
        ``wildcard_constraints`` already enforce this shape; the check is
        repeated here so the script is safe to call directly.
    """
    match = CF_FILE_RE.match(cf_file)
    if match is None:
        raise ValueError(
            f"{cf_file!r} is not a GODEEEP capacity-factor file name; expected e.g. "
            "'solar_gen_cf_2019_compressed' or 'wind_gen_cf_2019_125m_compressed'.",
        )
    return (
        match.group("technology"),
        match.group("wind_height") or "",
        int(match.group("year")),
    )


def pypsa_technology(technology: str) -> str:
    """A pypsa-usa carrier standing in for a GODEEEP technology family.

    ``resolve_cf`` takes the pypsa-usa technology wildcard, while the output
    path only carries the GODEEEP family. Every wind carrier shares one
    ``TechSpec`` (the hub height comes from ``godeeep_wind_height``), so any of
    them resolves the same file.
    """
    return "solar" if technology == "solar" else "onwind"


def resolve_request(config, scenario: str, cf_file: str) -> CfResolution:
    """Resolve the requested output path against the configured registry.

    The resolution is cross-checked against the wildcards: the registry
    resolves from the config alone, so a config that has drifted from the DAG
    (different hub height, different weather year) must fail loudly instead of
    filling the output path with a different file.

    Raises
    ------
    ValueError
        If the wildcards are malformed, the config is invalid, or the resolved
        file is not the one the output path names.
    CfNotAvailableError
        If no configured source declares the requested dataset and year.
    """
    technology, wind_height, year = parse_cf_file(cf_file)
    requested = cf_filename(technology, wind_height, scenario, year)

    resolution = resolve_cf(config, pypsa_technology(technology), planning_horizon=year)

    if (resolution.scenario, resolution.filename) != (scenario, requested):
        raise ValueError(
            f"rule retrieve_godeeep_cf was asked for '{scenario}/{requested}', but the configured "
            f"godeeep_cf_registry resolves to '{resolution.scenario}/{resolution.filename}' "
            f"(dataset {resolution.dataset_key!r}, year {resolution.year}). Retrieving it would "
            "silently substitute a different scenario, hub height or year. Fix the mismatch "
            "between the requested path and renewable_scenarios / renewable_weather_years / "
            "godeeep_wind_height, or re-run after clearing the stale target.",
        )
    return resolution


def local_source_root(config, resolution: CfResolution) -> Path:
    """Filesystem root of the local source that won the resolution."""
    return Path(load_sources(config)[resolution.source_index].root)


def read_checksum_manifest(manifest: Path) -> dict[str, str]:
    """Parse a ``sha256sum``-style manifest into ``{path: hash}``.

    Entry paths are kept as published (relative to the manifest's directory);
    the leading ``*``/space binary marker of ``sha256sum`` output is stripped.
    """
    entries: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        checksum, _, name = line.partition(" ")
        name = name.strip().lstrip("*")
        if checksum and name:
            entries[os.path.normpath(name)] = checksum
    return entries


def expected_checksum(entries: dict[str, str], relative: str, manifest: Path) -> str:
    """The published hash of ``relative``, keyed by path or by unique basename.

    Mirrors generated with ``sha256sum <dir>/*.nc`` key their entries on the
    bare file name, so both spellings are accepted — but only when the name is
    unambiguous.

    Raises
    ------
    ValueError
        If the manifest has no entry for the file, or several entries share its
        basename with different hashes.
    """
    relative = os.path.normpath(relative)
    if relative in entries:
        return entries[relative]

    basename = os.path.basename(relative)
    matches = {checksum for name, checksum in entries.items() if os.path.basename(name) == basename}
    if len(matches) == 1:
        return matches.pop()
    if len(matches) > 1:
        raise ValueError(
            f"{manifest} lists several conflicting hashes for files named {basename!r} and no entry "
            f"for {relative!r}; cannot tell which one belongs to the requested file.",
        )
    raise ValueError(
        f"{manifest} exists but publishes no checksum for {relative!r}, so the file's integrity "
        "cannot be verified. Regenerate the manifest at the source root, or remove it to skip "
        "checksum verification for this source.",
    )


def verify_local_checksum(source_path: Path, root: Path) -> None:
    """Verify ``source_path`` against the source's manifest, if it has one.

    A source without a ``SHA256SUMS`` manifest is used as-is (the file's
    existence is still required); a source with one must cover the file.

    Raises
    ------
    ValueError
        If the manifest does not cover the file or the hash does not match.
    """
    manifest = root / CHECKSUM_MANIFEST
    if not manifest.is_file():
        logger.info(f"No {CHECKSUM_MANIFEST} at {root}; skipping checksum verification.")
        return

    relative = os.path.relpath(source_path, root)
    checksum = expected_checksum(read_checksum_manifest(manifest), relative, manifest)
    try:
        validate_checksum(source_path, checksum=f"sha256:{checksum}")
    except AssertionError as exc:
        raise ValueError(
            f"Checksum mismatch for {source_path}: {manifest} publishes sha256 {checksum}. "
            "The local mirror is corrupt or out of date; re-stage the file rather than using it.",
        ) from exc
    logger.info(f"Verified sha256 of {relative} against {manifest}.")


def place_local(source_path: Path, out_path: Path, copy: bool) -> None:
    """Symlink (default) or copy the mirrored file to the rule's output path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Snakemake does not clear an existing output before a script runs, and
    # os.symlink refuses an existing target.
    if out_path.is_symlink() or out_path.exists():
        out_path.unlink()

    if copy:
        logger.info(f"Copying {source_path} -> {out_path}")
        shutil.copy2(source_path, out_path)
    else:
        logger.info(f"Linking {source_path} -> {out_path}")
        os.symlink(source_path, out_path)


def retrieve_local(config, resolution: CfResolution, out_path: Path) -> None:
    """Serve a resolved file from a local mirror.

    Raises
    ------
    FileNotFoundError
        If the mirror declares the dataset/year but does not hold the file.
    """
    source_path = Path(resolution.path)
    if not source_path.is_file():
        raise FileNotFoundError(
            f"The local GODEEEP CF source declares {resolution.dataset_key!r} for year "
            f"{resolution.year}, but {source_path} does not exist. Either stage the file there or "
            "drop the year from that source in godeeep_cf_registry so another source can serve it.",
        )
    verify_local_checksum(source_path, local_source_root(config, resolution))
    place_local(source_path, out_path, copy=resolution.copy_local)


def retrieve_zenodo(resolution: CfResolution, out_path: Path) -> None:
    """Download a resolved file from its Zenodo record (checksum-verified)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    downloader = ZenodoScenarioDownloader(download_dir=out_path.parent)
    logger.info(
        f"Downloading {resolution.filename} from Zenodo record {resolution.record_id} "
        f"({resolution.dataset_key}, year {resolution.year}).",
    )
    downloader.download_to_path(resolution.record_id, resolution.filename, out_path)


def retrieve_cf(config, scenario: str, cf_file: str, out_path) -> CfResolution:
    """Resolve and retrieve one GODEEEP CF file; raise on any failure."""
    out_path = Path(out_path)
    resolution = resolve_request(config, scenario, cf_file)
    logger.info(
        f"Resolved {resolution.dataset_key} year {resolution.year} to [{resolution.kind}] {resolution.location}.",
    )

    if resolution.kind == "local":
        retrieve_local(config, resolution, out_path)
    else:
        retrieve_zenodo(resolution, out_path)
    return resolution


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "retrieve_godeeep_cf",
            scenario="historical",
            cf_file="solar_gen_cf_2019_compressed",
        )
    configure_logging(snakemake)

    retrieve_cf(
        snakemake.config,
        snakemake.wildcards.scenario,
        snakemake.wildcards.cf_file,
        snakemake.output[0],
    )
