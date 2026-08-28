"""Tests for shared constant usage."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EULP_PATH = REPO_ROOT / "workflow/scripts/eulp.py"


def test_eulp_resampling_keeps_dataset_specific_2018_hour_count():
    """EULP resampling is tied to 2018 data and should keep explicit hour checks."""
    source = EULP_PATH.read_text()

    # One check, in Eulp._resample_data. Was two until the never-instantiated
    # EulpTotals near-clone was removed; the surviving resampler still guards
    # the 2018 hour count explicitly.
    assert source.count("assert len(resampled) == 8760") == 1


def test_hours_per_year_uses_shared_constant():
    """Avoid reintroducing bare 8760 literals in workflow source."""
    source_files = [
        *REPO_ROOT.glob("workflow/scripts/**/*.py"),
        *REPO_ROOT.glob("workflow/rules/**/*.smk"),
    ]

    offenders = []
    for path in source_files:
        if path.name == "constants.py" or path == EULP_PATH or "/test/" in path.as_posix():
            continue

        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            code = line.split("#", maxsplit=1)[0]
            if "8760" in code:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_number}: {line.strip()}")

    assert not offenders, "Use the shared HOURS_PER_YEAR constant instead of bare 8760 literals:\n" + "\n".join(
        offenders,
    )
