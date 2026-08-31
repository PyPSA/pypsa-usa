"""
Enumerate the raw GODEEEP capacity-factor archives and emit the compression task list.

The raw archives ship one NetCDF per (tech, year) inside a small number of zip64
files. This script reads only their *central directories* (no decompression, no
staging), so it is cheap enough to run on a login node, and writes a tab-separated
task list consumed by ``compress_godeeep_task.sh``:

    idx  tech  year  zip  member  member_bytes  dest_filename

``idx`` is the 1-based line number and doubles as the Slurm array task ID. Rows are
grouped by tech in the order given on the command line and, within a tech, ordered
by descending year (recent years first, so a partial sweep still yields the years
most runs care about).

Coverage is asserted, not repaired: every requested year must resolve to exactly one
member of exactly one archive, otherwise the script fails naming the tech and the
missing years. There is no partial task list.

Usage:
    python make_compress_tasks.py --output tasks.tsv
    python make_compress_tasks.py --techs solar --first-year 2000 --output solar.tsv
"""

import argparse
import re
import sys
import zipfile
from dataclasses import astuple, dataclass
from pathlib import Path

DEFAULT_ZIP_DIR = Path("/oak/stanford/groups/iazevedo/GoDEEEP_Capacity_Factors")
DEFAULT_TECHS = ("solar", "wind_125m", "wind_100m")
DEFAULT_FIRST_YEAR = 1980
DEFAULT_LAST_YEAR = 2022


@dataclass(frozen=True)
class Task:
    """One (tech, year) compression job: where the raw member lives, what to call the output."""

    idx: int
    tech: str
    year: int
    zip_path: str
    member: str
    member_bytes: int
    dest_filename: str

    def as_tsv(self) -> str:
        """Render the task as one tab-separated ``tasks.tsv`` line (no trailing newline)."""
        return "\t".join(str(field) for field in astuple(self))


def member_pattern(tech: str) -> re.Pattern[str]:
    """Regex matching the raw member *basenames* of ``tech``, capturing the year."""
    if tech == "solar":
        return re.compile(r"^solar_gen_cf_(?P<year>\d{4})\.nc$")
    wind = re.fullmatch(r"wind_(?P<height>\d+m)", tech)
    if wind is None:
        raise ValueError(f"unsupported tech {tech!r}: expected 'solar' or 'wind_<height>m' (e.g. wind_100m)")
    return re.compile(rf"^wind_gen_cf_(?P<year>\d{{4}})_{wind['height']}\.nc$")


def dest_filename(member: str) -> str:
    """Registry filename for a raw member (``.../wind_gen_cf_2012_125m.nc`` -> ``..._compressed.nc``)."""
    return Path(member).stem + "_compressed.nc"


def scan_tech(zip_dir: Path, tech: str, years: range) -> dict[int, tuple[Path, str, int]]:
    """
    Map ``year -> (zip_path, member, member_bytes)`` for one tech.

    Raises if the archives are missing, a year is served by two members, or any
    requested year is absent — the caller gets a complete map or an exception.
    """
    pattern = member_pattern(tech)
    archives = sorted(zip_dir.glob(f"{tech}_historical_*.zip"))
    if not archives:
        raise FileNotFoundError(f"{tech}: no archives matching '{tech}_historical_*.zip' under {zip_dir}")

    wanted = set(years)
    found: dict[int, tuple[Path, str, int]] = {}
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                match = pattern.match(Path(info.filename).name)
                if match is None:
                    continue
                year = int(match["year"])
                if year not in wanted:
                    continue
                if year in found:
                    prior_zip, prior_member, _ = found[year]
                    raise ValueError(
                        f"{tech} {year}: duplicate members "
                        f"{prior_zip.name}:{prior_member} and {archive.name}:{info.filename}"
                    )
                found[year] = (archive, info.filename, info.file_size)

    missing = sorted(wanted - found.keys())
    if missing:
        raise ValueError(
            f"{tech}: incomplete coverage — missing years {missing} "
            f"(searched {[a.name for a in archives]} in {zip_dir}); "
            f"found {len(found)}/{len(wanted)} of {min(years)}-{max(years)}"
        )
    return found


def enumerate_tasks(
    zip_dir: Path,
    techs: tuple[str, ...] = DEFAULT_TECHS,
    years: range = range(DEFAULT_FIRST_YEAR, DEFAULT_LAST_YEAR + 1),
) -> list[Task]:
    """Build the full, contiguously numbered task list: techs in order, years descending."""
    tasks: list[Task] = []
    for tech in techs:
        found = scan_tech(zip_dir, tech, years)
        for year in sorted(found, reverse=True):
            archive, member, size = found[year]
            tasks.append(
                Task(
                    idx=len(tasks) + 1,
                    tech=tech,
                    year=year,
                    zip_path=str(archive),
                    member=member,
                    member_bytes=size,
                    dest_filename=dest_filename(member),
                )
            )
    return tasks


def write_tasks_tsv(tasks: list[Task], out_path: str | Path) -> Path:
    """Write ``tasks`` to ``out_path`` (headerless TSV) and return the path."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(task.as_tsv() + "\n" for task in tasks))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip-dir", type=Path, default=DEFAULT_ZIP_DIR, help="Directory holding the raw GODEEEP zips.")
    ap.add_argument(
        "--techs",
        nargs="+",
        default=list(DEFAULT_TECHS),
        help="Techs to enumerate, in output order (solar, wind_100m, wind_125m, ...).",
    )
    ap.add_argument("--first-year", type=int, default=DEFAULT_FIRST_YEAR, help="First year required per tech.")
    ap.add_argument("--last-year", type=int, default=DEFAULT_LAST_YEAR, help="Last year required per tech.")
    ap.add_argument("--output", type=Path, default=Path("tasks.tsv"), help="Destination TSV.")
    args = ap.parse_args()

    years = range(args.first_year, args.last_year + 1)
    tasks = enumerate_tasks(args.zip_dir, tuple(args.techs), years)
    out = write_tasks_tsv(tasks, args.output)

    total_bytes = sum(task.member_bytes for task in tasks)
    print(f"[tasks] wrote {out}: {len(tasks)} tasks, {total_bytes / 1e12:.2f} TB raw", flush=True)
    for tech in args.techs:
        rows = [task for task in tasks if task.tech == tech]
        print(
            f"[tasks]   {tech}: {len(rows)} years {min(t.year for t in rows)}-{max(t.year for t in rows)}", flush=True
        )
    print(f"[tasks] submit with --array=1-{len(tasks)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
