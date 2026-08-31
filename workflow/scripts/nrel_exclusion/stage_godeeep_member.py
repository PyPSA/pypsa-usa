"""
Stream one member out of a raw GODEEEP zip64 archive onto local disk.

The raw archives are zip64 (~90 GB each, ~4.4 GB members), which rules out the
system ``unzip`` on some nodes; python's ``zipfile`` handles them and streams in
16 MiB blocks so peak memory stays flat regardless of member size.

The staged byte count is checked against the archive's central directory. A short
read (truncated stage, full scratch filesystem, interrupted job) therefore fails
here rather than surfacing later as a corrupt NetCDF.

Usage:
    python stage_godeeep_member.py \
        --zip  /oak/.../GoDEEEP_Capacity_Factors/solar_historical_1980_2022.zip \
        --member solar_historical/solar_gen_cf_2012.nc \
        --output $L_SCRATCH/solar_gen_cf_2012.nc \
        [--expect-bytes 4443329482]
"""

import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

COPY_BLOCK = 16 << 20  # bytes per copyfileobj block


def stage_member(
    zip_path: str | Path,
    member: str,
    dest: str | Path,
    expect_bytes: int | None = None,
) -> int:
    """
    Extract ``member`` from ``zip_path`` to ``dest`` and return the staged byte count.

    ``expect_bytes`` (e.g. the ``member_bytes`` column of ``tasks.tsv``) is checked
    against the archive's central directory before any bytes move, so a stale task
    list fails fast instead of after a multi-GB copy.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        try:
            info = zf.getinfo(str(member))
        except KeyError:
            raise KeyError(f"member {member!r} not found in {zip_path}") from None
        if expect_bytes is not None and info.file_size != expect_bytes:
            raise ValueError(
                f"{zip_path}:{member} central directory reports {info.file_size} bytes, task list expects {expect_bytes}",
            )
        with zf.open(info) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out, length=COPY_BLOCK)

    staged = dest.stat().st_size
    if staged != info.file_size:
        raise OSError(
            f"staged {staged} bytes to {dest}, central directory says {info.file_size} for {zip_path}:{member}",
        )
    return staged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, help="Raw GODEEEP zip64 archive.")
    ap.add_argument("--member", required=True, help="Member path inside the archive.")
    ap.add_argument("--output", required=True, help="Destination file (parent directories are created).")
    ap.add_argument(
        "--expect-bytes",
        type=int,
        default=None,
        help="Optional expected member size; cross-checked against the central directory.",
    )
    args = ap.parse_args()

    t0 = time.monotonic()
    staged = stage_member(args.zip, args.member, args.output, expect_bytes=args.expect_bytes)
    elapsed = time.monotonic() - t0
    print(
        f"[stage] {args.zip}:{args.member} -> {args.output} "
        f"({staged / 1e9:.2f} GB in {elapsed:.1f}s, {staged / 1e6 / max(elapsed, 1e-9):.0f} MB/s)",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
