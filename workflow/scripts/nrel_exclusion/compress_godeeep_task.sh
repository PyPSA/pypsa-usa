#!/bin/bash
# Compress one GODEEEP capacity-factor year: stage -> compress -> sanity -> publish.
#
# This is the per-task driver behind compress_godeeep_array.sbatch and the script
# that produced the Oak registry at
# /oak/stanford/groups/iazevedo/GoDEEEP_Capacity_Factors_compressed.
#
# The task index is the 1-based line number in tasks.tsv (see make_compress_tasks.py)
# and doubles as $SLURM_ARRAY_TASK_ID:
#
#   idx  tech  year  zip  member  member_bytes  dest_filename
#
# Flow, per task:
#   1. idempotent skip  — destination + SHA256SUMS.d sidecar agree -> exit 0
#   2. stage            — member out of the zip64 archive onto node-local disk
#                         ($L_SCRATCH when set, else $SCRATCH), removed by an EXIT trap
#   3. compress         — compress_godeeep.py (uint8, scale_factor 1/254, zlib 4)
#   4. sanity           — dims, uint8 dtype, NaN mask, 12-hour max|err| <= 1/254 vs raw
#   5. publish          — copy to <dest>.part, chmod 0664, re-hash, atomic rename,
#                         then write the sidecar
#
# Nothing here is wired into the Snakemake DAG: the registry is built once, out of
# band, and consumed read-only by the workflow.
#
# Usage:
#   bash compress_godeeep_task.sh 42
#   bash compress_godeeep_task.sh --index 42 --dest-root /path/to/registry --tasks /path/to/tasks.tsv
#
# Environment overrides: DEST_ROOT, TASKS_TSV, PYTHON, STAGE_ROOT,
#                        EXPECT_TIME / EXPECT_SOUTH_NORTH / EXPECT_WEST_EAST.

set -euo pipefail
umask 0002

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEST_ROOT="${DEST_ROOT:-/oak/stanford/groups/iazevedo/GoDEEEP_Capacity_Factors_compressed}"
TASKS_TSV="${TASKS_TSV:-}"
PYTHON="${PYTHON:-python}"
# Raw GODEEEP grid: 8760 hours on the 299x424 Lambert Conformal CONUS domain.
EXPECT_TIME="${EXPECT_TIME:-8760}"
EXPECT_SOUTH_NORTH="${EXPECT_SOUTH_NORTH:-299}"
EXPECT_WEST_EAST="${EXPECT_WEST_EAST:-424}"

TASK_IDX=""

die() {
    echo "[task] ERROR: $*" >&2
    exit 1
}

usage() {
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --index) TASK_IDX="${2:?--index needs a value}"; shift 2 ;;
        --tasks) TASKS_TSV="${2:?--tasks needs a value}"; shift 2 ;;
        --dest-root) DEST_ROOT="${2:?--dest-root needs a value}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) die "unknown flag $1" ;;
        *) TASK_IDX="$1"; shift ;;
    esac
done

: "${TASKS_TSV:=$DEST_ROOT/tasks.tsv}"
[ -n "$TASK_IDX" ] || die "no task index given (positional or --index); see --help"
[ -f "$TASKS_TSV" ] || die "task list not found: $TASKS_TSV (generate it with make_compress_tasks.py)"

# ---- resolve the task ------------------------------------------------------
line="$(awk -v n="$TASK_IDX" 'NR == n' "$TASKS_TSV")"
[ -n "$line" ] || die "task index $TASK_IDX is out of range for $TASKS_TSV ($(wc -l < "$TASKS_TSV") tasks)"

IFS=$'\t' read -r row_idx tech year zip_path member member_bytes dest_name <<<"$line"
[ "$row_idx" = "$TASK_IDX" ] || die "$TASKS_TSV line $TASK_IDX carries idx $row_idx — task list is malformed"

rel="historical/$tech/$dest_name"
dest="$DEST_ROOT/$rel"
sidecar_dir="$DEST_ROOT/SHA256SUMS.d"
sidecar="$sidecar_dir/$dest_name.sha256"

echo "[task $TASK_IDX] $tech $year -> $rel"

# ---- 1. idempotent skip ----------------------------------------------------
if [ -f "$dest" ] && [ -f "$sidecar" ]; then
    want="$(awk '{print $1; exit}' "$sidecar")"
    have="$(sha256sum "$dest" | awk '{print $1}')"
    if [ "$want" = "$have" ]; then
        echo "[task $TASK_IDX] SKIP $rel (sha256 verified against sidecar)"
        exit 0
    fi
    echo "[task $TASK_IDX] sidecar mismatch ($have != $want) — rebuilding"
fi

# ---- 2. stage --------------------------------------------------------------
if [ -z "${STAGE_ROOT:-}" ]; then
    if [ -n "${L_SCRATCH:-}" ]; then
        STAGE_ROOT="$L_SCRATCH"
    else
        STAGE_ROOT="${SCRATCH:?neither L_SCRATCH nor SCRATCH is set; pass STAGE_ROOT explicitly}"
    fi
fi
tmpd="$STAGE_ROOT/godeeep_compress/task_$TASK_IDX"
rm -rf "$tmpd"
mkdir -p "$tmpd"
trap 'rm -rf "$tmpd"; rm -f "$dest.part"' EXIT

raw="$tmpd/$(basename "$member")"
comp="$tmpd/$dest_name"

t0=$SECONDS
"$PYTHON" "$SCRIPT_DIR/stage_godeeep_member.py" \
    --zip "$zip_path" --member "$member" --output "$raw" --expect-bytes "$member_bytes"
t_stage=$((SECONDS - t0))

# ---- 3. compress -----------------------------------------------------------
t0=$SECONDS
"$PYTHON" "$SCRIPT_DIR/compress_godeeep.py" --input "$raw" --output "$comp"
t_comp=$((SECONDS - t0))

# ---- 4. sanity: dims, dtype, NaN mask, 12-hour round-trip vs the raw file ---
"$PYTHON" - "$raw" "$comp" "$EXPECT_TIME" "$EXPECT_SOUTH_NORTH" "$EXPECT_WEST_EAST" <<'PY'
import sys

import netCDF4
import numpy as np

raw_path, comp_path = sys.argv[1], sys.argv[2]
expected = dict(zip(("Time", "south_north", "west_east"), (int(a) for a in sys.argv[3:6])))
tol = 1.0 / 254.0


def fail(msg):
    raise SystemExit(f"[sanity] FAIL {msg}")


with netCDF4.Dataset(comp_path) as dc, netCDF4.Dataset(raw_path) as dr:
    dims = {name: len(dim) for name, dim in dc.dimensions.items()}
    for name, size in expected.items():
        if dims.get(name) != size:
            fail(f"dimension {name}={dims.get(name)}, expected {size} (got {dims})")

    cf = dc.variables["capacity_factor"]
    if cf.dtype != np.uint8:
        fail(f"capacity_factor dtype {cf.dtype}, expected uint8")

    raw_cf = dr.variables["capacity_factor"]
    raw_cf.set_auto_mask(False)  # raw carries NaN in-band; keep it as float, not masked
    for hour in np.linspace(0, expected["Time"] - 1, 12).astype(int):
        decoded = cf[int(hour)]  # auto-scaled to float, 255 -> masked
        decoded = np.ma.filled(decoded, np.nan) if np.ma.isMaskedArray(decoded) else np.asarray(decoded)
        decoded = decoded.astype(np.float64)
        reference = np.asarray(raw_cf[int(hour)], dtype=np.float64)

        nan_decoded, nan_reference = np.isnan(decoded), np.isnan(reference)
        if not np.array_equal(nan_decoded, nan_reference):
            fail(f"NaN mask differs at hour {hour}: {int(nan_decoded.sum())} vs {int(nan_reference.sum())}")

        valid = ~nan_reference
        if valid.any():
            err = float(np.abs(decoded[valid] - reference[valid]).max())
            if err > tol + 1e-9:
                fail(f"round-trip error {err:.6g} > {tol:.6g} at hour {hour}")

print("[sanity] OK dims/dtype/NaN-mask/round-trip", flush=True)
PY

# ---- 5. publish (atomic) ---------------------------------------------------
digest="$(sha256sum "$comp" | awk '{print $1}')"
mkdir -p "$(dirname "$dest")" "$sidecar_dir"

cp "$comp" "$dest.part"
chmod 0664 "$dest.part"
copied="$(sha256sum "$dest.part" | awk '{print $1}')"
[ "$copied" = "$digest" ] || die "destination hash mismatch after copy ($copied != $digest)"
mv -f "$dest.part" "$dest"

printf '%s  %s\n' "$digest" "$rel" > "$sidecar"
chmod 0664 "$sidecar"

echo "[task $TASK_IDX] OK $rel stage=${t_stage}s comp=${t_comp}s size=$(stat -c %s "$dest") sha256=$digest"
