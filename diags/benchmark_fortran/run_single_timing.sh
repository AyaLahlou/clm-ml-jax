#!/usr/bin/env bash
# run_single_timing.sh — Baseline single-site timing for CLM-ML Fortran.
#
# Runs CHATS7 three times (each = 6 timesteps: 1 warmup + 5 timed) and
# reports mean ms/timestep.  Also runs a 1-day (48-step) run as a sanity
# check for the total throughput.
#
# Usage:
#   cd clm-ml-fortran/benchmark
#   bash run_single_timing.sh
#
# Output:
#   benchmark/results/single_timing.txt   (human-readable log)
#   Prints ms/step summary to stdout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXE_DIR="${REPO_ROOT}/offline_executable"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

LOG="${RESULTS_DIR}/single_timing.txt"
> "${LOG}"

# ── Locate executable ─────────────────────────────────────────────────────────
if [ -f "${EXE_DIR}/prgm_opt.exe" ]; then
    EXE="${EXE_DIR}/prgm_opt.exe"
    echo "Using optimized build: prgm_opt.exe"
elif [ -f "${EXE_DIR}/prgm.exe" ]; then
    EXE="${EXE_DIR}/prgm.exe"
    echo "WARNING: prgm_opt.exe not found; using debug build prgm.exe (slower)." >&2
    echo "  Run build_optimized.sh first for accurate benchmarks." >&2
else
    echo "ERROR: no executable found in ${EXE_DIR}." >&2
    echo "  Run: cd ${EXE_DIR} && make" >&2
    exit 1
fi

# ── Locate CLM input file (glob for date-stamped filename) ────────────────────
CLM_FILE=$(ls "${REPO_ROOT}/input_files/clm5_0/CHATS7/CHATS_A15.clm2.h1."*.nc 2>/dev/null | head -1)
if [ -z "${CLM_FILE}" ]; then
    echo "ERROR: CLM input file not found under ${REPO_ROOT}/input_files/clm5_0/CHATS7/" >&2
    exit 1
fi

SOIL_FILE=$(ls "${REPO_ROOT}/input_files/clm5_0/CHATS7/CHATS_soil_moisture_correction_"*.nc 2>/dev/null | head -1)
if [ -z "${SOIL_FILE}" ]; then
    echo "ERROR: soil adjustment file not found under ${REPO_ROOT}/input_files/clm5_0/CHATS7/" >&2
    exit 1
fi

# Paths relative to EXE_DIR (where prgm.exe is run from)
REL_CLM=$(python3 -c "import os; print(os.path.relpath('${CLM_FILE}', '${EXE_DIR}'))")
REL_SOIL=$(python3 -c "import os; print(os.path.relpath('${SOIL_FILE}', '${EXE_DIR}'))")

echo "CLM file:  ${REL_CLM}"
echo "Soil file: ${REL_SOIL}"
echo ""

# ── Helper: write a short-run namelist ────────────────────────────────────────
# Arguments: $1=output_dir  $2=stop_n (number of steps)
write_namelist() {
    local outdir="$1"
    local stop_n="$2"
    cat <<EOF
&clmML_inparm
tower_name       = 'CHATS7'
start_ymd        = 20070501
start_tod        = 0
stop_option      = 'nsteps'
stop_n           = ${stop_n}
clm_start_ymd    = 20070401
clm_start_tod    = 0
fin_tower        = '../input_files/tower-forcing/CHATS7/2007-05.nc'
fin_clm          = '${REL_CLM}'
clm_phys         = 'CLM5_0'
fin_soil_adjust  = '${REL_SOIL}'
nlev_soil_adjust = 3
dirout           = '${outdir}'
met_type         = 3
dpai_min         = 0.01D0
pftcon_val       = 1
/
EOF
}

# ── Helper: time one run ───────────────────────────────────────────────────────
# Arguments: $1=run_label  $2=outdir  $3=stop_n
# Returns: wall-clock seconds via stdout of this function (captured by caller)
time_run() {
    local label="$1"
    local outdir="$2"
    local stop_n="$3"
    mkdir -p "${EXE_DIR}/${outdir}"

    local nl
    nl=$(write_namelist "${outdir}" "${stop_n}")

    echo "  Running ${label} (stop_n=${stop_n}) ..." | tee -a "${LOG}"
    local t0 t1 elapsed
    t0=$(date +%s%N)   # nanoseconds
    echo "${nl}" | "${EXE}" > /dev/null 2>&1
    t1=$(date +%s%N)
    elapsed=$(( (t1 - t0) ))   # nanoseconds
    local elapsed_s
    elapsed_s=$(awk "BEGIN { printf \"%.6f\", ${elapsed}/1e9 }")
    echo "    wall time: ${elapsed_s}s" | tee -a "${LOG}"
    echo "${elapsed_s}"
}

# ── Run 1: 1-day sanity check (48 steps) ─────────────────────────────────────
echo "=== 1-day sanity check (48 timesteps) ===" | tee -a "${LOG}"
elapsed_day=$(time_run "1day" "../output_files/bench_single_day" 48)
ms_per_step_day=$(awk "BEGIN { printf \"%.2f\", ${elapsed_day}*1000/48 }")
echo "  ms/step (48 steps total): ${ms_per_step_day}" | tee -a "${LOG}"
echo "" | tee -a "${LOG}"

# ── Runs 2-4: three repeated 6-step runs (1 warmup step + 5 timed) ────────────
# Strategy: run stop_n=6 three times.
# Total steps = 6, but step 1 includes initialization overhead.
# We measure total wall time / 6 (conservative) and also report time/5
# (excluding first step — use the average of 3 runs to estimate variance).
echo "=== Short-run timing (6 steps × 3 repetitions) ===" | tee -a "${LOG}"
echo "Note: step 1 has initialization overhead; ms/step reported two ways:" | tee -a "${LOG}"
echo "  (a) total / 6  — conservative (includes warmup)" | tee -a "${LOG}"
echo "  (b) total / 5  — aggressive (treat step 1 as warmup)" | tee -a "${LOG}"
echo "" | tee -a "${LOG}"

total_ns_arr=()
for rep in 1 2 3; do
    elapsed=$(time_run "rep${rep}" "../output_files/bench_single_rep${rep}" 6)
    total_ns_arr+=("${elapsed}")
done

# Compute mean across 3 reps
mean_s=$(python3 -c "
vals = [${total_ns_arr[0]}, ${total_ns_arr[1]}, ${total_ns_arr[2]}]
print(f'{sum(vals)/len(vals):.6f}')
")
ms_per_step_6=$(awk "BEGIN { printf \"%.2f\", ${mean_s}*1000/6 }")
ms_per_step_5=$(awk "BEGIN { printf \"%.2f\", ${mean_s}*1000/5 }")

echo "" | tee -a "${LOG}"
echo "=== Summary ===" | tee -a "${LOG}"
echo "  Mean wall time for 6-step run: ${mean_s}s" | tee -a "${LOG}"
echo "  ms/step (wall/6, includes warmup): ${ms_per_step_6}" | tee -a "${LOG}"
echo "  ms/step (wall/5, excludes warmup): ${ms_per_step_5}" | tee -a "${LOG}"
echo "  ms/step (1-day run / 48 steps):   ${ms_per_step_day}" | tee -a "${LOG}"
echo "" | tee -a "${LOG}"

# ── Write tidy summary file for parse_results.py ─────────────────────────────
cat > "${RESULTS_DIR}/single_timing_summary.txt" <<EOF
# Single-site timing summary
# Generated by run_single_timing.sh
exe=${EXE}
rep1_wall_s=${total_ns_arr[0]}
rep2_wall_s=${total_ns_arr[1]}
rep3_wall_s=${total_ns_arr[2]}
mean_6step_wall_s=${mean_s}
ms_per_step_incl_warmup=${ms_per_step_6}
ms_per_step_excl_warmup=${ms_per_step_5}
day_run_wall_s=${elapsed_day}
ms_per_step_day=${ms_per_step_day}
n_timed_steps=5
EOF

echo "Log written to: ${LOG}"
echo "Summary written to: ${RESULTS_DIR}/single_timing_summary.txt"
echo ""
echo "RESULT: Fortran CLM-ML single-site timing:"
echo "  ${ms_per_step_5} ms/step (5-step timed, after 1-step warmup)"
echo "  ${ms_per_step_day} ms/step (full 48-step day)"
