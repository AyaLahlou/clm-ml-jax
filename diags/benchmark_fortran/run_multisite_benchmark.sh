#!/usr/bin/env bash
# run_multisite_benchmark.sh — Multisite Fortran benchmark (N=1..32).
#
# For N=1,2,4,8,16,32 "sites":
#   Sequential: run N instances of prgm.exe back-to-back.
#               Reports total wall time and per-site time.
#   Parallel:   run N instances simultaneously (bash background jobs).
#               Reports wall-clock time for all N to complete.
#
# Each run uses stop_n=6 (1 warmup + 5 timed steps).
# All N parallel runs use isolated output directories to avoid file conflicts.
#
# Usage:
#   cd clm-ml-fortran/benchmark
#   bash run_multisite_benchmark.sh
#
# Output:
#   benchmark/results/multisite_benchmark_fortran.csv
#   benchmark/results/multisite_timing_raw.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXE_DIR="${REPO_ROOT}/offline_executable"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

RAW_LOG="${RESULTS_DIR}/multisite_timing_raw.txt"
CSV_OUT="${RESULTS_DIR}/multisite_benchmark_fortran.csv"
> "${RAW_LOG}"

# ── Locate executable ─────────────────────────────────────────────────────────
if [ -f "${EXE_DIR}/prgm_opt.exe" ]; then
    EXE="${EXE_DIR}/prgm_opt.exe"
    echo "Using optimized build: prgm_opt.exe"
elif [ -f "${EXE_DIR}/prgm.exe" ]; then
    EXE="${EXE_DIR}/prgm.exe"
    echo "WARNING: using debug build prgm.exe (slower). Run build_optimized.sh first." >&2
else
    echo "ERROR: no executable found in ${EXE_DIR}." >&2
    exit 1
fi

# ── Locate input files ────────────────────────────────────────────────────────
CLM_FILE=$(ls "${REPO_ROOT}/input_files/clm5_0/CHATS7/CHATS_A15.clm2.h1."*.nc 2>/dev/null | head -1)
SOIL_FILE=$(ls "${REPO_ROOT}/input_files/clm5_0/CHATS7/CHATS_soil_moisture_correction_"*.nc 2>/dev/null | head -1)
if [ -z "${CLM_FILE}" ] || [ -z "${SOIL_FILE}" ]; then
    echo "ERROR: input files not found under ${REPO_ROOT}/input_files/clm5_0/CHATS7/" >&2
    exit 1
fi
REL_CLM=$(python3 -c "import os; print(os.path.relpath('${CLM_FILE}', '${EXE_DIR}'))")
REL_SOIL=$(python3 -c "import os; print(os.path.relpath('${SOIL_FILE}', '${EXE_DIR}'))")

# ── Config ────────────────────────────────────────────────────────────────────
N_SIZES=(1 2 4 8 16 32)
N_STEPS=6          # total timesteps per run (1 warmup + 5 physics)
N_TIMED=5          # timed steps (exclude warmup)

echo "Benchmark config: N_STEPS=${N_STEPS}, N_TIMED=${N_TIMED}" | tee -a "${RAW_LOG}"
echo "EXE: ${EXE}" | tee -a "${RAW_LOG}"
echo "" | tee -a "${RAW_LOG}"

# ── Helper: write namelist for one instance ────────────────────────────────────
# Args: $1=output_dir_relative_to_exe (trailing slash ok)  $2=stop_n
write_nl() {
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

# ── CSV header ────────────────────────────────────────────────────────────────
echo "backend,N,seq_total_s,seq_ss_ms_per_site,parallel_wall_s,parallel_ms_per_site" > "${CSV_OUT}"

# ── Main benchmark loop ───────────────────────────────────────────────────────
for N in "${N_SIZES[@]}"; do
    echo "====== N = ${N} ======" | tee -a "${RAW_LOG}"

    # ── Sequential: run N instances back-to-back ──────────────────────────────
    echo "  Sequential (N=${N} runs one after another) ..." | tee -a "${RAW_LOG}"
    t0_seq=$(date +%s%N)

    for idx in $(seq 1 "${N}"); do
        outdir="../output_files/bench_ms_seq_N${N}_i${idx}/"
        mkdir -p "${EXE_DIR}/../output_files/bench_ms_seq_N${N}_i${idx}"
        nl=$(write_nl "${outdir}" "${N_STEPS}")
        echo "${nl}" | "${EXE}" > /dev/null 2>&1
    done

    t1_seq=$(date +%s%N)
    seq_ns=$(( t1_seq - t0_seq ))
    seq_s=$(awk "BEGIN { printf \"%.6f\", ${seq_ns}/1e9 }")
    seq_ms_per_site=$(awk "BEGIN { printf \"%.4f\", ${seq_ns}/1e6/${N}/${N_TIMED} }")

    echo "    seq total: ${seq_s}s  |  ms/site/step: ${seq_ms_per_site}" | tee -a "${RAW_LOG}"

    # ── Parallel: run N instances simultaneously ───────────────────────────────
    echo "  Parallel (N=${N} runs simultaneously) ..." | tee -a "${RAW_LOG}"
    t0_par=$(date +%s%N)
    pids=()

    for idx in $(seq 1 "${N}"); do
        outdir="../output_files/bench_ms_par_N${N}_i${idx}/"
        mkdir -p "${EXE_DIR}/../output_files/bench_ms_par_N${N}_i${idx}"
        nl=$(write_nl "${outdir}" "${N_STEPS}")
        echo "${nl}" | "${EXE}" > /dev/null 2>&1 &
        pids+=($!)
    done

    # Wait for all background jobs
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done

    t1_par=$(date +%s%N)
    par_ns=$(( t1_par - t0_par ))
    par_s=$(awk "BEGIN { printf \"%.6f\", ${par_ns}/1e9 }")
    par_ms_per_site=$(awk "BEGIN { printf \"%.4f\", ${par_ns}/1e6/${N}/${N_TIMED} }")
    # Note: parallel ms/site = wall_clock / N / n_timed_steps  (N sites ran concurrently)

    echo "    par wall: ${par_s}s  |  ms/site/step: ${par_ms_per_site}" | tee -a "${RAW_LOG}"
    echo "" | tee -a "${RAW_LOG}"

    # Write CSV row
    echo "fortran,${N},${seq_s},${seq_ms_per_site},${par_s},${par_ms_per_site}" >> "${CSV_OUT}"
done

echo ""
echo "=== DONE ===" | tee -a "${RAW_LOG}"
echo "Raw log: ${RAW_LOG}"
echo "CSV:     ${CSV_OUT}"
echo ""
echo "Contents of ${CSV_OUT}:"
cat "${CSV_OUT}"
