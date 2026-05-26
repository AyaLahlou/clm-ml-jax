"""
time_jax_run.py  —  Experiment 5: Wall-clock timing of CLM-ML-JAX for a
31-day simulation, comparing GPU vs CPU backends.

Usage (run from src/):
    python ../diags/time_jax_run.py [--namelist NL] [--backend gpu|cpu]

The script:
  1. Runs a short warm-up pass (1 day, same namelist with stop_n=1) to pay
     JIT compile cost up front and measure it separately.
  2. Runs the full 31-day simulation and records total wall-clock time.
  3. Reports: backend, n_timesteps, total_time_s, time_per_step_s,
     jit_warmup_time_s.

Output is written to diags/figures/timing_results.csv.

Mirrors the paper Experiment 5 timing methodology.
"""

import os
import sys
import csv
import time
import argparse
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent          # …/clm-ml-jax
_SRC_ROOT     = _PROJECT_ROOT / "src"
_DIAGS_FIGS   = _PROJECT_ROOT / "diags" / "figures"
_NAMELIST_31D = _SRC_ROOT / "offline_executable" / "nl.CHATS7.05.2007"
_NAMELIST_1D  = _SRC_ROOT / "offline_executable" / "nl.CHATS7.1day"


def _run_simulation(namelist: Path, backend: str, label: str) -> dict:
    """
    Run one full simulation and return timing info.

    Returns dict with keys: label, backend, returncode, wall_time_s,
    n_timesteps (parsed from stdout), time_per_step_s.
    """
    env = os.environ.copy()
    if backend == "cpu":
        env["JAX_PLATFORM_NAME"] = "cpu"
    elif backend == "gpu":
        env.pop("JAX_PLATFORM_NAME", None)   # let JAX pick GPU by default

    cmd = [sys.executable, "-m", "offline_executable.main", str(namelist)]

    print(f"\n{'='*70}", flush=True)
    print(f"  Running: {label}", flush=True)
    print(f"  Backend: {backend.upper()}", flush=True)
    print(f"  Namelist: {namelist.name}", flush=True)
    print(f"{'='*70}", flush=True)

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=str(_SRC_ROOT),
        env=env,
        capture_output=False,   # let stdout/stderr stream to terminal
        text=True,
    )
    wall_time = time.perf_counter() - t0

    # Parse number of timesteps from the namelist (stop_n * 48 for 30-min steps)
    import f90nml
    nml_data = f90nml.read(str(namelist))
    params   = nml_data.get("clmML_inparm", nml_data.get("clm_inparm", {}))
    stop_option = str(params.get("stop_option", "ndays")).strip()
    stop_n      = int(params.get("stop_n", 1))

    # CHATS7 uses 30-min timesteps → 48 per day
    if stop_option == "ndays":
        n_timesteps = stop_n * 48
    elif stop_option == "nsteps":
        n_timesteps = stop_n
    else:
        n_timesteps = stop_n

    time_per_step = wall_time / max(n_timesteps, 1)

    info = {
        "label":            label,
        "backend":          backend.upper(),
        "namelist":         namelist.name,
        "stop_n_days":      stop_n if stop_option == "ndays" else "n/a",
        "n_timesteps":      n_timesteps,
        "wall_time_s":      round(wall_time, 2),
        "time_per_step_s":  round(time_per_step, 4),
        "returncode":       result.returncode,
    }

    print(f"\n  --- {label} complete ---", flush=True)
    print(f"  Wall time   : {wall_time:.2f} s", flush=True)
    print(f"  Timesteps   : {n_timesteps}", flush=True)
    print(f"  Time/step   : {time_per_step:.4f} s", flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 5: time CLM-ML-JAX 31-day simulation"
    )
    parser.add_argument(
        "--namelist", default=str(_NAMELIST_31D),
        help="Path to 31-day namelist (default: nl.CHATS7.05.2007)"
    )
    parser.add_argument(
        "--warmup-namelist", default=str(_NAMELIST_1D),
        help="Path to warm-up namelist (default: nl.CHATS7.1day)"
    )
    parser.add_argument(
        "--backend", choices=["gpu", "cpu", "both"], default="gpu",
        help="JAX backend to benchmark (default: gpu)"
    )
    parser.add_argument(
        "--skip-warmup", action="store_true",
        help="Skip the warm-up (JIT compile) pass"
    )
    args = parser.parse_args()

    namelist_31d  = Path(args.namelist)
    namelist_1d   = Path(args.warmup_namelist)

    backends = ["gpu", "cpu"] if args.backend == "both" else [args.backend]

    _DIAGS_FIGS.mkdir(parents=True, exist_ok=True)
    results = []

    for backend in backends:
        # ------------------------------------------------------------------
        # Step 1: Warm-up run (pay JIT compile cost)
        # ------------------------------------------------------------------
        if not args.skip_warmup:
            warmup_info = _run_simulation(
                namelist_1d, backend,
                label=f"Warm-up / JIT compile (1-day, {backend.upper()})"
            )
            warmup_info["role"] = "warmup"
            results.append(warmup_info)
            jit_time = warmup_info["wall_time_s"]
            print(f"\n  JIT compile + 1-day run: {jit_time:.2f} s", flush=True)
        else:
            jit_time = None

        # ------------------------------------------------------------------
        # Step 2: Full 31-day timing run
        # ------------------------------------------------------------------
        full_info = _run_simulation(
            namelist_31d, backend,
            label=f"Full 31-day run ({backend.upper()})"
        )
        full_info["role"] = "timing"
        full_info["jit_warmup_time_s"] = jit_time if jit_time is not None else "n/a"
        results.append(full_info)

    # ------------------------------------------------------------------
    # Write CSV summary
    # ------------------------------------------------------------------
    outcsv = _DIAGS_FIGS / "timing_results.csv"
    fieldnames = [
        "role", "label", "backend", "namelist",
        "stop_n_days", "n_timesteps",
        "wall_time_s", "time_per_step_s",
        "jit_warmup_time_s", "returncode",
    ]
    with open(outcsv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*70}", flush=True)
    print("  TIMING SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    for r in results:
        if r.get("role") == "timing":
            print(
                f"  {r['backend']:4s}  |  "
                f"wall={r['wall_time_s']:.2f}s  |  "
                f"{r['n_timesteps']} steps  |  "
                f"{r['time_per_step_s']:.4f} s/step  |  "
                f"JIT warmup={r.get('jit_warmup_time_s', 'n/a')}s"
            )
    print(f"\n  Results written to: {outcsv}", flush=True)


if __name__ == "__main__":
    main()
