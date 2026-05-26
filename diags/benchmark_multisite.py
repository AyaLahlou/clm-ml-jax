"""
benchmark_multisite.py — Multi-site vmap GPU speedup benchmark.

Measures wall-clock time for running N identical sites simultaneously
using jax.vmap vs N sequential calls, on GPU and CPU.

Strategy (thin-wrapper, zero physics changes):
  - MLCanopyFluxes in _diff_mode (grid=GridInfo) is already pure-functional.
  - vmap operates on mlcanopy_inst from outside — physics is unchanged.
  - All other CLM state (atm2lnd_inst, canopystate_inst, etc.) is captured
    in the closure (same for all N sites = N replicas of CHATS7).
  - batched_ml = tree_map(stack, [mlcanopy_inst]*N)
  - output = jit(vmap(step))(batched_ml)

Metrics reported per N:
  - vmap_first_s   : time of first batched call (includes JIT recompile for new N)
  - vmap_steady_s  : time of subsequent batched call (pure GPU throughput)
  - seq_total_s    : time for N sequential jit(step) calls
  - speedup        : seq_total_s / vmap_steady_s
  - per_site_gain  : (seq_total_s/N) / (vmap_steady_s/N)  [should be ~1 if same]

Usage (from project root):
    python diags/benchmark_multisite.py [--n-sites 1,2,4,8,16,32] [--repeats 3]

Output: diags/figures/multisite_benchmark.csv + console table.
"""
from __future__ import annotations

import os
import sys
import csv
import time
import argparse
from pathlib import Path

# ── Project paths ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR      = _PROJECT_ROOT / "src"
_FIGURES_DIR  = _PROJECT_ROOT / "diags" / "figures"
_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--n-sites", default="1,2,4,8,16,32",
    help="Comma-separated list of N values to benchmark (default: 1,2,4,8,16,32)",
)
parser.add_argument(
    "--repeats", type=int, default=3,
    help="Number of timed repeats after first call (default: 3)",
)
parser.add_argument(
    "--backend", choices=["gpu", "cpu", "both"], default="gpu",
    help="JAX backend to benchmark (default: gpu)",
)
parser.add_argument(
    "--full-physics", action="store_true",
    help="Use full RK4 physics (runge_kutta_type=41, dtime_ml=300s, 6 sub-steps x 4 RK stages). "
         "Default is Euler (1 sub-step) as used by expt_init.",
)
args = parser.parse_args()
N_LIST    = [int(x) for x in args.n_sites.split(",")]
N_REPEATS = args.repeats

# ── JAX setup ─────────────────────────────────────────────────────────────────
os.environ["CLM_ML_NO_CHECKPOINT"] = "1"   # faster compilation, no grad needed

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

print(f"JAX devices: {jax.devices()}", flush=True)
print(f"JAX backend: {jax.default_backend()}", flush=True)

# ── Initialize single site (CHATS7) via shared expt_init ─────────────────────
# This runs a full warmup timestep and sets up all CLM singletons.
print("\n=== Initializing CHATS7 site (warmup ~5-10 min) ===", flush=True)
from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs, jax as _jax_check,
)
from multilayer_canopy.MLCanopyFluxesMod import MLCanopyFluxes

print(f"Site ready: p={grid.p}, ncan={grid.ncan}", flush=True)

# ── Physics settings ──────────────────────────────────────────────────────────
# expt_init sets Euler (runge_kutta_type=10) after warmup for fast experiments.
# --full-physics restores the production RK4 settings (41, dtime_ml=300s).
from multilayer_canopy import MLclm_varctl as _varctl
if args.full_physics:
    _varctl.runge_kutta_type = 41
    _varctl.dtime_ml         = 300.0
    _varctl.nrk              = 4
    print("Physics: FULL RK4  (runge_kutta_type=41, dtime_ml=300s, "
          "6 sub-steps x 4 RK stages)", flush=True)
else:
    print("Physics: EULER     (runge_kutta_type=10, 1 sub-step, 0 RK stages)",
          flush=True)

# ── Single-site step (pure functional, diff_mode via grid=GridInfo) ────────────
# Captures all CLM state in closure. Only mlcanopy_inst is the variable input.
def _single_site_step(ml_inst):
    """One CLM timestep for one site. Pure functional (no global mutations)."""
    return MLCanopyFluxes(mlcanopy_inst=ml_inst, **_mlcf_kwargs)

# JIT compile the single-site step (pay compile cost here, before timing)
print("\n=== Compiling single-site JIT step ===", flush=True)
t0 = time.perf_counter()
_single_step_jit = jax.jit(_single_site_step)
_ = _single_step_jit(mlcanopy_inst)
jax.effects_barrier()
print(f"  JIT compile done in {time.perf_counter()-t0:.1f}s", flush=True)


# ── Stacking helper ───────────────────────────────────────────────────────────
def _stack_n(inst, N: int):
    """Stack N copies of inst along a new leading batch axis (axis 0)."""
    return jax.tree_util.tree_map(
        lambda x: jnp.stack([x] * N, axis=0),
        inst,
    )


# ── Timing helpers ────────────────────────────────────────────────────────────
def _time_call(fn, *args, repeats: int = 1):
    """Time fn(*args) after one warm-up, return (first_s, mean_repeat_s)."""
    # First call — includes any JIT recompile for new shapes
    t0 = time.perf_counter()
    out = fn(*args)
    jax.effects_barrier()
    first_s = time.perf_counter() - t0

    # Timed repeats
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.effects_barrier()
        times.append(time.perf_counter() - t0)

    return first_s, float(np.mean(times)), out


def _time_sequential(N: int, repeats: int = 1):
    """Run N sequential single-site steps; return (first_s, mean_repeat_s)."""
    def _run_n_seq():
        inst = mlcanopy_inst
        for _ in range(N):
            inst = _single_step_jit(inst)
        jax.effects_barrier()
        return inst

    # Warm-up (N=1 already compiled)
    t0 = time.perf_counter()
    _run_n_seq()
    first_s = time.perf_counter() - t0

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _run_n_seq()
        times.append(time.perf_counter() - t0)

    return first_s, float(np.mean(times))


# ── Main benchmark loop ───────────────────────────────────────────────────────
def run_benchmark(backend_label: str) -> list[dict]:
    results = []

    print(f"\n{'='*70}", flush=True)
    print(f"  Backend: {backend_label.upper()}", flush=True)
    print(f"  Repeats per measurement: {N_REPEATS}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'N':>6}  {'vmap_1st_s':>12}  {'vmap_ss_s':>12}  "
          f"{'seq_ss_s':>12}  {'speedup':>9}  {'ms/site/step'}", flush=True)
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*9}  {'-'*14}", flush=True)

    for N in N_LIST:
        # Build batched input (N copies of same site)
        batched_ml = _stack_n(mlcanopy_inst, N)

        # Build vmapped step for this N (JIT wraps the vmap)
        batched_step = jax.jit(jax.vmap(_single_site_step))

        # Time batched
        vmap_first_s, vmap_ss_s, _ = _time_call(
            batched_step, batched_ml, repeats=N_REPEATS
        )

        # Time sequential
        seq_first_s, seq_ss_s = _time_sequential(N, repeats=N_REPEATS)

        speedup       = seq_ss_s / vmap_ss_s if vmap_ss_s > 0 else float("nan")
        ms_site_step  = vmap_ss_s / N * 1000

        print(
            f"  {N:>6}  {vmap_first_s:>12.3f}  {vmap_ss_s:>12.3f}  "
            f"{seq_ss_s:>12.3f}  {speedup:>9.2f}x  {ms_site_step:>10.1f} ms",
            flush=True,
        )

        results.append({
            "backend":      backend_label,
            "N":            N,
            "vmap_first_s": round(vmap_first_s, 4),
            "vmap_ss_s":    round(vmap_ss_s, 4),
            "seq_ss_s":     round(seq_ss_s, 4),
            "speedup":      round(speedup, 4),
            "ms_per_site":  round(ms_site_step, 4),
        })

    return results


# ── Run benchmarks ────────────────────────────────────────────────────────────
all_results = []

if args.backend in ("gpu", "both"):
    if jax.default_backend() != "gpu":
        print("WARNING: no GPU detected, skipping GPU benchmark", flush=True)
    else:
        all_results += run_benchmark("gpu")

if args.backend in ("cpu", "both"):
    # Re-run on CPU by moving arrays and setting default device
    cpu_dev = jax.devices("cpu")[0]
    print(f"\n=== Switching to CPU backend ({cpu_dev}) ===", flush=True)

    # Move mlcanopy_inst to CPU
    import functools
    _to_cpu = functools.partial(jax.device_put, device=cpu_dev)
    mlcanopy_inst_cpu = jax.tree_util.tree_map(_to_cpu, mlcanopy_inst)

    # Redefine _single_site_step for CPU
    def _single_site_step_cpu(ml_inst):
        return MLCanopyFluxes(mlcanopy_inst=ml_inst, **_mlcf_kwargs)

    # Redefine globals for _time_sequential and run_benchmark to use cpu inst
    _orig_ml = mlcanopy_inst
    # Monkeypatch: run_benchmark reads module-level mlcanopy_inst and _single_step_jit
    # Instead, just repeat the loop manually for CPU
    # (simpler than patching module state)

    jit_cpu = jax.jit(jax.device_put_replicated if False else _single_site_step_cpu)
    t0 = time.perf_counter()
    _ = jax.jit(_single_site_step_cpu)(mlcanopy_inst_cpu)
    jax.effects_barrier()
    print(f"  CPU JIT compile done in {time.perf_counter()-t0:.1f}s", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"  Backend: CPU", flush=True)
    print(f"  Repeats: {N_REPEATS}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'N':>6}  {'vmap_1st_s':>12}  {'vmap_ss_s':>12}  "
          f"{'seq_ss_s':>12}  {'speedup':>9}  {'ms/site/step'}", flush=True)
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*9}  {'-'*14}",
          flush=True)

    _jit_seq_cpu = jax.jit(_single_site_step_cpu)

    for N in N_LIST:
        batched_ml_cpu = jax.tree_util.tree_map(
            lambda x: jnp.stack([jax.device_put(x, cpu_dev)] * N, axis=0),
            _orig_ml,
        )
        batched_step_cpu = jax.jit(jax.vmap(_single_site_step_cpu))

        # vmap
        t0 = time.perf_counter()
        _ = batched_step_cpu(batched_ml_cpu); jax.effects_barrier()
        vmap_first_s = time.perf_counter() - t0
        vmap_times = []
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            batched_step_cpu(batched_ml_cpu); jax.effects_barrier()
            vmap_times.append(time.perf_counter() - t0)
        vmap_ss_s = float(np.mean(vmap_times))

        # sequential
        inst_cpu = jax.device_put(mlcanopy_inst, cpu_dev)
        t0 = time.perf_counter()
        for _ in range(N): inst_cpu = _jit_seq_cpu(inst_cpu)
        jax.effects_barrier()
        seq_first_s = time.perf_counter() - t0
        seq_times = []
        for _ in range(N_REPEATS):
            inst_cpu = jax.device_put(mlcanopy_inst, cpu_dev)
            t0 = time.perf_counter()
            for _ in range(N): inst_cpu = _jit_seq_cpu(inst_cpu)
            jax.effects_barrier()
            seq_times.append(time.perf_counter() - t0)
        seq_ss_s = float(np.mean(seq_times))

        speedup = seq_ss_s / vmap_ss_s if vmap_ss_s > 0 else float("nan")
        ms_site_step = vmap_ss_s / N * 1000

        print(
            f"  {N:>6}  {vmap_first_s:>12.3f}  {vmap_ss_s:>12.3f}  "
            f"{seq_ss_s:>12.3f}  {speedup:>9.2f}x  {ms_site_step:>10.1f} ms",
            flush=True,
        )
        all_results.append({
            "backend":      "cpu",
            "N":            N,
            "vmap_first_s": round(vmap_first_s, 4),
            "vmap_ss_s":    round(vmap_ss_s, 4),
            "seq_ss_s":     round(seq_ss_s, 4),
            "speedup":      round(speedup, 4),
            "ms_per_site":  round(ms_site_step, 4),
        })

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = _FIGURES_DIR / "multisite_benchmark.csv"
fieldnames = ["backend", "N", "vmap_first_s", "vmap_ss_s", "seq_ss_s",
              "speedup", "ms_per_site"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_results)

print(f"\nResults saved: {csv_path}", flush=True)
print("\n=== benchmark_multisite.py complete ===", flush=True)
