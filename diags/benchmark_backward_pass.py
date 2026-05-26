"""
benchmark_backward_pass.py — reverse-mode AD memory limits: float32 vs float64.

The forward-only benchmark (benchmark_precision.py) showed f32 ≈ f64 throughput
because CLM physics has no matrix multiplications — GPU tensor cores are idle.

The practical float32 advantage is MEMORY CAPACITY.  The reverse-mode backward
pass must store all intermediate activations for the entire batch in GPU HBM:

    activation memory ∝ N  ×  (activations per sample)  ×  (bytes per float)

float32 (4 B) fits 2× more samples than float64 (8 B).  The forward pass never
reveals this because it processes one batch at a time and discards activations
immediately; the backward pass must hold them all while traversing the compute
graph in reverse.

Gradient target:
    batch_loss(θ) = Σ_{i=1}^{N} Σ_k  output_k(θ_i)          [scalar]
    grads = jax.grad(batch_loss)(thetas_N)                     [N × 5]

This mirrors calibration workflows: differentiate a scalar loss through the full
physics stack for all N ensemble members simultaneously.

Key settings:
    XLA_PYTHON_CLIENT_PREALLOCATE=false   — JAX allocates memory on demand
                                            rather than pre-reserving 75% of HBM,
                                            so OOM reflects actual workload limits.

N sweep: [1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096]
Stops at first OOM; remaining N values recorded as 'skip'.

For each N:
  1. Fresh jax.jit(jax.grad(batch_loss)) — fresh JIT per N measures true compile time.
  2. Compile + first call → grad_compile_s
  3. REPEATS=3 subsequent calls → run_ms, ms_per_sample
  4. GPU memory before/after via nvidia-smi (MB)

Usage:
    python diags/benchmark_backward_pass.py --precision f64
    python diags/benchmark_backward_pass.py --precision f32

    # Or via SLURM array job (run_backward_pass_benchmark.sh, task 0=f64, 1=f32)

Output:
    diags/figures/backward_pass_f64.csv
    diags/figures/backward_pass_f32.csv

Columns: N, precision, grad_compile_s, run_ms, run_ms_std, ms_per_sample,
         mem_before_mb, mem_after_mb, status
"""
from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

# ── Precision must be set BEFORE any JAX import (happens inside expt_init) ───
_PREC = "f64"
for _i, _arg in enumerate(sys.argv[1:], 1):
    if _arg == "--precision" and _i < len(sys.argv):
        _PREC = sys.argv[_i + 1]
        break

import os
os.environ["CLM_ML_X64"]           = "0" if _PREC == "f32" else "1"
os.environ["CLM_ML_NO_CHECKPOINT"] = "1"
# Allocate GPU memory on demand rather than pre-reserving 75% of HBM.
# Without this, JAX grabs ~36 GB on a 48 GB A40 at startup, and all runs
# appear to OOM at the same low N because the working set is compared against
# the artificially-reduced headroom.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

_BENCH_CACHE = (
    f"/burg-archive/home/al4385/.cache/jax_compile_cache_backward_{_PREC}"
)
os.environ["JAX_COMPILATION_CACHE_DIR"] = _BENCH_CACHE
os.makedirs(_BENCH_CACHE, exist_ok=True)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import multilayer_canopy.MLpftconMod as _pftcon_mod

from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs,
    jax, jnp, MLCanopyFluxes,
    atm2lnd_inst, wateratm2lndbulk_inst, compute_gpp, compute_le, compute_h,
)

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_VALUES     = [1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096]
REPEATS      = 3
_float_dtype = jnp.float32 if _PREC == "f32" else jnp.float64

print(f"\n{'='*70}", flush=True)
print(f"Backward-pass memory benchmark: {_PREC}  "
      f"(jax_enable_x64={jax.config.jax_enable_x64})", flush=True)
print(f"GPU device: {jax.devices()[0]}", flush=True)
print(f"JAX cache dir: {_BENCH_CACHE}", flush=True)

# ── Model state (mirrors benchmark_precision.py) ──────────────────────────────
_orig_pftcon      = _pftcon_mod.MLpftcon
_p                = grid.p
_n                = grid.ncan
_canopystate_inst = _mlcf_kwargs["canopystate_inst"]
_mlcf_kwargs_base = {k: v for k, v in _mlcf_kwargs.items()
                     if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst",
                                  "canopystate_inst")}


def forward_multi(scales):
    """vmap target: scales (5,) → [GPP, LE, H]."""
    vc = scales[0] * jnp.asarray(_orig_pftcon.vcmaxpft, dtype=_float_dtype)
    mc = _canopystate_inst._replace(
        elai_patch=scales[4] * jnp.asarray(
            _canopystate_inst.elai_patch, dtype=_float_dtype),
        esai_patch=scales[4] * jnp.asarray(
            _canopystate_inst.esai_patch, dtype=_float_dtype))
    ma = atm2lnd_inst._replace(
        forc_t_downscaled_col=scales[1] * jnp.asarray(
            atm2lnd_inst.forc_t_downscaled_col, dtype=_float_dtype),
        forc_solad_downscaled_col=scales[2] * jnp.asarray(
            atm2lnd_inst.forc_solad_downscaled_col, dtype=_float_dtype),
        forc_solai_grc=scales[2] * jnp.asarray(
            atm2lnd_inst.forc_solai_grc, dtype=_float_dtype))
    mw = wateratm2lndbulk_inst._replace(
        forc_q_downscaled_col=scales[3] * jnp.asarray(
            wateratm2lndbulk_inst.forc_q_downscaled_col, dtype=_float_dtype))
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst, atm2lnd_inst=ma,
        wateratm2lndbulk_inst=mw, canopystate_inst=mc,
        vcmaxpft_jax=vc, **_mlcf_kwargs_base)
    return jnp.array([compute_gpp(inst, _p, _n),
                      compute_le(inst, _p, _n),
                      compute_h(inst, _p, _n)], dtype=_float_dtype)


def batch_loss(thetas):
    """Scalar aggregate loss over N ensemble members. Shape: [N,5] → scalar."""
    return jnp.sum(jax.vmap(forward_multi)(thetas))


# ── Memory helpers ────────────────────────────────────────────────────────────

def _gpu_mem_mb() -> float:
    """GPU memory in use (MB) via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5)
        return float(out.strip().split("\n")[0])
    except Exception:
        return float("nan")


def _is_oom(exc: Exception) -> bool:
    msg = str(exc).upper()
    return any(k in msg for k in ("RESOURCE_EXHAUSTED", "OUT OF MEMORY", "OOM",
                                   "CANNOT ALLOCATE"))


# ── Single-sample warmup ──────────────────────────────────────────────────────
# Triggers any lazy model initialisation so it doesn't inflate N=1 compile time.
print("\nWarmup: compiling single-sample grad...", flush=True)
_warmup_fn = jax.jit(jax.grad(batch_loss))
_ones_1x5  = jnp.ones((1, 5), dtype=_float_dtype)
t0 = time.perf_counter()
_ = _warmup_fn(_ones_1x5)
jax.effects_barrier()
print(f"  warmup compile: {time.perf_counter()-t0:.1f}s  "
      f"GPU mem: {_gpu_mem_mb():.0f} MB", flush=True)
del _warmup_fn  # discard so N=1 in the loop gets a fresh JIT

# ── Parameter samples ─────────────────────────────────────────────────────────
rng        = np.random.default_rng(seed=42)
all_thetas = rng.uniform(0.8, 1.2, size=(max(N_VALUES), 5)).astype(
    np.float32 if _PREC == "f32" else np.float64)

# ── Main sweep ────────────────────────────────────────────────────────────────
rows    = []
hit_oom = False

for N in N_VALUES:
    print(f"\n{'='*60}", flush=True)
    print(f"N = {N}", flush=True)

    if hit_oom:
        rows.append({"N": N, "precision": _PREC, "grad_compile_s": "nan",
                     "run_ms": "nan", "run_ms_std": "nan",
                     "ms_per_sample": "nan",
                     "mem_before_mb": "nan", "mem_after_mb": "nan",
                     "status": "skip"})
        print("  skipped (prior OOM)", flush=True)
        continue

    thetas_N  = jnp.array(all_thetas[:N])     # [N, 5]
    # Fresh JIT per N — ensures clean compilation timing for every N,
    # just as benchmark_precision.py does for the forward pass.
    grad_fn_N = jax.jit(jax.grad(batch_loss))

    # ── Compile + first call ──────────────────────────────────────────────────
    mem_before = _gpu_mem_mb()
    t0 = time.perf_counter()
    try:
        grads = grad_fn_N(thetas_N)
        jax.effects_barrier()
        compile_s  = time.perf_counter() - t0
        mem_after  = _gpu_mem_mb()
        print(f"  grad compile+first call: {compile_s:.1f}s", flush=True)
        print(f"  GPU mem: {mem_before:.0f} → {mem_after:.0f} MB", flush=True)
        print(f"  grads shape: {grads.shape}", flush=True)
    except Exception as exc:
        if _is_oom(exc):
            hit_oom = True
            print(f"  OOM during compile/first call", flush=True)
            rows.append({"N": N, "precision": _PREC, "grad_compile_s": "nan",
                         "run_ms": "nan", "run_ms_std": "nan",
                         "ms_per_sample": "nan",
                         "mem_before_mb": f"{mem_before:.0f}",
                         "mem_after_mb": "nan", "status": "oom"})
            continue
        raise

    # ── Steady-state timing ───────────────────────────────────────────────────
    times         = []
    oom_in_repeat = False
    for i in range(REPEATS):
        t0 = time.perf_counter()
        try:
            _ = grad_fn_N(thetas_N)
            jax.effects_barrier()
            times.append(time.perf_counter() - t0)
            print(f"  repeat {i+1}/{REPEATS}: {times[-1]*1000:.1f}ms", flush=True)
        except Exception as exc:
            if _is_oom(exc):
                hit_oom = True
                oom_in_repeat = True
                print(f"  OOM at repeat {i+1}", flush=True)
                break
            raise

    if oom_in_repeat:
        rows.append({"N": N, "precision": _PREC,
                     "grad_compile_s": f"{compile_s:.2f}",
                     "run_ms": "nan", "run_ms_std": "nan",
                     "ms_per_sample": "nan",
                     "mem_before_mb": f"{mem_before:.0f}",
                     "mem_after_mb": f"{mem_after:.0f}",
                     "status": "oom_repeat"})
        continue

    run_ms        = float(np.mean(times)) * 1000.0
    run_ms_std    = float(np.std(times))  * 1000.0
    ms_per_sample = run_ms / N
    print(f"  throughput: {run_ms:.1f}ms ± {run_ms_std:.1f}ms  "
          f"({ms_per_sample:.2f}ms/sample)", flush=True)

    rows.append({
        "N":              N,
        "precision":      _PREC,
        "grad_compile_s": f"{compile_s:.2f}",
        "run_ms":         f"{run_ms:.2f}",
        "run_ms_std":     f"{run_ms_std:.2f}",
        "ms_per_sample":  f"{ms_per_sample:.3f}",
        "mem_before_mb":  f"{mem_before:.0f}",
        "mem_after_mb":   f"{mem_after:.0f}",
        "status":         "ok",
    })

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
print(f"=== Backward-pass memory benchmark summary ({_PREC}) ===", flush=True)
print(f"  {'N':>6}  {'compile_s':>10}  {'run_ms':>10}  {'ms/sample':>10}  "
      f"{'mem_after_mb':>14}  status", flush=True)
print("  " + "-" * 70, flush=True)
for r in rows:
    print(f"  {r['N']:>6}  {r['grad_compile_s']:>10}  {r['run_ms']:>10}  "
          f"{r['ms_per_sample']:>10}  {r['mem_after_mb']:>14}  {r['status']}",
          flush=True)

# ── Write CSV ─────────────────────────────────────────────────────────────────
csv_path   = FIGURES_DIR / f"backward_pass_{_PREC}.csv"
fieldnames = ["N", "precision", "grad_compile_s", "run_ms", "run_ms_std",
              "ms_per_sample", "mem_before_mb", "mem_after_mb", "status"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"\nCSV saved: {csv_path}", flush=True)
print(f"\n=== benchmark_backward_pass.py ({_PREC}) complete ===", flush=True)
