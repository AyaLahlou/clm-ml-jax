"""
benchmark_precision.py — float32 vs float64 GPU throughput comparison.

Runs jax.jit(jax.vmap(forward_multi)) on GPU for each N in N_VALUES,
comparing float64 (JAX default for CLM-ML-JAX) against float32.

With jax_enable_x64=False, JAX silently maps all float64 dtypes to float32,
so the full physics model runs in reduced precision without code changes.
GPU hardware provides ~2× FP32 throughput over FP64 on Ampere/Turing GPUs.

This benchmark quantifies:
  1. Compile time (first JIT call per N)
  2. Steady-state throughput: mean of REPEATS calls, ms/sample
  3. Speedup: f64_ms_per_sample / f32_ms_per_sample

Usage:
    CLM_ML_X64=1 python diags/benchmark_precision.py --precision f64
    CLM_ML_X64=0 python diags/benchmark_precision.py --precision f32

    # Or via SLURM array job (run_precision_benchmark.sh, tasks 0=f64, 1=f32)

Output:
    diags/figures/precision_benchmark_f64.csv
    diags/figures/precision_benchmark_f32.csv

Columns: N, precision, compile_s, run_ms, run_ms_std, ms_per_sample
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

# ── Precision flag must be set BEFORE any JAX import (happens in expt_init) ──
_PREC = "f64"
for _i, _arg in enumerate(sys.argv[1:], 1):
    if _arg == "--precision" and _i < len(sys.argv):
        _PREC = sys.argv[_i + 1]
        break

import os
os.environ["CLM_ML_X64"] = "0" if _PREC == "f32" else "1"
os.environ["CLM_ML_NO_CHECKPOINT"] = "1"

# ── Use a separate compile cache to avoid cross-precision cache hits ──────────
_BENCH_CACHE = f"/burg-archive/home/al4385/.cache/jax_compile_cache_precision_{_PREC}"
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

N_VALUES = [1, 8, 32, 128, 512, 1024, 2048]
REPEATS  = 5

_float_dtype = jnp.float32 if _PREC == "f32" else jnp.float64

print(f"\n{'='*70}", flush=True)
print(f"Precision benchmark: {_PREC}  (jax_enable_x64={jax.config.jax_enable_x64})", flush=True)
print(f"JAX cache dir: {_BENCH_CACHE}", flush=True)
print(f"GPU device: {jax.devices()[0]}", flush=True)

_orig_pftcon     = _pftcon_mod.MLpftcon
_p               = grid.p
_n               = grid.ncan
_canopystate_inst = _mlcf_kwargs["canopystate_inst"]
_mlcf_kwargs_base = {k: v for k, v in _mlcf_kwargs.items()
                     if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst",
                                  "canopystate_inst")}


def forward_multi(scales):
    """vmap target: scales shape (5,), returns [GPP, H, LE]."""
    vc = scales[0] * jnp.asarray(_orig_pftcon.vcmaxpft, dtype=_float_dtype)
    mc = _canopystate_inst._replace(
        elai_patch=scales[4] * jnp.asarray(_canopystate_inst.elai_patch, dtype=_float_dtype),
        esai_patch=scales[4] * jnp.asarray(_canopystate_inst.esai_patch, dtype=_float_dtype))
    ma = atm2lnd_inst._replace(
        forc_t_downscaled_col     = scales[1] * jnp.asarray(atm2lnd_inst.forc_t_downscaled_col, dtype=_float_dtype),
        forc_solad_downscaled_col = scales[2] * jnp.asarray(atm2lnd_inst.forc_solad_downscaled_col, dtype=_float_dtype),
        forc_solai_grc            = scales[2] * jnp.asarray(atm2lnd_inst.forc_solai_grc, dtype=_float_dtype))
    mw = wateratm2lndbulk_inst._replace(
        forc_q_downscaled_col=scales[3] * jnp.asarray(wateratm2lndbulk_inst.forc_q_downscaled_col, dtype=_float_dtype))
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst, atm2lnd_inst=ma,
        wateratm2lndbulk_inst=mw, canopystate_inst=mc,
        vcmaxpft_jax=vc, **_mlcf_kwargs_base)
    return jnp.array([compute_gpp(inst, _p, _n),
                      compute_le(inst, _p, _n),
                      compute_h(inst, _p, _n)], dtype=_float_dtype)


# ── Single-sample warmup ──────────────────────────────────────────────────────
print("\nCompiling single-sample JIT warmup...", flush=True)
_fwd_jit = jax.jit(forward_multi)
_ones    = jnp.ones(5, dtype=_float_dtype)
t0 = time.perf_counter()
_ = _fwd_jit(_ones)
jax.effects_barrier()
warmup_s = time.perf_counter() - t0
print(f"  single JIT compile: {warmup_s:.1f}s", flush=True)

t0 = time.perf_counter()
_ = _fwd_jit(_ones)
jax.effects_barrier()
print(f"  single forward (post-JIT): {(time.perf_counter()-t0)*1000:.2f}ms", flush=True)

# ── Parameter samples ─────────────────────────────────────────────────────────
rng = np.random.default_rng(seed=42)
all_thetas = rng.uniform(0.8, 1.2, size=(max(N_VALUES), 5)).astype(
    np.float32 if _PREC == "f32" else np.float64)

# ── Main benchmark loop ───────────────────────────────────────────────────────
rows = []

for N in N_VALUES:
    print(f"\n{'='*60}", flush=True)
    print(f"N = {N}", flush=True)

    thetas_N   = jnp.array(all_thetas[:N])
    batched_fn = jax.jit(jax.vmap(forward_multi))

    # Compile + first call
    t0 = time.perf_counter()
    out = batched_fn(thetas_N)
    jax.effects_barrier()
    compile_s = time.perf_counter() - t0
    print(f"  compile + first call: {compile_s:.1f}s", flush=True)

    # Throughput: REPEATS subsequent calls
    times = []
    for i in range(REPEATS):
        t0 = time.perf_counter()
        _ = batched_fn(thetas_N)
        jax.effects_barrier()
        times.append(time.perf_counter() - t0)
        print(f"  repeat {i+1}/{REPEATS}: {times[-1]*1000:.2f}ms", flush=True)

    run_ms        = float(np.mean(times)) * 1000.0
    run_ms_std    = float(np.std(times))  * 1000.0
    ms_per_sample = run_ms / N

    print(f"  throughput: {run_ms:.2f}ms ± {run_ms_std:.2f}ms  ({ms_per_sample:.3f}ms/sample)", flush=True)

    rows.append({
        "N":             N,
        "precision":     _PREC,
        "compile_s":     f"{compile_s:.2f}",
        "run_ms":        f"{run_ms:.3f}",
        "run_ms_std":    f"{run_ms_std:.3f}",
        "ms_per_sample": f"{ms_per_sample:.4f}",
    })

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
print(f"=== Precision benchmark summary ({_PREC}) ===", flush=True)
print(f"  {'N':>6}  {'compile_s':>10}  {'run_ms':>10}  {'ms/sample':>12}", flush=True)
print("  " + "-" * 44, flush=True)
for r in rows:
    print(f"  {r['N']:>6}  {r['compile_s']:>10}  {r['run_ms']:>10}  {r['ms_per_sample']:>12}", flush=True)

# ── Write CSV ─────────────────────────────────────────────────────────────────
csv_path = FIGURES_DIR / f"precision_benchmark_{_PREC}.csv"
fieldnames = ["N", "precision", "compile_s", "run_ms", "run_ms_std", "ms_per_sample"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"\nCSV saved: {csv_path}", flush=True)
print(f"\n=== benchmark_precision.py ({_PREC}) complete ===", flush=True)
