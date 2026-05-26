"""
precision_roofline.py — f32 vs f64 roofline benchmark: matmul vs CLM-ML physics.

Two workloads side-by-side to explain why CLM-ML shows no f32 speedup:

  Workload A — Single large square GEMM:
      FMA / CUDA-core bound.  The Quadro RTX 8000 has ~32× more FP32 CUDA-core
      throughput than FP64; expect f32 to be ~16–32× faster at large M.

  Workload B — CLM-ML physics forward pass (vmapped):
      Dominated by transcendental functions (exp, log, sqrt in photosynthesis,
      stomatal conductance, radiation) that use the SFU rather than CUDA cores.
      SFU throughput is similar for f32 and f64; expect f32 ≈ f64.

Key timing fix: use jax.block_until_ready(out) not jax.effects_barrier().
effects_barrier() only waits for Python callbacks (io_callback), NOT GPU compute.
block_until_ready() waits for the specific output array to be ready on GPU.

Usage:
    CLM_ML_X64=1 python diags/precision_roofline.py --precision f64
    CLM_ML_X64=0 python diags/precision_roofline.py --precision f32

Each invocation appends its rows to:
    diags/figures/precision_roofline.csv

Columns: workload, precision, N_or_M, compile_s, median_ms, gflops

Run both precisions; second run auto-generates figure if both present.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description="f32 vs f64 roofline benchmark")
parser.add_argument("--precision", choices=["f32", "f64"], required=True)
args = parser.parse_args()
_PREC = args.precision

os.environ["CLM_ML_X64"] = "0" if _PREC == "f32" else "1"
os.environ["CLM_ML_NO_CHECKPOINT"] = "1"

_BENCH_CACHE = (
    f"/burg-archive/home/al4385/.cache/jax_compile_cache_roofline_{_PREC}"
)
os.environ["JAX_COMPILATION_CACHE_DIR"] = _BENCH_CACHE
os.makedirs(_BENCH_CACHE, exist_ok=True)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import multilayer_canopy.MLpftconMod as _pftcon_mod
from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs,
    MLCanopyFluxes,
    atm2lnd_inst, wateratm2lndbulk_inst, compute_gpp, compute_le, compute_h,
)

# ── Configuration ─────────────────────────────────────────────────────────────
# Workload A: single square GEMM of shape (M, M) @ (M, M)
# Large M saturates CUDA cores → exposes f32 vs f64 FLOP-rate difference.
GEMM_M_VALUES = [1024, 2048, 4096, 8192]

# Workload B: vmapped CLM-ML forward pass over N random parameter sets
N_VALUES_CLMML = [32, 128, 512, 1024, 2048]

REPEATS = 10   # more repeats → stable median

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = FIGURES_DIR / "precision_roofline.csv"

PAPER_FIG_DIR = Path(__file__).resolve().parent.parent / "Paper" / "jaxes_paper" / "figures"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

_clmml_dtype = jnp.float32 if _PREC == "f32" else jnp.float64

print(f"\n{'='*70}", flush=True)
print(f"Precision roofline benchmark — precision arg: {_PREC}", flush=True)
print(f"jax_enable_x64 = {jax.config.jax_enable_x64}", flush=True)
print(f"JAX cache dir: {_BENCH_CACHE}", flush=True)
print(f"GPU device: {jax.devices()[0]}", flush=True)
print(f"{'='*70}\n", flush=True)


def _time_fn(fn, *args, repeats=REPEATS):
    """Run fn(*args) `repeats` times after one warm call; return median wall time (s)."""
    out = fn(*args)
    jax.block_until_ready(out)   # wait for GPU compute, not just dispatch
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), out


# ─────────────────────────────────────────────────────────────────────────────
# Workload A — Single large square GEMM (FMA / CUDA-core bound)
# ─────────────────────────────────────────────────────────────────────────────

def _run_gemm_benchmark(dtype, dtype_label: str) -> list[dict]:
    """Time a single (M, M) @ (M, M) matmul for both precisions; returns rows."""
    print(f"\n{'─'*60}", flush=True)
    print(f"Workload A — Square GEMM (M×M @ M×M), dtype={dtype_label}", flush=True)

    rows = []
    for M in GEMM_M_VALUES:
        print(f"\n  M={M}", flush=True)
        # Memory estimate: 3 × M² × bytes_per_elem
        nbytes = 3 * M * M * (4 if dtype == jnp.float32 else 8)
        print(f"    memory estimate (3 tensors): {nbytes/1e9:.2f} GB", flush=True)

        try:
            rng = np.random.default_rng(0)
            dt = np.float32 if dtype == jnp.float32 else np.float64
            A = jnp.array(rng.standard_normal((M, M)).astype(dt), dtype=dtype)
            B = jnp.array(rng.standard_normal((M, M)).astype(dt), dtype=dtype)
            gemm_jit = jax.jit(jnp.matmul)

            # JIT compile
            t0 = time.perf_counter()
            out = gemm_jit(A, B)
            jax.block_until_ready(out)
            compile_s = time.perf_counter() - t0
            print(f"    compile + first call: {compile_s:.2f}s", flush=True)

            median_s, _ = _time_fn(gemm_jit, A, B)
            median_ms = median_s * 1000.0

            # GFLOPS = 2 × M³ / time_s / 1e9  (multiply-add counts as 2 ops)
            gflops = 2.0 * M**3 / median_s / 1e9

            print(f"    median: {median_ms:.3f}ms  →  {gflops:.1f} GFLOPS", flush=True)

        except Exception as e:
            print(f"    FAILED: {e}", flush=True)
            compile_s, median_ms, gflops = float("nan"), float("nan"), float("nan")

        rows.append({
            "workload":   "gemm",
            "precision":  dtype_label,
            "N_or_M":     M,
            "compile_s":  f"{compile_s:.3f}" if compile_s == compile_s else "nan",
            "median_ms":  f"{median_ms:.4f}" if median_ms == median_ms else "nan",
            "gflops":     f"{gflops:.1f}"    if gflops == gflops    else "nan",
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Workload B — CLM-ML physics forward pass (vmapped, transcendental-bound)
# ─────────────────────────────────────────────────────────────────────────────

_orig_pftcon      = _pftcon_mod.MLpftcon
_p                = grid.p
_n                = grid.ncan
_canopystate_inst = _mlcf_kwargs["canopystate_inst"]
_mlcf_kwargs_base = {
    k: v for k, v in _mlcf_kwargs.items()
    if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst", "canopystate_inst")
}


def forward_multi(scales):
    """vmap target: scales shape (5,); returns [GPP, H, LE]."""
    vc = scales[0] * jnp.asarray(_orig_pftcon.vcmaxpft, dtype=_clmml_dtype)
    mc = _canopystate_inst._replace(
        elai_patch=scales[4] * jnp.asarray(_canopystate_inst.elai_patch, dtype=_clmml_dtype),
        esai_patch=scales[4] * jnp.asarray(_canopystate_inst.esai_patch, dtype=_clmml_dtype))
    ma = atm2lnd_inst._replace(
        forc_t_downscaled_col     = scales[1] * jnp.asarray(atm2lnd_inst.forc_t_downscaled_col, dtype=_clmml_dtype),
        forc_solad_downscaled_col = scales[2] * jnp.asarray(atm2lnd_inst.forc_solad_downscaled_col, dtype=_clmml_dtype),
        forc_solai_grc            = scales[2] * jnp.asarray(atm2lnd_inst.forc_solai_grc, dtype=_clmml_dtype))
    mw = wateratm2lndbulk_inst._replace(
        forc_q_downscaled_col=scales[3] * jnp.asarray(wateratm2lndbulk_inst.forc_q_downscaled_col, dtype=_clmml_dtype))
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst, atm2lnd_inst=ma,
        wateratm2lndbulk_inst=mw, canopystate_inst=mc,
        vcmaxpft_jax=vc, **_mlcf_kwargs_base)
    return jnp.array([compute_gpp(inst, _p, _n),
                      compute_le(inst, _p, _n),
                      compute_h(inst, _p, _n)], dtype=_clmml_dtype)


def _run_clmml_benchmark() -> list[dict]:
    """Run CLM-ML vmapped forward pass benchmark; returns result dicts."""
    print(f"\n{'─'*60}", flush=True)
    print(f"Workload B — CLM-ML physics (vmapped), precision={_PREC}", flush=True)

    rng = np.random.default_rng(seed=42)
    dt = np.float32 if _PREC == "f32" else np.float64
    all_thetas = rng.uniform(0.8, 1.2, size=(max(N_VALUES_CLMML), 5)).astype(dt)

    rows = []
    _prev_fn = None
    for N in N_VALUES_CLMML:
        print(f"\n  N={N}", flush=True)

        try:
            thetas_N   = jnp.array(all_thetas[:N], dtype=_clmml_dtype)
            batched_fn = jax.jit(jax.vmap(forward_multi))

            t0 = time.perf_counter()
            out = batched_fn(thetas_N)
            jax.block_until_ready(out)
            compile_s = time.perf_counter() - t0
            print(f"    compile + first call: {compile_s:.1f}s", flush=True)

            median_s, _ = _time_fn(batched_fn, thetas_N)
            median_ms        = median_s * 1000.0
            ms_per_sample = median_ms / N

            print(f"    median={median_ms:.3f}ms  ({ms_per_sample:.4f}ms/sample)", flush=True)

        except Exception as e:
            print(f"    FAILED: {e}", flush=True)
            compile_s, median_ms, ms_per_sample = float("nan"), float("nan"), float("nan")

        rows.append({
            "workload":   "clm_ml",
            "precision":  _PREC,
            "N_or_M":     N,
            "compile_s":  f"{compile_s:.2f}" if compile_s == compile_s else "nan",
            "median_ms":  f"{ms_per_sample:.5f}" if ms_per_sample == ms_per_sample else "nan",
            "gflops":     "nan",
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Run benchmarks
# ─────────────────────────────────────────────────────────────────────────────

gemm_rows_f32 = _run_gemm_benchmark(jnp.float32, "f32")
gemm_rows_f64 = _run_gemm_benchmark(jnp.float64, "f64")
clmml_rows    = _run_clmml_benchmark()

all_rows = gemm_rows_f32 + gemm_rows_f64 + clmml_rows

# ─────────────────────────────────────────────────────────────────────────────
# Append to CSV
# ─────────────────────────────────────────────────────────────────────────────
fieldnames = ["workload", "precision", "N_or_M", "compile_s", "median_ms", "gflops"]
write_header = not CSV_PATH.exists()

with open(CSV_PATH, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerows(all_rows)

print(f"\nCSV appended: {CSV_PATH}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Summary — GEMM speedup
# ─────────────────────────────────────────────────────────────────────────────
def _lookup(rows_list, workload, precision, N):
    for r in rows_list:
        if r["workload"] == workload and r["precision"] == precision and r["N_or_M"] == N:
            v = float(r["median_ms"])
            return v if v == v else float("nan")
    return float("nan")

print(f"\n{'='*70}", flush=True)
print("GEMM speedup summary (f64_time / f32_time — larger = more f32 advantage):", flush=True)
for M in GEMM_M_VALUES:
    t32 = _lookup(gemm_rows_f32, "gemm", "f32", M)
    t64 = _lookup(gemm_rows_f64, "gemm", "f64", M)
    g32 = _lookup(gemm_rows_f32, "gemm", "f32", M)  # gflops stored separately
    ratio = t64 / t32 if t32 > 0 and t32 == t32 else float("nan")
    print(f"  M={M:5d}:  f32={t32:.3f}ms  f64={t64:.3f}ms  speedup={ratio:.1f}x", flush=True)

# Read CSV for CLM-ML cross-precision speedup
if CSV_PATH.exists():
    all_csv = []
    with open(CSV_PATH, newline="") as f:
        all_csv = list(csv.DictReader(f))

    def _csv_lookup(workload, precision, N):
        for r in all_csv:
            if (r["workload"] == workload and r["precision"] == precision
                    and int(r["N_or_M"]) == N):
                v = float(r["median_ms"])
                return v if v == v else float("nan")
        return float("nan")

    clmml_precs = {r["precision"] for r in all_csv if r["workload"] == "clm_ml"}
    print(f"\nCLM-ML speedup summary:", flush=True)
    if {"f32", "f64"}.issubset(clmml_precs):
        for N in N_VALUES_CLMML:
            t32 = _csv_lookup("clm_ml", "f32", N)
            t64 = _csv_lookup("clm_ml", "f64", N)
            ratio = t64 / t32 if t32 > 0 and t32 == t32 and t64 == t64 else float("nan")
            print(f"  N={N:5d}:  f32={t32:.5f}ms/sample  f64={t64:.5f}ms/sample  speedup={ratio:.2f}x",
                  flush=True)
    else:
        print(f"  CLM-ML data present for: {clmml_precs} (run both precisions for speedup ratio)",
              flush=True)

print(f"{'='*70}\n", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Plot (only when CSV has both precisions for CLM-ML)
# ─────────────────────────────────────────────────────────────────────────────

def _try_plot():
    if not CSV_PATH.exists():
        return

    rows_plot = []
    with open(CSV_PATH, newline="") as f:
        rows_plot = list(csv.DictReader(f))

    clmml_precs = {r["precision"] for r in rows_plot if r["workload"] == "clm_ml"}
    if not {"f32", "f64"}.issubset(clmml_precs):
        print(f"Plot deferred: CLM-ML data for {clmml_precs} only (need both).", flush=True)
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _series(workload, precision):
        pts = [(int(r["N_or_M"]), float(r["median_ms"]))
               for r in rows_plot
               if r["workload"] == workload and r["precision"] == precision
               and r["median_ms"] != "nan"]
        pts.sort()
        return [p[0] for p in pts], [p[1] for p in pts]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.5, 3.2))

    # ── Panel A — GEMM GFLOPS ────────────────────────────────────────────────
    def _gflops_series(precision):
        pts = [(int(r["N_or_M"]),
                2.0 * int(r["N_or_M"])**3 / (float(r["median_ms"]) * 1e-3) / 1e9)
               for r in rows_plot
               if r["workload"] == "gemm" and r["precision"] == precision
               and r["median_ms"] not in ("nan", "")]
        pts.sort()
        return [p[0] for p in pts], [p[1] for p in pts]

    gm_f32_x, gm_f32_y = _gflops_series("f32")
    gm_f64_x, gm_f64_y = _gflops_series("f64")

    axA.plot(gm_f32_x, gm_f32_y, "b-o",  lw=1.5, ms=5, label="f32")
    axA.plot(gm_f64_x, gm_f64_y, "r--s", lw=1.5, ms=5, label="f64")
    axA.set_xlabel("Matrix dimension M", fontsize=9)
    axA.set_ylabel("GFLOPS", fontsize=9)
    axA.set_title("Dense GEMM (M×M @ M×M)\nf32 ≫ f64 (CUDA-core bound)", fontsize=9)
    axA.legend(fontsize=8)
    axA.tick_params(labelsize=8)
    axA.set_xscale("log", base=2)
    axA.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v)}" if v > 0 else ""))

    # Annotate peak speedup
    if gm_f32_y and gm_f64_y:
        # Find matching M values
        f32_dict = dict(zip(gm_f32_x, gm_f32_y))
        f64_dict = dict(zip(gm_f64_x, gm_f64_y))
        common = sorted(set(f32_dict) & set(f64_dict))
        if common:
            best_M = max(common)
            sp = f32_dict[best_M] / f64_dict[best_M]
            axA.annotate(
                f"{sp:.0f}× at M={best_M}",
                xy=(best_M, f32_dict[best_M]),
                xytext=(best_M * 0.5, f32_dict[best_M] * 1.15),
                fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.8),
                ha="center",
            )

    # ── Panel B — CLM-ML ms/sample ───────────────────────────────────────────
    cl_f32_x, cl_f32_y = _series("clm_ml", "f32")
    cl_f64_x, cl_f64_y = _series("clm_ml", "f64")

    axB.plot(cl_f32_x, cl_f32_y, "b-o",  lw=1.5, ms=5, label="f32")
    axB.plot(cl_f64_x, cl_f64_y, "r--s", lw=1.5, ms=5, label="f64")
    axB.set_xlabel("Batch size N", fontsize=9)
    axB.set_ylabel("ms per sample", fontsize=9)
    axB.set_title("CLM-ML physics (vmapped)\nf32 ≈ f64 (SFU/transcendental bound)", fontsize=9)
    axB.legend(fontsize=8)
    axB.tick_params(labelsize=8)
    axB.set_xscale("log", base=2)
    axB.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v)}" if v > 0 else ""))

    # Annotate near-equal lines
    if cl_f32_y and cl_f64_y:
        f32d = dict(zip(cl_f32_x, cl_f32_y))
        f64d = dict(zip(cl_f64_x, cl_f64_y))
        common_n = sorted(set(f32d) & set(f64d))
        if common_n:
            mid = common_n[len(common_n) // 2]
            sp = f64d[mid] / f32d[mid]
            axB.annotate(
                f"{sp:.2f}× at N={mid}\n(SFU-bound)",
                xy=(mid, f32d[mid]),
                xytext=(mid * 1.6, f32d[mid] * 0.7),
                fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.8),
                ha="left",
            )

    fig.tight_layout()

    for fig_dir in [FIGURES_DIR, PAPER_FIG_DIR]:
        for ext in ["pdf", "png"]:
            path = fig_dir / f"precision_roofline.{ext}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved: {path}", flush=True)

    plt.close(fig)


_try_plot()
print(f"=== precision_roofline.py ({_PREC}) complete ===", flush=True)
