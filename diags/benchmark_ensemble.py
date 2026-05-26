"""
benchmark_ensemble.py — Parameter ensemble benchmark: GPU vs CPU speedup.

Demonstrates GPU advantage for CLM-ML-JAX by running N independent
parameter samples simultaneously via jax.vmap, vs N sequential calls on CPU.

The key insight: vmap over N parameter ensembles exploits GPU parallelism
across the batch dimension, while a single-column forward pass has too
little arithmetic intensity to benefit from GPU.

Parameters: 5 scale factors (Vcmax25, T_air, SW_rad, q_ref, dpai)
sampled uniformly from [0.8, 1.2] around baseline.

Outputs per sample: [GPP, H, LE]

N values tested: 1, 8, 32, 128, 512, 1024, 2048

Metrics:
  - vmap_first_s  : first call time (includes JIT compile for this N)
  - vmap_ss_s     : mean of `repeats` subsequent calls (pure throughput)
  - seq_ss_s      : N sequential calls to jit(forward_multi), mean of repeats
  - speedup       : seq_ss_s / vmap_ss_s
  - ms_per_sample : vmap_ss_s / N * 1000

Output:
  diags/figures/ensemble_benchmark.csv
  diags/figures/ensemble_benchmark.png
"""
from __future__ import annotations

import sys
import csv
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Import forward function dependencies from sensitivity_analysis ────────────
import multilayer_canopy.MLpftconMod as _pftcon_mod

from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs,
    jax, jnp, MLCanopyFluxes,
    atm2lnd_inst, wateratm2lndbulk_inst, compute_gpp, compute_le, compute_h,
)

_orig_pftcon = _pftcon_mod.MLpftcon   # original MLpftcon instance

_p = grid.p
_n = grid.ncan

# Build kwargs without the three instances we'll pass as traced args
_canopystate_inst = _mlcf_kwargs["canopystate_inst"]
_mlcf_kwargs_base = {k: v for k, v in _mlcf_kwargs.items()
                     if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst",
                                  "canopystate_inst")}


# ── forward_multi: copied exactly from sensitivity_analysis.py lines 148-218 ──
def forward_multi(scales: jnp.ndarray) -> jnp.ndarray:
    """
    Args:
        scales: float64 array of shape (5,) — scale factors at baseline = 1.0
          0: alpha_vcmax25  — scale on vcmaxpft (passed as vcmaxpft_jax arg)
          1: alpha_tair     — scale on atm2lnd_inst.forc_t_downscaled_col
          2: alpha_sw       — scale on atm2lnd_inst.forc_solad & forc_solai
          3: alpha_qref     — scale on wateratm2lndbulk_inst.forc_q_downscaled_col
          4: alpha_dpai     — scale on canopystate_inst.elai_patch & esai_patch

    Returns:
        jnp.ndarray of shape (3,): [GPP (umol CO2/m2/s), H_canopy (W/m2), LE_canopy (W/m2)]
    """
    # Scale vcmax via vcmaxpft_jax — bypasses JIT cache so gradient flows correctly.
    vcmaxpft_jax = scales[0] * _orig_pftcon.vcmaxpft

    # Scale LAI/SAI in canopystate_inst — actual source of dpai_profile.
    modified_canopy = _canopystate_inst._replace(
        elai_patch = scales[4] * jnp.asarray(_canopystate_inst.elai_patch, dtype=jnp.float64),
        esai_patch = scales[4] * jnp.asarray(_canopystate_inst.esai_patch, dtype=jnp.float64),
    )

    # Scale forcing via atm2lnd_inst (actual source used by physics).
    modified_atm = atm2lnd_inst._replace(
        forc_t_downscaled_col     = scales[1] * atm2lnd_inst.forc_t_downscaled_col,
        forc_solad_downscaled_col = scales[2] * atm2lnd_inst.forc_solad_downscaled_col,
        forc_solai_grc            = scales[2] * atm2lnd_inst.forc_solai_grc,
    )
    modified_watm = wateratm2lndbulk_inst._replace(
        forc_q_downscaled_col = scales[3] * wateratm2lndbulk_inst.forc_q_downscaled_col,
    )

    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=modified_watm,
        canopystate_inst=modified_canopy,
        vcmaxpft_jax=vcmaxpft_jax,
        **_mlcf_kwargs_base,
    )

    GPP = compute_gpp(inst, _p, _n)
    H   = compute_h(inst, _p, _n)
    LE  = compute_le(inst, _p, _n)

    return jnp.array([GPP, H, LE])


# ── Parameter sampling ────────────────────────────────────────────────────────
rng = np.random.default_rng(seed=42)
MAX_N = 2048
all_thetas_np = rng.uniform(0.8, 1.2, size=(MAX_N, 5)).astype(np.float64)

# ── Compile single-sample JIT first ──────────────────────────────────────────
_forward_jit = jax.jit(forward_multi)
print("Compiling single-sample JIT...", flush=True)
t0 = time.perf_counter()
_ = _forward_jit(jnp.ones(5, dtype=jnp.float64))
jax.effects_barrier()
print(f"  JIT compile: {time.perf_counter()-t0:.1f}s", flush=True)

# Warm up again to get pure inference time
t0 = time.perf_counter()
_ = _forward_jit(jnp.ones(5, dtype=jnp.float64))
jax.effects_barrier()
single_fwd_s = time.perf_counter() - t0
print(f"  Single forward (post-JIT): {single_fwd_s*1000:.1f}ms", flush=True)


# ── Benchmark helpers ─────────────────────────────────────────────────────────
N_VALS   = [1, 8, 32, 128, 512, 1024, 2048]
REPEATS  = 3
# Sequential benchmark is skipped for N > this threshold on GPU (still too fast)
# and for N > SEQ_CPU_CAP on CPU (too slow: ~32ms * N)
SEQ_GPU_CAP = 128   # GPU sequential capped to avoid hour-long runs
SEQ_CPU_CAP = 64    # CPU sequential capped (32ms * 64 = ~2s; >128 would be hours)


def run_vmap_benchmark(batched_fn, thetas, repeats=REPEATS):
    """First call (JIT) + timed repeats. Returns (first_s, ss_s)."""
    t0 = time.perf_counter()
    out = batched_fn(thetas)
    jax.effects_barrier()
    first_s = time.perf_counter() - t0

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = batched_fn(thetas)
        jax.effects_barrier()
        times.append(time.perf_counter() - t0)
    ss_s = float(np.mean(times))
    return first_s, ss_s, out


def run_seq_benchmark(forward_jit_fn, thetas, repeats=REPEATS):
    """N sequential calls per repeat. Returns mean total wall time over repeats."""
    N = thetas.shape[0]

    def run_n():
        for i in range(N):
            _ = forward_jit_fn(thetas[i])
        jax.effects_barrier()

    # Warm-up pass
    run_n()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run_n()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


# ─────────────────────────────────────────────────────────────────────────────
# GPU benchmark
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print("GPU BENCHMARK", flush=True)
print("="*60, flush=True)

gpu_dev = jax.devices("gpu")[0] if jax.devices("gpu") else None
if gpu_dev is None:
    print("WARNING: No GPU device found — running GPU section on default device.", flush=True)

all_thetas_gpu = jnp.array(all_thetas_np)  # place on default (GPU) device

gpu_rows = []  # list of dicts

for N in N_VALS:
    thetas_n = all_thetas_gpu[:N]
    batched_fn = jax.jit(jax.vmap(forward_multi))

    print(f"\n[GPU] N={N} vmap ...", flush=True)
    vmap_first_s, vmap_ss_s, _ = run_vmap_benchmark(batched_fn, thetas_n)
    ms_per_sample = vmap_ss_s / N * 1000.0
    print(f"  vmap first={vmap_first_s:.2f}s  steady={vmap_ss_s*1000:.1f}ms  "
          f"ms/sample={ms_per_sample:.2f}", flush=True)

    # Sequential: only for N <= SEQ_GPU_CAP
    if N <= SEQ_GPU_CAP:
        print(f"[GPU] N={N} sequential ...", flush=True)
        # Re-JIT forward on GPU device
        fwd_gpu = jax.jit(forward_multi)
        # Ensure warmup
        _ = fwd_gpu(thetas_n[0])
        jax.effects_barrier()
        seq_ss_s = run_seq_benchmark(fwd_gpu, thetas_n)
        speedup = seq_ss_s / vmap_ss_s if vmap_ss_s > 0 else float("nan")
        print(f"  sequential steady={seq_ss_s*1000:.1f}ms  speedup(seq/vmap)={speedup:.2f}x",
              flush=True)
    else:
        seq_ss_s = float("nan")
        speedup = float("nan")
        print(f"  [GPU] N={N} sequential skipped (N > {SEQ_GPU_CAP})", flush=True)

    gpu_rows.append(dict(
        backend="gpu",
        N=N,
        vmap_first_s=vmap_first_s,
        vmap_ss_s=vmap_ss_s,
        seq_ss_s=seq_ss_s,
        speedup_vs_seq_same=speedup,
        speedup_vs_cpu_seq=float("nan"),  # filled in later
        ms_per_sample=ms_per_sample,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# CPU benchmark
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print("CPU BENCHMARK", flush=True)
print("="*60, flush=True)

cpu_dev = jax.devices("cpu")[0]

# Move all closed-over arrays to CPU
atm2lnd_inst_cpu        = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_dev), atm2lnd_inst)
wateratm2lndbulk_cpu    = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_dev), wateratm2lndbulk_inst)
mlcanopy_inst_cpu       = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_dev), mlcanopy_inst)
canopystate_inst_cpu    = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_dev), _canopystate_inst)
orig_vcmaxpft_cpu       = jax.device_put(_orig_pftcon.vcmaxpft, cpu_dev)

# Also move the static kwargs that contain JAX arrays
_mlcf_kwargs_base_cpu = {}
for k, v in _mlcf_kwargs_base.items():
    try:
        _mlcf_kwargs_base_cpu[k] = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, cpu_dev) if hasattr(x, 'shape') else x, v
        )
    except Exception:
        _mlcf_kwargs_base_cpu[k] = v

all_thetas_cpu = jax.device_put(jnp.array(all_thetas_np), cpu_dev)


def forward_multi_cpu(scales: jnp.ndarray) -> jnp.ndarray:
    """CPU version of forward_multi — all arrays on CPU device."""
    vcmaxpft_jax = scales[0] * orig_vcmaxpft_cpu

    modified_canopy = canopystate_inst_cpu._replace(
        elai_patch = scales[4] * jnp.asarray(canopystate_inst_cpu.elai_patch, dtype=jnp.float64),
        esai_patch = scales[4] * jnp.asarray(canopystate_inst_cpu.esai_patch, dtype=jnp.float64),
    )
    modified_atm = atm2lnd_inst_cpu._replace(
        forc_t_downscaled_col     = scales[1] * atm2lnd_inst_cpu.forc_t_downscaled_col,
        forc_solad_downscaled_col = scales[2] * atm2lnd_inst_cpu.forc_solad_downscaled_col,
        forc_solai_grc            = scales[2] * atm2lnd_inst_cpu.forc_solai_grc,
    )
    modified_watm = wateratm2lndbulk_cpu._replace(
        forc_q_downscaled_col = scales[3] * wateratm2lndbulk_cpu.forc_q_downscaled_col,
    )

    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst_cpu,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=modified_watm,
        canopystate_inst=modified_canopy,
        vcmaxpft_jax=vcmaxpft_jax,
        **_mlcf_kwargs_base_cpu,
    )

    GPP = compute_gpp(inst, _p, _n)
    H   = compute_h(inst, _p, _n)
    LE  = compute_le(inst, _p, _n)

    return jnp.array([GPP, H, LE])


# Compile CPU JIT
print("Compiling CPU single-sample JIT...", flush=True)
_fwd_cpu_jit = jax.jit(forward_multi_cpu)
t0 = time.perf_counter()
_ = _fwd_cpu_jit(jax.device_put(jnp.ones(5, dtype=jnp.float64), cpu_dev))
jax.effects_barrier()
print(f"  CPU JIT compile: {time.perf_counter()-t0:.1f}s", flush=True)

t0 = time.perf_counter()
_ = _fwd_cpu_jit(jax.device_put(jnp.ones(5, dtype=jnp.float64), cpu_dev))
jax.effects_barrier()
cpu_single_fwd_s = time.perf_counter() - t0
print(f"  CPU single forward (post-JIT): {cpu_single_fwd_s*1000:.1f}ms", flush=True)

cpu_rows = []

for N in N_VALS:
    thetas_n_cpu = all_thetas_cpu[:N]
    batched_fn_cpu = jax.jit(jax.vmap(forward_multi_cpu))

    print(f"\n[CPU] N={N} vmap ...", flush=True)
    vmap_first_s, vmap_ss_s, _ = run_vmap_benchmark(batched_fn_cpu, thetas_n_cpu)
    ms_per_sample = vmap_ss_s / N * 1000.0
    print(f"  vmap first={vmap_first_s:.2f}s  steady={vmap_ss_s*1000:.1f}ms  "
          f"ms/sample={ms_per_sample:.2f}", flush=True)

    if N <= SEQ_CPU_CAP:
        print(f"[CPU] N={N} sequential ...", flush=True)
        _ = _fwd_cpu_jit(thetas_n_cpu[0])
        jax.effects_barrier()
        seq_ss_s = run_seq_benchmark(_fwd_cpu_jit, thetas_n_cpu)
        speedup = seq_ss_s / vmap_ss_s if vmap_ss_s > 0 else float("nan")
        print(f"  sequential steady={seq_ss_s*1000:.1f}ms  speedup(seq/vmap)={speedup:.2f}x",
              flush=True)
    else:
        seq_ss_s = float("nan")
        speedup = float("nan")
        print(f"  [CPU] N={N} sequential skipped (N > {SEQ_CPU_CAP})", flush=True)

    cpu_rows.append(dict(
        backend="cpu",
        N=N,
        vmap_first_s=vmap_first_s,
        vmap_ss_s=vmap_ss_s,
        seq_ss_s=seq_ss_s,
        speedup_vs_seq_same=speedup,
        speedup_vs_cpu_seq=float("nan"),
        ms_per_sample=ms_per_sample,
    ))


# ── Compute GPU speedup vs CPU seq ────────────────────────────────────────────
cpu_seq_map = {r["N"]: r["seq_ss_s"] for r in cpu_rows if not np.isnan(r["seq_ss_s"])}
for row in gpu_rows:
    N = row["N"]
    if N in cpu_seq_map:
        row["speedup_vs_cpu_seq"] = cpu_seq_map[N] / row["vmap_ss_s"]


# ── Save CSV ──────────────────────────────────────────────────────────────────
all_rows = gpu_rows + cpu_rows
csv_path = FIGURES_DIR / "ensemble_benchmark.csv"
fieldnames = ["backend", "N", "vmap_first_s", "vmap_ss_s", "seq_ss_s",
              "speedup_vs_seq_same", "speedup_vs_cpu_seq", "ms_per_sample"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)
print(f"\nCSV saved: {csv_path}", flush=True)


# ── Print summary table ────────────────────────────────────────────────────────
print("\n" + "="*80, flush=True)
print(f"{'backend':8s}  {'N':6s}  {'vmap_first_s':12s}  {'vmap_ss_ms':10s}  "
      f"{'seq_ss_ms':10s}  {'speedup_same':12s}  {'speedup_cpu':10s}  {'ms/sample':10s}")
print("-"*80, flush=True)
for row in all_rows:
    def fmt(v, fmt_str="{:.2f}"):
        return "N/A" if (isinstance(v, float) and np.isnan(v)) else fmt_str.format(v)
    print(f"{row['backend']:8s}  {row['N']:6d}  {fmt(row['vmap_first_s']):12s}  "
          f"{fmt(row['vmap_ss_s']*1000):10s}  {fmt(row['seq_ss_s']*1000 if not np.isnan(row['seq_ss_s']) else float('nan')):10s}  "
          f"{fmt(row['speedup_vs_seq_same']):12s}  {fmt(row['speedup_vs_cpu_seq']):10s}  "
          f"{fmt(row['ms_per_sample']):10s}", flush=True)


# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

gpu_N  = [r["N"] for r in gpu_rows]
gpu_ms = [r["ms_per_sample"] for r in gpu_rows]
cpu_N  = [r["N"] for r in cpu_rows]
cpu_ms = [r["ms_per_sample"] for r in cpu_rows]

ax0 = axes[0]
ax0.loglog(gpu_N, gpu_ms, "o-b", linewidth=2, markersize=7, label="GPU vmap")
ax0.loglog(cpu_N, cpu_ms, "s-r", linewidth=2, markersize=7, label="CPU vmap")

# Annotate single-forward reference lines
ax0.axhline(single_fwd_s * 1000, color="b", linestyle="--", alpha=0.5,
            label=f"GPU single ({single_fwd_s*1000:.0f}ms)")
ax0.axhline(cpu_single_fwd_s * 1000, color="r", linestyle="--", alpha=0.5,
            label=f"CPU single ({cpu_single_fwd_s*1000:.0f}ms)")

# Annotate crossover
for i, (gN, gms) in enumerate(zip(gpu_N, gpu_ms)):
    # find CPU at same N
    cpu_ms_at_N = {r["N"]: r["ms_per_sample"] for r in cpu_rows}
    if gN in cpu_ms_at_N and gms < cpu_ms_at_N[gN]:
        ax0.annotate(f"GPU faster\nN={gN}", xy=(gN, gms),
                     xytext=(gN * 1.5, gms * 2),
                     arrowprops=dict(arrowstyle="->", color="green"),
                     fontsize=8, color="green")
        break

ax0.set_xlabel("Ensemble size N", fontsize=12)
ax0.set_ylabel("ms / sample", fontsize=12)
ax0.set_title("(a) Throughput: ms per sample vs N", fontsize=11, fontweight="bold")
ax0.legend(fontsize=9)
ax0.grid(True, which="both", alpha=0.3)
ax0.set_xticks(N_VALS)
ax0.set_xticklabels([str(n) for n in N_VALS], fontsize=8)

# Right panel: speedup
ax1 = axes[1]

# GPU vmap vs GPU sequential (where available)
gpu_speedup_same_N = [r["N"] for r in gpu_rows if not np.isnan(r["speedup_vs_seq_same"])]
gpu_speedup_same   = [r["speedup_vs_seq_same"] for r in gpu_rows if not np.isnan(r["speedup_vs_seq_same"])]
if gpu_speedup_same_N:
    ax1.semilogx(gpu_speedup_same_N, gpu_speedup_same, "o-b", linewidth=2, markersize=7,
                 label="GPU vmap / GPU sequential")

# GPU vmap vs CPU sequential (where available)
gpu_speedup_cpu_N = [r["N"] for r in gpu_rows if not np.isnan(r["speedup_vs_cpu_seq"])]
gpu_speedup_cpu   = [r["speedup_vs_cpu_seq"] for r in gpu_rows if not np.isnan(r["speedup_vs_cpu_seq"])]
if gpu_speedup_cpu_N:
    ax1.semilogx(gpu_speedup_cpu_N, gpu_speedup_cpu, "^-g", linewidth=2, markersize=7,
                 label="GPU vmap / CPU sequential")

# CPU vmap vs CPU sequential (where available)
cpu_speedup_same_N = [r["N"] for r in cpu_rows if not np.isnan(r["speedup_vs_seq_same"])]
cpu_speedup_same   = [r["speedup_vs_seq_same"] for r in cpu_rows if not np.isnan(r["speedup_vs_seq_same"])]
if cpu_speedup_same_N:
    ax1.semilogx(cpu_speedup_same_N, cpu_speedup_same, "s--r", linewidth=2, markersize=7,
                 label="CPU vmap / CPU sequential")

ax1.axhline(1.0, color="k", linestyle=":", alpha=0.5, label="1x (no speedup)")
ax1.set_xlabel("Ensemble size N", fontsize=12)
ax1.set_ylabel("Speedup (×)", fontsize=12)
ax1.set_title("(b) Speedup vs sequential baseline", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, which="both", alpha=0.3)
ax1.set_xticks(N_VALS)
ax1.set_xticklabels([str(n) for n in N_VALS], fontsize=8)

fig.suptitle(
    "CLM-ML-JAX: Parameter ensemble benchmark — CHATS7, GPU vs CPU\n"
    "(5-param uniform prior [0.8,1.2], vmap N=1..2048, outputs: GPP/H/LE)",
    fontsize=10,
)
fig.tight_layout()

out_png = FIGURES_DIR / "ensemble_benchmark.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved: {out_png}", flush=True)

print("\n=== benchmark_ensemble.py complete ===", flush=True)
