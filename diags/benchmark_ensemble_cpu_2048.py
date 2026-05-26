"""CPU vmap benchmark for N=512 only. Appends one row to ensemble_benchmark.csv."""
from __future__ import annotations
import os, sys, csv, time
from pathlib import Path

_CACHE_DIR = "/burg-archive/home/al4385/.cache/jax_compile_cache"
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _CACHE_DIR)
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "10")
os.makedirs(_CACHE_DIR, exist_ok=True)
print(f"JAX compilation cache dir: {_CACHE_DIR}", flush=True)

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

N_TARGET = 2048

_orig_pftcon     = _pftcon_mod.MLpftcon
_p               = grid.p
_n               = grid.ncan
_canopystate_inst = _mlcf_kwargs["canopystate_inst"]
_mlcf_kwargs_base = {k: v for k, v in _mlcf_kwargs.items()
                     if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst", "canopystate_inst")}

cpu_dev = jax.devices("cpu")[0]
print(f"CPU device: {cpu_dev}", flush=True)

atm2lnd_cpu     = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_dev), atm2lnd_inst)
watm_cpu        = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_dev), wateratm2lndbulk_inst)
mlcanopy_cpu    = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_dev), mlcanopy_inst)
canopy_cpu      = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_dev), _canopystate_inst)
vcmaxpft_cpu    = jax.device_put(_orig_pftcon.vcmaxpft, cpu_dev)

_kwargs_cpu = {}
for k, v in _mlcf_kwargs_base.items():
    try:
        _kwargs_cpu[k] = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, cpu_dev) if hasattr(x, "shape") else x, v)
    except Exception:
        _kwargs_cpu[k] = v


def forward_multi_cpu(scales):
    vc = scales[0] * vcmaxpft_cpu
    mc = canopy_cpu._replace(
        elai_patch=scales[4] * jnp.asarray(canopy_cpu.elai_patch, dtype=jnp.float64),
        esai_patch=scales[4] * jnp.asarray(canopy_cpu.esai_patch, dtype=jnp.float64))
    ma = atm2lnd_cpu._replace(
        forc_t_downscaled_col     = scales[1] * atm2lnd_cpu.forc_t_downscaled_col,
        forc_solad_downscaled_col = scales[2] * atm2lnd_cpu.forc_solad_downscaled_col,
        forc_solai_grc            = scales[2] * atm2lnd_cpu.forc_solai_grc)
    mw = watm_cpu._replace(forc_q_downscaled_col=scales[3] * watm_cpu.forc_q_downscaled_col)
    inst = MLCanopyFluxes(mlcanopy_inst=mlcanopy_cpu, atm2lnd_inst=ma,
                          wateratm2lndbulk_inst=mw, canopystate_inst=mc,
                          vcmaxpft_jax=vc, **_kwargs_cpu)
    return jnp.array([compute_gpp(inst, _p, _n), compute_h(inst, _p, _n), compute_le(inst, _p, _n)])


# Warm up single-sample JIT first
print("Compiling single-sample CPU JIT...", flush=True)
_fwd_jit = jax.jit(forward_multi_cpu)
t0 = time.perf_counter()
_ = _fwd_jit(jax.device_put(jnp.ones(5, dtype=jnp.float64), cpu_dev))
jax.effects_barrier()
print(f"  single JIT compile: {time.perf_counter()-t0:.1f}s", flush=True)

rng = np.random.default_rng(seed=42)
all_thetas = rng.uniform(0.8, 1.2, size=(2048, 5)).astype(np.float64)
thetas = jax.device_put(jnp.array(all_thetas[:N_TARGET]), cpu_dev)

batched_fn = jax.jit(jax.vmap(forward_multi_cpu))

REPEATS = 3
print(f"\n[CPU] N={N_TARGET} vmap — compiling (may take hours)...", flush=True)

t0 = time.perf_counter()
_ = batched_fn(thetas)
jax.effects_barrier()
first_s = time.perf_counter() - t0
print(f"  first call (compile+run): {first_s:.1f}s", flush=True)

times = []
for i in range(REPEATS):
    t0 = time.perf_counter()
    _ = batched_fn(thetas)
    jax.effects_barrier()
    times.append(time.perf_counter() - t0)
    print(f"  repeat {i+1}: {times[-1]*1000:.1f}ms", flush=True)

ss_ms = float(np.mean(times)) * 1000.0
ms_per_sample = ss_ms / N_TARGET
print(f"\nN={N_TARGET}: vmap_ss={ss_ms:.1f}ms  ms/sample={ms_per_sample:.2f}", flush=True)

csv_path = Path(__file__).parent / "figures" / "ensemble_benchmark.csv"
file_exists = csv_path.exists() and csv_path.stat().st_size > 0
with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["backend", "N", "vmap_ss_ms", "seq_ss_ms", "ms_per_sample", "gpu_cpu_speedup"])
    writer.writerow(["cpu", N_TARGET, f"{ss_ms:.1f}", "", f"{ms_per_sample:.2f}", ""])
print(f"Appended row to {csv_path}", flush=True)
print("=== benchmark_ensemble_cpu_2048.py complete ===", flush=True)
