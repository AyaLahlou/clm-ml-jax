"""
aot_compile.py — Ahead-of-Time (AOT) compilation for the CLM-ML-JAX forward pass.

Two strategies are implemented:

1. **Persistent cache warm-up** (always runs):
   Triggers JIT compilation once and lets JAX's persistent cache save the
   XLA binary.  Subsequent runs (same hardware, same code) skip the 290s
   compile and load from cache instead.

2. **jax.export serialization** (optional, --export flag):
   Serializes the compiled XLA artifact to a .mlirbc file using the
   ``jax.export`` API (requires JAX ≥ 0.4.25).  The artifact can be
   loaded on the same hardware without any Python/JAX recompilation:
       loaded = jax.export.deserialize(open("forward.mlirbc","rb").read())
       result = loaded.call(mlcanopy_inst)

Usage:
    # Strategy 1 only (recommended for routine use):
    CLM_ML_NO_CHECKPOINT=1 python diags/aot_compile.py

    # Strategy 1 + export artifact:
    CLM_ML_NO_CHECKPOINT=1 python diags/aot_compile.py --export

    # Both with a custom artifact path:
    CLM_ML_NO_CHECKPOINT=1 python diags/aot_compile.py --export --out compiled_forward.mlirbc

Environment:
    JAX_COMPILATION_CACHE_DIR  — persistent cache directory (default: ~/.cache/jax_compile_cache)
    CLM_ML_NO_CHECKPOINT=1     — disable gradient-checkpoint recomputation during tracing
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR      = _PROJECT_ROOT / "src"
for _p in (_SRC_DIR, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── JAX setup ────────────────────────────────────────────────────────────────
os.environ.setdefault("CLM_ML_NO_CHECKPOINT", "1")

import jax
jax.config.update("jax_enable_x64", True)

# Enable persistent cache before importing anything else
_cache_dir = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.expanduser("~/.cache/jax_compile_cache"),
)
os.makedirs(_cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 10.0)
print(f"JAX compilation cache: {_cache_dir}", flush=True)
print(f"JAX devices: {jax.devices()}", flush=True)
print(f"JAX backend: {jax.default_backend()}", flush=True)

import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="AOT compile the CLM-ML-JAX forward pass")
    parser.add_argument("--export", action="store_true",
                        help="Also serialize compiled artifact via jax.export (JAX ≥ 0.4.25)")
    parser.add_argument("--out", default="compiled_forward.mlirbc",
                        help="Output path for serialized artifact (--export only)")
    args = parser.parse_args()

    # ── Initialise CHATS7 ──────────────────────────────────────────────────────
    print("\n=== Initialising CHATS7 ===", flush=True)
    t0 = time.perf_counter()
    from diags.expt_init import mlcanopy_inst, grid, forward_fn
    print(f"  init: {time.perf_counter()-t0:.1f}s", flush=True)

    # ── Strategy 1: JIT warm-up (populates persistent cache) ─────────────────
    print("\n=== Strategy 1: JIT warm-up + persistent cache ===", flush=True)
    jit_fn = jax.jit(forward_fn)

    print("  First call (compiles + caches) ...", flush=True)
    t0 = time.perf_counter()
    out = jit_fn(mlcanopy_inst)
    jax.effects_barrier()
    compile_s = time.perf_counter() - t0
    print(f"  compile time: {compile_s:.1f}s", flush=True)

    print("  Second call (should load from cache) ...", flush=True)
    t0 = time.perf_counter()
    out = jit_fn(mlcanopy_inst)
    jax.effects_barrier()
    second_s = time.perf_counter() - t0
    print(f"  second call:  {second_s:.3f}s", flush=True)

    if second_s < compile_s * 0.1:
        print("  Cache hit confirmed (second call >> 10× faster than compile)", flush=True)
    else:
        print("  Cache miss or cold — check JAX_COMPILATION_CACHE_DIR", flush=True)

    # Steady-state timing
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        jit_fn(mlcanopy_inst)
        jax.effects_barrier()
        times.append(time.perf_counter() - t0)
    ss_ms = float(np.mean(times)) * 1000
    print(f"  Steady-state: {ss_ms:.1f} ms/step  (mean of 5 repeats)", flush=True)

    # ── Strategy 2: jax.export serialization ─────────────────────────────────
    if args.export:
        print("\n=== Strategy 2: jax.export serialization ===", flush=True)
        try:
            import jax.export as jax_export
        except ImportError:
            print("  jax.export not available (requires JAX ≥ 0.4.25) — skipping", flush=True)
            return

        try:
            print("  Lowering ...", flush=True)
            t0 = time.perf_counter()
            exported = jax_export.export(jax.jit(forward_fn))(mlcanopy_inst)
            lower_s = time.perf_counter() - t0
            print(f"  Lowered in {lower_s:.1f}s", flush=True)

            artifact = exported.serialize()
            out_path = Path(args.out)
            out_path.write_bytes(artifact)
            print(f"  Artifact written: {out_path} ({len(artifact)/1024:.0f} KB)", flush=True)

            # Verify round-trip load
            print("  Verifying round-trip load ...", flush=True)
            loaded = jax_export.deserialize(artifact)
            t0 = time.perf_counter()
            out_loaded = loaded.call(mlcanopy_inst)
            jax.effects_barrier()
            load_s = time.perf_counter() - t0
            print(f"  Loaded + ran in {load_s:.3f}s", flush=True)
            print("  Serialization round-trip OK", flush=True)

        except Exception as e:
            print(f"  jax.export failed: {e}", flush=True)
            print("  Falling back to persistent cache only (Strategy 1 always works)", flush=True)


if __name__ == "__main__":
    main()
