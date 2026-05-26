"""
AD vs FD Scaling Experiment.

Empirically demonstrates the O(1) vs O(2p) scaling advantage of reverse-mode
AD over finite differences for gradient-based calibration.

For a scalar loss L(theta), p-vector theta:
  - T_AD = 1 backward pass  ≈ T_ratio * T_forward  (independent of p)
  - T_FD = 2p forward passes (central differences)
  - Crossover: AD wins when p > T_ratio / 2

Parameters used (5 verified params with correct gradients):
  0  alpha_sw    — shortwave radiation
  1  alpha_tref  — air temperature
  2  alpha_vcmax — Vcmax25
  3  alpha_iota  — WUE iota_SPA
  4  alpha_dpai  — dpai profile scaling (via mlcanopy_inst)

For p in [1, 2, 3, 5]:
  - Use first p params from the list above
  - Measure wall time of jax.grad(loss_fn)(theta_p) [AD]
  - Measure wall time of central FD gradient (2p forward calls)
  - Report T_FD / T_AD at each p

Outputs:
  diags/figures/ad_vs_fd_scaling.csv
  diags/figures/ad_vs_fd_scaling.png
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

os.environ["CLM_ML_NO_CHECKPOINT"] = "1"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _d in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared init ───────────────────────────────────────────────────────────────
print("=== ad_vs_fd_scaling.py: loading model ===", flush=True)
from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs,
    MLCanopyFluxes,
    atm2lnd_inst, wateratm2lndbulk_inst,
    compute_gpp,
)
import multilayer_canopy.MLpftconMod              as _MLpftconMod
import multilayer_canopy.MLLeafPhotosynthesisMod  as _LeafMod
import multilayer_canopy.MLCanopyNitrogenProfileMod as _NitroMod

_p    = grid.p
_ncan = grid.ncan

_mlcf_kwargs_no_atm = {k: v for k, v in _mlcf_kwargs.items()
                       if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst")}

_orig_pftcon = _MLpftconMod.MLpftcon
_orig_dpai   = mlcanopy_inst.dpai_profile  # shape (p, ncan+1)


# ── pftcon injection helpers ──────────────────────────────────────────────────

def _set_pftcon(new_inst):
    _MLpftconMod.MLpftcon = new_inst
    _LeafMod.MLpftcon     = new_inst
    _NitroMod.MLpftcon    = new_inst


def _restore_pftcon():
    _MLpftconMod.MLpftcon = _orig_pftcon
    _LeafMod.MLpftcon     = _orig_pftcon
    _NitroMod.MLpftcon    = _orig_pftcon


# ── p=5 forward function ──────────────────────────────────────────────────────
#
# theta = [alpha_sw, alpha_tref, alpha_vcmax, alpha_iota, alpha_dpai]
#
# We keep a full-length theta of size 5 but zero-pad when p < 5 by fixing
# non-active parameters to 1.0 in the wrapper functions below.

def _forward_gpp_full(theta5: jnp.ndarray) -> jnp.ndarray:
    """Full p=5 differentiable forward pass. Returns scalar GPP.

    theta5[0] alpha_sw    — scale on shortwave (direct + diffuse)
    theta5[1] alpha_tref  — scale on air temperature
    theta5[2] alpha_vcmax — global scale on vcmaxpft array
    theta5[3] alpha_iota  — global scale on iota_SPA array
    theta5[4] alpha_dpai  — scale on dpai_profile (leaf area index profile)
    """
    # Build modified atm2lnd_inst
    modified_atm = atm2lnd_inst._replace(
        forc_solad_downscaled_col = theta5[0] * atm2lnd_inst.forc_solad_downscaled_col,
        forc_solai_grc            = theta5[0] * atm2lnd_inst.forc_solai_grc,
        forc_t_downscaled_col     = theta5[1] * atm2lnd_inst.forc_t_downscaled_col,
    )

    # iota_SPA: module-global mutation (JAX traces through jnp.asarray)
    _set_pftcon(_orig_pftcon._replace(
        iota_SPA=theta5[3] * _orig_pftcon.iota_SPA,
    ))

    # vcmaxpft: explicit JAX arg to bypass JIT cache
    vcmaxpft_jax = theta5[2] * _orig_pftcon.vcmaxpft

    # dpai_profile: scale via mlcanopy_inst replacement
    modified_mcan = mlcanopy_inst._replace(
        dpai_profile=theta5[4] * _orig_dpai,
    )

    inst = MLCanopyFluxes(
        mlcanopy_inst=modified_mcan,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        vcmaxpft_jax=vcmaxpft_jax,
        **_mlcf_kwargs_no_atm,
    )
    _restore_pftcon()
    return compute_gpp(inst, _p, _ncan)


def _make_loss_fn_p(p: int):
    """Return a loss_fn(theta_p) → scalar that uses the first p params."""

    # Compute GPP_target once at theta = ones
    theta_star_full = jnp.ones(5, dtype=jnp.float64)
    GPP_target = _forward_gpp_full(theta_star_full)
    jax.block_until_ready(GPP_target)
    gpp_target_f = float(GPP_target)
    print(f"  GPP_target (p={p}): {gpp_target_f:.6f}", flush=True)

    def loss_fn(theta_p: jnp.ndarray) -> jnp.ndarray:
        # Pad to length-5 with ones for inactive params
        padding = jnp.ones(5 - p, dtype=jnp.float64)
        theta5  = jnp.concatenate([theta_p, padding])
        gpp     = _forward_gpp_full(theta5)
        return ((gpp - GPP_target) / (jnp.abs(GPP_target) + 1e-6)) ** 2

    return loss_fn, gpp_target_f


# ── Timing utilities ──────────────────────────────────────────────────────────

N_TIMING = 5   # median over this many warm runs

def _time_fn(fn, arg, n_runs: int = N_TIMING) -> float:
    """Time a JIT-compiled function (assumes JIT already warm). Returns median (s)."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = fn(arg)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _fd_gradient(loss_fn_scalar, theta_p: np.ndarray, eps: float = 1e-4) -> float:
    """Compute central-difference gradient and return wall time (s).

    Performs 2p forward evaluations.  Returns (grad_np, elapsed_s).
    """
    p      = len(theta_p)
    grad   = np.zeros(p)
    t0     = time.perf_counter()
    for i in range(p):
        th_plus  = theta_p.copy(); th_plus[i]  += eps
        th_minus = theta_p.copy(); th_minus[i] -= eps
        fp = float(loss_fn_scalar(jnp.array(th_plus,  dtype=jnp.float64)))
        fm = float(loss_fn_scalar(jnp.array(th_minus, dtype=jnp.float64)))
        grad[i] = (fp - fm) / (2.0 * eps)
    elapsed = time.perf_counter() - t0
    return grad, elapsed


# ── Main timing sweep ─────────────────────────────────────────────────────────

P_VALUES = [1, 2, 3, 5]

results = []  # list of dicts

for p in P_VALUES:
    print(f"\n{'='*60}", flush=True)
    print(f"=== p = {p} parameters ===", flush=True)

    loss_fn, gpp_target = _make_loss_fn_p(p)

    theta_ones = jnp.ones(p, dtype=jnp.float64)
    theta_np   = np.ones(p, dtype=np.float64)

    # ── JIT compile forward ───────────────────────────────────────────────────
    print(f"  JIT compile forward (p={p})...", flush=True)
    _loss_jit = jax.jit(loss_fn)
    t_compile_fwd = time.perf_counter()
    _ = jax.block_until_ready(_loss_jit(theta_ones))
    t_compile_fwd = time.perf_counter() - t_compile_fwd
    print(f"  Forward JIT compile+first eval: {t_compile_fwd:.2f}s", flush=True)

    # ── Measure T_forward ─────────────────────────────────────────────────────
    T_forward_s = _time_fn(_loss_jit, theta_ones)
    print(f"  T_forward (median {N_TIMING} runs): {T_forward_s:.4f}s", flush=True)

    # ── JIT compile backward ──────────────────────────────────────────────────
    print(f"  JIT compile backward (p={p})...", flush=True)
    _grad_jit = jax.jit(jax.grad(loss_fn))
    t_compile_bwd = time.perf_counter()
    _ = jax.block_until_ready(_grad_jit(theta_ones))
    t_compile_bwd = time.perf_counter() - t_compile_bwd
    print(f"  Backward JIT compile+first eval: {t_compile_bwd:.2f}s", flush=True)

    # ── Measure T_AD ─────────────────────────────────────────────────────────
    T_ad_s = _time_fn(_grad_jit, theta_ones)
    print(f"  T_AD (jax.grad, median {N_TIMING} runs): {T_ad_s:.4f}s", flush=True)

    # ── Measure T_FD (warm: JIT cache hit for each forward call) ─────────────
    print(f"  Measuring T_FD (central diff, 2p={2*p} forward evals)...", flush=True)
    # Run FD twice and take the second (ensures JIT warmup for all p directions)
    _, _ = _fd_gradient(_loss_jit, theta_np)   # warmup
    _, T_fd_s = _fd_gradient(_loss_jit, theta_np)
    print(f"  T_FD (central diff): {T_fd_s:.4f}s  (2p={2*p} calls)", flush=True)

    ratio = T_fd_s / T_ad_s
    ad_wins = ratio > 1.0
    T_ratio = T_ad_s / T_forward_s

    print(f"  T_FD / T_AD = {ratio:.2f}  | AD wins? {'YES' if ad_wins else 'NO'}", flush=True)
    print(f"  T_ratio = T_AD / T_forward = {T_ratio:.2f}", flush=True)

    results.append({
        "p":             p,
        "T_forward_s":   T_forward_s,
        "T_ad_s":        T_ad_s,
        "T_fd_s":        T_fd_s,
        "T_ratio":       T_ratio,
        "T_fd_over_ad":  ratio,
        "ad_wins":       ad_wins,
        "gpp_target":    gpp_target,
    })


# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("=== AD vs FD Scaling Summary ===", flush=True)
print(f"{'p':>4}  {'T_forward(s)':>14}  {'T_AD(s)':>10}  {'T_FD(s)':>10}  {'T_FD/T_AD':>10}  {'AD wins?':>9}", flush=True)
print("-" * 70, flush=True)
for r in results:
    wins_str = "YES" if r["ad_wins"] else "NO"
    print(f"  {r['p']:2d}  {r['T_forward_s']:14.4f}  {r['T_ad_s']:10.4f}  {r['T_fd_s']:10.4f}  {r['T_fd_over_ad']:10.2f}  {wins_str:>9}", flush=True)

# Effective T_ratio (mean across p values, should be ~constant)
t_ratios = [r["T_ratio"] for r in results]
print(f"\n  T_ratio (T_AD/T_forward) across p: {t_ratios}", flush=True)
print(f"  Mean T_ratio: {np.mean(t_ratios):.2f}", flush=True)
crossover_p = np.mean(t_ratios) / 2.0
print(f"  Theoretical crossover: p > {crossover_p:.1f}", flush=True)


# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = FIGURES_DIR / "ad_vs_fd_scaling.csv"
fieldnames = ["p", "T_forward_s", "T_ad_s", "T_fd_s", "T_ratio", "T_fd_over_ad", "ad_wins", "gpp_target"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
print(f"\n  CSV saved: {csv_path}", flush=True)


# ── Plot ──────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p_vals    = [r["p"] for r in results]
T_ad_vals = [r["T_ad_s"] for r in results]
T_fd_vals = [r["T_fd_s"] for r in results]
ratio_vals = [r["T_fd_over_ad"] for r in results]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: absolute times
x  = np.arange(len(p_vals))
w  = 0.35
ax = axes[0]
ax.bar(x - w/2, T_ad_vals, w, label="T_AD (jax.grad, 1 backward)", color="steelblue")
ax.bar(x + w/2, T_fd_vals, w, label="T_FD (central diff, 2p fwd)", color="coral", alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels([f"p={pv}" for pv in p_vals])
ax.set_ylabel("Wall time (s)")
ax.set_title("T_AD vs T_FD per gradient evaluation")
ax.legend(fontsize=9)
ax.set_xlabel("Number of calibration parameters p")

# Right: ratio T_FD / T_AD
ax2 = axes[1]
colors = ["green" if r > 1.0 else "red" for r in ratio_vals]
ax2.bar(x, ratio_vals, color=colors, alpha=0.85)
ax2.axhline(1.0, color="k", linestyle="--", lw=1.5, label="breakeven (T_FD = T_AD)")
ax2.set_xticks(x)
ax2.set_xticklabels([f"p={pv}" for pv in p_vals])
ax2.set_ylabel("T_FD / T_AD")
ax2.set_title("Speedup of AD over FD\n(ratio > 1 = AD wins, green)")
ax2.legend(fontsize=9)
ax2.set_xlabel("Number of calibration parameters p")

fig.suptitle(
    "CLM-ml-jax: Reverse-mode AD (O(1)) vs Finite Differences (O(2p))\n"
    "Scalar GPP loss — 5 physical parameters (alpha_sw, tref, vcmax, iota, dpai)",
    fontsize=11,
)
fig.tight_layout()
png_path = FIGURES_DIR / "ad_vs_fd_scaling.png"
fig.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Figure saved: {png_path}", flush=True)

print("\n=== ad_vs_fd_scaling.py complete ===", flush=True)
