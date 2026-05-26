"""
Experiment 4 (2-parameter): Gradient-based joint calibration of Vcmax25 and iota_SPA.

Demonstrates recovery of known physiological parameters:
  - vcmaxpft[7]  = 125.0 µmol CO2 m-2 s-1  (truth)
  - iota_SPA[7]  = 375.0 µmol CO2 mol-1 H2O (truth)

from a perturbed starting point:
  - vcmaxpft[7]  = 57.7 µmol CO2 m-2 s-1   (CLM PFT-7 default)
  - iota_SPA[7]  = 750.0 µmol CO2 mol-1 H2O (CLM PFT-7 default)

Calibration signal: synthetic GPP + LE observations generated at the truth.

Two methods:
  1. Adam in log-parameter space (gradient-based, 150 steps)
  2. Nelder-Mead baseline (gradient-free, budget = 300 forward evals)

Adam is expected to win in 2D because it has directional gradient information,
unlike the 1D alpha_sw case where Nelder-Mead can exploit the single-variable
bisection structure just as effectively.

Optimization in log-space: theta = [log(vcmax), log(iota)]
  theta_true = [log(125), log(375)]
  theta_init = [log(57.7), log(750)]

Loss: weighted relative squared error on GPP and LE:
  loss = 0.5 * ((GPP_pred - GPP_obs) / (|GPP_obs| + 1e-6))^2
       + 0.5 * ((LE_pred  - LE_obs ) / (|LE_obs|  + 1e-6))^2

Injection patterns (critical):
  - vcmaxpft:  vcmaxpft_jax explicit arg to MLCanopyFluxes (bypasses JIT cache)
  - iota_SPA:  _set_pftcon module-global mutation (JAX traces through jnp.asarray)

Usage (from project root):
    cd src && python ../diags/calibration_vcmax_iota.py

Output:
  - Console: per-step loss/parameter values plus final summary table
  - diags/figures/calibration_vcmax_iota_convergence.png: 2-panel figure
  - diags/figures/calibration_vcmax_iota_results.csv: final results table
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

# Ensure project root (parent of diags/) is on sys.path so 'diags' is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared init ───────────────────────────────────────────────────────────────
from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs,
    jax, jnp, MLCanopyFluxes,
    atm2lnd_inst, wateratm2lndbulk_inst,
    compute_gpp, compute_le,
)
import multilayer_canopy.MLpftconMod             as _MLpftconMod
import multilayer_canopy.MLLeafPhotosynthesisMod  as _LeafMod
import multilayer_canopy.MLCanopyNitrogenProfileMod as _NitroMod

_p    = grid.p
_ncan = grid.ncan

# Build kwargs without atm2lnd_inst/wateratm2lndbulk_inst so we keep them fixed
_mlcf_kwargs_no_atm = {k: v for k, v in _mlcf_kwargs.items()
                       if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst")}

# ── MLpftcon module injection helpers ─────────────────────────────────────────
# iota_SPA must be injected via module-global mutation so JAX traces through
# jnp.asarray(MLpftcon.iota_SPA) in MLLeafPhotosynthesisMod.
# vcmaxpft must be injected via the explicit vcmaxpft_jax arg to MLCanopyFluxes
# because CanopyNitrogenProfile is @jax.jit-cached and bakes vcmaxpft at compile time.

_orig_pftcon = _MLpftconMod.MLpftcon   # original MLpftcon_type instance


def _set_pftcon(new_inst):
    """Replace MLpftcon in all relevant module namespaces."""
    _MLpftconMod.MLpftcon = new_inst
    _LeafMod.MLpftcon     = new_inst
    _NitroMod.MLpftcon    = new_inst


def _restore_pftcon():
    """Restore MLpftcon to original in all relevant module namespaces."""
    _MLpftconMod.MLpftcon = _orig_pftcon
    _LeafMod.MLpftcon     = _orig_pftcon
    _NitroMod.MLpftcon    = _orig_pftcon


# ── Ground truth and starting point (PFT index 7 = broadleaf deciduous temperate) ──
VCMAX_TRUE   = 125.0    # µmol CO2 m-2 s-1
IOTA_TRUE    = 375.0    # µmol CO2 mol-1 H2O
VCMAX_INIT   = 57.7     # CLM PFT-7 default
IOTA_INIT    = 750.0    # CLM PFT-7 default
PFT_IDX      = 7        # CHATS7 PFT index

# Log-space true and initial theta vectors
theta_true = jnp.array([jnp.log(jnp.float64(VCMAX_TRUE)),
                         jnp.log(jnp.float64(IOTA_TRUE))], dtype=jnp.float64)
theta_init = jnp.array([jnp.log(jnp.float64(VCMAX_INIT)),
                         jnp.log(jnp.float64(IOTA_INIT))], dtype=jnp.float64)


# ── Joint forward function (returns GPP and LE) ───────────────────────────────
def forward_gpp_le(theta: jnp.ndarray):
    """
    Run model with vcmaxpft[7] = exp(theta[0]) and iota_SPA[7] = exp(theta[1]).
    Returns (GPP, LE) tuple of scalars.

    Injection pattern:
      - iota_SPA: module-global mutation via _set_pftcon so JAX traces through
                  jnp.asarray(MLpftcon.iota_SPA) inside MLLeafPhotosynthesisMod
      - vcmaxpft: explicit vcmaxpft_jax arg so JAX bypasses JIT cache and
                  gradient flows through CanopyNitrogenProfile
    """
    vcmax_val = jnp.exp(theta[0])
    iota_val  = jnp.exp(theta[1])

    # iota: update module-global pftcon (allows JAX to trace through it)
    new_pftcon = _orig_pftcon._replace(
        iota_SPA=_orig_pftcon.iota_SPA.at[PFT_IDX].set(iota_val)
    )
    _set_pftcon(new_pftcon)

    # vcmax: pass as explicit JAX arg (bypasses JIT cache)
    vcmaxpft_jax = _orig_pftcon.vcmaxpft.at[PFT_IDX].set(vcmax_val)

    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=atm2lnd_inst,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        vcmaxpft_jax=vcmaxpft_jax,
        **_mlcf_kwargs_no_atm,
    )
    _restore_pftcon()

    gpp = compute_gpp(inst, _p, _ncan)
    le  = compute_le(inst, _p, _ncan)
    return gpp, le


def forward_loss(theta: jnp.ndarray) -> jnp.ndarray:
    """
    Compute weighted relative MSE loss on GPP and LE.

    loss = 0.5 * ((GPP_pred - GPP_obs) / (|GPP_obs| + 1e-6))^2
         + 0.5 * ((LE_pred  - LE_obs ) / (|LE_obs|  + 1e-6))^2

    Args:
        theta: array [log(vcmax), log(iota)]
    Returns:
        scalar loss
    """
    gpp_pred, le_pred = forward_gpp_le(theta)
    loss_gpp = 0.5 * ((gpp_pred - obs_gpp) / (jnp.abs(obs_gpp) + 1e-6)) ** 2
    loss_le  = 0.5 * ((le_pred  - obs_le)  / (jnp.abs(obs_le)  + 1e-6)) ** 2
    return loss_gpp + loss_le


# ── Generate synthetic observations (at truth) ────────────────────────────────
print("\n=== Generating synthetic observations ===", flush=True)
print(f"  vcmaxpft[{PFT_IDX}] (truth) = {VCMAX_TRUE:.1f} µmol/m2/s", flush=True)
print(f"  iota_SPA[{PFT_IDX}] (truth) = {IOTA_TRUE:.1f} µmol CO2/mol H2O", flush=True)

t0 = time.time()
obs_gpp, obs_le = forward_gpp_le(theta_true)
jax.block_until_ready((obs_gpp, obs_le))
print(f"  Forward pass (truth) completed in {time.time() - t0:.2f}s", flush=True)
print(f"  GPP_obs = {float(obs_gpp):.4f} µmol CO2/m2/s", flush=True)
print(f"  LE_obs  = {float(obs_le):.4f} W/m2", flush=True)

# ── Sanity check: loss at truth ───────────────────────────────────────────────
loss_truth = float(forward_loss(theta_true))
loss_init  = float(forward_loss(theta_init))
print(f"\n  Loss at truth  [log({VCMAX_TRUE}), log({IOTA_TRUE})]: {loss_truth:.4e}  (should be ~0)", flush=True)
print(f"  Loss at init   [log({VCMAX_INIT}), log({IOTA_INIT})]: {loss_init:.4e}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Method 1: Adam optimizer in log-parameter space (gradient-based)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("=== Method 1: Adam optimizer (gradient-based, 150 steps) ===", flush=True)
print("=" * 60, flush=True)

# Adam hyperparameters
lr    = 0.05
beta1 = 0.9
beta2 = 0.999
eps   = 1e-8
N_ADAM_STEPS = 150

grad_fn = jax.jit(jax.grad(forward_loss))

# Warm up JIT (not counted in per-step timing)
print("  Warming up JIT compilation...", flush=True)
t_jit_start = time.time()
_ = grad_fn(theta_init)
jax.block_until_ready(_)
t_jit = time.time() - t_jit_start
print(f"  JIT compile time: {t_jit:.2f}s", flush=True)

# history_adam entries: (n_evals, loss, vcmax_val, iota_val)
history_adam = []

theta     = theta_init
m         = jnp.zeros_like(theta)
v         = jnp.zeros_like(theta)
t_adam    = 0
n_evals_adam = 0

t_adam_start = time.time()
for step in range(N_ADAM_STEPS):
    t_step = time.time()

    # Gradient step: ~2 forward passes
    g = grad_fn(theta)
    jax.block_until_ready(g)
    t_adam += 1
    n_evals_adam += 2

    # Adam update
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g ** 2
    m_hat = m / (1.0 - beta1 ** t_adam)
    v_hat = v / (1.0 - beta2 ** t_adam)
    theta = theta - lr * m_hat / (jnp.sqrt(v_hat) + eps)

    # Compute loss for logging: 1 additional forward pass
    l = float(forward_loss(theta))
    n_evals_adam += 1
    t_elapsed = time.time() - t_step

    vcmax_cur = float(jnp.exp(theta[0]))
    iota_cur  = float(jnp.exp(theta[1]))

    history_adam.append((n_evals_adam, l, vcmax_cur, iota_cur))
    print(
        f"  Adam step {step + 1:3d}: "
        f"vcmax={vcmax_cur:7.2f}  iota={iota_cur:7.1f}  "
        f"loss={l:.4e}  |g|={float(jnp.linalg.norm(g)):.3e}  "
        f"({t_elapsed:.2f}s/step)",
        flush=True,
    )

t_adam_total = time.time() - t_adam_start
vcmax_adam_final = float(jnp.exp(theta[0]))
iota_adam_final  = float(jnp.exp(theta[1]))
loss_adam_final  = float(forward_loss(theta))

err_vcmax_adam = abs(vcmax_adam_final - VCMAX_TRUE) / VCMAX_TRUE
err_iota_adam  = abs(iota_adam_final  - IOTA_TRUE)  / IOTA_TRUE

print(f"\n  Adam finished in {t_adam_total:.1f}s total ({n_evals_adam} evals)", flush=True)
print(f"  vcmax_final = {vcmax_adam_final:.4f}  (true={VCMAX_TRUE}, rel_err={err_vcmax_adam:.4f})", flush=True)
print(f"  iota_final  = {iota_adam_final:.4f}  (true={IOTA_TRUE},  rel_err={err_iota_adam:.4f})", flush=True)
print(f"  Final loss  = {loss_adam_final:.4e}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Method 2: Nelder-Mead (gradient-free baseline)
# Budget = 300 forward evals (~= 150 Adam steps × 2 fwd passes each)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("=== Method 2: Nelder-Mead baseline (gradient-free, 300 evals) ===", flush=True)
print("=" * 60, flush=True)

from scipy.optimize import minimize

n_evals_nm = [0]
history_nm = []   # list of (n_evals, loss, vcmax_val, iota_val)


def loss_np(x):
    """Wrapper for scipy: x = [log(vcmax), log(iota)], returns Python float."""
    n_evals_nm[0] += 1
    theta_np = jnp.array(x, dtype=jnp.float64)
    l = float(forward_loss(theta_np))
    vcmax_np = float(np.exp(x[0]))
    iota_np  = float(np.exp(x[1]))
    history_nm.append((n_evals_nm[0], l, vcmax_np, iota_np))
    print(
        f"  NM eval {n_evals_nm[0]:3d}: "
        f"vcmax={vcmax_np:7.2f}  iota={iota_np:7.1f}  loss={l:.4e}",
        flush=True,
    )
    return l


t_nm_start = time.time()
result = minimize(
    loss_np,
    x0=np.array([np.log(VCMAX_INIT), np.log(IOTA_INIT)]),
    method="Nelder-Mead",
    options={
        "maxiter": 300,
        "xatol": 1e-4,
        "fatol": 1e-6,
        "adaptive": True,   # adaptive Nelder-Mead — better for 2D
    },
)
t_nm_total = time.time() - t_nm_start

vcmax_nm_final = float(np.exp(result.x[0]))
iota_nm_final  = float(np.exp(result.x[1]))
loss_nm_final  = float(result.fun)
err_vcmax_nm   = abs(vcmax_nm_final - VCMAX_TRUE) / VCMAX_TRUE
err_iota_nm    = abs(iota_nm_final  - IOTA_TRUE)  / IOTA_TRUE

print(f"\n  Nelder-Mead finished in {t_nm_total:.1f}s total", flush=True)
print(f"  Converged: {result.success}  |  {result.message}", flush=True)
print(f"  vcmax_final = {vcmax_nm_final:.4f}  (true={VCMAX_TRUE}, rel_err={err_vcmax_nm:.4f})", flush=True)
print(f"  iota_final  = {iota_nm_final:.4f}  (true={IOTA_TRUE},  rel_err={err_iota_nm:.4f})", flush=True)
print(f"  Final loss  = {loss_nm_final:.4e}", flush=True)
print(f"  Function evaluations: {n_evals_nm[0]}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Results summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("=== Final Results Summary ===", flush=True)
print("=" * 70, flush=True)
hdr = f"  {'Method':<26} {'vcmax_final':>11} {'err_vcmax':>9} {'iota_final':>11} {'err_iota':>9} {'final_loss':>12} {'time(s)':>8}"
sep = "  " + "-" * 68
print(hdr, flush=True)
print(sep, flush=True)
print(
    f"  {'Adam (gradient-based)':<26} "
    f"{vcmax_adam_final:>11.3f} {err_vcmax_adam:>9.4f} "
    f"{iota_adam_final:>11.3f} {err_iota_adam:>9.4f} "
    f"{loss_adam_final:>12.4e} {t_adam_total:>8.1f}",
    flush=True,
)
print(
    f"  {'Nelder-Mead (grad-free)':<26} "
    f"{vcmax_nm_final:>11.3f} {err_vcmax_nm:>9.4f} "
    f"{iota_nm_final:>11.3f} {err_iota_nm:>9.4f} "
    f"{loss_nm_final:>12.4e} {t_nm_total:>8.1f}",
    flush=True,
)
print(
    f"\n  Truth:  vcmax={VCMAX_TRUE}  iota={IOTA_TRUE}",
    flush=True,
)
print(
    f"  Init:   vcmax={VCMAX_INIT}  iota={IOTA_INIT}",
    flush=True,
)
print(
    f"  Timing: Adam JIT compile = {t_jit:.2f}s | "
    f"{t_adam_total / N_ADAM_STEPS:.2f}s/step (amortized over {N_ADAM_STEPS} steps)",
    flush=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Save CSV
# ─────────────────────────────────────────────────────────────────────────────
csv_path = FIGURES_DIR / "calibration_vcmax_iota_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "method",
        "vcmax_true", "vcmax_init", "vcmax_final", "rel_err_vcmax",
        "iota_true",  "iota_init",  "iota_final",  "rel_err_iota",
        "final_loss", "n_evaluations", "wall_time_s",
    ])
    writer.writerow([
        "Adam",
        VCMAX_TRUE, VCMAX_INIT, f"{vcmax_adam_final:.4f}", f"{err_vcmax_adam:.6f}",
        IOTA_TRUE,  IOTA_INIT,  f"{iota_adam_final:.4f}",  f"{err_iota_adam:.6f}",
        f"{loss_adam_final:.4e}", n_evals_adam, f"{t_adam_total:.2f}",
    ])
    writer.writerow([
        "Nelder-Mead",
        VCMAX_TRUE, VCMAX_INIT, f"{vcmax_nm_final:.4f}", f"{err_vcmax_nm:.6f}",
        IOTA_TRUE,  IOTA_INIT,  f"{iota_nm_final:.4f}",  f"{err_iota_nm:.6f}",
        f"{loss_nm_final:.4e}", n_evals_nm[0], f"{t_nm_total:.2f}",
    ])
print(f"\nResults CSV saved: {csv_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Convergence figure
# ─────────────────────────────────────────────────────────────────────────────
# Unpack histories
if history_adam:
    adam_evals, adam_losses, adam_vcmax, adam_iota = zip(*history_adam)
else:
    adam_evals, adam_losses, adam_vcmax, adam_iota = [], [], [], []

if history_nm:
    nm_evals, nm_losses, nm_vcmax, nm_iota = zip(*history_nm)
else:
    nm_evals, nm_losses, nm_vcmax, nm_iota = [], [], [], []

adam_evals  = np.array(adam_evals,  dtype=float)
adam_losses = np.array(adam_losses, dtype=float)
adam_vcmax  = np.array(adam_vcmax,  dtype=float)
adam_iota   = np.array(adam_iota,   dtype=float)
nm_evals    = np.array(nm_evals,    dtype=float)
nm_losses   = np.array(nm_losses,   dtype=float)
nm_vcmax    = np.array(nm_vcmax,    dtype=float)
nm_iota     = np.array(nm_iota,     dtype=float)

# Normalize parameters by truth value for trajectory plot
adam_vcmax_norm = adam_vcmax / VCMAX_TRUE
adam_iota_norm  = adam_iota  / IOTA_TRUE
nm_vcmax_norm   = nm_vcmax   / VCMAX_TRUE
nm_iota_norm    = nm_iota    / IOTA_TRUE

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── Left panel: loss vs evaluations ─────────────────────────────────────────
ax = axes[0]
if len(adam_evals) > 0:
    ax.semilogy(adam_evals, adam_losses, color="steelblue", lw=2,
                label="Adam (gradient-based)", marker="o", markersize=3)
if len(nm_evals) > 0:
    ax.semilogy(nm_evals, nm_losses, color="darkorange", lw=2,
                label="Nelder-Mead (gradient-free)", marker="s", markersize=3)
ax.set_xlabel("Number of forward evaluations", fontsize=12)
ax.set_ylabel("Weighted relative MSE loss (log scale)", fontsize=12)
ax.set_title("Convergence: loss vs evaluations", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)
ax.set_xlim(left=0)

if len(adam_evals) > 0:
    ax.annotate(
        f"Adam: {loss_adam_final:.2e}",
        xy=(adam_evals[-1], adam_losses[-1]),
        xytext=(0.55, 0.65), textcoords="axes fraction",
        fontsize=9, color="steelblue",
        arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.2),
    )
if len(nm_evals) > 0:
    ax.annotate(
        f"NM: {loss_nm_final:.2e}",
        xy=(nm_evals[-1], nm_losses[-1]),
        xytext=(0.55, 0.45), textcoords="axes fraction",
        fontsize=9, color="darkorange",
        arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.2),
    )

# ── Right panel: parameter trajectory (normalized by truth) ─────────────────
ax = axes[1]
# Adam: vcmax and iota (both on normalized scale)
if len(adam_evals) > 0:
    ax.plot(adam_evals, adam_vcmax_norm, color="steelblue", lw=2,
            linestyle="-", marker="o", markersize=3,
            label="Adam — Vcmax25/Vcmax25_true")
    ax.plot(adam_evals, adam_iota_norm, color="royalblue", lw=2,
            linestyle="--", marker="o", markersize=3,
            label="Adam — iota/iota_true")
# Nelder-Mead
if len(nm_evals) > 0:
    ax.plot(nm_evals, nm_vcmax_norm, color="darkorange", lw=2,
            linestyle="-", marker="s", markersize=3,
            label="NM — Vcmax25/Vcmax25_true")
    ax.plot(nm_evals, nm_iota_norm, color="saddlebrown", lw=2,
            linestyle="--", marker="s", markersize=3,
            label="NM — iota/iota_true")
# Target line
ax.axhline(y=1.0, color="black", linestyle="--", lw=1.5, label="True value (= 1.0 norm.)")
ax.set_xlabel("Number of forward evaluations", fontsize=12)
ax.set_ylabel("Parameter value / True value", fontsize=12)
ax.set_title("Parameter trajectory vs evaluations\n(normalized by truth, both params on same scale)", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)

fig.suptitle(
    "CLM-ml-jax Exp 4: Joint calibration of Vcmax25 + iota_SPA via GPP+LE — CHATS7, May 1 2007\n"
    f"Adam (150 steps, grad-based) vs Nelder-Mead ({n_evals_nm[0]} evals, grad-free)\n"
    f"Truth: vcmax={VCMAX_TRUE}, iota={IOTA_TRUE}   Init: vcmax={VCMAX_INIT}, iota={IOTA_INIT}",
    fontsize=10,
)
fig.tight_layout()

fig_path = FIGURES_DIR / "calibration_vcmax_iota_convergence.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved: {fig_path}", flush=True)

print("\n=== calibration_vcmax_iota.py complete ===", flush=True)
