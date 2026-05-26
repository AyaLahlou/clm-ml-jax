"""
Experiment 4 — NM-only rerun.

Adam already finished (150 steps, results hardcoded from logs/7527834_calibration_vcmax_iota.out).
This script runs Nelder-Mead only (maxiter=80) and generates the comparison figure.

Usage (from project root):
    python diags/calibration_nm_only.py
Output:
    diags/figures/calibration_vcmax_iota_convergence.png
    diags/figures/calibration_vcmax_iota_results.csv
"""
from __future__ import annotations

import csv
import sys
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
_mlcf_kwargs_no_atm = {k: v for k, v in _mlcf_kwargs.items()
                       if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst")}

VCMAX_TRUE = 125.0
IOTA_TRUE  = 375.0
VCMAX_INIT = 57.7
IOTA_INIT  = 750.0
PFT_IDX    = 7

# ── Hardcoded Adam history from job 7527834 ────────────────────────────────
# Format: (n_evals, loss, vcmax, iota)  —  n_evals = step * 3
_adam_raw = [
    (1,60.66,713.4,1.1797e-01),(2,63.76,678.7,1.0224e-01),(3,67.02,645.8,8.7529e-02),
    (4,70.42,614.7,7.3897e-02),(5,73.97,585.3,6.1402e-02),(6,77.67,557.6,5.0091e-02),
    (7,81.51,531.7,3.9997e-02),(8,85.47,507.4,3.1141e-02),(9,89.54,484.7,2.3521e-02),
    (10,93.70,463.7,1.7118e-02),(11,97.93,444.2,1.1887e-02),(12,102.18,426.3,7.7611e-03),
    (13,106.43,409.9,4.6527e-03),(14,110.63,395.0,2.4532e-03),(15,114.74,381.6,1.0398e-03),
    (16,118.70,369.5,2.8056e-04),(17,122.48,358.8,4.1382e-05),(18,126.02,349.4,1.9241e-04),
    (19,129.29,341.2,6.1392e-04),(20,132.25,334.2,1.2006e-03),(21,134.88,328.3,1.8640e-03),
    (22,137.16,323.4,2.5337e-03),(23,139.08,319.5,3.1560e-03),(24,140.64,316.5,3.6933e-03),
    (25,141.85,314.4,4.1210e-03),(26,142.71,313.1,4.4259e-03),(27,143.24,312.5,4.6036e-03),
    (28,143.46,312.6,4.6569e-03),(29,143.39,313.4,4.5937e-03),(30,143.06,314.7,4.4262e-03),
    (31,142.50,316.6,4.1694e-03),(32,141.73,318.9,3.8404e-03),(33,140.78,321.7,3.4572e-03),
    (34,139.68,324.8,3.0385e-03),(35,138.46,328.3,2.6026e-03),(36,137.14,332.1,2.1668e-03),
    (37,135.76,336.0,1.7470e-03),(38,134.33,340.1,1.3569e-03),(39,132.89,344.3,1.0076e-03),
    (40,131.45,348.6,7.0719e-04),(41,130.04,352.9,4.6071e-04),(42,128.68,357.1,2.7003e-04),
    (43,127.38,361.2,1.3400e-04),(44,126.16,365.1,4.8783e-05),(45,125.03,368.8,8.3458e-06),
    (46,124.01,372.3,5.0077e-06),(47,123.09,375.4,3.0101e-05),(48,122.29,378.2,7.4622e-05),
    (49,121.61,380.7,1.2985e-04),(50,121.05,382.9,1.8787e-04),(51,120.61,384.6,2.4201e-04),
    (52,120.30,386.0,2.8711e-04),(53,120.09,387.0,3.1965e-04),(54,120.00,387.7,3.3777e-04),
    (55,120.01,388.0,3.4111e-04),(56,120.11,388.0,3.3063e-04),(57,120.29,387.7,3.0830e-04),
    (58,120.56,387.2,2.7675e-04),(59,120.88,386.5,2.3901e-04),(60,121.26,385.6,1.9816e-04),
    (61,121.68,384.6,1.5707e-04),(62,122.13,383.4,1.1825e-04),(63,122.60,382.2,8.3629e-05),
    (64,123.07,381.0,5.4576e-05),(65,123.55,379.7,3.1824e-05),(66,124.01,378.5,1.5541e-05),
    (67,124.46,377.4,5.4000e-06),(68,124.88,376.3,6.8925e-07),(69,125.26,375.3,4.2917e-07),
    (70,125.61,374.4,3.4931e-06),(71,125.91,373.6,8.7190e-06),(72,126.16,372.9,1.5004e-05),
    (73,126.37,372.4,2.1382e-05),(74,126.54,371.9,2.7071e-05),(75,126.65,371.6,3.1505e-05),
    (76,126.72,371.5,3.4339e-05),(77,126.75,371.4,3.5441e-05),(78,126.73,371.4,3.4858e-05),
    (79,126.69,371.6,3.2790e-05),(80,126.60,371.8,2.9539e-05),(81,126.50,372.1,2.5468e-05),
    (82,126.37,372.4,2.0962e-05),(83,126.22,372.8,1.6388e-05),(84,126.06,373.2,1.2066e-05),
    (85,125.89,373.7,8.2483e-06),(86,125.72,374.1,5.1093e-06),(87,125.55,374.6,2.7405e-06),
    (88,125.39,375.0,1.1560e-06),(89,125.23,375.4,3.0305e-07),(90,125.09,375.8,7.7153e-08),
    (91,124.96,376.1,3.3946e-07),(92,124.84,376.4,9.3461e-07),(93,124.74,376.6,1.7073e-06),
    (94,124.66,376.8,2.5163e-06),(95,124.60,377.0,3.2449e-06),(96,124.55,377.0,3.8074e-06),
    (97,124.53,377.1,4.1519e-06),(98,124.51,377.1,4.2591e-06),(99,124.52,377.0,4.1383e-06),
    (100,124.53,376.9,3.8217e-06),(101,124.56,376.8,3.3569e-06),(102,124.60,376.6,2.7996e-06),
    (103,124.65,376.5,2.2063e-06),(104,124.70,376.3,1.6289e-06),(105,124.76,376.1,1.1096e-06),
    (106,124.81,375.9,6.7893e-07),(107,124.87,375.7,3.5392e-07),(108,124.93,375.5,1.3913e-07),
    (109,124.98,375.3,2.7999e-08),(110,125.03,375.1,5.4468e-09),(111,125.08,375.0,5.0761e-08),
    (112,125.12,374.8,1.4063e-07),(113,125.15,374.7,2.5191e-07),(114,125.17,374.6,3.6391e-07),
    (115,125.19,374.6,4.6007e-07),(116,125.20,374.5,5.2885e-07),(117,125.21,374.5,5.6409e-07),
    (118,125.21,374.5,5.6463e-07),(119,125.20,374.5,5.3351e-07),(120,125.19,374.5,4.7695e-07),
    (121,125.17,374.5,4.0304e-07),(122,125.15,374.6,3.2057e-07),(123,125.13,374.6,2.3790e-07),
    (124,125.11,374.7,1.6218e-07),(125,125.08,374.7,9.8687e-08),(126,125.06,374.8,5.0613e-08),
    (127,125.03,374.9,1.9045e-08),(128,125.01,374.9,3.1951e-09),(129,124.99,375.0,8.0218e-10),
    (130,124.97,375.0,8.6258e-09),(131,124.95,375.1,2.2965e-08),(132,124.94,375.1,4.0141e-08),
    (133,124.93,375.1,5.6899e-08),(134,124.92,375.1,7.0687e-08),(135,124.92,375.2,7.9819e-08),
    (136,124.91,375.2,8.3504e-08),(137,124.92,375.2,8.1776e-08),(138,124.92,375.2,7.5336e-08),
    (139,124.92,375.1,6.5353e-08),(140,124.93,375.1,5.3239e-08),(141,124.94,375.1,4.0440e-08),
    (142,124.95,375.1,2.8257e-08),(143,124.96,375.1,1.7720e-08),(144,124.97,375.0,9.5047e-09),
    (145,124.98,375.0,3.9204e-09),(146,124.99,375.0,9.3131e-10),(147,125.00,375.0,2.2082e-10),
    (148,125.00,374.9,1.2766e-09),(149,125.01,374.9,3.4844e-09),(150,125.02,374.9,6.2181e-09),
]

# Convert step index to cumulative eval count (3 evals per step)
history_adam = [(step * 3, loss, vcmax, iota)
                for step, vcmax, iota, loss in _adam_raw]

vcmax_adam_final = 125.0167
iota_adam_final  = 374.9092
loss_adam_final  = 6.2181e-09
t_adam_total     = 51124.5
n_evals_adam     = 450
N_ADAM_STEPS     = 150

err_vcmax_adam = abs(vcmax_adam_final - VCMAX_TRUE) / VCMAX_TRUE
err_iota_adam  = abs(iota_adam_final  - IOTA_TRUE)  / IOTA_TRUE

print(f"Adam history loaded: {len(history_adam)} steps  (final loss={loss_adam_final:.2e})", flush=True)

# ── MLpftcon injection helpers ─────────────────────────────────────────────
_orig_pftcon = _MLpftconMod.MLpftcon


def _set_pftcon(new_inst):
    _MLpftconMod.MLpftcon = new_inst
    _LeafMod.MLpftcon     = new_inst
    _NitroMod.MLpftcon    = new_inst


def _restore_pftcon():
    _MLpftconMod.MLpftcon = _orig_pftcon
    _LeafMod.MLpftcon     = _orig_pftcon
    _NitroMod.MLpftcon    = _orig_pftcon


def forward_gpp_le(theta: jnp.ndarray):
    vcmax_val = jnp.exp(theta[0])
    iota_val  = jnp.exp(theta[1])
    new_pftcon = _orig_pftcon._replace(
        iota_SPA=_orig_pftcon.iota_SPA.at[PFT_IDX].set(iota_val)
    )
    _set_pftcon(new_pftcon)
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


theta_true = jnp.array([jnp.log(jnp.float64(VCMAX_TRUE)),
                         jnp.log(jnp.float64(IOTA_TRUE))], dtype=jnp.float64)

print("\n=== Generating synthetic observations ===", flush=True)
t0 = time.time()
obs_gpp, obs_le = forward_gpp_le(theta_true)
jax.block_until_ready((obs_gpp, obs_le))
print(f"  Warmup forward pass: {time.time()-t0:.2f}s", flush=True)
print(f"  GPP_obs={float(obs_gpp):.4f}  LE_obs={float(obs_le):.4f}", flush=True)


def forward_loss(theta: jnp.ndarray) -> jnp.ndarray:
    gpp_pred, le_pred = forward_gpp_le(theta)
    loss_gpp = 0.5 * ((gpp_pred - obs_gpp) / (jnp.abs(obs_gpp) + 1e-6)) ** 2
    loss_le  = 0.5 * ((le_pred  - obs_le)  / (jnp.abs(obs_le)  + 1e-6)) ** 2
    return loss_gpp + loss_le


# ── Nelder-Mead (gradient-free baseline) ──────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("=== Nelder-Mead baseline (gradient-free, maxiter=80) ===", flush=True)
print("=" * 60, flush=True)

from scipy.optimize import minimize

n_evals_nm = [0]
history_nm = []


def loss_np(x):
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
        "maxiter": 80,
        "xatol": 1e-3,
        "fatol": 1e-4,
        "adaptive": True,
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
print(f"  vcmax_final={vcmax_nm_final:.4f}  (true={VCMAX_TRUE}, rel_err={err_vcmax_nm:.4f})", flush=True)
print(f"  iota_final ={iota_nm_final:.4f}  (true={IOTA_TRUE},  rel_err={err_iota_nm:.4f})", flush=True)
print(f"  Final loss = {loss_nm_final:.4e}", flush=True)
print(f"  Evaluations: {n_evals_nm[0]}", flush=True)

# ── Results summary ────────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("=== Final Results Summary ===", flush=True)
hdr = f"  {'Method':<26} {'vcmax_final':>11} {'err_vcmax':>9} {'iota_final':>11} {'err_iota':>9} {'final_loss':>12}"
print(hdr, flush=True)
print("  " + "-" * 68, flush=True)
print(f"  {'Adam (gradient-based)':<26} {vcmax_adam_final:>11.3f} {err_vcmax_adam:>9.4f} "
      f"{iota_adam_final:>11.3f} {err_iota_adam:>9.4f} {loss_adam_final:>12.4e}", flush=True)
print(f"  {'Nelder-Mead (grad-free)':<26} {vcmax_nm_final:>11.3f} {err_vcmax_nm:>9.4f} "
      f"{iota_nm_final:>11.3f} {err_iota_nm:>9.4f} {loss_nm_final:>12.4e}", flush=True)
print(f"\n  Truth:  vcmax={VCMAX_TRUE}  iota={IOTA_TRUE}", flush=True)
print(f"  Init:   vcmax={VCMAX_INIT}  iota={IOTA_INIT}", flush=True)

# ── Save CSV ───────────────────────────────────────────────────────────────
csv_path = FIGURES_DIR / "calibration_vcmax_iota_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["method","vcmax_true","vcmax_init","vcmax_final","rel_err_vcmax",
                     "iota_true","iota_init","iota_final","rel_err_iota","final_loss","n_evaluations","wall_time_s"])
    writer.writerow(["Adam", VCMAX_TRUE, VCMAX_INIT, f"{vcmax_adam_final:.4f}", f"{err_vcmax_adam:.6f}",
                     IOTA_TRUE, IOTA_INIT, f"{iota_adam_final:.4f}", f"{err_iota_adam:.6f}",
                     f"{loss_adam_final:.4e}", n_evals_adam, f"{t_adam_total:.2f}"])
    writer.writerow(["Nelder-Mead", VCMAX_TRUE, VCMAX_INIT, f"{vcmax_nm_final:.4f}", f"{err_vcmax_nm:.6f}",
                     IOTA_TRUE, IOTA_INIT, f"{iota_nm_final:.4f}", f"{err_iota_nm:.6f}",
                     f"{loss_nm_final:.4e}", n_evals_nm[0], f"{t_nm_total:.2f}"])
print(f"\nCSV saved: {csv_path}", flush=True)

# ── Convergence figure ─────────────────────────────────────────────────────
adam_steps, adam_losses, adam_vcmax, adam_iota = zip(*history_adam)
adam_evals  = np.array(adam_steps,  dtype=float)
adam_losses = np.array(adam_losses, dtype=float)
adam_vcmax  = np.array(adam_vcmax,  dtype=float)
adam_iota   = np.array(adam_iota,   dtype=float)

nm_evals_arr = np.array([h[0] for h in history_nm], dtype=float)
nm_losses_arr = np.array([h[1] for h in history_nm], dtype=float)
nm_vcmax_arr  = np.array([h[2] for h in history_nm], dtype=float)
nm_iota_arr   = np.array([h[3] for h in history_nm], dtype=float)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.semilogy(adam_evals, adam_losses, color="steelblue", lw=2,
            label="Adam (gradient-based)", marker="o", markersize=3)
if len(nm_evals_arr) > 0:
    ax.semilogy(nm_evals_arr, nm_losses_arr, color="darkorange", lw=2,
                label="Nelder-Mead (gradient-free)", marker="s", markersize=3)
ax.set_xlabel("Number of forward evaluations", fontsize=12)
ax.set_ylabel("Weighted relative MSE loss (log scale)", fontsize=12)
ax.set_title("Convergence: loss vs evaluations", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)
ax.set_xlim(left=0)
ax.annotate(f"Adam: {loss_adam_final:.2e}",
            xy=(adam_evals[-1], adam_losses[-1]),
            xytext=(0.55, 0.65), textcoords="axes fraction", fontsize=9, color="steelblue",
            arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.2))
if len(nm_evals_arr) > 0:
    ax.annotate(f"NM: {loss_nm_final:.2e}",
                xy=(nm_evals_arr[-1], nm_losses_arr[-1]),
                xytext=(0.55, 0.45), textcoords="axes fraction", fontsize=9, color="darkorange",
                arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.2))

ax = axes[1]
ax.plot(adam_evals, adam_vcmax / VCMAX_TRUE, color="steelblue", lw=2,
        linestyle="-", marker="o", markersize=3, label="Adam — Vcmax25")
ax.plot(adam_evals, adam_iota / IOTA_TRUE, color="royalblue", lw=2,
        linestyle="--", marker="o", markersize=3, label="Adam — iota")
if len(nm_evals_arr) > 0:
    ax.plot(nm_evals_arr, nm_vcmax_arr / VCMAX_TRUE, color="darkorange", lw=2,
            linestyle="-", marker="s", markersize=3, label="NM — Vcmax25")
    ax.plot(nm_evals_arr, nm_iota_arr / IOTA_TRUE, color="saddlebrown", lw=2,
            linestyle="--", marker="s", markersize=3, label="NM — iota")
ax.axhline(y=1.0, color="black", linestyle="--", lw=1.5, label="True value (= 1.0)")
ax.set_xlabel("Number of forward evaluations", fontsize=12)
ax.set_ylabel("Parameter value / True value", fontsize=12)
ax.set_title("Parameter trajectory vs evaluations\n(normalized by truth)", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)

fig.suptitle(
    "CLM-ml-jax Exp 4: Joint calibration of Vcmax25 + iota_SPA via GPP+LE — CHATS7, May 1 2007\n"
    f"Adam (150 steps, grad-based, {n_evals_adam} evals) vs "
    f"Nelder-Mead ({n_evals_nm[0]} evals, grad-free)\n"
    f"Truth: vcmax={VCMAX_TRUE}, iota={IOTA_TRUE}   Init: vcmax={VCMAX_INIT}, iota={IOTA_INIT}",
    fontsize=10,
)
fig.tight_layout()

fig_path = FIGURES_DIR / "calibration_vcmax_iota_convergence.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved: {fig_path}", flush=True)

print("\n=== calibration_nm_only.py complete ===", flush=True)
