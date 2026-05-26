"""
Multi-parameter calibration experiment (p=10): single-step loss variant
with Tikhonov regularization to break equifinality.

Identical setup to multipar_calibration.py but uses only a single-step loss
(no multi-step JIT compilation).  The multi-step backward JIT takes >2h to
compile (it unrolls 8 forward passes into one XLA graph).  This script:

  1. Pre-warms only the single-step forward and backward (already ~30 min).
  2. Runs all four optimizer methods against the single-step loss.
  3. Finishes within ~1-2h total (vs 24h needed for multi-step).

Purpose: confirm Tikhonov regularization breaks the equifinality of the
single-step loss.  Without regularization, all optimizers reach loss≈0 at
parameter vectors far from theta_star (‖Δθ‖≈0.5–0.9).  With
L_reg = LAMBDA_REG * ‖theta - 1‖^2, the unique minimum is at theta_star.

Parameter set and forward function are identical to multipar_calibration.py.

Outputs:
  diags/output/multipar_calibration_singlestep_tikhonov_results.json
"""
from __future__ import annotations

import json
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
from scipy.optimize import minimize as _sp_minimize

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA_REG = 0.1  # Tikhonov weight: selects theta=theta_star among equifinal solutions

# ── Shared init ───────────────────────────────────────────────────────────────
print(f"=== multipar_calibration_singlestep.py (TIKHONOV lambda={LAMBDA_REG}): loading model ===",
      flush=True)
from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs,
    MLCanopyFluxes,
    atm2lnd_inst, wateratm2lndbulk_inst,
    compute_gpp, compute_h, compute_le,
)
import multilayer_canopy.MLpftconMod             as _MLpftconMod
import multilayer_canopy.MLLeafPhotosynthesisMod  as _LeafMod
import multilayer_canopy.MLCanopyNitrogenProfileMod as _NitroMod

_p    = grid.p
_ncan = grid.ncan

_mlcf_kwargs_no_atm = {k: v for k, v in _mlcf_kwargs.items()
                       if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst")}

_orig_pftcon = _MLpftconMod.MLpftcon

P = 10  # number of parameters — all active

# ── pftcon injection helpers ──────────────────────────────────────────────────

def _set_pftcon(new_inst):
    _MLpftconMod.MLpftcon = new_inst
    _LeafMod.MLpftcon     = new_inst
    _NitroMod.MLpftcon    = new_inst


def _restore_pftcon():
    _MLpftconMod.MLpftcon = _orig_pftcon
    _LeafMod.MLpftcon     = _orig_pftcon
    _NitroMod.MLpftcon    = _orig_pftcon


# ── SW waveband helpers ───────────────────────────────────────────────────────

def _scale_solad(arr, theta_vis, theta_nir):
    return arr.at[:, 1].mul(theta_vis).at[:, 2].mul(theta_nir)


def _scale_solai(arr, theta_vis, theta_nir):
    return arr.at[:, 1].mul(theta_vis).at[:, 2].mul(theta_nir)


# ── Per-timestep forward function ─────────────────────────────────────────────

def forward_theta(theta: jnp.ndarray):
    """p=10 single-step forward at the initialization timestep."""
    modified_atm = atm2lnd_inst._replace(
        forc_solad_downscaled_col = _scale_solad(
            atm2lnd_inst.forc_solad_downscaled_col, theta[0], theta[1]),
        forc_solai_grc            = _scale_solai(
            atm2lnd_inst.forc_solai_grc, theta[2], theta[3]),
        forc_t_downscaled_col     = theta[4] * atm2lnd_inst.forc_t_downscaled_col,
        forc_pbot_downscaled_col  = theta[8] * atm2lnd_inst.forc_pbot_downscaled_col,
        forc_u_grc                = theta[9] * atm2lnd_inst.forc_u_grc,
    )
    modified_wat = wateratm2lndbulk_inst._replace(
        forc_q_downscaled_col = theta[7] * wateratm2lndbulk_inst.forc_q_downscaled_col,
    )

    _set_pftcon(_orig_pftcon._replace(
        iota_SPA = theta[6] * _orig_pftcon.iota_SPA,
    ))

    vcmaxpft_jax = theta[5] * _orig_pftcon.vcmaxpft

    inst = MLCanopyFluxes(
        mlcanopy_inst         = mlcanopy_inst,
        atm2lnd_inst          = modified_atm,
        wateratm2lndbulk_inst = modified_wat,
        vcmaxpft_jax          = vcmaxpft_jax,
        **_mlcf_kwargs_no_atm,
    )
    _restore_pftcon()

    gpp = compute_gpp(inst, _p, _ncan)
    h   = compute_h(inst, _p, _ncan)
    le  = compute_le(inst, _p, _ncan)
    return gpp, h, le


# ── Generate single-step target ───────────────────────────────────────────────
theta_star = jnp.ones(P, dtype=jnp.float64)

print("\n=== Generating single-step target (theta_star = ones) ===", flush=True)
t0 = time.time()
GPP_target, H_target, LE_target = forward_theta(theta_star)
jax.block_until_ready((GPP_target, H_target, LE_target))
print(f"  GPP={float(GPP_target):.4f}  H={float(H_target):.4f}"
      f"  LE={float(LE_target):.4f}", flush=True)
print(f"  Target generation: {time.time()-t0:.2f}s", flush=True)

_EPS_NORM = 1e-6


# ── Loss function (single-step, Tikhonov-regularized) ────────────────────────

def loss_fn(theta: jnp.ndarray) -> jnp.ndarray:
    gpp, h, le = forward_theta(theta)
    data_loss = (
        ((gpp - GPP_target) / (jnp.abs(GPP_target) + _EPS_NORM)) ** 2
        + ((h   - H_target)   / (jnp.abs(H_target)   + _EPS_NORM)) ** 2
        + ((le  - LE_target)  / (jnp.abs(LE_target)  + _EPS_NORM)) ** 2
    )
    reg_loss = LAMBDA_REG * jnp.sum((theta - 1.0) ** 2)
    return data_loss + reg_loss


# ── JIT compile forward and backward ─────────────────────────────────────────
print("\n=== JIT-compiling single-step forward and backward ===", flush=True)
_loss_jit      = jax.jit(loss_fn)
_loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

print("  Warmup forward...", flush=True)
t_wu = time.time()
_ = jax.block_until_ready(_loss_jit(theta_star))
print(f"  Forward JIT compile + first eval: {time.time()-t_wu:.2f}s", flush=True)

print("  Warmup backward...", flush=True)
t_wu = time.time()
_ = jax.block_until_ready(_loss_and_grad(theta_star))
print(f"  Backward JIT compile + first eval: {time.time()-t_wu:.2f}s", flush=True)

N_TIMING = 5
t_fwd_runs = []; t_bwd_runs = []
for _ in range(N_TIMING):
    t0 = time.time(); _ = jax.block_until_ready(_loss_jit(theta_star))
    t_fwd_runs.append(time.time() - t0)
for _ in range(N_TIMING):
    t0 = time.time(); _ = jax.block_until_ready(_loss_and_grad(theta_star))
    t_bwd_runs.append(time.time() - t0)
T_forward_s  = float(np.median(t_fwd_runs))
T_backward_s = float(np.median(t_bwd_runs))
T_ratio      = T_backward_s / T_forward_s
crossover_p  = T_ratio / 2.0
print(f"  T_forward (median {N_TIMING}): {T_forward_s:.4f}s", flush=True)
print(f"  T_backward (median {N_TIMING}): {T_backward_s:.4f}s", flush=True)
print(f"  T_ratio = {T_ratio:.2f}  (AD faster when p > {crossover_p:.1f})", flush=True)

p_values = [1, 2, 5, 10, 20, 50]
T_ad_s   = [T_backward_s] * len(p_values)
T_fd_s   = [2 * pv * T_forward_s for pv in p_values]
for pv, tad, tfd in zip(p_values, T_ad_s, T_fd_s):
    print(f"  p={pv:3d}: T_AD={tad:.4f}s  T_FD={tfd:.4f}s  speedup={tfd/tad:.1f}x",
          flush=True)


# ── Initial perturbation ──────────────────────────────────────────────────────
rng = np.random.default_rng(42)
theta_0_np = rng.uniform(0.7, 1.3, size=P)
theta_0    = jnp.array(theta_0_np, dtype=jnp.float64)

loss_0 = float(_loss_jit(theta_0))
print(f"\n  Initial theta_0: {np.round(theta_0_np, 3).tolist()}", flush=True)
print(f"  Initial loss (single-step): {loss_0:.4e}", flush=True)
print(f"  Initial ||theta_0 - theta_star||_2 = "
      f"{float(jnp.linalg.norm(theta_0 - theta_star)):.4f}", flush=True)


# ── Cosine-annealing LR schedule ─────────────────────────────────────────────

def cosine_lr(step: int, n_steps: int, lr_max: float = 0.01,
              lr_min: float = 1e-4) -> float:
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * step / n_steps))


# ── Adam optimizer ────────────────────────────────────────────────────────────
# b2=0.9 (not 0.999): 10-step memory window so v adapts within ~10 steps when
# gradient magnitude drops (e.g., entering a flat plateau near loss=0).
# With b2=0.999 the ~1000-step window keeps v inflated from the high-gradient
# early phase, making g/sqrt(v) → 0 and stalling Adam on the flat landscape.
# clip_norm: global gradient clipping prevents the T_air Jacobian (~100×
# larger than other params) from dominating the update direction.

ADAM_B2       = 0.9
ADAM_CLIP_NORM = 10.0


def adam_step(params, m, v, grad, t, lr,
              b1=0.9, b2=ADAM_B2, eps=1e-8, clip_norm=ADAM_CLIP_NORM):
    g_norm = jnp.linalg.norm(grad)
    grad   = grad * jnp.minimum(1.0, clip_norm / (g_norm + 1e-12))
    m = b1 * m + (1.0 - b1) * grad
    v = b2 * v + (1.0 - b2) * grad ** 2
    m_hat = m / (1.0 - b1 ** t)
    v_hat = v / (1.0 - b2 ** t)
    return params - lr * m_hat / (jnp.sqrt(v_hat) + eps), m, v


print("\n" + "=" * 60, flush=True)
print(f"=== Method A: Adam + jax.grad (1000 steps, cosine LR 0.01→1e-4,"
      f" b2={ADAM_B2}, clip={ADAM_CLIP_NORM}) ===", flush=True)
print(f"    Loss: single-step + Tikhonov (lambda={LAMBDA_REG})", flush=True)
print("=" * 60, flush=True)

LR_MAX = 0.01; LR_MIN = 1e-4; N_ADAM = 1000

theta = theta_0
m = jnp.zeros(P, dtype=jnp.float64)
v = jnp.zeros(P, dtype=jnp.float64)

adam_loss_history = []; adam_param_err_history = []
adam_lr_history   = []; adam_grad_evals = []; adam_time_s = []

t_adam_start = time.time(); n_grad_evals = 0

for step in range(1, N_ADAM + 1):
    t_step = time.time()
    lr_t = cosine_lr(step, N_ADAM, LR_MAX, LR_MIN)
    loss_val, g = _loss_and_grad(theta)
    jax.block_until_ready((loss_val, g))
    n_grad_evals += 1

    theta, m, v = adam_step(theta, m, v, g, step, lr=lr_t)

    loss_f  = float(loss_val)
    err_f   = float(jnp.linalg.norm(theta - theta_star))
    t_cumul = time.time() - t_adam_start

    adam_loss_history.append(loss_f)
    adam_param_err_history.append(err_f)
    adam_lr_history.append(lr_t)
    adam_grad_evals.append(n_grad_evals)
    adam_time_s.append(t_cumul)

    if step % 50 == 0 or step == 1:
        print(f"  Adam step {step:3d}: loss={loss_f:.4e}  "
              f"||theta-theta*||={err_f:.4f}  lr={lr_t:.2e}  "
              f"({time.time()-t_step:.2f}s/step)", flush=True)

t_adam_total = time.time() - t_adam_start
print(f"\n  Adam done: {t_adam_total:.1f}s  ({n_grad_evals} grad evals)", flush=True)
print(f"  Final loss: {adam_loss_history[-1]:.4e}", flush=True)
print(f"  Final ||theta-theta*||_2: {adam_param_err_history[-1]:.4f}", flush=True)

adam_fwd_equiv   = [g * T_ratio for g in adam_grad_evals]
adam_theta_final = theta.tolist()


# ── L-BFGS-B + exact JAX gradient ────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("=== Method B: L-BFGS-B + jax.grad (exact gradient) ===", flush=True)
print(f"    Loss: single-step + Tikhonov (lambda={LAMBDA_REG})", flush=True)
print("=" * 60, flush=True)

lbfgsb_ad_loss_history = []; lbfgsb_ad_nfev_history = []; lbfgsb_ad_time_s = []
_lbfgsb_ad_evals = [0]; _t_lbfgsb_ad_start = time.time()


def _loss_and_grad_numpy(x):
    _lbfgsb_ad_evals[0] += 1
    theta = jnp.array(x, dtype=jnp.float64)
    loss_val, grad_val = _loss_and_grad(theta)
    jax.block_until_ready((loss_val, grad_val))
    lv = float(loss_val)
    lbfgsb_ad_loss_history.append(lv)
    lbfgsb_ad_nfev_history.append(_lbfgsb_ad_evals[0])
    lbfgsb_ad_time_s.append(time.time() - _t_lbfgsb_ad_start)
    if _lbfgsb_ad_evals[0] % 10 == 0 or _lbfgsb_ad_evals[0] == 1:
        print(f"  L-BFGS-B/AD eval {_lbfgsb_ad_evals[0]:4d}: loss={lv:.4e}  "
              f"({lbfgsb_ad_time_s[-1]:.1f}s)", flush=True)
    return lv, np.array(grad_val, dtype=np.float64)


_lbfgsb_ad_result = _sp_minimize(
    _loss_and_grad_numpy,
    x0=np.array(theta_0_np),
    method="L-BFGS-B",
    jac=True,
    options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-8},
)
t_lbfgsb_ad_total = time.time() - _t_lbfgsb_ad_start
print(f"\n  L-BFGS-B/AD: {_lbfgsb_ad_result.message}  "
      f"({_lbfgsb_ad_evals[0]} evals, {t_lbfgsb_ad_total:.1f}s)", flush=True)
print(f"  Final loss: {float(_lbfgsb_ad_result.fun):.4e}", flush=True)
lbfgsb_ad_theta_final = jnp.array(_lbfgsb_ad_result.x, dtype=jnp.float64)
print(f"  Final ||theta-theta*||_2: "
      f"{float(jnp.linalg.norm(lbfgsb_ad_theta_final - theta_star)):.4f}", flush=True)

# each AD eval = 1 backward pass = T_ratio forward-pass equivalents
lbfgsb_ad_fwd_equiv = [g * T_ratio for g in lbfgsb_ad_nfev_history]


# ── L-BFGS-B + FD gradient ───────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("=== Method C: L-BFGS-B + FD gradient (scipy jac=None) ===", flush=True)
print(f"    Loss: single-step + Tikhonov (lambda={LAMBDA_REG})", flush=True)
print("=" * 60, flush=True)

lbfgsb_fd_loss_history = []; lbfgsb_fd_nfev_history = []; lbfgsb_fd_time_s = []
_lbfgsb_fd_evals = [0]; _t_lbfgsb_fd_start = time.time()


def _loss_np_lbfgsb(x):
    _lbfgsb_fd_evals[0] += 1
    lv = float(_loss_jit(jnp.array(x, dtype=jnp.float64)))
    lbfgsb_fd_loss_history.append(lv)
    lbfgsb_fd_nfev_history.append(_lbfgsb_fd_evals[0])
    lbfgsb_fd_time_s.append(time.time() - _t_lbfgsb_fd_start)
    if _lbfgsb_fd_evals[0] % 20 == 0 or _lbfgsb_fd_evals[0] == 1:
        print(f"  L-BFGS-B/FD eval {_lbfgsb_fd_evals[0]:4d}: loss={lv:.4e}  "
              f"({lbfgsb_fd_time_s[-1]:.1f}s)", flush=True)
    return lv


_lbfgsb_fd_result = _sp_minimize(
    _loss_np_lbfgsb,
    x0=np.array(theta_0_np),
    method="L-BFGS-B",
    jac=None,
    options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-8},
)
t_lbfgsb_fd_total = time.time() - _t_lbfgsb_fd_start
print(f"\n  L-BFGS-B/FD: {_lbfgsb_fd_result.message}  "
      f"({_lbfgsb_fd_evals[0]} evals, {t_lbfgsb_fd_total:.1f}s)", flush=True)
print(f"  Final loss: {float(_lbfgsb_fd_result.fun):.4e}", flush=True)
lbfgsb_fd_theta_final = jnp.array(_lbfgsb_fd_result.x, dtype=jnp.float64)
print(f"  Final ||theta-theta*||_2: "
      f"{float(jnp.linalg.norm(lbfgsb_fd_theta_final - theta_star)):.4f}", flush=True)

# each FD eval = 1 forward pass (scipy calls this function p+1 times per gradient)
lbfgsb_fd_fwd_equiv = lbfgsb_fd_nfev_history


# ── Nelder-Mead ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("=== Method D: Nelder-Mead (gradient-free) ===", flush=True)
print(f"    Loss: single-step + Tikhonov (lambda={LAMBDA_REG})", flush=True)
print("=" * 60, flush=True)

nm_loss_history = []; nm_nfev_history = []; nm_time_s = []
_nm_evals = [0]; _t_nm_start = time.time()


def _loss_np_nm(x):
    _nm_evals[0] += 1
    lv = float(_loss_jit(jnp.array(x, dtype=jnp.float64)))
    nm_loss_history.append(lv)
    nm_nfev_history.append(_nm_evals[0])
    nm_time_s.append(time.time() - _t_nm_start)
    if _nm_evals[0] % 100 == 0 or _nm_evals[0] == 1:
        print(f"  NM eval {_nm_evals[0]:4d}: loss={lv:.4e}  "
              f"({nm_time_s[-1]:.1f}s)", flush=True)
    return lv


_nm_result = _sp_minimize(
    _loss_np_nm,
    x0=np.array(theta_0_np),
    method="Nelder-Mead",
    options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-8, "adaptive": True},
)
t_nm_total = time.time() - _t_nm_start
print(f"\n  Nelder-Mead: {_nm_result.message}  ({_nm_evals[0]} evals, {t_nm_total:.1f}s)",
      flush=True)
print(f"  Final loss: {float(_nm_result.fun):.4e}", flush=True)
nm_theta_final = jnp.array(_nm_result.x, dtype=jnp.float64)
print(f"  Final ||theta-theta*||_2: "
      f"{float(jnp.linalg.norm(nm_theta_final - theta_star)):.4f}", flush=True)


# ── Save results ──────────────────────────────────────────────────────────────
print("\n=== Saving results ===", flush=True)

results = {
    "p": P,
    "T_steps": 1,
    "mode": "single_step_tikhonov",
    "lambda_reg": LAMBDA_REG,
    "param_names": [
        "vis_dir", "nir_dir", "vis_dif", "nir_dif",
        "tref", "vcmax", "iota", "q", "pbot", "u",
    ],
    "theta_star":   theta_star.tolist(),
    "theta_0":      theta_0_np.tolist(),
    "GPP_target":   float(GPP_target),
    "H_target":     float(H_target),
    "LE_target":    float(LE_target),
    "T_forward_s":  T_forward_s,
    "T_backward_s": T_backward_s,
    "T_ratio":      T_ratio,
    "crossover_p":  crossover_p,
    "adam": {
        "loss_history":      adam_loss_history,
        "param_err_history": adam_param_err_history,
        "lr_history":        adam_lr_history,
        "grad_evals":        adam_grad_evals,
        "fwd_equiv":         adam_fwd_equiv,
        "time_s":            adam_time_s,
        "final_loss":        adam_loss_history[-1],
        "final_param_err":   adam_param_err_history[-1],
        "theta_final":       adam_theta_final,
        "total_time_s":      t_adam_total,
        "lr_max":            LR_MAX,
        "lr_min":            LR_MIN,
        "n_steps":           N_ADAM,
    },
    "lbfgsb_ad": {
        "loss_history":    lbfgsb_ad_loss_history,
        "nfev_history":    lbfgsb_ad_nfev_history,
        "fwd_equiv":       lbfgsb_ad_fwd_equiv,
        "time_s":          lbfgsb_ad_time_s,
        "total_time_s":    t_lbfgsb_ad_total,
        "final_loss":      float(_lbfgsb_ad_result.fun),
        "final_param_err": float(jnp.linalg.norm(lbfgsb_ad_theta_final - theta_star)),
        "theta_final":     lbfgsb_ad_theta_final.tolist(),
        "converged":       bool(_lbfgsb_ad_result.success),
    },
    "lbfgsb_fd": {
        "loss_history":    lbfgsb_fd_loss_history,
        "nfev_history":    lbfgsb_fd_nfev_history,
        "fwd_equiv":       lbfgsb_fd_fwd_equiv,
        "time_s":          lbfgsb_fd_time_s,
        "total_time_s":    t_lbfgsb_fd_total,
        "final_loss":      float(_lbfgsb_fd_result.fun),
        "final_param_err": float(jnp.linalg.norm(lbfgsb_fd_theta_final - theta_star)),
        "theta_final":     lbfgsb_fd_theta_final.tolist(),
        "converged":       bool(_lbfgsb_fd_result.success),
    },
    "nelder_mead": {
        "loss_history":    nm_loss_history,
        "nfev_history":    nm_nfev_history,
        "fwd_equiv":       nm_nfev_history,
        "time_s":          nm_time_s,
        "total_time_s":    t_nm_total,
        "final_loss":      float(_nm_result.fun),
        "final_param_err": float(jnp.linalg.norm(nm_theta_final - theta_star)),
        "theta_final":     nm_theta_final.tolist(),
        "converged":       bool(_nm_result.success),
    },
    "timing_sweep": {
        "p_values": p_values,
        "T_ad_s":   T_ad_s,
        "T_fd_s":   T_fd_s,
    },
}

out_path = OUTPUT_DIR / "multipar_calibration_singlestep_tikhonov_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved: {out_path}", flush=True)

print(f"\n=== multipar_calibration_singlestep.py (TIKHONOV lambda={LAMBDA_REG}) complete ===",
      flush=True)
