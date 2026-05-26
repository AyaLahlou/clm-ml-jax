"""
Minimal calibration experiment (p=3): JAX autodiff convergence demo for paper.

Recovers 3 physiological/forcing parameters from perturbed starting values
using Adam gradient descent, compared against L-BFGS-B+AD and Nelder-Mead.

Parameter set (p=3 scale factors, truth = 1.0 each):
  0  alpha_vcmax  — Vcmax25 (max carboxylation rate; drives GPP)
  1  alpha_iota   — iota_SPA WUE efficiency (couples GPP↔LE)
  2  alpha_tref   — air temperature (drives H and LE)

Timestep: step index 39 (1-based) in CHATS7 May 2007 forcing.
  Local noon (12:00 PDT = 19:00 UTC); confirmed daytime with GPP > 0.
  With 3 outputs (GPP, H, LE) and 3 parameters the system is exactly
  determined — no Tikhonov regularization needed.

Perturbation: theta_0 drawn uniformly from [0.7, 1.3] using rng seed 0.

Loss: normalized MSE over (GPP, H, LE):
  L(theta) = sum_i ((y_i(theta) - y_i*) / (|y_i*| + 1e-6))^2

Methods:
  Adam        — 100 steps, cosine LR 0.02→1e-4, b1=0.9, b2=0.9, grad clip=10
  L-BFGS-B    — exact JAX gradient via scipy jac=True
  Nelder-Mead — gradient-free baseline

Outputs:
  diags/output/minimal_calibration_results.json
  diags/figures/minimal_calibration.{pdf,png}
  Paper/jaxes_paper/figures/minimal_calibration.{pdf,png}
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR  = Path(__file__).parent / "output"
FIGURES_DIR = Path(__file__).parent / "figures"
PAPER_FIGS  = _PROJECT_ROOT / "Paper" / "jaxes_paper" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

STEP_INDEX = 39  # 1-based; local noon (12:00 PDT = 19:00 UTC), confirmed GPP > 0

# ── Shared init ───────────────────────────────────────────────────────────────
print("=== minimal_calibration.py (p=3): loading model ===", flush=True)
from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs,
    MLCanopyFluxes,
    compute_gpp, compute_h, compute_le,
)
import multilayer_canopy.MLpftconMod             as _MLpftconMod
import multilayer_canopy.MLLeafPhotosynthesisMod  as _LeafMod
import multilayer_canopy.MLCanopyNitrogenProfileMod as _NitroMod
from offline_driver.TowerMetMod import TowerMetCurr
from offline_executable.main import read_namelist, _resolve_path, build_bounds
from offline_driver import TowerDataMod
from clm_src_main import clm_instMod

_p    = grid.p
_ncan = grid.ncan

_mlcf_kwargs_no_atm = {k: v for k, v in _mlcf_kwargs.items()
                       if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst")}

_orig_pftcon = _MLpftconMod.MLpftcon

P = 3  # number of parameters

# ── pftcon injection helpers ──────────────────────────────────────────────────

def _set_pftcon(new_inst):
    _MLpftconMod.MLpftcon = new_inst
    _LeafMod.MLpftcon     = new_inst
    _NitroMod.MLpftcon    = new_inst


def _restore_pftcon():
    _MLpftconMod.MLpftcon = _orig_pftcon
    _LeafMod.MLpftcon     = _orig_pftcon
    _NitroMod.MLpftcon    = _orig_pftcon


# ── Load target timestep (step 39) ───────────────────────────────────────────
print(f"\n=== Loading forcing state for step {STEP_INDEX} ===", flush=True)

_NML_FULL   = _PROJECT_ROOT / "src" / "offline_executable" / "nl.CHATS7.05.2007"
_nml_full   = read_namelist(str(_NML_FULL))
_nml_params = _nml_full.get("clmML_inparm", _nml_full.get("clm_inparm", {}))
_fin_tower  = _resolve_path(str(_nml_params.get("fin_tower", "")))
_bounds_    = build_bounds(_nml_full)

(_atm_step, _watm_step, _frv_step) = TowerMetCurr(
    _fin_tower, STEP_INDEX,
    TowerDataMod.tower_num,
    _bounds_.begp, _bounds_.endp,
    clm_instMod.atm2lnd_inst,
    clm_instMod.wateratm2lndbulk_inst,
    clm_instMod.frictionvel_inst,
)
print(f"  Loaded step {STEP_INDEX}", flush=True)


# ── Per-timestep forward function (p=3) ──────────────────────────────────────

def forward_theta(theta: jnp.ndarray):
    """p=3 single-step forward at step 39.

    theta[0] alpha_vcmax — global Vcmax25 scale
    theta[1] alpha_iota  — WUE iota_SPA scale
    theta[2] alpha_tref  — air temperature scale
    """
    modified_atm = _atm_step._replace(
        forc_t_downscaled_col = theta[2] * _atm_step.forc_t_downscaled_col,
    )

    # iota_SPA via module-global mutation (JAX traces through jnp.asarray)
    _set_pftcon(_orig_pftcon._replace(
        iota_SPA = theta[1] * _orig_pftcon.iota_SPA,
    ))

    # vcmaxpft via explicit JAX arg to bypass JIT cache
    vcmaxpft_jax = theta[0] * _orig_pftcon.vcmaxpft

    inst = MLCanopyFluxes(
        mlcanopy_inst         = mlcanopy_inst,
        atm2lnd_inst          = modified_atm,
        wateratm2lndbulk_inst = _watm_step,
        vcmaxpft_jax          = vcmaxpft_jax,
        **_mlcf_kwargs_no_atm,
    )
    _restore_pftcon()

    gpp = compute_gpp(inst, _p, _ncan)
    h   = compute_h(inst, _p, _ncan)
    le  = compute_le(inst, _p, _ncan)
    return gpp, h, le


# ── Generate target (theta_star = ones) ──────────────────────────────────────
theta_star = jnp.ones(P, dtype=jnp.float64)

print("\n=== Generating target (theta_star = ones, step 39) ===", flush=True)
t0 = time.time()
GPP_target, H_target, LE_target = forward_theta(theta_star)
jax.block_until_ready((GPP_target, H_target, LE_target))
print(f"  GPP={float(GPP_target):.4f}  H={float(H_target):.4f}"
      f"  LE={float(LE_target):.4f}", flush=True)
print(f"  Target generation: {time.time()-t0:.2f}s", flush=True)

_EPS_NORM = 1e-6


# ── Loss function (normalized MSE, no regularization) ────────────────────────

def loss_fn(theta: jnp.ndarray) -> jnp.ndarray:
    """Normalized MSE over (GPP, H, LE). p=3 exactly determined — no Tikhonov."""
    gpp, h, le = forward_theta(theta)
    return (
        ((gpp - GPP_target) / (jnp.abs(GPP_target) + _EPS_NORM)) ** 2
        + ((h   - H_target)   / (jnp.abs(H_target)   + _EPS_NORM)) ** 2
        + ((le  - LE_target)  / (jnp.abs(LE_target)  + _EPS_NORM)) ** 2
    )


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
print(f"  T_forward  (median {N_TIMING}): {T_forward_s:.4f}s", flush=True)
print(f"  T_backward (median {N_TIMING}): {T_backward_s:.4f}s", flush=True)
print(f"  T_ratio = {T_ratio:.2f}", flush=True)


# ── Initial perturbation ──────────────────────────────────────────────────────
rng = np.random.default_rng(0)
theta_0_np = rng.uniform(0.7, 1.3, size=P)
theta_0    = jnp.array(theta_0_np, dtype=jnp.float64)

loss_0 = float(_loss_jit(theta_0))
print(f"\n  Initial theta_0: {np.round(theta_0_np, 3).tolist()}", flush=True)
print(f"  Initial loss: {loss_0:.4e}", flush=True)
print(f"  Initial ||theta_0 - theta_star||_2 = "
      f"{float(jnp.linalg.norm(theta_0 - theta_star)):.4f}", flush=True)


# ── Cosine-annealing LR schedule ─────────────────────────────────────────────

def cosine_lr(step: int, n_steps: int, lr_max: float = 0.02,
              lr_min: float = 1e-4) -> float:
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * step / n_steps))


# ── Adam optimizer ────────────────────────────────────────────────────────────
# b2=0.9 (not 0.999): 10-step memory window so v adapts within ~10 steps when
# gradient magnitude drops near loss=0. clip_norm prevents the T_air Jacobian
# (~100× larger than vcmax/iota) from dominating the update direction.

ADAM_B2        = 0.9
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
print(f"=== Method A: Adam + jax.grad (100 steps, cosine LR 0.02→1e-4,"
      f" b2={ADAM_B2}, clip={ADAM_CLIP_NORM}) ===", flush=True)
print("=" * 60, flush=True)

LR_MAX  = 0.02
LR_MIN  = 1e-4
N_ADAM  = 100

theta = theta_0
m     = jnp.zeros(P, dtype=jnp.float64)
v     = jnp.zeros(P, dtype=jnp.float64)

adam_loss_history      = []
adam_param_err_history = []
adam_grad_evals        = []
adam_time_s            = []

t_adam_start = time.time()
n_grad_evals = 0

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
    adam_grad_evals.append(n_grad_evals)
    adam_time_s.append(t_cumul)

    if step % 10 == 0 or step == 1:
        print(f"  Adam step {step:3d}: loss={loss_f:.4e}  "
              f"||theta-theta*||={err_f:.4f}  lr={lr_t:.2e}  "
              f"({time.time()-t_step:.2f}s/step)", flush=True)

t_adam_total = time.time() - t_adam_start
adam_theta_final = theta.tolist()

print(f"\n  Adam done: {t_adam_total:.1f}s  ({n_grad_evals} grad evals)", flush=True)
print(f"  Final loss: {adam_loss_history[-1]:.4e}", flush=True)
print(f"  Final theta: {[round(v, 4) for v in adam_theta_final]}", flush=True)
print(f"  Final ||theta-theta*||_2: {adam_param_err_history[-1]:.4f}", flush=True)

# Forward-pass equivalents: each grad eval = T_ratio forward passes
adam_fwd_equiv = [g * T_ratio for g in adam_grad_evals]


# ── L-BFGS-B + exact JAX gradient ────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("=== Method B: L-BFGS-B + jax.grad (exact gradient) ===", flush=True)
print("=" * 60, flush=True)

lbfgsb_loss_history = []; lbfgsb_nfev_history = []; lbfgsb_time_s = []
_lbfgsb_evals = [0]; _t_lbfgsb_start = time.time()


def _loss_and_grad_numpy(x):
    _lbfgsb_evals[0] += 1
    theta_arr = jnp.array(x, dtype=jnp.float64)
    loss_val, grad_val = _loss_and_grad(theta_arr)
    jax.block_until_ready((loss_val, grad_val))
    lv = float(loss_val)
    lbfgsb_loss_history.append(lv)
    lbfgsb_nfev_history.append(_lbfgsb_evals[0])
    lbfgsb_time_s.append(time.time() - _t_lbfgsb_start)
    if _lbfgsb_evals[0] % 10 == 0 or _lbfgsb_evals[0] == 1:
        print(f"  L-BFGS-B/AD eval {_lbfgsb_evals[0]:4d}: loss={lv:.4e}  "
              f"({lbfgsb_time_s[-1]:.1f}s)", flush=True)
    return lv, np.array(grad_val, dtype=np.float64)


_lbfgsb_result = _sp_minimize(
    _loss_and_grad_numpy,
    x0=np.array(theta_0_np),
    method="L-BFGS-B",
    jac=True,
    options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-8},
)
t_lbfgsb_total = time.time() - _t_lbfgsb_start
lbfgsb_theta_final = jnp.array(_lbfgsb_result.x, dtype=jnp.float64)

print(f"\n  L-BFGS-B/AD: {_lbfgsb_result.message}  "
      f"({_lbfgsb_evals[0]} evals, {t_lbfgsb_total:.1f}s)", flush=True)
print(f"  Final loss: {float(_lbfgsb_result.fun):.4e}", flush=True)
print(f"  Final theta: {[round(v, 4) for v in lbfgsb_theta_final.tolist()]}", flush=True)
print(f"  Final ||theta-theta*||_2: "
      f"{float(jnp.linalg.norm(lbfgsb_theta_final - theta_star)):.4f}", flush=True)

# each AD eval = T_ratio forward-pass equivalents
lbfgsb_fwd_equiv = [g * T_ratio for g in lbfgsb_nfev_history]


# ── Nelder-Mead ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("=== Method C: Nelder-Mead (gradient-free baseline) ===", flush=True)
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
nm_theta_final = jnp.array(_nm_result.x, dtype=jnp.float64)

print(f"\n  Nelder-Mead: {_nm_result.message}  "
      f"({_nm_evals[0]} evals, {t_nm_total:.1f}s)", flush=True)
print(f"  Final loss: {float(_nm_result.fun):.4e}", flush=True)
print(f"  Final theta: {[round(v, 4) for v in nm_theta_final.tolist()]}", flush=True)
print(f"  Final ||theta-theta*||_2: "
      f"{float(jnp.linalg.norm(nm_theta_final - theta_star)):.4f}", flush=True)


# ── Final summary table ───────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("=== FINAL SUMMARY ===", flush=True)
print(f"{'Method':<18} {'Final loss':>14} {'||theta-theta*||':>18} {'FP-equiv':>10} {'Wall (s)':>10}", flush=True)
print("-" * 74, flush=True)

_adam_fp_equiv = int(N_ADAM * T_ratio)
print(f"{'Adam+AD':<18} {adam_loss_history[-1]:>14.4e} "
      f"{adam_param_err_history[-1]:>18.4f} {_adam_fp_equiv:>10d} "
      f"{t_adam_total:>10.1f}", flush=True)

_lbfgsb_fp_eq = int(_lbfgsb_evals[0] * T_ratio)
print(f"{'L-BFGS-B+AD':<18} {float(_lbfgsb_result.fun):>14.4e} "
      f"{float(jnp.linalg.norm(lbfgsb_theta_final - theta_star)):>18.4f} "
      f"{_lbfgsb_fp_eq:>10d} {t_lbfgsb_total:>10.1f}", flush=True)

print(f"{'Nelder-Mead':<18} {float(_nm_result.fun):>14.4e} "
      f"{float(jnp.linalg.norm(nm_theta_final - theta_star)):>18.4f} "
      f"{_nm_evals[0]:>10d} {t_nm_total:>10.1f}", flush=True)
print("=" * 60, flush=True)

print(f"\nParameter recovery (truth = 1.0 for all):", flush=True)
param_names_display = ["vcmax", "iota", "tref"]
print(f"  {'Param':<8} {'theta_0':>8} {'Adam':>8} {'L-BFGS-B':>10} {'NM':>8}", flush=True)
for i, pname in enumerate(param_names_display):
    print(f"  {pname:<8} {theta_0_np[i]:>8.4f} {adam_theta_final[i]:>8.4f}"
          f" {lbfgsb_theta_final[i]:>10.4f} {float(nm_theta_final[i]):>8.4f}", flush=True)


# ── Save results JSON ─────────────────────────────────────────────────────────
print("\n=== Saving results ===", flush=True)

results = {
    "theta_star":  theta_star.tolist(),
    "theta_0":     theta_0_np.tolist(),
    "param_names": ["vcmax", "iota", "tref"],
    "step_index":  STEP_INDEX,
    "GPP_target":  float(GPP_target),
    "H_target":    float(H_target),
    "LE_target":   float(LE_target),
    "T_forward_s":  T_forward_s,
    "T_backward_s": T_backward_s,
    "T_ratio":      T_ratio,
    "methods": {
        "adam": {
            "theta_final":    adam_theta_final,
            "final_loss":     adam_loss_history[-1],
            "n_grad_evals":   N_ADAM,
            "wall_time_s":    t_adam_total,
            "loss_history":   adam_loss_history,
            "param_err_history": adam_param_err_history,
            "fwd_equiv":      adam_fwd_equiv,
            "grad_evals":     adam_grad_evals,
            "time_s":         adam_time_s,
            "lr_max":         LR_MAX,
            "lr_min":         LR_MIN,
        },
        "lbfgsb": {
            "theta_final":  lbfgsb_theta_final.tolist(),
            "final_loss":   float(_lbfgsb_result.fun),
            "n_fev":        _lbfgsb_evals[0],
            "wall_time_s":  t_lbfgsb_total,
            "loss_history": lbfgsb_loss_history,
            "nfev_history": lbfgsb_nfev_history,
            "fwd_equiv":    lbfgsb_fwd_equiv,
            "time_s":       lbfgsb_time_s,
            "converged":    bool(_lbfgsb_result.success),
        },
        "nelder_mead": {
            "theta_final":  nm_theta_final.tolist(),
            "final_loss":   float(_nm_result.fun),
            "n_fev":        _nm_evals[0],
            "wall_time_s":  t_nm_total,
            "loss_history": nm_loss_history,
            "nfev_history": nm_nfev_history,
            "fwd_equiv":    nm_nfev_history,
            "time_s":       nm_time_s,
            "converged":    bool(_nm_result.success),
        },
    },
}

out_path = OUTPUT_DIR / "minimal_calibration_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved: {out_path}", flush=True)


# ── Figure ────────────────────────────────────────────────────────────────────
print("\n=== Generating figure ===", flush=True)

matplotlib.rcParams.update({
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "font.family":      "sans-serif",
})

C_ADAM    = "#1f77b4"   # blue
C_LBFGSB  = "#ff7f0e"   # orange
C_NM      = "#2ca02c"   # green
C_INIT    = "#888888"   # grey (initial theta bar)
C_TRUTH   = "black"     # dashed truth line

fig, (ax_loss, ax_param) = plt.subplots(1, 2, figsize=(6, 3))

# ── Panel A: Loss curves (log scale) ─────────────────────────────────────────
ax = ax_loss

# Adam: x = grad eval count (= step number), y = loss
_ax_fp = adam_fwd_equiv  # forward-pass equivalents
ax.plot(_ax_fp, adam_loss_history, color=C_ADAM,   lw=1.5, label="Adam+AD")

# L-BFGS-B: x = forward-pass equivalents (nfev × T_ratio)
ax.plot(lbfgsb_fwd_equiv, lbfgsb_loss_history, color=C_LBFGSB, lw=1.5,
        label="L-BFGS-B+AD")

# Nelder-Mead: x = function evaluation count (= forward-pass count)
ax.plot(nm_nfev_history, nm_loss_history, color=C_NM, lw=1.5,
        label="Nelder-Mead")

# Mark final loss values
_x_adam_end = _ax_fp[-1]
ax.annotate(f"{adam_loss_history[-1]:.1e}",
            xy=(_x_adam_end, adam_loss_history[-1]),
            xytext=(5, 3), textcoords="offset points",
            fontsize=7, color=C_ADAM)

_x_lb_end = lbfgsb_fwd_equiv[-1] if lbfgsb_fwd_equiv else 0
ax.annotate(f"{float(_lbfgsb_result.fun):.1e}",
            xy=(_x_lb_end, float(_lbfgsb_result.fun)),
            xytext=(5, -8), textcoords="offset points",
            fontsize=7, color=C_LBFGSB)

_x_nm_end = nm_nfev_history[-1] if nm_nfev_history else 0
ax.annotate(f"{float(_nm_result.fun):.1e}",
            xy=(_x_nm_end, float(_nm_result.fun)),
            xytext=(5, 3), textcoords="offset points",
            fontsize=7, color=C_NM)

ax.set_yscale("log")
ax.set_xlabel("Gradient evaluations (forward-pass equivalents)")
ax.set_ylabel("Loss")
ax.set_title("(a) Convergence")
ax.set_xlim(0, 200)
ax.legend(loc="center right", framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Panel B: Parameter recovery (grouped bar chart) ──────────────────────────
ax = ax_param

param_labels = [r"Vcmax$_{25}$", r"$\iota$", r"$T_\mathrm{ref}$"]
x      = np.arange(len(param_labels))
width  = 0.18

bars_init  = ax.bar(x - 1.5*width, theta_0_np,                  width, color=C_INIT,
                    label=r"$\theta_0$ (init)", alpha=0.85)
bars_adam  = ax.bar(x - 0.5*width, adam_theta_final,             width, color=C_ADAM,
                    label=r"$\hat\theta$ (Adam)", alpha=0.85)
bars_lb    = ax.bar(x + 0.5*width, lbfgsb_theta_final.tolist(),  width, color=C_LBFGSB,
                    label=r"$\hat\theta$ (L-BFGS-B+AD)", alpha=0.85)
bars_nm    = ax.bar(x + 1.5*width, nm_theta_final.tolist(),      width, color=C_NM,
                    label=r"$\hat\theta$ (Nelder-Mead)", alpha=0.85)

# Dashed truth line at 1.0
ax.axhline(1.0, color=C_TRUTH, lw=1.2, ls="--", label=r"$\theta^\star = 1$")

ax.set_xticks(x)
ax.set_xticklabels(param_labels)
ax.set_ylabel("Parameter scale factor")
ax.set_title("(b) Parameter recovery")
ax.legend(loc="upper right", framealpha=0.9, fontsize=7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(False)
ax.set_axisbelow(False)

# y-axis range: give a little headroom around all values
_all_vals = list(theta_0_np) + adam_theta_final + lbfgsb_theta_final.tolist() + nm_theta_final.tolist() + [1.0]
_ylo = min(_all_vals) - 0.05
_yhi = max(_all_vals) + 0.08
ax.set_ylim(_ylo, _yhi)

plt.tight_layout(pad=0.8)

for _dest in (FIGURES_DIR, PAPER_FIGS):
    fig.savefig(_dest / "minimal_calibration.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(_dest / "minimal_calibration.png", dpi=300, bbox_inches="tight")
    print(f"  Figure saved: {_dest / 'minimal_calibration.{{pdf,png}}'}", flush=True)

plt.close(fig)

print("\n=== minimal_calibration.py complete ===", flush=True)
