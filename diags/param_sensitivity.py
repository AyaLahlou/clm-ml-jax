"""
Parameter sensitivity analysis for CLM-ML-JAX.

Computes dGPP/dθ and dLE/dθ for key plant physiological parameters
at the CHATS7 operating point.  Used as a prerequisite sanity check
before gradient-based parameter optimization: if any parameter has a
near-zero or non-finite gradient at the default operating point,
optimization cannot recover it from flux observations.

Parameters tested
-----------------
alpha_sw      : scale factor on incoming solar radiation (→ PAR → GPP)
alpha_tref    : scale factor on air temperature (→ enzyme kinetics)
alpha_vcmax25 : scale factor on Vcmax25 (maximum carboxylation rate)
alpha_iota    : scale factor on iota_SPA (WUE stomatal efficiency) — FD only,
                embedded as Python float in compiled kernels

Gradient methods
----------------
* alpha_sw, alpha_tref, alpha_vcmax25: jax.grad (exact autodiff)
* alpha_iota: central finite difference (FD) only, because iota_SPA
  is captured as a Python float (np.asarray → float()) at kernel-factory
  time in MLLeafPhotosynthesisMod.py.  Making iota fully differentiable
  requires passing it as a runtime JAX argument throughout the factory —
  planned for optimize_params.py Phase 2 refactoring.

Usage (from project root):
    cd src && python ../diags/param_sensitivity.py

Output:
  - Console table: parameter, dGPP/dα, dLE/dα, finite-difference check
  - diags/figures/param_sensitivity.png: bar chart
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared init ───────────────────────────────────────────────────────────────
from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs, jax, jnp, MLCanopyFluxes,
    atm2lnd_inst, wateratm2lndbulk_inst, compute_gpp, compute_le,
)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── MLpftconMod module-level singleton (for vcmax25 sensitivity) ──────────────
# NamedTuple fields are immutable.  The correct injection pattern is:
#   1. Create a new instance with `._replace(param=alpha * original.param)`
#   2. Update EVERY physics module that has `from MLpftconMod import MLpftcon`
#      because those local bindings were captured at import time and won't see
#      a change to `MLpftconMod.MLpftcon` alone.
import multilayer_canopy.MLpftconMod              as _pftcon_mod
import multilayer_canopy.MLLeafPhotosynthesisMod  as _leaf_mod
import multilayer_canopy.MLCanopyNitrogenProfileMod as _nitro_mod

from multilayer_canopy.MLclm_varpar import isun, isha

_p    = grid.p
_ncan = grid.ncan

# ── Build kwargs without atm2lnd so we can inject scaled versions ─────────────
_mlcf_kwargs_no_atm = {k: v for k, v in _mlcf_kwargs.items()
                       if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst")}

_orig_pftcon = _pftcon_mod.MLpftcon   # original MLpftcon_type instance


def _set_pftcon(new_inst):
    """Update MLpftcon in all physics module namespaces at once."""
    _pftcon_mod.MLpftcon = new_inst
    _leaf_mod.MLpftcon   = new_inst
    _nitro_mod.MLpftcon  = new_inst


def _restore_pftcon():
    _pftcon_mod.MLpftcon = _orig_pftcon
    _leaf_mod.MLpftcon   = _orig_pftcon
    _nitro_mod.MLpftcon  = _orig_pftcon


# ── Forward functions ──────────────────────────────────────────────────────────

def _run_inst(modified_atm=None, modified_mlcanopy=None, vcmaxpft_jax=None):
    """Run MLCanopyFluxes with optionally modified instances."""
    return MLCanopyFluxes(
        mlcanopy_inst=modified_mlcanopy or mlcanopy_inst,
        atm2lnd_inst=modified_atm or atm2lnd_inst,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        vcmaxpft_jax=vcmaxpft_jax,
        **_mlcf_kwargs_no_atm,
    )


# alpha_sw ──────────────────────────────────────────────────────────────────────
def _gpp_sw(alpha: jnp.ndarray) -> jnp.ndarray:
    inst = _run_inst(modified_atm=atm2lnd_inst._replace(
        forc_solad_downscaled_col=alpha * atm2lnd_inst.forc_solad_downscaled_col,
        forc_solai_grc           =alpha * atm2lnd_inst.forc_solai_grc,
    ))
    return compute_gpp(inst, _p, _ncan)


def _le_sw(alpha: jnp.ndarray) -> jnp.ndarray:
    inst = _run_inst(modified_atm=atm2lnd_inst._replace(
        forc_solad_downscaled_col=alpha * atm2lnd_inst.forc_solad_downscaled_col,
        forc_solai_grc           =alpha * atm2lnd_inst.forc_solai_grc,
    ))
    return compute_le(inst, _p, _ncan)


# alpha_tref ────────────────────────────────────────────────────────────────────
def _gpp_tref(alpha: jnp.ndarray) -> jnp.ndarray:
    inst = _run_inst(modified_atm=atm2lnd_inst._replace(
        forc_t_downscaled_col=alpha * atm2lnd_inst.forc_t_downscaled_col,
    ))
    return compute_gpp(inst, _p, _ncan)


def _le_tref(alpha: jnp.ndarray) -> jnp.ndarray:
    inst = _run_inst(modified_atm=atm2lnd_inst._replace(
        forc_t_downscaled_col=alpha * atm2lnd_inst.forc_t_downscaled_col,
    ))
    return compute_le(inst, _p, _ncan)


# alpha_vcmax25 — via explicit JAX argument ────────────────────────────────────
# CanopyNitrogenProfile is @jax.jit-cached; MLpftcon.vcmaxpft is baked in as an
# XLA constant at first trace.  Module-global mutation (_set_pftcon) cannot
# override the cached value.  Pass vcmaxpft_jax explicitly so JAX traces through it.
def _gpp_vcmax(alpha: jnp.ndarray) -> jnp.ndarray:
    vcmaxpft_jax = alpha * _orig_pftcon.vcmaxpft
    inst = _run_inst(vcmaxpft_jax=vcmaxpft_jax)
    return compute_gpp(inst, _p, _ncan)


def _le_vcmax(alpha: jnp.ndarray) -> jnp.ndarray:
    vcmaxpft_jax = alpha * _orig_pftcon.vcmaxpft
    inst = _run_inst(vcmaxpft_jax=vcmaxpft_jax)
    return compute_le(inst, _p, _ncan)


# ── Baseline ──────────────────────────────────────────────────────────────────
print("\n=== Baseline outputs ===", flush=True)
alpha1 = jnp.float64(1.0)
baseline_gpp = float(_gpp_sw(alpha1))
baseline_le  = float(_le_sw(alpha1))
print(f"  GPP proxy (baseline) = {baseline_gpp:.4f} (sum agross*dpai)", flush=True)
print(f"  LE  proxy (baseline) = {baseline_le:.4f} (sum lhleaf*dpai)", flush=True)


# ── Compute sensitivities ──────────────────────────────────────────────────────
print("\n=== Computing JAX autodiff sensitivities ===", flush=True)

_jit_grad_gpp_sw    = jax.jit(jax.grad(_gpp_sw))
_jit_grad_gpp_tref  = jax.jit(jax.grad(_gpp_tref))
_jit_grad_gpp_vcmax = jax.jit(jax.grad(_gpp_vcmax))

_jit_grad_le_sw    = jax.jit(jax.grad(_le_sw))
_jit_grad_le_tref  = jax.jit(jax.grad(_le_tref))
_jit_grad_le_vcmax = jax.jit(jax.grad(_le_vcmax))

params_jax = [
    ("alpha_sw",     _jit_grad_gpp_sw,    _jit_grad_le_sw),
    ("alpha_tref",   _jit_grad_gpp_tref,  _jit_grad_le_tref),
    ("alpha_vcmax25",_jit_grad_gpp_vcmax, _jit_grad_le_vcmax),
]

jax_results = {}
for name, grad_gpp_fn, grad_le_fn in params_jax:
    t0 = time.time()
    dgpp = float(grad_gpp_fn(alpha1))
    dle  = float(grad_le_fn(alpha1))
    elapsed = time.time() - t0
    jax_results[name] = (dgpp, dle)
    print(f"  {name:<18}  dGPP={dgpp:>12.4e}  dLE={dle:>12.4e}  ({elapsed:.1f}s)",
          flush=True)


# ── JAX grad sensitivity for alpha_iota (fixed in session 25) ─────────────────
# iota_SPA is now a JAX runtime broadcast arg in the WUE kernels (in_axes=None).
# Use the same _set_pftcon/_restore_pftcon pattern as vcmax.
print("\n=== Computing JAX grad sensitivity for alpha_iota ===", flush=True)

EPS_FD = 1e-4


def _gpp_iota(alpha: jnp.ndarray) -> jnp.ndarray:
    _set_pftcon(_orig_pftcon._replace(iota_SPA=alpha * _orig_pftcon.iota_SPA))
    try:
        inst = _run_inst()
        return compute_gpp(inst, _p, _ncan)
    finally:
        _restore_pftcon()


def _le_iota(alpha: jnp.ndarray) -> jnp.ndarray:
    _set_pftcon(_orig_pftcon._replace(iota_SPA=alpha * _orig_pftcon.iota_SPA))
    try:
        inst = _run_inst()
        return compute_le(inst, _p, _ncan)
    finally:
        _restore_pftcon()


t0 = time.time()
_iota_alpha = jnp.float64(1.0)
dgpp_iota = float(jax.jit(jax.grad(_gpp_iota))(_iota_alpha))
dle_iota  = float(jax.jit(jax.grad(_le_iota))(_iota_alpha))
elapsed_iota = time.time() - t0
print(f"  alpha_iota           dGPP={dgpp_iota:>12.4e}  dLE={dle_iota:>12.4e}  ({elapsed_iota:.1f}s, JAX grad)",
      flush=True)


# ── FD cross-check for JAX-differentiable parameters ──────────────────────────
print("\n=== FD cross-check for JAX-differentiable parameters ===", flush=True)

fd_fns = {
    "alpha_sw":     (_gpp_sw,    _le_sw),
    "alpha_tref":   (_gpp_tref,  _le_tref),
    "alpha_vcmax25":(_gpp_vcmax, _le_vcmax),
}

fd_results = {}
for name, (gpp_fn, le_fn) in fd_fns.items():
    gpp_p = float(gpp_fn(jnp.float64(1.0 + EPS_FD)))
    gpp_m = float(gpp_fn(jnp.float64(1.0 - EPS_FD)))
    le_p  = float(le_fn(jnp.float64(1.0 + EPS_FD)))
    le_m  = float(le_fn(jnp.float64(1.0 - EPS_FD)))
    fd_results[name] = (
        (gpp_p - gpp_m) / (2.0 * EPS_FD),
        (le_p  - le_m)  / (2.0 * EPS_FD),
    )
    print(f"  {name:<18}  dGPP_FD={fd_results[name][0]:>12.4e}  dLE_FD={fd_results[name][1]:>12.4e}",
          flush=True)


# ── Summary table ─────────────────────────────────────────────────────────────
print("\n=== Sensitivity summary ===")
print(f"{'Parameter':<18}  {'dGPP/dα (JAX)':>14}  {'dGPP/dα (FD)':>14}  "
      f"{'Rel err GPP':>12}  {'dLE/dα (JAX)':>13}  {'dLE/dα (FD)':>13}  "
      f"{'Rel err LE':>12}  {'Method'}")
print("-" * 130)

all_pass = True
records = []
for name in ["alpha_sw", "alpha_tref", "alpha_vcmax25"]:
    jax_gpp, jax_le = jax_results[name]
    fd_gpp,  fd_le  = fd_results[name]
    rel_gpp = abs(jax_gpp - fd_gpp) / (abs(fd_gpp) + 1e-30)
    rel_le  = abs(jax_le  - fd_le)  / (abs(fd_le)  + 1e-30)
    gpp_ok  = rel_gpp < 0.01
    le_ok   = rel_le  < 0.01
    status  = "PASS" if (gpp_ok and le_ok) else "FAIL"
    all_pass = all_pass and gpp_ok and le_ok
    records.append((name, jax_gpp, fd_gpp, rel_gpp, jax_le, fd_le, rel_le, "JAX+FD", status))
    print(f"  {name:<18}  {jax_gpp:>14.4e}  {fd_gpp:>14.4e}  {rel_gpp:>12.2e}  "
          f"{jax_le:>13.4e}  {fd_le:>13.4e}  {rel_le:>12.2e}  {status}")

# alpha_iota — FD only
records.append(("alpha_iota", None, dgpp_iota, None, None, dle_iota, None, "FD only", "N/A"))
print(f"  {'alpha_iota':<18}  {'N/A':>14}  {dgpp_iota:>14.4e}  {'N/A':>12}  "
      f"  {'N/A':>13}  {dle_iota:>13.4e}  {'N/A':>12}  FD only")

print()
print(f"{'ALL PASS' if all_pass else 'SOME FAILURES'} for JAX-differentiable parameters "
      f"(1% tolerance)")

# ── Identifiability assessment ─────────────────────────────────────────────────
print("\n=== Identifiability assessment ===")
print("Parameter is identifiable if |dGPP/dα| or |dLE/dα| >> noise level")
print("(noise level ≈ 0.1 μmol CO₂ m⁻² s⁻¹ for GPP, ≈ 5 W/m² for LE)")
for name, jax_gpp, fd_gpp, rel_gpp, jax_le, fd_le, rel_le, method, *_ in records:
    gpp_val = jax_gpp if jax_gpp is not None else fd_gpp
    le_val  = jax_le  if jax_le  is not None else fd_le
    gpp_snr = abs(gpp_val) / 0.1 if gpp_val else 0
    le_snr  = abs(le_val)  / 5.0 if le_val  else 0
    ident   = "YES" if max(gpp_snr, le_snr) > 10 else ("MARGINAL" if max(gpp_snr, le_snr) > 1 else "NO")
    print(f"  {name:<18}  |dGPP|={abs(gpp_val):.3e}  SNR_GPP={gpp_snr:.1f}  "
          f"|dLE|={abs(le_val):.3e}  SNR_LE={le_snr:.1f}  → {ident}")


# ── Figure ────────────────────────────────────────────────────────────────────
param_names = ["alpha_sw", "alpha_tref", "alpha_vcmax25", "alpha_iota"]
gpp_vals = [
    abs(jax_results.get("alpha_sw",     (0,0))[0]),
    abs(jax_results.get("alpha_tref",   (0,0))[0]),
    abs(jax_results.get("alpha_vcmax25",(0,0))[0]),
    abs(dgpp_iota),
]
le_vals = [
    abs(jax_results.get("alpha_sw",     (0,0))[1]),
    abs(jax_results.get("alpha_tref",   (0,0))[1]),
    abs(jax_results.get("alpha_vcmax25",(0,0))[1]),
    abs(dle_iota),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
x = np.arange(len(param_names))
w = 0.35

axes[0].bar(x, gpp_vals, color=["steelblue", "steelblue", "steelblue", "coral"])
axes[0].set_xticks(x)
axes[0].set_xticklabels(param_names, fontsize=9)
axes[0].set_ylabel("|dGPP/dα|  (agross·dpai units)")
axes[0].set_title("GPP sensitivity to scale factors")
axes[0].set_yscale("log")
axes[0].text(3, dle_iota * 0.5, "FD only", ha="center", fontsize=8, color="coral")

axes[1].bar(x, le_vals, color=["steelblue", "steelblue", "steelblue", "coral"])
axes[1].set_xticks(x)
axes[1].set_xticklabels(param_names, fontsize=9)
axes[1].set_ylabel("|dLE/dα|  (lhleaf·dpai units)")
axes[1].set_title("LE sensitivity to scale factors")
axes[1].set_yscale("log")

fig.suptitle("CLM-ML-JAX parameter sensitivity at CHATS7 operating point\n"
             "(blue=JAX grad, coral=FD only)", fontsize=11)
fig.tight_layout()
out = FIGURES_DIR / "param_sensitivity.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved: {out}")
