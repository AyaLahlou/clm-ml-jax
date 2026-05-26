"""
Experiment 2: Finite-difference gradient check.

Verifies that jax.grad produces accurate gradients through the full
CLM-ml-jax column by comparing against central finite differences for
parameters with clear, non-trivial pathways to canopy GPP:

  - alpha_sw:      global scale factor on swskyb_forcing & swskyd_forcing
                   (solar radiation → PAR → electron transport → GPP)
  - alpha_tref:    global scale factor on forc_t_downscaled_col
                   (temperature → enzyme kinetics → Vcmax(T)/Jmax(T) → GPP)
  - alpha_g1_MED:  global scale factor on g1_MED (Medlyn stomatal slope)
                   (g1 → Ci solver → gs → GPP; only active when gs_type==0)
  - alpha_iota:    global scale factor on iota_SPA (WUE efficiency parameter)
                   (iota → _bisect_gs_ift via IFT → gs_opt → GPP; gs_type==2)
  - alpha_vcmax:   global scale factor on vcmaxpft (max carboxylation at 25°C)
                   (vcmaxpft → vcmax25top → vcmax_leaf → ac → agross → GPP)

Output scalar: gppveg_canopy[p]  (total canopy GPP, umol CO2/m2/s)

Usage (from project root):
    cd src && python ../diags/fd_grad_check.py

Output:
  - Console table: parameter, jax.grad value, FD value, relative error
  - diags/figures/fd_grad_check.png: bar chart of relative errors
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root (parent of diags/) is on sys.path so 'diags' is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared init ────────────────────────────���──────────────────────────────────
from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs, jax, jnp, MLCanopyFluxes,
    atm2lnd_inst, wateratm2lndbulk_inst, compute_gpp,
)
import multilayer_canopy.MLpftconMod              as _MLpftconMod
import multilayer_canopy.MLLeafPhotosynthesisMod  as _LeafMod
import multilayer_canopy.MLCanopyNitrogenProfileMod as _NitroMod

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_p = grid.p

# Build kwargs without atm2lnd_inst so we can pass it as a traced arg
_mlcf_kwargs_no_atm = {k: v for k, v in _mlcf_kwargs.items()
                       if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst")}

# ── MLpftcon module injection helpers ─────────────────────────────────────────
# NamedTuples are immutable — we cannot set `MLpftcon.g1_MED = ...`.
# We must replace the module-level variable AND update all already-imported
# physics module references so they see the new instance.
# Each physics module does `from MLpftconMod import MLpftcon` at import time,
# binding the NAME to the original object.  Replacing `MLpftconMod.MLpftcon`
# alone does NOT update those local bindings.  We must also update:
#   _LeafMod.MLpftcon  — for g1_MED / g0_MED / iota_SPA / gsmin_SPA
#   _NitroMod.MLpftcon — for vcmaxpft / clump_fac

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


# ── GPP scalar forward functions ─────────────────────────────────────────────
# IMPORTANT: MLCanopyFluxes.__init__ overwrites swskyb_cur_forcing and
# tref_cur_forcing from atm2lnd_inst (lines 910-921 of MLCanopyFluxesMod.py).
# Gradients must flow through atm2lnd_inst, NOT mlcanopy_inst forcing fields.

def forward_gpp_sw(alpha: jnp.ndarray) -> jnp.ndarray:
    """Forward pass: scale beam+diffuse SW by alpha via atm2lnd_inst, return GPP."""
    modified_atm = atm2lnd_inst._replace(
        forc_solad_downscaled_col = alpha * atm2lnd_inst.forc_solad_downscaled_col,
        forc_solai_grc            = alpha * atm2lnd_inst.forc_solai_grc,
    )
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        **_mlcf_kwargs_no_atm,
    )
    return compute_gpp(inst, _p, grid.ncan)


def forward_gpp_tref(alpha: jnp.ndarray) -> jnp.ndarray:
    """Forward pass: scale air temperature by alpha via atm2lnd_inst, return GPP.

    Temperature affects photosynthesis via enzyme kinetics (Vcmax(T), Jmax(T)).
    """
    modified_atm = atm2lnd_inst._replace(
        forc_t_downscaled_col = alpha * atm2lnd_inst.forc_t_downscaled_col,
    )
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        **_mlcf_kwargs_no_atm,
    )
    return compute_gpp(inst, _p, grid.ncan)


def forward_gpp_g1_MED(alpha: jnp.ndarray) -> jnp.ndarray:
    """Forward pass: scale g1_MED[all PFTs] by alpha, return GPP.

    g1_MED is the Medlyn stomatal slope.  Active when gs_type == 0 (Medlyn).
    Gradient path: g1_MED → gs (quadratic solve in kernel) → Ci → GPP.
    """
    _set_pftcon(_orig_pftcon._replace(g1_MED=alpha * _orig_pftcon.g1_MED))
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=atm2lnd_inst,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        **_mlcf_kwargs_no_atm,
    )
    _restore_pftcon()
    return compute_gpp(inst, _p, grid.ncan)


def forward_gpp_iota(alpha: jnp.ndarray) -> jnp.ndarray:
    """Forward pass: scale iota_SPA[all PFTs] by alpha, return GPP.

    iota_SPA is the WUE efficiency parameter.  Active when gs_type == 2 (WUE).
    Gradient path: iota → _bisect_gs_ift (IFT) → gs_opt → Ci → GPP.
    """
    _set_pftcon(_orig_pftcon._replace(iota_SPA=alpha * _orig_pftcon.iota_SPA))
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=atm2lnd_inst,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        **_mlcf_kwargs_no_atm,
    )
    _restore_pftcon()
    return compute_gpp(inst, _p, grid.ncan)


def forward_gpp_vcmaxpft(alpha: jnp.ndarray) -> jnp.ndarray:
    """Forward pass: scale vcmaxpft[all PFTs] by alpha, return GPP.

    vcmaxpft is the maximum carboxylation rate at 25°C.
    Gradient path: vcmaxpft → vcmax25top (NitrogenProfile) → vcmax25_leaf →
                   vcmax_leaf (T-response in kernel) → ac (Rubisco) → agross → GPP.
    Note: vcmaxpft_jax is passed directly as a JAX arg to MLCanopyFluxes so that
    jax.grad can trace through it.  Module-global mutation (_set_pftcon) is NOT
    used here because CanopyNitrogenProfile is @jax.jit-cached and bakes in
    MLpftcon.vcmaxpft as a compile-time constant.
    """
    vcmaxpft_jax = alpha * _orig_pftcon.vcmaxpft
    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=atm2lnd_inst,
        wateratm2lndbulk_inst=wateratm2lndbulk_inst,
        vcmaxpft_jax=vcmaxpft_jax,
        **_mlcf_kwargs_no_atm,
    )
    return compute_gpp(inst, _p, grid.ncan)


# ── Print baseline values ────────────────────────────────────────────────────
from multilayer_canopy.MLclm_varctl import gs_type as _gs_type
print("\n=== Baseline outputs ===", flush=True)
baseline_gpp = float(forward_gpp_sw(jnp.float64(1.0)))
print(f"  GPP proxy (baseline) = {baseline_gpp:.4f} (agross_leaf weighted sum)", flush=True)
_gs_name = {0: "Medlyn (gs_type=0)", 1: "Ball-Berry (gs_type=1)", 2: "WUE (gs_type=2)"}.get(_gs_type, f"unknown (gs_type={_gs_type})")
print(f"  Stomatal model: {_gs_name}", flush=True)
print(f"  NOTE: dGPP/d(alpha_g1) active only for gs_type==0; dGPP/d(alpha_iota) active only for gs_type==2", flush=True)

# ── Compute JAX gradients ─────────────────────────────────────────────────────
print("\n=== Computing JAX gradients ===", flush=True)

t0 = time.time()
grad_sw_jax = float(jax.jit(jax.grad(forward_gpp_sw))(jnp.float64(1.0)))
print(f"  dGPP/d(alpha_sw)   [JAX] = {grad_sw_jax:.6e}  ({time.time()-t0:.1f}s)",
      flush=True)

t0 = time.time()
grad_tref_jax = float(jax.jit(jax.grad(forward_gpp_tref))(jnp.float64(1.0)))
print(f"  dGPP/d(alpha_tref) [JAX] = {grad_tref_jax:.6e}  ({time.time()-t0:.1f}s)",
      flush=True)

t0 = time.time()
grad_g1_jax = float(jax.jit(jax.grad(forward_gpp_g1_MED))(jnp.float64(1.0)))
print(f"  dGPP/d(alpha_g1)   [JAX] = {grad_g1_jax:.6e}  ({time.time()-t0:.1f}s)",
      flush=True)

t0 = time.time()
grad_iota_jax = float(jax.jit(jax.grad(forward_gpp_iota))(jnp.float64(1.0)))
print(f"  dGPP/d(alpha_iota) [JAX] = {grad_iota_jax:.6e}  ({time.time()-t0:.1f}s)",
      flush=True)

t0 = time.time()
grad_vcmax_jax = float(jax.jit(jax.grad(forward_gpp_vcmaxpft))(jnp.float64(1.0)))
print(f"  dGPP/d(alpha_vcmax)[JAX] = {grad_vcmax_jax:.6e}  ({time.time()-t0:.1f}s)",
      flush=True)

# ── Compute finite-difference gradients ───────────────────────────────────────
print("\n=== Computing finite-difference gradients ===", flush=True)
EPS = 1e-4

t0 = time.time()
f_plus  = float(forward_gpp_sw(jnp.float64(1.0 + EPS)))
f_minus = float(forward_gpp_sw(jnp.float64(1.0 - EPS)))
grad_sw_fd = (f_plus - f_minus) / (2 * EPS)
print(f"  dGPP/d(alpha_sw)   [FD]  = {grad_sw_fd:.6e}  ({time.time()-t0:.1f}s)",
      flush=True)

t0 = time.time()
f_plus  = float(forward_gpp_tref(jnp.float64(1.0 + EPS)))
f_minus = float(forward_gpp_tref(jnp.float64(1.0 - EPS)))
grad_tref_fd = (f_plus - f_minus) / (2 * EPS)
print(f"  dGPP/d(alpha_tref) [FD]  = {grad_tref_fd:.6e}  ({time.time()-t0:.1f}s)",
      flush=True)

t0 = time.time()
f_plus  = float(forward_gpp_g1_MED(jnp.float64(1.0 + EPS)))
f_minus = float(forward_gpp_g1_MED(jnp.float64(1.0 - EPS)))
grad_g1_fd = (f_plus - f_minus) / (2 * EPS)
print(f"  dGPP/d(alpha_g1)   [FD]  = {grad_g1_fd:.6e}  ({time.time()-t0:.1f}s)",
      flush=True)

t0 = time.time()
f_plus  = float(forward_gpp_iota(jnp.float64(1.0 + EPS)))
f_minus = float(forward_gpp_iota(jnp.float64(1.0 - EPS)))
grad_iota_fd = (f_plus - f_minus) / (2 * EPS)
print(f"  dGPP/d(alpha_iota) [FD]  = {grad_iota_fd:.6e}  ({time.time()-t0:.1f}s)",
      flush=True)

t0 = time.time()
f_plus  = float(forward_gpp_vcmaxpft(jnp.float64(1.0 + EPS)))
f_minus = float(forward_gpp_vcmaxpft(jnp.float64(1.0 - EPS)))
grad_vcmax_fd = (f_plus - f_minus) / (2 * EPS)
print(f"  dGPP/d(alpha_vcmax)[FD]  = {grad_vcmax_fd:.6e}  ({time.time()-t0:.1f}s)",
      flush=True)

# ── Report ────────────────────────────────────────────────────────────────────
print("\n=== Gradient accuracy summary ===")
print(f"{'Parameter':<24}  {'JAX grad':>14}  {'FD grad':>14}  {'Rel error':>12}  {'Pass?':>6}")
print("-" * 80)

results = []
for name, jax_val, fd_val in [
    ("dGPP/d(alpha_sw)",    grad_sw_jax,    grad_sw_fd),
    ("dGPP/d(alpha_tref)",  grad_tref_jax,  grad_tref_fd),
    ("dGPP/d(alpha_g1)",    grad_g1_jax,    grad_g1_fd),
    ("dGPP/d(alpha_iota)",  grad_iota_jax,  grad_iota_fd),
    ("dGPP/d(alpha_vcmax)", grad_vcmax_jax, grad_vcmax_fd),
]:
    both_tiny = abs(jax_val) < 1e-6 and abs(fd_val) < 1e-6
    if both_tiny:
        rel_err = 0.0
        passed  = True
        status  = "INACT"   # parameter inactive for current gs_type
    else:
        rel_err = abs(jax_val - fd_val) / (abs(fd_val) + 1e-30)
        passed  = rel_err < 0.01  # 1% tolerance
        status  = "PASS" if passed else "FAIL"
    results.append((name, jax_val, fd_val, rel_err, passed))
    print(f"  {name:<20}  {jax_val:>14.4e}  {fd_val:>14.4e}  {rel_err:>12.2e}  {status:>6}")

all_pass = all(r[4] for r in results)
print(f"\n{'ALL PASS' if all_pass else 'SOME FAILURES'} — "
      f"autodiff accuracy confirmed to {'<1%' if all_pass else '>1%'} relative error")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

names    = [r[0] for r in results]
jax_vals = np.abs([r[1] for r in results])
fd_vals  = np.abs([r[2] for r in results])
rel_errs = [r[3] for r in results]

x = np.arange(len(names))
w = 0.35
axes[0].bar(x - w/2, jax_vals, w, label="jax.grad", color="steelblue")
axes[0].bar(x + w/2, fd_vals,  w, label="FD (central)", color="coral", alpha=0.8)
axes[0].set_yscale("log")
axes[0].set_xticks(x)
axes[0].set_xticklabels(names)
axes[0].set_ylabel("|gradient|")
axes[0].set_title("Gradient magnitudes: JAX vs FD")
axes[0].legend()

axes[1].bar(x, rel_errs, color=["green" if r[4] else "red" for r in results])
axes[1].axhline(0.01, color="k", linestyle="--", lw=1.2, label="1% threshold")
axes[1].set_xticks(x)
axes[1].set_xticklabels(names)
axes[1].set_ylabel("Relative error")
axes[1].set_title("Gradient relative error (< 1% = pass)")
axes[1].legend()

fig.suptitle("CLM-ml-jax: Autodiff accuracy — dGPP/d(param) via jax.grad vs central FD\n(Exp 2)", fontsize=12)
fig.tight_layout()
out = FIGURES_DIR / "fd_grad_check.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved: {out}")
