"""
sensitivity_analysis_v2.py — Updated 7-parameter Jacobian under WUE stomatal model.

Replaces sensitivity_analysis.py (which used {vcmax, tair, sw, q_ref, dpai}).
This version uses all 7 parameters confirmed correct in job 7589868 + 7600635:

  0: alpha_vcmax  — Vcmax25 (vcmaxpft_jax explicit arg)
  1: alpha_tref   — air temperature (forc_t_downscaled_col)
  2: alpha_sw     — shortwave radiation (forc_solad + forc_solai)
  3: alpha_iota   — WUE stomatal efficiency iota_SPA (module-global mutation)
  4: alpha_q      — specific humidity (forc_q_downscaled_col)
  5: alpha_pbot   — atmospheric pressure (forc_pbot_downscaled_col)
  6: alpha_u      — wind u-component (forc_u_grc)

Stomatal model: WUE (gs_type=2) — iota_SPA is active, g1_MED is not.
All 7 parameters confirmed correct (rel err < 1e-3) under WUE at CHATS7.

Outputs:
  diags/figures/sensitivity_jacobian_v2.csv
  diags/figures/sensitivity_jacobian_v2.png
  diags/figures/sensitivity_jacobian_v2.pdf

Usage:
  python diags/sensitivity_analysis_v2.py
  python diags/sensitivity_analysis_v2.py --plot-only
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

_PLOT_ONLY = "--plot-only" in sys.argv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _d in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_PARAMS  = 7
N_OUTPUTS = 3

PARAM_NAMES = [
    r"$V_{c,\max25}$",
    r"$T_\mathrm{air}$",
    r"$SW_\mathrm{rad}$",
    r"$\iota_\mathrm{SPA}$",
    r"$q$",
    r"$P_\mathrm{bot}$",
    r"$u$",
]
PLOT_PARAM_INDICES = (0, 1, 2, 3, 4, 6)
PLOT_PARAM_NAMES = [PARAM_NAMES[i] for i in PLOT_PARAM_INDICES]
OUTPUT_NAMES = ["GPP", "H", "LE"]


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_jacobian(J_np: np.ndarray, param_names: list[str]) -> None:
    n_out, n_par = J_np.shape
    output_scale = np.abs(J_np).max(axis=1, keepdims=True) + 1e-30
    J_norm = J_np / output_scale
    log_J  = np.log10(np.abs(J_np) + 1e-30)

    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))
    fig.subplots_adjust(wspace=0.35)

    # Panel (a): log magnitude
    vmax = max(abs(log_J.max()), abs(log_J.min()))
    im0 = axes[0].imshow(log_J, aspect="auto", cmap="RdYlBu_r",
                          vmin=-vmax, vmax=vmax)
    axes[0].set_xticks(range(n_par))
    axes[0].set_xticklabels(param_names, fontsize=10)
    axes[0].set_yticks(range(n_out))
    axes[0].set_yticklabels(OUTPUT_NAMES, fontsize=10)
    for i in range(n_out):
        for j in range(n_par):
            axes[0].text(j, i, f"{log_J[i,j]:.2f}",
                         ha="center", va="center", fontsize=8, color="k")
    axes[0].set_title(r"(a) $\log_{10}|\partial\,\mathrm{output}/\partial\alpha_i|$",
                      fontsize=11, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], label=r"$\log_{10}|J|$")

    # Panel (b): normalised sensitivity
    divnorm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im1 = axes[1].imshow(J_norm, aspect="auto", cmap="RdBu_r", norm=divnorm)
    axes[1].set_xticks(range(n_par))
    axes[1].set_xticklabels(param_names, fontsize=10)
    axes[1].set_yticks(range(n_out))
    axes[1].set_yticklabels(OUTPUT_NAMES, fontsize=10)
    for i in range(n_out):
        for j in range(n_par):
            axes[1].text(j, i, f"{J_norm[i,j]:+.2f}",
                         ha="center", va="center", fontsize=8, color="k")
    axes[1].set_title(r"(b) Normalised sensitivity $\partial\,\mathrm{output}/\partial\alpha_i$",
                      fontsize=11, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], ticks=[-1, -0.5, 0, 0.5, 1])

    #fig.suptitle(
    #    r"CLM-ml-jax: 7-parameter Jacobian $\partial(\mathrm{GPP},H,\mathrm{LE})/\partial\boldsymbol{\theta}$"
    #    "\nWUE stomatal model · CHATS7 · 1 May 2007 · GPU · jax.jacrev",
    #    fontsize=9, y=1.02,
    #)
    fig.tight_layout()

    for fmt in ("png", "pdf"):
        out = FIGURES_DIR / f"sensitivity_jacobian_v2.{fmt}"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"Saved: {out}", flush=True)
    plt.close(fig)


# ── --plot-only: reload CSV and replot ────────────────────────────────────────
if _PLOT_ONLY:
    csv_path = FIGURES_DIR / "sensitivity_jacobian_v2.csv"
    if not csv_path.exists():
        sys.exit(f"ERROR: {csv_path} not found.")
    J_np = np.zeros((N_OUTPUTS, len(PLOT_PARAM_INDICES)), dtype=np.float64)
    with open(csv_path, newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0:
                continue
            values = [float(v) for v in row[1:]]
            J_np[i - 1] = [values[j] for j in PLOT_PARAM_INDICES]
    plot_jacobian(J_np, PLOT_PARAM_NAMES)
    sys.exit(0)


# ── Full run ──────────────────────────────────────────────────────────────────
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import multilayer_canopy.MLpftconMod             as _pftcon_mod
import multilayer_canopy.MLLeafPhotosynthesisMod  as _leaf_mod
import multilayer_canopy.MLCanopyNitrogenProfileMod as _nitro_mod

from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs,
    MLCanopyFluxes,
    atm2lnd_inst, wateratm2lndbulk_inst,
    compute_gpp, compute_le, compute_h,
)

_orig_pftcon = _pftcon_mod.MLpftcon
_p   = grid.p
_n   = grid.ncan

_mlcf_kwargs_base = {k: v for k, v in _mlcf_kwargs.items()
                     if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst")}


def _set_pftcon(new_inst):
    _pftcon_mod.MLpftcon = new_inst
    _leaf_mod.MLpftcon   = new_inst
    _nitro_mod.MLpftcon  = new_inst


def _restore_pftcon():
    _pftcon_mod.MLpftcon = _orig_pftcon
    _leaf_mod.MLpftcon   = _orig_pftcon
    _nitro_mod.MLpftcon  = _orig_pftcon


def forward_multi(scales: jnp.ndarray) -> jnp.ndarray:
    """7-output forward function for jax.jacrev.

    scales[0] alpha_vcmax  — Vcmax25 (explicit JAX arg, bypasses JIT cache)
    scales[1] alpha_tref   — air temperature
    scales[2] alpha_sw     — shortwave radiation (direct + diffuse)
    scales[3] alpha_iota   — WUE iota_SPA (module-global mutation)
    scales[4] alpha_q      — specific humidity
    scales[5] alpha_pbot   — atmospheric pressure
    scales[6] alpha_u      — wind u-component
    """
    # Vcmax25: explicit JAX arg — bypasses lru_cache, gradient confirmed.
    vcmaxpft_jax = scales[0] * jnp.asarray(_orig_pftcon.vcmaxpft)

    # iota_SPA: module-global mutation — confirmed correct in job 7589868.
    _set_pftcon(_orig_pftcon._replace(
        iota_SPA=scales[3] * jnp.asarray(_orig_pftcon.iota_SPA)
    ))

    modified_atm = atm2lnd_inst._replace(
        forc_t_downscaled_col     = scales[1] * atm2lnd_inst.forc_t_downscaled_col,
        forc_solad_downscaled_col = scales[2] * atm2lnd_inst.forc_solad_downscaled_col,
        forc_solai_grc            = scales[2] * atm2lnd_inst.forc_solai_grc,
        forc_pbot_downscaled_col  = scales[5] * atm2lnd_inst.forc_pbot_downscaled_col,
        forc_u_grc                = scales[6] * atm2lnd_inst.forc_u_grc,
    )
    modified_wat = wateratm2lndbulk_inst._replace(
        forc_q_downscaled_col = scales[4] * wateratm2lndbulk_inst.forc_q_downscaled_col,
    )

    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=modified_wat,
        vcmaxpft_jax=vcmaxpft_jax,
        **_mlcf_kwargs_base,
    )
    _restore_pftcon()

    return jnp.array([
        compute_gpp(inst, _p, _n),
        compute_h(inst, _p, _n),
        compute_le(inst, _p, _n),
    ])


# ── Baseline ──────────────────────────────────────────────────────────────────
print("\n=== Baseline outputs ===", flush=True)
scales0  = jnp.ones(N_PARAMS, dtype=jnp.float64)
baseline = forward_multi(scales0)
jax.block_until_ready(baseline)
print(f"  GPP = {float(baseline[0]):.4f} umol CO2/m2/s", flush=True)
print(f"  H   = {float(baseline[1]):.4f} W/m2", flush=True)
print(f"  LE  = {float(baseline[2]):.4f} W/m2", flush=True)

# ── Jacobian ──────────────────────────────────────────────────────────────────
print(f"\n=== jax.jacrev ({N_OUTPUTS} outputs × {N_PARAMS} params) ===", flush=True)
jacrev_fn = jax.jit(jax.jacrev(forward_multi))

t0 = time.time()
J  = jacrev_fn(scales0)
jax.block_until_ready(J)
t_jacrev = time.time() - t0
print(f"  jacrev completed in {t_jacrev:.1f}s", flush=True)

J_np = np.array(J)
header = "  ".join(f"{p:>14s}" for p in PARAM_NAMES)
print(f"\n  {'':6s}  {header}", flush=True)
for i, oname in enumerate(OUTPUT_NAMES):
    row = "  ".join(f"{J_np[i,j]:>14.4e}" for j in range(N_PARAMS))
    print(f"  {oname:<6s}  {row}", flush=True)

# ── Timing comparison ─────────────────────────────────────────────────────────
t0    = time.time(); _ = forward_multi(scales0); jax.block_until_ready(_)
t_fwd = time.time() - t0
print(f"\n  Single forward pass:  {t_fwd:.2f}s", flush=True)
print(f"  jacrev total:         {t_jacrev:.1f}s", flush=True)
print(f"  FD equivalent (2p={2*N_PARAMS} calls): ~{2*N_PARAMS*t_fwd:.1f}s", flush=True)
print(f"  T_ratio (jacrev/fwd): {t_jacrev/t_fwd:.2f}", flush=True)

# ── CSV ───────────────────────────────────────────────────────────────────────
csv_path = FIGURES_DIR / "sensitivity_jacobian_v2.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["output"] + PARAM_NAMES)
    for i, oname in enumerate(OUTPUT_NAMES):
        w.writerow([oname] + list(J_np[i]))
print(f"\nCSV saved: {csv_path}", flush=True)

# ── Figure ────────────────────────────────────────────────────────────────────
plot_jacobian(J_np, PARAM_NAMES)
print("\n=== sensitivity_analysis_v2.py complete ===", flush=True)
