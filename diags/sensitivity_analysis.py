"""
Experiment 3: Jacobian-based global sensitivity analysis.

Computes the full output-parameter Jacobian J = d[GPP, H, LE] / d[theta]
via reverse-mode autodiff (jax.jacrev), where theta is a vector of
five physiological and forcing scale factors:
  0: alpha_vcmax25  — scale on vcmax25_profile & vcmax25_leaf (Vcmax25)
  1: alpha_tref     — scale on tref_forcing (air temperature)
  2: alpha_sw       — scale on swskyb_forcing & swskyd_forcing (solar radiation)
  3: alpha_qref     — scale on qref_forcing (specific humidity / VPD)
  4: alpha_dpai     — scale on dpai_profile (canopy leaf area)

Outputs: [GPP, H_sum, LE_sum] at the single CHATS7 patch.

Usage (from project root):
    cd src && python ../diags/sensitivity_analysis.py           # full run
    python diags/sensitivity_analysis.py --plot-only            # replot from CSV

Output:
  - Console: Jacobian values and runtime comparison with FD cost estimate
  - diags/figures/sensitivity_jacobian.png: normalised heatmap (paper figure)
  - diags/figures/sensitivity_jacobian.pdf: vector version
  - diags/figures/sensitivity_jacobian.csv: raw Jacobian values
"""
from __future__ import annotations

import sys
import csv
import time
from pathlib import Path

# ── Argument parsing (before any heavy imports) ───────────────────────────────
_PLOT_ONLY = "--plot-only" in sys.argv

# Ensure project root (parent of diags/) is on sys.path so 'diags' is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Parameter / output names ──────────────────────────────────────────────────
PARAM_NAMES  = ["Vcmax25", "T_air", "SW_rad", "q_ref\n", "dpai\n"]
OUTPUT_NAMES = ["GPP", "H", "LE\n"]


# ── Plotting function (used in both modes) ────────────────────────────────────
def plot_jacobian(J_np: np.ndarray) -> None:
    """Save sensitivity_jacobian.png and .pdf from a (3, 5) Jacobian array."""
    output_scale = np.abs(J_np).max(axis=1, keepdims=True) + 1e-30
    J_norm = J_np / output_scale

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left: absolute Jacobian (log scale magnitude)
    log_J = np.log10(np.abs(J_np) + 1e-30)
    im0 = axes[0].imshow(log_J, aspect="auto", cmap="RdYlBu_r", vmin= log_J.min(), vmax=log_J.max())
    axes[0].set_xticks(range(5))
    axes[0].set_xticklabels(PARAM_NAMES, fontsize=9)
    axes[0].set_yticks(range(3))
    axes[0].set_yticklabels(OUTPUT_NAMES, fontsize=9)
    for i in range(3):
        for j in range(5):
            axes[0].text(j, i, f"{log_J[i,j]:.2f}", ha="center", va="center",
                         fontsize=9, color="k")
    axes[0].set_title("(a) Log scale magnitude: log₁₀|∂output/∂scale|", fontsize=11, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], label="log₁₀|J|", ticks=[-1, 0, 1, 2, 3])

    # Right: row-normalised (relative sensitivity within each output)
    divnorm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im1 = axes[1].imshow(J_norm, aspect="auto", cmap="RdYlBu_r", norm=divnorm)
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels(PARAM_NAMES, fontsize=9)
    axes[1].set_yticks(range(3))
    axes[1].set_yticklabels(OUTPUT_NAMES, fontsize=9)
    for i in range(3):
        for j in range(5):
            axes[1].text(j, i, f"{J_norm[i,j]:+.2f}", ha="center", va="center",
                         fontsize=9, color="k")
    axes[1].set_title("(b) Normalised sensitivity: ∂output/∂scale", fontsize=11, fontweight="bold")
    plt.colorbar(im1, ax=axes[1],
                 ticks=[-1, -0.5, 0, 0.5, 1])

    fig.tight_layout()

    out_png = FIGURES_DIR / "sensitivity_jacobian.png"
    out_pdf = FIGURES_DIR / "sensitivity_jacobian.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_png}")
    print(f"Figure saved: {out_pdf}")


# ── --plot-only mode: load CSV and replot, then exit ─────────────────────────
if _PLOT_ONLY:
    csv_path = FIGURES_DIR / "sensitivity_jacobian.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run without --plot-only first.", flush=True)
        sys.exit(1)
    print(f"Loading Jacobian from {csv_path} ...", flush=True)
    J_np = np.zeros((3, 5), dtype=np.float64)
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for i, row in enumerate(reader):
            if i >= 3:
                break
            J_np[i] = [float(v) for v in row[1:6]]  # skip output-name column
    print("Jacobian loaded:")
    print(f"  {'':18s}  " + "  ".join(f"{p.split(chr(10))[0]:>8s}" for p in PARAM_NAMES))
    for i, oname in enumerate(OUTPUT_NAMES):
        row_str = "  ".join(f"{J_np[i,j]:>10.4e}" for j in range(5))
        print(f"  {oname.split(chr(10))[0]:18s}  {row_str}")
    plot_jacobian(J_np)
    sys.exit(0)


# ── Full-run mode: import JAX + model, compute Jacobian ──────────────────────
import multilayer_canopy.MLpftconMod as _pftcon_mod

from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs,
    jax, jnp, MLCanopyFluxes,
    atm2lnd_inst, wateratm2lndbulk_inst, compute_gpp, compute_le, compute_h,
)

_orig_pftcon = _pftcon_mod.MLpftcon   # original MLpftcon instance

_p   = grid.p
_n   = grid.ncan

# Build kwargs without atm2lnd_inst/wateratm2lndbulk_inst/canopystate_inst
# so we can pass them as traced args in forward_multi.
_canopystate_inst = _mlcf_kwargs["canopystate_inst"]
_mlcf_kwargs_base = {k: v for k, v in _mlcf_kwargs.items()
                     if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst",
                                  "canopystate_inst")}

# ── Multi-output forward function ────────────────────────────────────────────
def forward_multi(scales: jnp.ndarray) -> jnp.ndarray:
    """
    Args:
        scales: float64 array of shape (5,) — scale factors at baseline = 1.0
          0: alpha_vcmax25  — scale on vcmaxpft (passed as vcmaxpft_jax arg)
          1: alpha_tair     — scale on atm2lnd_inst.forc_t_downscaled_col
          2: alpha_sw       — scale on atm2lnd_inst.forc_solad & forc_solai
          3: alpha_qref     — scale on wateratm2lndbulk_inst.forc_q_downscaled_col
          4: alpha_dpai     — scale on canopystate_inst.elai_patch & esai_patch

    Returns:
        jnp.ndarray of shape (3,): [GPP (umol CO2/m2/s), H_canopy (W/m2), LE_canopy (W/m2)]

    NOTE: Vcmax25 is passed via vcmaxpft_jax (bypasses JIT cache so gradient flows).
    Scaling vcmax25_profile/vcmax25_leaf directly in mlcanopy_inst does NOT work
    because CanopyNitrogenProfile (called inside _physics_step_fn) recomputes them
    from MLpftcon.vcmaxpft, overwriting the scaled values.

    T_air, SW_rad, q_ref are scaled via atm2lnd_inst/wateratm2lndbulk_inst because
    MLCanopyFluxes.__init__ copies forcing fields from those instances into
    mlcanopy_inst, overwriting any mlcanopy_inst forcing fields.

    dpai MUST be scaled via canopystate_inst.elai_patch / esai_patch, NOT via
    mlcanopy_inst.dpai_profile directly.  MLCanopyFluxes recomputes dpai_profile
    at the start of every call (lines ~430-434 of MLCanopyFluxesMod.py):
        dpai[p, 1:ncan+1] = dlai_frac_profile * elai_patch[p]
                           + dsai_frac_profile * esai_patch[p]
    This overwrites any scaled dpai_profile stored in mlcanopy_inst, breaking
    the gradient tape.  Scaling elai_patch/esai_patch propagates through this
    recomputation so the gradient flows correctly.

    H and LE are computed from shleaf_leaf and lhleaf_leaf (leaf-level, weighted over
    canopy layers) — updated inside _physics_step_fn even in diff mode.
    """
    # Scale vcmax via vcmaxpft_jax — bypasses JIT cache so gradient flows correctly.
    vcmaxpft_jax = scales[0] * _orig_pftcon.vcmaxpft

    # Scale LAI/SAI in canopystate_inst — this is the actual source of dpai_profile.
    # (scaling mlcanopy_inst.dpai_profile directly is silently overwritten by MLCanopyFluxes)
    modified_canopy = _canopystate_inst._replace(
        elai_patch = scales[4] * jnp.asarray(_canopystate_inst.elai_patch, dtype=jnp.float64),
        esai_patch = scales[4] * jnp.asarray(_canopystate_inst.esai_patch, dtype=jnp.float64),
    )

    # Scale forcing via atm2lnd_inst (actual source used by physics).
    modified_atm = atm2lnd_inst._replace(
        forc_t_downscaled_col     = scales[1] * atm2lnd_inst.forc_t_downscaled_col,
        forc_solad_downscaled_col = scales[2] * atm2lnd_inst.forc_solad_downscaled_col,
        forc_solai_grc            = scales[2] * atm2lnd_inst.forc_solai_grc,
    )
    modified_watm = wateratm2lndbulk_inst._replace(
        forc_q_downscaled_col = scales[3] * wateratm2lndbulk_inst.forc_q_downscaled_col,
    )

    inst = MLCanopyFluxes(
        mlcanopy_inst=mlcanopy_inst,
        atm2lnd_inst=modified_atm,
        wateratm2lndbulk_inst=modified_watm,
        canopystate_inst=modified_canopy,
        vcmaxpft_jax=vcmaxpft_jax,
        **_mlcf_kwargs_base,
    )

    # Use leaf-level proxies (shleaf_leaf, lhleaf_leaf, agross_leaf) — all updated
    # by _physics_step_fn in diff mode.  _CanopyFluxesDiagnostics is skipped in
    # diff mode so gppveg_canopy, eflx_sh_tot, eflx_lh_tot are NOT updated.
    GPP = compute_gpp(inst, _p, _n)
    H   = compute_h(inst, _p, _n)
    LE  = compute_le(inst, _p, _n)

    return jnp.array([GPP, H, LE])


# ── Baseline outputs ─────────────────────────────────────────────────────────
print("\n=== Computing baseline outputs ===", flush=True)
scales0  = jnp.ones(5, dtype=jnp.float64)
baseline = forward_multi(scales0)
jax.block_until_ready(baseline)
print(f"  GPP = {float(baseline[0]):.3f} umol CO2/m2/s")
print(f"  H   = {float(baseline[1]):.3f} W/m2 (top canopy layer)")
print(f"  LE  = {float(baseline[2]):.6f} mol H2O/m2/s (top canopy layer)")

# ── Jacobian via jacfwd ───────────────────────────────────────────────────────
print("\n=== Computing Jacobian via jax.jacrev ===", flush=True)
jacrev_fn = jax.jit(jax.jacrev(forward_multi))

t0 = time.time()
J = jacrev_fn(scales0)      # shape: (3, 5)
jax.block_until_ready(J)
t_jacrev = time.time() - t0
print(f"  jacrev completed in {t_jacrev:.1f}s  (shape: {J.shape})")

J_np = np.array(J)
print("\n  Raw Jacobian J[i,j] = d(output_i)/d(scale_j):")
print(f"  {'':15s}  " + "  ".join(f"{p:>12s}" for p in PARAM_NAMES))
for i, oname in enumerate(OUTPUT_NAMES):
    row = "  ".join(f"{J_np[i,j]:>12.4e}" for j in range(5))
    print(f"  {oname:15s}  {row}")

# ── Normalised sensitivity ────────────────────────────────────────────────────
# Normalize: J_norm[i,j] = J[i,j] / std(output_i)  so rows are comparable
# Using output baseline values as approximate std (1% perturbation interpretation)
output_scale = np.abs(J_np).max(axis=1, keepdims=True) + 1e-30
J_norm = J_np / output_scale

# ── Runtime comparison ────────────────────────────────────────────────────────
n_params = 5
n_outputs = 3
print(f"\n=== Runtime comparison ===")
print(f"  jacrev (autodiff, {n_outputs} outputs, {n_params} params): {t_jacrev:.1f}s")
print(f"  (jacrev cost ≈ {n_outputs} backward passes; FD would need {n_params}×2 = {n_params*2} forward evaluations)")
# Estimate single forward pass time for comparison
t0 = time.time()
_ = forward_multi(scales0)
jax.block_until_ready(_)
t_fwd = time.time() - t0
print(f"  Single forward pass: {t_fwd:.1f}s")
print(f"  FD equivalent cost estimate: ~{n_params * 2 * t_fwd:.1f}s")

# ── Save CSV ──────────────────────────────────────────────────────────────────
import csv
csv_path = FIGURES_DIR / "sensitivity_jacobian.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["output"] + PARAM_NAMES)
    for i, oname in enumerate(OUTPUT_NAMES):
        writer.writerow([oname] + list(J_np[i]))
print(f"\nRaw Jacobian saved: {csv_path}")

# ── Figure: normalised heatmap ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Left: absolute Jacobian (log scale magnitude)
log_J = np.log10(np.abs(J_np) + 1e-30)
im0 = axes[0].imshow(log_J, aspect="auto", cmap="RdYlBu_r",
                      vcenter=0.0, vmax=-log_J.min())
axes[0].set_xticks(range(5))
axes[0].set_xticklabels(PARAM_NAMES, fontsize=9)
axes[0].set_yticks(range(3))
axes[0].set_yticklabels(OUTPUT_NAMES, fontsize=9)
for i in range(3):
    for j in range(5):
        axes[0].text(j, i, f"{J_np[i,j]:.2f}", ha="center", va="center",
                     fontsize=7, color="k")
axes[0].set_title("log₁₀|∂output/∂scale|", fontsize=11)
plt.colorbar(im0, ax=axes[0], label="log₁₀|J|", ticks=[])

# Right: row-normalised (relative sensitivity within each output)
divnorm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
im1 = axes[1].imshow(J_norm, aspect="auto", cmap="RdYlBu_r", norm=divnorm)
axes[1].set_xticks(range(5))
axes[1].set_xticklabels(PARAM_NAMES, fontsize=9)
axes[1].set_yticks(range(3))
axes[1].set_yticklabels(OUTPUT_NAMES, fontsize=9)
for i in range(3):
    for j in range(5):
        axes[1].text(j, i, f"{J_norm[i,j]:+.2f}", ha="center", va="center",
                     fontsize=9, color="k")
axes[1].set_title("Normalised sensitivity", fontsize=11)
plt.colorbar(im1, ax=axes[1], label="Normalised ∂output/∂scale", ticks=[-1, -0.5, 0, 0.5, 1])

#fig.suptitle(
#    "CLM-ml-jax: Jacobian sensitivity analysis — CHATS7, May 1 2007, t=1\n"
#    "(computed via jax.jacrev in a single reverse-mode pass)",
#    fontsize=11,
#)
fig.tight_layout()

out = FIGURES_DIR / "sensitivity_jacobian.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
out_pdf = FIGURES_DIR / "sensitivity_jacobian.pdf"
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved: {out}")
