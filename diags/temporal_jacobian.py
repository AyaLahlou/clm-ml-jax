"""
temporal_jacobian.py — Time-varying Jacobian analysis for CHATS7, May 2007.

Computes the output-parameter Jacobian J_t = d[GPP, H, LE]/d[theta] at
each selected timestep over the 31-day May 2007 CHATS7 simulation, using
jax.jacrev (reverse-mode autodiff).  The time-varying sensitivity reveals
how parameter control shifts between phenological and meteorological regimes.

Parameters (theta, 5 dimensions):
  0: alpha_vcmax25 — scale on Vcmax25 (max carboxylation rate, physiological)
  1: alpha_tair    — scale on air temperature (meteorological forcing)
  2: alpha_sw      — scale on shortwave radiation (meteorological forcing)
  3: alpha_q       — scale on specific humidity (moisture forcing)
  4: alpha_dpai    — scale on canopy leaf area (structural parameter)

Outputs (y, 3 dimensions):
  0: GPP (gross primary productivity, umol CO2 m-2 s-1 proxy)
  1: H   (sensible heat flux, W m-2 proxy)
  2: LE  (latent heat flux, W m-2 proxy)

Sampling strategy:
  Every STEP_STRIDE timestep (default 6 = every 3 hours) over 1488 total steps.
  Full 31-day run produces ~248 Jacobian evaluations.

Usage (from project root):
    python diags/temporal_jacobian.py
    python diags/temporal_jacobian.py --fast       # every 12th step (~124 evals)
    python diags/temporal_jacobian.py --plot-only  # replot from saved CSV

Output:
  diags/figures/temporal_jacobian.csv   — raw Jacobian time series
  diags/figures/temporal_jacobian.png   — main time-series figure
  diags/figures/temporal_jacobian.pdf   — vector version (for paper)
  Paper/jaxes_paper/figures/temporal_jacobian.png  (also written)
  Paper/jaxes_paper/figures/temporal_jacobian.pdf  (also written)
"""
from __future__ import annotations

import csv
import sys
import os
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ── Configuration ─────────────────────────────────────────────────────────────
_FAST_MODE   = "--fast"      in sys.argv
_PLOT_ONLY   = "--plot-only" in sys.argv
STEP_STRIDE  = 12 if _FAST_MODE else 6    # timestep stride (30-min steps)
N_TOTAL      = 1488                         # total timesteps in May 2007
PARAM_NAMES  = [r"$V_{c,\max25}$", r"$T_\mathrm{air}$", r"$SW_\mathrm{rad}$",
                r"$q$", r"$\mathrm{dpai}$"]
OUTPUT_NAMES = ["GPP", "H", "LE"]
PARAM_KEYS   = ["Vcmax25", "T_air", "SW_rad", "q", "dpai"]

FIGURES_DIR  = Path(__file__).parent / "figures"
PAPER_FIGS   = _PROJECT_ROOT / "Paper" / "jaxes_paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

CSV_PATH     = FIGURES_DIR / "temporal_jacobian.csv"
PNG_PATH     = FIGURES_DIR / "temporal_jacobian.png"
PDF_PATH     = FIGURES_DIR / "temporal_jacobian.pdf"

# ── May 2007 datetime index (30-min steps, 1-indexed) ────────────────────────
_T0 = datetime(2007, 5, 1, 0, 0)
def _step_to_dt(step_1based: int) -> datetime:
    """Convert 1-based step index to datetime (30-min steps)."""
    return _T0 + timedelta(minutes=30 * (step_1based - 1))


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_temporal_jacobian(steps, J_series, forcing_series):
    """
    Plot time-varying Jacobian J_t = d[GPP, H, LE]/d[theta] as line plots.

    Args:
        steps: list of 1-based step indices
        J_series: ndarray (n_steps, 3, 5) — raw Jacobians
        forcing_series: dict of forcing arrays (SW, T_air, etc.)
    """
    dts = [_step_to_dt(s) for s in steps]
    n_steps = len(steps)

    # Normalize each row (output) by its max absolute value so all outputs
    # are on the same scale for the relative-sensitivity plot.
    J_norm = np.zeros_like(J_series)  # (n_steps, 3, 5)
    for i in range(3):
        max_abs = np.max(np.abs(J_series[:, i, :])) + 1e-30
        J_norm[:, i, :] = J_series[:, i, :] / max_abs

    # Colours for the 5 parameters
    param_colors = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]

    # ── Figure: 3 rows (one per output) × 2 cols (raw + normalised) ─────────
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    fig.suptitle(
        "CLM-ml-jax: Time-varying parameter sensitivity — CHATS7 walnut orchard, May 2007\n"
        r"$\partial$(GPP, H, LE) / $\partial\theta$  computed via jax.jacrev at each timestep",
        fontsize=12, fontweight="bold",
    )

    for i, (oname, label) in enumerate(zip(OUTPUT_NAMES, [
            r"$\partial$GPP / $\partial\alpha_j$",
            r"$\partial$H / $\partial\alpha_j$",
            r"$\partial$LE / $\partial\alpha_j$",
    ])):
        ax_raw  = axes[i, 0]
        ax_norm = axes[i, 1]

        for j, (pname, col) in enumerate(zip(PARAM_NAMES, param_colors)):
            ax_raw.plot(dts, J_series[:, i, j], color=col, lw=0.8,
                        label=pname, alpha=0.85)
            ax_norm.plot(dts, J_norm[:, i, j], color=col, lw=0.8,
                         label=pname if i == 0 else None, alpha=0.85)

        ax_raw.set_ylabel(label, fontsize=10)
        ax_raw.axhline(0, color="k", lw=0.5, ls="--")
        ax_norm.axhline(0, color="k", lw=0.5, ls="--")
        if i == 0:
            ax_raw.set_title("(a) Raw Jacobian", fontsize=10)
            ax_norm.set_title("(b) Row-normalised sensitivity", fontsize=10)
            ax_norm.legend(loc="upper right", fontsize=8, ncol=2)
        ax_norm.set_ylim(-1.2, 1.2)

    # Format x-axis
    for ax in axes[-1]:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    # Copy to paper figures
    import shutil
    shutil.copy(PNG_PATH, PAPER_FIGS / "temporal_jacobian.png")
    shutil.copy(PDF_PATH, PAPER_FIGS / "temporal_jacobian.pdf")
    plt.close(fig)
    print(f"Figure saved: {PNG_PATH}")
    print(f"Figure saved: {PDF_PATH}")

    # ── Secondary figure: diurnal-composite heatmap ──────────────────────────
    # Group steps by hour-of-day, average |J| for each hour × parameter
    hour_of_day = np.array([(s - 1) % 48 * 30 // 60 for s in steps])  # 0-23
    J_diurnal = np.zeros((3, 24, 5))   # (output, hour, param)
    counts     = np.zeros((3, 24, 5))
    for idx, h in enumerate(hour_of_day):
        J_diurnal[:, h, :] += np.abs(J_series[idx])
        counts[:, h, :] += 1
    counts[counts == 0] = 1
    J_diurnal /= counts

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle(
        "Diurnal composite of |∂output/∂θ| — CHATS7 walnut orchard, May 2007",
        fontsize=11, fontweight="bold",
    )
    for i, (oname, ax) in enumerate(zip(OUTPUT_NAMES, axes2)):
        im = ax.imshow(J_diurnal[i].T, aspect="auto", origin="lower",
                       cmap="YlOrRd")
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)], fontsize=8, rotation=30)
        ax.set_yticks(range(5))
        ax.set_yticklabels(PARAM_NAMES, fontsize=9)
        ax.set_xlabel("Hour of day (local)")
        ax.set_title(f"|∂{oname}/∂θ| diurnal composite", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig2.tight_layout()
    diurnal_png = FIGURES_DIR / "temporal_jacobian_diurnal.png"
    diurnal_pdf = FIGURES_DIR / "temporal_jacobian_diurnal.pdf"
    fig2.savefig(diurnal_png, dpi=150, bbox_inches="tight")
    fig2.savefig(diurnal_pdf, bbox_inches="tight")
    shutil.copy(diurnal_png, PAPER_FIGS / "temporal_jacobian_diurnal.png")
    shutil.copy(diurnal_pdf, PAPER_FIGS / "temporal_jacobian_diurnal.pdf")
    plt.close(fig2)
    print(f"Figure saved: {diurnal_png}")

    return fig


# ── --plot-only mode ──────────────────────────────────────────────────────────
if _PLOT_ONLY:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run without --plot-only first.")
        sys.exit(1)
    print(f"Loading Jacobian from {CSV_PATH} ...")
    rows = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    steps    = [int(r["step"]) for r in rows]
    J_series = np.array([[
        [float(r[f"J_{o}_{p}"]) for p in PARAM_KEYS]
        for o in OUTPUT_NAMES
    ] for r in rows])
    # Build dummy forcing series
    forcing_series = {}
    plot_temporal_jacobian(steps, J_series, forcing_series)
    sys.exit(0)


# ── Full compute mode ─────────────────────────────────────────────────────────
print("=== temporal_jacobian.py: loading model ===", flush=True)

# ─ shared init (loads model at step 1) ─
from diags.expt_init import (
    mlcanopy_inst, grid, _mlcf_kwargs,
    jax, jnp, MLCanopyFluxes,
    atm2lnd_inst, wateratm2lndbulk_inst,
    compute_gpp, compute_le, compute_h,
)
import multilayer_canopy.MLpftconMod              as _pftcon_mod
import multilayer_canopy.MLLeafPhotosynthesisMod  as _leaf_mod
import multilayer_canopy.MLCanopyNitrogenProfileMod as _nitro_mod
from multilayer_canopy import MLclm_varctl
from clm_src_utils import clm_time_manager
from clm_src_cpl.lnd_comp_nuopc import InitializeRealize, ModelAdvance
from offline_driver.TowerMetMod import TowerMetCurr, TowerMetNext
from clm_src_main import clm_instMod
from offline_executable.main import read_namelist, _resolve_path, build_bounds
from offline_driver import controlMod, TowerDataMod

_p    = grid.p
_ncan = grid.ncan
_orig_pftcon = _pftcon_mod.MLpftcon

# Build kwargs without the instances we'll be overriding at each timestep
_canopystate_inst = _mlcf_kwargs["canopystate_inst"]
_mlcf_kwargs_base = {k: v for k, v in _mlcf_kwargs.items()
                     if k not in ("atm2lnd_inst", "wateratm2lndbulk_inst",
                                  "canopystate_inst")}

# ── Load namelist for the full 31-day run ────────────────────────────────────
_NML_PATH = _PROJECT_ROOT / "src" / "offline_executable" / "nl.CHATS7.05.2007"
nml    = read_namelist(str(_NML_PATH))
params = nml.get("clmML_inparm", nml.get("clm_inparm", {}))

fin_tower = _resolve_path(str(params.get("fin_tower", "")))
ntim      = 1488   # full May 2007

print(f"Tower forcing file: {fin_tower}")
print(f"Total timesteps: {ntim}, stride: {STEP_STRIDE}, "
      f"evaluations: {len(range(1, ntim+1, STEP_STRIDE))}", flush=True)


# ── Forward function with explicit forcing arguments ──────────────────────────
# Design: atm_inst and watm_inst are EXPLICIT JAX pytree arguments, not closure
# constants.  JAX traces once for abstract shapes → one XLA binary for all 248
# steps.  Previous design baked forcing as constants → recompilation per step.

def forward_with_forcing(scales: jnp.ndarray,
                         atm_inst,
                         watm_inst) -> jnp.ndarray:
    """
    Args:
        scales:    float64 (5,) — scale factors, baseline = 1.0
          0: alpha_vcmax25   1: alpha_tair   2: alpha_sw
          3: alpha_q         4: alpha_dpai
        atm_inst:  atm2lnd_type NamedTuple for this timestep (JAX pytree)
        watm_inst: wateratm2lndbulk_type NamedTuple for this timestep (JAX pytree)

    Returns: float64 (3,) — [GPP, H, LE]
    """
    vcmaxpft_jax = scales[0] * _orig_pftcon.vcmaxpft

    modified_canopy = _canopystate_inst._replace(
        elai_patch = scales[4] * jnp.asarray(_canopystate_inst.elai_patch, jnp.float64),
        esai_patch = scales[4] * jnp.asarray(_canopystate_inst.esai_patch, jnp.float64),
    )
    modified_atm = atm_inst._replace(
        forc_t_downscaled_col     = scales[1] * atm_inst.forc_t_downscaled_col,
        forc_solad_downscaled_col = scales[2] * atm_inst.forc_solad_downscaled_col,
        forc_solai_grc            = scales[2] * atm_inst.forc_solai_grc,
    )
    modified_watm = watm_inst._replace(
        forc_q_downscaled_col = scales[3] * watm_inst.forc_q_downscaled_col,
    )

    inst = MLCanopyFluxes(
        mlcanopy_inst         = mlcanopy_inst,
        atm2lnd_inst          = modified_atm,
        wateratm2lndbulk_inst = modified_watm,
        canopystate_inst      = modified_canopy,
        vcmaxpft_jax          = vcmaxpft_jax,
        **_mlcf_kwargs_base,
    )

    GPP = compute_gpp(inst, _p, _ncan)
    H   = compute_h(inst,   _p, _ncan)
    LE  = compute_le(inst,  _p, _ncan)
    return jnp.array([GPP, H, LE])


# ── Compile jacrev ONCE (all steps share this XLA binary) ────────────────────
scales0      = jnp.ones(5, dtype=jnp.float64)
bounds_local = build_bounds(nml)

# JIT-compile jacrev w.r.t. scales only (argnums=0).
# atm_inst / watm_inst are JAX pytrees → abstract shapes, not constants.
_jac_fn = jax.jit(jax.jacrev(forward_with_forcing, argnums=0))
_fwd_fn = jax.jit(forward_with_forcing)

print("\n=== Pre-JIT: loading step 1 and compiling jacrev once ===", flush=True)

(atm_step1, watm_step1, _) = TowerMetCurr(
    fin_tower, 1,
    TowerDataMod.tower_num,
    bounds_local.begp, bounds_local.endp,
    clm_instMod.atm2lnd_inst,
    clm_instMod.wateratm2lndbulk_inst,
    clm_instMod.frictionvel_inst,
)

t0 = time.time()
J0 = _jac_fn(scales0, atm_step1, watm_step1)
jax.block_until_ready(J0)
t_compile = time.time() - t0
print(f"First JIT+jacrev (XLA compile): {t_compile:.1f}s", flush=True)
print(f"All subsequent steps reuse this binary (no recompilation).", flush=True)

# ── Time-stepping loop ───────────────────────────────────────────────────────
steps_computed = list(range(1, ntim + 1, STEP_STRIDE))
J_series   = np.full((len(steps_computed), 3, 5), np.nan)
fwd_series = {"GPP": [], "H": [], "LE": [], "SW": [], "Tair": []}

print(f"\n=== Computing Jacobians at {len(steps_computed)} timesteps ===", flush=True)
t_loop_start = time.time()

for loop_idx, step in enumerate(steps_computed):
    (atm_s, watm_s, _frv) = TowerMetCurr(
        fin_tower, step,
        TowerDataMod.tower_num,
        bounds_local.begp, bounds_local.endp,
        clm_instMod.atm2lnd_inst,
        clm_instMod.wateratm2lndbulk_inst,
        clm_instMod.frictionvel_inst,
    )

    t_step = time.time()
    try:
        J = _jac_fn(scales0, atm_s, watm_s)
        jax.block_until_ready(J)
        J_series[loop_idx] = np.array(J)

        y0 = _fwd_fn(scales0, atm_s, watm_s)
        jax.block_until_ready(y0)
        fwd_series["GPP"].append(float(y0[0]))
        fwd_series["H"].append(float(y0[1]))
        fwd_series["LE"].append(float(y0[2]))

        sw_val   = float(jnp.sum(atm_s.forc_solad_downscaled_col[_p]))
        tair_val = float(atm_s.forc_t_downscaled_col[_p]) - 273.15
        fwd_series["SW"].append(sw_val)
        fwd_series["Tair"].append(tair_val)

    except Exception as exc:
        print(f"  step {step}: ERROR — {exc}", flush=True)
        fwd_series["GPP"].append(np.nan)
        fwd_series["H"].append(np.nan)
        fwd_series["LE"].append(np.nan)
        fwd_series["SW"].append(np.nan)
        fwd_series["Tair"].append(np.nan)

    dt_step = time.time() - t_step
    if loop_idx % 20 == 0 or loop_idx < 3:
        elapsed = time.time() - t_loop_start
        remain  = (elapsed / max(loop_idx + 1, 1)) * (len(steps_computed) - loop_idx - 1)
        print(f"  step {step:4d} ({loop_idx+1}/{len(steps_computed)})  "
              f"GPP={fwd_series['GPP'][-1]:.3f}  "
              f"SW={fwd_series['SW'][-1]:.1f}  "
              f"T={fwd_series['Tair'][-1]:.1f}°C  "
              f"dt={dt_step:.2f}s  ETA={remain/60:.1f}min",
              flush=True)

t_total = time.time() - t_loop_start
print(f"\nTotal loop time: {t_total/60:.1f} min  "
      f"({t_total/len(steps_computed):.2f}s/step)", flush=True)

# ── Print summary statistics ──────────────────────────────────────────────────
print("\n=== Jacobian statistics (time-averaged) ===")
J_mean = np.nanmean(J_series, axis=0)   # (3, 5)
J_std  = np.nanstd(J_series,  axis=0)   # (3, 5)
J_max  = np.nanmax(np.abs(J_series), axis=0)

print(f"\n  Mean |J| across all timesteps:")
header = "  {:5s}  " + "  ".join(f"{p:>15s}" for p in PARAM_KEYS)
print(header.format(""))
for i, oname in enumerate(OUTPUT_NAMES):
    row = "  ".join(f"{J_mean[i,j]:>15.4e}" for j in range(5))
    print(f"  {oname:5s}  {row}")

print(f"\n  Std(J) across all timesteps:")
for i, oname in enumerate(OUTPUT_NAMES):
    row = "  ".join(f"{J_std[i,j]:>15.4e}" for j in range(5))
    print(f"  {oname:5s}  {row}")

print(f"\n  Max |J| across all timesteps:")
for i, oname in enumerate(OUTPUT_NAMES):
    row = "  ".join(f"{J_max[i,j]:>15.4e}" for j in range(5))
    print(f"  {oname:5s}  {row}")

# Coefficient of variation: std/|mean|  (relative temporal variability)
print(f"\n  CV = std/|mean| (temporal variability of sensitivity):")
for i, oname in enumerate(OUTPUT_NAMES):
    row = "  ".join(f"{J_std[i,j]/(abs(J_mean[i,j])+1e-30):>15.3f}" for j in range(5))
    print(f"  {oname:5s}  {row}")

# Diurnal split: daytime (SW>10) vs nighttime
daytime_mask = np.array(fwd_series["SW"]) > 10.0
print(f"\n  Daytime steps: {daytime_mask.sum()}, Nighttime steps: {(~daytime_mask).sum()}")
if daytime_mask.sum() > 0 and (~daytime_mask).sum() > 0:
    J_day   = np.nanmean(J_series[daytime_mask],  axis=0)
    J_night = np.nanmean(J_series[~daytime_mask], axis=0)
    print(f"\n  Mean |J| — DAYTIME only:")
    for i, oname in enumerate(OUTPUT_NAMES):
        row = "  ".join(f"{J_day[i,j]:>15.4e}" for j in range(5))
        print(f"  {oname:5s}  {row}")
    print(f"\n  Mean |J| — NIGHTTIME only:")
    for i, oname in enumerate(OUTPUT_NAMES):
        row = "  ".join(f"{J_night[i,j]:>15.4e}" for j in range(5))
        print(f"  {oname:5s}  {row}")

# Early May vs late May split
n_half = len(steps_computed) // 2
J_early = np.nanmean(J_series[:n_half], axis=0)
J_late  = np.nanmean(J_series[n_half:], axis=0)
print(f"\n  Mean |J| — EARLY MAY (steps 1-{steps_computed[n_half-1]}):")
for i, oname in enumerate(OUTPUT_NAMES):
    row = "  ".join(f"{J_early[i,j]:>15.4e}" for j in range(5))
    print(f"  {oname:5s}  {row}")
print(f"\n  Mean |J| — LATE MAY (steps {steps_computed[n_half]}-{steps_computed[-1]}):")
for i, oname in enumerate(OUTPUT_NAMES):
    row = "  ".join(f"{J_late[i,j]:>15.4e}" for j in range(5))
    print(f"  {oname:5s}  {row}")


# ── Save CSV ──────────────────────────────────────────────────────────────────
with open(CSV_PATH, "w", newline="") as f:
    fieldnames = ["step", "datetime", "GPP", "H", "LE", "SW_rad", "T_air_C"] + \
                 [f"J_{o}_{p}" for o in OUTPUT_NAMES for p in PARAM_KEYS]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for idx, step in enumerate(steps_computed):
        row = {
            "step":     step,
            "datetime": _step_to_dt(step).strftime("%Y-%m-%d %H:%M"),
            "GPP":      fwd_series["GPP"][idx],
            "H":        fwd_series["H"][idx],
            "LE":       fwd_series["LE"][idx],
            "SW_rad":   fwd_series["SW"][idx],
            "T_air_C":  fwd_series["Tair"][idx],
        }
        for i, o in enumerate(OUTPUT_NAMES):
            for j, p in enumerate(PARAM_KEYS):
                row[f"J_{o}_{p}"] = float(J_series[idx, i, j])
        writer.writerow(row)
print(f"\nCSV saved: {CSV_PATH}")


# ── Plot ──────────────────────────────────────────────────────────────────────
print("\n=== Generating figures ===", flush=True)
plot_temporal_jacobian(steps_computed, J_series, fwd_series)

print("\n=== temporal_jacobian.py complete ===", flush=True)
