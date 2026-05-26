"""
Generate publication figure for multi-parameter calibration experiment.

Loads diags/output/multipar_calibration_results.json and produces a 3-panel
figure at 10.5"×4" (NeurIPS double-column width):

  Panel A (left): Loss vs forward-equivalent evaluations — convergence.
    Inset shows cosine-annealing LR schedule for Adam.
    - Adam + jax.grad (green solid)
    - L-BFGS-B + jax.grad (purple dash-dot)
    - L-BFGS-B + FD (orange dashed)
    - Nelder-Mead (blue dotted)

  Panel B (centre): Per-parameter recovery  |θ_final - θ★| for each method.
    Grouped bar chart: 10 parameter groups × 4 methods.
    Shows whether equifinality is broken by the multi-step loss.

  Panel C (right): Gradient cost scaling vs number of parameters p.
    AD cost flat at T_ratio; FD cost slopes as 2p.

Outputs:
  diags/figures/multipar_calibration.pdf
  diags/figures/multipar_calibration.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path(__file__).parent / "output" / "multipar_calibration_results.json"
if not DATA_PATH.exists():
    print(f"ERROR: {DATA_PATH} not found. Run diags/multipar_calibration.py first.")
    sys.exit(1)

with open(DATA_PATH) as f:
    res = json.load(f)

P           = res["p"]
T_steps     = res.get("T_steps", 1)
T_ratio     = res["T_ratio"]
crossover   = res["crossover_p"]
T_fwd       = res["T_forward_s"]
T_bwd       = res["T_backward_s"]
param_names = res["param_names"]
theta_star  = np.array(res["theta_star"])

adam      = res["adam"]
lbfgsb_ad = res.get("lbfgsb_ad", None)
lbfgsb_fd = res["lbfgsb_fd"]
nm        = res["nelder_mead"]
timing    = res["timing_sweep"]

# ── Colors and labels ─────────────────────────────────────────────────────────
C_ADAM   = "#2ca02c"
C_LBFGSB = "#9467bd"
C_FD     = "#ff7f0e"
C_NM     = "#1f77b4"

METHODS = [
    ("Adam + AD",        adam,      C_ADAM,   "-"),
    ("L-BFGS-B + AD",   lbfgsb_ad, C_LBFGSB, "-."),
    ("L-BFGS-B + FD",   lbfgsb_fd, C_FD,     "--"),
    ("Nelder-Mead",      nm,        C_NM,     ":"),
]

# ── Forward-equivalent axes ───────────────────────────────────────────────────
adam_x     = np.array(adam["fwd_equiv"], dtype=float)
adam_y     = np.array(adam["loss_history"], dtype=float)

if lbfgsb_ad is not None:
    lbfgsb_ad_x = np.array(lbfgsb_ad["fwd_equiv"], dtype=float)
    lbfgsb_ad_y = np.array(lbfgsb_ad["loss_history"], dtype=float)

lbfgsb_fd_x = np.array(lbfgsb_fd["nfev_history"], dtype=float)
lbfgsb_fd_y = np.array(lbfgsb_fd["loss_history"], dtype=float)

nm_x = np.array(nm["nfev_history"], dtype=float)
nm_y = np.array(nm["loss_history"], dtype=float)

# ── Cosine LR schedule ────────────────────────────────────────────────────────
lr_max    = adam.get("lr_max", 0.01)
lr_min    = adam.get("lr_min", 1e-4)
n_adam    = adam.get("n_steps", len(adam_x))
lr_sched  = lr_min + 0.5 * (lr_max - lr_min) * (
    1.0 + np.cos(np.pi * np.arange(1, n_adam + 1) / n_adam))

# ── Timing sweep ──────────────────────────────────────────────────────────────
p_vals    = np.array(timing["p_values"], dtype=float)
T_ad_norm = np.array(timing["T_ad_s"], dtype=float) / T_fwd
T_fd_norm = np.array(timing["T_fd_s"], dtype=float) / T_fwd
p_fine    = np.logspace(np.log10(0.5), np.log10(60), 200)

# ── Per-parameter error ───────────────────────────────────────────────────────
def _param_errors(method_dict):
    tf = method_dict.get("theta_final")
    if tf is None:
        return None
    return np.abs(np.array(tf) - theta_star)

adam_perr    = _param_errors(adam)
lbfgsb_perr  = _param_errors(lbfgsb_ad) if lbfgsb_ad else None
lbfgsb_fd_pe = _param_errors(lbfgsb_fd)
nm_perr      = _param_errors(nm)

# ── Figure layout: 3 panels ───────────────────────────────────────────────────
fig = plt.figure(figsize=(10.5, 4.0))
gs  = GridSpec(1, 3, figure=fig, wspace=0.38, left=0.07, right=0.97)
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])
axC = fig.add_subplot(gs[2])

# ══════════════════════════════════════════════════════════════════════════════
# Panel A — Convergence
# ══════════════════════════════════════════════════════════════════════════════
def _label(name, d):
    pe = d.get("final_param_err", float("nan")) if d else float("nan")
    return f"{name}  ($\|\|\\Delta\\theta\|\|={pe:.3f}$)"

if len(adam_x) > 0:
    axA.semilogy(adam_x, np.maximum(adam_y, 1e-20),
                 color=C_ADAM, lw=1.8,
                 label=_label("Adam + AD", adam), zorder=4)

if lbfgsb_ad is not None and len(lbfgsb_ad_x) > 0:
    axA.semilogy(lbfgsb_ad_x, np.maximum(lbfgsb_ad_y, 1e-20),
                 color=C_LBFGSB, lw=1.8, linestyle="-.",
                 label=_label("L-BFGS-B + AD", lbfgsb_ad), zorder=5)

if len(lbfgsb_fd_x) > 0:
    axA.semilogy(lbfgsb_fd_x, np.maximum(lbfgsb_fd_y, 1e-20),
                 color=C_FD, lw=1.8, linestyle="--",
                 label=_label("L-BFGS-B + FD", lbfgsb_fd), zorder=2)

if len(nm_x) > 0:
    axA.semilogy(nm_x, np.maximum(nm_y, 1e-20),
                 color=C_NM, lw=1.8, linestyle=":",
                 label=_label("Nelder-Mead", nm), zorder=1)

two_p = 2 * P
axA.axvline(two_p, color="gray", linestyle="-.", lw=1.0, alpha=0.7)
axA.text(two_p * 1.04, axA.get_ylim()[1] if hasattr(axA, '_ylim') else 1.0,
         f"$2p={two_p}$", fontsize=6, color="gray", va="top")

axA.set_xlabel("Forward-pass equivalents", fontsize=8)
axA.set_ylabel(f"Loss (multi-step, $T={T_steps}$)", fontsize=8)
axA.set_title(f"(a) Convergence — $p={P}$ parameters", fontsize=8.5, fontweight="bold")
axA.legend(fontsize=6, loc="upper right")
axA.grid(True, which="both", alpha=0.2)
axA.tick_params(labelsize=7)
axA.set_xlim(0, 600)

# Inset: cosine LR schedule
ax_inset = inset_axes(axA, width="38%", height="30%", loc="lower left",
                      bbox_to_anchor=(0.05, 0.06, 1, 1),
                      bbox_transform=axA.transAxes)
ax_inset.plot(np.arange(1, n_adam + 1), lr_sched, color=C_ADAM, lw=1.2)
ax_inset.set_yscale("log")
ax_inset.set_xlabel("step", fontsize=5.5)
ax_inset.set_ylabel("LR", fontsize=5.5)
ax_inset.set_title("Adam LR schedule", fontsize=5.5)
ax_inset.tick_params(labelsize=5)
ax_inset.grid(True, alpha=0.2)
ax_inset.yaxis.set_minor_formatter(mticker.NullFormatter())

# ══════════════════════════════════════════════════════════════════════════════
# Panel B — Per-parameter recovery
# ══════════════════════════════════════════════════════════════════════════════
# Short display names for the 10 parameters
SHORT_NAMES = [r"$\alpha_{V\!D}$", r"$\alpha_{ND}$", r"$\alpha_{VF}$", r"$\alpha_{NF}$",
               r"$T$", r"$V_{cmax}$", r"$\iota$", r"$q$", r"$p_{bot}$", r"$u$"]

bar_data = [
    (adam_perr,    C_ADAM,   "Adam + AD"),
    (lbfgsb_perr,  C_LBFGSB, "L-BFGS-B + AD"),
    (lbfgsb_fd_pe, C_FD,     "L-BFGS-B + FD"),
    (nm_perr,      C_NM,     "Nelder-Mead"),
]
bar_data = [(d, c, n) for d, c, n in bar_data if d is not None]
n_methods = len(bar_data)

x      = np.arange(P)
width  = 0.8 / n_methods
offset = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * width

for idx, (errs, col, lbl) in enumerate(bar_data):
    axB.bar(x + offset[idx], errs, width=width,
            color=col, alpha=0.8, label=lbl, edgecolor="white", linewidth=0.3)

axB.axhline(0, color="k", lw=0.5)
axB.set_xticks(x)
axB.set_xticklabels(SHORT_NAMES, fontsize=7, rotation=0)
axB.set_ylabel(r"$|\theta_{\rm final} - \theta^\star|$", fontsize=8)
axB.set_title(f"(b) Per-parameter recovery  ($T={T_steps}$ steps)", fontsize=8.5, fontweight="bold")
axB.legend(fontsize=6, loc="upper right")
axB.grid(True, axis="y", alpha=0.2)
axB.tick_params(labelsize=7)

# ══════════════════════════════════════════════════════════════════════════════
# Panel C — Gradient cost scaling
# ══════════════════════════════════════════════════════════════════════════════
ad_line = np.full_like(p_fine, T_ratio)
fd_line = 2.0 * p_fine

axC.fill_between(p_fine, ad_line, fd_line,
                 where=(fd_line >= ad_line),
                 color=C_ADAM, alpha=0.12, label="AD faster region")
axC.fill_between(p_fine, ad_line, fd_line,
                 where=(fd_line < ad_line),
                 color=C_FD, alpha=0.12, label="FD faster region")

axC.loglog(p_fine, ad_line, color=C_ADAM, lw=2.0, label="AD cost (const)")
axC.loglog(p_fine, fd_line, color=C_FD,   lw=2.0, linestyle="--",
           label="FD cost ($2p$)")

axC.scatter(p_vals, T_ad_norm, color=C_ADAM, s=35, zorder=5)
axC.scatter(p_vals, T_fd_norm, color=C_FD,   s=35, marker="s", zorder=5)

axC.axvline(crossover, color="gray", linestyle="-.", lw=1.0, alpha=0.7)
axC.text(crossover * 1.1, 1.4,
         f"crossover\n$p={crossover:.1f}$", fontsize=6.5, color="gray")

axC.axvline(P, color=C_ADAM, linestyle=":", lw=1.0, alpha=0.6)
axC.text(P * 1.08, T_ratio * 4,
         f"$p={P}$\nhere", fontsize=6, color=C_ADAM, va="top")

axC.set_xlabel("Number of parameters $p$", fontsize=8)
axC.set_ylabel(r"Cost (units of $T_{\rm fwd}$)", fontsize=8)
axC.set_title("(c) Gradient cost scaling", fontsize=8.5, fontweight="bold")
axC.legend(fontsize=6.5, loc="upper left")
axC.grid(True, which="both", alpha=0.2)
axC.tick_params(labelsize=7)
axC.xaxis.set_major_formatter(mticker.ScalarFormatter())
axC.set_xlim(0.9, 65)

axC.text(0.97, 0.06,
         f"$T_{{\\rm ratio}}={T_ratio:.1f}\\times$",
         transform=axC.transAxes, fontsize=7.5, color=C_ADAM,
         ha="right",
         bbox=dict(facecolor="white", edgecolor=C_ADAM, alpha=0.85, lw=0.8))

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    f"CLM-ml-jax: gradient-based vs gradient-free calibration — "
    f"$p={P}$ parameters, $T={T_steps}$-step loss  ·  CHATS7 walnut orchard, May 2007",
    fontsize=8.5,
)

for ext in ("pdf", "png"):
    out = FIGURES_DIR / f"multipar_calibration.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)

# ── Copy to paper figures ─────────────────────────────────────────────────────
import shutil
PAPER_FIGS = _PROJECT_ROOT / "Paper" / "jaxes_paper" / "figures"
if PAPER_FIGS.exists():
    for ext in ("pdf", "png"):
        src = FIGURES_DIR / f"multipar_calibration.{ext}"
        shutil.copy(src, PAPER_FIGS / f"multipar_calibration.{ext}")
        print(f"Copied to paper: {PAPER_FIGS / f'multipar_calibration.{ext}'}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"Multi-step loss: T={T_steps} steps, "
      f"step indices = {res.get('multi_step_indices', 'N/A')}")
print(f"\n{'Method':<22}  {'Final loss':>12}  {'‖Δθ‖₂':>8}  "
      f"{'Converged':>9}  {'Time(s)':>8}")
print("-" * 70)
for lbl, d, _, _ in METHODS:
    if d is None:
        continue
    conv = d.get("converged", "—")
    if isinstance(conv, bool):
        conv = "yes" if conv else "no"
    t_s  = d.get("time_s", d.get("total_time_s", float("nan")))
    if isinstance(t_s, list):
        t_s = t_s[-1]
    print(f"  {lbl:<20}  {d['final_loss']:>12.4e}  "
          f"{d['final_param_err']:>8.4f}  {str(conv):>9}  {t_s:>8.1f}")

if adam_perr is not None:
    print("\n  Per-parameter |θ_final - θ★|:")
    print(f"  {'param':<12}" + "".join(f"  {n:<16}" for n in
          [b[2] for b in bar_data]))
    for i, pn in enumerate(param_names):
        row = f"  {pn:<12}"
        for errs, _, _ in bar_data:
            row += f"  {errs[i]:>16.4f}"
        print(row)

print("\n=== plot_multipar_calibration.py complete ===")
