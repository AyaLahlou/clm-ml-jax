"""
Publication figure: 10-parameter gradient accuracy check (job 7589868).

Two-panel layout (7.5" × 4.5"):
  (a) Grouped horizontal bars — JAX vs FD gradient magnitude for each parameter,
      colored by PASS/NaN/INACT status.
  (b) Relative error for PASS parameters on a log10 scale, with 1% / 0.1% / 1e-6
      reference lines.

Key finding: 6 of 10 parameters pass (<1% rel err); alpha_pbot gives NaN gradient
(blocked path); lwrad/v/pco2 are inactive (zero gradient expected and observed).

Data: diags/figures/check_10param_grads.csv
Out:  diags/figures/10param_grads.{pdf,png}
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

# ── Paths ─────────────────────────────────────────────────────────────────────
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA = FIGURES_DIR / "check_10param_grads.csv"

if not DATA.exists():
    sys.exit(f"ERROR: {DATA} not found. Run diags/check_10param_grads.py first.")

# ── Load CSV ──────────────────────────────────────────────────────────────────
def _f(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")

with open(DATA, newline="") as fh:
    rows = list(csv.DictReader(fh))

rows     = [r for r in rows if r["status"] != "INACT"]
params   = [r["param"] for r in rows]
statuses = [r["status"] for r in rows]
grad_jax = np.array([_f(r["grad_jax"]) for r in rows])
grad_fd  = np.array([_f(r["grad_fd"])  for r in rows])
rel_err  = np.array([_f(r["rel_err"])  for r in rows])

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "figure.dpi": 150,
})

STATUS_COLOR = {
    "PASS":  "#2ca02c",
    "NaN":   "#d62728",
    "INACT": "#9467bd",
}

PARAM_LABELS = {
    "alpha_sw":    r"$\alpha_{\rm sw}$  shortwave rad.",
    "alpha_tref":  r"$\alpha_{T}$  air temperature",
    "alpha_vcmax": r"$\alpha_{V_{\rm cmax}}$  Vcmax25",
    "alpha_iota":  r"$\alpha_{\iota}$  WUE iota",
    "alpha_q":     r"$\alpha_{q}$  specific humidity",
    "alpha_pbot":  r"$\alpha_{P}$  atm. pressure",
    "alpha_lwrad": r"$\alpha_{\rm lw}$  longwave rad.",
    "alpha_u":     r"$\alpha_{u}$  wind $u$",
    "alpha_v":     r"$\alpha_{v}$  wind $v$",
    "alpha_pco2":  r"$\alpha_{p_{\rm CO_2}}$  CO$_2$ pres.",
}

ytick_labels = [PARAM_LABELS.get(p, p) for p in params]
colors       = [STATUS_COLOR[s] for s in statuses]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.5),
                         gridspec_kw={"width_ratios": [1.4, 1]})
fig.subplots_adjust(wspace=0.55)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (a): grouped horizontal bars
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[0]

n      = len(rows)
y      = np.arange(n, dtype=float)
bar_h  = 0.32
gap    = 0.05

jax_abs  = np.abs(grad_jax)
fd_abs   = np.abs(grad_fd)
jax_plot = np.where(np.isnan(jax_abs), 0.0, jax_abs)

valid_fd = fd_abs[~np.isnan(fd_abs)]
max_val  = float(valid_fd.max()) if len(valid_fd) else 1.0

# FD reference bars (light blue)
ax.barh(y + (bar_h / 2 + gap / 2), fd_abs,
        height=bar_h, color="#aec7e8", edgecolor="white",
        linewidth=0.4, label="FD (reference)", zorder=2)

# JAX bars
for i in range(n):
    st  = statuses[i]
    col = STATUS_COLOR[st]
    htch = "///" if st in ("NaN", "INACT") else None
    alph = 1.0 if st == "PASS" else 0.65
    width = max(jax_plot[i], max_val * 0.003)
    ax.barh(y[i] - (bar_h / 2 + gap / 2), width,
            height=bar_h, color=col, hatch=htch, alpha=alph,
            edgecolor="white", linewidth=0.4, zorder=3)
    if st == "NaN":
        ax.text(max_val * 0.01, y[i] - (bar_h / 2 + gap / 2),
                "NaN", va="center", ha="left", fontsize=7,
                color="#d62728", fontweight="bold")
    elif st == "INACT":
        ax.text(max_val * 0.008, y[i] - (bar_h / 2 + gap / 2),
                "inactive (zero)", va="center", ha="left",
                fontsize=6.5, color="#9467bd", style="italic")

ax.set_yticks(y)
ax.set_yticklabels(ytick_labels, fontsize=7.8)
ax.invert_yaxis()
ax.set_xlabel(r"Gradient magnitude $|\partial\,\overline{\rm GPP}\,/\,\partial\alpha_i|$",
              fontsize=8.5)
ax.set_title("(a) AD vs finite-difference gradients\n"
             r"$p=10$ atmospheric / physiological parameters",
             fontsize=8.5)
ax.set_xlim(0, max_val * 1.22)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{x:.0f}" if x >= 1 else f"{x:.2f}"))
ax.tick_params(labelsize=8)

patch_fd  = mpatches.Patch(color="#aec7e8", label="FD (reference)")
patch_p   = mpatches.Patch(color="#2ca02c", label="AD/JAX — PASS")
patch_nan = mpatches.Patch(color="#d62728", hatch="///", alpha=0.65,
                           label="AD/JAX — NaN")
ax.legend(handles=[patch_fd, patch_p, patch_nan],
          fontsize=7, loc="lower right", handlelength=1.2,
          handletextpad=0.3, borderpad=0.5)

n_pass = statuses.count("PASS")
n_nan  = statuses.count("NaN")
ax.text(0.97, 0.02,
        f"{n_pass} PASS  ·  {n_nan} NaN",
        transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
        color="gray")

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (b): relative error for PASS params
# ═══════════════════════════════════════════════════════════════════════════════
ax2 = axes[1]

pass_idx  = [i for i, s in enumerate(statuses) if s == "PASS"]
pass_err  = np.array([rel_err[i] for i in pass_idx])
pass_labs = [PARAM_LABELS.get(params[i], params[i]) for i in pass_idx]

# Sort by rel_err descending (worst at top)
order      = np.argsort(pass_err)[::-1]
pass_err_s = pass_err[order]
pass_labs_s = [pass_labs[j] for j in order]

yp = np.arange(len(pass_idx))

ax2.barh(yp, pass_err_s, height=0.55,
         color="#2ca02c", edgecolor="white", linewidth=0.4, zorder=3)

for thresh, ls, lab in [
    (1e-2, "--", "1%"),
    (1e-3, ":",  "0.1%"),
    (1e-6, "-.", r"10$^{-6}$"),
]:
    ax2.axvline(thresh, color="gray", linestyle=ls, lw=0.85, alpha=0.7, zorder=1)
    ax2.text(thresh * 1.15, len(pass_idx) - 0.08, lab,
             fontsize=6.5, color="gray", va="top", ha="left")

ax2.set_xscale("log")
ax2.set_yticks(yp)
ax2.set_yticklabels(pass_labs_s, fontsize=7.8)
ax2.invert_yaxis()
ax2.set_xlabel(r"Relative error  $|\rm JAX - FD| \,/\, |FD|$", fontsize=8.5)
ax2.set_title("(b) Gradient accuracy\n(PASS parameters only)", fontsize=8.5)
ax2.grid(True, axis="x", which="both", alpha=0.18, zorder=0)
ax2.tick_params(labelsize=8)

# Annotate exact values
xlim_right = pass_err_s.max() * 80
for ypos, val in zip(yp, pass_err_s):
    ax2.text(val * 4.5, ypos,
             f"{val:.1e}", va="center", ha="left", fontsize=7,
             color="#2ca02c")

ax2.set_xlim(pass_err_s.min() * 0.05, xlim_right)

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "CLM-ml-jax: gradient verification — 7 active parameters  "
    r"($\partial\,\overline{\rm GPP}_{\rm day}\,/\,\partial\,\alpha_i$)"
    "\nMedlyn stomata · CHATS7 · 1 May 2007 · GPU · reverse-mode AD vs central FD (eps=1e-4)",
    fontsize=8, y=1.01,
)

for ext in ("pdf", "png"):
    out = FIGURES_DIR / f"10param_grads.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")

plt.close(fig)
print("=== plot_10param_grads.py complete ===")
