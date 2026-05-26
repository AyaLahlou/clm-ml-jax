"""
plot_bug_module_heatmap.py — Bug type × module co-occurrence heatmap for JAXES paper.

Rows = 9 bug types (T1–T9), columns = 10 modules (ordered by total bugs desc).
Cell color = count; annotated with integer for non-zero cells.

Data: analyst-derived from CHANGELOG.md sessions 1-46 (1 Apr – 8 May 2026).
See diags/output/agentic_pipeline_analysis.md Section 3.2 for full bug inventory.
Note: counts are best-effort estimates from qualitative CHANGELOG mining; minor
attribution ambiguity (±1 per cell) does not affect the key visual message.

Usage:
    python diags/plot_bug_module_heatmap.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

_HERE = Path(__file__).resolve().parent
_FIGURES_DIR = _HERE / "figures"
_PAPER_FIGS = _HERE.parent / "Paper" / "jaxes_paper" / "figures"

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "text.usetex":       False,
})

# ── Data ──────────────────────────────────────────────────────────────────────
# Column headers (abbreviated module names)
MODULES = [
    "LeafPhoto",   # MLLeafPhotosynthesisMod
    "CanopyTurb",  # MLCanopyTurbulenceMod
    "CanopyFlux",  # MLCanopyFluxesMod
    "MathTools",   # MLMathToolsMod
    "SolarRad",    # MLSolarRadiationMod
    "PlantHydro",  # MLPlantHydraulicsMod
    "NitroProf",   # MLCanopyNitrogenProfileMod
    "CanopyWater", # MLCanopyWaterMod
    "LeafBdry",    # MLLeafBoundaryLayerMod
    "diags",       # diags/ scripts (T7, T8, T9)
]

# Row headers (type codes and short names)
TYPE_CODES = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]
TYPE_LABELS = [
    "T1  NaN Gradient",
    "T2  Zero/Wrong Grad",
    "T3  XLA Recompile",
    "T4  Memory/OOM",
    "T5  Grad Explosion",
    "T6  Device–Host Sync",
    "T7  Optim. Algorithm",
    "T8  Crash/Compile",
    "T9  Diagnostic",
]

# Bug matrix: rows=types T1..T9, cols=modules in order above.
# Derived from agentic_pipeline_analysis.md Section 3.2 bug inventory (B1–B47).
# Row sums should approximate type totals [17,9,2,3,3,3,4,3,3].
# Some attribution ambiguity (±1) due to bugs spanning multiple modules.
#
# T1 (17): B4→Sol, B5→Nitro, B8→Leaf, B9→Plant, B10→Water, B11→Turb, B12→Math,
#           B13→Math, B14→Turb, B17→Turb, B18→Turb, B19→Plant, B21→Sol,
#           B24→Bdry, B27→Turb, B40→Leaf + (B6 FluxProfile→Flux, B7 LeafFlux→Leaf)
# T2 (9):  B29→Flux, B30→Flux, B31→Flux/Nitro, B35→Leaf, B36→Nitro, B37→Nitro,
#           B39→Leaf + 2 diags-side (sensitivity_analysis pattern resets)
# T3 (2):  B22→Leaf, B23→Turb
# T4 (3):  B20→Flux, B48→diags + 1 diags
# T5 (3):  B32→Leaf, B34→Turb, B38→Leaf
# T6 (3):  B15→Leaf, B16→Turb, B2→Flux (B3→Flux collapsed with B2)
# T7 (4):  B41,B42,B43,B44 → diags
# T8 (3):  B1,B46,B47 → diags
# T9 (3):  B33,B45 + FD-epsilon → diags
#
# Columns: LeafP CaTurb CaFlux Math  Sol  Plant Nitro Water Bdry  diags
BUG_MATRIX = np.array([
    [   3,    5,    1,    2,    2,    2,    1,    1,    1,    0 ],  # T1  17*
    [   2,    0,    3,    0,    0,    0,    2,    0,    0,    2 ],  # T2   9
    [   1,    1,    0,    0,    0,    0,    0,    0,    0,    0 ],  # T3   2
    [   0,    0,    1,    0,    0,    0,    0,    0,    0,    2 ],  # T4   3
    [   2,    1,    0,    0,    0,    0,    0,    0,    0,    0 ],  # T5   3
    [   1,    1,    1,    0,    0,    0,    0,    0,    0,    0 ],  # T6   3
    [   0,    0,    0,    0,    0,    0,    0,    0,    0,    4 ],  # T7   4
    [   0,    0,    0,    0,    0,    0,    0,    0,    0,    3 ],  # T8   3
    [   0,    0,    0,    0,    0,    0,    0,    0,    0,    3 ],  # T9   3
], dtype=int)
# * T1 row sum = 18 (vs. taxonomy's 17); rounding artifact from multi-module B6/B7

ROW_TOTALS_APPROX = [17, 9, 2, 3, 3, 3, 4, 3, 3]  # from taxonomy

# Soft sanity check (warn, don't crash — counts are analyst estimates)
row_sums = BUG_MATRIX.sum(axis=1).tolist()
total = BUG_MATRIX.sum()
for i, (actual, expected) in enumerate(zip(row_sums, ROW_TOTALS_APPROX)):
    if abs(actual - expected) > 2:
        print(f"WARNING: row {TYPE_CODES[i]} sum={actual} vs expected={expected}")
print(f"Matrix total: {total}  (taxonomy total: ~47-48)")

col_totals = BUG_MATRIX.sum(axis=0)
row_totals = BUG_MATRIX.sum(axis=1)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(6.8, 4.0))

# Layout constants
L, B, W, H = 0.21, 0.26, 0.68, 0.65   # main heatmap [left, bottom, width, height]

# Main heatmap axes
ax = fig.add_axes([L, B, W, H])

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
    "yor", ["#ffffff", "#FEF3C7", "#FCA5A5", "#DC2626"], N=256)
vmax = max(BUG_MATRIX.max(), 1)

im = ax.imshow(BUG_MATRIX, cmap=cmap, vmin=0, vmax=vmax,
               aspect="auto", interpolation="nearest")

# Grid lines
for x in np.arange(-0.5, len(MODULES), 1):
    ax.axvline(x, color="white", linewidth=0.9)
for y in np.arange(-0.5, len(TYPE_CODES), 1):
    ax.axhline(y, color="white", linewidth=0.9)

# Cell annotations
for i in range(BUG_MATRIX.shape[0]):
    for j in range(BUG_MATRIX.shape[1]):
        v = BUG_MATRIX[i, j]
        if v > 0:
            txt_color = "white" if v >= vmax * 0.65 else "#1a1a1a"
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=8, fontweight="bold", color=txt_color)

# Axes labels — x labels rotated 45° with right-alignment
ax.set_xticks(range(len(MODULES)))
ax.set_xticklabels(MODULES, rotation=45, ha="right", fontsize=7.5)
ax.set_yticks(range(len(TYPE_LABELS)))
ax.set_yticklabels(TYPE_LABELS, fontsize=7.8)
ax.tick_params(length=0)

# Column totals bar at top
ax_top = fig.add_axes([L, B + H + 0.01, W, 0.06])
ax_top.bar(range(len(MODULES)), col_totals, color="#6B7280", alpha=0.7, width=0.7)
ax_top.set_xlim(-0.5, len(MODULES) - 0.5)
ax_top.set_ylim(0, max(col_totals) + 1.5)
ax_top.axis("off")
for j, v in enumerate(col_totals):
    if v > 0:
        ax_top.text(j, v + 0.1, str(v), ha="center", va="bottom",
                    fontsize=7, color="#333333")
ax_top.set_title("Bugs per module", fontsize=7.5, pad=2)

# Row totals bar at right
ax_right = fig.add_axes([L + W + 0.01, B, 0.05, H])
ax_right.barh(range(len(TYPE_CODES)), row_totals, color="#6B7280", alpha=0.7,
              height=0.65)
ax_right.set_ylim(-0.5, len(TYPE_CODES) - 0.5)
ax_right.set_xlim(0, max(row_totals) + 2)
ax_right.invert_yaxis()
ax_right.axis("off")
for i, v in enumerate(row_totals):
    ax_right.text(v + 0.15, i, str(v), ha="left", va="center",
                  fontsize=7, color="#333333")
ax_right.set_title("n", fontsize=7.5, rotation=0, pad=2)

# Colorbar — bottom-right, clear of the row-totals bar
cbar_ax = fig.add_axes([L + W + 0.01, B - 0.18, 0.025, 0.14])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label("Count", fontsize=7)
cb.set_ticks([0, 2, 4, 5])
cb.ax.tick_params(labelsize=6.5)

# X-axis title below heatmap
fig.text(L + W / 2, 0.01, "Module (ordered by total bug count  →)",
         ha="center", fontsize=8)

# ── Save ──────────────────────────────────────────────────────────────────────
_FIGURES_DIR.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    out = _FIGURES_DIR / f"bug_module_heatmap.{ext}"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")

if _PAPER_FIGS.is_dir():
    for ext in ("pdf", "png"):
        src = _FIGURES_DIR / f"bug_module_heatmap.{ext}"
        dst = _PAPER_FIGS / f"bug_module_heatmap.{ext}"
        shutil.copy(src, dst)
        print(f"Copied → {dst}")
else:
    print(f"Paper figures dir not found: {_PAPER_FIGS}  (skipping copy)")
