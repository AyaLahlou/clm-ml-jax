"""
plot_bug_taxonomy.py — Bug type distribution for Phase 5b differentiability repair campaign.

Horizontal bar chart: 9 bug types, ordered by count, with a twin top-axis scatter
showing average debugging attempts per type. Intended for JAXES paper appendix.

Data source: CHANGELOG.md sessions 1-46 (1 Apr – 8 May 2026).

Usage:
    python diags/plot_bug_taxonomy.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_HERE = Path(__file__).resolve().parent
_FIGURES_DIR = _HERE / "figures"
_PAPER_FIGS = _HERE.parent / "Paper" / "jaxes_paper" / "figures"

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   7.5,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "text.usetex":       False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

TYPES = [
    ("T1", "NaN Gradient\n(jnp.where)",     17, 35.4, 1.0,  True),
    ("T2", "Zero / Wrong\nGradient",          9, 18.8, 1.3,  False),
    ("T7", "Optimization\nAlgorithm",         4,  8.3, 1.5,  False),
    ("T4", "Memory / OOM",                    3,  6.3, 1.7,  False),
    ("T5", "Gradient\nExplosion",             3,  6.3, 2.0,  False),
    ("T6", "Device–Host\nSync",               3,  6.3, 1.0,  True),
    ("T8", "Crash / Compile",                 3,  6.3, 1.3,  False),
    ("T9", "Diagnostic\nReliability",         3,  6.3, 1.3,  False),
    ("T3", "XLA\nRecompilation",              2,  4.2, 1.0,  True),
]

TYPE_COLORS = {
    "T1": "#DC2626", "T2": "#EA580C", "T3": "#CA8A04",
    "T4": "#65A30D", "T5": "#7C3AED", "T6": "#0891B2",
    "T7": "#1D4ED8", "T8": "#6B7280", "T9": "#9CA3AF",
}

codes    = [t[0] for t in TYPES]
labels   = [f"{t[0]}  {t[1]}" for t in TYPES]
counts   = [t[2] for t in TYPES]
pcts     = [t[3] for t in TYPES]
avg_att  = [t[4] for t in TYPES]
one_shot = [t[5] for t in TYPES]
colors   = [TYPE_COLORS[c] for c in codes]

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.2, 3.8))

# ax_top shares the y-axis; its x-axis (avg attempts) appears at the TOP
ax_top = ax.twiny()

y = np.arange(len(TYPES))
bar_h = 0.60

# ── Bars (bottom x-axis = bug count) ──────────────────────────────────────────
bars = ax.barh(y, counts, height=bar_h, color=colors, alpha=0.82, zorder=3)

for bar, is_one in zip(bars, one_shot):
    if is_one:
        bar.set_hatch("///")
        bar.set_edgecolor("white")
        bar.set_linewidth(0.4)

for bar, n, p in zip(bars, counts, pcts):
    x_txt = n * 0.5 if n > 2 else n + 0.2
    ha    = "center" if n > 2 else "left"
    clr   = "white"  if n > 2 else "#333333"
    ax.text(x_txt, bar.get_y() + bar.get_height() / 2,
            f"{n}  ({p:.0f}%)", va="center", ha=ha,
            color=clr, fontsize=7.5, fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7.8)
ax.set_xlabel("Number of bugs  ($n$ = 48 total)", fontsize=8.5)
ax.set_xlim(0, 19)
ax.set_ylim(-0.5, len(TYPES) - 0.5)
ax.xaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.spines["bottom"].set_visible(True)
ax.spines["left"].set_visible(True)

# ── Scatter (top x-axis = avg attempts) ───────────────────────────────────────
ax_top.scatter(avg_att, y, color="black", marker="D", s=30, zorder=5)
ax_top.axvline(1.33, color="#999999", linestyle="--", linewidth=0.9, zorder=2)

ax_top.set_xlabel("Avg. attempts to fix", fontsize=8.5, labelpad=4)
ax_top.set_xlim(0.7, 2.5)
ax_top.set_ylim(-0.5, len(TYPES) - 0.5)
ax_top.set_xticks([1.0, 1.33, 1.5, 2.0])
ax_top.set_xticklabels(["1.0", "1.33\n(mean)", "1.5", "2.0"], fontsize=7.5)
ax_top.spines["right"].set_visible(False)
ax_top.spines["bottom"].set_visible(False)

# ── Legend (inside axes, lower right) ─────────────────────────────────────────
patch_auto  = mpatches.Patch(facecolor="white", hatch="///", edgecolor="gray",
                              linewidth=0.5, label="Always 1-shot (autonomous)")
patch_multi = mpatches.Patch(facecolor="#999999", alpha=0.5,
                              label="Multi-attempt (human re-spec.)")
handle_mean = plt.Line2D([0], [0], color="#999999", linestyle="--",
                          linewidth=0.9, label="Mean attempts = 1.33")
handle_dot  = plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="black",
                          markersize=5, label="Avg. attempts (top axis)")
ax.legend(handles=[patch_auto, patch_multi, handle_mean, handle_dot],
          loc="lower right", fontsize=7, framealpha=0.88,
          handlelength=1.3, handletextpad=0.4, borderpad=0.5)

ax.set_title("Phase 5b repair campaign: bug type distribution", fontsize=9, pad=4)

fig.tight_layout()

# ── Save ───────────────────────────────────────────────────────────────────────
_FIGURES_DIR.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    out = _FIGURES_DIR / f"bug_taxonomy_chart.{ext}"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")

if _PAPER_FIGS.is_dir():
    for ext in ("pdf", "png"):
        shutil.copy(_FIGURES_DIR / f"bug_taxonomy_chart.{ext}",
                    _PAPER_FIGS / f"bug_taxonomy_chart.{ext}")
        print(f"Copied → {_PAPER_FIGS / f'bug_taxonomy_chart.{ext}'}")
else:
    print(f"Paper figures dir not found: {_PAPER_FIGS}  (skipping copy)")
