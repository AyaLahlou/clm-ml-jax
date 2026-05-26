"""
plot_repair_timeline.py — Differentiability repair campaign timeline for JAXES paper.

Shows session density (bugs fixed per session), cumulative bug count, and phase
annotations over the 5-week repair campaign (1 Apr – 8 May 2026, sessions 1-46).

Data: hardcoded from CHANGELOG.md sessions 1-46.

Usage:
    python diags/plot_repair_timeline.py
"""
from __future__ import annotations

import shutil
from pathlib import Path
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "text.usetex":       False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

_RAW = [
    ("2026-04-01", 12),
    ("2026-04-02",  5),
    ("2026-04-03",  4),
    ("2026-04-06",  5),
    ("2026-04-08",  4),
    ("2026-04-09",  3),
    ("2026-04-10",  5),
    ("2026-04-14",  2),
    ("2026-04-22",  2),
    ("2026-04-23",  1),
    ("2026-04-24",  1),
    ("2026-04-28",  2),
    ("2026-04-29",  1),
    ("2026-05-08",  3),
]

dates_raw = [date.fromisoformat(d) for d, _ in _RAW]
bugs_raw  = [b for _, b in _RAW]

d_start = date(2026, 4, 1)
d_end   = date(2026, 5, 9)
n_days  = (d_end - d_start).days
all_dates = [d_start + timedelta(days=i) for i in range(n_days)]

bugs_per_day = np.zeros(n_days, dtype=int)
for d, b in zip(dates_raw, bugs_raw):
    bugs_per_day[(d - d_start).days] += b

cumulative = np.cumsum(bugs_per_day)

# Phase bands — label placed inside band at mid-height of the bar region
PHASES = [
    (date(2026, 4,  1), date(2026, 4,  6), "#FEF3C7", "NaN\nelim. (T1)"),
    (date(2026, 4,  6), date(2026, 4, 12), "#FECACA", "Zero-grad\nroot cause"),
    (date(2026, 4, 12), date(2026, 4, 27), "#DBEAFE", "IFT + param\ninjection"),
    (date(2026, 4, 27), date(2026, 5,  9), "#DCFCE7", "Calibration\n(T7)"),
]

# Milestones: (date, label, prefer_left) — prefer_left=True annotates text to the left
MILESTONES = [
    (date(2026, 4,  1), "First jax.grad\n12 NaN bugs",      False),
    (date(2026, 4, 10), "5 params\nGPU-verified",            False),
    (date(2026, 4, 27), "7-param Jacobian\n(all non-zero)",  False),
    (date(2026, 5,  8), "p=10 calibration\n+ Tikhonov",      True),
]

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.8, 3.2))
ax_r = ax.twinx()

dates_num = mdates.date2num(all_dates)
ymax_bar  = max(bugs_per_day) * 1.2   # bar axis ceiling
cum_max   = max(cumulative)

# Phase bands
for p_start, p_end, color, label in PHASES:
    xs  = mdates.date2num([p_start, p_end])
    mid = mdates.date2num([p_start + (p_end - p_start) / 2])[0]
    ax.axvspan(xs[0], xs[1], color=color, alpha=0.85, zorder=0)
    # Label at ~20% height inside the band
    ax.text(mid, ymax_bar * 0.18, label,
            ha="center", va="bottom", fontsize=6.5,
            style="italic", color="#555555", zorder=4)

# Session bars
nonzero = bugs_per_day > 0
for dn, b, nz in zip(dates_num, bugs_per_day, nonzero):
    if nz:
        ax.bar(dn, b, width=0.65, color="#1D4ED8", alpha=0.75, zorder=3)

# Cumulative step line on right axis
ax_r.step(dates_num, cumulative, where="post", color="#6B7280",
          linewidth=1.5, linestyle="--", zorder=2)
ax_r.fill_between(dates_num, 0, cumulative, step="post",
                  color="#6B7280", alpha=0.07, zorder=1)

# Milestone annotations — alternate y positions to avoid overlap
y_fracs = [0.88, 0.70, 0.52, 0.70]
for (m_date, label, prefer_left), y_frac in zip(MILESTONES, y_fracs):
    mx   = mdates.date2num([m_date])[0]
    y_pos = cum_max * y_frac
    ax.axvline(mx, color="#555555", linewidth=0.85, linestyle=":", zorder=4, alpha=0.7)
    x_off = -0.8 if prefer_left else 0.6
    ha    = "right" if prefer_left else "left"
    ax_r.annotate(label, xy=(mx, y_pos),
                  xytext=(mx + x_off, y_pos),
                  fontsize=6.2, color="#333333", ha=ha, va="center",
                  bbox=dict(boxstyle="round,pad=0.22", fc="white",
                            ec="#aaaaaa", alpha=0.92, linewidth=0.5),
                  zorder=6)

# Axes
x_lo = mdates.date2num([d_start])[0] - 0.5
x_hi = mdates.date2num([d_end])[0]   + 0.5
ax.set_xlim(x_lo, x_hi)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
ax.set_ylim(0, ymax_bar)
ax.set_ylabel("Bugs fixed\nper session", fontsize=8.5)
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))
ax.tick_params(axis="x", rotation=25)

ax_r.set_ylabel("Cumulative bugs fixed", fontsize=8.5, color="#6B7280")
ax_r.tick_params(axis="y", labelcolor="#6B7280")
ax_r.set_ylim(0, cum_max * 1.15)
ax_r.spines["top"].set_visible(False)
ax_r.spines["right"].set_visible(True)

# Legend
bar_patch = mpatches.Patch(color="#1D4ED8", alpha=0.75, label="Bugs fixed (left axis)")
cum_line  = plt.Line2D([0], [0], color="#6B7280", linestyle="--", linewidth=1.5,
                        label="Cumulative (right axis)")
ax.legend(handles=[bar_patch, cum_line], loc="upper left", fontsize=7,
          framealpha=0.9, handlelength=1.4)

ax.set_title("Phase 5b differentiability repair campaign  "
             "(1 Apr – 8 May 2026,  46 sessions)", fontsize=8.5, pad=5)

fig.tight_layout()

# ── Save ───────────────────────────────────────────────────────────────────────
_FIGURES_DIR.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    out = _FIGURES_DIR / f"repair_timeline.{ext}"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")

if _PAPER_FIGS.is_dir():
    for ext in ("pdf", "png"):
        shutil.copy(_FIGURES_DIR / f"repair_timeline.{ext}",
                    _PAPER_FIGS / f"repair_timeline.{ext}")
        print(f"Copied → {_PAPER_FIGS / f'repair_timeline.{ext}'}")
else:
    print(f"Paper figures dir not found: {_PAPER_FIGS}  (skipping copy)")
