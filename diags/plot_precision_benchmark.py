"""
Publication figure: float32 vs float64 GPU throughput benchmark (jobs 7578655_0/1).

Two-panel layout (7" × 3.8"):
  (a) Throughput (ms/sample) vs ensemble size N — f32 vs f64 lines, plus
      hypothetical "ideal 2× f32 speedup" reference.
  (b) XLA compile time vs N for both precisions.

Key finding: f32 ≈ f64 (no speedup). Users get float64 fidelity at no extra cost.

Data: diags/figures/precision_benchmark_f64.csv
      diags/figures/precision_benchmark_f32.csv
Out:  diags/figures/precision_benchmark.{pdf,png}
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ─────────────────────────────────────────────────────────────────────
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

for fname in ("precision_benchmark_f64.csv", "precision_benchmark_f32.csv"):
    if not (FIGURES_DIR / fname).exists():
        sys.exit(f"ERROR: {FIGURES_DIR / fname} not found. "
                 "Run diags/benchmark_precision.py first.")

def _load(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    N         = np.array([float(r["N"])           for r in rows])
    compile_s = np.array([float(r["compile_s"])   for r in rows])
    ms_sample = np.array([float(r["ms_per_sample"]) for r in rows])
    return N, compile_s, ms_sample

N64, c64, t64 = _load(FIGURES_DIR / "precision_benchmark_f64.csv")
N32, c32, t32 = _load(FIGURES_DIR / "precision_benchmark_f32.csv")

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

C64  = "#1f77b4"   # blue for f64
C32  = "#ff7f0e"   # orange for f32
CREF = "#bbbbbb"   # grey for hypothetical 2× line

fig, axes = plt.subplots(1, 2, figsize=(7, 3.8))
fig.subplots_adjust(wspace=0.38)

xticks = [1, 8, 32, 128, 512, 1024, 2048]
x_fmt  = mticker.FuncFormatter(lambda x, _: str(int(x)))

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (a): throughput (ms/sample) vs N
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[0]

ax.plot(N64, t64, color=C64, lw=2.0, marker="o", ms=5, zorder=3,
        label="float64 (JAX default)")
ax.plot(N32, t32, color=C32, lw=2.0, marker="s", ms=5, zorder=3,
        linestyle="--", label="float32")

# Ideal 2× f32 speedup reference
t_ideal = t64 / 2.0
ax.plot(N64, t_ideal, color=CREF, lw=1.4, linestyle=":", zorder=1,
        label="Ideal f32 speedup (2×)")

# Shade gap between actual f32 and ideal (where f32 fails to speed up)
ax.fill_between(N64, t_ideal, np.minimum(t64, t32),
                color=C32, alpha=0.08, zorder=0)

# Mean throughput annotation for large N
mask = N64 >= 32
mean_t64 = float(t64[mask].mean())
mean_t32 = float(t32[N32 >= 32].mean())
ratio = mean_t32 / mean_t64

ax.axhline(mean_t64, color=C64, lw=0.7, linestyle="-.", alpha=0.5)
ax.text(N64[-1] * 1.06, mean_t64 + 0.4,
        f"f64 ≈ {mean_t64:.1f} ms", fontsize=7, color=C64, va="bottom")
ax.text(N32[-1] * 1.06, mean_t32 - 0.6,
        f"f32 ≈ {mean_t32:.1f} ms\n({ratio:.2f}× f64)",
        fontsize=7, color=C32, va="top")

ax.set_xscale("log")
ax.set_xlabel("Ensemble size $N$ (columns)", fontsize=9)
ax.set_ylabel("Throughput  (ms sample$^{-1}$)", fontsize=9)
ax.set_title("(a) GPU throughput  —  float32 vs float64", fontsize=9)
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(True, which="both", alpha=0.18)
ax.tick_params(labelsize=8)
ax.set_xticks(xticks)
ax.xaxis.set_major_formatter(x_fmt)
ax.set_ylim(0, max(t64.max(), t32.max()) * 1.30)

ax.text(0.04, 0.07,
        "No f32 speedup observed:\nf32 ≈ f64 for all $N$\n"
        r"(memory-bandwidth bound,$\not{\rm FLOP}$-bound)",
        transform=ax.transAxes, fontsize=6.8, color="#444444",
        va="bottom", ha="left",
        bbox=dict(facecolor="white", edgecolor="#cccccc",
                  linewidth=0.5, boxstyle="round,pad=0.3"))

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (b): XLA compile time vs N
# ═══════════════════════════════════════════════════════════════════════════════
ax2 = axes[1]

ax2.plot(N64, c64, color=C64, lw=2.0, marker="o", ms=5, zorder=3,
         label="float64")
ax2.plot(N32, c32, color=C32, lw=2.0, marker="s", ms=5, zorder=3,
         linestyle="--", label="float32")

ax2.set_xscale("log")
ax2.set_xlabel("Ensemble size $N$ (columns)", fontsize=9)
ax2.set_ylabel("XLA JIT compile time (s)", fontsize=9)
ax2.set_title("(b) XLA compile time vs $N$", fontsize=9)
ax2.legend(fontsize=7.5)
ax2.grid(True, which="both", alpha=0.18)
ax2.tick_params(labelsize=8)
ax2.set_xticks(xticks)
ax2.xaxis.set_major_formatter(x_fmt)

# Annotate compile time at each point
for N_arr, c_arr, col in [(N64, c64, C64), (N32, c32, C32)]:
    for ni, ci in zip(N_arr, c_arr):
        ax2.text(ni, ci + 15, f"{ci:.0f}", fontsize=5.5, color=col,
                 ha="center", va="bottom")

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "CLM-ml-jax GPU throughput: float32 vs float64  "
    "—  CHATS7 · Quadro RTX 8000\n"
    "(median of 10 vmapped forward passes per $N$, after JIT warm-up)",
    fontsize=8, y=1.01,
)

for ext in ("pdf", "png"):
    out = FIGURES_DIR / f"precision_benchmark.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")

plt.close(fig)
print("=== plot_precision_benchmark.py complete ===")
