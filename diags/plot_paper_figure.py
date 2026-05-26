"""
plot_paper_figure.py — Publication figure for CLM-ML-JAX paper (NeurIPS AI4Science).

Three-panel narrative:
  A. Differentiable mode speedup  — lax.scan removes 164–3100× Python-loop overhead,
                                    enabling gradient computation at near-zero extra cost.
  B. Amortized per-sample cost    — JAX GPU drops from 25ms to 11.5ms/sample as N grows;
                                    Fortran is flat at ~54ms (cannot batch).
  C. Total ensemble wall-clock    — N=2048 samples: Fortran 553s (O(N)) vs JAX GPU 23.4s;
                                    24× wall-clock reduction.

Data sources:
  diags/figures/ensemble_benchmark.csv           (JAX GPU + CPU, A40, Euler)
  diags/figures/ensemble_benchmark_fortran.csv   (Fortran sequential, CHATS7 RK4)

Usage:
  python diags/plot_paper_figure.py
  python diags/plot_paper_figure.py --out path/to/paper_figure.pdf
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.gridspec import GridSpec

_HERE        = Path(__file__).resolve().parent
_FIGURES_DIR = _HERE / "figures"

# ── Color palette ─────────────────────────────────────────────────────────────
C_GPU    = "#2563EB"   # blue
C_CPU    = "#DC2626"   # red
C_FORT   = "#D97706"   # amber
C_DIFF   = "#16A34A"   # green  (lax.scan diff mode)
C_NDIFF  = "#9333EA"   # purple (Python loop non-diff)
ALPHA    = 0.88

# ── lax.scan confirmed measurements (A100, jobs 7329052+7329441) ──────────────
_LAXSCAN = {
    "Euler": {"diff_ms": 37.2,  "nondiff_ms": 6_095.0,  "speedup": 164},
    "RK4":   {"diff_ms": 37.8,  "nondiff_ms": 117_164.0, "speedup": 3100},
}

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_jax_ensemble(path: Path):
    gpu, cpu = {}, {}
    with open(path) as f:
        for row in csv.DictReader(f):
            N = int(row["N"])
            ms_sample = float(row["ms_per_sample"]) if row["ms_per_sample"] else np.nan
            vmap_total = float(row["vmap_ss_ms"]) if row["vmap_ss_ms"] else np.nan
            if row["backend"] == "gpu":
                gpu[N] = {"ms_per_sample": ms_sample, "total_ms": vmap_total}
            else:
                cpu[N] = {"ms_per_sample": ms_sample, "total_ms": vmap_total}
    return gpu, cpu


def _load_fortran_ensemble(path: Path):
    seq = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if "seq" in row["backend"]:
                N = int(row["N"])
                seq[N] = {
                    "ms_per_sample": float(row["ms_per_sample"]),
                    "total_ms": float(row["run_wall_s"]) * 1000.0,
                    "measured": "extrapolated" not in row["notes"],
                }
    return seq


# ── Panel helpers ─────────────────────────────────────────────────────────────

def _panel_a(ax):
    """Panel A: lax.scan diff vs Python-loop non-diff speedup (bar chart)."""
    modes = ["Euler", "RK4"]
    x = np.arange(len(modes))
    w = 0.33
    gap = 0.04

    diff_ms  = [_LAXSCAN[m]["diff_ms"]   for m in modes]
    ndiff_ms = [_LAXSCAN[m]["nondiff_ms"] for m in modes]

    ax.bar(x - w/2 - gap/2, diff_ms,  width=w, color=C_DIFF,  alpha=ALPHA,
           label="Differentiable\n(lax.scan)", zorder=3)
    ax.bar(x + w/2 + gap/2, ndiff_ms, width=w, color=C_NDIFF, alpha=ALPHA,
           label="Non-differentiable\n(Python loop)", zorder=3)

    # Speedup badges
    for i, mode in enumerate(modes):
        sp = _LAXSCAN[mode]["speedup"]
        y_pos = diff_ms[i] * 3.5
        ax.text(x[i], y_pos, f"{sp:,}×",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF9C3",
                          edgecolor="#D97706", linewidth=0.8, alpha=0.9))

    # Value labels on non-diff bars
    for i, (xi, val) in enumerate(zip(x, ndiff_ms)):
        label = f"{val/1000:.0f}s" if val >= 1000 else f"{val:.0f}ms"
        ax.text(xi + w/2 + gap/2, val * 1.08, label,
                ha="center", va="bottom", fontsize=7.5, color=C_NDIFF)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["Euler\n(1st order)", "RK4\n(4th order)"], fontsize=9.5)
    ax.set_ylabel("Steady-state time (ms / step)", fontsize=9)
    ax.set_title("A", fontweight="bold", fontsize=11, loc="left", pad=4)
    ax.set_xlabel("Time-integration scheme", fontsize=9)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.yaxis.grid(True, which="both", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Subtitle
    ax.text(0.5, -0.27,
            "Gradient computation via lax.scan\neliminates Python-loop overhead",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            style="italic", color="#374151")


def _panel_b(ax, gpu: dict, cpu: dict, fort: dict):
    """Panel B: ms/sample vs N — amortized cost."""
    Ns_gpu  = sorted(gpu)
    Ns_cpu  = sorted(cpu)
    Ns_fort = sorted(fort)

    mss_gpu  = [gpu[n]["ms_per_sample"]  for n in Ns_gpu]
    mss_cpu  = [cpu[n]["ms_per_sample"]  for n in Ns_cpu]
    mss_fort = [fort[n]["ms_per_sample"] for n in Ns_fort]

    ax.plot(Ns_gpu, mss_gpu, "o-", color=C_GPU, lw=2.2, ms=6.5,
            label="JAX GPU (A40)", zorder=4)
    ax.plot(Ns_cpu, mss_cpu, "s-", color=C_CPU, lw=2.2, ms=6.5,
            label="JAX CPU", zorder=4)

    # Fortran: separate measured vs extrapolated markers
    meas_N  = [n for n in Ns_fort if fort[n]["measured"]]
    meas_ms = [fort[n]["ms_per_sample"] for n in meas_N]
    extr_N  = [n for n in Ns_fort if not fort[n]["measured"]]
    extr_ms = [fort[n]["ms_per_sample"] for n in extr_N]

    # Draw a flat reference line spanning full x range
    all_N_range = [min(Ns_gpu + Ns_cpu + Ns_fort), max(Ns_gpu + Ns_cpu + Ns_fort)]
    fort_ref = np.mean(mss_fort)
    ax.hlines(fort_ref, all_N_range[0], all_N_range[1],
              colors=C_FORT, lw=1.6, ls="--", zorder=3,
              label=f"Fortran sequential (~{fort_ref:.0f} ms/sample, no batching)")
    ax.scatter(meas_N, meas_ms, marker="D", color=C_FORT, s=40, zorder=5)
    ax.scatter(extr_N, extr_ms, marker="D", color=C_FORT, s=40, zorder=5,
               facecolors="none", linewidths=1.5)

    # Annotate GPU plateau
    n_plateau, ms_plateau = Ns_gpu[-1], mss_gpu[-1]
    ax.annotate(f"{ms_plateau:.1f} ms/sample",
                xy=(n_plateau, ms_plateau), xytext=(-55, -22),
                textcoords="offset points", fontsize=8, color=C_GPU,
                arrowprops=dict(arrowstyle="-", color=C_GPU, lw=1))

    # Annotate Fortran flat line
    ax.annotate(f"{fort_ref:.0f} ms/sample",
                xy=(all_N_range[1], fort_ref), xytext=(-80, 12),
                textcoords="offset points", fontsize=8, color=C_FORT,
                arrowprops=dict(arrowstyle="-", color=C_FORT, lw=1))

    # Speedup at N=2048
    gpu_ms_max = gpu[max(Ns_gpu)]["ms_per_sample"]
    speedup = fort_ref / gpu_ms_max
    ax.text(0.97, 0.08,
            f"GPU {speedup:.1f}× faster\nthan Fortran\n(N = {max(Ns_gpu):,})",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, fontweight="bold", color=C_GPU,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=C_GPU, linewidth=0.9, alpha=0.9))

    all_Ns_ticks = sorted({1, 8, 32, 128, 512, 1024, 2048})
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(all_Ns_ticks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Ensemble size N (parameter samples)", fontsize=9)
    ax.set_ylabel("ms / sample", fontsize=9)
    ax.set_title("B", fontweight="bold", fontsize=11, loc="left", pad=4)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.yaxis.grid(True, which="major", alpha=0.25, lw=0.6)
    ax.xaxis.grid(True, which="major", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.text(0.5, -0.2,
            "JAX amortizes cost across samples via vmap;\nFortran must execute each sample serially",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            style="italic", color="#374151")


def _panel_c(ax, gpu: dict, cpu: dict, fort: dict):
    """Panel C: total wall-clock vs N — the O(N) vs amortized story."""
    Ns_gpu  = sorted(gpu)
    Ns_cpu  = sorted(cpu)
    Ns_fort = sorted(fort)

    total_gpu  = [gpu[n]["total_ms"]  / 1000.0 for n in Ns_gpu]   # seconds
    total_cpu  = [cpu[n]["total_ms"]  / 1000.0 for n in Ns_cpu if not np.isnan(cpu[n]["total_ms"])]
    Ns_cpu_ok  = [n for n in Ns_cpu if not np.isnan(cpu[n]["total_ms"])]
    total_fort = [fort[n]["total_ms"] / 1000.0 for n in Ns_fort]

    # Shaded region between GPU and Fortran
    # Interpolate GPU onto Fortran N values for shading
    fort_Ns_arr = np.array(Ns_fort)
    gpu_Ns_arr  = np.array(Ns_gpu)
    gpu_tot_arr = np.array(total_gpu)
    gpu_interp  = np.interp(np.log2(fort_Ns_arr), np.log2(gpu_Ns_arr), gpu_tot_arr)
    ax.fill_between(fort_Ns_arr, gpu_interp, total_fort,
                    color=C_GPU, alpha=0.08, zorder=1)

    ax.plot(Ns_gpu, total_gpu, "o-", color=C_GPU, lw=2.2, ms=6.5,
            label="JAX GPU (A40)", zorder=4)
    ax.plot(Ns_cpu_ok, total_cpu, "s-", color=C_CPU, lw=2.2, ms=6.5,
            label="JAX CPU", zorder=4)

    # Fortran: solid line through measured points, dashed through extrapolated
    meas_N_fort   = [n for n in Ns_fort if fort[n]["measured"]]
    extr_N_fort   = [n for n in Ns_fort if not fort[n]["measured"]]
    meas_tot_fort = [fort[n]["total_ms"] / 1000.0 for n in meas_N_fort]
    extr_tot_fort = [fort[n]["total_ms"] / 1000.0 for n in extr_N_fort]

    # Draw as one continuous line (solid measured → dashed extrapolated)
    # Connect them with a boundary point
    boundary_N  = meas_N_fort[-1]  # N=32
    boundary_ms = fort[boundary_N]["total_ms"] / 1000.0

    ax.plot(meas_N_fort, meas_tot_fort, "D-", color=C_FORT, lw=2.2, ms=6,
            label="Fortran sequential (O(N))", zorder=4)
    ax.plot([boundary_N] + extr_N_fort, [boundary_ms] + extr_tot_fort,
            "D--", color=C_FORT, lw=2.2, ms=6, zorder=4, markerfacecolor="none",
            markeredgewidth=1.5)

    # O(N) slope annotation (triangle)
    slope_x = [128, 1024]
    slope_y = [fort[128]["total_ms"] / 1000.0, fort[1024]["total_ms"] / 1000.0]
    ax.annotate("", xy=(slope_x[1], slope_y[1]), xytext=(slope_x[0], slope_y[1]),
                arrowprops=dict(arrowstyle="-", color=C_FORT, lw=0.8, ls=":"))
    ax.annotate("", xy=(slope_x[1], slope_y[1]), xytext=(slope_x[1], slope_y[0]),
                arrowprops=dict(arrowstyle="-", color=C_FORT, lw=0.8, ls=":"))
    ax.text(np.sqrt(slope_x[0] * slope_x[1]), slope_y[0] * 0.6,
            "slope = 1\n(O(N))", ha="center", va="top", fontsize=7.5,
            color=C_FORT, style="italic")

    # Endpoint annotations at N=2048
    N_end = max(Ns_gpu)
    gpu_end  = gpu[N_end]["total_ms"] / 1000.0
    fort_end = fort[N_end]["total_ms"] / 1000.0
    total_speedup = fort_end / gpu_end

    # Arrow+label for GPU endpoint
    ax.annotate(f"{gpu_end:.1f}s",
                xy=(N_end, gpu_end), xytext=(-65, 12),
                textcoords="offset points", fontsize=8.5, color=C_GPU,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C_GPU, lw=1))

    # Arrow+label for Fortran endpoint
    ax.annotate(f"{fort_end:.0f}s",
                xy=(N_end, fort_end), xytext=(-65, 12),
                textcoords="offset points", fontsize=8.5, color=C_FORT,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C_FORT, lw=1))

    # Big speedup badge
    ax.text(0.97, 0.10,
            f"{total_speedup:.0f}× faster\ntotal wall-clock\n(N = {N_end:,})",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold", color=C_GPU,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=C_GPU, linewidth=1.2, alpha=0.95))

    all_Ns_ticks = sorted({1, 8, 32, 128, 512, 1024, 2048})
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(all_Ns_ticks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Ensemble size N (parameter samples)", fontsize=9)
    ax.set_ylabel("Total wall-clock time (s)", fontsize=9)
    ax.set_title("C", fontweight="bold", fontsize=11, loc="left", pad=4)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.yaxis.grid(True, which="major", alpha=0.25, lw=0.6)
    ax.xaxis.grid(True, which="major", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.text(0.5, -0.2,
            "Fortran scales O(N); JAX GPU wall-clock\ngrows sub-linearly via vmap parallelism",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            style="italic", color="#374151")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jax-ensemble",
                        default=str(_FIGURES_DIR / "ensemble_benchmark.csv"))
    parser.add_argument("--fortran-ensemble",
                        default=str(_FIGURES_DIR / "ensemble_benchmark_fortran.csv"))
    parser.add_argument("--out",
                        default=str(_FIGURES_DIR / "paper_figure.pdf"))
    args = parser.parse_args()

    gpu, cpu = _load_jax_ensemble(Path(args.jax_ensemble))
    fort     = _load_fortran_ensemble(Path(args.fortran_ensemble))

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13.5, 4.2))
    fig.suptitle(
        "CLM-ML-JAX: differentiable multi-layer canopy physics on GPU\n"
        "CHATS7 (46-layer canopy column, May 2007)  ·  "
        "Ensemble: 5-parameter perturbations, Euler timestepping, A40 GPU",
        fontsize=9.5, y=1.01, color="#1F2937",
    )

    # Panels: A narrow (lax.scan), B medium (ms/sample), C wide (total wall-clock)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1.5, 1.5], wspace=0.38)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    _panel_a(ax_a)
    _panel_b(ax_b, gpu, cpu, fort)
    _panel_c(ax_c, gpu, cpu, fort)

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as both PDF (vector) and PNG (raster preview)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    print(f"Saved: {png_path}")

    # Print headline numbers
    N_max = max(gpu)
    gpu_total_s  = gpu[N_max]["total_ms"] / 1000.0
    fort_total_s = fort[N_max]["total_ms"] / 1000.0
    gpu_ms_samp  = gpu[N_max]["ms_per_sample"]
    fort_ms_mean = np.mean([fort[n]["ms_per_sample"] for n in fort])

    print(f"\n── Key numbers ────────────────────────────────────────────")
    print(f"  N={N_max}: Fortran sequential total  = {fort_total_s:.0f}s")
    print(f"  N={N_max}: JAX GPU vmap total        = {gpu_total_s:.1f}s")
    print(f"  Total wall-clock speedup             = {fort_total_s/gpu_total_s:.0f}×")
    print(f"  Per-sample: GPU {gpu_ms_samp:.1f}ms vs Fortran ~{fort_ms_mean:.0f}ms "
          f"= {fort_ms_mean/gpu_ms_samp:.1f}× per-sample speedup")
    print(f"  lax.scan: Euler {_LAXSCAN['Euler']['speedup']}×, "
          f"RK4 {_LAXSCAN['RK4']['speedup']:,}× over Python loop")


if __name__ == "__main__":
    main()
