"""
plot_ensemble_benchmark.py — GPU vs CPU ensemble benchmark figure.

Reads diags/figures/ensemble_benchmark.csv produced from log of job 7527971 (A40).
Produces diags/figures/ensemble_benchmark.png (2-panel).

Optionally overlays Fortran reference timing from a CSV produced by the
Fortran benchmark suite (clm-ml-fortran/benchmark/run_ensemble_benchmark.sh):
  --fortran path/to/ensemble_benchmark_fortran.csv

Panel A: ms/sample vs N (log-log) for GPU and CPU vmap
         + optional Fortran sequential and parallel lines
Panel B: GPU/CPU speedup vs N (ms/sample ratio)
         + optional Fortran/GPU speedup lines

Usage:
    python diags/plot_ensemble_benchmark.py
    python diags/plot_ensemble_benchmark.py \
        --fortran path/to/ensemble_benchmark_fortran.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

_HERE        = Path(__file__).resolve().parent
_FIGURES_DIR = _HERE / "figures"

C_GPU       = "#2563EB"   # blue
C_CPU       = "#DC2626"   # red
C_FORT_SEQ  = "#D97706"   # amber  (Fortran sequential)
C_FORT_PAR  = "#92400E"   # brown  (Fortran parallel)
ALPHA       = 0.85


def _load(path: Path):
    gpu, cpu = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            d = {
                "N":            int(row["N"]),
                "ms_per_sample": float(row["ms_per_sample"]) if row["ms_per_sample"] else np.nan,
                "vmap_ss_ms":   float(row["vmap_ss_ms"])   if row["vmap_ss_ms"]   else np.nan,
                "seq_ss_ms":    float(row["seq_ss_ms"])    if row["seq_ss_ms"]    else np.nan,
            }
            if row["backend"] == "gpu":
                gpu.append(d)
            else:
                cpu.append(d)
    return sorted(gpu, key=lambda r: r["N"]), sorted(cpu, key=lambda r: r["N"])


def _load_fortran_ensemble(path: Path | None) -> dict[str, list[dict]]:
    """
    Load ensemble_benchmark_fortran.csv produced by run_ensemble_benchmark.sh.

    Expected columns: backend,N,run_wall_s,ms_per_sample,notes
    backend = 'fortran_seq' or 'fortran_par'

    Returns {"seq": [...], "par": [...]} each sorted by N.
    Returns empty lists if path is None or file does not exist.
    """
    result: dict[str, list[dict]] = {"seq": [], "par": []}
    if path is None:
        return result
    if not path.exists():
        print(f"  [plot] INFO: Fortran ensemble CSV not found: {path}", file=sys.stderr)
        return result
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                N = int(row["N"])
                ms = float(row["ms_per_sample"])
            except (KeyError, ValueError) as e:
                print(f"  [plot] Skipping malformed Fortran row {row}: {e}", file=sys.stderr)
                continue
            backend = row.get("backend", "")
            if "seq" in backend:
                result["seq"].append({"N": N, "ms_per_sample": ms,
                                      "notes": row.get("notes", "")})
            elif "par" in backend:
                result["par"].append({"N": N, "ms_per_sample": ms,
                                      "notes": row.get("notes", "")})
    for k in result:
        result[k].sort(key=lambda r: r["N"])
    print(f"  [plot] Loaded Fortran ensemble data: "
          f"{len(result['seq'])} seq + {len(result['par'])} par N-values from {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Plot CLM-ML-JAX ensemble benchmark results")
    parser.add_argument("--csv",      default=str(_FIGURES_DIR / "ensemble_benchmark.csv"),
                        help="JAX ensemble benchmark CSV (default: diags/figures/ensemble_benchmark.csv)")
    parser.add_argument("--out",      default=str(_FIGURES_DIR / "ensemble_benchmark.png"),
                        help="Output PNG path")
    parser.add_argument(
        "--fortran",
        default=None,
        metavar="FORTRAN_CSV",
        help="Optional: path to ensemble_benchmark_fortran.csv from the Fortran benchmark "
             "suite (clm-ml-fortran/benchmark/run_ensemble_benchmark.sh). "
             "Overlays Fortran sequential and parallel lines on Panel A, "
             "and Fortran/GPU speedup on Panel B.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    gpu, cpu = _load(csv_path)
    fortran = _load_fortran_ensemble(Path(args.fortran) if args.fortran else None)

    gpu_Ns  = [r["N"] for r in gpu]
    gpu_mss = [r["ms_per_sample"] for r in gpu]
    cpu_Ns  = [r["N"] for r in cpu]
    cpu_mss = [r["ms_per_sample"] for r in cpu]

    # Speedup: CPU ms / GPU ms at matching N values
    cpu_by_N = {r["N"]: r["ms_per_sample"] for r in cpu}
    speedup_Ns   = [r["N"] for r in gpu if r["N"] in cpu_by_N]
    speedup_vals = [cpu_by_N[r["N"]] / r["ms_per_sample"]
                    for r in gpu if r["N"] in cpu_by_N]

    # Fortran/GPU speedup: fortran_seq ms / GPU ms
    fort_seq_by_N = {r["N"]: r["ms_per_sample"] for r in fortran["seq"]}
    fort_speedup_Ns   = [r["N"] for r in gpu if r["N"] in fort_seq_by_N]
    fort_speedup_vals = [fort_seq_by_N[r["N"]] / r["ms_per_sample"]
                         for r in gpu if r["N"] in fort_seq_by_N]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Panel A: ms/sample ─────────────────────────────────────────────────────
    ax_a.plot(gpu_Ns, gpu_mss, "o-", color=C_GPU, lw=2, ms=7, label="JAX GPU (Quadro RTX 8000)", zorder=3)
    ax_a.plot(cpu_Ns, cpu_mss, "s-", color=C_CPU, lw=2, ms=7, label="JAX CPU", zorder=3)
    for i, (n, m) in enumerate(zip(gpu_Ns, gpu_mss)):
        xoff = -8 if i == len(gpu_Ns) - 1 else 6
        yoff = 10 if i == len(gpu_Ns) - 1 else 3
        ax_a.annotate(f"{m:.1f}", (n, m), textcoords="offset points",
                      xytext=(xoff, yoff), fontsize=7.5, color=C_GPU)
    for i, (n, m) in enumerate(zip(cpu_Ns, cpu_mss)):
        xoff = -8 if i == 0 else 6
        yoff = -16 if i == 0 else -10
        ax_a.annotate(f"{m:.1f}", (n, m), textcoords="offset points",
                      xytext=(xoff, yoff), fontsize=7.5, color=C_CPU)

    # Optional Fortran lines
    if fortran["seq"]:
        fort_seq_Ns  = [r["N"] for r in fortran["seq"]]
        fort_seq_mss = [r["ms_per_sample"] for r in fortran["seq"]]
        ax_a.plot(fort_seq_Ns, fort_seq_mss, "D--", color=C_FORT_SEQ, lw=1.8, ms=6,
                  label="Fortran sequential (N runs)", zorder=3, alpha=0.9)
        for i, (n, m) in enumerate(zip(fort_seq_Ns, fort_seq_mss)):
            xoff = -8 if i == len(fort_seq_Ns) - 1 else 6
            yoff = 10 if i == len(fort_seq_Ns) - 1 else 5
            ax_a.annotate(f"{m:.1f}", (n, m), textcoords="offset points",
                          xytext=(xoff, yoff), fontsize=7, color=C_FORT_SEQ)

    ax_a.set_xscale("log", base=2)
    ax_a.set_yscale("log")
    xticks = sorted(set(gpu_Ns + cpu_Ns + [r["N"] for r in fortran["seq"]]))
    ax_a.set_xticks(xticks)
    ax_a.set_xticklabels([str(t) for t in xticks])
    ax_a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_a.set_xlabel("Ensemble size N", fontsize=10)
    ax_a.set_ylabel("ms / sample", fontsize=10)
    ax_a.set_title("(a) Amortized cost per sample", fontsize=13)
    ax_a.legend(fontsize=9)
    ax_a.grid(False)

    # ── Panel B: speedup ───────────────────────────────────────────────────────
    ax_b.plot(speedup_Ns, speedup_vals, "s-", color="#16A34A", lw=2, ms=7,
              label="JAX GPU speedup vs JAX CPU", zorder=3)
    # Annotate first and last points only
    for idx, (n, s) in [(0, (speedup_Ns[0], speedup_vals[0])),
                         (-1, (speedup_Ns[-1], speedup_vals[-1]))]:
        xoff = 6 if idx == 0 else -6
        yoff = 18 if idx == 0 else 6
        ha   = "left" if idx == 0 else "right"
        ax_b.annotate(f"{s:.2f}×", (n, s), textcoords="offset points",
                      xytext=(xoff, yoff), fontsize=8, color="#16A34A", ha=ha)

    # Optional: Fortran sequential / GPU speedup
    if fort_speedup_Ns:
        ax_b.plot(fort_speedup_Ns, fort_speedup_vals, "D--", color=C_FORT_SEQ,
                  lw=1.8, ms=6, label="JAX GPU speedup vs Fortran seq", zorder=3, alpha=0.9)
        for idx, (n, s) in [(0, (fort_speedup_Ns[0], fort_speedup_vals[0])),
                             (-1, (fort_speedup_Ns[-1], fort_speedup_vals[-1]))]:
            xoff = 6 if idx == 0 else -6
            ha   = "left" if idx == 0 else "right"
            ax_b.annotate(f"{s:.1f}×", (n, s), textcoords="offset points",
                          xytext=(xoff, -14), fontsize=7.5, color=C_FORT_SEQ, ha=ha)

    all_speedup_Ns = sorted(set(speedup_Ns + fort_speedup_Ns))
    ax_b.set_xscale("log", base=2)
    b_ticks = all_speedup_Ns if all_speedup_Ns else speedup_Ns
    ax_b.set_xticks(b_ticks)
    ax_b.set_xticklabels([str(t) for t in b_ticks])
    b_yticks = [1, 2, 3, 4, 5]
    ax_b.set_yticks(b_yticks)
    ax_b.set_yticklabels([str(t) for t in b_yticks])
    ax_b.set_xlabel("Ensemble size N", fontsize=10)
    ax_b.set_ylabel("Reference ms/sample  /  GPU ms/sample", fontsize=10)
    ax_b.set_title("(b) GPU speedup over reference (per sample)", fontsize=13)
    ax_b.legend(fontsize=9)
    ax_b.grid(False)

    plt.tight_layout()
    _paper_figs = Path(__file__).resolve().parent.parent / "Paper" / "jaxes_paper" / "figures"
    for dest_dir in [out_path.parent, _paper_figs]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            p = dest_dir / out_path.with_suffix(f".{ext}").name
            fig.savefig(p, dpi=300, bbox_inches="tight")
            print(f"Saved: {p}")

    # Summary
    print("\n── JAX GPU/CPU speedup summary ──────────────────────────────────")
    for n, s in zip(speedup_Ns, speedup_vals):
        print(f"  N={n:4d}  {s:.2f}×")
    if gpu_mss:
        print(f"\n  JAX GPU wall-clock N=2048: {gpu_mss[-1] * 2048 / 1000:.1f}s")
    if cpu_mss:
        print(f"  JAX CPU sequential N=2048 estimate: {cpu_mss[0] * 2048 / 1000:.1f}s")

    if fort_speedup_Ns:
        print("\n── JAX GPU vs Fortran sequential speedup ────────────────────────")
        for n, s in zip(fort_speedup_Ns, fort_speedup_vals):
            print(f"  N={n:4d}  {s:.2f}×")


if __name__ == "__main__":
    main()
