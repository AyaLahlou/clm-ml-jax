#!/usr/bin/env python3
"""
parse_results.py — Parse Fortran CLM-ML benchmark outputs and write summary CSVs.

Reads:
  results/single_timing_summary.txt     (from run_single_timing.sh)
  results/multisite_benchmark_fortran.csv   (from run_multisite_benchmark.sh)
  results/ensemble_benchmark_fortran.csv    (from run_ensemble_benchmark.sh)

Writes (overwriting in-place, adding computed columns):
  results/multisite_benchmark_fortran.csv   (enriched with ms/site annotation)
  results/ensemble_benchmark_fortran.csv    (enriched)

Also prints a human-readable summary table.

Usage:
  python3 parse_results.py
  python3 parse_results.py --results-dir /path/to/results
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _load_single_timing(path: Path) -> dict:
    """Parse results/single_timing_summary.txt → {key: value} dict."""
    d: dict[str, str] = {}
    if not path.exists():
        print(f"  [parse] WARNING: {path} not found — skipping single timing.", file=sys.stderr)
        return d
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                d[k.strip()] = v.strip()
    return d


def _load_csv(path: Path) -> list[dict]:
    """Load a CSV file, return list of row dicts."""
    if not path.exists():
        print(f"  [parse] WARNING: {path} not found.", file=sys.stderr)
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    """Write list of dicts to CSV."""
    if not rows:
        return
    fnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote: {path}")


def process_multisite(results_dir: Path) -> list[dict]:
    """
    Load multisite_benchmark_fortran.csv, compute derived columns, print summary.

    The script run_multisite_benchmark.sh already writes the CSV with columns:
      backend,N,seq_total_s,seq_ss_ms_per_site,parallel_wall_s,parallel_ms_per_site

    This function validates the data, adds a 'speedup_par_vs_seq' column, and
    prints a summary table.
    """
    csv_path = results_dir / "multisite_benchmark_fortran.csv"
    rows = _load_csv(csv_path)
    if not rows:
        print("  [parse] No multisite data to process.")
        return []

    enriched = []
    for r in rows:
        try:
            N = int(r["N"])
            seq_s = float(r["seq_total_s"])
            seq_ms = float(r["seq_ss_ms_per_site"])
            par_s = float(r["parallel_wall_s"])
            par_ms = float(r["parallel_ms_per_site"])
        except (KeyError, ValueError) as e:
            print(f"  [parse] Skipping malformed row {r}: {e}", file=sys.stderr)
            continue

        # Speedup: how much faster is parallel than sequential (ms/site)?
        speedup = seq_ms / par_ms if par_ms > 0 else float("nan")

        enriched.append({
            "backend": r.get("backend", "fortran"),
            "N": N,
            "seq_total_s": f"{seq_s:.4f}",
            "seq_ss_ms_per_site": f"{seq_ms:.4f}",
            "parallel_wall_s": f"{par_s:.4f}",
            "parallel_ms_per_site": f"{par_ms:.4f}",
            "speedup_par_vs_seq": f"{speedup:.3f}",
        })

    # Print table
    print("\n── Multisite Fortran benchmark (ms/site/step) ───────────────────")
    print(f"  {'N':>5}  {'seq ms/site':>14}  {'par ms/site':>14}  {'par speedup':>13}")
    for r in enriched:
        print(f"  {r['N']:>5}  {float(r['seq_ss_ms_per_site']):>14.2f}  "
              f"{float(r['parallel_ms_per_site']):>14.2f}  "
              f"{float(r['speedup_par_vs_seq']):>13.2f}×")

    _write_csv(csv_path, enriched,
               ["backend", "N", "seq_total_s", "seq_ss_ms_per_site",
                "parallel_wall_s", "parallel_ms_per_site", "speedup_par_vs_seq"])
    return enriched


def process_ensemble(results_dir: Path) -> list[dict]:
    """
    Load ensemble_benchmark_fortran.csv, compute derived columns, print summary.

    The script run_ensemble_benchmark.sh writes columns:
      backend,N,run_wall_s,ms_per_sample,notes

    backend is either 'fortran_seq' or 'fortran_par'.

    This function splits by backend and prints a summary.
    """
    csv_path = results_dir / "ensemble_benchmark_fortran.csv"
    rows = _load_csv(csv_path)
    if not rows:
        print("  [parse] No ensemble data to process.")
        return []

    seq_rows = [r for r in rows if r.get("backend", "").startswith("fortran_seq")]
    par_rows = [r for r in rows if r.get("backend", "").startswith("fortran_par")]

    # Build lookup for speedup calculation
    seq_by_N = {int(r["N"]): float(r["ms_per_sample"]) for r in seq_rows
                if r.get("ms_per_sample")}
    par_by_N = {int(r["N"]): float(r["ms_per_sample"]) for r in par_rows
                if r.get("ms_per_sample")}

    print("\n── Ensemble Fortran benchmark (ms/sample) ───────────────────────")
    print(f"  {'N':>6}  {'seq ms/sample':>16}  {'par ms/sample':>16}  {'par speedup':>13}")
    all_Ns = sorted(set(list(seq_by_N) + list(par_by_N)))
    for N in all_Ns:
        s = seq_by_N.get(N, float("nan"))
        p = par_by_N.get(N, float("nan"))
        sp = s / p if (p > 0 and not (p != p)) else float("nan")  # nan check
        s_str = f"{s:.2f}" if s == s else "n/a"
        p_str = f"{p:.2f}" if p == p else "n/a"
        sp_str = f"{sp:.2f}×" if sp == sp else "n/a"
        print(f"  {N:>6}  {s_str:>16}  {p_str:>16}  {sp_str:>13}")

    # Single-run baseline from single_timing_summary.txt if available
    single = _load_single_timing(results_dir / "single_timing_summary.txt")
    if single.get("ms_per_step_excl_warmup"):
        print(f"\n  Single-run baseline: {single['ms_per_step_excl_warmup']} ms/step "
              f"(from run_single_timing.sh)")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Parse Fortran CLM-ML benchmark results.")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Path to results directory (default: benchmark/results/ relative to this script)"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    results_dir = Path(args.results_dir) if args.results_dir else (script_dir / "results")

    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing results from: {results_dir}")

    # Single timing
    single_path = results_dir / "single_timing_summary.txt"
    single = _load_single_timing(single_path)
    if single:
        print("\n── Single-site baseline timing ──────────────────────────────────")
        for k, v in single.items():
            if not k.startswith("exe"):
                print(f"  {k}: {v}")

    # Multisite
    process_multisite(results_dir)

    # Ensemble
    process_ensemble(results_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
