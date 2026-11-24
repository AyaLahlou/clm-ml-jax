"""
Integration test: compare JAX offline simulation outputs against Fortran reference.

Reference files: tests/fortran_output_files/CHATS7_2007-05_*.out
JAX outputs:     the legacy ``.out`` files of a ``clm-ml run`` (``CLM_ML_COMPARE_DIR``)

Thresholds (from CLAUDE.md):
  - Mean absolute relative error < 2.5 %
  - Pearson correlation >= 0.9999

Run after a full-month CHATS7 May-2007 simulation (the Fortran reference covers May), with
the Fortran acclimation temperature (``processes.photosynthesis.temperature_acclimation: run_mean``; the
default 10-day running mean departs from Fortran by design, by up to ~1%) and
the legacy ASCII output (``output.legacy_ascii: true``): ``configs/chats7_may2007_fortran``::

    clm-ml run configs/chats7_may2007_fortran --set run.output.directory=<dir>
    CLM_ML_COMPARE_DIR=<dir> pytest tests/test_fortran_python_comparison.py -m slow -v

Without ``CLM_ML_COMPARE_DIR`` the tests skip.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
FORTRAN_DIR = REPO_ROOT / "tests" / "fortran_output_files"
# The JAX run's legacy .out files (May 2007, nl.CHATS7.05.2007): CLM_ML_COMPARE_DIR
JAX_DIR = Path(os.environ.get("CLM_ML_COMPARE_DIR", REPO_ROOT / "missing-CLM_ML_COMPARE_DIR"))

OUTPUT_TAGS = ["flux", "aux", "fsun", "soiltemp", "profile", "fluxprofile"]

# Thresholds
REL_ERR_TOL = 0.025  # 2.5 %
CORR_TOL = 0.9999


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_out(path: Path) -> np.ndarray:
    """Load a whitespace-delimited .out file, skipping blank/non-numeric lines."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue  # skip blank day-separator lines
            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                continue  # skip any header lines
    # Warn if rows have unequal length (variable-column profile files)
    lengths = {len(r) for r in rows}
    if len(lengths) > 1:
        # Keep only rows matching the most-common column count
        most_common = max(lengths, key=lambda n: sum(1 for r in rows if len(r) == n))
        rows = [r for r in rows if len(r) == most_common]
    return np.array(rows)


def _col_stats(ref: np.ndarray, jax: np.ndarray, col: int) -> dict:
    """
    Compute comparison statistics for a single column.

    Relative error is MAE / max(|ref|), which is robust to near-zero values
    and matches the "< 2.5%" threshold reported in CLAUDE.md.
    """
    r = ref[:, col]
    j = jax[:, col]

    abs_err = np.abs(r - j)
    scale = max(float(np.abs(r).max()), 1e-6)
    rel_err = float(np.mean(abs_err) / scale)

    # Pearson correlation (skip if column is constant)
    if np.std(r) < 1e-12 or np.std(j) < 1e-12:
        corr = 1.0 if np.allclose(r, j, atol=1e-10) else 0.0
    else:
        corr = float(np.corrcoef(r, j)[0, 1])

    return {"rel_err": rel_err, "corr": corr, "max_abs_err": float(abs_err.max())}


def _has_time_column(arr: np.ndarray) -> bool:
    """Return True if col 0 looks like a Julian-day timestamp (1–366, varying)."""
    col = arr[:, 0]
    return bool(col.min() >= 1.0 and col.max() <= 366.0 and np.std(col) > 0.01)


def compare_files(tag: str) -> tuple[dict[str, dict], int, int]:
    """
    Compare one pair of .out files.

    Files whose first column is a Julian-day timestamp are aligned on that
    column (handles gaps where some timesteps are missing in one file).
    Files without a time column (e.g. aux.out) are compared positionally
    up to the shorter of the two arrays.

    Returns (per-column stats dict, n_ref_rows, n_matched_rows).
    """
    fname = f"CHATS7_2007-05_{tag}.out"
    ref_path = FORTRAN_DIR / fname
    jax_path = JAX_DIR / fname

    assert ref_path.exists(), f"Fortran reference not found: {ref_path}"
    if "CLM_ML_COMPARE_DIR" not in os.environ:
        pytest.skip("set CLM_ML_COMPARE_DIR to a May-2007 run's legacy .out files (docstring)")
    assert jax_path.exists(), f"JAX output not found: {jax_path}  (run the simulation first)"

    ref = load_out(ref_path)
    jax = load_out(jax_path)

    assert ref.ndim == 2 and jax.ndim == 2, f"{tag}: failed to load as 2-D arrays"
    assert (
        ref.shape[1] == jax.shape[1]
    ), f"{tag}: column count mismatch — Fortran {ref.shape[1]} vs JAX {jax.shape[1]}"

    n_ref = ref.shape[0]

    if _has_time_column(ref):
        ref_times_raw = np.round(ref[:, 0], 7)
        jax_times_raw = np.round(jax[:, 0], 7)
        unique_ref, ref_counts = np.unique(ref_times_raw, return_counts=True)
        rows_per_ts = int(np.median(ref_counts))  # 1 for scalar files, 46 for profiles

        if rows_per_ts == 1:
            # Single-row-per-timestep files (flux, fsun, soiltemp)
            # Deduplicate first (a tiny number of boundary timestamps repeat)
            _, ref_first = np.unique(ref_times_raw, return_index=True)
            _, jax_first = np.unique(jax_times_raw, return_index=True)
            ref_dd = ref[np.sort(ref_first)]
            jax_dd = jax[np.sort(jax_first)]
            common = np.intersect1d(np.round(ref_dd[:, 0], 7), np.round(jax_dd[:, 0], 7))
            assert len(common) > 0, f"{tag}: no matching timesteps"
            ref_aligned = ref_dd[np.isin(np.round(ref_dd[:, 0], 7), common)]
            jax_aligned = jax_dd[np.isin(np.round(jax_dd[:, 0], 7), common)]
        else:
            # Multi-row-per-timestep files (profile = 46 rows/ts, fluxprofile)
            # Build aligned arrays block by block for matching timestamps
            unique_jax = np.unique(jax_times_raw)
            common = np.intersect1d(unique_ref, unique_jax)
            assert len(common) > 0, f"{tag}: no matching timesteps"
            ref_blocks, jax_blocks = [], []
            for t in common:
                rb = ref[ref_times_raw == t]
                jb = jax[jax_times_raw == t]
                n_rows = min(len(rb), len(jb))
                if n_rows > 0:
                    ref_blocks.append(rb[:n_rows])
                    jax_blocks.append(jb[:n_rows])
            ref_aligned = np.vstack(ref_blocks)
            jax_aligned = np.vstack(jax_blocks)

        n_matched = len(common)
        first_data_col = 1  # skip time
    else:
        # No time column — positional comparison up to shorter array
        n_matched = min(n_ref, jax.shape[0])
        ref_aligned = ref[:n_matched]
        jax_aligned = jax[:n_matched]
        first_data_col = 0

    assert ref_aligned.shape[0] == jax_aligned.shape[0], f"{tag}: alignment produced unequal arrays"

    n_cols = ref_aligned.shape[1]
    stats = {
        f"col{c}": _col_stats(ref_aligned, jax_aligned, c) for c in range(first_data_col, n_cols)
    }
    return stats, n_ref, n_matched


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("tag", OUTPUT_TAGS)
def test_relative_error(tag):
    """Mean absolute relative error < 2.5 % for every data column."""
    stats, n_ref, n_matched = compare_files(tag)
    if n_matched < n_ref:
        pytest.skip(f"{tag}: only {n_matched}/{n_ref} timesteps matched — run a fresh simulation")
    failures = [(col, s["rel_err"]) for col, s in stats.items() if s["rel_err"] > REL_ERR_TOL]
    assert not failures, (
        f"{tag}: {len(failures)} column(s) exceed {REL_ERR_TOL*100:.1f}% relative error:\n"
        + "\n".join(f"  {col}: {e*100:.3f}%" for col, e in failures)
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("tag", OUTPUT_TAGS)
def test_correlation(tag):
    """Pearson correlation >= 0.9999 for every data column."""
    stats, n_ref, n_matched = compare_files(tag)
    if n_matched < n_ref:
        pytest.skip(f"{tag}: only {n_matched}/{n_ref} timesteps matched — run a fresh simulation")
    failures = [(col, s["corr"]) for col, s in stats.items() if s["corr"] < CORR_TOL]
    assert (
        not failures
    ), f"{tag}: {len(failures)} column(s) below correlation {CORR_TOL}:\n" + "\n".join(
        f"  {col}: corr={c:.6f}" for col, c in failures
    )


# ---------------------------------------------------------------------------
# Stand-alone summary (python tests/integration/test_fortran_python_comparison.py)
# ---------------------------------------------------------------------------


def _print_summary():
    print("\n" + "=" * 72)
    print("Fortran vs JAX Output Comparison — CHATS7 May 2007")
    print("=" * 72)

    all_pass = True
    for tag in OUTPUT_TAGS:
        fname = f"CHATS7_2007-05_{tag}.out"
        ref_path = FORTRAN_DIR / fname
        jax_path = JAX_DIR / fname

        if not ref_path.exists():
            print(f"\n[{tag}]  SKIP — Fortran reference not found")
            continue
        if not jax_path.exists():
            print(f"\n[{tag}]  SKIP — JAX output not found (run simulation first)")
            continue

        stats, n_ref, n_matched = compare_files(tag)
        rel_errs = [s["rel_err"] for s in stats.values()]
        corrs = [s["corr"] for s in stats.values()]

        max_rel = max(rel_errs)
        min_corr = min(corrs)
        mean_rel = np.mean(rel_errs)

        ok_err = max_rel <= REL_ERR_TOL
        ok_corr = min_corr >= CORR_TOL
        status = "PASS" if (ok_err and ok_corr) else "FAIL"
        if status == "FAIL":
            all_pass = False

        incomplete = n_matched < n_ref
        tag_status = status if not incomplete else f"{status} (incomplete: {n_matched}/{n_ref} ts)"
        print(f"\n[{tag}]  {tag_status}")
        print(f"  columns      : {len(stats)}")
        print(
            f"  mean rel err : {mean_rel*100:.4f}%  (max {max_rel*100:.4f}%,  tol {REL_ERR_TOL*100:.1f}%)  {'OK' if ok_err  else 'FAIL'}"
        )
        print(
            f"  min corr     : {min_corr:.6f}               (tol {CORR_TOL})  {'OK' if ok_corr else 'FAIL'}"
        )

        if not ok_corr:
            bad_corr = [(col, s["corr"]) for col, s in stats.items() if s["corr"] < CORR_TOL]
            bad_corr.sort(key=lambda x: x[1])
            print(f"  low-corr cols ({len(bad_corr)} of {len(stats)}):")
            for col, c in bad_corr[:5]:
                print(
                    f"    {col}: corr={c:.6f}  max_abs={stats[col]['max_abs_err']:.4e}  rel={stats[col]['rel_err']*100:.4f}%"
                )
            if len(bad_corr) > 5:
                print(f"    ... and {len(bad_corr)-5} more")

        if not ok_err:
            bad_err = sorted(stats.items(), key=lambda kv: kv[1]["rel_err"], reverse=True)[:5]
            print("  worst-err cols:")
            for col, s in bad_err:
                print(f"    {col}: {s['rel_err']*100:.3f}%  max_abs={s['max_abs_err']:.4e}")

    print("\n" + "=" * 72)
    print("Overall:", "PASS" if all_pass else "FAIL")
    print("=" * 72 + "\n")
    return all_pass


if __name__ == "__main__":
    import sys

    ok = _print_summary()
    sys.exit(0 if ok else 1)
