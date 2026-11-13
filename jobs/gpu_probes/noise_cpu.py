"""The regression case on the CPU with another psihat table search, against the committed
reference by the noise-twin criterion (tests/regression/test_end_to_end_regression.py)."""

import sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from tests.regression.record_run import REFERENCE_FILE, TWIN_FILES, recorded_fields, run_case
from clm_ml import driver

method = sys.argv[1]
driver.table_search = lambda choice: method
load = lambda p: {k: v for k, v in np.load(p).items() if not k.startswith("meta.")}
ref, plus, minus = load(REFERENCE_FILE), load(TWIN_FILES["plus"]), load(TWIN_FILES["minus"])
run, out = run_case(Path(tempfile.mkdtemp()))
new = recorded_fields(out["fields"].raw)
valid = lambda v: np.isfinite(v) & (np.abs(v) < 1e30)
worst, nbit, fails, rows = 0.0, 0, [], []
for f, r in ref.items():
    n = new[f]
    if not np.issubdtype(r.dtype, np.floating):
        assert np.array_equal(n, r), f
        nbit += 1
        continue
    ok = valid(r) & valid(n) & valid(plus[f]) & valid(minus[f])
    if not ok.any():
        nbit += 1
        continue
    diff = float(np.max(np.abs(n[ok] - r[ok])))
    spread = float(np.max(np.abs(plus[f][ok] - minus[f][ok])))
    scale = float(np.max(np.abs(r[ok])))
    tol = 2.0 * spread + 1e-12 * scale
    nbit += diff == 0.0
    if tol > 0:
        worst = max(worst, diff / tol)
    if diff > tol:
        fails.append((f, diff, tol))
    rows.append((diff / tol if tol else 0, diff / scale if scale else 0, diff, spread, scale, f))
print(
    f"{method}: {len(ref)} fields, {nbit} bit for bit, largest difference {worst:.3f} x the allowed, {len(fails)} fail"
)
for f in fails[:10]:
    print("  FAIL", f)
print("  worst by ratio to allowed: (ratio, relative diff, abs diff, twin spread, scale)")
for r in sorted(rows, reverse=True)[:6]:
    print("   %.2f  rel %.1e  abs %.1e  twins %.1e  scale %.1e  %s" % r)
print("  largest relative difference:", "%.1e %s" % max((r[1], r[5]) for r in rows))
