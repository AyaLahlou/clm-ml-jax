"""Compare two saved global outputs (gpu0.py --save): per leaf, the largest absolute and
relative difference; relative to the leaf's largest magnitude. Usage: compare.py A.npz B.npz"""

import sys, numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
rows, nid = [], 0
for k in a.files:
    x, y = a[k], b[k]
    if x.dtype.kind != "f":
        nid += np.array_equal(x, y)
        continue
    ok = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < 1e30)
    if not np.array_equal(np.isfinite(x), np.isfinite(y)):
        print("NaN pattern differs:", k)
    if not ok.any():
        continue
    d = np.abs(x[ok] - y[ok])
    scale = np.abs(x[ok]).max()
    if d.max() == 0:
        nid += 1
        continue
    rows.append(
        (
            d.max() / scale if scale else 0,
            d.max(),
            float(np.percentile(d, 99.9)) / scale if scale else 0,
            scale,
            k,
            x.shape,
        )
    )
print(f"{len(a.files)} leaves, {nid} identical")
for r in sorted(rows, reverse=True)[:8]:
    print("  max rel %.1e  max abs %.1e  99.9%% rel %.1e  scale %.1e  %s %s" % r)
