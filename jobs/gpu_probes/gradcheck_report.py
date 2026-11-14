"""Classify the output of gradcheck0.py:  python gradcheck_report.py gradcheck_params.npz [-v]

Per input direction and output variable, over the steps of the day:
  nan      automatic differentiation (AD) gives a non-finite value
  none     AD and finite differences (FD) both zero: no dependence
  negl     negligible dependence (AD below 1e-8 of the output per 100 % of the input)
  lost     AD zero where FD is not
  ok/fair/bad   largest |AD - FD| over the day relative to the largest |FD|, at the FD
           step that agrees best: ok < 1e-3, fair < 5e-2, bad above
and the steps where AD differs from FD at every FD step size by more than 1 % (+ 1e-4 of
the day's largest value).
"""

import sys

import numpy as np

z = np.load(sys.argv[1])
verbose = "-v" in sys.argv
ad, fd, hs, names, vars_, y0, scale = (z[k] for k in ("ad", "fd", "hs", "names", "vars", "y0", "scale"))
nd, nv, nt = ad.shape
ymax = np.abs(y0).max(axis=1)  # (variable,)
print(f"{sys.argv[1]}: {nd} input directions, {nv} variables, {nt} steps; FD steps {list(hs)}")
rv, fw = z["rev_dirs"], z["fwd_dirs"]
print("reverse against forward mode, largest relative difference:",
      f"{np.max(np.abs(rv - fw) / (np.abs(fw).max(axis=1, keepdims=True) + 1e-300)):.1e}")  # fmt: skip

cls = np.empty((nd, nv), dtype=object)
err = np.full((nd, nv), np.nan)
besth = np.zeros((nd, nv))
badsteps = {}
for i in range(nd):
    for v in range(nv):
        a, f = ad[i, v], fd[:, i, v]
        # FD numerators at round-off level are zero
        noise = 1e-10 * ymax[v] / (2 * hs[:, None] * scale[i])
        fz = np.where(np.abs(f) <= noise, 0.0, f)
        if not np.isfinite(a).all():
            cls[i, v] = "nan"
            continue
        if not a.any() and not fz.any():
            cls[i, v] = "none"
            continue
        # negligible: a 100 % change of the input moves the output by less than 1e-8 of its
        # largest value by AD, and FD (largest step) sees less than 1e-6: below what FD resolves
        if np.abs(a).max() * scale[i] < 1e-8 * ymax[v] and np.abs(f[0]).max() * scale[i] < 1e-6 * ymax[v]:
            cls[i, v] = "negl"
            continue
        if not a.any():
            cls[i, v] = "lost"
            err[i, v] = np.abs(fz).max()
            continue
        e = np.abs(a - f).max(axis=1) / (np.abs(f).max(axis=1) + 1e-300)
        j = int(np.argmin(e))
        err[i, v], besth[i, v] = e[j], hs[j]
        cls[i, v] = "ok" if e[j] < 1e-3 else "fair" if e[j] < 5e-2 else "bad"
        dev = np.abs(a - f)
        tol = 0.01 * np.abs(f) + 1e-4 * np.abs(f).max(axis=1, keepdims=True) + noise
        bad = np.flatnonzero((dev > tol).all(axis=0))
        if bad.size:
            badsteps[i, v] = bad

kinds = ["ok", "fair", "bad", "lost", "none", "negl", "nan"]
print("\npairs (input direction, variable) by class:",
      {k: int((cls == k).sum()) for k in kinds})  # fmt: skip
short = [s[:10] for s in vars_]
print("\n" + " " * 44 + " ".join(f"{s:>10s}" for s in short))
for i in range(nd):
    if not verbose and all(c in ("ok", "none", "negl") for c in cls[i]):
        continue
    cells = []
    for v in range(nv):
        c = cls[i, v]
        cells.append(f"{c:>10s}" if c in ("none", "nan", "lost", "negl") else f"{c[:1]} {err[i, v]:8.1e}")
    print(f"{str(names[i])[-43:]:43s} " + " ".join(cells))
n_ok = sum(all(c in ("ok", "none", "negl") for c in cls[i]) for i in range(nd))
print(f"({n_ok} of {nd} directions are ok or without dependence in every variable"
      + ("" if verbose else "; not listed") + ")")
allnone = [str(names[i]) for i in range(nd) if all(c == "none" for c in cls[i])]
print(f"\nno dependence in any variable ({len(allnone)}):", ", ".join(allnone[:60]))
if badsteps:
    print("\nsteps where AD differs from FD at every step size (direction, variable: steps):")
    for (i, v), b in list(badsteps.items())[: 80 if verbose else 40]:
        print(f"  {str(names[i])[-40:]:40s} {short[v]:10s} {list(b[:12])}{' ...' if b.size > 12 else ''}")
    steps = np.concatenate(list(badsteps.values()))
    print("  count by step:", dict(zip(*np.unique(steps, return_counts=True))))
