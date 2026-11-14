"""Gradient accuracy probe on real 1 degree patches (docs/design/gradient_progress.md): as
gradcheck0.py, for every patch of a box of an experiment folder with a grid. The derivative
of each patch's outputs over a day with respect to each parameter (the PFT rows of the
parameter scaled together: a patch only sees its own PFT), by forward-mode automatic
differentiation (AD) and by central finite differences (FD). Prints a line per patch.

    JAX_PLATFORMS=cpu python gradcheck_grid.py TREE "{lat: [35, 37], lon: [-100, -98]}" [--out f.npz]
"""

import argparse
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("tree")
ap.add_argument("region")
ap.add_argument("overrides", nargs="*")
ap.add_argument("--folder", default="configs/grid_1deg_2000")
ap.add_argument("--steps", type=int, default=47)
ap.add_argument("--first", type=int, default=1)
ap.add_argument("--h", default="1e-4,1e-5,1e-6")
ap.add_argument("--out", default="gradcheck_grid.npz")
args = ap.parse_intermixed_args()
TREE = Path(args.tree)
sys.path.insert(0, str(TREE / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from clm_ml.experiment import load_experiment  # noqa: E402
from clm_ml.model import Model  # noqa: E402
from clm_ml.output_streams import StreamSpec  # noqa: E402

VARS = ("latent_heat_flux", "sensible_heat_flux", "net_radiation", "gpp", "ground_heat_flux",
        "canopy_top_air_temperature", "friction_velocity")  # fmt: skip
run = load_experiment(
    TREE / args.folder,
    [f"domain.grid.region={args.region}", "run.execution.devices=1", *args.overrides],
)
model = Model.from_experiment(run)
sim = model._sim
first, n = args.first, args.steps
state, params, forcing = model.initial_state(), model.default_params(), model.forcing(first, n)
stream = StreamSpec("s", "step", "mean", VARS + ("leaf_area_index",))
zero = jax.tree.map(jnp.zeros_like, params)
names, tangents = [], []  # one direction per parameter (and band): every PFT row scaled by itself
for field in params._fields:
    v = np.asarray(getattr(params, field))
    for band in range(v.shape[1]) if v.ndim == 2 else [None]:
        d = np.zeros_like(v)
        if band is None:
            d[:] = np.where(v != 0, np.abs(v), 1.0)
        else:
            d[:, band] = np.where(v[:, band] != 0, np.abs(v[:, band]), 1.0)
        names.append(field + ("" if band is None else f"[{band}]"))
        tangents.append(zero._replace(**{field: jnp.asarray(d)}))


def F(p):  # (variable, patch, step)
    _, out = model.run_chunk(state, forcing, first, p, streams=[stream])
    return jnp.stack([out["s"][v].T for v in VARS]), out["s"]["leaf_area_index"].mean(axis=0)


def shifted(t, e):
    return jax.tree.map(lambda x, d: x + e * d, params, t)


y0, lai = (np.asarray(a) for a in F(params))
npatch = y0.shape[1]
pft = [sim.pft_names[int(r)] for r in np.asarray(sim.pft_row)[:npatch]]
print("patches", npatch, "outputs", y0.shape, "PFTs", sorted(set(pft)), flush=True)
print("forward finite:", bool(np.isfinite(y0).all()), flush=True)
stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *tangents)
t0 = time.perf_counter()
ad = np.asarray(jax.jit(jax.vmap(lambda t: jax.jvp(lambda p: F(p)[0], (params,), (t,))[1]))(stacked))
print("forward mode", ad.shape, round(time.perf_counter() - t0, 1), "s", flush=True)
Fj = jax.jit(lambda p: F(p)[0])
hs = [float(h) for h in args.h.split(",")]
fd = np.zeros((len(hs),) + ad.shape)
for j, h in enumerate(hs):
    for i, t in enumerate(tangents):
        fd[j, i] = (np.asarray(Fj(shifted(t, h))) - np.asarray(Fj(shifted(t, -h)))) / (2 * h)
    print("finite differences h =", h, round(time.perf_counter() - t0, 1), "s", flush=True)
np.savez(args.out, ad=ad, fd=fd, hs=hs, names=np.array(names), vars=np.array(VARS), y0=y0,
         lai=lai, pft=np.array(pft))  # fmt: skip

# classes as gradcheck_report.py, per (parameter, variable) of each patch
ymax = np.abs(y0).max(axis=2)  # (variable, patch)
print(f"\n{'patch':>5s} {'PFT':38s} {'LAI':>5s}  ok fair  bad lost none negl  nan  worst")
for k in range(npatch):
    count = dict.fromkeys(("ok", "fair", "bad", "lost", "none", "negl", "nan"), 0)
    worst = (0.0, "")
    for i in range(len(names)):
        for v in range(len(VARS)):
            a, f = ad[i, v, k], fd[:, i, v, k]
            noise = 1e-10 * ymax[v, k] / (2 * np.array(hs)[:, None])
            fz = np.where(np.abs(f) <= noise, 0.0, f)
            if not np.isfinite(a).all():
                c = "nan"
            elif not a.any() and not fz.any():
                c = "none"
            elif np.abs(a).max() < 1e-8 * ymax[v, k] and np.abs(f[0]).max() < 1e-6 * ymax[v, k]:
                c = "negl"
            elif not a.any():
                c = "lost"
            else:
                e = (np.abs(a - f).max(axis=1) / (np.abs(f).max(axis=1) + 1e-300)).min()
                c = "ok" if e < 1e-3 else "fair" if e < 5e-2 else "bad"
                if e > worst[0]:
                    worst = (e, f"{names[i]} -> {VARS[v]}")
            count[c] += 1
    print(f"{k:5d} {pft[k]:38s} {lai[k]:5.2f} " + " ".join(f"{c:4d}" for c in count.values())
          + f"  {worst[0]:.1e} {worst[1]}")  # fmt: skip
