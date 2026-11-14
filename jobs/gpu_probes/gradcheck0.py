"""Gradient accuracy probe (docs/design/gradient_progress.md): the derivatives of a day of
CHATS7 outputs with respect to every parameter, state value or forcing field, by automatic
differentiation (forward mode, checked against reverse mode) and by central finite
differences at several step sizes. Saves everything to an npz file; `gradcheck_report.py`
classifies it.

    JAX_PLATFORMS=cpu python gradcheck0.py TREE --wrt params|state|forcing [--steps 47] [overrides]
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("tree")
ap.add_argument("overrides", nargs="*")
ap.add_argument("--wrt", default="params", choices=["params", "state", "forcing"])
ap.add_argument("--steps", type=int, default=47)
ap.add_argument("--first", type=int, default=1)
ap.add_argument("--h", default="1e-2,1e-3,1e-4,1e-5,1e-6", help="relative finite-difference steps")
ap.add_argument("--out", default="")
ap.add_argument("--pft", default="", help="another PFT for the patch (e.g. c4_grass)")
ap.add_argument("--rain", type=float, default=0.0, help="rain (kg/m2/s) in 3 of every 12 steps")
ap.add_argument("--snow", type=float, default=0.0, help="snow (kg/m2/s) in 2 of every 12 steps")
args = ap.parse_intermixed_args()
TREE = Path(args.tree)
sys.path.insert(0, str(TREE / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from clm_ml.experiment import load_experiment  # noqa: E402
from clm_ml.io.namelist import convert_namelist  # noqa: E402
from clm_ml.model import Model  # noqa: E402
from clm_ml.output_streams import StreamSpec  # noqa: E402

VARS = ("latent_heat_flux", "sensible_heat_flux", "net_radiation", "gpp", "ground_heat_flux",
        "canopy_top_air_temperature", "friction_velocity")  # fmt: skip
tmp = Path(tempfile.mkdtemp())
convert_namelist(TREE / "data" / "namelists" / "nl.CHATS7.1day", tmp)
if args.pft:
    text = (tmp / "domain.yaml").read_text()
    assert "pft: broadleaf_deciduous_temperate_tree" in text
    (tmp / "domain.yaml").write_text(
        text.replace("pft: broadleaf_deciduous_temperate_tree", f"pft: {args.pft}")
    )
model = Model.from_experiment(load_experiment(tmp, args.overrides))
first, n = args.first, args.steps
state, params, forcing = model.initial_state(), model.default_params(), model.forcing(first, n)
if args.rain or args.snow:  # invented showers: the forcing is an input like any other
    xg, xp = forcing
    k = (np.arange(n) % 12).reshape((-1,) + (1,) * (xg.rain.ndim - 1))
    xg = xg._replace(rain=jnp.where(k < 3, args.rain, 0.0) + 0 * xg.rain,
                     snow=jnp.where((k >= 6) & (k < 8), args.snow, 0.0) + 0 * xg.snow)  # fmt: skip
    forcing = (xg, xp)
stream = StreamSpec("s", "step", "mean", VARS)
base = {"params": params, "state": state, "forcing": forcing}[args.wrt]

# the float leaves of the input as one vector; names of its elements
paths, leaves = zip(*[(jax.tree_util.keystr(k), v) for k, v in jax.tree_util.tree_leaves_with_path(base)])
treedef = jax.tree.structure(base)
isf = [np.issubdtype(np.asarray(v).dtype, np.floating) for v in leaves]
shapes = [np.shape(v) for v in leaves]
sizes = [int(np.prod(s)) if f else 0 for s, f in zip(shapes, isf)]
x0 = np.concatenate([np.ravel(np.asarray(v)) for v, f in zip(leaves, isf) if f])
names = [f"{p}{list(i) if s else ''}" for p, s, f in zip(paths, shapes, isf) if f
         for i in (np.ndindex(*s) if s else [()])]  # fmt: skip
leaf_of = np.concatenate([np.full(sz, k) for k, sz in enumerate(sizes) if sz])


def unravel(x):
    out, at = [], 0
    for v, f, s, sz in zip(leaves, isf, shapes, sizes):
        if f:
            out.append(x[at:at + sz].reshape(s))
            at += sz
        else:
            out.append(v)
    return jax.tree.unflatten(treedef, out)


def F(x):  # (variable, step)
    s, p, f = state, params, forcing
    full = unravel(x)
    if args.wrt == "params":
        p = full
    elif args.wrt == "state":
        s = full
    else:
        f = full
    _, out = model.run_chunk(s, f, first, p, streams=[stream])
    return jnp.stack([out["s"][v][:, 0] for v in VARS])


y0 = np.asarray(F(jnp.asarray(x0)))  # also fills the host caches outside a trace
Fj = jax.jit(F)
print(args.wrt, "inputs", x0.size, "outputs", y0.shape, flush=True)

# directions: one per element (params, state); one per field (forcing: the field scaled at
# all steps together, or +1 everywhere where the field is zero)
if args.wrt == "forcing":
    dirs, dnames = [], []
    for k in sorted(set(leaf_of)):
        sel = leaf_of == k
        d = np.where(sel, np.where(x0 != 0, np.abs(x0), 0.0), 0.0)
        if not d.any():
            d = sel.astype(float)
        dirs.append(d)
        dnames.append([p for p, f in zip(paths, isf) if f][list(sorted(set(leaf_of))).index(k)])
    dirs = np.array(dirs)
    scale = np.ones(len(dirs))
else:
    ok = np.abs(x0) < 1e30  # not the unused layer slots (spval)
    idx = np.flatnonzero(ok)
    dirs = np.eye(x0.size)[idx]
    dnames = [names[i] for i in idx]
    scale = np.where(x0[idx] != 0, np.abs(x0[idx]), 1.0)

t = time.perf_counter()
jvps = jax.jit(jax.vmap(lambda d: jax.jvp(F, (jnp.asarray(x0),), (d,))[1]))
ad = np.asarray(jvps(jnp.asarray(dirs)))  # (direction, variable, step)
print("forward mode", ad.shape, round(time.perf_counter() - t, 1), "s", flush=True)

# reverse mode against forward mode: random cotangents and each variable's day mean
rng = np.random.default_rng(0)
cots = [rng.standard_normal(y0.shape) for _ in range(3)]
cots += [np.eye(len(VARS))[k][:, None] * np.ones(n) / n for k in range(len(VARS))]
vjp = jax.jit(lambda w: jax.grad(lambda x: jnp.sum(F(x) * w))(jnp.asarray(x0)))
t = time.perf_counter()
rev = np.array([np.asarray(vjp(jnp.asarray(w))) for w in cots])  # (cotangent, input)
print("reverse mode", rev.shape, round(time.perf_counter() - t, 1), "s", flush=True)
rev_dirs = rev @ dirs.T  # (cotangent, direction)
fwd_dirs = np.einsum("cvt,dvt->cd", np.array(cots), ad)

hs = [float(h) for h in args.h.split(",")]
fd = np.zeros((len(hs),) + ad.shape)
t = time.perf_counter()
for j, h in enumerate(hs):
    for i, d in enumerate(dirs):
        e = h * scale[i]
        up, dn = Fj(jnp.asarray(x0 + e * d)), Fj(jnp.asarray(x0 - e * d))
        fd[j, i] = (np.asarray(up) - np.asarray(dn)) / (2 * e)
    print("finite differences h =", h, round(time.perf_counter() - t, 1), "s", flush=True)

out = args.out or f"gradcheck_{args.wrt}.npz"
np.savez(out, ad=ad, fd=fd, hs=hs, names=np.array(dnames), vars=np.array(VARS), y0=y0, x0=x0,
         scale=scale, rev_dirs=rev_dirs, fwd_dirs=fwd_dirs, rev=rev)  # fmt: skip
print("saved", out)
