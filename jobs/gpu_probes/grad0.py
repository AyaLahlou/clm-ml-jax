"""Gradient speed probe, round 1 (branch gpu_grad): CHATS7 tower, one compiled chunk.

Times, in one process: the forward chunk, one forward-mode derivative (jvp, one direction),
and the reverse-mode gradient (jax.grad) of the chunk-mean latent heat flux, with respect to
the parameters, the initial state or the forcing.

    JAX_PLATFORMS=cpu python grad0.py TREE [--steps 8] [--wrt params] [--modes fwd,jvp,grad] [overrides]
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("tree")
ap.add_argument("overrides", nargs="*")
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--wrt", default="params", choices=["params", "state", "forcing"])
ap.add_argument("--modes", default="fwd,jvp,grad")
ap.add_argument("--stop", default="", help="ablation: module:function[+...] whose results get stop_gradient")
ap.add_argument("--repeat", type=int, default=3)
ap.add_argument("--label", default="")
ap.add_argument("--out", default="results_grad0.jsonl")
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

if args.stop:  # ablation: no derivative through these processes
    import importlib

    def _stopped(f):
        return lambda *a, **k: jax.tree.map(jax.lax.stop_gradient, f(*a, **k))

    for item in args.stop.split("+"):
        mod, fn = item.split(":")
        m = importlib.import_module("clm_ml.processes." + mod)
        setattr(m, fn, _stopped(getattr(m, fn)))

tmp = Path(tempfile.mkdtemp())
convert_namelist(TREE / "data" / "namelists" / "nl.CHATS7.1day", tmp)
model = Model.from_experiment(load_experiment(tmp, args.overrides))
first, n = 1, args.steps
state, params, forcing = model.initial_state(), model.default_params(), model.forcing(first, n)
stream = StreamSpec("s", "step", "mean", ("latent_heat_flux",))


def loss(x):
    s, p, f = state, params, forcing
    if args.wrt == "params":
        p = x
    elif args.wrt == "state":
        s = x
    else:
        f = x
    _, out = model.run_chunk(s, f, first, p, streams=[stream])
    return out["s"]["latent_heat_flux"].mean()


loss(params if args.wrt == "params" else state if args.wrt == "state" else forcing)  # host caches
x0 = {"params": params, "state": state, "forcing": forcing}[args.wrt]
isf = lambda a: jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating)  # noqa: E731
tangent = jax.tree.map(
    lambda a: jnp.ones_like(a) if isf(a) else np.zeros(np.shape(a), jax.dtypes.float0), x0
)
fns = {
    "fwd": loss,
    "jvp": lambda x: jax.jvp(loss, (x,), (tangent,))[1],
    "grad": jax.grad(loss, allow_int=True),
    "jgrad": jax.jit(jax.grad(loss, allow_int=True)),
    "jacfwd": jax.jit(jax.jacfwd(loss)),
    "jjvp": jax.jit(lambda x: jax.jvp(loss, (x,), (tangent,))[1]),
}
res = {
    "label": args.label, "stop": args.stop, "overrides": args.overrides, "steps": n, "wrt": args.wrt,
    "n_inputs": int(sum(np.size(a) for a in jax.tree.leaves(x0) if isf(a))),
    "device": str(jax.devices()[0]), "patches": model.n_patches,
}  # fmt: skip
for name in args.modes.split(","):
    times = []
    for _ in range(args.repeat + 1):
        t = time.perf_counter()
        out = jax.block_until_ready(fns[name](x0))
        times.append(time.perf_counter() - t)
    print(name, "times", [round(t, 3) for t in times], flush=True)
    res[f"{name}_first_s"] = round(times[0], 1)
    res[f"{name}_ms_per_step"] = round(1000 * min(times[1:]) / n, 2)
    if name in ("grad", "jgrad", "jacfwd"):
        g = [np.asarray(a) for a in jax.tree.leaves(out) if np.asarray(a).dtype.kind == "f"]
        res[f"{name}_finite"] = bool(all(np.isfinite(a).all() for a in g))
        res[f"{name}_nonzero"] = int(sum((a != 0).sum() for a in g))
    else:
        res[f"{name}_value"] = float(out)
    print(name, res[f"{name}_first_s"], "s first,", res[f"{name}_ms_per_step"], "ms/step", flush=True)
res["date"] = time.strftime("%Y-%m-%d %H:%M")
print(json.dumps(res), flush=True)
with open(args.out, "a") as fh:
    fh.write(json.dumps(res) + "\n")
