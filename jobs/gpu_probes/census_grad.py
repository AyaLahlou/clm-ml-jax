"""Static census of the forward and gradient programs of the CHATS7 chunk (CPU): the
primitive operations each runs per model step, weighted by the values they produce, by
source file; and the values the loops stack for the backward pass.

    JAX_PLATFORMS=cpu python census_grad.py TREE [overrides]
"""

import collections
import sys
import tempfile
from pathlib import Path

TREE = Path(sys.argv[1])
sys.path.insert(0, str(TREE / "src"))
import jax  # noqa: E402
import numpy as np  # noqa: E402

from clm_ml.experiment import load_experiment  # noqa: E402
from clm_ml.io.namelist import convert_namelist  # noqa: E402
from clm_ml.model import Model  # noqa: E402
from clm_ml.output_streams import StreamSpec  # noqa: E402

tmp = Path(tempfile.mkdtemp())
convert_namelist(TREE / "data" / "namelists" / "nl.CHATS7.1day", tmp)
model = Model.from_experiment(load_experiment(tmp, sys.argv[2:]))
N = 2
state, params, forcing = model.initial_state(), model.default_params(), model.forcing(1, N)
stream = StreamSpec("s", "step", "mean", ("latent_heat_flux",))


def loss(p):
    _, out = model.run_chunk(state, forcing, 1, p, streams=[stream])
    return out["s"]["latent_heat_flux"].mean()


def src(eqn):
    tb = eqn.source_info.traceback if eqn.source_info else None
    fr = [f for f in tb.frames if "/clm_ml/" in f.file_name] if tb else []
    return f"{Path(fr[0].file_name).stem}:{fr[0].function_name}" if fr else "?"


def size(vs):
    return sum(int(np.prod(v.aval.shape)) for v in vs if hasattr(v.aval, "shape"))


def subs(eqn):
    out = []
    for v in eqn.params.values():
        for x in v if isinstance(v, (tuple, list)) else (v,):
            j = getattr(x, "jaxpr", x)
            if hasattr(j, "eqns"):
                out.append(j)
    return out


def census(fn):
    j = jax.make_jaxpr(fn)(params).jaxpr
    work, ops, stacked, loops = (collections.Counter() for _ in range(4))

    def walk(j, mult, inside):
        for e in j.eqns:
            name = e.primitive.name
            if name == "scan":
                L, p = e.params["length"], e.params
                body = p["jaxpr"].jaxpr
                ys = body.outvars[p["num_carry"] :]
                key = inside or src(e)
                stacked[key] += mult * L * size(ys)
                loops[key] += mult * L
                walk(body, mult * L, inside or src(e))
            elif name == "while":
                loops[inside or src(e)] += mult
                walk(e.params["body_jaxpr"].jaxpr, mult, inside or src(e))
            elif name == "cond":
                walk(e.params["branches"][-1].jaxpr, mult, inside)
            elif subs(e):
                for s in subs(e):
                    walk(s, mult, inside)
            else:
                k = src(e)
                work[k] += mult * size(e.outvars)
                ops[k] += mult

    # the chunk's scan over steps and the sub-step scan are not "inside a loop" for the keys
    def top(j, mult, depth):
        for e in j.eqns:
            if e.primitive.name == "scan" and e.params["length"] in (N, 6) and depth < 4:
                top(e.params["jaxpr"].jaxpr, mult * e.params["length"], depth + 1)
            elif e.primitive.name not in ("scan", "while", "cond") and subs(e):
                for s in subs(e):
                    top(s, mult, depth)
            else:
                walk(type("J", (), {"eqns": [e]}), mult, None)

    top(j, 1, 0)
    return work, ops, stacked, loops


loss(params)  # fills the host caches (output shapes) outside a trace
f, g = census(loss), census(jax.grad(loss))
for title, i, unit in (("values produced per step (thousands)", 0, 1e3 * N),
                       ("operations per step", 1, N),
                       ("values stacked by loops per step (thousands)", 2, 1e3 * N),
                       ("loop iterations per step", 3, N)):  # fmt: skip
    print(f"\n== {title}: forward, gradient, ratio")
    keys = sorted(set(f[i]) | set(g[i]), key=lambda k: -g[i][k])
    tf, tg = sum(f[i].values()) / unit, sum(g[i].values()) / unit
    print(f"{'TOTAL':45s} {tf:10.1f} {tg:10.1f} {tg / max(tf, 1e-9):6.1f}")
    for k in keys[:22]:
        a, b = f[i][k] / unit, g[i][k] / unit
        print(f"{k:45s} {a:10.1f} {b:10.1f} {b / a if a else float('nan'):6.1f}")
