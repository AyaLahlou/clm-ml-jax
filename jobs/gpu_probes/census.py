"""Loop census of the compiled chunk (step 0): every loop in one model step, how often it
runs per step, its trip count, what it carries, and how many primitive operations its body
has. Static (from the jaxpr); run on the CPU with a small region.

    JAX_PLATFORMS=cpu python census.py TREE
"""

import collections, json, sys
from pathlib import Path

TREE = Path(sys.argv[1])
sys.path.insert(0, str(TREE / "src"))
import jax, jax.numpy as jnp, numpy as np
from jax._src import source_info_util
from clm_ml import driver as D, output_streams as S
from clm_ml.experiment import load_experiment
from clm_ml.processes import photosynthesis, turbulence

NAMES = ("latent_heat_flux", "gpp")
run = load_experiment(
    TREE / "configs/grid_1deg_2000",
    ["domain.grid.region={lat: [35, 37], lon: [-100, -98]}", "run.execution.devices=1"],
)
sim = D.simulation(run).laid_out(1, None, None)
schedule, length = D.chunk_schedule(sim, 0, 2)
first, n = schedule[0]
stream = S.StreamSpec("d", "daily", "mean", NAMES)
gvars = [D.gridcell_variables(NAMES, ("patch", "gridcell"))]
shapes = {v: (sim.n_patches,) + s for v, s in D.output_shapes(sim, NAMES).items()}
shapes.update({f"gridcell:{v}": (sim.n_gridcells,) + shapes[v][1:] for v in gvars[0]})
rc = S.chunk_records(sim.step_starts(first, n), first, sim.step_seconds, "daily", sim.calendar)
buffers = (
    S.init_buffers("mean", D.max_chunk_records("daily", length, sim.step_seconds), shapes, jnp),
)
slots = (jnp.asarray(D.pad_steps(rc.index, length)),)
xg, xp = D.read_chunk_inputs(sim, first, n, length)
a = (
    D.place_patches(sim, sim.state0),
    xg,
    xp,
    slots,
    buffers,
    sim.patch_params(),
    jnp.asarray(sim.weight),
    jnp.ones(sim.n_patches, dtype=bool),
    jnp.asarray(sim.runner_gidx),
    D.place_patches(sim, sim.constants),
    jnp.asarray([True] * n + [False] * (length - n)),
)
runner = D.make_runner(sim, [stream], gvars)
closed = jax.make_jaxpr(lambda *x: runner(*x, tables=sim.tables))(*a)
NP = sim.layout[2]  # C3 slots: sizes are reported per patch of the C3 group
print("layout", sim.layout, "patch slots", sim.n_patches)

FORI = {photosynthesis.GS_HALVINGS, turbulence.OBU_ITERATIONS}


def where(eqn):
    tb = eqn.source_info.traceback if eqn.source_info else None
    if tb is None:
        return "?"
    fr = [f for f in tb.frames if "/clm_ml/" in f.file_name]
    if not fr:
        return "?"
    # innermost clm_ml frame, and the first one that is not a generic math helper
    out = [f"{Path(f.file_name).name}:{f.line_num} {f.function_name}" for f in fr[:3]]
    return " < ".join(out)


def nbytes(vs):
    return sum(
        int(np.prod(v.aval.shape)) * v.aval.dtype.itemsize for v in vs if hasattr(v.aval, "shape")
    )


def subjaxprs(eqn):
    out = []
    for k, v in eqn.params.items():
        for x in (v if isinstance(v, (tuple, list)) else (v,)):
            j = getattr(x, "jaxpr", x)
            if hasattr(j, "eqns"):
                out.append((k, j))
    return out


def count_flat(j):  # primitive ops in a body, not descending into loops
    c = 0
    for e in j.eqns:
        subs = subjaxprs(e)
        if e.primitive.name in ("scan", "while"):
            continue
        c += sum(count_flat(s) for _, s in subs) if subs else 1
    return c


loops = []


def walk(j, mult, depth, path):
    for e in j.eqns:
        name = e.primitive.name
        if name == "scan":
            L, body = e.params["length"], e.params["jaxpr"].jaxpr
            nc, nk = e.params["num_consts"], e.params["num_carry"]
            carry = body.invars[nc : nc + nk]
            loops.append(
                dict(
                    kind="scan",
                    where=where(e),
                    trips=L,
                    calls=mult,
                    depth=depth,
                    path=path,
                    carry_n=nk,
                    carry_bytes=nbytes(carry),
                    body_ops=count_flat(body),
                    xs_n=len(body.invars) - nc - nk,
                    ys_n=len(body.outvars) - nk,
                )
            )
            walk(body, mult * L, depth + 1, path + [f"scan{L}"])
        elif name == "while":
            body = e.params["body_jaxpr"].jaxpr
            nk = e.params["body_nconsts"]
            carry = body.invars[nk:]
            loops.append(
                dict(
                    kind="while",
                    where=where(e),
                    trips=None,
                    calls=mult,
                    depth=depth,
                    path=path,
                    carry_n=len(carry),
                    carry_bytes=nbytes(carry),
                    body_ops=count_flat(body),
                    xs_n=0,
                    ys_n=0,
                )
            )
            i = len(loops) - 1
            walk(body, mult, depth + 1, path + [f"while#{i}"])
        else:
            for k, s in subjaxprs(e):
                walk(s, mult, depth, path)


walk(closed.jaxpr, 1, 0, [])
json.dump(loops, open("census.json", "w"), indent=1)
print(f"{len(loops)} loops; top-level ops outside loops: {count_flat(closed.jaxpr)}")
for l in loops:
    print(
        f"{'  ' * l['depth']}{l['kind']} trips={l['trips']} calls={l['calls']} carry={l['carry_n']} "
        f"({l['carry_bytes'] / 1024:.1f} kB) ops={l['body_ops']} xs={l['xs_n']} ys={l['ys_n']}  {l['where']}"
    )
