"""GPU optimization probe (docs/design/gpu_progress.md; plan steps 0a/0b/0d/0f/0g).

Times the compiled chunk of an experiment folder: sets the run up, compiles a chunk of
--steps steps, runs it --repeat times and reports each run. As timing/time_program.py, with a
chosen chunk length, repeats, the January run's eight output variables, and an optional nsys
capture of the last run (--profile, with `nsys profile --capture-range=cudaProfilerApi`).

    python gpu0.py TREE configs/grid_1deg_2000 domain.grid.region=global --steps 48 --repeat 3
"""

import argparse
import ctypes
import glob
import json
import os
import platform
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("tree", help="the code snapshot (a git worktree)")
parser.add_argument("folder", help="experiment folder, relative to the tree")
parser.add_argument("overrides", nargs="*")
parser.add_argument("--steps", type=int, default=48)
parser.add_argument("--repeat", type=int, default=3)
parser.add_argument("--label", default="")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--real", type=int, default=0, help="real steps; the rest of the chunk padded")
parser.add_argument("--searchsorted", default="", help="prototype: jnp.searchsorted method")
parser.add_argument(
    "--unroll", default="", help="prototype: loop groups to unroll, e.g. layers+obukhov"
)
parser.add_argument("--save", default="", help="save the output buffers and state (npz)")
parser.add_argument("--out", default="results_gpu0.jsonl")
args = parser.parse_args()

TREE = Path(args.tree)
sys.path.insert(0, str(TREE / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

if args.searchsorted:  # prototype (step 0): another method for the psihat table searches
    _ss = jnp.searchsorted

    def _patched(a, v, side="left", sorter=None, *, method="scan"):
        return _ss(a, v, side=side, sorter=sorter, method=args.searchsorted)

    jnp.searchsorted = _patched
    jax.numpy.searchsorted = _patched

UNROLL = {  # prototype (step 2a): loops written out, by the file and name of the loop body
    "layers": {"math:_forward", "math:_back", "math:_eliminate", "math:_substitute",
               "solar_radiation:_tbi_step", "solar_radiation:_upward",
               "solar_radiation:_downward", "solar_radiation:_sum", "soil:add"},
    "obukhov": {"turbulence:_secant_step"},
    "z0m": {"math:body"},
    "gs": {"photosynthesis:body"},
}  # fmt: skip
if args.unroll:
    _wanted = set().union(*(UNROLL[g] for g in args.unroll.split("+")))
    _scan, _fori = jax.lax.scan, jax.lax.fori_loop
    _seen = {}

    def _key(f):
        f = getattr(f, "func", f)
        return Path(f.__code__.co_filename).stem + ":" + f.__name__

    def _patched_scan(f, init, xs=None, length=None, reverse=False, unroll=1, **kw):
        hit = _key(f) in _wanted
        _seen[_key(f)] = hit
        return _scan(
            f, init, xs, length=length, reverse=reverse, unroll=True if hit else unroll, **kw
        )

    def _patched_fori(lower, upper, body_fun, init_val, **kw):
        hit = _key(body_fun) in _wanted
        _seen[_key(body_fun)] = hit
        if hit:
            kw["unroll"] = True
        return _fori(lower, upper, body_fun, init_val, **kw)

    jax.lax.scan, jax.lax.fori_loop = _patched_scan, _patched_fori

from clm_ml import driver as D  # noqa: E402
from clm_ml import output_streams as S  # noqa: E402
from clm_ml.experiment import load_experiment  # noqa: E402

NAMES = ("latent_heat_flux", "sensible_heat_flux", "net_radiation", "gpp", "ground_heat_flux",
         "canopy_top_air_temperature", "leaf_area_index", "soil_water_stress")  # fmt: skip

run = load_experiment(TREE / args.folder, args.overrides)
ex = run.run_file.execution
res = {
    "label": args.label,
    "overrides": args.overrides,
    "steps": args.steps,
    "searchsorted": args.searchsorted or "scan",
    "unroll": args.unroll,
}
t0 = time.perf_counter()
sim = D.simulation(run).laid_out(int(ex.devices or 1), ex.patch_count, ex.gridcell_count)
res["setup_s"] = round(time.perf_counter() - t0, 1)
res.update(patches=sim.n_real, gridcells=sim.n_real_gridcells, layout=sim.layout,
           checkpoint=bool(sim.checkpoint), slots=sim.n_patches)  # fmt: skip
print("setup", res["setup_s"], "s", sim.n_real, "patches", sim.layout, flush=True)

schedule, length = D.chunk_schedule(sim, 0, args.steps)
first, n = schedule[0]
if args.real:
    n = args.real
res["real_steps"] = n
stream = S.StreamSpec("d", "daily", "mean", NAMES)
gvars = [D.gridcell_variables(NAMES, ("patch", "gridcell"))]
shapes = {v: (sim.n_patches,) + s for v, s in D.output_shapes(sim, NAMES).items()}
shapes.update({f"gridcell:{v}": (sim.n_gridcells,) + shapes[v][1:] for v in gvars[0]})
rc = S.chunk_records(sim.step_starts(first, n), first, sim.step_seconds, "daily", sim.calendar)
n_slots = D.max_chunk_records("daily", length, sim.step_seconds)
buffers = (S.init_buffers("mean", n_slots, shapes, jnp),)
slots = (jnp.asarray(D.pad_steps(rc.index, length)),)
xg, xp = D.read_chunk_inputs(sim, first, n, length)
jax.block_until_ready((xg, xp))
state, constants = D.place_patches(sim, sim.state0), D.place_patches(sim, sim.constants)
a = (state, xg, xp, slots, buffers, sim.patch_params(), jnp.asarray(sim.weight),
     jnp.ones(sim.n_patches, dtype=bool), jnp.asarray(sim.runner_gidx), constants,
     jnp.asarray([True] * n + [False] * (length - n)))  # fmt: skip
runner = D.make_runner(sim, [stream], gvars)
t0 = time.perf_counter()
compiled = runner.lower(*a, tables=sim.tables).compile()
res["compile_s"] = round(time.perf_counter() - t0, 1)
try:
    res["code_mb"] = round(compiled.memory_analysis().generated_code_size_in_bytes / 1e6, 2)
except Exception as exc:  # noqa: BLE001
    res["code_mb"] = f"unavailable: {exc}"
print("compile", res["compile_s"], "s, code", res["code_mb"], "MB", flush=True)
if args.unroll:
    print("loops", {k: ("unrolled" if v else "loop") for k, v in sorted(_seen.items())}, flush=True)

cudart = None
if args.profile:
    import nvidia.cuda_runtime

    lib = glob.glob(os.path.join(list(nvidia.cuda_runtime.__path__)[0], "lib", "libcudart.so*"))
    cudart = ctypes.CDLL(lib[0])

runs = []
for i in range(args.repeat):
    last = i == args.repeat - 1
    if cudart and last:
        cudart.cudaProfilerStart()
    t0 = time.perf_counter()
    out = compiled(*a, tables=sim.tables)
    jax.block_until_ready(out)
    runs.append(round(time.perf_counter() - t0, 4))
    if cudart and last:
        cudart.cudaProfilerStop()
    print(f"run {i}: {runs[-1]} s, {1000 * runs[-1] / length:.1f} ms/step", flush=True)

steady = runs[1:] if len(runs) > 1 else runs
best = min(steady)
res.update(run_s=runs, ms_per_step=[round(1000 * r / length, 2) for r in runs],
           steady_ms_per_step=round(1000 * best / length, 2),
           patch_steps_per_s=round(sim.n_real * n / best, 1),
           us_per_patch_step=round(1e6 * best / (length * sim.n_real), 3))  # fmt: skip
errors = out[2]
if args.save:
    import numpy as np

    leaves = [np.asarray(x) for x in jax.tree.leaves((out[0], out[1]))]
    np.savez(args.save, *leaves)
res["finite"] = bool(jnp.all(jnp.isfinite(errors)))
stats = [d.memory_stats() or {} for d in jax.local_devices()]
res["peak_device_gb"] = [round(s.get("peak_bytes_in_use", 0) / 1e9, 2) for s in stats]
res.update(device=str(jax.devices()[0]), jax_version=jax.__version__, host=platform.node(),
           ncpus=os.environ.get("NCPUS", ""), xla_flags=os.environ.get("XLA_FLAGS", ""),
           no_substep_checkpoint=os.environ.get("CLM_ML_NO_CHECKPOINT", "0"),
           profiled=bool(cudart), date=time.strftime("%Y-%m-%d %H:%M"))  # fmt: skip
print(json.dumps(res, indent=1), flush=True)
with open(args.out, "a") as fh:
    fh.write(json.dumps(res) + "\n")
