"""
Program size and compile time of a run's compiled chunk.

Sets the run up (``clm_ml.driver.simulation``), builds the compiled chunk of
``execution.chunk`` steps with one daily-mean stream, compiles it (ahead of time) and runs it
once on the first chunk. Records: setup time, compile time, the program's generated code
size (``compiled.memory_analysis().generated_code_size_in_bytes``), the chunk's run time and
patch-steps per second, and peak device memory. Appends one JSON line to
``timing/results_program.jsonl``.

    python timing/time_program.py configs/grid_1deg_2000 \
        domain.grid.region=global run.execution.devices=1 [--label ...]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", help="experiment folder")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    from clm_ml import driver as D
    from clm_ml import output_streams as S
    from clm_ml.experiment import load_experiment

    run = load_experiment(args.folder, args.overrides)
    ex = run.run_file.execution
    result = {"label": args.label, "run": str(args.folder), "overrides": args.overrides}
    t0 = time.perf_counter()
    sim = D.simulation(run).laid_out(int(ex.devices or 1), ex.patch_count, ex.gridcell_count)
    result["setup_s"] = round(time.perf_counter() - t0, 1)
    result.update(patches=sim.n_real, gridcells=sim.n_real_gridcells, layout=sim.layout)
    print("setup", result["setup_s"], "s", sim.n_real, "patches", flush=True)

    steps = run.chunk_steps or 480
    schedule, length = D.chunk_schedule(sim, 0, steps)
    first, n = schedule[0]
    names = ("latent_heat_flux", "gpp")
    stream = S.StreamSpec("d", "daily", "mean", names)
    gvars = [D.gridcell_variables(names, ("patch", "gridcell"))]
    shapes = {v: (sim.n_patches,) + s for v, s in D.output_shapes(sim, names).items()}
    shapes.update({f"gridcell:{v}": (sim.n_gridcells,) + shapes[v][1:] for v in gvars[0]})
    rc = S.chunk_records(sim.step_starts(first, n), first, sim.step_seconds, "daily", sim.calendar)
    n_slots = D.max_chunk_records("daily", length, sim.step_seconds)
    buffers = (S.init_buffers("mean", n_slots, shapes, jnp),)
    slots = (jnp.asarray(D.pad_steps(rc.index, length)),)
    t0 = time.perf_counter()
    xg, xp = D.read_chunk_inputs(sim, first, n, length)
    jax.block_until_ready((xg, xp))
    result["read_s"] = round(time.perf_counter() - t0, 1)
    state, constants = D.place_patches(sim, sim.state0), D.place_patches(sim, sim.constants)
    args_ = (state, xg, xp, slots, buffers, sim.patch_params(), jnp.asarray(sim.weight),
             jnp.ones(sim.n_patches, dtype=bool), jnp.asarray(sim.runner_gidx), constants,
             jnp.asarray([True] * n + [False] * (length - n)))  # fmt: skip
    runner = D.make_runner(sim, [stream], gvars)
    t0 = time.perf_counter()
    compiled = runner.lower(*args_, tables=sim.tables).compile()
    result["compile_s"] = round(time.perf_counter() - t0, 1)
    try:
        result["generated_code_gb"] = round(
            compiled.memory_analysis().generated_code_size_in_bytes / 1e9, 3
        )
    except Exception as exc:  # noqa: BLE001 - not every backend reports it
        result["generated_code_gb"] = f"unavailable: {exc}"
    print("compile", result["compile_s"], "s, code", result["generated_code_gb"], "GB", flush=True)
    t0 = time.perf_counter()
    out = compiled(*args_, tables=sim.tables)
    jax.block_until_ready(out)
    secs = time.perf_counter() - t0
    result.update(chunk_steps=length, chunk_s=round(secs, 1),
                  patch_steps_per_s=round(sim.n_real * n / secs, 1))  # fmt: skip
    stats = [d.memory_stats() or {} for d in jax.local_devices()]
    result["peak_device_gb"] = [round(s.get("peak_bytes_in_use", 0) / 1e9, 2) for s in stats]
    result.update(device=str(jax.devices()[0]), jax_version=jax.__version__,
                  machine=platform.machine(), date=time.strftime("%Y-%m-%d %H:%M"))  # fmt: skip
    print(json.dumps(result, indent=1))
    with open(ROOT / "timing" / "results_program.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
