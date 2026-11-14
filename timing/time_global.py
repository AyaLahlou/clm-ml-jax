"""
Timing test of a synthetic global-size domain (plan Phase 6, step 6): many gridcells with
several patches each, set up by ``clm_ml.setup`` (Phase 7.4) and run as one compiled chunk
spread over the devices (``clm_ml.driver``).

The domain is a towers.yaml of gridcells that all read CHATS7's forcing files, with every
other input at random: latitude and longitude, sand, clay, organic matter, soil colour,
bedrock depth, PFT (three C3 and one C4), canopy height, root biomass, explicit layer counts
(5-15 within, 5-15 above) and patch weights. Records setup time, forcing read time, chunk
compile and steady times (the first step is part of the chunk), patch-steps per second and
peak device memory in ``timing/results_global.jsonl``.

    python timing/time_global.py [--patches 262144] [--per-gridcell 4] [--devices 4]
                                 [--steps 24] [--label ...]
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PFTS = (  # (name, C3, canopy height range m, root biomass g m-2)
    ("broadleaf_deciduous_temperate_tree", True, (8.0, 20.0), 500.0),
    ("needleleaf_evergreen_temperate_tree", True, (8.0, 20.0), 800.0),
    ("broadleaf_deciduous_boreal_tree", True, (5.0, 15.0), 500.0),
    ("c4_grass", False, (0.3, 1.0), 400.0),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--patches", type=int, default=262144)
    parser.add_argument("--per-gridcell", type=int, default=4)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np
    import yaml

    from clm_ml import driver as D
    from clm_ml import output_streams as S
    from clm_ml.experiment import load_experiment
    from clm_ml.io.namelist import convert_namelist

    n_p, k = args.patches, args.per_gridcell
    n_g = n_p // k
    rng = np.random.default_rng(0)
    result = {"label": args.label, "patches": n_p, "gridcells": n_g, "devices": args.devices}

    # the synthetic domain: CHATS7 forcing for every gridcell, every other input at random
    tmp = Path(tempfile.mkdtemp())
    convert_namelist(ROOT / "data" / "namelists" / "nl.CHATS7.1day", tmp)
    chats = yaml.safe_load(
        (ROOT / "src" / "clm_ml" / "compsets" / "towers" / "chats7.yaml").read_text()
    )["CHATS7"]
    towers = {}
    for g in range(n_g):
        t = copy.deepcopy(chats)
        sand = float(rng.uniform(5.0, 80.0))
        t.update(latitude=float(rng.uniform(-60.0, 75.0)), longitude=float(rng.uniform(-180, 180)))
        t["soil"] = {"texture": {"given_as": "percentages", "sand_percent": sand,
                                 "clay_percent": float(rng.uniform(5.0, 95.0 - sand))},
                     "organic_matter": float(rng.uniform(0.0, 100.0)),
                     "color_class": int(rng.integers(1, 21)),
                     "bedrock_depth": float(rng.uniform(0.5, 10.0))}  # fmt: skip
        kinds = [0] + list(rng.integers(0, len(PFTS), size=k - 1))
        w = rng.random(k) + 0.1
        t["patches"] = [
            {"pft": PFTS[j][0], "weight": float(wi / w.sum()),
             "canopy_height": float(rng.uniform(*PFTS[j][2])), "root_biomass": PFTS[j][3],
             "canopy_layers": {"within": int(rng.integers(5, 16)), "above": int(rng.integers(5, 16))}}
            for j, wi in zip(kinds, w)
        ]  # fmt: skip
        towers[f"G{g}"] = t
    d = yaml.safe_load((tmp / "domain.yaml").read_text())
    d["towers"] = towers
    (tmp / "domain.yaml").write_text(yaml.safe_dump(d))
    d = yaml.safe_load((tmp / "run.yaml").read_text())
    d["execution"]["devices"] = args.devices
    (tmp / "run.yaml").write_text(yaml.safe_dump(d))

    t0 = time.perf_counter()
    sim = D.simulation(load_experiment(tmp)).laid_out(args.devices)
    result["setup_s"] = round(time.perf_counter() - t0, 1)
    print("setup", result["setup_s"], "s", flush=True)

    names_out = ("latent_heat_flux", "gpp")
    first, n = 0, args.steps
    t0 = time.perf_counter()
    xg, xp = D.read_chunk_inputs(sim, first, n)
    jax.block_until_ready((xg, xp))
    result["read_s"] = round(time.perf_counter() - t0, 1)

    stream = S.StreamSpec("h", "hourly", "mean", names_out)
    gvars = [D.gridcell_variables(stream.variables, ("patch", "gridcell"))]
    runner = D.make_runner(sim, [stream], gvars)
    rc = S.chunk_records(sim.step_starts(first, n), first, sim.step_seconds, "hourly")
    shapes = {v: (sim.n_patches,) for v in names_out}
    shapes.update({f"gridcell:{v}": (sim.n_gridcells,) for v in gvars[0]})
    state, constants = D.place_patches(sim, sim.state0), D.place_patches(sim, sim.constants)
    params = sim.patch_params()
    weight = jnp.asarray(sim.weight)
    alive = jnp.ones(sim.n_patches, dtype=bool)
    gi = jnp.asarray(sim.runner_gidx)
    times = []
    for _ in range(2):
        buffers = (S.init_buffers("mean", len(rc.keys), shapes, jnp),)
        t0 = time.perf_counter()
        _, out, errors = runner(
            state, xg, xp, (jnp.asarray(rc.index),), buffers, params, weight, alive, gi,
            constants, tables=sim.tables,
        )  # fmt: skip
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    lh = np.asarray(D.real_part(out[0], sim)["sum"]["latent_heat_flux"])
    result.update(
        chunk_first_s=round(times[0], 1),
        chunk_steady_s=round(times[1], 2),
        steady_per_step_s=round(times[1] / n, 3),
        patch_steps_per_s=round(n_p * n / times[1], 1),
        finite_fraction=float(np.isfinite(lh).mean()),
        steps=n,
    )
    stats = [dev.memory_stats() for dev in jax.devices()[: args.devices]]
    peak = [s["peak_bytes_in_use"] / 1e9 for s in stats if s]
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    result.update(
        peak_device_gb=[round(x, 2) for x in peak] or None,
        peak_rss_gb=round(peak_rss, 1),
        device=str(jax.devices()[0]),
        jax_version=jax.__version__,
        machine=platform.machine(),
        date=time.strftime("%Y-%m-%d %H:%M"),
    )
    print(json.dumps(result, indent=1))
    with open(ROOT / "timing" / "results_global.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
