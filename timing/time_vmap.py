"""
Timing test of ``vmap`` over patches (plan Phase 6, step 1b): the compiled CHATS7 chunk run on
the patch and N - 1 identical copies (``execution.patch_count`` padding), batched over the
patch axis as a domain runs (``clm_ml.model.Model``; Phase 7.4).

Measures, for one N per process (so peak device memory is per N):

- forward: one compiled chunk of ``--steps`` steps over N patches, called four times (the
  first call compiles; the steady cost is the fastest of the third and fourth); patch-steps
  per second;
- gradient (``--grad``): d(mean latent heat of the CHATS7 patch)/d(parameters), likewise.
  The second call of an un-jitted ``jax.grad`` is not steady (``gradient_second_s``): until
  2026-11-14 this script reported it as the steady cost, so older gradient lines of the
  results file are too high;
- peak device memory (``memory_stats``; GPU only) and peak resident set size;
- check: the CHATS7 patch equals its one-patch run (largest relative difference).

Appends one JSON line to ``timing/results_vmap.jsonl``.

    python timing/time_vmap.py --patches 1024 [--steps 8] [--grad] [--label ...]
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--patches", type=int, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--grad", action="store_true")
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--devices", type=int, default=1, help="split the patches over devices (shard_map)"
    )
    args = parser.parse_args()

    import jax
    import numpy as np

    from clm_ml.experiment import load_experiment
    from clm_ml.io.namelist import convert_namelist
    from clm_ml.model import Model
    from clm_ml.output_streams import StreamSpec

    tmp = Path(tempfile.mkdtemp())
    convert_namelist(ROOT / "data" / "namelists" / "nl.CHATS7.1day", tmp)
    first, n, npatch = 1, args.steps, args.patches
    # the CHATS7 patch and N - 1 padded copies of it (execution.patch_count; weight 0 for the
    # copies, which compute everything all the same): one batched program, as a domain runs
    # (with one padding gridcell for the copies, as execution.patch_count requires)
    extra = (
        [f"run.execution.patch_count={npatch}", "run.execution.gridcell_count=2"]
        if npatch > 1
        else []
    )
    if args.devices > 1:
        extra.append(f"run.execution.devices={args.devices}")
    model = Model.from_experiment(load_experiment(tmp, extra))
    single = Model.from_experiment(load_experiment(tmp))
    state, params, forcing = model.initial_state(), model.default_params(), model.forcing(first, n)
    stream = StreamSpec("s", "step", "mean", ("latent_heat_flux",))
    jax.block_until_ready((state, forcing))

    def forward(p):
        _, out = model.run_chunk(state, forcing, first, p, streams=[stream])
        return out["s"]["latent_heat_flux"]  # (step, real patches): patch 0

    def loss(p):
        return forward(p).mean()

    result = {"label": args.label, "patches": npatch, "steps": n, "devices": args.devices}
    fns = [("forward", forward)] + ([("gradient", jax.grad(loss))] if args.grad else [])
    for name, fn in fns:
        times = []
        for _ in range(4):
            t = time.perf_counter()
            out = jax.block_until_ready(fn(params))
            times.append(time.perf_counter() - t)
        steady = min(times[2:])
        result[f"{name}_first_s"] = round(times[0], 2)
        result[f"{name}_second_s"] = round(times[1], 4)
        result[f"{name}_steady_s"] = round(steady, 4)
        result[f"{name}_steady_per_step_ms"] = round(1000 * steady / n, 2)
        result[f"{name}_patch_steps_per_s"] = round(npatch * n / steady, 1)
        if name == "forward":
            _, ref = single.run_chunk(single.initial_state(), single.forcing(first, n), first,
                                      streams=[stream])  # fmt: skip
            ref = np.asarray(ref["s"]["latent_heat_flux"])
            lh = np.asarray(out)
            diff = np.abs(lh - ref) / np.maximum(np.abs(ref), 1e-30)
            result["max_rel_diff_vs_single"] = float(diff.max())
            result["finite"] = bool(np.isfinite(lh).all())
        else:
            leaves = jax.tree.leaves(out)
            result["gradient_finite"] = bool(all(np.isfinite(np.asarray(x)).all() for x in leaves))

    dev = jax.devices()[0]
    stats = dev.memory_stats() if hasattr(dev, "memory_stats") else None
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result.update(
        peak_device_gb=round(stats["peak_bytes_in_use"] / 1e9, 3) if stats else None,
        peak_rss_gb=round(peak_rss / 1e9 if sys.platform == "darwin" else peak_rss / 1e6, 2),
        device=str(dev),
        jax_version=jax.__version__,
        machine=platform.machine(),
        date=time.strftime("%Y-%m-%d %H:%M"),
    )
    print(json.dumps(result, indent=1))
    with open(ROOT / "timing" / "results_vmap.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
