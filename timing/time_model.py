"""
Timing test of the compiled model (clm_ml.model.Model) on CHATS7:

- forward: one compiled chunk of N steps, called four times; the first call includes the
  compilation, the steady cost is the fastest of the third and fourth;
- gradient: d(mean latent heat over the chunk)/d(parameters), likewise. The second call of
  an un-jitted ``jax.grad`` is not steady (several times slower than the calls after it;
  reported as ``gradient_second_s``): until 2026-11-14 this script reported it as the
  steady cost, so older gradient lines of the results file are too high.

Peak memory is the process's maximum resident set size. Appends one JSON line to
``timing/results_model.jsonl``. Used before and after Phase 5b (mask-aware layers).

    python timing/time_model.py [--steps 8] [--label "before 5b"] [--no-grad]
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


def _peak_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6  # bytes on macOS, kB on Linux


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--label", default="")
    parser.add_argument("--no-grad", action="store_true")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="override, the file named first (run.time.length=2 days, model.processes...)",
    )
    args = parser.parse_args()

    import jax

    from clm_ml.experiment import load_experiment
    from clm_ml.io.namelist import convert_namelist
    from clm_ml.model import Model
    from clm_ml.output_streams import StreamSpec

    tmp = Path(tempfile.mkdtemp())
    convert_namelist(ROOT / "data" / "namelists" / "nl.CHATS7.1day", tmp)
    run = load_experiment(tmp, args.set)
    t0 = time.perf_counter()
    model = Model.from_experiment(run)
    setup_s = time.perf_counter() - t0
    state, params = model.initial_state(), model.default_params()
    first, n = 1, args.steps
    forcing = model.forcing(first, n)
    stream = StreamSpec("s", "step", "mean", ("latent_heat_flux",))

    def forward(p):
        _, out = model.run_chunk(state, forcing, first, p, streams=[stream])
        return out["s"]["latent_heat_flux"].mean()

    result = {"label": args.label, "steps": n, "overrides": args.set, "setup_s": round(setup_s, 1)}
    for name, fn in (("forward", forward), ("gradient", jax.grad(forward))):
        if name == "gradient" and args.no_grad:
            continue
        times = []
        for _ in range(4):
            t = time.perf_counter()
            jax.block_until_ready(fn(params))
            times.append(time.perf_counter() - t)
        steady = min(times[2:])
        result[f"{name}_first_s"] = round(times[0], 2)
        result[f"{name}_second_s"] = round(times[1], 3)
        result[f"{name}_steady_s"] = round(steady, 3)
        result[f"{name}_steady_per_step_ms"] = round(1000 * steady / n, 1)
    result.update(
        peak_rss_gb=round(_peak_gb(), 2),
        device=str(jax.devices()[0]),
        jax_version=jax.__version__,
        machine=platform.machine(),
        date=time.strftime("%Y-%m-%d %H:%M"),
    )
    print(json.dumps(result, indent=1))
    with open(ROOT / "timing" / "results_model.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
