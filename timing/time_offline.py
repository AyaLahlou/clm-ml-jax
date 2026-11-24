"""
Timing test: wall-clock time of a whole run through ``clm-ml run``.

Converts a namelist (``clm-ml convert``), then runs ``clm_ml.cli run`` in a fresh process and
reports the total wall time (imports, setup, compilation or cache load, the run, writing the
output files) and the seconds per simulated day. Steps run inside compiled chunks, so there is
no per-step time; ``timing/time_model.py`` times the compiled chunk alone.

Results are printed and appended to ``timing/results.jsonl``.

Usage::

    python timing/time_offline.py                      # CHATS7, 50 steps
    python timing/time_offline.py --namelist data/namelists/nl.CHATS7.1day
    python timing/time_offline.py --cold-cache         # empty JAX compilation cache
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

DEFAULT_NAMELIST = PROJECT_ROOT / "data" / "namelists" / "nl.CHATS7.50steps"
RESULTS_FILE = Path(__file__).resolve().parent / "results.jsonl"


def run(namelist: Path, cold_cache: bool) -> dict:
    from clm_ml.experiment import load_experiment
    from clm_ml.io.namelist import convert_namelist

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        convert_namelist(namelist, tmp)
        run_cfg = load_experiment(tmp)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC_DIR), env.get("PYTHONPATH")]))
        if cold_cache:
            env["JAX_COMPILATION_CACHE_DIR"] = str(tmp / "jax_cache")
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-m", "clm_ml.cli", "run", str(tmp)],
            env=env, capture_output=True, text=True,
        )  # fmt: skip
        total = time.perf_counter() - t0
        if proc.returncode != 0:
            sys.stdout.write(proc.stdout[-3000:] + proc.stderr[-3000:])
            raise SystemExit(f"model run failed with exit code {proc.returncode}")

    import jax

    days = run_cfg.n_steps * run_cfg.step_seconds / 86400.0
    return {
        "namelist": namelist.name,
        "n_steps": run_cfg.n_steps,
        "cold_cache": cold_cache,
        "total_s": round(total, 1),
        "s_per_simulated_day": round(total / days, 1),
        "device": str(jax.devices()[0]),
        "jax_version": jax.__version__,
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}",
        "git_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "date": time.strftime("%Y-%m-%d"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--namelist", type=Path, default=DEFAULT_NAMELIST)
    parser.add_argument("--cold-cache", action="store_true", help="use an empty JAX cache")
    parser.add_argument("--no-save", action="store_true", help="do not append to results.jsonl")
    args = parser.parse_args()

    result = run(args.namelist.resolve(), args.cold_cache)
    print(json.dumps(result, indent=2))
    if not args.no_save:
        with RESULTS_FILE.open("a") as fh:
            fh.write(json.dumps(result) + "\n")
        print(f"Appended to {RESULTS_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
