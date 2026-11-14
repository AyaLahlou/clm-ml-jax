"""
Batch job script for an experiment folder (decision 112): on NCAR's Derecho,
:func:`clm_ml.new_experiment` (``clm-ml new --machine derecho``) writes ``submit.pbs`` into
the folder, which runs ``clm-ml run`` on the folder; the user submits it with ``qsub``. On
any other machine no script is written.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from clm_ml.io.yaml_schema import ConfigError

#: machines a job script is written for
MACHINES = ("derecho",)
#: the job script's name in the experiment folder
JOB_SCRIPT = "submit.pbs"

# Derecho nodes: GPU nodes have 4 A100s, 64 cores and ~480 GB usable memory; CPU nodes 128
# cores and ~235 GB. jobs/program_gpu.pbs asks 200 GB for a one-GPU global chunk.
_GPUS_PER_NODE = 4


def check_machine(
    machine: Optional[str], device: str, devices: int, account: Optional[str]
) -> Optional[str]:
    """``machine`` lower-cased if a job script is written for it (None otherwise); raises
    ConfigError for a Derecho case without an account or with devices one node lacks."""
    if machine is None or machine.lower() not in MACHINES:
        return None
    if not account:
        raise ConfigError("machine derecho: give the project account the job is charged to")
    if device not in ("gpu", "cpu"):
        raise ConfigError(f"device {device!r}: gpu or cpu")
    if device == "gpu" and not 1 <= devices <= _GPUS_PER_NODE:
        raise ConfigError(f"devices {devices}: 1 to {_GPUS_PER_NODE} GPUs (one Derecho node)")
    if device == "cpu" and devices != 1:
        raise ConfigError(f"devices {devices}: a CPU run uses 1 device")
    return machine.lower()


def write_job_script(
    folder: Path | str, *, device: str, devices: int, account: str, walltime: str
) -> Path:
    """Write ``folder/submit.pbs``: a Derecho PBS job that runs the experiment in ``folder``
    with the Python environment running this function; its log goes to ``folder/run.log``."""
    folder = Path(folder).resolve()
    if device == "gpu":
        mem = min(max(200, 120 * devices), 480)
        select = f"select=1:ncpus={16 * devices}:ngpus={devices}:mem={mem}GB"
        platform, cache = "cuda", "jax_compile_cache_gpu"
        what = f"{devices} GPU{'s' if devices > 1 else ''}"
    else:
        select, platform, what = "select=1:ncpus=128:mem=235GB", "cpu", "a CPU node"
        cache = "jax_compile_cache"
    text = f"""#!/bin/bash
#PBS -N {folder.name[:15]}
#PBS -A {account}
#PBS -q main
#PBS -l {select}
#PBS -l walltime={walltime}
#PBS -j oe
#PBS -o {folder}/run.log
# Runs the experiment in this folder on Derecho ({what}). Written by `clm-ml new`; edit the
# walltime (or anything else) as needed and submit from the command line:
#
#   qsub {folder}/{JOB_SCRIPT}
#
# The log goes to run.log here, the output to the folder named in run.yaml (output).
set -eu
SCRATCH=${{SCRATCH:-/glade/derecho/scratch/$USER}}
export PATH={Path(sys.prefix) / "bin"}:$PATH
export JAX_PLATFORMS={platform}
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/{cache}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=$SCRATCH/tmp
mkdir -p $TMPDIR
cd {folder}
date
clm-ml run {folder}
date
"""
    path = folder / JOB_SCRIPT
    path.write_text(text)
    return path
