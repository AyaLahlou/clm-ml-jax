"""CLM-ML-JAX: multilayer canopy model (CLM-ml v2) in JAX.

An experiment is a folder with run.yaml, domain.yaml, model.yaml and params.yaml
(docs/design/experiment_folders.md)::

    import clm_ml

    clm_ml.new_experiment("my_site", "chats7_1day")    # write a complete folder from a compset
    exp = clm_ml.load_experiment("my_site")              # the four files, validated
    results = exp.execute()                              # run; outputs per stream
    model = clm_ml.Model.from_experiment(exp)            # chunks and gradients (calibration)

The public API (stable from 1.0; decision 96) is these names: ``load_experiment``,
``new_experiment``, ``upgrade_experiment``, ``Experiment`` and ``Model``. Everything else
(``clm_ml.processes``, ``clm_ml.driver``, ``clm_ml.setup``, ...) is internal and may change.
"""

import os as _os

# XLA's CPU compiler splits a program into pieces compiled in parallel and links them with
# LLVM's JIT; for the large batched programs (e.g. C3 and C4 patch groups in one chunk) the
# linker's recursion overflowed a compile thread's stack: an intermittent segmentation fault
# during compilation (JAX 0.9.2 on Derecho, Phase 6; backtrace in llvm::orc
# RTDyldObjectLinkingLayer::emit). One piece avoids it (5 of 5 compilations clean; without it
# about half of the compilations of these programs crashed). XLA reads the flag when it
# starts, so it is set on import, unless the user's XLA_FLAGS already choose it. With JAX
# 0.10.2 the crash was not seen without the flag (5 of 5 cold compiles, decision 105); the flag
# stays as a precaution.
_SPLIT = "--xla_cpu_parallel_codegen_split_count=1"
if "xla_cpu_parallel_codegen_split_count" not in _os.environ.get("XLA_FLAGS", ""):
    _os.environ["XLA_FLAGS"] = (_os.environ.get("XLA_FLAGS", "") + " " + _SPLIT).strip()


_EXPORTS = {
    "load_experiment": "clm_ml.experiment",
    "new_experiment": "clm_ml.experiment",
    "upgrade_experiment": "clm_ml.experiment",
    "Experiment": "clm_ml.experiment",
    "Model": "clm_ml.model",
}


def __getattr__(name):
    """The public names, imported when first used (``import clm_ml`` stays light)."""
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'clm_ml' has no attribute {name!r}")
