"""
Padding test: the same CHATS7 patch (46 canopy-air layers) with
``max_layers`` = 46 (no padding), 50 (the default) and 100 gives the
same results. Unused layer slots must not change anything; the only allowed difference is
round-off from sums over arrays of different lengths (~1e-11 relative, the size of the
model's own round-off noise).

The gradient version (46 and 60 slots) checks that padded layers give finite gradients equal to the unpadded
ones (a masked ``jnp.where`` differentiates both branches, so arithmetic on empty layers can
produce NaN gradients even when the forward values are right). Slow: compiles gradients.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SIZES = (46, 50, 100)
# Gradients: 46 (no padding) and 60. Each size runs in its own process: gradient
# compilation memory grows with max_layers (the tridiagonal solves are unrolled over every
# slot), and 100 slots, or three sizes in one process, do not fit in 17 GB.
GRADIENT_SIZES = (46, 60)
RTOL = 1e-9  # relative to each variable's largest magnitude
# Gradients amplify round-off: they pass through the implicit-function derivatives of the
# iterative solvers, where a ~1e-11 change of a converged value moves the derivative by
# ~1e-9 (measured: up to 5e-9 between 46, 50 and 100 slots). 1e-7 is still far below the
# accuracy that matters (the finite-difference gradient checks pass at 5 %).
RTOL_GRADIENT = 1e-7

SCRIPT = """
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, {src!r})
import jax
import numpy as np
from clm_ml.io.namelist import convert_namelist
from clm_ml.experiment import load_experiment
from clm_ml.model import Model
from clm_ml.outputs import VARIABLES
from clm_ml.output_streams import StreamSpec

tmp = Path(tempfile.mkdtemp())
convert_namelist(Path({namelist!r}), tmp)
names = tuple(v.name for v in VARIABLES)
stream = StreamSpec("s", "step", "instantaneous", names)
first, n = 1, 6
out = {{}}
for size in {sizes!r}:
    model = Model.from_experiment(load_experiment(tmp,
                                    ["model.canopy_layers.max_layers=" + str(size)]))
    state, params = model.initial_state(), model.default_params()
    forcing = model.forcing(first, n)
    if {gradient!r}:
        def loss(p):
            _, rec = model.run_chunk(state, forcing, first, p, streams=[stream])
            return rec["s"]["latent_heat_flux"].mean() + rec["s"]["gpp"].mean()
        g = jax.grad(loss)(params)
        out[size] = {{k: np.asarray(v).tolist() for k, v in g._asdict().items()}}
    else:
        _, rec = model.run_chunk(state, forcing, first, params, streams=[stream])
        out[size] = {{k: np.asarray(v).tolist() for k, v in rec["s"].items()}}
print("RESULT" + json.dumps(out))
"""


def _run(gradient: bool, sizes=SIZES, separate: bool = False) -> dict:
    """Results per size; ``separate``: one process per size (memory freed in between)."""
    out = {}
    for group in [(size,) for size in sizes] if separate else [tuple(sizes)]:
        script = SCRIPT.format(
            src=str(ROOT / "src"),
            namelist=str(ROOT / "data" / "namelists" / "nl.CHATS7.1day"),
            sizes=group,
            gradient=gradient,
        )
        run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert run.returncode == 0, run.stderr[-3000:]
        line = next(ln for ln in run.stdout.splitlines() if ln.startswith("RESULT"))
        out.update({int(k): v for k, v in json.loads(line[len("RESULT") :]).items()})
    return out


def _common(a: np.ndarray, shape) -> np.ndarray:
    """``a`` cut to ``shape`` (layer axes cut to the smallest size)."""
    return a[tuple(slice(0, s) for s in shape)]


def _compare(results: dict, rtol: float = RTOL) -> list[str]:
    problems = []
    sizes = sorted(results)
    exact = results[sizes[0]]
    for size in sizes[1:]:
        for name, ref in exact.items():
            ref = np.asarray(ref, dtype=float)
            new = _common(np.asarray(results[size][name], dtype=float), ref.shape)
            if not np.array_equal(np.isfinite(new), np.isfinite(ref)):
                problems.append(f"max_layers {size}: {name}: NaN pattern differs")
                continue
            ok = np.isfinite(ref)
            scale = float(np.max(np.abs(ref[ok]), initial=0.0)) or 1.0
            diff = float(np.max(np.abs(new[ok] - ref[ok]), initial=0.0))
            if diff > rtol * scale:
                problems.append(
                    f"max_layers {size}: {name}: relative difference {diff / scale:.2e}"
                )
    return problems


@pytest.mark.regression
def test_padding_does_not_change_results():
    results = _run(gradient=False)
    problems = _compare(results)
    assert not problems, "\n".join(problems)


@pytest.mark.slow
def test_padding_gives_finite_equal_gradients():
    results = _run(gradient=True, sizes=GRADIENT_SIZES, separate=True)
    for size, grads in results.items():
        for name, g in grads.items():
            assert np.all(np.isfinite(np.asarray(g, dtype=float))), f"{size}: d/d{name} not finite"
    problems = _compare(results, RTOL_GRADIENT)
    assert not problems, "\n".join(problems)
