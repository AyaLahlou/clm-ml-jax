"""Gradient accuracy over days in a box of real 1 degree patches (or the CHATS7 day)
(docs/design/gradient_progress.md, rounds 7-18): the forward run, a twin
(iota * (1 + 1e-14)), the forward-mode derivative with respect to iota and the response to
a finite change of iota (+-1 % unless FDH), for the output variables VARS at every step.
Saves ``y``, ``tw``, ``J``, ``fd`` (variable, patch, step) to an npz file and prints the
derivative of the run-mean latent heat against the finite change per patch.

    JAX_PLATFORMS=cpu python gradbox.py TREE chats7|cerrado|plains START OUT.npz [override ...]

Environment: NSTEP (steps, default 47); FDH (relative finite change, default 0.01); REF
(an npz file of a run with the same forward values: its twin and finite change are used
and only the forward run and the derivative are computed, for derivative rules that leave
the forward values alone, decision 124).
"""

import os
import sys
import tempfile
import time
from pathlib import Path

TREE = Path(sys.argv[1])
case, start, out, extra = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
sys.path.insert(0, str(TREE / "src"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from clm_ml.experiment import load_experiment  # noqa: E402
from clm_ml.model import Model  # noqa: E402
from clm_ml.output_streams import StreamSpec  # noqa: E402

VARS = ("latent_heat_flux", "sensible_heat_flux", "gpp", "canopy_top_air_temperature",
        "friction_velocity", "leaf_area_index", "state.intercepted_water")  # fmt: skip
BOX = {"cerrado": "{lat: [-12, -10], lon: [-52, -50]}", "plains": "{lat: [35, 37], lon: [-100, -98]}"}
if case == "chats7":
    from clm_ml.io.namelist import convert_namelist

    tmp = Path(tempfile.mkdtemp())
    convert_namelist(TREE / "data" / "namelists" / "nl.CHATS7.1day", tmp)
    run = load_experiment(tmp, list(extra))
else:
    run = load_experiment(
        TREE / "configs/grid_1deg_2000",
        [f"domain.grid.region={BOX[case]}", "run.execution.devices=1", f"run.time.start={start}",
         *extra],
    )  # fmt: skip
model = Model.from_experiment(run)
nstep = int(os.environ.get("NSTEP", 47))
state, params, forcing = model.initial_state(), model.default_params(), model.forcing(1, nstep)
stream = StreamSpec("s", "step", "mean", VARS)
iota0 = params.iota


def F(e):
    _, o = model.run_chunk(
        state, forcing, 1, params._replace(iota=iota0 * (1 + e)), streams=[stream]
    )
    return jnp.stack([(o["s"][v].sum(-1) if o["s"][v].ndim == 3 else o["s"][v]).T for v in VARS])


t0 = time.perf_counter()
if os.environ.get("REF"):
    ref = np.load(os.environ["REF"])
    y = np.asarray(F(0.0))
    yr, tw, fd = (ref[k][:, :, : y.shape[2]] for k in ("y", "tw", "fd"))
    print("forward identical to the reference:", np.array_equal(y, yr),
          "largest difference %.3g" % np.nanmax(np.abs(y - yr)), flush=True)  # fmt: skip
    _, J = jax.jit(lambda e: jax.jvp(F, (e,), (1.0,)))(0.0)
    J = np.asarray(J)
else:
    np.asarray(F(0.0))
    Fj = jax.jit(F)
    y = np.asarray(Fj(0.0))
    tw = np.asarray(Fj(1e-14))
    _, J = jax.jit(lambda e: jax.jvp(F, (e,), (1.0,)))(0.0)
    J = np.asarray(J)
    h = float(os.environ.get("FDH", 0.01))
    fd = (np.asarray(Fj(h)) - np.asarray(Fj(-h))) / (2 * h)
np.savez(out, y=y, tw=tw, J=J, fd=fd, vars=np.array(VARS))
am, fm = J[0].mean(1), fd[0].mean(1)
print(" run-mean LE, AD vs the finite change, per patch:\n  AD",
      np.array2string(am, precision=2, max_line_width=250), "\n  FD",
      np.array2string(fm, precision=2, max_line_width=250),
      "\n  worst relative error %.1f %%"
      % (100 * np.nanmax(np.abs(am - fm) / np.maximum(np.abs(fm), 0.5))))  # fmt: skip
print(extra, f"{time.perf_counter() - t0:.0f} s; NaN y {int(np.isnan(y).sum())} J {int(np.isnan(J).sum())}")
a = np.abs(J[0])
d = np.abs(tw[0] - y[0])
print(" patches with max|dLE/diota| > 1e4:", int((a.max(1) > 1e4).sum()), "of", a.shape[0],
      "; largest %.1e" % np.nanmax(a),
      "; twin LE diff max per patch: median %.1e max %.1e" % (np.median(d.max(1)), d.max()),
      "; mean LE %.2f H %.2f" % (y[0].mean(), y[1].mean()), flush=True)  # fmt: skip
