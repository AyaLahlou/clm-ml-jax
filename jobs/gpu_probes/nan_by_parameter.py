import sys
from pathlib import Path
TREE = Path("/glade/work/linnia/clm-ml-jax"); sys.path.insert(0, str(TREE / "src"))
import jax, jax.numpy as jnp, numpy as np
from clm_ml.experiment import load_experiment
from clm_ml.model import Model
from clm_ml.output_streams import StreamSpec
n = int(sys.argv[1])
run = load_experiment(TREE / "configs/grid_1deg_2000", ["domain.grid.region={lat: [-12, -10], lon: [-52, -50]}", "run.execution.devices=1"])
model = Model.from_experiment(run)
state, params, forcing = model.initial_state(), model.default_params(), model.forcing(1, n)
stream = StreamSpec("s", "step", "mean", ("latent_heat_flux",))
zero = jax.tree.map(jnp.zeros_like, params)
def F(p):
    _, out = model.run_chunk(state, forcing, 1, p, streams=[stream])
    return out["s"]["latent_heat_flux"]  # (step, patch)
F(params)
one = jax.jit(lambda t: jax.jvp(F, (params,), (t,))[1])
for field in params._fields:
    v = np.asarray(getattr(params, field))
    d = np.where(v != 0, np.abs(v), 1.0)
    a = np.asarray(one(zero._replace(**{field: jnp.asarray(d)})))
    bad = ~np.isfinite(a)
    print(f"{field:22s} NaN patches {np.flatnonzero(bad.any(axis=0)).tolist()} first step {int(np.argmax(bad.any(axis=1))) if bad.any() else None}  max|d| {np.nanmax(np.abs(a)):.3g}", flush=True)
