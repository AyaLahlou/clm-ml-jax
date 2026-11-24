"""
Sensitivity of a tower's fluxes to its parameters, by automatic differentiation.

Runs one day at the CHATS walnut orchard (1 May 2007) and computes, in one backward pass,
how the day's mean latent heat flux and gross primary production (GPP) change with three
parameters of the walnut PFT: the maximum carboxylation rate (vcmax25), the stomatal
water-use efficiency (iota) and the leaf water potential at half stomatal closure
(psi50_gs). Each derivative is given as the change for a 10 % increase of the parameter and
is checked against a finite difference (two more runs with the parameter 0.1 % higher and
lower). Saves a bar chart of the sensitivities.

    python examples/sensitivity_tower.py                     # the CHATS7 day (~1-2 min on a CPU)
    python examples/sensitivity_tower.py --steps 12          # a short check (as CI runs it)
    python examples/sensitivity_tower.py --folder my_tower   # your own tower (any experiment folder)

The same pattern computes the derivative of any output with respect to any parameter, the
forcing or the initial state: see ``clm_ml.model.Model``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

import clm_ml
from clm_ml.model import Model
from clm_ml.output_streams import StreamSpec

jax.config.update("jax_enable_x64", True)

PARAMETERS = ("vcmax25", "iota", "psi50_gs")
OUTPUTS = ("latent_heat_flux", "gpp")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--folder", default="chats7_example", help="experiment folder")
    parser.add_argument("--compset", default="chats7_1day", help="compset for a new folder")
    parser.add_argument("--steps", type=int, help="model steps to run (default: the whole run)")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        clm_ml.new_experiment(folder, args.compset)
        print(f"wrote {folder}/ from the compset {args.compset}")
    model = Model.from_experiment(clm_ml.load_experiment(folder))

    # The model is set up with its first step done; run the rest as one chunk
    first = 1
    n_steps = args.steps or sum(n for _, n in model.chunks())
    state = model.initial_state()
    forcing = model.forcing(first, n_steps)
    params = model.default_params()  # one row per PFT in the domain
    stream = StreamSpec("steps", "step", "mean", OUTPUTS)  # every step's value

    def run_means(scale):
        """Run-mean of each output (patch 0) with each parameter multiplied by 1 + scale[k]."""
        p = params._replace(
            **{name: getattr(params, name) * (1.0 + scale[k]) for k, name in enumerate(PARAMETERS)}
        )
        _, records = model.run_chunk(state, forcing, first, p, streams=[stream])
        return jax.numpy.stack([records["steps"][v][:, 0].mean() for v in OUTPUTS])

    zero = jax.numpy.zeros(len(PARAMETERS))
    base = np.asarray(run_means(zero))
    # d(output)/d(scale): the change of each output per relative change of each parameter
    grad = np.asarray(jax.jacrev(run_means)(zero))  # (output, parameter)

    # Check: central finite differences, 0.1 % of each parameter
    h = 1e-3
    fd = np.zeros_like(grad)
    for k in range(len(PARAMETERS)):
        step = np.zeros(len(PARAMETERS))
        step[k] = h
        fd[:, k] = (np.asarray(run_means(step)) - np.asarray(run_means(-step))) / (2 * h)

    units = {"latent_heat_flux": "W m-2", "gpp": "umol m-2 s-1"}
    print(f"\n{n_steps} steps; means: " + ", ".join(
        f"{v} {base[i]:.2f} {units[v]}" for i, v in enumerate(OUTPUTS)))  # fmt: skip
    print("\nchange of the run mean for a 10 % increase of each parameter:")
    print(f"  {'parameter':10s} {'output':18s} {'gradient':>10s} {'finite diff.':>13s}")
    for k, name in enumerate(PARAMETERS):
        for i, var in enumerate(OUTPUTS):
            print(f"  {name:10s} {var:18s} {0.1 * grad[i, k]:10.4f} {0.1 * fd[i, k]:13.4f}")

    figure = plot(0.1 * grad, model_name=folder.name, path=folder / "sensitivity.png")
    print(f"\nfigure: {figure}")


def plot(change, model_name: str, path: Path) -> Path:
    """One panel per output: the change for a 10 % increase of each parameter."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"latent_heat_flux": "latent heat flux (W m-2)", "gpp": "GPP (umol m-2 s-1)"}
    fig, axes = plt.subplots(1, len(OUTPUTS), figsize=(9, 3.5))
    for ax, var, row in zip(axes, OUTPUTS, change):
        ax.barh(PARAMETERS, row, color="#4c72b0")
        ax.axvline(0.0, color="k", lw=0.8)
        ax.set_xlabel(f"change of {labels[var]}")
    fig.suptitle(f"{model_name}: change of the run mean for a 10 % parameter increase", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


if __name__ == "__main__":
    main()
