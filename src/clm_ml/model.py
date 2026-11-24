"""
The model in this Python process: initial state, compiled chunks, gradients.

``Experiment.execute()`` runs the model and writes files. :class:`Model` is for working in
a notebook or script instead: it exposes the pieces a calibration or sensitivity study needs::

    import clm_ml
    from clm_ml.model import Model

    model = Model.from_experiment(clm_ml.load_experiment("configs/chats7_1day"))
    state = model.initial_state()          # after the first (initializing) step
    params = model.default_params()        # CanopyParams table, one row per PFT of the domain
    first, n = model.chunks()[0]           # fixed-length chunks
    forcing = model.forcing(first, n)
    daily = StreamSpec("daily", "daily", "mean", ("latent_heat_flux",))

    def loss(p):
        new_state, records = model.run_chunk(state, forcing, first, p, streams=[daily])
        return records["daily"]["latent_heat_flux"].mean()

    jax.grad(loss)(params)                 # d(loss)/d(every parameter)

``run_chunk`` is differentiable with respect to the state, the parameters, the forcing and
the patch weights. All patches of the run (any number of gridcells and patches) advance
together, the time step batched over the patch axis: state and outputs have
a leading patch axis, and gridcell outputs (``gridcell:<variable>``) are the area-weighted
sums of the patches.

Each model holds its own options, parameters, constants and state (no module globals), so a
process can hold several models, of the same or different towers and soil layer structures,
e.g. to compare schemes, parameter sets or sites.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Sequence

import numpy as np

# Gradients over a compiled chunk crash (segmentation fault) in XLA's newer CPU runtime
# (the "thunk" runtime, default in JAX 0.6.2): reproduced with a 2-step chunk; the older
# runtime computes them, with or without jax.checkpoint on the step. XLA reads this flag
# when it starts, so it is set here, on import, unless already chosen. (Normal runs,
# Experiment.execute and the command line, keep the default runtime.) JAX 0.9.2 no longer supports
# the flag (the new runtime is the only one) and computes these gradients without crashing,
# so it is set only for JAX < 0.7.
_LEGACY_CPU_RUNTIME = "--xla_cpu_use_thunk_runtime=false"


def _jax_older_than(major: int, minor: int) -> bool:
    from importlib.metadata import version

    return tuple(int(x) for x in version("jax").split(".")[:2]) < (major, minor)


if _jax_older_than(0, 7) and "xla_cpu_use_thunk_runtime" not in os.environ.get("XLA_FLAGS", ""):
    import jax._src.xla_bridge as _xla_bridge

    if _xla_bridge.backends_are_initialized():
        warnings.warn(
            "clm_ml.model: JAX has already started, so XLA's older CPU runtime cannot be "
            f"selected; gradients over chunks may crash. Import clm_ml.model first, or set "
            f"XLA_FLAGS={_LEGACY_CPU_RUNTIME} before starting Python.",
            stacklevel=2,
        )
    else:
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "") + " " + _LEGACY_CPU_RUNTIME
        ).strip()


class Model:
    """A run's patches set up in this process (``clm_ml.driver.Simulation``; any number of
    gridcells and patches, the step batched over the patch axis). State, constants and
    outputs have a leading patch axis; the forcing comes per gridcell and per patch
    (:meth:`forcing`); the parameters are a table with one row per PFT of the domain
    (:attr:`pft_names`), gathered to the patches, so their gradients are per PFT (summed
    over each PFT's patches)."""

    def __init__(self, run, sim, state):
        self.run = run
        self._sim = sim
        self._state = state
        self._runners: dict = {}
        self.pft_names: tuple = sim.pft_names

    @classmethod
    def from_experiment(cls, run) -> "Model":
        """Set the model up in this process for an experiment
        (:class:`clm_ml.experiment.Experiment`): every patch, and its first step (run as a
        one-step chunk)."""
        import jax.numpy as jnp

        from clm_ml import driver

        ex = run.run_file.execution
        sim = driver.simulation(run).laid_out(
            int(ex.devices or 1), ex.patch_count, ex.gridcell_count
        )
        model = cls(run, sim, sim.state0)
        forcing = model.forcing(0, 1)
        state, _ = model._advance(jax_tree_asarray(sim.state0), forcing, 0, None, (), None, None,
                                  None, None)  # fmt: skip
        model._state = jax_tree_map(jnp.asarray, state)
        return model

    # -- pieces ------------------------------------------------------------------------

    @property
    def n_patches(self) -> int:
        return self._sim.n_patches

    def initial_state(self):
        """The :class:`clm_ml.state.State` after the first step (the start of chunk 1), with
        a leading patch axis."""
        return self._state

    def default_params(self):
        """The :class:`CanopyParams` table from the run's params.yaml: one row per PFT of
        the domain (:attr:`pft_names`), leading axis."""
        return self._sim.params_table

    def patch_params(self, params):
        """A PFT table (:meth:`default_params`) gathered to the patches (leading patch axis)."""
        return self._sim.patch_params(params)

    def weights(self):
        """The patches' area fractions of their gridcells, in the model's order (domain order,
        then padding patches with 0; device blocks with ``execution.devices``)."""
        import jax.numpy as jnp

        return jnp.asarray(self._sim.weight, dtype=jnp.float64)

    def constants(self):
        """The patches' :class:`clm_ml.step.PatchConstants` (leading patch axis)."""
        return self._sim.constants

    def chunks(self) -> list:
        """``(first_step, n_steps)`` of the run's chunks after the first step
        (``execution.chunk``; the last one may be shorter; None: calendar
        months)."""
        from clm_ml import driver

        sim, steps = self._sim, self.run.chunk_steps
        if steps is None:
            return driver.month_chunks(sim, 1)
        length = max(1, min(steps, sim.n_steps - 1))
        return [(k, min(length, sim.n_steps - k)) for k in range(1, sim.n_steps, length)]

    def forcing(self, first_step: int, n_steps: int):
        """The forcing of run steps ``first_step ... first_step + n_steps - 1``: ``(xg, xp)``,
        the meteorology per gridcell (``clm_ml.step.MetInputs``) and the land data per patch
        (``clm_ml.step.LandInputs``; grid runs: the monthly records and weights), stacked per
        step (differentiable input of :meth:`run_chunk`)."""
        import jax
        import jax.numpy as jnp

        return jax.tree.map(jnp.asarray, self._sim.read(first_step, n_steps))

    # -- running -----------------------------------------------------------------------

    def _runner(self, streams, gvars):
        from clm_ml import driver

        key = tuple((s.name, s.frequency, s.statistic, s.variables) for s in streams)
        if key not in self._runners:
            self._runners[key] = driver.make_runner(self._sim, streams, gvars)
        return self._runners[key]

    def _advance(self, state, forcing, first_step, params, streams, weights, program, alive,
                 tables, constants=None):  # fmt: skip
        import jax.numpy as jnp

        from clm_ml import driver
        from clm_ml import output_streams as S

        sim = self._sim
        streams = tuple(streams)
        params = self.default_params() if params is None else params
        weights = self.weights() if weights is None else weights
        alive = jnp.ones(self.n_patches, dtype=bool) if alive is None else jnp.asarray(alive)
        xg, xp = forcing
        n = int(xg.calday.shape[0])
        n_g = sim.n_gridcells
        names = sorted({v for s in streams for v in s.variables})
        gvars = [driver.gridcell_variables(s.variables, ("gridcell",)) for s in streams]
        if program is not None:  # its compiled program holds its options: they must be ours
            mine, theirs = self._sim, program._sim
            same = (
                mine.options,
                mine.acclimation,
                mine.acclimation_days,
                mine.max_layers,
                mine.n_patches,
                mine.layout,
                mine.checkpoint,
            ) == (
                theirs.options,
                theirs.acclimation,
                theirs.acclimation_days,
                theirs.max_layers,
                theirs.n_patches,
                theirs.layout,
                theirs.checkpoint,
            )
            if not same or (mine.land_fn is None) != (theirs.land_fn is None):
                raise ValueError(
                    "program: the other model's compiled program has other options (time step, "
                    "orbit of its start year, soil structure, acclimation, layer slots, patches "
                    "or layout); its program would run this model with them"
                )  # fmt: skip
        runner = (program or self)._runner(streams, gvars)
        shapes = {v: (self.n_patches,) + sh for v, sh in driver.output_shapes(sim, names).items()}
        starts = sim.step_starts(first_step, n)
        records = [
            S.chunk_records(starts, first_step, sim.step_seconds, s.frequency, sim.calendar)
            for s in streams
        ]

        def buffer_shapes(s, gv):
            out = {v: shapes[v] for v in s.variables}
            out.update({f"gridcell:{v}": (n_g,) + shapes[v][1:] for v in gv})
            return out

        buffers = tuple(
            S.init_buffers(s.statistic, len(rc.keys), buffer_shapes(s, gv), jnp)
            for s, rc, gv in zip(streams, records, gvars)
        )
        slots = tuple(jnp.asarray(rc.index) for rc in records)
        state, buffers, errors = runner(
            state, xg, xp, slots, buffers, self.patch_params(params), weights, alive,
            jnp.asarray(sim.runner_gidx),
            jax_tree_map(jnp.asarray, sim.constants if constants is None else constants),
            tables=sim.tables if tables is None else tables,
        )  # fmt: skip
        from clm_ml.processes.math import is_concrete

        if is_concrete(errors):  # not under jax.grad: Fortran's conservation checks
            driver.check_batch_balance(
                errors, sim, f"the chunk from run step {first_step}", np.asarray(alive)
            )
        out = {}
        for s, b in zip(streams, (driver.real_part(b, sim) for b in buffers)):
            if s.statistic == "mean":
                count = b["count"]
                out[s.name] = {
                    v: x / count.reshape((-1,) + (1,) * (x.ndim - 1)) for v, x in b["sum"].items()
                }
            else:
                out[s.name] = dict(b["value"])
        return state, out

    def run_chunk(
        self,
        state,
        forcing,
        first_step: int,
        params=None,
        streams: Sequence = (),
        weights=None,
        program: Optional["Model"] = None,
        alive=None,
        tables=None,
        constants=None,
    ):
        """Advance ``state`` over the chunk starting at run step ``first_step`` (``forcing``
        from :meth:`forcing`) as one compiled scan, accumulating ``streams``
        (:class:`clm_ml.output_streams.StreamSpec`) inside it. ``params``: a PFT table
        (:meth:`default_params`); ``weights``: the patches' area fractions (default
        :meth:`weights`). ``program``: run this model's inputs through another model's
        compiled program (same number of patches, layer slots and soil levels). ``alive``:
        the patches still in the run (default all): the others are left out of
        gridcell values and of parameter gradients. ``tables``: the psihat lookup tables
        (``clm_ml.processes.turbulence.RSLTables``; default: the package's). ``constants``: the
        patches' :class:`clm_ml.step.PatchConstants` (default :meth:`constants`).

        Returns ``(state, {stream: {variable: (record, patch, ...), "gridcell:<variable>":
        (record, gridcell, ...)}})`` with the chunk's records (partial records at the chunk's
        ends included); the gridcell values are area-weighted sums over each gridcell's
        patches, formed at every step (variables without a layer axis). Differentiable with
        respect to the state, the forcing, the parameters and the weights."""
        return self._advance(state, forcing, first_step, params, streams, weights, program,
                             alive, tables, constants)  # fmt: skip


def jax_tree_map(f, tree):
    import jax

    return jax.tree.map(f, tree)


def jax_tree_asarray(tree):
    import jax.numpy as jnp

    return jax_tree_map(jnp.asarray, tree)
