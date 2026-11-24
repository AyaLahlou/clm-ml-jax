"""The forcing of grid runs at setup and in the run: each patch's first-step values from its
own gridcell's gridded data, the chunk reader (meteorology per gridcell slot, monthly
vegetation and soil-water records with per-step weights) and the land data formed inside
the compiled chunk.

The gridded data are read and interpolated by :class:`clm_ml.grid_forcing.GridInputs`
(CRUJRA/CO2/LAI/soil water as CDEPS and CTSM handle them).
"""

from __future__ import annotations

import numpy as np

from clm_ml.forcing import chunk_run_steps


def grid_inputs(run):
    """The :class:`clm_ml.grid_forcing.GridInputs` of a grid run."""
    from clm_ml.grid_forcing import GridInputs

    return GridInputs(
        run.domain, run.path(run.domain_file.grid.surface_data), run.run_file.forcing.gridded,
        run.start, run.step_seconds, run.n_steps, resolve=run.path, calendar=run.calkindflag,
    )  # fmt: skip


def met_records(met: dict, steps) -> dict:
    """Rows ``steps`` of :meth:`GridInputs.met_at` output."""
    return {k: (tuple(x[steps] for x in v) if isinstance(v, tuple) else v[steps])
            for k, v in met.items()}  # fmt: skip


def first_values(inputs, n_steps: int) -> dict:
    """Per patch (P,) the first step's forcing (``uref``, ``tref``, ``qref``, ``pref``,
    ``pco2``) of its gridcell, its initial soil temperature ``tsoi`` (P, nlevgrnd; the soil
    temperature stream at the first step) and acclimation temperature ``tacclim`` (the
    first step's air temperature, for both acclimation options)."""
    gi = inputs.domain.patches.gridcell_index
    met = inputs.met_at(chunk_run_steps(n_steps, 0, 1))  # [halo, step 0, next]
    row = {k: np.asarray(met[k], dtype=float)[1, gi] for k in ("uref", "tref", "qref", "pref",
                                                              "pco2")}  # fmt: skip
    row["tsoi"] = np.asarray(inputs.soil_at_first_step("TSOI"), dtype=float)[gi]
    row["tacclim"] = row["tref"].copy()
    return row


class GridChunkReader:
    """A grid run's chunk inputs for a :class:`clm_ml.driver.Simulation`'s slots (padding
    gridcells read the last real gridcell's data, padding patches the patch they copy):
    ``xg`` (``clm_ml.step.MetInputs`` per gridcell slot, calendar days ``(n,)`` without a
    gridcell axis) and ``xp`` = ``(xw, land)``: per-step vegetation and soil-water weights and
    the chunk's monthly records (:func:`clm_ml.grid_forcing.land_record`). Host arrays."""

    def __init__(self, inputs):
        self.inputs = inputs

    def __call__(self, sim, first_step: int, n_steps: int):
        import jax

        from clm_ml.chunk_inputs import met_inputs

        inp = self.inputs
        src_g = np.minimum(np.arange(sim.n_gridcells), sim.n_real_gridcells - 1)
        src_p = np.asarray(sim.source)
        rs = chunk_run_steps(sim.n_steps, first_step, n_steps)
        met = inp.met_at(rs)
        met = {k: (tuple(x[:, src_g] for x in v) if isinstance(v, tuple) else v[:, src_g])
               for k, v in met.items()}  # fmt: skip
        n = n_steps
        times = [sim.time_of(k) for k in range(first_step, first_step + n)]
        xg = met_inputs(
            met_records(met, slice(0, n)), met_records(met, slice(1, n + 1)),
            met_records(met, slice(2, n + 2)), [t.calday for t in times],
            [t.calday_beg for t in times],
        )  # fmt: skip
        m1, m2, w1, w2 = inp.vegetation_weights(first_step, n)
        h, lb, f1, f2 = inp.soil_window(first_step, n)
        xw = (m1, m2, w1, w2, lb, f1, f2)
        land = (inp.lai[:, src_p], inp.sai[:, src_p], h[:, src_g])
        return jax.tree.map(np.asarray, xg), jax.tree.map(np.asarray, (xw, land))


def grid_simulation_parts(run) -> dict:
    """The grid-specific parts of a run's :class:`clm_ml.driver.Simulation`: the first-step
    values per patch, the chunk reader and the land-data function of the compiled chunk."""
    from clm_ml.grid_forcing import land_record

    inputs = grid_inputs(run)
    return {
        "first": first_values(inputs, run.n_steps),
        "reader": GridChunkReader(inputs),
        "land_fn": land_record,
    }
