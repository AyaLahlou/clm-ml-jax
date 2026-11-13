"""Running the model (Phase 7.4): the patches set up in this process, advanced together in
compiled chunks that accumulate the output streams.

- :class:`Simulation`: a run's patches (constants, initial state, parameters, C3/C4 flag,
  gridcell map and weights; padded or laid out for devices), its clock, options and forcing
  reader. :func:`simulation` builds it from a :class:`clm_ml.experiment.Experiment`.
- :func:`make_runner`: the compiled chunk: a ``jax.lax.scan`` over steps whose step is
  ``clm_ml.step.step`` vmapped over the patches (C3 and C4 patches in contiguous groups,
  decision 22), with the output streams accumulated inside (plan §6.9), split over devices
  with ``jax.shard_map`` (Phase 6 step 5).
- :func:`run_streams`: the run loop: chunks of fixed length (decision 33) from the first
  step (an ordinary step since Phase 7.4: the setup gives the state before it), the next
  chunk's forcing read on a background thread, conservation and health checks, restarts.

Moved from ``offline_executable.main`` (the legacy driver), without the legacy containers:
the constants of every patch are an input of the compiled chunk (C23 b).
"""

from __future__ import annotations

import dataclasses
import datetime
import os
import time as _time
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from clm_ml import calendar
from clm_ml.checks import BALANCE_CHECKS, ModelError
from clm_ml.io.restart import FORMAT_VERSION as RESTART_VERSION
from clm_ml.io.restart import read_restart, write_restart  # noqa: F401  (netCDF restarts, 7.5)

jax.config.update("jax_enable_x64", True)


def _default_cache_dir() -> str:
    """``$SCRATCH/jax_compile_cache`` where the system defines a scratch directory (NCAR's
    machines do; the cache grows to gigabytes and home quotas are small, decision 106),
    ``~/.cache/jax_compile_cache`` otherwise."""
    scratch = os.environ.get("SCRATCH", "")
    if scratch and os.path.isdir(scratch):
        return os.path.join(scratch, "jax_compile_cache")
    return os.path.expanduser("~/.cache/jax_compile_cache")


def _setup_compilation_cache() -> None:
    """Every compiled program cached on disk (``JAX_COMPILATION_CACHE_DIR``, default
    :func:`_default_cache_dir`; 2 GB with ``filelock``): a repeated run in a new process is
    ~3x faster."""
    cache = os.environ.get("JAX_COMPILATION_CACHE_DIR", _default_cache_dir())
    os.makedirs(cache, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    try:
        import filelock  # noqa: F401

        jax.config.update("jax_compilation_cache_max_size", 2 * 1024**3)
    except ImportError:
        pass


_setup_compilation_cache()


# ---------------------------------------------------------------------------------------
# The simulation: patches, clock, options, forcing
# ---------------------------------------------------------------------------------------


@dataclasses.dataclass
class Simulation:
    """A run's patches and everything the run loop needs. Per-patch arrays (NumPy) have a
    leading patch axis: the domain's patches, then padding (copies of a real patch with
    weight 0) or the device blocks' slots (:meth:`blocked`)."""

    start: datetime.datetime  # start of the first step
    step_seconds: int
    n_steps: int
    calendar: str  # clm_ml.calendar name
    options: object  # clm_ml.step.StepOptions (static)
    acclimation: str  # running_mean | run_mean
    acclimation_days: float
    tables: object  # psihat lookup tables (RSLTables), an input of the compiled chunk
    max_layers: int
    params_table: object  # CanopyParams, one row per PFT of the domain
    pft_names: tuple  # the PFT of each row of params_table
    constants: object  # clm_ml.step.PatchConstants (P, ...)
    state0: object  # clm_ml.state.State (P, ...), before the first step
    pft_row: np.ndarray  # (P,) row of params_table
    c3: np.ndarray  # (P,) bool
    gridcell_index: np.ndarray  # (P,)
    weight: np.ndarray  # (P,)
    n_gridcells: int
    names: list  # per patch, for messages
    soil_depth: np.ndarray  # (real patches, nlevgrnd), output metadata
    reader: object = None  # (sim, first_step, n_steps) -> (xg, xp), host arrays
    land_fn: object = None  # grid runs: (xw, land, gidx) -> LandInputs inside the chunk
    n_real: Optional[int] = None
    n_real_gridcells: Optional[int] = None
    source: Optional[np.ndarray] = None  # the domain patch each slot holds (padding: copied)
    real_index: Optional[np.ndarray] = None  # the slot of each domain patch
    # jax.checkpoint on the step of compiled chunks, as in jax-gcm (model.yaml
    # numerics.checkpoint_steps): gradients recompute each step instead of storing it
    checkpoint: bool = True
    layout: Optional[tuple] = None  # (devices, gridcells per device, C3 slots, C4 slots)

    def __post_init__(self):
        n = len(self.pft_row)
        if self.n_real is None:
            self.n_real = n
        if self.n_real_gridcells is None:
            self.n_real_gridcells = self.n_gridcells
        if self.source is None:
            self.source = np.arange(n)
        if self.real_index is None:
            self.real_index = np.arange(self.n_real)

    # -- time ------------------------------------------------------------------------

    def time_of(self, run_step: int) -> calendar.StepTime:
        """``StepTime`` of a 0-based run step."""
        return calendar.step_time(self.start, self.step_seconds, run_step + 1, self.calendar)

    def step_starts(self, first_step: int, n_steps: int) -> list:
        """Start times of run steps ``first_step ... first_step + n_steps - 1``."""
        return [
            calendar.add_seconds(self.start, k * self.step_seconds, self.calendar)
            for k in range(first_step, first_step + n_steps)
        ]

    def step_end(self, run_step: int) -> datetime.datetime:
        return calendar.add_seconds(self.start, (run_step + 1) * self.step_seconds, self.calendar)

    # -- patches ---------------------------------------------------------------------

    @property
    def n_patches(self) -> int:
        return len(self.pft_row)

    @property
    def n_devices(self) -> int:
        return 1 if self.layout is None else self.layout[0]

    @property
    def groups(self) -> list:
        """``[(c3, patch indices), ...]``: the C3 group, then the C4 group (decision 22),
        each a contiguous range of slots, present only if it has patches."""
        if self.layout is not None:
            _, _, n3, n4 = self.layout
            return [(c3, np.arange(lo, lo + n)) for c3, lo, n in ((True, 0, n3), (False, n3, n4))
                    if n]  # fmt: skip
        out = []
        for c3 in (True, False):
            idx = np.flatnonzero(self.c3 == c3)
            if idx.size:
                out.append((c3, idx))
        if len(out) > 1:
            raise ValueError("C3 and C4 patches need the block layout: Simulation.blocked")
        return out

    @property
    def runner_gidx(self) -> np.ndarray:
        """The gridcell of each patch as the compiled chunk sees it: within its device's
        block of gridcells (the whole domain on one device)."""
        if self.layout is None:
            return self.gridcell_index
        _, gd, n3, n4 = self.layout
        device = np.arange(self.n_patches) // (n3 + n4)
        return self.gridcell_index - device * gd

    def patch_params(self, table=None):
        """A parameter table (default the run's) gathered to the patches."""
        table = self.params_table if table is None else table
        rows = jnp.asarray(self.pft_row)
        return jax.tree.map(lambda x: jnp.take(x, rows, axis=0), table)

    def _with_slots(self, src, **changes) -> "Simulation":
        take = lambda tree: jax.tree.map(lambda a: np.asarray(a)[src], tree)  # noqa: E731
        return dataclasses.replace(
            self,
            constants=take(self.constants),
            state0=take(self.state0),
            pft_row=self.pft_row[src],
            c3=self.c3[src],
            source=self.source[src],
            **changes,
        )

    def padded(self, patch_count: Optional[int], gridcell_count: Optional[int]) -> "Simulation":
        """This simulation padded to ``patch_count`` patches and ``gridcell_count``
        gridcells (domains of different sizes share one compiled program): padding
        gridcells read the last real gridcell's forcing; padding patches copy the last real
        patch with weight 0, on the last padding gridcell (or the last real one)."""
        n_p, n_g = patch_count or self.n_patches, gridcell_count or self.n_gridcells
        if n_p < self.n_patches or n_g < self.n_gridcells:
            raise ValueError(
                f"execution.patch_count/gridcell_count ({n_p}, {n_g}) is smaller than the "
                f"domain ({self.n_patches} patches, {self.n_gridcells} gridcells)"
            )
        k = n_p - self.n_patches
        src = np.concatenate([np.arange(self.n_patches), np.full(k, self.n_patches - 1)])
        return self._with_slots(
            src,
            gridcell_index=np.concatenate([self.gridcell_index, np.full(k, n_g - 1)]),
            weight=np.concatenate([self.weight, np.zeros(k)]),
            n_gridcells=n_g,
            names=self.names + [f"padding {i}" for i in range(k)],
            n_real=self.n_patches,
            n_real_gridcells=self.n_gridcells,
        )

    def blocked(self, n_devices: int) -> "Simulation":
        """This (domain-order, unpadded) simulation laid out for ``n_devices`` devices: every
        device gets the same block, ``Gd`` consecutive gridcells (padded) and their patches in
        ``N3`` C3 slots followed by ``N4`` C4 slots, so every device runs the same program and
        the forcing gather and gridcell sums stay on the device. Empty slots hold weight-0
        copies of a patch of the same pathway, on that patch's own gridcell when it is in the
        block (so the copy sees its own forcing), else on the block's last gridcell; padding
        gridcells read the last real gridcell's forcing. Padding slots are not checked
        (:func:`real_mask`). Outputs are put back in domain order (``real_index``)."""
        if self.n_real != self.n_patches or self.n_real_gridcells != self.n_gridcells:
            raise ValueError("blocked(): start from the domain's own patches (no padding)")
        gidx, c3 = self.gridcell_index, self.c3
        n_g, d_n = self.n_gridcells, n_devices
        gd = -(-n_g // d_n)
        blocks = []
        for d in range(d_n):
            on = [p for p in range(self.n_patches) if d * gd <= gidx[p] < (d + 1) * gd]
            blocks.append(([p for p in on if c3[p]], [p for p in on if not c3[p]]))
        n3 = max(len(b[0]) for b in blocks)
        n4 = max(len(b[1]) for b in blocks)
        any3 = int(np.flatnonzero(c3)[0]) if c3.any() else None
        any4 = int(np.flatnonzero(~c3)[0]) if (~c3).any() else None
        src, real, gi, names = [], [], [], []
        real_index = np.zeros(self.n_patches, dtype=int)
        for d, (p3, p4) in enumerate(blocks):
            for plist, n, fallback in ((p3, n3, any3), (p4, n4, any4)):
                for k in range(n):
                    if k < len(plist):
                        real_index[plist[k]] = len(src)
                        src.append(plist[k])
                        real.append(True)
                        gi.append(int(gidx[plist[k]]))
                    else:
                        src.append(plist[-1] if plist else fallback)
                        real.append(False)
                        # the copied patch's gridcell when it is in this block (open item O29)
                        gi.append(int(gidx[plist[-1]]) if plist else d * gd + gd - 1)
                    names.append(self.names[src[-1]] if real[-1] else f"padding {len(src) - 1}")
        src, real = np.array(src), np.array(real)
        return self._with_slots(
            src,
            gridcell_index=np.array(gi),
            weight=np.where(real, self.weight[src], 0.0),
            n_gridcells=d_n * gd,
            names=names,
            n_real=self.n_patches,
            n_real_gridcells=n_g,
            real_index=real_index,
            layout=(d_n, gd, n3, n4),
        )

    def laid_out(self, devices: int = 1, patch_count=None, gridcell_count=None):
        """Laid out for ``devices`` devices and C3/C4 groups (block layout), or padded to
        ``patch_count``/``gridcell_count``."""
        if devices > 1 or len(set(self.c3.tolist())) > 1:
            if patch_count or gridcell_count:
                raise ValueError(
                    "execution.patch_count/gridcell_count: not supported with several devices "
                    "or with C3 and C4 patches in one run (the blocks are padded automatically)"
                )
            return self.blocked(devices)
        return self.padded(patch_count, gridcell_count)

    def read(self, first_step: int, n_steps: int):
        """A chunk's forcing ``(xg, xp)`` for this simulation's slots (host arrays)."""
        return self.reader(self, first_step, n_steps)


# ---------------------------------------------------------------------------------------
# Tower runs: the simulation and its forcing reader
# ---------------------------------------------------------------------------------------


class TowerReader:
    """The forcing of a tower run: each gridcell's :class:`clm_ml.tower_forcing.
    TowerForcingSource` read chunk by chunk (``clm_ml.chunk_inputs.forcing_inputs``), the
    meteorology per gridcell slot (padding gridcells read the last real gridcell), the land
    data per patch slot (its gridcell's CLM history)."""

    def __init__(self, sources, patch_gridcell):
        self.sources = sources
        self.patch_gridcell = np.asarray(patch_gridcell)  # domain patch -> domain gridcell

    def __call__(self, sim: Simulation, first_step: int, n_steps: int):
        from clm_ml.chunk_inputs import forcing_inputs

        times = [sim.time_of(k) for k in range(first_step, first_step + n_steps)]
        days = [t.calday for t in times]
        days_beg = [t.calday_beg for t in times]
        parts = [forcing_inputs(s.chunk(first_step, n_steps), days, days_beg) for s in self.sources]
        met, land = jax.tree.map(lambda *x: np.concatenate(x, axis=1), *parts)
        src_g = np.minimum(np.arange(sim.n_gridcells), sim.n_real_gridcells - 1)
        land_g = self.patch_gridcell[sim.source]
        xg = jax.tree.map(lambda a: np.take(a, src_g, axis=1), met)
        xp = jax.tree.map(lambda a: np.take(a, land_g, axis=1), land)
        return xg, xp


def table_search(choice: str) -> str:
    """The psihat table search of a run (model.yaml ``numerics.table_search``; decision 114):
    ``auto`` is ``unrolled`` on a GPU (1.9x faster for the 1 degree globe: a loop costs one
    command per iteration there) and ``loop`` on the CPU (faster there, and the arithmetic of
    the regression reference)."""
    if choice != "auto":
        return choice
    return "unrolled" if jax.default_backend() == "gpu" else "loop"


def step_options(run):
    """The time step's static choices (``clm_ml.step.StepOptions``) of a run."""
    from clm_ml.options import MLOptions, substep_options
    from clm_ml.processes.orbit import shr_orb_params
    from clm_ml.setup.soil import NLEVGRND, NLEVSOI
    from clm_ml.step import StepOptions

    f = run.run_file.forcing
    interpolation = (f.gridded if run.is_grid else f.meteorology).interpolation
    opts = MLOptions.from_codes(run.config.to_fortran_codes(interpolation))
    soil_water = getattr(f, "soil_water", None)
    adjust = None if run.is_grid or soil_water is None else soil_water.adjustment
    eccen, _obliq, _mvelp, obliqr, lambm0, mvelpp = shr_orb_params(run.start.year)
    return (
        StepOptions(
            substep=substep_options(
                opts.step_options(), table_search(run.config.numerics.table_search)
            ),
            step_seconds=int(run.step_seconds),
            layer_pai_floor=float(opts.layer_pai_floor),
            flux_profile_type=int(opts.flux_profile_type),
            n_soil_adjust=int(adjust.levels) if adjust else 0,
            n_soil=NLEVSOI,
            eccen=float(eccen),
            mvelpp=float(mvelpp),
            lambm0=float(lambm0),
            obliqr=float(obliqr),
        ),
        opts,
        NLEVGRND,
    )


def simulation(run) -> Simulation:
    """The :class:`Simulation` of ``run`` (``clm_ml.experiment.Experiment``): every patch set up
    (``clm_ml.setup``), domain order, not yet laid out for devices (:meth:`Simulation.
    laid_out`)."""
    from clm_ml.io.rsl_tables import default_rsl_tables
    from clm_ml.params import PFT_NAMES
    from clm_ml.setup.patches import build_patches

    if run.is_grid:
        from clm_ml.setup.grid import grid_simulation_parts

        parts = grid_simulation_parts(run)
    else:
        from clm_ml.setup.towers import first_values, gridcell_sources

        sources = gridcell_sources(run)
        first = first_values(run, sources)
        gi = run.domain.patches.gridcell_index
        parts = {
            "first": {k: v[gi] for k, v in first.items()},
            "reader": TowerReader(sources, gi),
            "land_fn": None,
        }
    options, opts, _ = step_options(run)
    if opts.mlcan_to_clm != 0:
        raise NotImplementedError("mlcan_to_clm = 1 (fluxes passed to CLM) is not supported")
    setup = build_patches(run.domain, run.params, run.config, parts["first"])
    d = run.domain
    pfts = [PFT_NAMES[int(i)] for i in d.patches.pft_index]
    pft_names = tuple(dict.fromkeys(pfts))
    table = jax.tree.map(
        lambda *x: jnp.stack([jnp.asarray(v) for v in x]),
        *[run.params.canopy_params(n) for n in pft_names],
    )
    return Simulation(
        start=run.start,
        step_seconds=int(run.step_seconds),
        n_steps=int(run.n_steps),
        calendar=run.calkindflag,
        options=options,
        # the state's acclimation temperature: the whole-run mean, or the running mean (also
        # carried when acclimation is off, as before; the photosynthesis does not use it then)
        acclimation=(
            "run_mean"
            if run.config.processes.photosynthesis.temperature_acclimation == "run_mean"
            else "running_mean"
        ),
        acclimation_days=float(run.config.processes.photosynthesis.acclimation_days),
        tables=default_rsl_tables(),
        max_layers=int(run.config.canopy_layers.max_layers),
        params_table=table,
        pft_names=pft_names,
        constants=setup.constants,
        state0=setup.state,
        pft_row=np.array([pft_names.index(n) for n in pfts]),
        c3=setup.c3,
        gridcell_index=np.asarray(d.patches.gridcell_index, dtype=int),
        weight=np.asarray(d.patches.weight, dtype=float),
        n_gridcells=int(d.n_gridcells),
        names=list(_patch_names(run)),
        soil_depth=setup.soil_depth,
        checkpoint=bool(run.config.numerics.checkpoint_steps),
        reader=parts["reader"],
        land_fn=parts["land_fn"],
    )


def _patch_names(run) -> list:
    """Each patch's name in messages: its gridcell's name, with ``#k`` (its place on the
    gridcell) when the gridcell has several patches."""
    gidx = run.domain.patches.gridcell_index
    names = []
    for p, g in enumerate(gidx):
        name = run.domain.gridcells.names[int(g)]
        if int(np.sum(gidx == g)) > 1:
            name = f"{name}#{int(np.sum(gidx[:p] == g))}"
        names.append(name)
    return names


# ---------------------------------------------------------------------------------------
# The compiled chunk
# ---------------------------------------------------------------------------------------


def advance_acclimation(sim: Simulation, state, tref):
    """The state with the acclimation temperature's running mean advanced by this step's
    forcing air temperature ``tref`` (decision 15): ``mean += (T - mean) / min(n, N)``, with
    ``n`` the steps averaged so far (this one included) and ``N`` the steps in
    ``acclimation_days`` (CLM's ``runmean`` accumulation). The setup starts it at the first
    step's air temperature with no steps, so the first step leaves it unchanged. Unchanged
    for ``acclimation = run_mean``."""
    if sim.acclimation != "running_mean":
        return state
    c = state.canopy
    period = sim.acclimation_days * 86400.0 / sim.step_seconds
    n = c.acclimation_steps + 1.0
    t = jnp.asarray(tref, dtype=jnp.float64)
    mean = c.acclimation_temperature + (t - c.acclimation_temperature) / jnp.minimum(n, period)
    return state._replace(canopy=c._replace(acclimation_temperature=mean, acclimation_steps=n))


def step_variables(names, max_layers: int):
    """``variables(fields) -> {name: value}``: the output variables ``names`` of one patch's
    step from its fields (``clm_ml.outputs.step_fields``), computed with jax.numpy; raw
    field names (``diagnostics.<name>``, ``state.<name>``, ``layers.<name>``) give the field
    itself (validation)."""
    from clm_ml.outputs import VARIABLES_BY_NAME, _Raw

    def variables(fields) -> dict:
        r = _Raw({k: jnp.asarray(v)[None, None] for k, v in fields.items()}, max_layers, xp=jnp)
        out = {}
        for n in names:
            if n == "soil_temperature":
                out[n] = fields["state.soil_temperature"]
            elif "." in n:  # raw step field (validation)
                out[n] = jnp.asarray(fields[n], dtype=float)
            else:
                out[n] = VARIABLES_BY_NAME[n].compute(r)[0, 0]
        return out

    return variables


def patch_step(sim: Simulation, c3: bool, names):
    """One patch's time step for compiled chunks: ``step(constants, state, x, params,
    tables) -> (state, {variable: value}, errors)``: the acclimation update, then
    ``clm_ml.step.step``; the output variables ``names`` from the step's fields; ``errors``
    the step's largest error of each conservation check."""
    from clm_ml import step as S
    from clm_ml.outputs import step_fields

    variables = step_variables(names, sim.max_layers)
    options = sim.options

    def step(constants, state, x, params, tables):
        state = advance_acclimation(sim, state, x.tref)
        result = S.step(state, x, constants, params, tables, options=options, c3=c3)
        fields = step_fields(result, constants.layers)
        return result.state, variables(fields), result.diagnostics.balance_error

    return step


def make_runner(sim: Simulation, streams, gridcell_vars):
    """A compiled function ``(state, xg, xp, slots, buffers, params, weight, alive, gidx,
    constants, active=None, *, tables) -> (state, buffers, errors)`` advancing all patches
    of ``sim`` over a chunk as one ``jax.lax.scan``, the one-patch step (:func:`patch_step`)
    ``vmap``ped over the patch axis (Phase 6 step 3).

    ``state``, ``params`` (``CanopyParams`` per patch) and ``constants``
    (``PatchConstants``) have a leading patch axis; ``constants`` is an input, so patches
    with other layer grids or soils reuse the program. The forcing comes per gridcell
    (``xg``: ``clm_ml.step.MetInputs``, axes (step, gridcell, ...)) and is gathered to the
    patches; the land data per patch (``xp``: ``clm_ml.step.LandInputs``, (step, patch,
    ...); grid runs: the inputs of ``sim.land_fn``). Stream buffers hold every variable per
    patch ``(record, patch, ...)`` and, for ``gridcell_vars[i]`` of stream ``i``, the
    area-weighted sum over each gridcell's patches formed at every step
    (``gridcell:<variable>``). ``errors``: ``(patch, check)``. ``active`` (``(step,)``
    bool): steps marked False (the padding of a short last chunk) change nothing.

    ``alive`` ``(patch,)`` marks the patches still in the run (decision 25): an excluded
    patch, or one whose value is not finite, is left out of its gridcell's sum (the others'
    weights renormalized) and its parameters enter through ``stop_gradient``.
    ``tables``: the psihat lookup tables, one copy for every patch and step (C22)."""
    from clm_ml import output_streams as S
    from clm_ml.step import step_inputs as combine_inputs

    names = sorted({n for spec in streams for n in spec.variables})
    groups = sim.groups
    group_steps = [patch_step(sim, bool(c3), names) for c3, _ in groups]
    if sim.layout is None:
        n_p, n_g = sim.n_patches, sim.n_gridcells
    else:
        n_p, n_g = sim.layout[2] + sim.layout[3], sim.layout[1]

    def batched_step(constants, state, x_patch, params, tables):
        """Every patch's step: per group (contiguous slot ranges), vmapped; the groups'
        results joined in patch order (slices and concatenation only: scattering into the
        scan's carry gave wrong states on CPU, Phase 6 step 5)."""
        vmapped = [jax.vmap(s, in_axes=(0, 0, 0, 0, None)) for s in group_steps]
        if len(groups) == 1:
            return vmapped[0](constants, state, x_patch, params, tables)
        outs = []
        for (_, idx), step_g in zip(groups, vmapped):
            lo, hi = int(idx[0]), int(idx[-1]) + 1
            cut = lambda t: jax.tree.map(lambda a: a[lo:hi], t)  # noqa: E731
            outs.append(step_g(cut(constants), cut(state), cut(x_patch), cut(params), tables))
        return jax.tree.map(lambda *parts: jnp.concatenate(parts), *outs)

    def gridcell_sum(v, weight, alive, gidx):  # (P, ...) -> (G, ...): area-weighted sum
        def per(a):
            return a.reshape((-1,) + (1,) * (v.ndim - 1))

        def seg(a):
            return jax.ops.segment_sum(a, gidx, num_segments=n_g)

        ok = alive & jnp.all(jnp.isfinite(v.reshape(n_p, -1)), axis=1)
        v_safe = jnp.where(per(ok), v, 0.0)  # NaN never enters a sum (nor its gradient)
        plain = seg(v_safe * per(weight))
        w_ok = jnp.where(ok, weight, 0.0)
        den = seg(w_ok)
        den_b = den.reshape((-1,) + (1,) * (v.ndim - 1))
        renorm = seg(v_safe * per(w_ok)) / jnp.where(den_b > 0.0, den_b, 1.0)
        renorm = jnp.where(den_b > 0.0, renorm, jnp.nan)  # every patch of the gridcell out
        some_out = (seg((~ok).astype(jnp.int32)) > 0).reshape(den_b.shape)
        return jnp.where(some_out, renorm, plain)

    land_fn = sim.land_fn

    def to_patches(a, gidx):
        """A per-gridcell input (gridcell, ...) gathered to the patches; a value without a
        gridcell axis (the step's calendar days in grid runs) broadcast to them."""
        return jnp.take(a, gidx, 0) if a.ndim >= 1 else jnp.broadcast_to(a, (n_p,))

    def step(constants, land, consts, carry, x):
        params, weight, alive, gidx, tables = consts
        xg, xp, slots, *active = x

        def advance(carry):
            state, buffers, errors = carry
            met = jax.tree.map(lambda a: to_patches(a, gidx), xg)
            # grid runs: the step's land data from the monthly records
            x_patch = xp if land_fn is None else land_fn(xp, land, gidx)
            state, values, step_errors = batched_step(
                constants, state, combine_inputs(met, x_patch), params, tables
            )
            new_buffers = []
            for spec, b, slot, gvars in zip(streams, buffers, slots, gridcell_vars):
                vals = {n: values[n] for n in spec.variables}
                vals.update(
                    {f"gridcell:{n}": gridcell_sum(values[n], weight, alive, gidx) for n in gvars}
                )
                new_buffers.append(S.accumulate(spec.statistic, b, slot, vals))
            return state, tuple(new_buffers), jnp.maximum(errors, step_errors)

        if not active:
            return advance(carry), None
        # A padded step of a short last chunk changes nothing (decision 33) and is not
        # computed (decision 116): lax.cond runs only the branch taken, where jnp.where on the
        # results computed the whole step and discarded it
        return jax.lax.cond(active[0], advance, lambda c: c, carry), None

    checkpointed_step = jax.checkpoint(step) if sim.checkpoint else step

    def run_chunk(
        state, xg, xp, slots, buffers, params, weight, alive, gidx, constants, active=None,
        *, tables,
    ):  # fmt: skip
        params = jax.tree.map(
            lambda p: jnp.where(
                alive.reshape((-1,) + (1,) * (p.ndim - 1)), p, jax.lax.stop_gradient(p)
            ),
            params,
        )
        # the same at every step: held once, not copied along the step axis
        consts = (params, weight, alive, gidx, tables)
        errors = jnp.zeros((n_p, len(BALANCE_CHECKS)))
        land = None
        if land_fn is not None:
            xp, land = xp  # per-step weights (scanned), monthly records (chunk constant)
        (state, buffers, errors), _ = jax.lax.scan(
            lambda c, x: checkpointed_step(constants, land, consts, c, x),
            (state, buffers, errors),
            (xg, xp, slots) + (() if active is None else (active,)),
        )
        return state, buffers, errors

    if sim.n_devices == 1:
        return jax.jit(run_chunk)
    return jax.jit(_sharded(run_chunk, sim.n_devices))


def _sharded(run_chunk, n_devices: int):
    """``run_chunk`` run on ``n_devices`` devices with ``jax.shard_map`` (Phase 6 step 5):
    every array with a patch or gridcell axis is split into the devices' blocks along that
    axis (axis 0, or axis 1 behind a time/record axis); each device runs the chunk on its
    block. The parameters come per patch, so their gradients are summed over the devices
    by the gather's transpose outside the shard_map."""
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as Pspec

    devices = jax.devices()
    if len(devices) < n_devices:
        raise RuntimeError(f"execution.devices is {n_devices} but JAX sees {len(devices)}")
    mesh = Mesh(np.array(devices[:n_devices]), ("patch",))
    lead, behind, whole = Pspec("patch"), Pspec(None, "patch"), Pspec()

    def wrapper(
        state, xg, xp, slots, buffers, params, weight, alive, gidx, constants, active=None,
        *, tables,
    ):  # fmt: skip

        def spec_tree(tree, spec):
            return jax.tree.map(lambda _: spec, tree)

        def by_ndim(tree):  # (time/record, patch or gridcell, ...) split; (time,) replicated
            return jax.tree.map(lambda x: behind if x.ndim >= 2 else whole, tree)

        buffer_specs = by_ndim(buffers)
        in_specs = (
            spec_tree(state, lead), by_ndim(xg), by_ndim(xp), spec_tree(slots, whole),
            buffer_specs, spec_tree(params, lead), lead, lead, lead,
            spec_tree(constants, lead), spec_tree(tables, whole),
        )  # fmt: skip
        out_specs = (spec_tree(state, lead), buffer_specs, lead)
        args = (state, xg, xp, slots, buffers, params, weight, alive, gidx, constants, tables)
        if active is not None:  # (step,): the same on every device
            in_specs, args = in_specs + (whole,), args + (active,)

        def chunk(*a):  # shard_map passes positional arguments
            *rest, tables = a[:11]
            return run_chunk(*rest, *a[11:], tables=tables)

        f = jax.shard_map(chunk, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
        return f(*args)

    return wrapper


def place_inputs(sim: Simulation, tree):
    """A chunk's inputs moved to the devices as the compiled chunk takes them: with several
    devices, every array with a patch or gridcell axis (axis 1) split into the devices'
    blocks, the others replicated (each device receives only its own block)."""
    if sim.n_devices == 1:
        return jax.device_put(tree)
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as Pspec

    mesh = Mesh(np.array(jax.devices()[: sim.n_devices]), ("patch",))
    split, whole = NamedSharding(mesh, Pspec(None, "patch")), NamedSharding(mesh, Pspec())
    return jax.tree.map(lambda a: jax.device_put(a, split if np.ndim(a) >= 2 else whole), tree)


def place_patches(sim: Simulation, tree):
    """Per-patch arrays (leading patch axis) on the devices: split by device blocks with
    several devices."""
    if sim.n_devices == 1:
        return jax.tree.map(jnp.asarray, tree)
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as Pspec

    mesh = Mesh(np.array(jax.devices()[: sim.n_devices]), ("patch",))
    return jax.device_put(tree, NamedSharding(mesh, Pspec("patch")))


def pad_steps(tree, length: int):
    """Every leaf's leading (step) axis extended to ``length`` by repeating its last entry
    (the padded steps of a short last chunk, decision 33); numpy leaves stay numpy."""

    def pad(a):
        n = a.shape[0]
        if n == length:
            return a
        xp = np if isinstance(a, np.ndarray) else jnp
        return xp.concatenate([a, xp.repeat(a[-1:], length - n, axis=0)])

    return jax.tree.map(pad, tree)


def read_chunk_inputs(sim: Simulation, first_step: int, n_steps: int, length=None):
    """The chunk's forcing padded to ``length`` steps (None: not padded) and placed on the
    devices. Runs on the reader thread, so the transfer overlaps the running chunk."""
    xg, xp = sim.read(first_step, n_steps)
    if length is not None and length != n_steps:
        xg = pad_steps(xg, length)
        xp = (pad_steps(xp[0], length), xp[1]) if sim.land_fn is not None else pad_steps(xp, length)
    return place_inputs(sim, (xg, xp))


def output_shapes(sim: Simulation, names) -> dict:
    """Per-patch shape of each output variable (for ``init_buffers``), from patch 0's first
    step (abstract evaluation: nothing is computed), found once per set of names and kept
    on the simulation (it traces a step: ~1 s)."""
    cache = sim.__dict__.setdefault("_output_shapes", {})
    key = tuple(sorted(names))
    if key not in cache:
        cache[key] = _output_shapes(sim, key)
    return dict(cache[key])


def _output_shapes(sim: Simulation, names) -> dict:
    from clm_ml.step import step_inputs as combine

    step = patch_step(sim, bool(sim.c3[0]), tuple(names))
    xg, xp = sim.read(0, 1)
    g0 = int(sim.runner_gidx[0])
    met = jax.tree.map(lambda a: a[0][g0] if np.ndim(a) > 1 else a[0], xg)
    if sim.land_fn is not None:
        xw, land = xp
        gidx = jnp.asarray(sim.runner_gidx)
        lands = sim.land_fn(jax.tree.map(lambda a: a[0], xw), land, gidx)
        land0 = jax.tree.map(lambda a: a[0], lands)
    else:
        land0 = jax.tree.map(lambda a: a[0][0], xp)

    def one(tree):
        return jax.tree.map(lambda a: jnp.asarray(np.asarray(a)[0]), tree)

    x = combine(jax.tree.map(jnp.asarray, met), jax.tree.map(jnp.asarray, land0))
    args = (one(sim.constants), one(sim.state0), x, one(sim.patch_params()), sim.tables)
    out = jax.eval_shape(step, *args)
    return {n: tuple(v.shape) for n, v in out[1].items()}


# ---------------------------------------------------------------------------------------
# Chunks, checks, restarts
# ---------------------------------------------------------------------------------------


def month_chunks(sim: Simulation, first_step: int) -> list:
    """``(first_step, n_steps)`` chunks covering run steps ``first_step ... n_steps - 1``, one
    per calendar month of the steps' start times (plan decision 17)."""
    chunks = []
    for k, start in zip(range(first_step, sim.n_steps), sim.step_starts(first_step,
                                                                          sim.n_steps - first_step)):  # fmt: skip
        key = (start.year, start.month)
        if chunks and chunks[-1][0] == key:
            chunks[-1][2] += 1
        else:
            chunks.append([key, k, 1])
    return [(k, n) for _, k, n in chunks]


def chunk_schedule(sim: Simulation, first_step: int, chunk_steps: Optional[int] = None):
    """``(chunks, length)``: the ``(first_step, n_steps)`` chunks covering run steps
    ``first_step ... n_steps - 1`` and the length every chunk is run at (decision 33).
    ``chunk_steps``: fixed-length chunks counted from step 0, each run at ``length =
    min(chunk_steps, n_steps)`` steps (a short last chunk padded, so one compiled program
    serves the run; a run resumed from a restart written at a chunk end keeps the same
    chunks). ``None``: calendar months, each at its own length (``length`` None)."""
    if chunk_steps is None:
        return month_chunks(sim, first_step), None
    length = max(1, min(chunk_steps, sim.n_steps))
    if first_step % length:
        raise ValueError(
            f"run step {first_step} is not the start of a {length}-step chunk (a restart "
            "written with another execution.chunk?)"
        )
    chunks = [(k, min(length, sim.n_steps - k)) for k in range(first_step, sim.n_steps, length)]
    return chunks, length


#: Shortest period (s) of each output frequency, for the record slots a chunk can touch
_SHORTEST_PERIOD = {"hourly": 3600, "daily": 86400, "monthly": 28 * 86400, "yearly": 365 * 86400}


def max_chunk_records(frequency: str, length: int, step_seconds: int) -> int:
    """The most records of a stream of ``frequency`` that ``length`` consecutive steps can
    touch, wherever they start (every chunk then has the same buffer shapes)."""
    if frequency == "step":
        return length
    period = max(1, _SHORTEST_PERIOD[frequency] // step_seconds)
    return min(length, (length - 1) // period + 2)


#: State fields that are temperatures (K), checked against a plausible range
_TEMPERATURE_FIELDS = (
    "air_temperature", "leaf_temperature", "ground_temperature", "acclimation_temperature",
    "temperature",
)  # fmt: skip
_TEMPERATURE_RANGE = (150.0, 350.0)


def health_problems(state, layers) -> list:
    """The problems of one patch's state (empty: healthy): non-finite values, or
    temperatures outside 150-350 K, in the layers in use (``layers``: its
    ``PatchLayers``; every soil level and scalar). Temperatures of exactly 0 mark unused
    entries (leafless layers) and are not checked."""
    in_use = np.arange(np.shape(layers.dz)[-1]) < int(np.asarray(layers.n_layers))
    problems = []
    for part in ("canopy", "soil"):
        for name, value in getattr(state, part)._asdict().items():
            x = np.asarray(value, dtype=float)
            mask = np.ones(x.shape, dtype=bool)
            if part == "canopy" and x.ndim >= 1:  # (layer, ...)
                mask = in_use.reshape((-1,) + (1,) * (x.ndim - 1)) & mask
            bad = mask & ~np.isfinite(x)
            if bad.any():
                problems.append(f"{part}.{name}: {int(bad.sum())} non-finite values")
            elif name in _TEMPERATURE_FIELDS:
                used = mask & (x != 0.0)
                if used.any():
                    lo, hi = float(x[used].min()), float(x[used].max())
                    if lo < _TEMPERATURE_RANGE[0] or hi > _TEMPERATURE_RANGE[1]:
                        problems.append(f"{part}.{name}: range {lo:.1f}-{hi:.1f} K")
    return problems


def check_health(state, layers, where: str) -> None:
    """Stop the run if one patch's state is unhealthy (:func:`health_problems`)."""
    problems = health_problems(state, layers)
    if problems:
        raise ModelError(f"unhealthy model state after {where}: " + "; ".join(problems))


def restart_boundary(start: datetime.datetime, end: datetime.datetime, every) -> bool:
    """Whether a chunk from ``start`` to ``end`` reaches a ``restart_every`` boundary
    (``(n, "month")`` or ``(n, "year")``; None: never): a boundary ``b`` with
    ``start < b <= end``."""
    if every is None:
        return False
    n, unit = every
    month = start.year * 12 + start.month
    while True:
        y, m = divmod(month, 12)
        b = datetime.datetime(y, m + 1, 1)
        if b > end:
            return False
        if b > start and (((y * 12 + m) % n == 0) if unit == "month" else (m == 0 and y % n == 0)):
            return True
        month += 1


def gridcell_variables(names, level) -> tuple:
    """The variables of a stream accumulated per gridcell inside the chunk (level
    ``gridcell``): named variables without a ``layer`` axis."""
    from clm_ml.outputs import VARIABLES_BY_NAME

    if "gridcell" not in level:
        return ()
    return tuple(
        n for n in names if n in VARIABLES_BY_NAME and "layer" not in VARIABLES_BY_NAME[n].dims
    )


def ignored_check_indices(names) -> list:
    """Indices into ``BALANCE_CHECKS`` of model.yaml ``numerics.ignore_checks``."""
    out = []
    for name in names:
        hits = [i for i, (full, _) in enumerate(BALANCE_CHECKS) if name in full]
        if len(hits) != 1:
            raise ValueError(
                f"model.yaml numerics.ignore_checks: {name!r} matches {len(hits)} checks; "
                f"checks: {[full for full, _ in BALANCE_CHECKS]}"
            )
        out += hits
    return out


def report_ignored_checks(errors, ignored, alive, sim: Simulation, where: str):
    """``errors`` with the ignored checks set to 0, after printing how many patches exceeded
    each one's tolerance in the chunk and the largest error."""
    if not ignored:
        return errors
    errors = np.array(errors, dtype=float).reshape(sim.n_patches, len(BALANCE_CHECKS))
    use = real_mask(sim) & np.asarray(alive, dtype=bool)
    for i in ignored:
        name, tol = BALANCE_CHECKS[i]
        e = errors[use, i]
        n_over = int(np.sum(e > tol))
        if n_over:
            print(f"ignored check ({where}): {name}: {n_over} patches above {tol:g}, "
                  f"largest {e.max():.3g}", flush=True)  # fmt: skip
        errors[:, i] = 0.0
    return errors


def _patch(tree, p):
    return jax.tree.map(lambda x: np.asarray(x)[p], tree)


def real_mask(sim: Simulation) -> np.ndarray:
    """``(slots,)`` True for the slots that hold a domain patch (not padding)."""
    real = np.zeros(sim.n_patches, dtype=bool)
    real[np.asarray(sim.real_index)] = True
    return real


def batch_failures(errors, state, sim: Simulation, alive=None) -> dict:
    """``{patch: [problem, ...]}`` for the patches (of those ``alive``) that fail a
    conservation check or whose state is unhealthy; padding slots are not checked (they are
    copies with weight 0, outside every output and gradient; open item O29)."""
    errors = np.asarray(errors, dtype=float).reshape(sim.n_patches, len(BALANCE_CHECKS))
    state = jax.tree.map(np.asarray, state)  # one transfer from the devices
    failed = {}
    real = real_mask(sim)
    for p in range(sim.n_patches):
        if not real[p] or (alive is not None and not alive[p]):
            continue
        problems = [
            f"{name}: error {e:.3g} exceeds {tol:g}"
            for (name, tol), e in zip(BALANCE_CHECKS, errors[p])
            if e > tol
        ]
        problems += health_problems(_patch(state, p), _patch(sim.constants.layers, p))
        if problems:
            failed[p] = problems
    return failed


def check_batch_balance(errors, sim: Simulation, where: str, alive=None) -> None:
    """Fortran's conservation checks per patch: stop the run naming the patches that fail
    (decision 25, default); padding slots are not checked."""
    errors = np.asarray(errors, dtype=float).reshape(sim.n_patches, len(BALANCE_CHECKS))
    failed = []
    real = real_mask(sim)
    for p in range(sim.n_patches):
        if not real[p] or (alive is not None and not alive[p]):
            continue
        for (name, tol), e in zip(BALANCE_CHECKS, errors[p]):
            if e > tol:
                failed.append(f"{sim.names[p]}: {name}: error {e:.3g} exceeds {tol:g}")
    if failed:
        raise ModelError(f"ERROR: ({where}) " + "; ".join(failed))


def check_batch(errors, state, sim: Simulation, where: str) -> None:
    """Conservation checks and the state's health, per patch: stop the run naming the
    patches that fail (decision 25, default)."""
    check_batch_balance(errors, sim, where)
    state = jax.tree.map(np.asarray, state)  # one transfer from the devices
    for p in np.flatnonzero(real_mask(sim)):
        check_health(_patch(state, p), _patch(sim.constants.layers, p), f"{where}, {sim.names[p]}")


def freeze_patches(state, previous, alive):
    """``state`` with the patches not ``alive`` taken from ``previous`` (their last good
    state, so they stay finite while excluded)."""
    keep = jnp.asarray(alive)
    return jax.tree.map(
        lambda new, old: jnp.where(keep.reshape((-1,) + (1,) * (new.ndim - 1)), new, old),
        state,
        previous,
    )


def fill_patches(buffers, alive):
    """Stream buffers with the patch values of the patches not ``alive`` set to NaN."""
    dead = jnp.asarray(~np.asarray(alive))

    def fill(values: dict) -> dict:
        out = {}
        for k, v in values.items():
            if k.startswith("gridcell:") or v.ndim < 2:
                out[k] = v
            else:
                mask = dead.reshape((1, -1) + (1,) * (v.ndim - 2))
                out[k] = jnp.where(mask, jnp.nan, v)
        return out

    return {k: fill(v) if isinstance(v, dict) else v for k, v in buffers.items()}


def real_part(buffers, sim: Simulation):
    """Stream buffers without the padding patches and gridcells, in domain order."""
    if sim.n_real == sim.n_patches and sim.n_real_gridcells == sim.n_gridcells:
        if sim.layout is None:
            return buffers
    real = jnp.asarray(sim.real_index)

    def cut(values: dict) -> dict:
        return {
            k: (
                v[:, : sim.n_real_gridcells]
                if k.startswith("gridcell:")
                else jnp.take(v, real, axis=1)
            )
            for k, v in values.items()
        }

    return {k: cut(v) if isinstance(v, dict) else v for k, v in buffers.items()}


# ---------------------------------------------------------------------------------------
# The run loop
# ---------------------------------------------------------------------------------------


def run_streams(
    sim: Simulation,
    streams,
    on_records,
    params=None,
    *,
    restart: Optional[dict] = None,
    restart_path: Optional[Path] = None,
    restart_every=None,
    save_extra=None,
    restore_extra=None,
    levels=None,
    on_patch_error: str = "stop",
    max_failed_patch_fraction: float = 0.01,
    ignore_checks=(),
    chunk_steps: Optional[int] = None,
    timing: bool = False,
) -> None:
    """Run the model with output streams (plan §6.9): compiled chunks of ``chunk_steps``
    steps from the first step (decision 33; a short last chunk padded, so one program serves
    the run; None: calendar months), accumulating every stream inside the scan. The next
    chunk's forcing is read and placed on the devices while one runs. After each chunk the
    conservation checks and the state's health are checked, and a restart is written
    (``restart_path``) at the first chunk end on or after each ``restart_every`` boundary
    and at the end of the run.

    Output values have a patch axis, ``(patch, ...)``; streams whose ``levels[i]`` include
    ``gridcell`` also carry ``gridcell:<variable>`` values ``(gridcell, ...)``.
    ``on_records(stream_index, records)`` receives each stream's completed records
    ``[(key, end_time, {variable: value}), ...]`` as soon as they are complete.

    ``restart`` (from :func:`read_restart`) starts from a saved state: inside the run, the
    run continues from there with the saved output bookkeeping (results equal an
    uninterrupted run); at the run's start, a new run starts from that state.
    ``save_extra()``/``restore_extra(saved)`` carry the caller's bookkeeping (e.g. output
    files in progress). ``params`` (per patch) replaces the patches' own parameters.

    A patch that fails a conservation check or the health check (decision 25): with
    ``on_patch_error="stop"`` the run stops naming it; with ``"exclude"`` it is left out from
    then on (its state frozen at its last good chunk, its outputs fill values, gridcell values
    from the other patches with renormalized weights), and the run stops when more than
    ``max_failed_patch_fraction`` of the patches have failed."""
    from concurrent.futures import ThreadPoolExecutor

    from clm_ml import output_streams as S

    n_p, n_g = sim.n_patches, sim.n_gridcells
    levels = levels or [("patch",)] * len(streams)
    names = sorted({n for spec in streams for n in spec.variables})
    gvars = [gridcell_variables(spec.variables, lv) for spec, lv in zip(streams, levels)]
    collectors = [S.StreamCollector(spec) for spec in streams]
    step_seconds = sim.step_seconds
    if params is None:
        params = sim.patch_params()
    if on_patch_error not in ("stop", "exclude"):
        raise ValueError(f"on_patch_error must be stop or exclude, got {on_patch_error!r}")
    ignored = ignored_check_indices(ignore_checks)
    alive = np.ones(n_p, dtype=bool)
    shapes = {n: (n_p,) + sh for n, sh in output_shapes(sim, names).items()}

    def buffer_shapes(spec, gv):
        out = {n: shapes[n] for n in spec.variables}
        out.update({f"gridcell:{n}": (n_g,) + shapes[n][1:] for n in gv})
        return out

    state = sim.state0
    first = 0
    if restart is not None:
        if restart.get("calendar", sim.calendar) != sim.calendar:
            raise ValueError(
                f"restart written with the {restart['calendar']} calendar, this run uses "
                f"{sim.calendar}"
            )
        elapsed = calendar.seconds_between(sim.start, restart["time"], sim.calendar)
        first = int(round(elapsed / step_seconds))
        if first * step_seconds != elapsed or not 0 <= first < sim.n_steps:
            raise ValueError(
                f"restart time {restart['time']} is not a step of this run "
                f"({sim.start} + {sim.n_steps} steps of {step_seconds} s)"
            )
        if restart.get("layout") != sim.layout:
            raise ValueError(
                f"restart written with device layout {restart.get('layout')}, this run has "
                f"{sim.layout} (execution.devices must be the same)"
            )
        restored = jax.tree.map(np.asarray, restart["state"])
        for (where, a), b in zip(
            jax.tree_util.tree_leaves_with_path(restored), jax.tree.leaves(state)
        ):
            if np.shape(a) != np.shape(b):
                raise ValueError(
                    f"restart state {jax.tree_util.keystr(where)} has shape {np.shape(a)}, "
                    f"this run {np.shape(b)} (e.g. a different model.yaml max_layers or "
                    "number of patches)"
                )
        state = restored
        if first > 0:  # continuing the same run: restore the output bookkeeping
            saved = restart.get("streams")
            mine = [(sp.name, sp.statistic) for sp in streams]
            if saved is not None and (len(saved) != len(mine) or any(
                    a is not None and tuple(a) != b for a, b in zip(saved, mine))):  # fmt: skip
                raise ValueError(
                    f"restart written with output streams {saved}, this run has {mine} (the "
                    "streams of a continued run must be the same)"
                )
            alive = np.asarray(restart.get("alive", alive), dtype=bool)
            for collector, pending in zip(collectors, restart["collectors"]):
                collector._pending = pending
            if restore_extra is not None:
                restore_extra(restart.get("extra"))
    state = place_patches(sim, state)
    constants = place_patches(sim, sim.constants)
    runner = make_runner(sim, streams, gvars)

    schedule, length = chunk_schedule(sim, first, chunk_steps)
    reader = ThreadPoolExecutor(max_workers=1)
    pending = reader.submit(read_chunk_inputs, sim, *schedule[0], length) if schedule else None
    try:
        weight = jnp.asarray(sim.weight, dtype=jnp.float64)
        gidx = jnp.asarray(sim.runner_gidx)
        for c, (first_step, n) in enumerate(schedule):
            xg, xp = pending.result()
            if c + 1 < len(schedule):
                pending = reader.submit(read_chunk_inputs, sim, *schedule[c + 1], length)
            starts = sim.step_starts(first_step, n)
            records = [
                S.chunk_records(starts, first_step, step_seconds, s.frequency, sim.calendar)
                for s in streams
            ]
            n_run = n if length is None else length
            n_slots = [
                (
                    len(rc.keys)
                    if length is None
                    else max_chunk_records(s.frequency, length, step_seconds)
                )
                for s, rc in zip(streams, records)
            ]
            buffers = tuple(
                S.init_buffers(s.statistic, k, buffer_shapes(s, gv), jnp)
                for s, k, gv in zip(streams, n_slots, gvars)
            )
            slots = tuple(jnp.asarray(pad_steps(rc.index, n_run)) for rc in records)
            active = jnp.asarray(np.arange(n_run) < n)
            chunk_start_state = state
            t_chunk = _time.perf_counter()
            state, buffers, errors = runner(
                state, xg, xp, slots, buffers, params, weight, jnp.asarray(alive), gidx, constants,
                active, tables=sim.tables,
            )  # fmt: skip
            del xg, xp  # the chunk's forcing is not kept on the devices while the next one runs
            buffers = tuple(
                jax.tree.map(lambda a, k=len(rc.keys): a[:k], b) for b, rc in zip(buffers, records)
            )
            if timing:
                jax.block_until_ready(state)
                secs = _time.perf_counter() - t_chunk
                print(f"timing: chunk {starts[0]:%Y-%m-%d} {n} steps {secs:.1f} s "
                      f"({sim.n_real * n / secs:.0f} patch-steps/s)", flush=True)  # fmt: skip
            end = sim.step_end(first_step + n - 1)
            where = f"the chunk {starts[0]} to {end}"
            errors = report_ignored_checks(errors, ignored, alive, sim, where)
            if on_patch_error == "stop":
                check_batch(errors, state, sim, where)
            else:
                failed = batch_failures(errors, state, sim, alive)
                for p, problems in failed.items():
                    print(f"excluding {sim.names[p]} after {where}: " + "; ".join(problems))
                    alive[p] = False
                n_out = int((~alive[sim.real_index]).sum())
                if n_out > max_failed_patch_fraction * sim.n_real:
                    raise ModelError(
                        f"ERROR: {n_out} of {sim.n_real} patches failed (more than "
                        f"max_failed_patch_fraction {max_failed_patch_fraction:g}), after {where}"
                    )
                if not alive.all():
                    state = freeze_patches(state, chunk_start_state, alive)
                    buffers = tuple(fill_patches(b, alive) for b in buffers)
            for i, rc in enumerate(records):
                on_records(i, collectors[i].add_chunk(rc, real_part(buffers[i], sim)))
            last = first_step + n == sim.n_steps
            if restart_path is not None and (
                last or restart_boundary(starts[0], end, restart_every)
            ):
                write_restart(
                    restart_path,
                    {
                        "version": RESTART_VERSION,
                        "time": end,
                        "state": jax.tree.map(np.asarray, state),
                        "collectors": [col._pending for col in collectors],
                        "streams": [(sp.name, sp.statistic) for sp in streams],
                        "alive": alive.copy(),
                        "layout": sim.layout,
                        "extra": save_extra() if save_extra is not None else None,
                    },
                    calendar=sim.calendar,
                )
    finally:
        reader.shutdown(cancel_futures=True)
    for i, collector in enumerate(collectors):
        on_records(i, collector.close())
