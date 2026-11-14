"""
Experiment folders: an experiment is a folder with four complete files,

- ``run.yaml`` (:mod:`clm_ml.run`): when, with which forcing and land data, what it writes;
- ``domain.yaml`` (:mod:`clm_ml.domain`): where: towers or a grid;
- ``model.yaml`` (:mod:`clm_ml.config`): how: each process's parameterization and the
  numerics (static);
- ``params.yaml`` (:mod:`clm_ml.params`): which plants: every parameter of every PFT.

:func:`load_experiment` reads a folder strictly (every key present; nothing read from the
package), :func:`new_experiment` writes one from a compset (``clm_ml/compsets``),
:func:`upgrade_experiment` rewrites one in the current layout. ``Experiment.execute()`` sets
the run up (``clm_ml.setup``) and runs it in this process (``clm_ml.driver``), with one clock
for all gridcells (decision 24).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import os
from importlib import resources
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from clm_ml.config import ModelConfig, model_yaml_text
from clm_ml.domain import HEADER as DOMAIN_HEADER
from clm_ml.domain import Domain, DomainFile
from clm_ml.io.yaml_schema import (
    ConfigError,
    MissingKeys,
    apply_overrides,
    from_dict,
    read_yaml,
    render,
)
from clm_ml.params import PFT_NAMES, Params
from clm_ml.run import HEADER as RUN_HEADER
from clm_ml.run import RunFile, TimeSection, parse_chunk, parse_every, parse_length

#: The four files of an experiment folder, by the prefix of their ``--set`` keys
FILES = {"run": "run.yaml", "domain": "domain.yaml", "model": "model.yaml", "params": "params.yaml"}


def version() -> str:
    """The installed clm-ml-jax version (recorded in the files it writes)."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    try:
        return _version("clm-ml-jax")
    except PackageNotFoundError:
        return "dev"


def compsets_dir() -> Path:
    return Path(str(resources.files("clm_ml") / "compsets"))


def compset_names() -> list[str]:
    return sorted(p.stem for p in (compsets_dir() / "experiments").glob("*.yaml"))


# --------------------------------------------------------------------------------------
# The loaded experiment
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Experiment:
    """An experiment folder, loaded and validated (:func:`load_experiment`): the four files
    as typed objects (``run``, ``domain_file``, ``model``, ``params``), the domain built from
    ``domain.yaml`` (gridcells, patches, soil columns) and the run's time steps."""

    run_file: RunFile  # run.yaml
    domain_file: DomainFile  # domain.yaml
    config: ModelConfig  # model.yaml
    params: Params  # params.yaml
    domain: Domain  # the gridcells, patches and soil columns of domain.yaml
    base_dir: Path  # the folder; relative paths are resolved against it
    start: dt.datetime
    n_steps: int
    overrides: tuple = ()

    @property
    def run(self) -> RunFile:
        """run.yaml (the same object as ``run_file``)."""
        return self.run_file

    @property
    def model(self) -> ModelConfig:
        """model.yaml (the same object as ``config``)."""
        return self.config

    @property
    def source(self) -> str:
        """The folder's run.yaml, for messages."""
        return f"{self.base_dir.name}/run.yaml"

    @property
    def is_grid(self) -> bool:
        return self.domain_file.type == "grid"

    # -- paths -------------------------------------------------------------------------

    def path(self, text: str, tower: Optional[str] = None) -> Path:
        if tower is not None:
            text = text.replace("{tower}", tower)
        path = Path(text).expanduser()
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    def files(self, entry, tower: str, key: str = "files") -> list[Path]:
        """The files of a ``files`` entry, in order. ``{year}`` and ``{month}`` are
        expanded for every month from one month before the run to one month after it (the
        neighbouring records interpolation needs); files for months inside the run must
        exist, the others are used when they exist."""
        entries = [entry] if isinstance(entry, str) else list(entry)
        end = self.time_after((self.n_steps - 1) * self.step_seconds)  # last step
        months, (y, m) = [], (self.start.year, self.start.month)
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
        while (y, m) <= ((end.year, end.month + 1) if end.month < 12 else (end.year + 1, 1)):
            months.append((y, m))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        run_months = set(months[1:-1])
        out = []
        for text in entries:
            if "{year" not in text and "{month" not in text:
                path = self.path(text.replace("{tower}", tower))
                if not path.exists():
                    raise ConfigError(f"{self.source}: {key}: file not found: {path}")
                out.append(path)
                continue
            for y, m in months:
                try:
                    name = text.format(tower=tower, year=y, month=m)
                except (KeyError, IndexError, ValueError) as err:
                    raise ConfigError(f"{self.source}: {key}: {text!r}: {err}") from None
                path = self.path(name)
                if path.exists():
                    out.append(path)
                elif (y, m) in run_months:
                    raise ConfigError(
                        f"{self.source}: {key}: file not found for {y}-{m:02d}: {path}"
                    )
        return list(dict.fromkeys(out))

    @property
    def output_directory(self) -> Path:
        return self.path(self.run_file.output.directory)

    @property
    def calkindflag(self) -> str:
        """The model calendar as the time manager names it (``GREGORIAN``, ``NOLEAP``)."""
        from clm_ml.calendar import CALENDARS

        return CALENDARS[self.run_file.time.calendar]

    def time_after(self, seconds: float) -> dt.datetime:
        """The date-time ``seconds`` after the run start, in the model calendar."""
        from clm_ml.calendar import add_seconds

        return add_seconds(self.start, seconds, self.calkindflag)

    @property
    def chunk_steps(self) -> Optional[int]:
        """Model steps per compiled chunk (``execution.chunk``; None: calendar months)."""
        days = parse_chunk(self.run_file.execution.chunk, "execution.chunk")
        return None if days is None else days * 86400 // self.step_seconds

    @property
    def step_seconds(self) -> int:
        """The model step (model.yaml numerics.step_seconds)."""
        return int(self.config.numerics.step_seconds)

    @property
    def restart_path(self) -> Path:
        """Where this run writes its restart (the previous one is kept as ``.prev``)."""
        return self.output_directory / "restart" / f"{self.name}_restart.nc"

    # -- running -----------------------------------------------------------------------

    def execute(
        self, *, write: bool = True, quiet: bool = False, return_outputs: bool = True
    ) -> dict:
        """Run the model in this process and (if ``write``) write the output streams configured
        in run.yaml:
        one netCDF file per stream, level and ``file_per`` period, each written as soon as
        its period is complete.

        Returns ``{stream name: Outputs}`` (all records in memory), or ``{}`` with
        ``return_outputs=False``, so long runs keep only the files' records in memory.
        """
        from clm_ml.outputs import Outputs

        out = self.run_file.output
        if write:  # the four files as read, a folder that repeats the run (decision 91)
            self.write_copy(self.output_directory / "config")
        arrays = self._run_in_process(write, return_outputs)

        if not return_outputs:
            return {}
        from clm_ml.outputs import domain_meta

        meta = dict(domain_meta(self.domain), soil_depth=arrays["static.soil_depth"])
        results = {}
        for stream in out.streams:
            prefix = f"{stream.name}/"
            records = {k[len(prefix) :]: v for k, v in arrays.items() if k.startswith(prefix)}
            results[stream.name] = Outputs.from_records(records, meta)
        return results

    def _run_in_process(self, write: bool, return_outputs: bool) -> dict:
        """Set the run up and run it in this process (``clm_ml.driver``), writing the output
        files as their periods complete. Returns the records as arrays
        (``"<stream>/<key>"``) and ``"static.soil_depth"``."""
        import time

        from clm_ml import driver
        from clm_ml.output_streams import StreamSpec
        from clm_ml.outputs import StreamFileWriter, domain_meta, records_to_arrays

        out = self.run_file.output
        ex = self.run_file.execution
        t0 = time.perf_counter()
        sim = driver.simulation(self).laid_out(
            int(ex.devices or 1), ex.patch_count, ex.gridcell_count
        )
        if self.is_grid:  # progress of long runs
            print(f"timing: setup {time.perf_counter() - t0:.1f} s ({sim.n_real} patches, "
                  f"{sim.n_real_gridcells} gridcells, layout {sim.layout})", flush=True)  # fmt: skip
        streams = [
            StreamSpec(st.name, st.frequency, st.statistic, self.stream_variables(st))
            for st in out.streams
        ]
        levels = [tuple(st.level) for st in out.streams]
        keys = [
            tuple(s.variables)
            + tuple(f"gridcell:{n}" for n in driver.gridcell_variables(s.variables, level))
            for s, level in zip(streams, levels)
        ]
        meta = dict(domain_meta(self.domain), soil_depth=sim.soil_depth)
        writers = None
        if write:
            writers = [
                StreamFileWriter(st, k, meta, self.output_directory, self.name, self.start)
                for st, k in zip(out.streams, keys)
            ]
        collected: list[list] = [[] for _ in streams]
        ascii_writer = None
        if write and out.legacy_ascii:  # one patch (validation): an internal per-step stream
            from clm_ml.io import legacy as legacy_io

            streams.append(StreamSpec("_legacy_ascii", "step", "instantaneous",
                                      legacy_io._variables()))  # fmt: skip
            levels.append(("patch",))
            interpolation = self.run_file.forcing.meteorology.interpolation
            ascii_writer = legacy_io.LegacyAsciiWriter(
                self.output_directory, self.legacy_tower_name(0), self.start.year,
                self.start.month, met_type=3 if interpolation == "three_point" else 0,
                step_seconds=self.step_seconds, soil_z=sim.soil_depth[0],
                append=self.run_file.initial_conditions.source == "restart",
            )  # fmt: skip

        def on_records(i, records):
            if i == len(keys):  # the legacy ASCII stream: patch 0, legacy patch axis
                for key, _end, values in records:
                    fields = {k: np.asarray(v)[0] for k, v in values.items()}  # patch 0
                    ascii_writer.write(sim.time_of(int(key)).calday, fields)
                return
            if writers is not None:
                writers[i].add(records)  # files are written as their periods complete
            if return_outputs:
                collected[i].extend(records)

        def save_extra():
            # the output files in progress; a restart holds what the next run needs, not the
            # records so far (review R6)
            return {"writers": [w.state() for w in writers] if writers is not None else None}

        def restore_extra(saved):
            if saved is not None and writers is not None and saved.get("writers") is not None:
                for w, ws in zip(writers, saved["writers"]):
                    w.restore(ws)

        ic = self.run_file.initial_conditions
        restart = None
        if ic.source == "restart":
            name = self.domain.gridcells.names[0]
            restart = driver.read_restart(self.path(ic.restart_file, name))
        every = parse_every(ex.restart_every, "restart_every")
        numerics = self.config.numerics
        try:
            driver.run_streams(
                sim,
                streams,
                on_records,
                restart=restart,
                restart_path=self.restart_path if write else None,
                restart_every=tuple(every) if every else None,
                save_extra=save_extra,
                restore_extra=restore_extra,
                levels=levels,
                on_patch_error=numerics.on_patch_error,
                max_failed_patch_fraction=numerics.max_failed_patch_fraction,
                ignore_checks=tuple(numerics.ignore_checks),
                chunk_steps=self.chunk_steps,
                timing=self.is_grid,
            )
        finally:  # also when the run stops: the files hold the records before the stop
            for w in writers or ():
                w.close()
            if ascii_writer is not None:
                ascii_writer.close()
        if self.is_grid:
            import jax

            stats = [dev.memory_stats() or {} for dev in jax.local_devices()]
            peak = [round(st.get("peak_bytes_in_use", 0) / 1e9, 2) for st in stats]
            print(f"timing: peak device memory GB {peak}", flush=True)
        arrays = {"static.soil_depth": sim.soil_depth}
        if return_outputs:
            for stream, stream_keys, records in zip(streams, keys, collected):
                if not records:
                    raise RuntimeError(f"output stream {stream.name} produced no records")
                for key, value in records_to_arrays(records, stream_keys).items():
                    arrays[f"{stream.name}/{key}"] = value
        return arrays

    def stream_variables(self, stream) -> tuple[str, ...]:
        """A stream's variable names, with ``all`` expanded."""
        from clm_ml.outputs import VARIABLES

        names = []
        for v in stream.variables:
            names += [x.name for x in VARIABLES] + ["soil_temperature"] if v == "all" else [v]
        return tuple(dict.fromkeys(names))

    @property
    def name(self) -> str:
        """The run's name in file names: its tower, or ``domain`` for several gridcells."""
        names = self.domain.gridcells.names
        return names[0] if len(names) == 1 else "domain"

    def legacy_tower_name(self, patch: int) -> str:
        """Patch ``patch``'s entry in the legacy tower table: its gridcell's name, with
        ``#k`` (its place on the gridcell) when the gridcell has several patches."""
        gidx = self.domain.patches.gridcell_index
        g = int(gidx[patch])
        name = self.domain.gridcells.names[g]
        if int(np.sum(gidx == g)) == 1:
            return name
        return f"{name}#{int(np.sum(gidx[:patch] == g))}"

    def write_copy(self, folder: Path) -> Path:
        """Write the four files as this experiment read them (``--set`` changes applied) to
        ``folder``, every path absolute: a folder that repeats the run."""
        return write_experiment(
            folder, self.run_file, self.domain_file, self.config, self.params,
            generated=f"Written by clm-ml {version()} for the run of {self.base_dir}"
            + (f" with {', '.join(self.overrides)}" if self.overrides else "") + ".",
            params_about=f"As read from {self.base_dir / 'params.yaml'}.",
            path=lambda p: str(self.path(p)), force=True,
        )  # fmt: skip

    def summary(self) -> str:
        """What ``clm-ml check`` prints: the values the run will use and where they come from
        (where one setting decides another, e.g. layer counts)."""
        from clm_ml.config import changed_options

        d, c, rf = self.domain, self.config, self.run_file
        end = self.time_after(self.n_steps * self.step_seconds)
        lines = [
            f"Experiment: {self.base_dir}",
            f"  run        {self.start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}: {self.n_steps} "
            f"steps of {self.step_seconds} s (time.length: {rf.time.length})",
        ]
        if not self.is_grid:
            met = rf.forcing.meteorology
            lines.append(f"  forcing    tower data every {met.interval_seconds} s, "
                         f"{met.interpolation} within each step")  # fmt: skip
        lines.append(f"  domain     {self.domain_file.type}: {d.n_gridcells} gridcell(s), "
                     f"{d.n_patches} patch(es)")  # fmt: skip
        layers = d.layer_counts_with_source(c)
        if not self.is_grid:
            p = 0
            for name, tower in self.domain_file.towers.items():
                soil, tex = tower.soil, tower.soil.texture
                texture = (
                    f"{tex.sand_percent:g} % sand, {tex.clay_percent:g} % clay (percentages)"
                    if tex.given_as == "percentages" else f"{tex.texture_class} (class)"
                )  # fmt: skip
                bedrock = "none" if soil.bedrock_depth is None else f"{soil.bedrock_depth:g} m"
                lines.append(
                    f"  tower {name}: {tower.latitude:g} N, {tower.longitude:g} E, forcing height "
                    f"{tower.measurement_height:g} m; soil {texture}, organic matter "
                    f"{soil.organic_matter:g} kg m-3, colour {soil.color_class}, bedrock {bedrock}"
                )
                for patch in tower.patches:
                    if patch.weight < self.domain_file.min_patch_weight:
                        continue
                    h = d.patches.canopy_height[p]
                    height = ("PFT default" if isinstance(patch.canopy_height, str)
                              else f"{h:g} m")  # fmt: skip
                    profiles = "PFT default" if isinstance(patch.area_profiles, str) else "given"
                    n, why = layers[p]
                    lines.append(
                        f"    patch {p}: {patch.pft}, weight {d.patches.weight[p]:g}, canopy "
                        f"height {height}, root biomass {patch.root_biomass:g} g m-2, area "
                        f"profiles {profiles}; {n} canopy layers ({why}; max_layers "
                        f"{c.canopy_layers.max_layers})"
                    )
                    p += 1
        else:
            for i, (n, why) in enumerate(layers[:20]):
                g = d.patches.gridcell_index[i]
                lines.append(
                    f"    patch {i}: {d.gridcells.names[g]}  {PFT_NAMES[d.patches.pft_index[i]]}  "
                    f"weight {d.patches.weight[i]:g}  {n} canopy layers ({why})"
                )
            if d.n_patches > 20:
                lines.append(f"    ... ({d.n_patches - 20} more patches)")
        changed = changed_options(c)
        lines.append("  model      " + ("differs from the defaults in: " + ", ".join(changed)
                                        if changed else "all settings at their defaults"))  # fmt: skip
        unvalidated = c.non_default_options()
        if unvalidated:
            lines.append(
                "  note: these options differ from the CLM-ml v2 defaults and have not been "
                f"validated against Fortran in the JAX translation: {', '.join(unvalidated)}"
            )
        lines.append("  OK")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_experiment(folder: Path | str, overrides: tuple[str, ...] | list[str] = ()) -> Experiment:
    """Read an experiment folder's four files and validate them.

    Every file must be complete (``clm-ml upgrade`` adds keys a newer version introduced).
    ``overrides`` change settings without editing the files, with the file named first, as
    on the command line (``clm-ml run --set``)::

        load_experiment("my_site", ["run.time.length=7 days",
                                    "model.processes.stomatal_conductance.scheme=medlyn",
                                    "params.broadleaf_deciduous_temperate_tree.vcmax25=100"])
    """
    folder = Path(folder).resolve()
    if folder.is_file():
        raise ConfigError(
            f"{folder}: give the experiment folder (with run.yaml, domain.yaml, model.yaml and "
            "params.yaml), not a file"
        )
    absent = [name for name in FILES.values() if not (folder / name).exists()]
    if absent:
        hint = (
            f"`clm-ml upgrade {folder}` converts this folder (the layout before experiment "
            "folders)"
            if (folder / "run.yaml").exists()
            else f"`clm-ml new {folder} --from COMPSET` writes a complete folder "
            f"(compsets: {', '.join(compset_names())})"
        )
        raise ConfigError(f"{folder}: missing {', '.join(absent)}; {hint}")
    data = {key: read_yaml(folder / name) for key, name in FILES.items()}
    _apply_overrides(data, overrides, folder)
    return _experiment(data, folder, tuple(overrides))


def _apply_overrides(data: dict, overrides, folder: Path) -> None:
    for item in overrides:
        key, _, _ = str(item).partition("=")
        prefix, _, rest = key.partition(".")
        if prefix not in FILES or not rest:
            raise ConfigError(
                f"--set {item!r}: name the file first: run., domain., model. or params. "
                "(e.g. run.time.length=3 days)"
            )
        value = str(item)[len(prefix) + 1 :]
        if prefix == "params":  # params.<pft>.<name>=value
            apply_overrides(data["params"].setdefault("pft", {}), [value], source="params.yaml")
        else:
            apply_overrides(data[prefix], [value], source=FILES[prefix])


def _strict(cls, data: dict, folder: Path, name: str, context: Optional[str],
            ignore_unused: bool = False):  # fmt: skip
    source = f"{folder.name}/{name}"
    try:
        return from_dict(cls, data, source=source, strict=True, context=context,
                         ignore_unused=ignore_unused)  # fmt: skip
    except MissingKeys as exc:
        raise ConfigError(
            f"{exc}. Every setting must be listed: `clm-ml upgrade {folder}` adds missing keys "
            "at their defaults"
        ) from None
    except ConfigError as exc:
        if "unknown key" in str(exc):
            raise ConfigError(
                f"{exc} (a folder written by another version? `clm-ml upgrade {folder}`)"
            ) from None
        raise


def _experiment(data: dict, folder: Path, overrides: tuple) -> Experiment:
    kind = (data["domain"] or {}).get("type")
    if kind not in ("towers", "grid"):
        raise ConfigError(f"{folder.name}/domain.yaml: type: {kind!r} is not one of [towers, grid]")
    # a --set change may make a setting of the file unused (another choice): ignored then
    unused = {k: any(str(o).startswith(k + ".") for o in overrides) for k in FILES}
    run_file = _strict(RunFile, data["run"], folder, "run.yaml", kind, unused["run"])
    domain_file = _strict(DomainFile, data["domain"], folder, "domain.yaml", kind,
                          unused["domain"])  # fmt: skip
    config = _strict(ModelConfig, data["model"], folder, "model.yaml", None, unused["model"])
    params = Params.from_data(data["params"], f"{folder.name}/params.yaml")
    source = f"{folder.name}/run.yaml"

    _check_supported(run_file, source)
    _check_model(config)
    model_step = int(config.numerics.step_seconds)
    if model_step <= 0:
        raise ConfigError(f"{folder.name}/model.yaml: numerics.step_seconds must be positive")
    substeps = model_step / config.numerics.substep_seconds
    if substeps != int(substeps):
        raise ConfigError(
            f"{folder.name}/model.yaml: numerics.substep_seconds "
            f"({config.numerics.substep_seconds:g}) must divide the model step ({model_step} s)"
        )
    start, n_steps = _resolve_time(run_file.time, source, model_step)
    if kind == "grid":
        domain = _grid_domain(run_file, domain_file, config, params, folder, start, model_step)
    else:
        domain = Domain.from_towers(
            domain_file.towers, params=params, config=config,
            min_patch_weight=domain_file.min_patch_weight,
        )  # fmt: skip
    _check_execution(run_file, domain, source)
    exp = Experiment(run_file, domain_file, config, params, domain, folder, start, n_steps,
                     overrides)  # fmt: skip
    if kind == "towers":
        _check_towers_run(exp, source, model_step)
    return exp


def _check_model(config: ModelConfig) -> None:
    ps = config.processes
    if ps.radiation.solar == "norman":
        raise ConfigError(
            "model.yaml: processes.radiation.solar: norman is not supported: its JAX translation "
            "was never validated and stops at the first CHATS7 step with a conservation error "
            "(plan, open issues of Phases 4-5); use two_stream"
        )
    if ps.stomatal_conductance.scheme == "wue" and ps.stomatal_conductance.wue_solver == "brent":
        raise ConfigError(
            "model.yaml: processes.stomatal_conductance.wue_solver: brent is not supported: the "
            "JAX model has no traceable Brent solver (Phase 7 decision C16; global_input.md "
            "issue 18); use bisection"
        )
    if ps.canopy_air.flux_profile_solution == "well_mixed":
        raise ConfigError(
            "model.yaml: processes.canopy_air.flux_profile_solution: well_mixed is not "
            "supported: Fortran stops for it too (legacy code; Phase 7 decision D8); use implicit"
        )


def _grid_domain(run_file, domain_file, config, params, folder, start, model_step) -> Domain:
    """The domain of a grid experiment, from the surface data (:mod:`clm_ml.grid`)."""
    from clm_ml.grid import load_grid_domain

    src = f"{folder.name}/run.yaml"
    if config.processes.photosynthesis.temperature_acclimation == "run_mean":
        raise ConfigError(
            f"{folder.name}/model.yaml: processes.photosynthesis.temperature_acclimation: "
            "run_mean is not available for grid runs (the Fortran whole-run mean needs all of a "
            "run's forcing before it starts)"
        )
    if run_file.output.legacy_ascii:
        raise ConfigError(f"{src}: output.legacy_ascii: not available for grid runs")

    def resolve(text):
        p = Path(str(text)).expanduser()
        return p if p.is_absolute() else (folder / p).resolve()

    return load_grid_domain(
        domain_file.grid, params=params, config=config, start=start, step_seconds=model_step,
        min_patch_weight=domain_file.min_patch_weight, resolve=resolve,
    )  # fmt: skip


def _check_execution(run_file: RunFile, domain: Domain, source: str) -> None:
    ex = run_file.execution
    for key, count, own in (
        ("patch_count", ex.patch_count, domain.n_patches),
        ("gridcell_count", ex.gridcell_count, domain.n_gridcells),
    ):
        if count is not None and count < own:
            raise ConfigError(f"{source}: execution.{key} ({count}) is less than {own}")
    if (ex.patch_count or domain.n_patches) > domain.n_patches and not (
        (ex.gridcell_count or domain.n_gridcells) > domain.n_gridcells
    ):
        raise ConfigError(
            f"{source}: execution.patch_count pads the patches: set gridcell_count above "
            f"{domain.n_gridcells} too (padding patches go on a padding gridcell)"
        )
    if ex.devices < 1:
        raise ConfigError(f"{source}: execution.devices must be at least 1")
    if ex.devices > 1 and (ex.patch_count or ex.gridcell_count):
        raise ConfigError(
            f"{source}: execution.patch_count/gridcell_count with devices > 1 is not "
            "supported (the device blocks are padded automatically)"
        )


def _check_towers_run(exp: Experiment, source: str, model_step: int) -> None:
    run_file, domain = exp.run_file, exp.domain
    if run_file.output.legacy_ascii and domain.n_patches > 1:
        raise ConfigError(
            f"{source}: output.legacy_ascii: the legacy .out files (validation) hold one "
            f"patch; this domain has {domain.n_patches}"
        )
    if run_file.time.calendar != "gregorian":
        raise ConfigError(
            f"{source}: time.calendar: {run_file.time.calendar} only for grid runs; tower "
            "forcing is a real (Gregorian) time series"
        )
    forcing = run_file.forcing
    interval = forcing.meteorology.interval_seconds
    if interval <= 0 or interval % model_step:
        raise ConfigError(
            f"model.yaml: numerics.step_seconds ({model_step}) must divide the forcing "
            f"interval (run.yaml forcing.meteorology.interval_seconds, {interval} s)"
        )
    for tower in domain.gridcells.names:  # every gridcell's forcing files ({tower} expanded)
        exp.files(forcing.meteorology.files, tower, "forcing.meteorology.files")
        for key in ("vegetation", "soil_water"):
            section = getattr(forcing, key)
            if section.source == "file":
                if not section.files:
                    raise ConfigError(f"{source}: forcing.{key}.files: required with source file")
                exp.files(section.files, tower, f"forcing.{key}.files")
        files = {"forcing.clm_history.file": forcing.clm_history.file}
        if forcing.soil_water.adjustment:
            files["forcing.soil_water.adjustment.file"] = forcing.soil_water.adjustment.file
        for key, text in files.items():
            if not exp.path(text, tower).exists():
                raise ConfigError(f"{source}: {key}: file not found: {exp.path(text, tower)}")
    met = {tuple(exp.files(forcing.meteorology.files, t)) for t in domain.gridcells.names}
    if run_file.execution.devices > 1 and len(met) > 1:
        # the device layout's padding slots can make a gridcell read another tower's forcing
        # (Phase 6 review R1); several towers on several devices is not a use case
        raise ConfigError(
            f"{source}: execution.devices > 1 with towers that have different meteorology "
            "files is not supported (one tower, or towers sharing their forcing)"
        )


def _check_supported(run_file: RunFile, source: str) -> None:
    ic = run_file.initial_conditions
    if ic.source == "restart" and not ic.restart_file:
        raise ConfigError(
            f"{source}: initial_conditions.restart_file: required with source restart"
        )
    parse_chunk(run_file.execution.chunk, f"{source}: execution.chunk")
    parse_every(run_file.execution.restart_every, f"{source}: execution.restart_every")
    names = [st.name for st in run_file.output.streams]
    if len(set(names)) != len(names):
        raise ConfigError(f"{source}: output.streams: stream names must be unique ({names})")
    if not names:
        raise ConfigError(f"{source}: output.streams: at least one stream is needed")
    from clm_ml.outputs import VARIABLES_BY_NAME, field_names

    # raw step fields (validation): "diagnostics.<name>", "state.<name>", "layers.<name>"
    raw = set(field_names())
    for st in run_file.output.streams:
        unknown = [
            v for v in st.variables
            if v not in ("all", "soil_temperature")
            and v not in VARIABLES_BY_NAME
            and v not in raw
        ]  # fmt: skip
        if unknown:
            raise ConfigError(
                f"{source}: output.streams[{st.name}].variables: unknown {unknown}; "
                "see clm_ml.outputs.VARIABLES"
            )


def _parse_datetime(text: str, key: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(str(text))
    except ValueError:
        raise ConfigError(f"run.yaml: {key}: {text!r} is not a date-time like 2007-05-01T00:00")


def _resolve_time(t: TimeSection, source: str, step_seconds: int) -> tuple[dt.datetime, int]:
    """Start and number of model steps (``step_seconds`` long), in the model calendar."""
    from clm_ml.calendar import CALENDARS, seconds_between

    calendar = CALENDARS[t.calendar]
    start = _parse_datetime(t.start, "time.start")
    kind, value = parse_length(t.length, f"{source}: time.length")
    try:
        if kind == "steps":
            n = value
            seconds_between(start, start, calendar)  # the start exists in the calendar
        else:
            seconds = (
                seconds_between(start, _parse_datetime(value, "time.length"), calendar)
                if kind == "until"
                else value * 86400.0
            )
            n = seconds / step_seconds
            if n != int(n):
                raise ConfigError(f"{source}: time: run length is not a whole number of steps")
    except ValueError as e:  # 29 February in the noleap calendar
        raise ConfigError(f"{source}: time: {e}")
    if int(n) < 1:
        raise ConfigError(f"{source}: time: the run must be at least one step long")
    return start, int(n)


# --------------------------------------------------------------------------------------
# Writing: complete, commented files
# --------------------------------------------------------------------------------------


def map_paths(obj, fn: Callable[[str], str]):
    """A copy of dataclass ``obj`` with every file path (fields declared ``path=True``)
    passed through ``fn``."""
    if dataclasses.is_dataclass(obj):
        changes = {}
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            if f.metadata.get("path") and value is not None:
                changes[f.name] = (
                    tuple(fn(v) for v in value) if isinstance(value, tuple) else fn(value)
                )
            elif value is not None:
                new = map_paths(value, fn)
                if new is not value:
                    changes[f.name] = new
        return dataclasses.replace(obj, **changes) if changes else obj
    if isinstance(obj, tuple) and obj and dataclasses.is_dataclass(obj[0]):
        return tuple(map_paths(v, fn) for v in obj)
    if isinstance(obj, dict):
        return {k: map_paths(v, fn) for k, v in obj.items()}
    return obj


def write_experiment(
    folder: Path | str,
    run_file: RunFile,
    domain_file: DomainFile,
    config: ModelConfig,
    params: Params,
    *,
    generated: str,
    params_about: str,
    path: Optional[Callable[[str], str]] = None,
    force: bool = False,
) -> Path:
    """Write the four complete files of an experiment to ``folder``: every setting with its
    explanation (``generated``: the line saying what wrote them; ``path``: applied to every
    file path). Refuses a folder that already holds experiment files unless ``force``."""
    from clm_ml.params_convert import params_table, params_text

    folder = Path(folder)
    existing = [n for n in FILES.values() if (folder / n).exists()]
    if existing and not force:
        raise ConfigError(f"{folder}: already holds {', '.join(existing)}")
    if path is not None:
        run_file, domain_file = map_paths(run_file, path), map_paths(domain_file, path)
    kind = domain_file.type
    texts = {
        "run.yaml": render(run_file, RUN_HEADER.format(generated=generated), context=kind),
        "domain.yaml": render(domain_file, DOMAIN_HEADER.format(generated=generated), context=kind),
        "model.yaml": model_yaml_text(config, generated=generated),
        "params.yaml": params_text(
            params_table(params),
            [
                "params.yaml: which plants: every parameter of every PFT 1-16 (the numbers "
                "gradients come back for). Part of an experiment folder (run.yaml, "
                "domain.yaml, model.yaml, params.yaml). Each patch uses the row of its PFT.",
                params_about,
                generated,
            ],
        ),
    }
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (folder / name).write_text(text)
    return folder


def relative_to(folder: Path, text: str) -> str:
    """``text`` (an absolute path, or a path relative to the current directory) relative to
    ``folder`` when it is nearby (at most three levels up), otherwise absolute."""
    p = Path(text).expanduser()
    p = p if p.is_absolute() else Path.cwd() / p
    rel = os.path.relpath(p, Path(folder).resolve())
    return str(p) if rel.startswith(os.path.join("..", "..", "..", "..")) else rel


def compose(recipe: dict, source: str) -> tuple[RunFile, DomainFile, ModelConfig, Params]:
    """The four objects of a compset recipe or a converted namelist (``run``, ``domain``,
    ``model`` settings that differ from the defaults; ``params`` a compset name or path;
    ``domain.towers`` a list of tower compsets or a mapping of towers)."""
    domain = dict(recipe.get("domain") or {})
    kind = domain.get("type")
    if kind == "towers" and isinstance(domain.get("towers"), list):
        towers = {}
        for name in domain["towers"]:
            compset = compsets_dir() / "towers" / f"{str(name).lower()}.yaml"
            if not compset.exists():
                raise ConfigError(
                    f"{source}: tower {name!r} has no compset (compsets: "
                    f"{', '.join(sorted(p.stem for p in compset.parent.glob('*.yaml')))}); "
                    "describe it in domain.yaml"
                )
            towers.update({k: v for k, v in read_yaml(compset).items()})
        domain["towers"] = towers
    domain_file = from_dict(DomainFile, domain, source=source, context=kind)
    run_file = from_dict(RunFile, recipe.get("run") or {}, source=source, context=kind)
    config = ModelConfig.from_dict(recipe.get("model") or {}, source=source)
    params = Params.from_yaml(recipe["params"])
    return run_file, domain_file, config, params


def new_experiment(
    folder: Path | str,
    compset: str,
    *,
    force: bool = False,
    machine: Optional[str] = None,
    device: str = "gpu",
    devices: Optional[int] = None,
    account: Optional[str] = None,
    walltime: str = "04:00:00",
) -> Path:
    """Write a complete experiment folder from a compset (``clm_ml/compsets/experiments``).

    ``devices`` (if given) is written to ``run.execution.devices``. On ``machine`` derecho the
    folder also gets ``submit.pbs``, a job running it on ``devices`` GPUs (``device`` gpu) or
    a CPU node, charged to ``account`` (required) (decision 112); other machines get none."""
    from clm_ml.batch import check_machine, write_job_script

    path = compsets_dir() / "experiments" / f"{compset}.yaml"
    if not path.exists():
        raise ConfigError(f"unknown compset {compset!r}; compsets: {', '.join(compset_names())}")
    machine = check_machine(machine, device, devices or 1, account)
    recipe = read_yaml(path)
    run_file, domain_file, config, params = compose(recipe, path.name)
    if devices is not None:
        execution = dataclasses.replace(run_file.execution, devices=devices)
        run_file = dataclasses.replace(run_file, execution=execution)
    folder = Path(folder)
    repo = Path(__file__).resolve().parents[2]  # compset paths starting data/: the repository

    def place(text: str) -> str:
        return relative_to(folder, str(repo / text)) if text.startswith("data/") else text

    write_experiment(
        folder, run_file, domain_file, config, params,
        generated=f"Generated by clm-ml {version()} (`clm-ml new {folder.name} --from {compset}`).",
        params_about=f"Copied from the compset {recipe['params']}.", path=place, force=force,
    )  # fmt: skip
    if machine is not None:
        write_job_script(folder, device=device, devices=devices or 1, account=account,
                         walltime=walltime)  # fmt: skip
    return folder


#: model.yaml settings renamed or moved (decision 93): old dotted key -> new dotted key
RENAMED_MODEL = {
    "physics.stomatal_conductance": "processes.stomatal_conductance.scheme",
    "physics.water_stress_on_conductance": "processes.stomatal_conductance.water_stress",
    "physics.wue_solver": "processes.stomatal_conductance.wue_solver",
    "physics.photosynthesis_colimitation": "processes.photosynthesis.colimitation",
    "physics.acclimation_days": "processes.photosynthesis.acclimation_days",
    "physics.roughness_sublayer": "processes.turbulence.roughness_sublayer",
    "physics.sparse_canopy": "processes.turbulence.sparse_canopy",
    "physics.ground_conductance": "processes.turbulence.ground_conductance",
    "physics.leaf_boundary_layer": "processes.turbulence.leaf_boundary_layer",
    "physics.flux_profile_solution": "processes.canopy_air.flux_profile_solution",
    "physics.solar_radiation": "processes.radiation.solar",
    "physics.longwave_radiation": "processes.radiation.longwave",
    "physics.leaf_optics_vary_with_height": "processes.radiation.leaf_optics_vary_with_height",
    "canopy_layers.min_plant_area_per_layer": "canopy_layers.merge_thin_layers_below",
    "canopy_layers.min_layer_plant_area": "canopy_layers.minimum_leafy_layer_plant_area",
}


def _migrate(key: str, data: dict, report: list) -> dict:
    """A file's contents in an earlier layout moved to the current one (decisions 91, 93):
    renamed keys, one key for each choice, named words for the old nulls."""
    name = FILES[key]
    data = dict(data or {})

    def note(text):
        report.append(f"{name}: {text}")

    if key == "model":
        physics = data.get("physics")
        if isinstance(physics, dict):
            on, mode = physics.pop("temperature_acclimation", True), physics.pop(
                "acclimation_temperature", "running_mean"
            )
            kn = physics.pop("nitrogen_decay_kn", None)
            photo = data.setdefault("processes", {}).setdefault("photosynthesis", {})
            photo["temperature_acclimation"] = mode if on else "none"
            photo["nitrogen_profile"] = "computed" if kn is None else kn
        for old, new in RENAMED_MODEL.items():
            if _pop_dotted(data, old, into=new):
                note(f"{old} -> {new}")
        if isinstance(physics, dict):
            if physics:
                raise ConfigError(f"{name}: physics: unknown key(s) {sorted(physics)}")
            data.pop("physics")
        layers = data.get("canopy_layers") or {}
        if "n_layers_within" in layers or "n_layers_above" in layers:
            w, a = layers.pop("n_layers_within", None), layers.pop("n_layers_above", None)
            layers["layer_counts"] = "from_spacing" if w is None else {"within": w, "above": a}
            note("canopy_layers.n_layers_within/n_layers_above -> canopy_layers.layer_counts")
    elif key == "run":
        time = data.get("time") or {}
        if "length" not in time and any(k in time for k in ("end", "days", "steps")):
            end, days, steps = (
                time.pop("end", None),
                time.pop("days", None),
                time.pop("steps", None),
            )
            given = [x for x in (end, days, steps) if x is not None]
            if len(given) != 1:
                raise ConfigError(f"{name}: time: give exactly one of end, days, steps")
            time["length"] = (f"until {end}" if end is not None else
                              f"{days:g} days" if days is not None else f"{steps} steps")  # fmt: skip
            note(f"time.end/days/steps -> time.length: {time['length']}")
        ex = data.get("execution") or {}
        if "restart_every" in ex and ex["restart_every"] is None:
            ex["restart_every"] = "end_of_run"
        co2 = ((data.get("forcing") or {}).get("gridded") or {}).get("co2")
        if isinstance(co2, dict) and "source" not in co2:
            co2["source"] = "file" if co2.get("file") else "constant"
    elif key == "domain":
        for tname, tower in (data.get("towers") or {}).items():
            if not isinstance(tower, dict):
                continue
            if tower.get("measurement_height", 0) is None:
                tower["measurement_height"] = 30.0
            soil = tower.get("soil") or {}
            if "texture" not in soil and any(
                k in soil for k in ("texture_class", "sand_percent", "clay_percent")
            ):
                cls_, sand, clay = (soil.pop(k, None) for k in
                                    ("texture_class", "sand_percent", "clay_percent"))  # fmt: skip
                soil["texture"] = (
                    {"given_as": "percentages", "sand_percent": sand, "clay_percent": clay}
                    if sand is not None and clay is not None
                    else {"given_as": "class", "texture_class": cls_}
                )
                note(f"towers.{tname}.soil -> texture given as {soil['texture']['given_as']}")
            if "organic_matter" in soil and soil["organic_matter"] is None:
                raise ConfigError(
                    f"{name}: towers.{tname}.soil.organic_matter: a value is required now (kg "
                    "m-3; a missing value was read as -999, giving a negative organic fraction: "
                    "open issue 22)"
                )
            for patch in tower.get("patches") or []:
                if "canopy_height" in patch and patch["canopy_height"] is None:
                    patch["canopy_height"] = "pft_default"
                lb, sb = patch.pop("leaf_area_profile_beta", None), patch.pop(
                    "stem_area_profile_beta", None
                )
                if lb is not None and sb is not None:
                    patch["area_profiles"] = {"leaf_beta": lb, "stem_beta": sb}
                elif lb is not None or sb is not None:
                    note(f"towers.{tname}: a leaf or stem profile given alone is not used "
                         "(Fortran needs both); area_profiles: pft_default")  # fmt: skip
                w, a = patch.pop("n_layers_within", None), patch.pop("n_layers_above", None)
                if w is not None:
                    patch["canopy_layers"] = {"within": w, "above": a}
        grid = data.get("grid")
        if isinstance(grid, dict) and "region" in grid and grid["region"] in (None, "whole_globe"):
            grid["region"] = "global"
    return data


def upgrade_experiment(folder: Path | str) -> list[str]:
    """Rewrite an experiment folder in the current layout, every setting with its current
    explanation: the values the user set are kept, keys added since are written at their
    defaults, renamed or regrouped keys moved (:func:`_migrate`); a folder in the layout before experiment
    folders (run.yaml naming a params compset or file, towers and a model.yaml) becomes the four
    files. The old files are kept in ``folder/.previous/``. Returns what was done."""
    import shutil

    folder = Path(folder).resolve()
    if not (folder / "run.yaml").exists():
        raise ConfigError(f"{folder}: no run.yaml (`clm-ml new {folder} --from COMPSET`)")
    run_data = read_yaml(folder / "run.yaml")
    old_layout = not (folder / "domain.yaml").exists() or bool(
        {"model", "params", "domain"} & set(run_data)
    )
    report: list[str] = []
    if old_layout:
        recipe, previous = _from_old_layout(folder, run_data, report)
    else:
        recipe, previous = {}, list(FILES.values())
        for key, name in FILES.items():
            recipe[key] = read_yaml(folder / name)
    for key in ("run", "domain", "model"):
        recipe[key] = _migrate(key, recipe[key], report)
    kind = (recipe["domain"] or {}).get("type")
    if kind not in ("towers", "grid"):
        raise ConfigError(f"{folder.name}/domain.yaml: type: {kind!r} is not one of [towers, grid]")
    source = folder.name
    domain_file = from_dict(DomainFile, recipe["domain"], source=f"{source}/domain.yaml",
                            context=kind)  # fmt: skip
    run_file = from_dict(RunFile, recipe["run"], source=f"{source}/run.yaml", context=kind)
    config = from_dict(ModelConfig, recipe["model"], source=f"{source}/model.yaml")
    params = (
        recipe["params"]
        if isinstance(recipe["params"], Params)
        else Params.from_data(recipe["params"], f"{source}/params.yaml")
    )
    for key, obj in (("run", run_file), ("domain", domain_file), ("model", config)):
        added = _missing(obj, recipe[key], kind)
        if added:
            report.append(f"{FILES[key]}: added at their defaults: {', '.join(added)}")
    backup = folder / ".previous"
    backup.mkdir(exist_ok=True)
    for name in previous:
        if (folder / name).exists():
            shutil.copy2(folder / name, backup / name)
    write_experiment(
        folder, run_file, domain_file, config, params,
        generated=f"Upgraded by clm-ml {version()} (`clm-ml upgrade`).",
        params_about=recipe.get("params_about", "As it was in this folder."), force=True,
    )  # fmt: skip
    report.append(f"wrote the four files of {folder} (old files in {backup})")
    return report


def _pop_dotted(data: dict, old: str, *, into: str) -> bool:
    parts = old.split(".")
    node = data
    for part in parts[:-1]:
        node = node.get(part)
        if not isinstance(node, dict):
            return False
    if parts[-1] not in node:
        return False
    value = node.pop(parts[-1])
    target = data
    for part in into.split(".")[:-1]:
        target = target.setdefault(part, {})
    target[into.split(".")[-1]] = value
    return True


def _missing(obj, data, context: Optional[str], key: str = "") -> list[str]:
    """Dotted keys of ``obj``'s settings absent from ``data`` (what upgrade adds)."""
    out = []
    data = data if isinstance(data, dict) else {}
    for f in dataclasses.fields(obj):
        if f.metadata.get("hidden"):
            continue
        want = f.metadata.get("applies")
        if want is not None and want != context:
            continue
        sub = f"{key}.{f.name}" if key else f.name
        value = getattr(obj, f.name)
        if f.name not in data:
            out.append(sub)
        elif dataclasses.is_dataclass(value):
            out += _missing(value, data[f.name], context, sub)
    return out


def _from_old_layout(folder: Path, run_data: dict, report: list[str]) -> tuple[dict, list]:
    """The four files' contents from a folder in the layout before experiment folders."""
    run_data = dict(run_data)
    model_ref = run_data.pop("model", None)
    model_data = read_yaml(folder / model_ref) if model_ref else {}
    params_ref = run_data.pop("params", "clm5_defaults")
    if str(params_ref).endswith((".yaml", ".yml")) and not Path(params_ref).is_absolute():
        params_ref = str(folder / params_ref)
    params = Params.from_yaml(params_ref)
    old_domain = dict(run_data.pop("domain", None) or {"type": "towers"})
    kind = old_domain.get("type", "towers")
    time = dict(run_data.get("time") or {})
    data_step = time.pop("step_seconds", 1800)
    run_data["time"] = time
    numerics = model_data.setdefault("numerics", {})
    if numerics.get("step_seconds") is None:
        numerics["step_seconds"] = int(data_step)
    domain = {"type": kind, "min_patch_weight": old_domain.get("min_patch_weight", 0.0)}
    if kind == "towers":
        available = {}
        for compset in sorted((compsets_dir() / "towers").glob("*.yaml")):
            available.update(read_yaml(compset))
        if old_domain.get("towers_file"):
            available.update(read_yaml(folder / old_domain["towers_file"]))
        names = list(old_domain.get("towers") or [])
        unknown = [n for n in names if n not in available]
        if unknown:
            raise ConfigError(f"{folder.name}/run.yaml: domain.towers: unknown tower(s) {unknown}")
        towers = {}
        for name in names:
            entry = dict(available[name])
            minutes = entry.pop("forcing_step_minutes", None)
            if minutes is not None and int(minutes) * 60 != int(data_step):
                raise ConfigError(
                    f"{folder.name}: tower {name} has forcing_step_minutes {minutes} but "
                    f"time.step_seconds is {data_step}"
                )
            towers[name] = entry
        domain["towers"] = towers
        forcing = run_data.setdefault("forcing", {})
        met = forcing.setdefault("meteorology", {})
        met.setdefault("interval_seconds", int(data_step))
    else:
        domain["grid"] = old_domain.get("grid")
    towers_file = old_domain.get("towers_file")
    moved = "numerics.step_seconds"
    if kind == "towers":
        moved += " and forcing.meteorology.interval_seconds"
    report.append(
        f"converted from the layout before experiment folders: params from {params.source}, "
        f"domain from run.yaml domain"
        + (f" and {towers_file}" if towers_file else "")
        + f", model from {model_ref or 'the defaults'}; time.step_seconds -> {moved}"
    )
    recipe = {"run": run_data, "domain": domain, "model": model_data, "params": params,
              "params_about": f"Copied from {params.source} by `clm-ml upgrade`."}  # fmt: skip
    previous = ["run.yaml"] + ([model_ref] if model_ref else [])
    return recipe, previous
