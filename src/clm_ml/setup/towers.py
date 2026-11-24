"""The forcing of tower runs at setup: one :class:`TowerForcingSource` per
gridcell (its tower meteorology files and CLM history file), and the first step's values
the setup needs (meteorology, initial soil temperature, acclimation temperature).

Fortran reference: CLM-ml v2 ``CLMml_driver.F90`` (``CLMml_drv`` up to its time loop:
the history file's time slices, ``init_acclim``, ``SoilInit``).
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np

from clm_ml.experiment import _parse_datetime
from clm_ml.tower_forcing import TowerForcingSource


def history_index_function(run_start: dt.datetime, step_seconds: int, clm_start, history_seconds):
    """``k -> 1-based CLM history time slice of 0-based run step k``: the slice whose
    interval contains the step's end (slice ``s`` ends at the history start + ``s``
    intervals)."""

    def index(run_step: int) -> int:
        step_end = run_start + dt.timedelta(seconds=(run_step + 1) * step_seconds)
        elapsed = (step_end - clm_start).total_seconds()
        return int(-(-elapsed // history_seconds))  # ceil

    return index


def gridcell_sources(run) -> list:
    """The :class:`TowerForcingSource` of each gridcell of a tower run
    (``clm_ml.experiment.Experiment``)."""
    from clm_ml.io.clm_history import history_interval
    from clm_ml.io.tower import TowerFiles

    f = run.run_file.forcing
    met = f.meteorology
    clm_start = _parse_datetime(f.clm_history.start, "forcing.clm_history.start")
    adjust = f.soil_water.adjustment
    g = run.domain.gridcells
    step = run.step_seconds
    sources = []
    record_seconds = int(met.interval_seconds)
    for i, name in enumerate(g.names):
        if record_seconds % step:
            raise ValueError(
                f"model step {step} s must divide the forcing interval {record_seconds} s"
            )
        tower_files = [str(p) for p in run.files(met.files, name, "forcing.meteorology")]
        history = str(run.path(f.clm_history.file, name))

        def land_files(key, name=name):
            section = getattr(f, key)
            if section.source != "file":
                return None
            return tuple(str(p) for p in run.files(section.files, name, f"forcing.{key}"))

        series = TowerFiles.open(tower_files, record_seconds)
        height = float(g.measurement_height[i])
        sources.append(
            TowerForcingSource(
                tower_file=series,
                clm_history_file=history,
                soil_adjust_file=(
                    str(run.path(adjust.file, name)) if adjust and adjust.levels > 0 else None
                ),
                first_record=series.first_record(run.start),
                n_steps=run.n_steps,
                history_index=history_index_function(
                    run.start, step, clm_start, history_interval(history)
                ),
                tower_height=-999.0 if math.isnan(height) else height,
                record_seconds=record_seconds,
                step_seconds=step,
                run_start=run.start,
                lat=float(g.latitude[i]),
                lon=float(g.longitude[i]),
                vegetation_file=land_files("vegetation"),
                soil_water_file=land_files("soil_water"),
                calendar=run.calkindflag,
                shortwave=met.shortwave.strip(),
                precipitation=met.precipitation.strip(),
            )
        )
    return sources


def acclimation_temperature(run, source, first_tref: float) -> float:
    """The initial acclimation temperature of a gridcell (K): the first step's air
    temperature (``running_mean``), or the mean over the whole run, read chunk
    by chunk (``run_mean``, Fortran ``init_acclim``; validation only)."""
    if run.config.processes.photosynthesis.temperature_acclimation != "run_mean":
        return (0.0 + float(first_tref)) / float(1)
    total, n, chunk = 0.0, run.n_steps, 1000
    for first in range(0, n, chunk):
        forcing = source.chunk(first, min(chunk, n - first))
        for t in range(forcing.n_steps):
            total = total + float(forcing.tower.air_temperature[t + 1, 0])
    return total / float(n)


def first_values(run, sources) -> dict:
    """Per gridcell (G,) the first step's forcing (``uref``, ``tref``, ``qref``, ``pref``,
    ``pco2``), its initial soil temperature ``tsoi`` (G, nlevgrnd; the CLM history's
    ``TSOI`` at the first step's time slice) and acclimation temperature ``tacclim``."""
    from clm_ml.setup.soil import NLEVGRND, initial_soil_temperature

    out: dict = {k: [] for k in ("uref", "tref", "qref", "pref", "pco2", "tsoi", "tacclim")}
    for source in sources:
        tower = source.chunk(0, 1).tower
        row = {
            "uref": tower.wind_speed[1, 0],
            "tref": tower.air_temperature[1, 0],
            "qref": tower.specific_humidity[1, 0],
            "pref": tower.air_pressure[1, 0],
            "pco2": tower.co2_partial_pressure[1, 0],
        }
        for k, v in row.items():
            out[k].append(float(v))
        out["tsoi"].append(
            initial_soil_temperature(source.clm_history_file, source.history_index(0), NLEVGRND)
        )
        out["tacclim"].append(acclimation_temperature(run, source, row["tref"]))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}
