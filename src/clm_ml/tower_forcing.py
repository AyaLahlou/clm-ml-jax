"""Tower forcing read chunk by chunk: the tower meteorology files and the CLM history file of
one gridcell as :class:`clm_ml.forcing.Forcing` chunks (one halo record at each end),
interpolated to the model step when it is shorter than the data interval, with user land-data
files in place of the history's leaf/stem area or soil water when given.

Fortran reference: CLM-ml v2 ``TowerMetMod.F90``, ``clmDataMod.F90`` (readers);
``CLMml_driver.F90`` (the time loop's reads).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Callable, Optional

import numpy as np

from clm_ml.forcing import ClmHistoryForcing, Forcing, TowerForcing, chunk_run_steps
from clm_ml.io.clm_history import read_clm_history_series
from clm_ml.io.tower import as_tower_files, derive_tower_met


def load_forcing(
    tower_file,
    clm_history_file: str,
    soil_adjust_file: str,
    *,
    first_record: int,
    run_n_steps: int,
    first_step: int,
    n_steps: int,
    history_index: Callable[[int], int],
    tower_height: float,
) -> Forcing:
    """Read and derive one chunk of forcing for one tower (one gridcell).

    Args:
        tower_file, clm_history_file, soil_adjust_file: Input files (``tower_file``: a
            path, or a :class:`TowerMetMod.TowerFiles` series split in time).
        first_record: 0-based record of ``tower_file`` at the run's first step
            (``TowerMetMod.tower_first_record``).
        run_n_steps: Steps in the whole run (to know where the run ends).
        first_step, n_steps: The chunk, in 0-based run steps.
        history_index: 1-based CLM history time slice for a 0-based run step.
        tower_height: Forcing height from the tower table (m).
    """
    run_steps = chunk_run_steps(run_n_steps, first_step, n_steps)
    raws = as_tower_files(tower_file).read([first_record + k for k in run_steps])
    mets = [derive_tower_met(raw, tower_height) for raw in raws]
    slices = [history_index(k) for k in run_steps]
    return _forcing(mets, slices, clm_history_file, soil_adjust_file, first_step)


def _forcing(mets, slices, clm_history_file, soil_adjust_file, first_step):
    """A chunk from per-step derived meteorology and CLM history time slices."""

    def col(name):
        return np.array([[getattr(m, name)] for m in mets], dtype=float)

    tower = TowerForcing(
        wind_speed=col("uref"),
        air_temperature=col("tref"),
        specific_humidity=col("qref"),
        air_pressure=col("pref"),
        longwave_down=col("lwsky"),
        rainfall=col("rain"),
        forcing_height=col("hgt"),
        co2=col("co2_ppm"),
        co2_partial_pressure=col("pco2"),
        o2_partial_pressure=col("po2"),
        shortwave_direct=np.array([[m.solad] for m in mets], dtype=float),
        shortwave_diffuse=np.array([[m.solai] for m in mets], dtype=float),
    )

    records = read_clm_history_series(clm_history_file, slices, soil_adjust_file)
    steps = [records[i] for i in slices]
    factors = [r.h2osoi_factor for r in steps]
    clm = ClmHistoryForcing(
        history_index=np.asarray(slices, dtype=int),
        leaf_area_index=np.array([[r.elai] for r in steps], dtype=float),
        stem_area_index=np.array([[r.esai] for r in steps], dtype=float),
        soil_water=np.array([[np.asarray(r.h2osoi)] for r in steps], dtype=float),
        soil_water_factor=(
            None if factors[0] is None else np.array([[f] for f in factors], dtype=float)
        ),
    )
    return Forcing(tower=tower, clm_history=clm, first_step=first_step)


@dataclasses.dataclass(frozen=True)
class TowerForcingSource:
    """:class:`clm_ml.forcing.ForcingSource` for one tower: the tower meteorology file and
    the CLM history file (leaf/stem area, soil water), read chunk by chunk with
    :func:`load_forcing`."""

    tower_file: object  # a path or a TowerMetMod.TowerFiles series
    clm_history_file: str
    soil_adjust_file: Optional[str]  # None: no soil-water adjustment factor
    first_record: int  # 0-based tower-file record at the run's first step
    n_steps: int  # steps in the whole run
    history_index: Callable[[int], int]
    tower_height: float
    # The model step and the data interval: equal -> records are steps;
    # otherwise the records are interpolated to the model step.
    record_seconds: int = 1800
    step_seconds: int = 1800
    run_start: Optional[dt.datetime] = None  # needed when interpolating
    lat: float = 0.0  # degrees, for the solar zenith angle (coszen interpolation)
    lon: float = 0.0
    # Prescribed land data from user files instead of the CLM history file:
    # daily or monthly records, linear between interval midpoints. None: CLM history.
    vegetation_file: Optional[str] = None  # LAI, SAI (m2 m-2)
    soil_water_file: Optional[str] = None  # H2OSOI (m3 m-3), the model's soil layers
    # (several files split in time may be given as a tuple, for both)
    calendar: str = "GREGORIAN"  # the model calendar (clm_ml.calendar)
    shortwave: str = "coszen"  # data interval -> model step: coszen | linear | constant
    precipitation: str = "constant"  # constant | linear

    def chunk(self, first_step: int, n_steps: int) -> Forcing:
        forcing = self._chunk(first_step, n_steps)
        if self.vegetation_file or self.soil_water_file:
            forcing = _with_land_data(self, forcing, first_step, n_steps)
        return forcing

    def _chunk(self, first_step: int, n_steps: int) -> Forcing:
        if self.step_seconds != self.record_seconds:
            return _interpolated_chunk(self, first_step, n_steps)
        return load_forcing(
            self.tower_file,
            self.clm_history_file,
            self.soil_adjust_file,
            first_record=self.first_record,
            run_n_steps=self.n_steps,
            first_step=first_step,
            n_steps=n_steps,
            history_index=self.history_index,
            tower_height=self.tower_height,
        )


_COSZ_FUNCTIONS: dict = {}  # (lat, lon, year) -> mean_cosz function


def _interpolated_chunk(self: TowerForcingSource, first_step: int, n_steps: int) -> Forcing:
    """The chunk at the model step, interpolated from the data records
    (:mod:`clm_ml.forcing_interpolation`), then derived step by step."""
    import math

    from clm_ml.forcing_interpolation import interpolate_records, mean_cosz_function
    from clm_ml.io.tower import TowerMetRaw

    run_steps = chunk_run_steps(self.n_steps, first_step, n_steps)
    step = dt.timedelta(seconds=self.step_seconds)
    starts = [self.run_start + k * step for k in run_steps]
    per = self.record_seconds // self.step_seconds
    # records covering the steps, plus one on each side for linear interpolation
    lo = max(min(run_steps) // per - 1, -self.first_record)
    files = as_tower_files(self.tower_file, self.record_seconds)
    hi = min(max(run_steps) // per + 1, files.n_records - 1 - self.first_record)
    records = list(range(lo, hi + 1))
    raws = files.read([self.first_record + j for j in records])
    rec = dt.timedelta(seconds=self.record_seconds)
    rec_starts = [self.run_start + j * rec for j in records]
    values = {f: np.array([getattr(r, f) for r in raws], dtype=float) for f in TowerMetRaw._fields}
    key = (self.lat, self.lon, self.run_start.year, self.calendar)
    if key not in _COSZ_FUNCTIONS:
        _COSZ_FUNCTIONS[key] = mean_cosz_function(
            math.radians(self.lat), math.radians(self.lon), self.run_start.year,
            calendar=self.calendar,
        )  # fmt: skip
    at_step = interpolate_records(
        values,
        rec_starts,
        self.record_seconds,
        starts,
        self.step_seconds,
        _COSZ_FUNCTIONS[key],
        rules={
            "fsdsbot": self.shortwave,
            "fsds_direct": self.shortwave,
            "fsds_diffuse": self.shortwave,
            "prect": self.precipitation,
        },
    )
    mets = [
        derive_tower_met(
            TowerMetRaw(**{f: float(at_step[f][i]) for f in at_step}), self.tower_height
        )
        for i in range(len(starts))
    ]
    slices = [self.history_index(k) for k in run_steps]
    return _forcing(mets, slices, self.clm_history_file, self.soil_adjust_file, first_step)


def _with_land_data(
    self: TowerForcingSource, forcing: Forcing, first_step: int, n_steps: int
) -> Forcing:
    """Replace the CLM-history leaf/stem area and/or soil water of a chunk with values
    interpolated from the user's land-data files (the CHATS soil-moisture adjustment
    factor, if any, still multiplies the soil water)."""
    from clm_ml.forcing_interpolation import linear_between_midpoints, read_land_data

    run_steps = chunk_run_steps(self.n_steps, first_step, n_steps)
    step = dt.timedelta(seconds=self.step_seconds)
    starts = [self.run_start + k * step for k in run_steps]
    clm = forcing.clm_history

    def at_steps(path, names):
        rec_starts, values, end = read_land_data(path, names, starts[0], starts[-1] + step)
        return {n: linear_between_midpoints(rec_starts, values[n], starts, self.step_seconds, end)
                for n in names}  # fmt: skip

    changes = {}
    if self.vegetation_file:
        v = at_steps(self.vegetation_file, ("LAI", "SAI"))
        changes["leaf_area_index"] = v["LAI"].reshape(clm.leaf_area_index.shape)
        changes["stem_area_index"] = v["SAI"].reshape(clm.stem_area_index.shape)
    if self.soil_water_file:
        h2osoi = at_steps(self.soil_water_file, ("H2OSOI",))["H2OSOI"]
        nlev = clm.soil_water.shape[-1]  # nlevsoi
        if h2osoi.ndim != 2 or h2osoi.shape[1] < nlev:
            raise ValueError(
                f"{self.soil_water_file}: H2OSOI needs (time, level) with at least {nlev} "
                f"levels; got {h2osoi.shape}"
            )
        changes["soil_water"] = h2osoi[:, None, :nlev]
    return dataclasses.replace(forcing, clm_history=dataclasses.replace(clm, **changes))
