"""
Forcing: one chunk of everything prescribed from outside the model.

A run reads its forcing a chunk at a time, so memory depends on
the chunk length, not the run length. A chunk of ``n_steps`` model steps holds
``n_steps + 2`` records along its time axis: one **halo** record before the first step, the
steps themselves, and one halo record after the last step. The halos are the neighbouring
records the three-point time interpolation needs at the chunk edges. At the very start and
end of a run there is no neighbour, so the halo repeats the edge record, as the Fortran
driver does (``*_bef`` seeded with the current forcing; ``itim_next = min(itim+1, ntim)``).

Meteorology is per gridcell, shape ``(time, gridcell)``; CLM history data (leaf/stem area
and soil water) is per patch/column, shape ``(time, patch)`` / ``(time, column, layer)``.
Wavebands are ``(vis, nir)``.

``Forcing`` holds arrays only: no files are opened while a chunk runs. It is built by
:func:`clm_ml.tower_forcing.load_forcing`, which applies the same derivations as the
Fortran ``TowerMetMod`` (humidity from relative humidity, longwave from emissivity, solar
partitioning).
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Protocol

import numpy as np


@dataclasses.dataclass(frozen=True)
class TowerForcing:
    """Meteorology at the forcing height, per gridcell. Shape ``(n_steps + 2, gridcell)``."""

    wind_speed: np.ndarray  # m s-1
    air_temperature: np.ndarray  # K
    specific_humidity: np.ndarray  # kg kg-1
    air_pressure: np.ndarray  # Pa
    longwave_down: np.ndarray  # W m-2
    rainfall: np.ndarray  # kg H2O m-2 s-1 (= mm s-1)
    forcing_height: np.ndarray  # m
    co2: np.ndarray  # umol mol-1
    co2_partial_pressure: np.ndarray  # Pa
    o2_partial_pressure: np.ndarray  # Pa
    shortwave_direct: np.ndarray  # (time, gridcell, 2) W m-2
    shortwave_diffuse: np.ndarray  # (time, gridcell, 2) W m-2


@dataclasses.dataclass(frozen=True)
class ClmHistoryForcing:
    """Prescribed land state from the CLM history file, per patch / soil column."""

    history_index: np.ndarray  # (n_steps + 2,) int, 1-based time slice in the CLM history file
    leaf_area_index: np.ndarray  # (time, patch) m2 m-2
    stem_area_index: np.ndarray  # (time, patch) m2 m-2
    soil_water: np.ndarray  # (time, column, nlevsoi) m3 m-3
    soil_water_factor: Optional[np.ndarray]  # (time, column) adjustment factor, or None


@dataclasses.dataclass(frozen=True)
class Forcing:
    """One chunk: ``n_steps`` model steps plus a halo record at each end."""

    tower: TowerForcing
    clm_history: ClmHistoryForcing
    first_step: int  # 0-based run step of the chunk's first step

    @property
    def n_steps(self) -> int:
        """Model steps in the chunk (the two halo records excluded)."""
        return self.tower.air_temperature.shape[0] - 2

    def window_indices(self, t: int) -> tuple[int, int, int]:
        """Array indices of the (previous, current, next) records for chunk step ``t``
        (0-based). The halos make these simply ``t``, ``t + 1``, ``t + 2``."""
        if not 0 <= t < self.n_steps:
            raise IndexError(f"chunk step {t} outside 0..{self.n_steps - 1}")
        return t, t + 1, t + 2


def chunk_run_steps(run_n_steps: int, first_step: int, n_steps: int) -> list[int]:
    """0-based run steps of a chunk's records: ``[halo before, steps..., halo after]``.

    Inside a run the halos are the neighbouring steps. At the run's first and last steps
    there is no neighbour, so the halo repeats the edge step (Fortran: ``*_bef`` seeded
    with the current forcing; ``itim_next = min(itim+1, ntim)``).
    """
    if n_steps < 1 or first_step < 0 or first_step + n_steps > run_n_steps:
        raise ValueError(
            f"chunk of {n_steps} steps from step {first_step} is outside a run of "
            f"{run_n_steps} steps"
        )
    steps = list(range(first_step, first_step + n_steps))
    return [max(first_step - 1, 0), *steps, min(first_step + n_steps, run_n_steps - 1)]


class ForcingSource(Protocol):
    """Where a run's forcing comes from: any object that returns one chunk.

    ``chunk(first_step, n_steps)`` returns the :class:`Forcing` for run steps
    ``first_step ... first_step + n_steps - 1`` with one halo record at each end, already at
    the model step. Implementations read only the files covering those times. A run asks
    for consecutive chunks; the next one is read in the background while the current one
    runs. Implemented by :class:`clm_ml.tower_forcing.TowerForcingSource` (tower meteorology
    and the CLM history file, as CHATS7); readers for coarser streams (6-hourly meteorology,
    daily or monthly LAI, SAI and soil water, interpolated to the model step) implement the
    same interface.
    """

    n_steps: int  # steps in the whole run

    def chunk(self, first_step: int, n_steps: int) -> Forcing: ...
