"""CLM history files of tower runs: leaf and stem area index, volumetric soil water, the
optional soil-water adjustment factor and the initial soil temperature.

The records a run needs are read before its time loop, chunk by chunk
(:func:`read_clm_history_series`). The history must have the CLM soil levels (``nlevsoi``
levels of ``H2OSOI``).

Fortran reference: CLM-ml v2 ``clmDataMod.F90`` (``readCLMveg``, ``readCLMsoil``,
``readSoilWatFactor``) and ``CLMml_driver.F90::SoilInit``.
"""

from __future__ import annotations

from typing import Dict, Iterable, NamedTuple, Optional

import netCDF4 as nc
import numpy as np

from clm_ml.io import netcdf
from clm_ml.setup.soil import NLEVSOI


class ClmHistoryRecord(NamedTuple):
    """One time slice of the CLM history (and soil adjustment) files, as read."""

    elai: float  # leaf area index (m2/m2)
    esai: float  # stem area index (m2/m2)
    h2osoi: np.ndarray  # (nlevsoi,) volumetric soil water (m3/m3)
    h2osoi_factor: Optional[float]  # soil water adjustment factor (None if not used)


def read_clm_history_series(
    path: str,
    time_slices: Iterable[int],
    soil_adjust_path: Optional[str] = None,
) -> Dict[int, ClmHistoryRecord]:
    """The CLM history records of the given 1-based time slices: ``ELAI``, ``ESAI``,
    ``H2OSOI`` (the ``nlevsoi`` soil levels) and, with ``soil_adjust_path``, the
    adjustment factor ``FACTOR``."""
    records = {}
    netcdf.LOCK.acquire()  # the adjustment file stays open while the history is read
    adjust = nc.Dataset(soil_adjust_path, "r") if soil_adjust_path else None
    try:
        with netcdf.dataset(path, "r") as ds:
            n_time = ds.dimensions["time"].size
            levels = ds.variables["H2OSOI"].shape[1]
            if levels != NLEVSOI:
                raise ValueError(
                    f"{path}: H2OSOI has {levels} soil levels; the CLM soil layer structure "
                    f"needs a history with {NLEVSOI}"
                )
            for strt in sorted(set(time_slices)):
                if not 1 <= strt <= n_time:
                    raise ValueError(
                        f"{path}: time slice {strt} outside 1..{n_time} (does the run start "
                        "before forcing.clm_history.start or end after the file?)"
                    )
                t = strt - 1
                elai = float(ds.variables["ELAI"][t, 0])
                esai = float(ds.variables["ESAI"][t, 0])
                h2osoi = np.array(ds.variables["H2OSOI"][t, :NLEVSOI, 0], dtype=np.float64)
                factor = float(adjust.variables["FACTOR"][t]) if adjust is not None else None
                records[strt] = ClmHistoryRecord(elai, esai, h2osoi, factor)
    finally:
        if adjust is not None:
            adjust.close()
        netcdf.LOCK.release()
    return records


def history_interval(path: str) -> int:
    """The interval of a CLM history file's time axis (s)."""
    with netcdf.dataset(path) as ds:
        t = ds.variables["time"]
        return int(round(float(t[1] - t[0]) * 86400.0))
