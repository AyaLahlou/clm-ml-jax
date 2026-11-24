"""Tower meteorology files.

The tower records a run (or a chunk of it) needs are read before its time loop
(:func:`read_tower_met_series`, :class:`TowerFiles` for forcing split in time);
:func:`derive_tower_met` derives the missing quantities (specific humidity, longwave
radiation) and partitions solar radiation into direct and diffuse components per waveband,
with the arithmetic of Fortran ``TowerMetCurr``/``TowerMetNext``.

Fortran reference: CLM-ml v2 ``TowerMetMod.F90``.
"""

from __future__ import annotations

import dataclasses
import datetime
from typing import List, NamedTuple, Sequence, Tuple

import jax.numpy as jnp

from clm_ml.io import netcdf
from clm_ml.processes.constants import mmdry, mmh2o, sb
from clm_ml.processes.water_vapor import sat_vap


class TowerMetError(ValueError):
    """A tower forcing file that cannot be used."""


def endrun(msg: str):
    raise TowerMetError(msg.strip())


def handle_err(_status, name: str):
    raise TowerMetError(f"tower forcing file: variable {name} not found")


def TowerMetCO2() -> float:
    """
    Return the atmospheric CO2 concentration (ppm).

    Mirrors Fortran function ``TowerMetCO2`` (lines 35-48).
    The commented-out alternative value of 367 ppm is preserved.

    Returns:
        Atmospheric CO2 concentration in ppm (umol/mol).
    """
    #   return 367.0    # Fortran line 44 (commented out)
    return 383.0  # Fortran line 45


# ---------------------------------------------------------------------------
# Private: solar radiation partitioning
# ---------------------------------------------------------------------------


def TowerMetSolarRad(
    fsds: float,
) -> Tuple[float, float, float, float]:
    """
    Partition total solar radiation into direct and diffuse components
    for visible and near-infrared wavebands.

    Mirrors Fortran subroutine ``TowerMetSolarRad`` (lines 50-100).

    Total solar radiation is split 50/50 between the visible and
    near-infrared wavebands (Fortran lines 82-83). Within each waveband
    the direct beam fraction is computed from a third-order polynomial
    in the waveband irradiance and clamped to ``[0.01, 0.99]``:

    .. code-block:: none

        fsds_vis = 0.5 * fsds
        rvis = a0 + fsds_vis*(a1 + fsds_vis*(a2 + fsds_vis*a3))
        rvis = clamp(rvis, 0.01, 0.99)

        fsds_nir = 0.5 * fsds
        rnir = b0 + fsds_nir*(b1 + fsds_nir*(b2 + fsds_nir*b3))
        rnir = clamp(rnir, 0.01, 0.99)

    Args:
        fsds: Total incident solar radiation (W/m2).

    Returns:
        Tuple of ``(forc_solad_vis, forc_solai_vis, forc_solad_nir,
        forc_solai_nir)`` in W/m2.
    """
    # Polynomial coefficients — Fortran lines 68-76
    a0 = 0.17639
    a1 = 0.00380
    a2 = -9.0039e-6
    a3 = 8.1351e-9
    b0 = 0.29548
    b1 = 0.00504
    b2 = -1.4957e-5
    b3 = 1.4881e-8

    # Visible waveband — Fortran lines 82-86
    fsds_vis = 0.5 * fsds
    rvis = a0 + fsds_vis * (a1 + fsds_vis * (a2 + fsds_vis * a3))
    rvis = float(jnp.maximum(0.01, jnp.minimum(0.99, rvis)))

    # Near-infrared waveband — Fortran lines 88-90
    fsds_nir = 0.5 * fsds
    rnir = b0 + fsds_nir * (b1 + fsds_nir * (b2 + fsds_nir * b3))
    rnir = float(jnp.maximum(0.01, jnp.minimum(0.99, rnir)))

    # Direct beam and diffuse per waveband — Fortran lines 92-95
    forc_solad_vis = fsds_vis * rvis
    forc_solai_vis = fsds_vis * (1.0 - rvis)
    forc_solad_nir = fsds_nir * rnir
    forc_solai_nir = fsds_nir * (1.0 - rnir)

    return forc_solad_vis, forc_solai_vis, forc_solad_nir, forc_solai_nir


# ---------------------------------------------------------------------------
# Private: atmospheric emissivity
# ---------------------------------------------------------------------------


def TowerMetEmiss(eair: float, tair: float) -> float:
    """
    Compute atmospheric emissivity from vapor pressure and temperature.

    Used when longwave radiation is missing from the tower forcing file.
    Mirrors Fortran function ``TowerMetEmiss`` (lines 102-120).

    .. code-block:: none

        emiss = 0.7 + 5.95e-5 * 0.01 * eair * exp(1500 / tair)

    Args:
        eair: Atmospheric vapor pressure (Pa).
        tair: Atmospheric temperature (K).

    Returns:
        Dimensionless atmospheric emissivity.
    """
    return 0.7 + 5.95e-5 * 0.01 * eair * float(jnp.exp(1500.0 / tair))


# ---------------------------------------------------------------------------
# Public: read forcing for the current CLM time step
# ---------------------------------------------------------------------------


class TowerMetRaw(NamedTuple):
    """One record of the tower netCDF file, as read (``-999.0`` = variable absent)."""

    zbot: float  # Observational height (m)
    tbot: float  # Air temperature (K)
    rhbot: float  # Relative humidity (%)
    qbot: float  # Specific humidity (kg/kg)
    ubot: float  # Wind speed (m/s)
    fsdsbot: float  # Solar radiation (W/m2)
    fldsbot: float  # Longwave radiation (W/m2)
    pbot: float  # Atmospheric pressure (Pa)
    prect: float  # Precipitation (mm/s)
    # Direct and diffuse solar radiation (W/m2), when the file has them instead of FSDS
    # (gridded CRUJRA-style forcing written as tower files); -999 = absent
    fsds_direct: float = -999.0
    fsds_diffuse: float = -999.0


class TowerMet(NamedTuple):
    """Forcing derived from one tower record (Fortran TowerMetCurr/TowerMetNext).

    Wavebands are (vis, nir).
    """

    uref: float  # Wind speed (m/s)
    solad: Tuple[float, float]  # Direct beam solar radiation (W/m2)
    solai: Tuple[float, float]  # Diffuse solar radiation (W/m2)
    tref: float  # Air temperature (K)
    qref: float  # Specific humidity (kg/kg)
    pref: float  # Air pressure (Pa)
    lwsky: float  # Atmospheric longwave radiation (W/m2)
    rain: float  # Rainfall (kg H2O/m2/s)
    hgt: float  # Forcing (reference) height (m)
    co2_ppm: float  # Atmospheric CO2 (umol/mol)
    pco2: float  # CO2 partial pressure (Pa)
    po2: float  # O2 partial pressure (Pa)
    # Snowfall (kg H2O/m2/s): 0 for tower forcing (Fortran: forc_snow = 0); gridded forcing
    # splits precipitation into rain and snow as CLM's data atmosphere does
    snow: float = 0.0


def read_tower_met_series(ncfilename: str, records: Sequence[int]) -> List[TowerMetRaw]:
    """
    Read the given 0-based records of a tower netCDF file.

    Same variables, defaults and value conversion as the per-step
    ``readTowerMet`` (Fortran lines 218-280). Only the span of records
    requested is read, so a run can read its forcing a chunk at a time and
    no file is opened inside the time loop.
    """
    if not records:
        return []
    lo, hi = min(records), max(records) + 1
    with netcdf.dataset(ncfilename, "r") as ds:
        split = "FSDS_DIRECT" in ds.variables and "FSDS_DIFFUSE" in ds.variables
        for name in ("PRECTmms", "TBOT", "WIND") + (() if split else ("FSDS",)):
            if name not in ds.variables:
                handle_err(-1, name)
        n_file = ds.dimensions["time"].size
        if lo < 0 or hi > n_file:
            endrun(
                msg=f" TowerMet error: records {lo}..{hi - 1} outside {ncfilename} (0..{n_file - 1})"
            )

        def span(name: str):
            if name not in ds.variables:
                return None
            return ds.variables[name][lo:hi, 0, 0]

        columns = {
            "zbot": span("ZBOT"),
            "tbot": span("TBOT"),
            "rhbot": span("RH"),
            "qbot": span("QBOT"),
            "ubot": span("WIND"),
            "fsdsbot": span("FSDS"),
            "fldsbot": span("FLDS"),
            "pbot": span("PSRF"),
            "prect": span("PRECTmms"),
            "fsds_direct": span("FSDS_DIRECT"),
            "fsds_diffuse": span("FSDS_DIFFUSE"),
        }

    def value(values, r: int) -> float:
        return -999.0 if values is None else float(values[r - lo])

    return [TowerMetRaw(**{k: value(v, r) for k, v in columns.items()}) for r in records]


def tower_first_record(ncfilename: str, start: datetime.datetime, step_seconds: int) -> int:
    """
    0-based record of a tower file at date-time ``start``, from its time axis.

    The time variable must have units "days since <date>" and a constant
    interval of ``step_seconds``.
    """
    with netcdf.dataset(ncfilename, "r") as ds:
        time = ds.variables["time"]
        origin = datetime.datetime.fromisoformat(time.units.split("since")[1].strip())
        first_day = float(time[0])
        n_file = ds.dimensions["time"].size
    days = (start - origin).total_seconds() / 86400.0 - first_day
    record = round(days * 86400.0 / step_seconds)
    if abs(record * step_seconds - days * 86400.0) > 1.0 or not 0 <= record < n_file:
        endrun(msg=f" TowerMet error: {start} is not a record time in {ncfilename}")
    return record


@dataclasses.dataclass(frozen=True)
class TowerFiles:
    """Tower forcing split in time over several files (e.g. one per month), read as one
    series of records. Record 0 is the first record of the first file; each file must
    start one interval after the previous one ends (no gaps, no overlap)."""

    paths: tuple[str, ...]
    offsets: tuple[int, ...]  # series index of each file's first record
    n_records: int
    origin: datetime.datetime  # time of record 0
    step_seconds: int

    @classmethod
    def open(cls, paths: Sequence[str], step_seconds: int) -> "TowerFiles":
        paths = tuple(str(p) for p in paths)
        if not paths:
            endrun(msg=" TowerMet error: no tower forcing files")
        offsets, total, origin = [], 0, None
        for path in paths:
            with netcdf.dataset(path, "r") as ds:
                time = ds.variables["time"]
                base = datetime.datetime.fromisoformat(time.units.split("since")[1].strip())
                first = base + datetime.timedelta(days=float(time[0]))
                size = ds.dimensions["time"].size
            if origin is None:
                origin = first
            expected = origin + datetime.timedelta(seconds=total * step_seconds)
            if abs((first - expected).total_seconds()) > 1.0:
                endrun(
                    msg=f" TowerMet error: {path} starts at {first}, but the previous file "
                    f"ends at {expected} (files must follow each other without gaps)"
                )
            offsets.append(total)
            total += size
        return cls(paths, tuple(offsets), total, origin, step_seconds)

    def first_record(self, start: datetime.datetime) -> int:
        """Series record at date-time ``start``."""
        seconds = (start - self.origin).total_seconds()
        record = round(seconds / self.step_seconds)
        if abs(record * self.step_seconds - seconds) > 1.0 or not 0 <= record < self.n_records:
            endrun(msg=f" TowerMet error: {start} is not a record time in {self.paths[0]} ...")
        return record

    def read(self, records: Sequence[int]) -> List[TowerMetRaw]:
        """The given series records, each read from the file holding it."""
        out: dict = {}
        bounds = list(self.offsets) + [self.n_records]
        for r in records:
            if not 0 <= r < self.n_records:
                endrun(msg=f" TowerMet error: record {r} outside the forcing files")
        for i, path in enumerate(self.paths):
            mine = [r for r in records if bounds[i] <= r < bounds[i + 1]]
            if mine:
                raws = read_tower_met_series(path, [r - bounds[i] for r in mine])
                out.update(zip(mine, raws))
        return [out[r] for r in records]


def as_tower_files(files, step_seconds: int = 1800) -> TowerFiles:
    """A :class:`TowerFiles` from one path, a list of paths, or a ``TowerFiles``."""
    if isinstance(files, TowerFiles):
        return files
    return TowerFiles.open([files] if isinstance(files, (str, bytes)) else files, step_seconds)


def derive_tower_met(raw: TowerMetRaw, tower_height: float) -> TowerMet:
    """
    Derive the forcing for one tower record.

    The calculations shared by Fortran ``TowerMetCurr`` (lines 122-215) and
    ``TowerMetNext`` (lines 216-280), in the same order:

    - shortwave partitioned into direct/diffuse vis/nir (``TowerMetSolarRad``)
    - forcing height from the tower table (30 m if missing)
    - pressure 101325 Pa if missing
    - specific humidity from relative humidity when available
    - longwave from atmospheric emissivity when missing
    - CO2 from ``TowerMetCO2``, O2 = 0.209 * pressure
    """
    if round(raw.fsds_direct) != -999 and round(raw.fsds_diffuse) != -999:
        # direct and diffuse given: half visible, half near-infrared each (CLM's data
        # atmosphere, datm_datamode_clmncep_advance)
        solad_vis = solad_nir = raw.fsds_direct * 0.5
        solai_vis = solai_nir = raw.fsds_diffuse * 0.5
    else:
        fsds = float(jnp.maximum(raw.fsdsbot, 0.0))
        solad_vis, solai_vis, solad_nir, solai_nir = TowerMetSolarRad(fsds)

    hgt = float(tower_height)
    if round(hgt) == -999:
        hgt = 30.0
    pref = raw.pbot
    if round(pref) == -999:
        pref = 101325.0

    tref = raw.tbot
    qref = raw.qbot
    if round(raw.rhbot) != -999:
        esat, _ = sat_vap(tref)
        eair = (raw.rhbot / 100.0) * esat
        qref = mmh2o / mmdry * eair / (pref - (1.0 - mmh2o / mmdry) * eair)
    elif round(qref) != -999:
        eair = qref * pref / (mmh2o / mmdry + (1.0 - mmh2o / mmdry) * qref)
    else:
        endrun(msg=" TowerMet error: rhref and qref not valid")
        eair = 0.0  # Unreachable; satisfies type checker

    lwsky = raw.fldsbot
    if round(lwsky) == -999:
        emiss = TowerMetEmiss(eair, tref)
        lwsky = emiss * sb * tref**4

    co2 = TowerMetCO2()
    return TowerMet(
        uref=raw.ubot,
        solad=(solad_vis, solad_nir),
        solai=(solai_vis, solai_nir),
        tref=tref,
        qref=float(qref),
        pref=pref,
        lwsky=float(lwsky),
        rain=raw.prect,
        hgt=hgt,
        co2_ppm=co2,
        pco2=(co2 / 1.0e6) * pref,
        po2=0.209 * pref,
    )
