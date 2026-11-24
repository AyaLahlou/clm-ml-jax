"""
Gridded forcing and land data for grid runs, handled as CLM handles
them: the data atmosphere (CDEPS ``datm``, ``CLMNCEP`` mode with the three CRUJRA streams),
CLM's satellite phenology, and CLM's prescribed soil-moisture stream.

**Meteorology** (CDEPS ``datm_datamode_clmncep_mod``, ``dshr_strdata_mod``,
``dshr_tinterp_mod``):

- Spatial mapping 0.5° -> gridcell: bilinear between the four surrounding source points;
  where one of them is outside the forcing's land mask, the nearest valid source point (CDEPS:
  ESMF bilinear with the source mask and nearest-source extrapolation for unmapped points).
  Here the bilinear weights are planar in latitude/longitude (ESMF works on the sphere; the
  difference is second order in the 0.5° spacing).
- Time: CLM's run sequence (``nuopc.runseq``) runs the land before the data atmosphere in
  each coupling interval, so the land step ``[t, t + Δt]`` uses the data atmosphere evaluated
  at ``t``, the step's start. Each stream is interpolated in its own calendar (``noleap`` for
  CRUJRA); a Gregorian model date of 29 February uses 28 February (CDEPS). Stamps are those
  of the files. Each stream's data years form a loop (:class:`StreamYears`, CDEPS
  ``taxmode = cycle`` with ``stream_year_first/last`` and ``year_align``, as CLM runs these
  streams): a model year reads its data year, and the first data year's first steps
  interpolate from the last record of the last data year.
  - ``TPQW`` (temperature, pressure, humidity, wind, longwave): **linear** between the
    stamps bracketing ``t`` (lower bound: last stamp <= t);
  - ``Precip``: **nearest** stamp (a tie goes to the earlier one);
  - ``Solar``: **coszen**: the lower-bound record times ``cosz(t) / <cosz>``, with ``cosz``
    the instantaneous cosine of the solar zenith angle (at least 0.001) and ``<cosz>`` its
    mean over the record's interval sampled at the model step from the record's stamp; zero
    when ``cosz(t) <= 0.001``. The model's orbital parameters (the run's year).
- Derived as ``datm`` does: ``TBOT >= 180 K``; specific humidity capped at saturation
  (``datm_eSat``, over ice below 0 °C); ``PSRF = 0`` -> 101325 Pa; direct and diffuse shortwave
  as given, each split half visible / half near-infrared; precipitation split into rain and
  snow by the ramp ``frac_rain = (T - 273.15) / 2`` bounded to [0, 1] (``datm``, and CLM's
  ``repartition_rain_snow`` with its default thresholds 0 and 2 °C gives the same split);
  forcing height 30 m. Wind: ``datm`` sets ``u = v = WIND / sqrt(2)``; the model only uses
  the speed ``sqrt(u^2 + v^2)``, so ``u = WIND, v = 0`` is used (the same speed without the
  round-off of the split).
- CO2: a ``co2tseries``-style stream (global annual values, **linear** in time at ``t``) or
  a constant; O2 partial pressure ``0.209 * PSRF`` (CLM's ``o2_molar_const``).

**Vegetation**: the surface data's monthly ``MONTHLY_LAI``/``MONTHLY_SAI`` per PFT and
gridcell, interpolated as CLM's satellite phenology does (:func:`clm_ml.grid.clm_monthly_weights`
at the end of the step plus one step), then ``elai = tlai`` and ``esai = tsai`` (no snow
burial: this model has no snow), not below 0. Unlike CLM, values below 0.05 are *not* set to
0: CLM does it to protect its big-leaf canopy from very small plant area, which
the minimum plant area of every leafy layer (model.yaml
``canopy_layers.minimum_leafy_layer_plant_area``) does here; without the cutoff LAI and SAI
enter continuously, so the model is differentiable with respect to them.

**Soil water**: ``H2OSOI`` from monthly CLM history files per gridcell (the CLM run's
``landmask`` grid), **linear** between the files' time stamps (the middle of each month in
CTSM 5.4 history files) at the end of the step (CLM's ``SoilMoistureStreamMod``:
``get_curr_date``, ``soilm_tintalgo = linear``). The initial soil temperature and water
(``TSOI``, ``H2OSOI``) are the same interpolation at the end of the first step (what the
tower setup reads from its CLM history slice).

Meteorology is built on the host for each chunk at the model step, per gridcell (``xg``, as
for towers). Vegetation and soil water stay as monthly records and are interpolated inside the
compiled chunk (:func:`land_record`), so they cost no memory per step.

Fortran reference: CDEPS datm_datamode_clmncep_mod.F90 (datm_datamode_clmncep_advance,
datm_eSat), dshr_tinterp_mod.F90 (shr_tInterp_getFactors, shr_tInterp_getAvgCosz,
shr_tInterp_getCosz), shr_precip_mod.F90; CTSM SatellitePhenologyMod.F90,
SoilMoistureStreamMod.F90, atm2lndMod.F90 (partition_precip).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np

from clm_ml.io import netcdf
from clm_ml.io.yaml_schema import ConfigError, option

TKFRZ = 273.15  # SHR_CONST_TKFRZ
PSTD = 101325.0  # SHR_CONST_PSTD
SOLZEN_MIN = 0.001  # dshr_tInterp_mod solZenMin
O2_MOLAR = 0.209  # CLM o2_molar_const

#: datm stream -> (variables, time interpolation)
STREAMS = {
    "tpqw": (("TBOT", "PSRF", "QBOT", "WIND", "FLDS"), "linear"),
    "solar": (("FSDS_DIRECT", "FSDS_DIFFUSE"), "coszen"),
    "precipitation": (("PRECTmms",), "nearest"),
}

# ---------------------------------------------------------------------------------------
# run.yaml schema (forcing.gridded)
# ---------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StreamYears:
    """A stream's time axis as CDEPS handles it (``taxmode``, ``stream_year_first``,
    ``stream_year_last``, ``year_align``): with ``cycle`` the data years
    ``year_first..year_last`` repeat, model year ``year_align`` reading ``year_first``, so a
    run in the first data year interpolates from the last record of ``year_last`` (CDEPS
    ``shr_stream_findBounds``: the greatest valid date, shifted back by the loop length), and
    a run past ``year_last`` starts the loop again; with ``limit`` the run must lie inside
    the data."""

    taxmode: Literal["cycle", "limit"] = option(
        "cycle",
        "Time axis outside the data years: cycle (repeat them, CLM's default for "
        "these streams) or limit (stop).",
        "taxmode",
    )
    year_first: Optional[int] = option(None, "First data year of the loop.",
                                       "stream_year_first", required=True)  # fmt: skip
    year_last: Optional[int] = option(None, "Last data year of the loop.", "stream_year_last",
                                      required=True)  # fmt: skip
    year_align: Union[Literal["year_first"], int] = option(
        "year_first",
        "The model year that reads year_first (year_first: model years are data " "years).",
        "year_align",
        when={"taxmode": "cycle"},
    )

    def data_year(self, model_year: int) -> int:
        """The data year read for ``model_year`` (CDEPS ``dYear``)."""
        if self.taxmode != "cycle":
            return model_year
        n = self.year_last - self.year_first + 1
        align = self.year_first if self.year_align == "year_first" else int(self.year_align)
        return self.year_first + (model_year - align + 2 * n) % n


def shift_years(stamps, calendar: str, years: int) -> np.ndarray:
    """Stamps (seconds of ``calendar``, :data:`_EPOCH`) moved by ``years`` calendar years
    (CDEPS shifts the date's year; a 29 February with no counterpart is dropped: NaN)."""
    import cftime

    stamps = np.asarray(stamps, dtype=float)
    if years == 0:
        return stamps
    out = np.empty_like(stamps)
    for i, d in enumerate(cftime.num2date(stamps, _EPOCH, calendar=calendar)):
        try:
            out[i] = cftime.date2num(d.replace(year=d.year + years), _EPOCH, calendar=calendar)
        except ValueError:
            out[i] = np.nan
    return out


@dataclasses.dataclass(frozen=True)
class DatmStreamsSection:
    """The three CRUJRA datm streams ({year} is expanded), mapped and interpolated in time as
    CDEPS datm does."""

    tpqw: str = option(doc="Temperature, pressure, humidity, wind and longwave stream.",
                       path=True)  # fmt: skip
    solar: str = option(doc="Solar radiation stream.", path=True)
    precipitation: str = option(doc="Precipitation stream.", path=True)
    mask: str = option(doc="The streams' 0.5 degree domain file with mask.", path=True)
    years: Optional[StreamYears] = option(
        None,
        "The streams' data years and time axis (CLM: DATM_YR_START, DATM_YR_END, "
        "DATM_YR_ALIGN).",
        required=True,
    )


@dataclasses.dataclass(frozen=True)
class Co2Section:
    """Atmospheric CO2."""

    source: Literal["file", "constant"] = option(
        "file", "A CO2 time series file, or a constant value."
    )
    file: Optional[str] = option(
        None,
        "CO2 time series file (CO2, time with units and calendar).",
        path=True,
        when={"source": "file"},
        required=True,
    )
    ppm: Optional[float] = option(None, "Constant CO2 (ppm).", when={"source": "constant"},
                                  required=True)  # fmt: skip


@dataclasses.dataclass(frozen=True)
class SoilHistorySection:
    """Monthly CLM history files with H2OSOI and TSOI on the surface-data grid: prescribed
    soil water, and the initial soil temperature."""

    files: str = option(doc="The files, with {year} and {month} expanded.", path=True)
    years: Optional[StreamYears] = option(
        None,
        "The files' data years and time axis (CLM: stream_year_first_soilm, "
        "stream_year_last_soilm, model_year_align_soilm).",
        required=True,
    )


@dataclasses.dataclass(frozen=True)
class GriddedForcingSection:
    """Gridded forcing (grid runs)."""

    meteorology: DatmStreamsSection = option(doc="The meteorology streams.")
    co2: Co2Section = option(doc="Atmospheric CO2.")
    soil: SoilHistorySection = option(doc="Monthly soil water and temperature.")
    interpolation: Literal["none", "three_point"] = option(
        "three_point",
        "Forcing within a model step's sub-steps, as for towers: three-point interpolation "
        "(CLM-ml v2) or the step's value.",
        fortran="met_type 3/0",
    )


# ---------------------------------------------------------------------------------------
# Time in a stream's calendar
# ---------------------------------------------------------------------------------------

_EPOCH = "seconds since 1700-01-01 00:00:00"


def stream_time(t: dt.datetime, calendar: str) -> float:
    """Model date-time ``t`` (Gregorian) in ``calendar`` seconds (CDEPS ``ymdmod``: a
    29 February in a no-leap stream is 28 February)."""
    import cftime

    cal = calendar.lower()
    if cal in ("noleap", "365_day") and t.month == 2 and t.day == 29:
        t = t.replace(day=28)
    c = cftime.datetime(t.year, t.month, t.day, t.hour, t.minute, t.second, calendar=cal)
    return float(cftime.date2num(c, _EPOCH, calendar=cal))


def stamps_of(time_var) -> tuple[np.ndarray, str]:
    """A netCDF time variable's stamps in seconds (:data:`_EPOCH`) of its calendar."""
    import cftime

    cal = getattr(time_var, "calendar", "standard").lower()
    cal = "standard" if cal == "gregorian" else cal
    dates = cftime.num2date(np.asarray(time_var[:], dtype=float), time_var.units, calendar=cal)
    return np.asarray(cftime.date2num(dates, _EPOCH, calendar=cal), dtype=float), cal


def calday_in(t: dt.datetime, calendar: str) -> float:
    """Julian day (1.0 = 0Z 1 January) of model date ``t`` in ``calendar`` (CDEPS
    ``shr_cal_date2julian``, after the 29 February mapping)."""
    cal = calendar.lower()
    leap_cal = cal not in ("noleap", "365_day")
    if not leap_cal and t.month == 2 and t.day == 29:
        t = t.replace(day=28)
    doy = t.timetuple().tm_yday
    if (
        not leap_cal
        and t.month > 2
        and (t.year % 4 == 0 and (t.year % 100 != 0 or t.year % 400 == 0))
    ):
        doy -= 1
    return doy + (t.hour * 3600 + t.minute * 60 + t.second) / 86400.0


def lower_index(stamps: np.ndarray, t: float) -> int:
    """CDEPS lower bound: the last stamp <= t."""
    i = int(np.searchsorted(stamps, t, side="right")) - 1
    if i < 0 or i + 1 >= len(stamps):
        raise ValueError("forcing stream: the files do not cover the model time")
    return i


def linear_factors(stamps: np.ndarray, t: float) -> tuple[int, float, float]:
    """``(lb, f_lb, f_ub)``: CDEPS ``linear`` (``f1 = (t_ub - t) / (t_ub - t_lb)``,
    ``f2 = 1 - f1``)."""
    lb = lower_index(stamps, t)
    f1 = (stamps[lb + 1] - t) / (stamps[lb + 1] - stamps[lb])
    return lb, f1, 1.0 - f1


def nearest_index(stamps: np.ndarray, t: float) -> int:
    """CDEPS ``nearest`` (a tie goes to the lower bound)."""
    lb = lower_index(stamps, t)
    return lb if abs(stamps[lb] - t) <= abs(stamps[lb + 1] - t) else lb + 1


# ---------------------------------------------------------------------------------------
# Spatial mapping
# ---------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PointMap:
    """Destination values as weighted sums of source points: ``index`` (dst, 4) into
    ``points`` (the source points read, flat ``j * nlon + i``), ``weight`` (dst, 4)."""

    points: np.ndarray  # (n_src,) flat source index
    index: np.ndarray  # (dst, 4) into points
    weight: np.ndarray  # (dst, 4)

    def apply(self, values: np.ndarray) -> np.ndarray:
        """``values`` (..., n_src) at the points -> (..., dst)."""
        return np.sum(values[..., self.index] * self.weight, axis=-1)


def bilinear_map(src_lat, src_lon, src_mask, dst_lat, dst_lon) -> PointMap:
    """Bilinear weights from a regular lat/lon grid (centers ``src_lat`` (nlat,), ``src_lon``
    (nlon,), increasing, ``src_mask`` (nlat, nlon) valid points) to destination points; the
    nearest valid source point where a corner is masked (module docstring)."""
    from scipy.spatial import cKDTree

    src_lat, src_lon = np.asarray(src_lat, float), np.asarray(src_lon, float)
    nlat, nlon = src_lat.size, src_lon.size
    dlat, dlon = src_lat[1] - src_lat[0], src_lon[1] - src_lon[0]
    lat = np.asarray(dst_lat, float)
    lon = np.mod(np.asarray(dst_lon, float) - src_lon[0], 360.0) + src_lon[0]
    y = (lat - src_lat[0]) / dlat
    j0 = np.clip(np.floor(y).astype(int), 0, nlat - 2)
    fy = np.clip(y - j0, 0.0, 1.0)
    x = (lon - src_lon[0]) / dlon
    i0 = np.floor(x).astype(int) % nlon
    fx = x - np.floor(x)
    i1 = (i0 + 1) % nlon
    corners = np.stack(
        [j0 * nlon + i0, j0 * nlon + i1, (j0 + 1) * nlon + i0, (j0 + 1) * nlon + i1], 1
    )
    weight = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], 1)
    valid = np.asarray(src_mask, bool).ravel()
    bad = ~valid[corners].all(axis=1)
    if bad.any():
        jj, ii = np.nonzero(np.asarray(src_mask, bool))

        def xyz(la, lo):
            la, lo = np.radians(la), np.radians(lo)
            return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)

        tree = cKDTree(xyz(src_lat[jj], src_lon[ii]))
        _, k = tree.query(xyz(lat[bad], lon[bad]))
        corners[bad] = (jj[k] * nlon + ii[k])[:, None]
        weight[bad] = np.array([1.0, 0.0, 0.0, 0.0])
    points, index = np.unique(corners, return_inverse=True)
    return PointMap(points, index.reshape(corners.shape), weight)


# ---------------------------------------------------------------------------------------
# datm derivations
# ---------------------------------------------------------------------------------------


def datm_esat(tk, tkbot):
    """CDEPS ``datm_eSat``: saturation vapour pressure (Pa), over ice when ``tkbot`` is
    below freezing."""
    a = (6.107799961, 4.436518521e-01, 1.428945805e-02, 2.650648471e-04, 3.031240396e-06,
         2.034080948e-08, 6.136820929e-11)  # fmt: skip
    b = (6.109177956, 5.034698970e-01, 1.886013408e-02, 4.176223716e-04, 5.824720280e-06,
         4.838803174e-08, 1.838826904e-10)  # fmt: skip
    t = np.minimum(50.0, np.maximum(-50.0, tk - TKFRZ))

    def poly(c):
        return 100.0 * (
            c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * (c[4] + t * (c[5] + t * c[6])))))
        )

    return np.where(tkbot < TKFRZ, poly(b), poly(a))


def datm_derive(tbot, psrf, qbot, wind, flds, fsds_dir, fsds_dif, prect, co2_ppm, height):
    """The forcing ``datm`` (``CLMNCEP``) and CLM's precipitation repartition derive from the
    stream values (arrays of any shape), as the ``TowerMet`` fields plus ``snow``."""
    tbot = np.maximum(180.0, tbot)
    pbot = np.where(psrf == 0.0, PSTD, psrf)
    e = datm_esat(tbot, tbot)
    qsat = (0.622 * e) / (pbot - 0.378 * e)
    q = np.minimum(qbot, qsat)
    # datm: precn as 10 % convective and 90 % large-scale rain, then split by the ramp
    frac = np.minimum(1.0, np.maximum(0.0, (tbot - TKFRZ) * 0.5))
    rainc, rainl = prect * 0.1, prect * 0.9
    snow_d = np.maximum(0.0, rainc * (1.0 - frac)) + np.maximum(0.0, rainl * (1.0 - frac))
    rain_d = np.maximum(0.0, rainc * frac) + np.maximum(0.0, rainl * frac)
    # CLM (atm2lndMod, repartition_rain_snow_one_col; non-glacier: all snow at 0 C, all rain
    # at 2 C): the total repartitioned with the same ramp on the same temperature
    frac_rain = np.minimum(1.0, np.maximum(0.0, (tbot - (TKFRZ + 0.0)) * (1.0 / (2.0 - 0.0))))
    total = rain_d + snow_d
    rain = total * frac_rain
    snow = total - rain
    return {
        "uref": wind,
        "solad": (fsds_dir * 0.5, fsds_dir * 0.5),
        "solai": (fsds_dif * 0.5, fsds_dif * 0.5),
        "tref": tbot,
        "qref": q,
        "pref": pbot,
        "lwsky": flds,
        "rain": rain,
        "hgt": np.broadcast_to(height, np.shape(tbot)),
        "co2_ppm": co2_ppm,
        "pco2": (co2_ppm / 1.0e6) * pbot,
        "po2": O2_MOLAR * pbot,
        "snow": snow,
    }


# ---------------------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------------------


class YearlyStream:
    """One ``datm`` stream split into yearly files (``{year}`` in the path), read for the
    source points of a :class:`PointMap`."""

    def __init__(self, pattern: str, variables, point_map: PointMap, nlon: int,
                 years: Optional[StreamYears] = None):  # fmt: skip
        self.pattern, self.variables, self.map, self.nlon = (
            pattern,
            tuple(variables),
            point_map,
            nlon,
        )
        self.years = years or StreamYears(taxmode="limit")
        self._index: dict = {}  # year -> (stamps, calendar)

    def path(self, year: int) -> Path:
        return Path(self.pattern.format(year=year))

    def stamps(self, year: int):
        if year not in self._index:

            path = self.path(year)
            if not path.exists():
                self._index[year] = None
            else:
                with netcdf.dataset(path) as ds:
                    self._index[year] = stamps_of(ds.variables["time"])
        return self._index[year]

    def window(self, first: dt.datetime, last: dt.datetime):
        """Stamps (seconds, the stream's calendar, on the model's years) of the records from
        the year before ``first`` to the year after ``last``, with ``(data year, k)`` per
        record: each model year reads its data year (:class:`StreamYears`), its stamps moved
        to the model year."""
        stamps, where, cal = [], [], None
        for year in range(first.year - 1, last.year + 2):
            data_year = self.years.data_year(year)
            entry = self.stamps(data_year)
            if entry is None:
                continue
            s, cal = entry
            s = shift_years(s, cal, year - data_year)
            keep = np.isfinite(s)
            stamps.append(s[keep])
            where += [(data_year, k) for k in range(len(s)) if keep[k]]
        if not stamps:
            raise ConfigError(f"forcing: no file for {first.year} ({self.path(first.year)})")
        return np.concatenate(stamps), where, cal

    def read(self, records):
        """``{variable: (len(records), dst)}`` mapped to the destination points, for
        ``records`` [(year, k), ...] (each year's span read once)."""

        jj, ii = np.divmod(self.map.points, self.nlon)
        out = {v: np.empty((len(records), self.map.index.shape[0])) for v in self.variables}
        by_year: dict = {}
        for n, (year, k) in enumerate(records):
            by_year.setdefault(year, []).append((n, k))
        for year, items in by_year.items():
            ks = [k for _, k in items]
            lo, hi = min(ks), max(ks) + 1
            with netcdf.dataset(self.path(year)) as ds:
                for v in self.variables:
                    slab = np.ma.filled(ds.variables[v][lo:hi], np.nan)
                    pts = np.asarray(slab[:, jj, ii], dtype=float)
                    mapped = self.map.apply(pts)
                    for n, k in items:
                        out[v][n] = mapped[k - lo]
        return out


class Co2Stream:
    """Global CO2 (ppm): a ``co2tseries`` file, linear in time, or a constant."""

    def __init__(self, section: Co2Section, resolve):
        self.ppm = section.ppm if section.source == "constant" else None
        if section.source == "file":

            with netcdf.dataset(resolve(section.file)) as ds:
                self.stamps, self.calendar = stamps_of(ds.variables["time"])
                self.values = np.asarray(ds.variables["CO2"][:], dtype=float).reshape(-1)

    def at(self, t: dt.datetime) -> float:
        if self.ppm is not None:
            return float(self.ppm)
        lb, f1, f2 = linear_factors(self.stamps, stream_time(t, self.calendar))
        return self.values[lb] * f1 + self.values[lb + 1] * f2


class MonthlyHistory:
    """Monthly CLM history files (``{year}``, ``{month}``): ``H2OSOI``/``TSOI`` per domain
    gridcell (rows/cols on the history grid), with the files' time stamps."""

    def __init__(self, pattern: str, rows, cols, years: Optional[StreamYears] = None):
        self.pattern, self.rows, self.cols = pattern, np.asarray(rows), np.asarray(cols)
        self.years = years or StreamYears(taxmode="limit")
        self._stamp: dict = {}

    def path(self, year: int, month: int) -> Path:
        return Path(self.pattern.format(year=year, month=month))

    def months_around(self, first: dt.datetime, last: dt.datetime):
        """``[(data year, month, stamp)]`` from the month before ``first`` to the month after
        ``last``, files that exist, in the files' calendar seconds on the model's years (each
        model year reads its data year, :class:`StreamYears`)."""

        y, m = (first.year, first.month - 1) if first.month > 1 else (first.year - 1, 12)
        out, cal = [], None
        while (y, m) <= ((last.year, last.month + 1) if last.month < 12 else (last.year + 1, 1)):
            dy = self.years.data_year(y)
            path = self.path(dy, m)
            if path.exists():
                if (dy, m) not in self._stamp:
                    with netcdf.dataset(path) as ds:
                        s, c = stamps_of(ds.variables["time"])
                        self._stamp[(dy, m)] = (float(s[0]), c)
                s, cal = self._stamp[(dy, m)]
                s = float(shift_years([s], cal, y - dy)[0])
                if np.isfinite(s):
                    out.append((dy, m, s))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        if len(out) < 2:
            raise ConfigError(f"forcing.gridded.soil: no monthly files around {first:%Y-%m}")
        return out, cal

    def read(self, year: int, month: int, name: str) -> np.ndarray:
        """``name`` (levels, gridcell) of one month."""

        with netcdf.dataset(self.path(year, month)) as ds:
            v = np.ma.filled(ds.variables[name][0], np.nan).astype(float)
        out = v[:, self.rows, self.cols]
        if not np.all(np.isfinite(out)) or np.any(np.abs(out) > 1e30):
            bad = int(np.sum(~np.isfinite(out) | (np.abs(out) > 1e30)))
            raise ValueError(
                f"{self.path(year, month)}: {name} has {bad} missing values in the domain"
            )
        return out


# ---------------------------------------------------------------------------------------
# The inputs of a grid run
# ---------------------------------------------------------------------------------------


class GridInputs:
    """Everything a grid run reads per chunk: the meteorology at the model steps per
    gridcell, and the monthly vegetation and soil-water records with their time weights.

    ``domain``: the run's grid :class:`clm_ml.domain.Domain`; ``surface_data``: its surface
    dataset; ``section``: run.yaml ``forcing.gridded``; ``start``, ``step_seconds``,
    ``n_steps``, ``calendar`` (``GREGORIAN`` or ``NOLEAP``): the run's clock (one for all
    gridcells)."""

    def __init__(self, domain, surface_data, section: GriddedForcingSection, start, step_seconds,
                 n_steps, resolve=lambda p: Path(p), calendar="GREGORIAN"):  # fmt: skip

        from clm_ml.processes.orbit import shr_orb_params

        self.domain, self.start, self.calendar = domain, start, calendar
        self.step_seconds, self.n_steps = int(step_seconds), int(n_steps)
        g = domain.gridcells
        met = section.meteorology
        with netcdf.dataset(resolve(met.mask)) as ds:
            mask = np.asarray(ds.variables["mask"][:]) > 0
            src_lat = np.asarray(ds.variables["yc"][:, 0], dtype=float)
            src_lon = np.asarray(ds.variables["xc"][0, :], dtype=float)
        self.point_map = bilinear_map(src_lat, src_lon, mask, g.latitude, g.longitude)
        self.streams = {
            name: YearlyStream(str(resolve(getattr(met, name))), STREAMS[name][0], self.point_map,
                               src_lon.size, met.years)  # fmt: skip
            for name in STREAMS
        }
        self.co2 = Co2Stream(section.co2, resolve)
        rows, cols = g.grid_index[:, 0], g.grid_index[:, 1]
        self.history = MonthlyHistory(str(resolve(section.soil.files)), rows, cols,
                                      section.soil.years)  # fmt: skip
        self.lat_rad, self.lon_rad = np.radians(g.latitude), np.radians(g.longitude)
        _e, _o, _m, self.obliqr, self.lambm0, self.mvelpp = shr_orb_params(start.year)
        self.eccen = _e
        # monthly vegetation per patch, (12, P), from the surface data
        p = domain.patches
        prow, pcol = rows[p.gridcell_index], cols[p.gridcell_index]
        with netcdf.dataset(resolve(surface_data)) as ds:
            self.lai = self._monthly(ds.variables["MONTHLY_LAI"], p.pft_index, prow, pcol)
            self.sai = self._monthly(ds.variables["MONTHLY_SAI"], p.pft_index, prow, pcol)

    @staticmethod
    def _monthly(var, pft, rows, cols) -> np.ndarray:
        out = np.empty((12, len(pft)))
        for m in range(12):
            out[m] = np.ma.filled(var[m], np.nan)[pft, rows, cols]
        if not np.all(np.isfinite(out)):
            raise ValueError(f"{var.name}: missing values for some patches")
        return out

    # -- time ------------------------------------------------------------------------------

    def step_start(self, k: int) -> dt.datetime:
        """Start of run step ``k`` in the model calendar (a noleap run never reaches 29
        February; a Gregorian one reads the no-leap streams' 28 February then)."""
        from clm_ml.calendar import add_seconds

        return add_seconds(self.start, k * self.step_seconds, self.calendar)

    # -- meteorology -----------------------------------------------------------------------

    def _cosz(self, t: dt.datetime, calendar: str) -> np.ndarray:
        """CDEPS ``shr_tInterp_getCosz`` at model time ``t`` for every gridcell."""
        from clm_ml.processes.orbit import shr_orb_decl

        day = calday_in(t, calendar)
        declin, _ = shr_orb_decl(day, self.eccen, self.mvelpp, self.lambm0, self.obliqr)
        declin = float(declin)
        # shr_orb_cosz, on arrays
        cosz = np.sin(self.lat_rad) * np.sin(declin) - np.cos(self.lat_rad) * np.cos(
            declin
        ) * np.cos(day * 2.0 * np.pi + self.lon_rad)
        return np.maximum(SOLZEN_MIN, cosz)

    def met_at(self, run_steps) -> dict:
        """The derived meteorology (``datm_derive`` fields) at the starts of ``run_steps``,
        ``{field: (len(run_steps), gridcell)}``."""
        times = [self.step_start(k) for k in run_steps]
        values: dict = {}
        for name, stream in self.streams.items():
            stamps, where, cal = stream.window(times[0], times[-1])
            ts = [stream_time(t, cal) for t in times]
            rule = STREAMS[name][1]
            if rule == "linear":
                lbs = [linear_factors(stamps, t) for t in ts]
                need = sorted({lb for lb, _, _ in lbs} | {lb + 1 for lb, _, _ in lbs})
            else:
                lbs = [nearest_index(stamps, t) if rule == "nearest" else lower_index(stamps, t)
                       for t in ts]  # fmt: skip
                need = sorted(set(lbs))
            data = stream.read([where[i] for i in need])
            pos = {i: n for n, i in enumerate(need)}
            for v, arr in data.items():
                if rule == "linear":
                    values[v] = np.stack([arr[pos[lb]] * f1 + arr[pos[lb + 1]] * f2
                                          for lb, f1, f2 in lbs])  # fmt: skip
                elif rule == "nearest":
                    values[v] = np.stack([arr[pos[i]] for i in lbs])
                else:  # coszen
                    rows = []
                    for t, lb in zip(times, lbs):
                        cosz = self._cosz(t, cal)
                        tav = self._mean_cosz(stamps[lb], stamps[lb + 1], cal)
                        rows.append(np.where(cosz > SOLZEN_MIN, arr[pos[lb]] * cosz / tav, 0.0))
                    values[v] = np.stack(rows)
        co2 = np.array([self.co2.at(t) for t in times])[:, None] * np.ones(len(self.lat_rad))
        return datm_derive(
            values["TBOT"], values["PSRF"], values["QBOT"], values["WIND"], values["FLDS"],
            values["FSDS_DIRECT"], values["FSDS_DIFFUSE"], values["PRECTmms"], co2,
            float(self.domain.gridcells.measurement_height[0]),
        )  # fmt: skip

    def _mean_cosz(self, lb: float, ub: float, calendar: str) -> np.ndarray:
        """CDEPS ``shr_tInterp_getAvgCosz``: mean of ``cosz`` over ``[lb, ub)`` sampled every
        model step from ``lb`` (stream-calendar seconds)."""
        import cftime

        cache = self.__dict__.setdefault("_tav_cache", {})
        key = (lb, ub, calendar)
        if key not in cache:
            ldt = self.step_seconds
            span = int(round(ub - lb))
            if span % ldt:
                ldt = span // (span // ldt + 1)
            total, n, s = 0.0, 0, lb
            while s < ub:
                c = cftime.num2date(s, _EPOCH, calendar=calendar)
                t = dt.datetime(c.year, c.month, c.day, c.hour, c.minute, c.second)
                total = total + self._cosz(t, calendar) * ldt
                n += ldt
                s += ldt
            cache[key] = total / n
        return cache[key]

    # -- land data -------------------------------------------------------------------------

    def soil_window(self, first_step: int, n: int, n_records: Optional[int] = 3):
        """Soil-water records and weights for steps ``first_step ... first_step + n - 1``:
        ``(h2osoi (n_records, gridcell, levels), lb (n,), f_lb (n,), f_ub (n,))``, linear at
        each step's end; the records are those the steps need, padded to ``n_records``
        (a fixed count, so every chunk has the same shapes; None: as many as needed)."""
        ends = [self.step_start(k + 1) for k in range(first_step, first_step + n)]
        months, cal = self.history.months_around(ends[0], ends[-1])
        stamps = np.array([s for _, _, s in months])
        f = [linear_factors(stamps, stream_time(t, cal)) for t in ends]
        lo = min(lb for lb, _, _ in f)
        hi = max(lb for lb, _, _ in f) + 1
        n_records = hi - lo + 1 if n_records is None else n_records
        if hi - lo + 1 > n_records:
            raise ValueError("soil_window: a chunk spans more monthly soil-water records than "
                             f"{n_records}")  # fmt: skip
        recs = [self.history.read(*months[i][:2], "H2OSOI").T for i in range(lo, hi + 1)]
        recs += [recs[-1]] * (n_records - len(recs))
        lb = np.array([x - lo for x, _, _ in f])
        return np.stack(recs), lb, np.array([a for _, a, _ in f]), np.array([b for _, _, b in f])

    def soil_at_first_step(self, name: str) -> np.ndarray:
        """``name`` (``TSOI``, ``H2OSOI``) at the end of the first step, (gridcell, levels)."""
        end = self.step_start(1)
        months, cal = self.history.months_around(end, end)
        stamps = np.array([s for _, _, s in months])
        lb, f1, f2 = linear_factors(stamps, stream_time(end, cal))
        a = self.history.read(*months[lb][:2], name)
        b = self.history.read(*months[lb + 1][:2], name)
        return (a * f1 + b * f2).T

    def vegetation_weights(self, first_step: int, n: int):
        """CLM's satellite-phenology months and weights per step, ``(m1, m2, w1, w2)``
        (n,) each, months 0-based."""
        from clm_ml.grid import clm_monthly_weights, satellite_phenology_date

        out = []
        for k in range(first_step, first_step + n):
            when = satellite_phenology_date(self.step_start(k), self.step_seconds, self.calendar)
            m1, m2, w1, w2 = clm_monthly_weights(when.month, when.day)
            out.append((m1 - 1, m2 - 1, w1, w2))
        m1, m2, w1, w2 = (np.array(c) for c in zip(*out))
        return m1, m2, w1, w2


def exposed_area(tlai, tsai, xp=np):
    """``(elai, esai)`` from the interpolated LAI and SAI: CLM's satellite phenology without
    snow burial, ``max(., 0)``, and without CLM's cutoff of values below 0.05 (see the
    module docstring) (``xp``: ``numpy`` or ``jax.numpy``, same operations).

    Fortran reference: CTSM SatellitePhenologyMod.F90 (UpdateSatellitePhenologyCanopy), whose
    ``if (elai(p) < 0.05_r8) elai(p) = 0._r8`` (and the same for esai) is left out."""
    return xp.maximum(tlai, 0.0), xp.maximum(tsai, 0.0)


def land_record(xw, land, gidx):
    """The land data of every patch at one step (``clm_ml.step.LandInputs``), inside the
    compiled chunk:
    vegetation from the monthly records with CLM's satellite-phenology weights (``xw``: m1,
    m2, w1, w2), soil water linear between the monthly records (``xw``: lb, f_lb, f_ub;
    ``land``: ``(lai (12, P), sai (12, P), h2osoi (records, G, nlevsoi))``)."""
    import jax.numpy as jnp

    from clm_ml.step import LandInputs

    m1, m2, w1, w2, lb, f1, f2 = xw
    lai, sai, h2osoi = land
    tlai = w1 * jnp.take(lai, m1, axis=0) + w2 * jnp.take(lai, m2, axis=0)
    tsai = w1 * jnp.take(sai, m1, axis=0) + w2 * jnp.take(sai, m2, axis=0)
    elai, esai = exposed_area(tlai, tsai, jnp)
    h = jnp.take(h2osoi, lb, axis=0) * f1 + jnp.take(h2osoi, lb + 1, axis=0) * f2  # (G, lev)
    h = jnp.take(h, gidx, axis=0)
    n_p = elai.shape[0]
    return LandInputs(
        elai=elai,
        esai=esai,
        h2osoi=h,
        h2osoi_factor=jnp.ones((n_p,), dtype=jnp.float64),  # no adjustment factor
    )
