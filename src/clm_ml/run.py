"""
``run.yaml``: when an experiment runs, with which forcing and land data, and what it writes
(one of the four files of an experiment folder, decision 91; ``docs/design/experiment_folders.md``).

Every setting is declared here with :func:`clm_ml.io.yaml_schema.option` and its
explanation, which ``clm-ml new`` and ``clm-ml upgrade`` write into the file. The folder is
loaded by :func:`clm_ml.experiment.load_experiment`.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Optional

from clm_ml.grid_forcing import GriddedForcingSection
from clm_ml.io.yaml_schema import ConfigError, option

HEADER = """\
run.yaml: when the experiment runs, with which forcing and land data, and what it writes.
Part of an experiment folder (run.yaml, domain.yaml, model.yaml, params.yaml). Every setting
is listed; REQUIRED keys have no default. Paths are relative to this folder. {generated}"""


@dataclasses.dataclass(frozen=True)
class TimeSection:
    """The simulated period and calendar (the model time step is model.yaml
    numerics.step_seconds)."""

    start: str = option(doc="First model time, ISO date-time (e.g. 2007-05-01T00:00).",
                        fortran="start_ymd, start_tod")  # fmt: skip
    length: str = option(
        doc="How long the run lasts: a number of days (e.g. 1 day, 31 days), of model steps "
        "(e.g. 48 steps), or until a date-time (e.g. until 2001-01-01T00:00).",
        fortran="stop_option, stop_n",
    )
    calendar: Literal["gregorian", "noleap"] = option(
        "gregorian",
        "Model calendar, as CTSM's GREGORIAN / NO_LEAP: dates, solar geometry, run length and "
        "output records follow it. noleap only for grid runs (tower forcing is a real time "
        "series).",
    )


#: A ``files`` entry: one path or a list; ``{tower}``, ``{year}`` and ``{month}`` (e.g.
#: ``{year}-{month:02d}.nc``) are expanded over the run's months.
Files = tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class MeteorologySection:
    """Tower meteorology."""

    files: Files = option(
        doc="Tower forcing netCDF file(s); may be split in time with {year} and {month} "
        "(e.g. {year}-{month:02d}.nc) and shared by towers with {tower}.",
        fortran="fin_tower",
        path=True,
    )
    interval_seconds: int = option(
        1800,
        "Time between the records of the forcing files (s). The model step (model.yaml "
        "numerics.step_seconds) must divide it; shorter steps interpolate the forcing.",
    )
    interpolation: Literal["none", "three_point"] = option(
        "three_point",
        "Forcing within a model step's sub-steps: three-point interpolation of the previous, "
        "current and next records (CLM-ml v2), or the step's record unchanged.",
        fortran="met_type 3/0",
    )
    shortwave: Literal["coszen", "linear", "constant"] = option(
        "coszen",
        "Shortwave radiation from the data interval to a shorter model step: following the "
        "cosine of the solar zenith angle (conserves interval means), linear, or held.",
    )
    precipitation: Literal["constant", "linear"] = option(
        "constant",
        "Precipitation from the data interval to a shorter model step: held (conserves "
        "totals) or linear.",
    )


@dataclasses.dataclass(frozen=True)
class ClmHistorySection:
    """The CLM history file of the tower."""

    file: str = option(
        doc="CLM history file (netCDF, one patch, at the forcing interval) with ELAI, ESAI, "
        "H2OSOI and TSOI: the initial soil temperature, and LAI, SAI and soil water when their "
        "source is clm_history.",
        fortran="fin_clm",
        path=True,
    )
    start: str = option(doc="Date-time of the history file's first record.",
                        fortran="clm_start_ymd, clm_start_tod")  # fmt: skip


@dataclasses.dataclass(frozen=True)
class SoilMoistureAdjustmentSection:
    """Soil water adjusted to observations (CHATS)."""

    file: str = option(doc="Adjustment factors per CLM history record (netCDF, FACTOR).",
                       fortran="fin_soil_adjust", path=True)  # fmt: skip
    levels: int = option(3, "Number of soil levels, from the top, multiplied by the factor.",
                         fortran="nlev_soil_adjust")  # fmt: skip


@dataclasses.dataclass(frozen=True)
class LandDataSection:
    """Prescribed leaf and stem area: from the CLM history file, or from files with daily or
    monthly records (netCDF with time in "days since ..."), interpolated linearly between the
    middles of their intervals."""

    source: Literal["clm_history", "file"] = option(
        "clm_history", "Where the values come from: the CLM history file, or files."
    )
    files: Optional[Files] = option(None, "The netCDF file(s).", path=True,
                                    when={"source": "file"}, required=True)  # fmt: skip
    interpolation: Literal["linear"] = option(
        "linear", "Between the middles of the files' intervals.", when={"source": "file"}
    )


@dataclasses.dataclass(frozen=True)
class SoilWaterSection(LandDataSection):
    """Prescribed volumetric soil water (H2OSOI), as the leaf and stem area."""

    adjustment: Optional[SoilMoistureAdjustmentSection] = option(
        None, "Soil water adjusted to observations by factors (CHATS); null: none."
    )


@dataclasses.dataclass(frozen=True)
class ForcingSection:
    """Meteorology and prescribed land data."""

    meteorology: MeteorologySection = option(doc="Tower meteorology.", applies="towers")
    clm_history: ClmHistorySection = option(doc="The CLM history file.", applies="towers")
    gridded: GriddedForcingSection = option(
        doc="Gridded forcing: CRUJRA datm streams, CO2, monthly CLM soil history (vegetation "
        "comes from the surface data).",
        applies="grid",
    )
    vegetation: LandDataSection = option(
        LandDataSection(), "Leaf and stem area index.", applies="towers"
    )
    soil_water: SoilWaterSection = option(
        SoilWaterSection(), "Volumetric soil water.", applies="towers"
    )


def parse_length(text: str, key: str):
    """``time.length``: ``("days", 31.0)``, ``("steps", 48)`` or ``("until", "2001-01-01T00:00")``."""
    words = str(text).split()
    if len(words) == 2 and words[0] == "until":
        return "until", words[1]
    if len(words) == 2:
        try:
            n = float(words[0])
        except ValueError:
            n = None
        if n is not None and n > 0 and words[1] in ("day", "days"):
            return "days", n
        if n is not None and n >= 1 and n == int(n) and words[1] in ("step", "steps"):
            return "steps", int(n)
    raise ConfigError(
        f"{key}: {text!r}: expected e.g. '1 day', '31 days', '48 steps' or "
        "'until 2001-01-01T00:00'"
    )


def parse_chunk(text: str, key: str) -> Optional[int]:
    """``"10 days"`` -> 10; ``"1 month"`` -> None (calendar months), decision 33."""
    parts = str(text).split()
    if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) >= 1:
        if parts[1] in ("day", "days"):
            return int(parts[0])
        if parts[1] in ("month", "months") and int(parts[0]) == 1:
            return None
    raise ConfigError(f"{key}: {text!r}: expected e.g. '10 days' or '1 month'")


def parse_every(text: Optional[str], key: str):
    """``"3 months"`` -> ``(3, "month")``; ``"end_of_run"`` (or None) -> ``None``."""
    if text is None or text == "end_of_run":
        return None
    parts = str(text).split()
    units = {"month": "month", "months": "month", "year": "year", "years": "year"}
    if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) < 1 or parts[1] not in units:
        raise ConfigError(f"{key}: {text!r}: expected e.g. '1 month', '6 months' or '10 years'")
    return int(parts[0]), units[parts[1]]


@dataclasses.dataclass(frozen=True)
class ExecutionSection:
    """How the run is divided into compiled chunks, where it runs and how often it saves
    restarts. None of this changes the results."""

    chunk: str = option(
        "10 days",
        "Compiled chunk length: 'N days' (one compiled program; a short last chunk is padded) "
        "or '1 month' (calendar months). Sets how much forcing is in memory and how often the "
        "checks run.",
    )
    restart_every: str = option(
        "end_of_run",
        "When restart files are written: end_of_run, or every 'N months' or "
        "'N years' (and at the end).",
    )
    devices: int = option(
        1, "Number of devices (e.g. 4 GPUs on a Derecho node) the patches are spread over."
    )
    # Test and timing settings (hidden): pad the domain to this many patches / gridcells
    # (weight-0 copies, not in the output), so domains of different sizes share one program
    patch_count: Optional[int] = option(None, hidden=True)
    gridcell_count: Optional[int] = option(None, hidden=True)


@dataclasses.dataclass(frozen=True)
class InitialConditionsSection:
    """The state before the first step."""

    source: Literal["clm_history", "restart"] = option(
        "clm_history",
        "clm_history: canopy air and leaves from the first forcing record, soil temperature "
        "from the CLM history (towers) or the monthly soil history (grid); restart: a restart "
        "file of an earlier run.",
    )
    restart_file: Optional[str] = option(None, "The restart file.", path=True,
                                         when={"source": "restart"}, required=True)  # fmt: skip


@dataclasses.dataclass(frozen=True)
class OutputStreamSection:
    """One output stream, accumulated during the run (decision 16)."""

    name: str = option(doc="Name of the stream (in its file names).")
    frequency: Literal["step", "hourly", "daily", "monthly", "yearly"] = option(
        "step", "How often a record is written."
    )
    statistic: Literal["mean", "instantaneous", "min", "max"] = option(
        "mean", "What a record holds for its interval."
    )
    variables: tuple[str, ...] = option(
        ("all",),
        "Variables: all, or names from clm_ml.outputs.VARIABLES (e.g. " "latent_heat_flux, gpp).",
    )
    level: tuple[Literal["patch", "gridcell"], ...] = option(
        ("patch", "gridcell"),
        "Patch values, gridcell means (over the patches, by area), or both. A grid run's "
        "gridcell file is on the latitude-longitude grid of the surface data (the region's "
        "box), NaN where no "
        "gridcell was run; its values are per m2 of vegetated land, and it holds area, landfrac "
        "and vegetated_fraction (a total is value * area * landfrac * vegetated_fraction).",
    )
    file_per: Literal["run", "year", "month"] = option("run", "One file per run, year or month.")


@dataclasses.dataclass(frozen=True)
class OutputSection:
    """What the run writes."""

    directory: str = option("output", "Output directory.", path=True)
    streams: tuple[OutputStreamSection, ...] = option(
        (OutputStreamSection(name="step"),), "The output streams."
    )
    legacy_ascii: bool = option(
        False,
        "Also write the six Fortran-format .out files (one patch; for the comparison with "
        "Fortran CLM-ml v2 only).",
        applies="towers",
    )


@dataclasses.dataclass(frozen=True)
class RunFile:
    """The contents of ``run.yaml``."""

    time: TimeSection = option(doc="")
    forcing: ForcingSection = option(doc="")
    initial_conditions: InitialConditionsSection = option(InitialConditionsSection(), "")
    output: OutputSection = option(OutputSection(), "")
    execution: ExecutionSection = option(ExecutionSection(), "")
