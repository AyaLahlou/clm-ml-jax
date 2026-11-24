"""Restart files as netCDF.

A restart holds what a run needs to continue from the end of a chunk; the setup (constants,
parameters, layer grid) is rebuilt from the run's configuration, deterministically, so it is
not stored. One file per run, written beside the previous one and renamed over it, keeping
the previous one as ``<file>.prev`` (a run killed while writing resumes from it).

Schema (format version :data:`FORMAT_VERSION`; netCDF4, groups):

- global attributes: ``title``, ``format_version``, ``time`` (the restart's time: the end of
  the last step run, ISO 8601, model calendar), ``calendar`` (``GREGORIAN`` | ``NOLEAP``),
  ``layout`` (the device layout ``[devices, gridcells per device, C3 slots, C4 slots]`` as
  JSON, or ``null``), ``writers`` (the output files in progress, JSON);
- dimensions ``patch``, ``layer``, ``leaf`` (0 sunlit, 1 shaded) and ``soil_level``;
- group ``state``: the prognostic state (``clm_ml.state.State``) of every patch, one variable
  per field (``canopy_air_temperature`` (patch, layer) K, ...), with ``units`` and
  ``long_name``;
- variable ``alive`` (patch): 1 for the patches still in the run;
- group ``streams``, one subgroup per output stream (``stream_0``, ...) holding the record in
  progress (attributes ``name``, ``statistic``, ``pending``: 0 or 1, ``key`` (JSON, typed),
  ``end`` (ISO 8601)) and its partial buffers (``sum``/``count`` or ``value`` subgroups, one
  variable per output variable: dimensions ``record``, ``patch`` or ``gridcell``, then the
  variable's own axes; ``units``).

:func:`write_restart`/:func:`read_restart` take and return the restart as the run loop uses
it (``clm_ml.driver.run_streams``): ``{"version", "time", "state", "collectors", "alive",
"layout", "extra"}``.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import numpy as np

from clm_ml.io import netcdf

#: The restart format this code writes and reads (1-5: pickles of earlier phases)
FORMAT_VERSION = 6

#: State field -> (netCDF variable, dimensions after patch, units, long name)
STATE_FIELDS = {
    ("canopy", "air_temperature"): ("layer",), ("canopy", "vapor_pressure"): ("layer",),
    ("canopy", "co2"): ("layer",), ("canopy", "leaf_temperature"): ("layer", "leaf"),
    ("canopy", "leaf_water_potential"): ("layer", "leaf"),
    ("canopy", "intercepted_water"): ("layer",), ("canopy", "ground_temperature"): (),
    ("canopy", "acclimation_temperature"): (), ("canopy", "acclimation_steps"): (),
    ("soil", "temperature"): ("soil_level",),
}  # fmt: skip
UNITS = {
    "air_temperature": ("K", "canopy air temperature"),
    "vapor_pressure": ("Pa", "canopy air water vapor pressure"),
    "co2": ("umol mol-1", "canopy air CO2"),
    "leaf_temperature": ("K", "leaf temperature"),
    "leaf_water_potential": ("MPa", "leaf water potential"),
    "intercepted_water": ("kg m-2", "intercepted water"),
    "ground_temperature": ("K", "ground surface temperature"),
    "acclimation_temperature": ("K", "acclimation temperature (running mean)"),
    "acclimation_steps": ("1", "steps averaged into the acclimation temperature"),
    "temperature": ("K", "soil temperature"),
}


class RestartError(RuntimeError):
    """A restart file that cannot be used."""


# -- record keys ---------------------------------------------------------------------------


def _encode_key(key) -> str:
    """A stream record's key (``output_streams.record_key``) as typed JSON."""
    if isinstance(key, dt.datetime):
        return json.dumps({"type": "datetime", "value": key.isoformat()})
    if isinstance(key, dt.date):
        return json.dumps({"type": "date", "value": key.isoformat()})
    if isinstance(key, tuple):
        return json.dumps({"type": "month", "value": [int(k) for k in key]})
    return json.dumps({"type": "int", "value": int(key)})


def _decode_key(text: str):
    d = json.loads(text)
    kind, value = d["type"], d["value"]
    if kind == "datetime":
        return dt.datetime.fromisoformat(value)
    if kind == "date":
        return dt.date.fromisoformat(value)
    if kind == "month":
        return tuple(value)
    return int(value)


# -- nested buffers as groups --------------------------------------------------------------


def _dimension(group, name: str, size: int) -> str:
    """Dimension ``name`` of ``size`` visible from ``group`` (created in it if needed)."""
    g = group
    while g is not None:
        if name in g.dimensions:
            if len(g.dimensions[name]) == size:
                return name
            break
        g = g.parent
    if name not in group.dimensions:
        group.createDimension(name, size)
        return name
    return _dimension(group, f"{name}_{size}", size)


def _write_tree(group, tree: dict) -> None:
    """``{name: array | dict}`` into ``group``: dicts as subgroups, arrays as variables
    (names in a ``name`` attribute). A stream buffer's axes are ``record`` (the record in
    progress), then ``patch`` (or ``gridcell`` for ``gridcell:<variable>``), then the output
    variable's own axes (``axis_<k>``); output variables carry their ``units``."""
    from clm_ml.outputs import VARIABLES_BY_NAME

    for i, (name, value) in enumerate(tree.items()):
        if isinstance(value, dict):
            sub = group.createGroup(f"g{i}")
            sub.setncattr("name", name)
            _write_tree(sub, value)
            continue
        a = np.asarray(value)
        names = ["record", "gridcell" if name.startswith("gridcell:") else "patch"]
        dims = [
            _dimension(group, names[k] if k < 2 else f"axis_{k}", n) for k, n in enumerate(a.shape)
        ]
        var = group.createVariable(f"v{i}", a.dtype if a.dtype != bool else "i1", dims)
        var.setncattr("name", name)
        spec = VARIABLES_BY_NAME.get(name.split(":", 1)[-1])
        if spec is not None:
            var.units = spec.units
        var[...] = a.astype("i1") if a.dtype == bool else a


def _read_tree(group) -> dict:
    out = {}
    entries = [(v.getncattr("name"), v[...]) for v in group.variables.values()]
    entries += [(g.getncattr("name"), _read_tree(g)) for g in group.groups.values()]
    for name, value in entries:
        out[name] = value if isinstance(value, dict) else np.asarray(value)
    return out


# -- writing and reading -------------------------------------------------------------------


def write_restart(path: Path, content: dict, calendar: str = "GREGORIAN") -> None:
    """Write ``content`` (``clm_ml.driver.run_streams``'s restart) to the netCDF file
    ``path``: written beside it and renamed over it, the previous file kept as
    ``<path>.prev``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    state = content["state"]
    extra = content.get("extra") or {}
    with netcdf.dataset(tmp, "w", format="NETCDF4") as ds:
        ds.title = "CLM-ML-JAX restart"
        ds.format_version = FORMAT_VERSION
        ds.time = content["time"].isoformat()
        ds.calendar = calendar
        layout = content.get("layout")
        ds.layout = json.dumps(None if layout is None else [int(x) for x in layout])
        ds.writers = json.dumps(extra.get("writers"))
        temp = np.asarray(state.canopy.leaf_temperature)
        ds.createDimension("patch", temp.shape[0])
        ds.createDimension("layer", temp.shape[1])
        ds.createDimension("leaf", temp.shape[2])
        ds.createDimension("soil_level", np.asarray(state.soil.temperature).shape[1])
        g = ds.createGroup("state")
        for (part, name), dims in STATE_FIELDS.items():
            var = g.createVariable(f"{part}_{name}", "f8", ("patch",) + dims)
            var.units, var.long_name = UNITS[name]
            var[...] = np.asarray(getattr(getattr(state, part), name), dtype=np.float64)
        alive = ds.createVariable("alive", "i1", ("patch",))
        alive.long_name = "1: the patch is still in the run"
        alive[...] = np.asarray(content.get("alive", np.ones(temp.shape[0], bool)), dtype="i1")
        streams = ds.createGroup("streams")
        names = content.get("streams") or [None] * len(content.get("collectors", []))
        for i, pending in enumerate(content.get("collectors", [])):
            s = streams.createGroup(f"stream_{i}")
            if names[i] is not None:
                s.setncattr("name", names[i][0])
                s.statistic = names[i][1]
            s.pending = int(pending is not None)
            if pending is not None:
                key, end, buffers = pending
                s.key, s.end = _encode_key(key), end.isoformat()
                _write_tree(s, buffers)
    if path.exists():
        os.replace(path, path.with_name(path.name + ".prev"))
    os.replace(tmp, path)


def _read_one(path: Path) -> dict:

    from clm_ml.state import CanopyState, SoilState, State

    with open(path, "rb") as fh:
        if fh.read(1) == b"\x80":  # a pickle (formats 1-5: older restart files were pickles)
            raise RestartError(
                "a pickle restart (format 1-5, an older version of clm-ml): this code reads netCDF "
                f"restarts (format {FORMAT_VERSION}); run again from the start"
            )
    with netcdf.dataset(path) as ds:
        ds.set_auto_mask(False)  # every value as written (NaN, inf and spval included)
        version = int(getattr(ds, "format_version", -1))
        if version != FORMAT_VERSION:
            raise RestartError(
                f"restart format version {version}, this code reads version {FORMAT_VERSION} "
                "(write it again with the current code)"
            )
        g = ds.groups["state"]
        parts: dict = {"canopy": {}, "soil": {}}
        for part, name in STATE_FIELDS:
            parts[part][name] = np.asarray(g.variables[f"{part}_{name}"][...], dtype=np.float64)
        state = State(canopy=CanopyState(**parts["canopy"]), soil=SoilState(**parts["soil"]))
        collectors, streams = [], []
        for i in range(len(ds.groups["streams"].groups)):
            s = ds.groups["streams"].groups[f"stream_{i}"]
            attrs = s.ncattrs()
            streams.append((s.getncattr("name"), s.statistic) if "name" in attrs else None)
            if not int(s.pending):
                collectors.append(None)
                continue
            collectors.append((_decode_key(s.key), dt.datetime.fromisoformat(s.end), _read_tree(s)))
        layout = json.loads(ds.layout)
        return {
            "version": version,
            "time": dt.datetime.fromisoformat(ds.time),
            "calendar": ds.calendar,
            "state": state,
            "collectors": collectors,
            "streams": streams,
            "alive": np.asarray(ds.variables["alive"][...], dtype=bool),
            "layout": None if layout is None else tuple(layout),
            "extra": {"writers": json.loads(ds.writers)},
        }


def read_restart(path: Path) -> dict:
    """A restart written by :func:`write_restart`; falls back to ``<path>.prev`` if the file
    is missing or unreadable."""
    path = Path(path)
    errors = []
    for candidate in (path, path.with_name(path.name + ".prev")):
        try:
            content = _read_one(candidate)
        except Exception as exc:  # noqa: BLE001 - report every candidate
            errors.append(f"{candidate}: {exc}")
            continue
        if errors:
            print(f"warning: restart {path} not readable ({errors[0]}); using {candidate}",
                  flush=True)  # fmt: skip
        return content
    raise RestartError("no readable restart: " + "; ".join(errors))
