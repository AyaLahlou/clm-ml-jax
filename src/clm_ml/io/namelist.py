"""
Convert a Fortran-style ``clmML_inparm`` namelist into an experiment folder (decision 91).

Used by ``clm-ml convert``. Paths in the namelist are interpreted the way
the legacy driver did (``../input_files/...`` relative to ``data/namelists``) and
written relative to the output directory.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

from clm_ml.io.yaml_schema import ConfigError

#: The repository's data directory (the CHATS7 namelists' ``../input_files``)
_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def _resolve_legacy_path(raw: str, namelist_dir: Path) -> Path | None:
    """The legacy driver's search order: as given, relative to the namelist, then with
    leading ``../`` stripped under the repository's ``data/``."""
    raw = raw.strip()
    if not raw:
        return None
    p = Path(raw)
    for candidate in (p, namelist_dir / p):
        if candidate.exists():
            return candidate.resolve()
    parts = p.parts
    while parts and parts[0] == "..":
        parts = parts[1:]
    candidate = _DATA_DIR / Path(*parts)
    return candidate.resolve() if candidate.exists() else (namelist_dir / p).resolve()


def _datetime(ymd: int, tod: int) -> dt.datetime:
    day = dt.datetime.strptime(str(int(ymd)), "%Y%m%d")
    return day + dt.timedelta(seconds=int(tod))


def convert_namelist(namelist_path: Path, out_dir: Path) -> Path:
    """Write the experiment folder of ``namelist_path`` into ``out_dir``: the four complete
    files (decision 91), the tower from its compset, ``pftcon_val = 1`` the ``chats``
    parameters (``clm5_defaults`` otherwise). Returns the folder."""
    import f90nml

    namelist_path = Path(namelist_path).resolve()
    nml = f90nml.read(str(namelist_path))
    if "clmml_inparm" not in nml:
        raise ConfigError(f"{namelist_path.name}: no &clmML_inparm group")
    g = nml["clmml_inparm"]
    known = {
        "tower_name", "start_ymd", "start_tod", "stop_option", "stop_n", "clm_start_ymd",
        "clm_start_tod", "fin_tower", "fin_clm", "clm_phys", "fin_soil_adjust",
        "nlev_soil_adjust", "dirout", "met_type", "dpai_min", "pftcon_val", "progress_nsteps",
    }  # fmt: skip
    unknown = sorted(set(g) - known)
    if unknown:
        raise ConfigError(f"{namelist_path.name}: cannot convert unknown setting(s) {unknown}")

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def rel(raw: str) -> str | None:
        """Relative to the output directory when nearby, otherwise absolute."""
        path = _resolve_legacy_path(str(raw), namelist_path.parent)
        if path is None:
            return None
        relative = os.path.relpath(path, out_dir)
        return str(path) if relative.startswith(os.path.join("..", "..", "..")) else relative

    tower = str(g["tower_name"]).strip()
    start = _datetime(g.get("start_ymd"), g.get("start_tod", 0))
    stop_option = str(g.get("stop_option", "ndays")).strip()
    stop_n = int(g.get("stop_n", 1))
    if stop_option == "ndays":
        length = f"{stop_n} day" + ("s" if stop_n != 1 else "")
    elif stop_option == "nsteps":
        length = f"{stop_n} step" + ("s" if stop_n != 1 else "")
    else:
        raise ConfigError(f"{namelist_path.name}: stop_option {stop_option!r} not supported")
    time = {"start": start.isoformat(timespec="minutes"), "length": length}
    clm_phys = str(g.get("clm_phys", "CLM5_0")).strip()
    if clm_phys != "CLM5_0":
        raise ConfigError(
            f"{namelist_path.name}: clm_phys {clm_phys!r} is not supported (CLM5_0 only: "
            "the model has the CLM soil layer structure only)"
        )

    met_type = int(g.get("met_type", 0))
    if met_type not in (0, 3):
        raise ConfigError(f"{namelist_path.name}: met_type {met_type} is not supported (0 or 3)")
    forcing = {
        "meteorology": {
            "files": rel(g["fin_tower"]),
            "interval_seconds": 1800,  # CLM-ml v2 tower forcing: 30 minutes
            "interpolation": "three_point" if met_type == 3 else "none",
        },
        "clm_history": {
            "file": rel(g["fin_clm"]),
            "start": _datetime(g.get("clm_start_ymd"), g.get("clm_start_tod", 0)).isoformat(
                timespec="minutes"
            ),
        },
    }
    adjust = rel(g.get("fin_soil_adjust", ""))
    if adjust:
        forcing["soil_water"] = {"source": "clm_history"}
        forcing["soil_water"]["adjustment"] = {
            "file": adjust,
            "levels": int(g.get("nlev_soil_adjust", 0)),
        }

    run = {
        "time": time,
        "forcing": forcing,
        "initial_conditions": {"source": "clm_history"},
        "output": {
            # an absolute dirout is kept; a relative one (the CHATS7 namelists'
            # ../output_files/) becomes "output" next to run.yaml, not inside the repository
            "directory": (
                str(g["dirout"]).strip()
                if Path(str(g.get("dirout", "")).strip()).is_absolute()
                else "output"
            ),
            "streams": [{"name": "step", "frequency": "step", "variables": ["all"]}],
        },
    }
    recipe = {
        "run": run,
        "domain": {"type": "towers", "towers": [tower]},  # the tower's compset
        "model": {"canopy_layers": {"merge_thin_layers_below": float(g.get("dpai_min", 0.01))}},
        "params": "chats" if int(g.get("pftcon_val", 0)) == 1 else "clm5_defaults",
    }
    from clm_ml.experiment import compose, version, write_experiment

    run_file, domain_file, config, params = compose(recipe, namelist_path.name)
    return write_experiment(
        out_dir, run_file, domain_file, config, params,
        generated=f"Converted from {namelist_path.name} by clm-ml {version()} (`clm-ml convert`).",
        params_about=f"Copied from the compset {recipe['params']} (pftcon_val "
        f"{int(g.get('pftcon_val', 0))}).",
        force=True,
    )  # fmt: skip
