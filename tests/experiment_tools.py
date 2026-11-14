"""Experiment folders for tests (decision 91): a complete folder from a compset or a namelist,
then changes to its files (dicts merged into them, or functions that edit them)."""

from __future__ import annotations

from pathlib import Path

FILES = {"run": "run.yaml", "domain": "domain.yaml", "model": "model.yaml"}


def make_experiment(folder, *, compset=None, namelist=None, **changes) -> Path:
    """Write an experiment folder (``compset``, default ``chats7_1day``, or ``namelist``) and
    apply ``changes`` (:func:`edit`). Returns the folder."""
    from clm_ml.experiment import new_experiment
    from clm_ml.io.namelist import convert_namelist

    folder = Path(folder)
    if namelist is not None:
        convert_namelist(Path(namelist), folder)
    else:
        new_experiment(folder, compset or "chats7_1day", force=True)
    return edit(folder, **changes)


def chats7_tower() -> dict:
    """The CHATS7 tower compset (``clm_ml/compsets/towers/chats7.yaml``) as a dict."""
    import yaml

    from clm_ml.experiment import compsets_dir

    return yaml.safe_load((compsets_dir() / "towers" / "chats7.yaml").read_text())["CHATS7"]


def edit(folder, *, run=None, domain=None, model=None, params=None) -> Path:
    """Change an experiment folder's files: a dict is merged into the file (nested sections
    merged, other values replaced: so towers given in a dict are added to the file's; give a
    function to replace them), a function gets the file's contents to edit in place;
    ``params``: a compset name, a params file or a :class:`clm_ml.params.Params`, written as the
    folder's complete params.yaml. The files are then completed (`clm-ml upgrade`: keys left
    out, e.g. of a new output stream, at their defaults)."""
    import yaml

    from clm_ml.experiment import upgrade_experiment

    folder = Path(folder)
    for key, change in (("run", run), ("domain", domain), ("model", model)):
        if change is None:
            continue
        path = folder / FILES[key]
        data = yaml.safe_load(path.read_text()) or {}
        if callable(change):
            change(data)
        else:
            _merge(data, change)
        path.write_text(yaml.safe_dump(data, sort_keys=False))
    if params is not None:
        from clm_ml.params import Params
        from clm_ml.params_convert import params_table, write_table

        table = params if isinstance(params, Params) else Params.from_yaml(params)
        write_table(params_table(table), folder / "params.yaml", ["test parameters"])
    if any(c is not None for c in (run, domain, model)):
        upgrade_experiment(folder)
    return folder


def _merge(data: dict, change: dict) -> None:
    for k, v in change.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            _merge(data[k], v)
        else:
            data[k] = v
