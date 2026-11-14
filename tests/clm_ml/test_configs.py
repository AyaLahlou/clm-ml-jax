"""The experiment folders in ``configs/`` (decision 91): each is its compset written out by
``clm-ml new``, complete and current (every file as the code writes it today)."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from clm_ml.config import ModelConfig, model_yaml_text
from clm_ml.domain import DomainFile
from clm_ml.experiment import compset_names, map_paths, new_experiment
from clm_ml.io.yaml_schema import from_dict, read_yaml, render
from clm_ml.params import Params
from clm_ml.params_convert import params_table, params_text
from clm_ml.run import RunFile

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
FOLDERS = sorted(p.parent.name for p in CONFIGS.glob("*/run.yaml"))


def _objects(folder: Path):
    """The four files of ``folder`` as objects (strict), every path absolute."""
    kind = read_yaml(folder / "domain.yaml")["type"]

    def absolute(p):
        q = Path(p).expanduser()
        return str(q if q.is_absolute() else (folder / q).resolve())

    run = from_dict(RunFile, read_yaml(folder / "run.yaml"), source="run.yaml", strict=True,
                    context=kind)  # fmt: skip
    domain = from_dict(DomainFile, read_yaml(folder / "domain.yaml"), source="domain.yaml",
                       strict=True, context=kind)  # fmt: skip
    model = from_dict(ModelConfig, read_yaml(folder / "model.yaml"), source="model.yaml",
                      strict=True)  # fmt: skip
    params = Params.from_yaml(folder / "params.yaml")
    # the data paths absolute; the output directory stays the folder's own
    output = dataclasses.replace(run.output, directory=run.output.directory)
    run = dataclasses.replace(map_paths(run, absolute), output=output)
    return kind, run, map_paths(domain, absolute), model, params


def _body(text: str) -> str:
    """The text after the header comment (the settings and their explanations)."""
    return text.split("\n\n", 1)[1]


def test_every_compset_has_a_folder():
    assert FOLDERS == compset_names()


@pytest.mark.parametrize("name", FOLDERS)
def test_folder_is_its_compset(name, tmp_path):
    """configs/<name> holds the same settings as `clm-ml new --from <name>` writes."""
    fresh = new_experiment(tmp_path / name, name)
    a, b = _objects(CONFIGS / name), _objects(fresh)
    hint = f"configs/{name} differs from its compset: `clm-ml new configs/{name} --from {name} --force`"
    assert a[:4] == b[:4], hint
    for k in a[4].values:
        np.testing.assert_array_equal(a[4].values[k], b[4].values[k])


@pytest.mark.parametrize("name", FOLDERS)
def test_folder_is_complete_and_current(name):
    """Every file lists every setting with today's explanations (as the code writes it)."""
    folder = CONFIGS / name
    kind = read_yaml(folder / "domain.yaml")["type"]
    run = from_dict(RunFile, read_yaml(folder / "run.yaml"), source="run.yaml", strict=True,
                    context=kind)  # fmt: skip
    domain = from_dict(DomainFile, read_yaml(folder / "domain.yaml"), source="domain.yaml",
                       strict=True, context=kind)  # fmt: skip
    model = ModelConfig.from_yaml(folder / "model.yaml")
    params = Params.from_yaml(folder / "params.yaml")
    hint = f"run `clm-ml new configs/{name} --from {name} --force`"
    assert _body((folder / "run.yaml").read_text()) == _body(render(run, "x", context=kind)), hint
    assert _body((folder / "domain.yaml").read_text()) == _body(
        render(domain, "x", context=kind)
    ), hint
    assert _body((folder / "model.yaml").read_text()) == _body(model_yaml_text(model)), hint
    assert _body((folder / "params.yaml").read_text()) == _body(
        params_text(params_table(params), ["x"])
    ), hint
