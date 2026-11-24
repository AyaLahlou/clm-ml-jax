"""Sphinx extension: write the settings reference from the code.

Every setting of ``run.yaml``, ``domain.yaml`` and ``model.yaml`` is declared once in the
code with ``clm_ml.io.yaml_schema.option`` (explanation, default, Fortran name, when it
applies), and every parameter of ``params.yaml`` in ``clm_ml.params.PARAM_SPECS``. At the
start of each build this writes ``reference/settings.md`` from them, so the reference always
matches the code.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path

FILES = (
    ("run.yaml", "clm_ml.run", "RunFile", "When and with what: dates, forcing and land-data "
     "files, initial conditions, output streams, chunks and restarts."),
    ("domain.yaml", "clm_ml.domain", "DomainFile", "Where: towers (location, forcing height, "
     "soil, patches) or the gridcells of a CLM surface dataset."),
    ("model.yaml", "clm_ml.config", "ModelConfig", "How: the scheme of each process, numerics, "
     "canopy layers and soil. A change recompiles the model."),
)  # fmt: skip


def _unwrap(t):
    """(the dataclass inside ``t`` or None, how it nests: '', '.<name>' or '[]')."""
    origin = typing.get_origin(t)
    if origin in (typing.Union, types.UnionType):
        for a in typing.get_args(t):
            if a is not type(None):
                inner, how = _unwrap(a)
                if inner is not None:
                    return inner, how
        return None, ""
    if origin is dict:
        inner, _ = _unwrap(typing.get_args(t)[1])
        return inner, ".<name>"
    if origin in (tuple, list):
        inner, _ = _unwrap(typing.get_args(t)[0])
        return inner, "[]"
    if dataclasses.is_dataclass(t):
        return t, ""
    return None, ""


def _type_text(t) -> str:
    origin = typing.get_origin(t)
    if origin is typing.Literal:
        return " / ".join(f"`{a}`" for a in typing.get_args(t))
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(t) if a is not type(None)]
        text = " or ".join(_type_text(a) for a in args)
        return text + (" or empty" if len(args) < len(typing.get_args(t)) else "")
    if origin in (tuple, list):
        args = typing.get_args(t)
        return f"list of {_type_text(args[0])}" if args else "list"
    if origin is dict:
        return "mapping"
    return getattr(t, "__name__", str(t))


def _default_text(f: dataclasses.Field) -> str:
    if f.metadata.get("required"):
        return "required"
    if f.default is not dataclasses.MISSING:
        value = f.default
    elif f.default_factory is not dataclasses.MISSING:
        value = f.default_factory()
    else:
        return "required"
    if dataclasses.is_dataclass(value):
        return ""
    from clm_ml.io.yaml_schema import flow

    text = flow(value)
    return f"`{text}`" if len(text) < 40 else "see generated file"


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _rows(cls, prefix: str, rows: list, sections: list, inherited: tuple = ()):
    hints = typing.get_type_hints(cls)
    for f in dataclasses.fields(cls):
        meta = f.metadata
        if meta.get("hidden"):
            continue
        key = f"{prefix}{f.name}"
        inner, how = _unwrap(hints[f.name])
        notes = list(inherited)
        if meta.get("applies") and f"{meta['applies']} only" not in notes:
            notes.append(f"{meta['applies']} only")
        if meta.get("when"):
            when = " and ".join(
                f"`{k}`: " + " or ".join(f"`{x}`" for x in (v if isinstance(v, tuple) else (v,)))
                for k, v in meta["when"].items()
            )
            notes.append(f"used when {when}")
        if meta.get("fortran"):
            notes.append(f"Fortran: `{meta['fortran']}`")
        doc = meta.get("doc", "")
        if inner is not None:
            sections.append((f"{key}{how}", doc or (inner.__doc__ or "").strip(), notes))
            keep = tuple(n for n in notes if not n.startswith("Fortran"))
            _rows(inner, f"{key}{how}.", rows, sections, keep)
            continue
        rows.append((key, _type_text(hints[f.name]), _default_text(f), doc, notes))


def settings_markdown() -> str:
    import importlib

    out = [
        "# Settings reference",
        "",
        "Every setting of the four files of an experiment folder, generated from the code. "
        "The files written by `clm-ml new` list the same settings with the same explanations; "
        "`clm-ml check` prints where each value comes from. Settings marked *towers only* or "
        "*grid only* appear only in that kind of run; *used when* settings appear only with "
        "that choice above them. A key written `.<name>` is a name you choose (a tower, a "
        "patch); `[]` is a list.",
        "",
    ]
    for fname, module, cls_name, about in FILES:
        cls = getattr(importlib.import_module(module), cls_name)
        rows, sections = [], []
        _rows(cls, "", rows, sections)
        out += [f"## `{fname}`", "", about, ""]
        out += ["| Setting | Type or choices | Default | Explanation |", "|---|---|---|---|"]
        for key, typ, default, doc, notes in rows:
            text = _cell(doc) + (f" *({'; '.join(notes)})*" if notes else "")
            out.append(f"| `{key}` | {_cell(typ)} | {_cell(default)} | {text} |")
        out.append("")
    out += _params_markdown()
    return "\n".join(out) + "\n"


def _params_markdown() -> list:
    from clm_ml.params import PARAM_SPECS

    out = [
        "## `params.yaml`",
        "",
        "Every parameter of every plant functional type (PFT 1-16), under the PFT's name "
        "(e.g. `broadleaf_deciduous_temperate_tree.vcmax25`). Parameters are differentiable: "
        "`clm_ml.model.Model` gives gradients with respect to each of them, per PFT. Spectral "
        "parameters are `[visible, near-infrared]` pairs. *Optional* parameters are needed "
        "only by some runs (for example grid runs).",
        "",
        "| Parameter | Units | Description | Fortran (table: field) |",
        "|---|---|---|---|",
    ]
    for s in PARAM_SPECS:
        desc = _cell(s.description) + (" (spectral)" if s.spectral else "")
        desc += " *(optional)*" if s.optional else ""
        out.append(f"| `{s.name}` | {_cell(s.units) or '-'} | {desc} | "
                   f"`{s.legacy_table}`: `{s.legacy_field}` |")  # fmt: skip
    return out + [""]


def _write(app):
    path = Path(app.srcdir) / "reference" / "settings.md"
    path.parent.mkdir(exist_ok=True)
    text = settings_markdown()
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def setup(app):
    app.connect("builder-inited", _write)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
