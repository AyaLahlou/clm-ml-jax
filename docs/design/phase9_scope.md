# Phase 9: documentation and release (scope)

Written at the close of Phase 8a (2026-11-01) as the scope of Phase 8b; moved to Phase 9 when
the maintainer reduced 8b (decision 103). Phase 9 documents the model as it now is and
releases version 1.0.0. **No science changes** in Phase 9: anything found goes into
`open_items.md`. Start by discussing the decisions in §2 with the maintainer (CLAUDE.md: stop
before a new phase to discuss it), on a new branch `phase9/docs` from `main`.

## 0. Already done in Phase 8b (2026-11-01, decision 103)

- `README.md`: install, run the CHATS7 day (`clm-ml new | check | run`), read the output with
  xarray, change settings, pointers to other towers and global runs.
- `docs/how_the_model_works.md`: how the code is organized, a skeleton of what runs when
  (host and compiled parts), static/data/differentiable values, the differences from Fortran
  (how the code is written, and the science departures), gradients.
- `examples/run_tower.py` (any tower; CHATS7 by default; prints mean fluxes, saves a figure of
  the fluxes and canopy air temperature profiles) and `examples/run_global.py` (the 1° grid,
  whole globe by default, `--region` for a box; maps of the gridcell means). Linted in CI;
  not yet run in CI (9.5).

These are starting points for 9.2, 9.4 and 9.5, not replacements.

## 1. What Phase 9 documents (settled in 8a)

- Experiment folders: `run.yaml`, `domain.yaml`, `model.yaml`, `params.yaml`, complete and
  strict, `clm-ml new | check | run | upgrade | convert | params` (decisions 91, 93).
- The public API: `load_experiment`, `new_experiment`, `upgrade_experiment`, `Experiment`,
  `Model` (decision 96); everything else is internal.
- The compsets and `configs/` (CHATS7 one day, the May 2007 Fortran comparison, the 1° grid
  run), the gridded streams' time axis (decision 102).
- Authorship, citation and the canonical repository (decision 97); release on GitHub, 1.0.0
  tagged at the end of Phase 9, no PyPI for now (decisions 98, 103).
- How the model works: `model_overview.md` and the schematic.

## 2. Decisions to make at the start of Phase 9

1. **Documentation tool and hosting.** Recommended: Sphinx with MyST Markdown and
   autodoc/napoleon (as jax-gcm), hosted on Read the Docs, or GitHub Pages built by CI.
   Alternative: MkDocs Material with mkdocstrings (lighter). Either needs new dev packages:
   ask before installing (CLAUDE.md).
2. **Examples: scripts or notebooks.** Recommended: plain Python scripts under `examples/`
   (run in CI, diff-friendly), rendered into the docs; notebooks optional.
3. **The calibration example (O47).** A twin experiment (recover a known Vcmax25 or iota
   from a synthetic CHATS7 run; needs no new data) or calibration against CHATS observations
   (needs observation files in the repository: which, and their licence).
4. **Where the user documentation lives.** Recommended: `docs/` for users (getting started,
   guides, reference, examples), `docs/design/` kept as the development record, linked but
   not part of the user guide.

## 3. Packages

| Package | Content | Open items |
|---|---|---|
| 9.1 | Documentation site: tool, structure, theme, CI build with warnings as errors | — |
| 9.2 | Getting started: install, `clm-ml new`, check, run one CHATS7 day, read the netCDF with xarray, a first plot; the four files explained | O45 |
| 9.3 | Reference of every setting, generated from the code (`yaml_schema.option` explanations, choices, defaults, Fortran names, `when` conditions); params table from `PARAM_SPECS`; a test that it is current | O46 |
| 9.4 | Guides: your own tower (domain.yaml), your own parameters (params.yaml), output streams, restarts and long runs, grid runs, GPUs and devices, calibration with `Model` | — |
| 9.5 | Examples run in CI (short versions): gradient calibration, ensemble (`vmap` over parameters), multi-tower and multi-patch, grid | O47 |
| 9.6 | For Fortran CLM-ml users: namelist keys and option codes to the YAML settings, generated from `clm_ml/io/namelist.py` and `config.to_fortran_codes` | O48 |
| 9.7 | API reference of the five public names (docstrings), and a pass over docstrings still worded in legacy terms | O49 |
| 9.8 | Known issues and limitations page (from `open_items.md`, §4) and output caveats | O51 |
| 9.9 | CONTRIBUTING final pass (process variants, tests, regression policy) | O50 |
| 9.10 | Formatting clean-up: consistent style of the documents and docstrings, the README folded into the docs site, `docs/how_the_model_works.md` and `docs/design/model_overview.md` reconciled | — |
| 9.11 | Release 1.0.0: version, CHANGELOG (Phases 0-8 summarized from `decisions.md`), `CITATION.cff` version, tag and GitHub release; `main` moved to the release commit | O52, O24 |

## 4. Known issues the documentation must state (deferred to after Phase 8)

From `open_items.md`: O4 (Fortran's checks of the implicit solution not run), O5 (solver
stopping rules differ from Fortran, within the validation tolerance), O6 (transpiration and
canopy-evaporation latent heat use the last sub-step's latent heat on steps crossing 0 °C),
O18/O19 (schemes that are refused: Norman solar, well-mixed, Brent), O20/O66 (settings and
parameters that cannot be calibrated by gradient: the nitrogen profile coefficient; root
profile, C3/C4, default heights act only at setup), O28 (a failing patch still in its
chunk's gridcell sums), O30 (per-variable exclusion in gridcell sums), O32 (`Model.chunks()`
starts at step 1), O53 (round-off amplification in real 1° patches), O54 (no bare ground),
O58 (orbital parameters of the start year), and the model limitations O61-O64.

## 5. Testing in Phase 9

Per commit: lint, the fast tests, the docs build (warnings as errors) and the examples' short
versions. Code is not expected to change; if it does, the tiers of CLAUDE.md "Rules for
changes" apply (bit for bit and noise twin before each commit). At the end: all of
`tests/regression`, the gradient tier, the Fortran comparison of
`configs/chats7_may2007_fortran`, and a clean install from the tag in a fresh environment
(with the maintainer's permission).

## 6. Done when

The documentation builds in CI and is published; a new user can follow getting started from
a fresh clone; every setting appears in the generated reference; the examples run; the known
issues are stated; 1.0.0 is tagged and released and `main` points to it.
