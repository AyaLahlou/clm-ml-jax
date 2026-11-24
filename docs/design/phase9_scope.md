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

## 0.5 Phase 9.0: clean-up first (decision 128, 2026-11-24)

Before the documentation: 9.0a the code's comments, docstrings, messages and option
explanations (no phase, decision, issue or refactor references; Fortran module and
subroutine; departures stated plainly), in groups (processes; step and state; driver, setup
and forcing; configuration and io, then `configs/` regenerated; tests, timing, jobs). Every
commit is checked to leave the code identical apart from docstrings, comments and strings,
plus bit for bit and the noise twin. 9.0b the documents and other files (development logs
deleted, CLAUDE.md shortened, README and CONTRIBUTING cleaned). 9.0c a discussion of what
documentation to write (§2 below), before any is written.

Status: 9.0a done 2026-11-24 (4f33cd7); 9.0b done 2026-11-24 (13 development logs deleted,
CLAUDE.md shortened, the remaining documents cleaned); 9.0c done 2026-11-24 (decision 129). 9.1-9.6 built
2026-11-24 on `phase9/docs`; GitHub Pages to be enabled in the repository settings.

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

## 3. Packages (narrowed by decision 129, 2026-11-24)

Goal: colleagues can pick the model up and get started quickly; not a formal release.

| Package | Content | Open items |
|---|---|---|
| 9.1 | Sphinx site in `docs/` (MyST, furo), CI build with warnings as errors, GitHub Pages on every push to `main` | — |
| 9.2 | Home and getting started: install, run one CHATS7 day, read the output with xarray, a plot, `--set`, a paragraph on `clm-ml convert` | O45, O48 |
| 9.3 | Your own tower: `domain.yaml`, forcing and CLM history files, layer counts, `params.yaml` | — |
| 9.4 | Settings reference generated from the code (option declarations, `PARAM_SPECS`) | O46 |
| 9.5 | How the model works (existing page) with known limitations and output caveats | O51 |
| 9.6 | Examples: `run_tower.py`, `run_global.py` and `plot_global_diag.py` (kept apart, decision 110), new `sensitivity_tower.py`; tower examples in CI (short) | O47 |

Not done (decision 129): calibration example, per-topic guides, Fortran-users page, API
reference, changelog, version tag and release.

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
