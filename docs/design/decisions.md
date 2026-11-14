# Decision register (CLM-ML-JAX modernization)

The meaningful decisions of the modernization in one place: what was decided, why, and
whether the model departs from Fortran CLM-ml v2 / CTSM because of it.

## How to use

- **One entry per decision**, grouped by topic. IDs are stable: never renumber; a new
  decision gets the next free ID (append it to its topic) in the same commit as the change.
- **Entry format:** `**N. Title**`, then one line `date · phase · who · status · departure`,
  then the decision (one or two sentences), the reason (one sentence) and `Source:` (the
  original IDs).
  - *who:* **maintainer** (decided by the maintainer) or **implementation** (decided while
    implementing; science choices are never decided this way).
  - *status:* in force / superseded by N / revised by N / completed (a one-time choice that
    has been carried out) / open (decided to wait).
  - *departure:* **Departure from Fortran: yes/no**: whether the decision makes the model
    compute something different from CLM-ml v2 (or, for global runs, from CTSM 5.4/CDEPS).
- **Superseded decisions stay**, marked, with a pointer to what replaced them.
- The older documents remain the detailed record (measurements, alternatives, probes):
  `modernization_plan.md` §0 and §8, `phase7_scope.md` (D1-D12), `phase7_decisions.md`
  (C0-C35), `global_input.md` (input decisions 1-8, G1-G20, open issues),
  `numerics_and_regression.md`, `phase6_progress.md` (review items R1-R11).
- Open questions and tasks are not decisions: they belong in [`open_items.md`](open_items.md).

### Old IDs to new IDs

| Old ID | New | Old ID | New | Old ID | New |
|---|---|---|---|---|---|
| plan §0 1 | 7 | plan §0 26 | 26 | C20 | 17, 67 |
| plan §0 2 | 8 | plan §0 27 | 14 | C21 | 17, 54, 81 |
| plan §0 3 | 21 | plan §0 28 | 27 | C22 | 62 |
| plan §0 4 | 22 | plan §0 29 | 28 | C23 | 63 |
| plan §0 5 | 23 | plan §0 30 | 49 | C24 | 47, 84 |
| plan §0 6 | 24 | plan §0 31 | 45 | C25 | 77 (plan of 7.4) |
| plan §0 7 | 61 | plan §0 32 | 50 | C26 | 63 |
| plan §0 8 | 9 | plan §0 33 | 38 | C27 | 55, 77, 83 |
| plan §0 9 | 12 | plan §0 34 | 42 | C28 | 15, 20, 38, 77, 78 |
| plan §0 10 | 10 | plan §0 35 | 15, 17, 18, 25, 47, 70, 76, 79, 86 | C29 | 83 |
| plan §0 11 | 11 | plan §0 36 | 47 | C30 | 11, 74, 85 |
| plan §0 12 | 36 | plan §0 37 | 19 | C31 | 19, 56, 77 |
| plan §0 13 | 40 | §8 1 | 7 | C32 | 73 |
| plan §0 14 | 41 | §8 2 | 24 | C33 | 75, 85 |
| plan §0 15 | 48 | §8 3 | 66 | C34 | 17 |
| plan §0 16 | 72 | §8 4 | 11 | C35 | 19 |
| plan §0 17 | 37 | §8 5 | 80 | G1, G2, G4-G8 | 43 |
| plan §0 18 | 80 | §8 6 | 74 | G3 | 42 |
| plan §0 19 | 24 | §8 7 | 65 | G9 | 44, 45 |
| plan §0 20 | 15 | §8 8 | 29 | G10, G11 | 46 |
| plan §0 21 | 80 | §8 9 | 13 | G12, G13 | 27 |
| plan §0 22 | 25 | D1 | 79 | G14 | 33 |
| plan §0 23 | 26 | D2 | 73 | G15 | 12, 14 |
| plan §0 24 | 39 | D3 | 47 | G16 | 35 |
| plan §0 25 | 30 | D4 | 42 | G17 | 34 |
| C0 | 69, 70 | D5 | 86 | G18 | 29 |
| C1 | 85 | D6 | 76 | G19 | 28 |
| C2, C7 | 75 (transition, superseded) | D7 | 67 | G20 | 49 |
| C3 | 70 | D8 | 17 | input dec. 1, 3 | 43 |
| C4 | 68 | D9 | 18 | input dec. 2 | 33 |
| C5 | 82 | D10 | 25 | input dec. 4, 5, 8 | 32 |
| C6 | 75, 76 | D11 | 15 | input dec. 6 | 14 |
| C8 | 76 | D12 | 70 | input dec. 7 | 27 |
| C9 | 53, 80 | R1 | 31 | numerics §1 | 58 |
| C10 | 67, 80 | R2 | 42 | numerics §2 | 53, 59 |
| C11 | 57 | R3, R6 | 73 | numerics §3 | 60 |
| C12, C13 | 17, 80 | R8 | 19 (obsolete) | numerics §4 | 53 |
| C14 | 55, 67 | R9 | 44 | numerics §4.4 | 51 |
| C15 | 62, 64 | issue 12 | 50 | numerics §5 | 52 |
| C16 | 17, 67, 80 | issue 14 | 49 | numerics §6 | 64, 66, 68 |
| C17 | 17, 54, 67 | issue 17a | 45 | numerics §6.1 | 69 |
| C18 | 16, 80 | issue 18 | 17 | | |
| C19 | 67, 81 | | | | |

---

## A. Working rules

**1. Science is the maintainer's decision; where a choice comes up, copy CLM**
2026-10-29/30 · Phase 6b · maintainer · in force · Departure from Fortran: no
Bugs and science questions are diagnosed first with probes, without changing code, and brought
to the maintainer; no shortcuts or dropped patches; where a modelling choice arises, CTSM 5.4
(or CLM-ml v1/v2) is copied; each science change is recorded as a decision.
Reason: the model is a scientific tool and must be accurate and traceable.
Source: CLAUDE.md ("Bugs and science"); `global_input.md` "Implementation decisions" preamble.

**2. Every phase gives the same answer; adapters first; refactors and science apart**
2026-10-27 · plan · maintainer · in force · Departure from Fortran: no
Each phase reproduces the previous phase's results; new interfaces go around the translated code
first and internals are rewritten later; refactors and science changes are separate commits.
Reason: any change in results can then be traced to one deliberate science change.
Source: plan §3.2 principle 8; CLAUDE.md "Rules for changes".

**3. Branches, commits and pace**
2026-10-28/30 · all · maintainer · in force · Departure from Fortran: no
One branch per phase; commit and push when the required tests pass; never open PRs; stop before
a new phase to discuss it; a measured slowdown above ~1.5× (timing test or gradient) is
discussed before going on; no changes to the maintainer's Python environment without asking.
Reason: the maintainer reviews at phase boundaries and keeps control of performance and the
environment.
Source: CLAUDE.md "Standing preferences". (Note: no phase branch has been merged into `main`.)

**4. Vocabulary**
2026-10-27 · plan · maintainer · in force · Departure from Fortran: no
**gridcell** (own forcing), **patch** (one PFT's canopy with an area weight; the unit of
computation), **soil column**, **tower** (a named entry of `towers.yaml`); "site" only for
observation locations; "timing test" for wall-clock measurements ("benchmark" keeps its paper
meaning: the reference multilayer simulation).
Reason: precise terms avoid confusion with CLM's land hierarchy and with the papers.
Source: plan §3.1; CLAUDE.md.

**5. Where decisions are recorded**
2026-10-30 · Phase 7 · maintainer · revised by this register · Departure from Fortran: no
Maintainer decisions went to plan §0 (and §8), implementation decisions to the phase decision
log (C1, C2, ...), global-input choices to `global_input.md`; this register now collects the
meaningful ones.
Reason: one place to see what was decided and why.
Source: `phase7_decisions.md` header; CLAUDE.md.

**6. Phase 8 in two parts**
2026-11-01 · Phase 8 · maintainer · revised by 103 · Departure from Fortran: no
Phase 8a: housekeeping and clean-up (science issues, repository organization, one open-items
document, this register, a schematic of the model); Phase 8b: the documentation and release.
Reason: the model and repository should be in final shape before they are documented.
Source: maintainer, 2026-11-01 (this session).

**103. Phase 8 ends with a short 8b; formal documentation and the release move to Phase 9**
2026-11-01 · Phase 8b · maintainer · in force · Departure from Fortran: no
Phase 8b is only: a README that says how to install and run the CHATS tower; one page,
`docs/how_the_model_works.md` (how the model is organized, how it works, how the JAX version
differs from Fortran, and a skeleton of what runs when); and two scripts, `examples/run_tower.py`
(any tower, CHATS7 by default) and `examples/run_global.py` (the 1° grid, whole globe by
default). Phase 8 closes with them. The documentation site, the generated settings reference,
the guides, the further examples (calibration, ensemble), the Fortran-users page, the API
reference, the formatting clean-up and the 1.0.0 release are Phase 9 (`phase9_scope.md`;
open items O45-O52, O24).
Reason: the maintainer's decision: enough for a user to install and run now; the formal
documentation later.
Source: maintainer, 2026-11-01.

**110. The global example is a case-setup script**
2026-11-05 · after Phase 8 · maintainer · revised by 112 (revises 103's `examples/run_global.py`) · Departure from Fortran: no
`examples/run_global.py` reads as instructions for setting up a case: settings at the top
(case folder under `$SCRATCH`, start, length, calendar, region, GPUs, chunk), then create
the case from the compset, write the settings into its files (`Experiment.write_copy`), print
the check summary and run (or leave the case for `clm-ml run`). No command-line options,
functions or plotting.
Reason: quick to reuse by changing a few values; the case folder records the run.
Source: maintainer, 2026-11-05.

## B. Configuration and user interface

**7. Names and platform**
2026-10-27 · plan · maintainer · in force · Departure from Fortran: no
Import as `import clm_ml`; install name `clm-ml-jax`; command-line tool `clm-ml`
(`run | check | convert | params`); Python ≥ 3.10.
Reason: short, conventional names; 3.10 is the oldest Python the dependencies support well.
Source: plan §0 1; §8 1.

**8. Readable option names**
2026-10-27 · plan · maintainer · in force · Departure from Fortran: no
Options are words (`stomatal_conductance: wue`); Fortran integer codes are accepted only by the
namelist converter.
Reason: configuration files should be readable without the Fortran source.
Source: plan §0 2.

**9. Configuration in YAML files**
2026-10-27 · plan · maintainer · in force · Departure from Fortran: no
`run.yaml` (the experiment: towers or grid, dates, forcing, initial conditions, output),
`model.yaml` (scheme choices and numerics: static), `params.yaml` (parameters per PFT:
differentiable), `towers.yaml` (tower descriptions; built-in or the user's); loaded into typed
objects that JAX runs on.
Reason: separates what changes the compiled program (model) from what can be differentiated
(params) and from the experiment (run).
Source: plan §0 8; §4.

**10. Plain YAML with our own validation**
2026-10-27 · plan · maintainer · in force · Departure from Fortran: no
Dataclasses with load-time validation whose messages name the file, key and allowed values; no
Hydra, no Flax.
Reason: few dependencies and clear errors for users who are not software engineers.
Source: plan §0 10; §6.8.

**11. Namelists: a permanent converter; the legacy driver retired**
2026-10-27, revised 2026-11-01 · plan, Phase 7.4f · maintainer · in force · Departure from Fortran: no
`clm-ml convert nl.*` writes run.yaml and model.yaml and stays permanently; the legacy
`clm-ml-offline` driver was kept until Phase 7 and then deleted (its namelists stay as test
data).
Reason: Fortran users can bring their namelists; one run path to maintain.
Source: plan §0 11; §8 4; C30.

**12. Parameters: a per-PFT table in YAML only**
2026-10-27, revised 2026-10-29 · plan, Phase 6b · maintainer · in force (revised by 89: no `extends`) · Departure from Fortran: no
Each patch looks up its PFT's row, so gradients come out per PFT; `params.yaml` used
`extends:` with overrides (until 89); params files are YAML only: a CLM parameter netCDF file is converted
once (`clm-ml params FILE.nc -o FILE.yaml`), which writes a complete table.
Reason: calibration targets PFT parameters; one format avoids two code paths.
Source: plan §0 9; G15; `phase6_progress.md` 2026-10-29 ("params files are YAML only").

**13. Built-in towers: CHATS7 only**
2026-10-27 · plan · maintainer · in force · Departure from Fortran: no
Only CHATS7 is in the built-in `towers.yaml`; other towers go in a user `towers.yaml` with
explicit layer counts where the default spacing gives too many layers.
Reason: CHATS7 is the only tower validated against Fortran with input data in the repository.
Source: plan §8 9 (and §8 2).

**14. Parameters for global runs: CLM6**
2026-10-29 · Phase 6b · maintainer · in force · Departure from Fortran: no (CTSM 6.0 values)
Global runs use `params: clm6` (the CTSM 6.0 parameter file converted) for every PFT, with the
CLM-ml-specific parameters (ι, plant hydraulics, ...) from `clm5_defaults`; CHATS keeps its
overrides (`params: chats`).
Reason: current CLM values for the CLM part; CLM has no values for the CLM-ml-specific part.
Source: plan §0 27 (first written "CLM5 defaults", corrected after review R11); input
decision 6; G15.

**15. Configuration passed explicitly; runs in-process**
2026-10-28, extended 2026-10-31 · Phase 5a, 7.4 · maintainer · in force · Departure from Fortran: no
No module globals hold configuration or model state: options are static arguments, tunable
numbers are differentiable parameters; a process can hold several models (any towers or soils);
the worker subprocess is gone and runs execute in-process.
Reason: several models per session (scheme comparisons, ensembles), in-process tests, and no
hidden state.
Source: plan §0 20; D11; plan §0 35; C28.

**16. Static choices versus differentiable numbers**
2026-10-27 · plan · maintainer · in force (one exception) · Departure from Fortran: no
Flags and scheme names that choose code paths are static; every numeric tunable is
differentiable and never hidden from `jax.grad`.
Reason: calibration and sensitivity analysis need gradients for every parameter.
Source: plan §3.2; CLAUDE.md (jax-gcm guidance). (Unclear: the nitrogen decay coefficient
`kn_val` is still a static option, C18, which this rule would make a parameter.)

**17. Unsupported variants are refused with a clear error**
2026-10-30 to 11-01 · Phase 6 review, 7.2, 7.4f · maintainer (D8, issue 18) and implementation · revised (code kept only in git history, C34) · Departure from Fortran: yes (variants not available)
These stay registered by name but stop at load with a `ConfigError`: Norman solar radiation,
the well-mixed flux-profile solution and the dataset profiles, Brent's method for the WUE
stomatal optimization (`wue_solver: brent`), two-point time interpolation (`met_type 2`, which
CLM-ml v2 itself refuses), and `mlcan_to_clm = 1` (writes CLM containers nothing reads
offline). Their legacy translations were deleted with the containers in 7.4f and are in git
history at `1f1b651`.
Reason: none was validated in the JAX translation (Norman fails at the first CHATS7 step;
Fortran itself stops after well-mixed); making them run compiled is work after Phase 8.
Source: D8; plan §0 35; C12; C13; C16 (issue 18); C17; C20; C21; C34. (Unclear: D8 said
"kept"; C34 asks whether that meant keeping the code in the tree.)

**18. `gs_solver` and `mlcan_to_clm` options kept**
2026-10-30 · Phase 7 scope · maintainer · revised by 17 · Departure from Fortran: no
The options stay in model.yaml.
Reason: they are Fortran options.
Source: D9. Revised: `brent` (C16) and `mlcan_to_clm = 1` (C21) are refused (entry 17).

**19. CLM soil only**
2026-11-01 · Phase 7 review · maintainer · in force · Departure from Fortran: no (CLM-ml v2
offers CLM4.5, never validated here)
One soil layer structure, CLM5's (20 hydrologically active, 25 ground levels), called "CLM";
model.yaml has no `soil.layer_structure`; CLM histories must have 20 `H2OSOI` levels;
`clm-ml convert` refuses `clm_phys = 'CLM4_5'`.
Reason: only CLM5 soil is ever used; the CLM4.5 path never ran (no reference).
Source: plan §0 37; C35; C31 (history validation); R8 (now obsolete).

**20. Errors and checks**
2026-10-27, revised 2026-10-31 · plan, 7.4 · implementation · in force · Departure from Fortran: no
YAML is validated at load (`clm-ml check` validates without running); model errors raise
`clm_ml.checks.ModelError` (a Python exception) instead of exiting the process as Fortran's
`endrun`.
Reason: runs are in-process (entry 15); a caller must be able to catch a failure.
Source: plan §6.8; C28.

**89. Params files are complete; every run names one**
2026-11-01 · Phase 8a · maintainer · in force (revised by 91: the run's params.yaml is a file of its folder) · Departure from Fortran: no
A params file gives every parameter of every PFT 1-16 itself: no `extends`, no values filled in
from another file (`extends` and missing values are errors; only the canopy roughness ratios,
used by grid runs alone, may be left out, and a grid run then refuses the file). `run.yaml`
must name a params file (no default). The built-in `chats` is now a complete table
(`clm5_defaults` with the walnut values; same numbers as before); a new parameter set is a
copy (`clm-ml params chats -o mine.yaml`) edited.
Reason: a run's parameters are all in the one file it names, so what a run used is never hidden
in a chain of files.
Source: maintainer, 2026-11-01 (open_items O17).

**90. `configs/`: runnable configurations with complete model.yaml files**
2026-11-01 · Phase 8a · maintainer · revised by 91 (to be implemented) · Departure from Fortran: no
Runnable experiments live in `configs/` (one folder each: `chats7_1day`, `chats7_may2007_fortran`,
`grid_1deg_2000`), replacing `docs/design/examples/`. `configs/model.yaml` lists every model
setting at its default with its explanation, generated from `clm_ml.config` (`clm-ml
model-yaml`; the explanations are in the code, field metadata); an experiment that changes
settings carries its own complete model.yaml; `clm-ml convert` writes a complete one; a test
checks every model.yaml in `configs/` is complete and current and every run.yaml loads. The
built-in params, towers and psihat files stay package data (used by name).
Reason: validated configurations as named, commented files (as jax-gcm); one source for the
defaults (the code), so the file cannot drift.
Source: maintainer, 2026-11-01 (open_items O17).

**91. Experiment folders: four complete files with fixed names, strict, with `clm-ml upgrade`**
2026-11-01 · Phase 8a · maintainer · in force (implemented 2026-11-01) · Departure from Fortran: no
An experiment is a folder with `run.yaml` (when, forcing, output), `domain.yaml` (towers or
grid), `model.yaml` (schemes, numerics) and `params.yaml` (every parameter of every PFT); a run
reads only these and the data files they name, never a compset by name. Every file is complete
(each key with its explanation, choices, default or REQUIRED); missing, unknown or misplaced
keys are errors, and `clm-ml upgrade DIR` rewrites a folder to the current layout keeping the
user's values. Folders are generated from compsets (`clm-ml new DIR --from COMPSET`); each setting
lives in one file; each run writes a copy of the four files into its output. Details:
[`experiment_folders.md`](experiment_folders.md).
Reason: one start point where every user setting is visible and the required ones are marked,
for ecologists; one definition of every option in the code (as jax-gcm), so files, errors and
reference documentation agree.
Implemented (2026-11-01): `clm_ml.experiment` (load, new, upgrade, the configuration copy),
`run.py`/`domain.py`/`config.py` options declared with `yaml_schema.option`, compsets in
`clm_ml/compsets/`, `configs/` the compsets written out. Behaviour that changed with it: the
model step is `numerics.step_seconds` (default 1800 s; no longer "null = the forcing
interval"); the tower forcing interval is run.yaml `forcing.meteorology.interval_seconds`
(was run.yaml `time.step_seconds` and towers.yaml `forcing_step_minutes`); towers are written
in domain.yaml (no towers file, no registry at run time); `--set` keys name the file first
(`run.`, `domain.`, `model.`, `params.<pft>.<name>`); hidden test settings
(`execution.patch_count`/`gridcell_count`, `grid.max_gridcells`) are given only as `--set` or
from Python; `clm-ml model-yaml` and `configs/model.yaml` (decision 90) removed. CHATS7 bit for
bit and noise twin unchanged.
Source: maintainer, 2026-11-01 (strict with upgrade, fixed names; names kept, data interval in
run.yaml, test-only settings hidden).

**96. The public API: five names**
2026-11-01 · Phase 8a · maintainer · in force · Departure from Fortran: no
`import clm_ml` offers `load_experiment`, `new_experiment`, `upgrade_experiment`,
`Experiment` and `Model`; they are documented and stable from 1.0. Everything else
(`clm_ml.processes`, `driver`, `setup`, ...) is internal and may change.
Reason: enough for runs, compsets, upgrades and calibration; a small surface to keep stable.
Source: open_items O21.

**97. Authorship, citation and repository**
2026-11-01 · Phase 8a · maintainer · in force · Departure from Fortran: n/a
Authors: Linnia Hawkins (lead) and Aya Lahlou (the original translation); the canonical
repository is github.com/linniahawkins/clm-ml-jax. `CITATION.cff`, the README citation and
badges, `pyproject.toml` authors and the LICENSE copyright (2024-2026, both) say so; users
also cite the CLM-ml v2 papers (Bonan et al. 2021, 2026).
Reason: the maintainer's decision.
Source: open_items O23, O37.

**98. Release: `main` at the end of 8a, 1.0.0 at the end of 8b, GitHub only**
2026-11-01 · Phase 8a · maintainer · in force, revised by 103 (`main` moved at the end of 8a, 2026-11-01; 1.0.0 at the end of Phase 9) · Departure from Fortran: n/a
`main` is fast-forwarded to the phase branch when 8a closes (no pull request; it holds nothing
of its own); version 1.0.0 is tagged at the end of 8b as a GitHub release; no PyPI for now
(install with pip from git).
Reason: the maintainer's decision.
Source: open_items O24.

**99. Stale files deleted**
2026-11-01 · Phase 8a · maintainer · completed · Departure from Fortran: n/a
`.github/copilot-instructions.md` (described deleted packages; CLAUDE.md is the guidance for
AI assistants) and the unused namelists `nl.CHATS7.debug`, `nl.CHATS7.test3`,
`nl.CHATS7.04.2007` and `nl.US-UMB.test` (scratch paths) are deleted; the tests use
`nl.CHATS7.1day`, `nl.CHATS7.50steps` and `nl.CHATS7.05.2007`.
Reason: the maintainer's decision.
Source: open_items O36, O40.

**107. The grid compset runs the whole globe; the region word is `global`**
2026-11-04 · after Phase 8 · maintainer · in force · Departure from Fortran: no
The compset `grid_1deg_2000` has no region, so it runs the whole globe (the default); a
box in `domain.yaml grid.region` runs a region. The named word for the whole globe is
`global` (was `whole_globe`, decision 93); `clm-ml upgrade` rewrites `whole_globe` and the
old null as `global`.
Reason: a global compset should run the globe; `global` is the plainer word (maintainer).
Source: maintainer, 2026-11-04.

## C. Spatial structure: gridcells, patches, layers, soil

**21. One tower up to global**
2026-10-27 · plan · maintainer · in force · Departure from Fortran: no
A run is one or many gridcells with no lateral exchange.
Reason: the same code serves towers and global runs.
Source: plan §0 3.

**22. The patch is the unit of computation**
2026-10-27 · plan · maintainer · in force · Departure from Fortran: no
Physics is written for one patch and batched with `vmap`; patches are a flat list with
`gridcell_index`/`column_index`; forcing is gathered from gridcells, parameters from the PFT
table; gridcell output is the area-weighted sum of its patches.
Reason: gridcells with different numbers of patches need no padded grid, and the same code runs
one patch or a globe.
Source: plan §0 4; §3.2 principles 4-5.

**23. Soil under patches: option A now, option B later**
2026-10-27 · plan · maintainer · in force · Departure from Fortran: no
Each patch has its own soil column (`soil.columns: per_patch`, as standalone CLM-ml); shared
columns (option B) are a later, separately validated science change.
Reason: option A reproduces CLM-ml exactly; B needs a new rule for per-patch ground
temperatures.
Source: plan §0 5; §6.7.

**24. Fixed layer slots with masks**
2026-10-27, 2026-10-28 · plan, Phase 5 · maintainer · in force · Departure from Fortran: no
Every patch has `max_layers` slots (default 50; Fortran's limit was 100) with masks for the
layers in use and the leafy ones; padding must not change results or give NaN gradients;
masks rather than grouping patches by layer structure, so one compiled program serves every
patch.
Reason: one program for mixed patches (simpler batching, sharding and gradients); the cost is
computing on padded slots.
Source: plan §0 6; §0 19; §8 2.

**25. C3/C4 as a static choice per group of patches**
2026-10-28 · Phase 5b/6 · maintainer · in force (to revisit with the GPU work) · Departure from Fortran: no
Patches are sorted into a C3 and a C4 group inside one compiled program; each patch computes
only its own pathway.
Reason: computing both pathways for every patch (tried, measured) wastes about half the
photosynthesis work on a GPU; masks stay for layer counts, which take many values.
Source: plan §0 22; D10; plan "After Phase 8: GPU performance".

**26. Design target: global runs**
2026-10-28, revised 2026-10-29 · Phase 6, 6b · maintainer · in force · Departure from Fortran: no
Phase 6 was designed for global runs with several patches per gridcell (first ~0.5°, then
changed to 1°, CESM f09, to match the surface data and soil moisture); real global input was
brought in before the Phase 7 restructure (Phase 6b). Tower runs stay supported as a special
case.
Reason: where the two pull apart the global run decides; Phase 7 was designed against real
requirements.
Source: plan §0 23; §0 26.

**27. Canopy layers and heights in global runs**
2026-10-29 · Phase 6b · maintainer · in force · Departure from Fortran: no (CLM-ml v2 profile)
Constant layer counts for every patch (10 within, 10 above the canopy; model.yaml
`canopy_layers.n_layers_within/above`); canopy top from the surface dataset's
`MONTHLY_HEIGHT_TOP` at the first step (layers fixed at setup); `MONTHLY_HEIGHT_BOT` read but
not used (the leaf-area profile is a beta distribution from the ground, as CLM-ml v2).
Reason: 5-10 within-canopy layers reproduce the multilayer benchmark (Bonan et al. 2021); using
the canopy bottom would be a science change.
Source: plan §0 28; input decision 7; G12; G13.

**28. Non-vegetated land left out**
2026-10-29 · Phase 6b · maintainer · in force (bare ground after Phase 8, issue 13) · Departure from Fortran: yes (versus CLM's gridcell)
Bare ground and other non-vegetated land units are skipped; each gridcell's vegetated patch
weights are renormalized to sum to 1; gridcell output is the area-weighted mean over vegetated
patches; gridcells without vegetation are left out.
Reason: CLM-ml has no bare-ground canopy model yet (new code, deferred).
Source: plan §0 29; G19.

**29. Minimum patch weight**
2026-10-27, 2026-10-29 · plan, Phase 6b · maintainer · in force · Departure from Fortran: no
`domain.min_patch_weight` (default 0: keep all); applied to the fraction of the vegetated area,
then renormalized; a gridcell keeps at least its largest patch.
Reason: dropping tiny patches is a science choice for the user, not a default.
Source: plan §8 8; G18.

**30. A patch that fails a conservation check**
2026-10-28 · Phase 6 · maintainer · in force · Departure from Fortran: no (default)
Failures are recorded per patch; default: stop and name the patches (as Fortran);
`numerics.on_patch_error: exclude` drops the patch from then on (fill values, left out of
gridcell sums with weights renormalized, zero gradient contribution) and stops only above a set
fraction of failures.
Reason: one bad patch must not end a global run, and a NaN must never reach a gridcell sum or a
shared PFT gradient.
Source: plan §0 25.

**31. Several devices only for towers sharing one meteorology file**
2026-10-30 · Phase 6 review · maintainer · in force · Departure from Fortran: no
`load_run` refuses `devices > 1` with towers of different meteorology files.
Reason: not a use case; a defect (review R1) made such runs read another tower's forcing.
Source: R1 (`phase6_progress.md`, review fixes).

**32. Global input choices**
2026-10-29 · Phase 6b · maintainer · in force · Departure from Fortran: no
1° (f09) grid; domain = the CLM run's land mask (14 346 gridcells, 72 575 vegetated patches);
soil texture from the top level of the surface data; year 2000 for forcing, soil moisture,
initial soil temperature and land cover.
Reason: matches the provided CLM inputs.
Source: input decisions 4, 5, 8 (`global_input.md`).

**33. Forcing height in global runs: CLM's rule**
2026-10-29 · Phase 6b · maintainer · in force · Departure from Fortran: no versus CLM5 (ZengWang2007); yes versus CLM6's default roughness (Meier2022, open issue 1)
Per patch `zref = 30 m + (z0mr + displar)·htop` with the PFT's ratios in params.yaml.
Reason: CLM's data atmosphere sets 30 m above the displacement height plus roughness length.
Source: input decision 2; G14.

**34. Root biomass in global runs**
2026-10-29 · Phase 6b · implementation (copying CLM-ml towers) · in force · Departure from Fortran: n/a (CLM has no fine-root biomass)
500 g m⁻² for every patch (`domain.grid.root_biomass`).
Reason: the value of every CLM-ml forest tower.
Source: G17.

**35. Organic matter capped**
2026-10-29 · Phase 6b · implementation (copying CLM) · in force · Departure from Fortran: yes (CLM-ml's formula has no cap)
Soil organic matter capped at CLM-ml's `organic_max` (130 kg m⁻³).
Reason: CLM caps the organic fraction at 1; 4 021 patches' soils exceed the cap.
Source: G16.

## D. Time, calendar, forcing and land data

**36. Memory does not depend on run length**
2026-10-27 · plan v4 · maintainer · in force · Departure from Fortran: no
Forcing comes from streams read in chunks with one halo record at each end; a run is a loop
over chunks carrying the state, reading the next chunk in the background and writing output per
chunk; never the whole run's forcing or per-step output in memory.
Reason: design target is a global run with 150 years of 6-hourly forcing.
Source: plan §0 12; §6.9.

**37. Chunks of calendar months**
2026-10-28 · Phase 4 · maintainer · superseded by 38 · Departure from Fortran: no
Chunks aligned to months; each distinct length compiled once.
Source: plan §0 17.

**38. Fixed-length chunks, one program per run**
2026-10-30, revised 2026-10-31 · Phase 6b, 7.4 · maintainer (implementation for counting) · in force · Departure from Fortran: no
`execution.chunk: N days` (default 10); a short last chunk is padded with inactive steps; output
buffers sized for the most records a chunk can touch; chunks counted from step 0 (the first step
is an ordinary step).
Reason: every month length compiled anew (~1000 s each on GPU) and a month of forcing ran a GPU
out of memory; chunk length barely changes throughput beyond a few days.
Source: plan §0 33; C28.

**39. One clock for the whole run**
2026-10-28 · Phase 6 · maintainer · in force · Departure from Fortran: no
All gridcells share calendar, start, model step and chunking; latitude and longitude are
per-gridcell data; towers with different periods are separate runs.
Reason: as in a global run.
Source: plan §0 24.

**40. Configurable model step with interpolated forcing**
2026-10-27, done 2026-10-28 · Phase 4b · maintainer · in force · Departure from Fortran: no at the forcing interval (CHATS7 unchanged); the interpolation itself is new
`numerics.step_seconds` independent of the forcing interval; tower forcing records are
interpreted as interval means: linear between midpoints for temperature, humidity, wind,
pressure, longwave; shortwave scaled by `cos(zenith)` conserving each interval's mean;
precipitation held constant.
Reason: long and global runs use 3- or 6-hourly forcing.
Source: plan §0 13; Phase 4b.

**41. Prescribed land data from daily or monthly files**
2026-10-27, done 2026-10-28 · Phase 4b · maintainer · in force · Departure from Fortran: no (CLM history still supported)
LAI, SAI and soil water from user files (linear between interval midpoints); the CLM history
file at the forcing step stays supported (CHATS7).
Reason: long runs cannot use a CLM history at every forcing step.
Source: plan §0 14; Phase 4b.

**42. Exact model calendar**
2026-10-30 · Phase 6 review · maintainer · in force (grid compset noleap: 109) · Departure from Fortran: yes (fixes CLM-ml's year counting; orbit kept fixed)
`time.calendar: gregorian` (default) or `noleap` (grid runs), with exact date arithmetic as
CTSM's clock; a Gregorian 29 February reads no-leap streams' 28 February. Orbital parameters are
those of the start year (not per year: D4), a known departure from CLM's historical runs
(~0.02° of declination over 150 years).
Reason: Fortran CLM-ml counted years with the start year's length (37 days off after 150 years);
TRENDY CLM runs use no-leap.
Source: plan §0 34; D4; G3; R2; open issue 2 (resolved).

**43. Gridded forcing copies CLM's data atmosphere (CDEPS)**
2026-10-29 · Phase 6b · maintainer (copy CLM) · in force · Departure from Fortran: no versus CDEPS, except planar bilinear weights (ESMF uses the sphere)
The step `[t, t+Δt]` uses the data atmosphere at its start (G1); per-stream time interpolation:
TPQW linear, precipitation nearest, shortwave `coszen` (G2); 0.5°→1° bilinear with nearest
valid point outside the source mask (G4); datm derivations (TBOT ≥ 180 K, QBOT capped at
saturation, half visible/half NIR, 30 m) (G5); rain/snow ramp and CLM's repartition, snowfall
intercepted by the canopy (G6); wind speed only (G7); TRENDY CO₂ read as CDEPS reads it, O₂ =
0.209 × pressure (G8); CRUJRA's direct/diffuse split as given.
Reason: the global runs should see the forcing CLM sees.
Source: input decisions 1, 3; G1, G2, G4-G8.

**44. LAI and SAI in global runs**
2026-10-29, revised 2026-10-30 · Phase 6b · maintainer (copy CLM) · in force · Departure from Fortran: no versus CTSM (except 45)
Monthly LAI/SAI per PFT and gridcell with CLM's satellite-phenology weights, evaluated at the
end of the step (CTSM's ordering) and with CLM's 28-day February; `elai = tlai`, `esai = tsai`
(no snow burial: no snow model).
Reason: copies `SatellitePhenologyMod`; review R9 found the weights one step early.
Source: G9; R9.

**45. CLM's 0.05 LAI/SAI cutoff not copied**
2026-10-29 · Phase 6b · maintainer · in force · Departure from Fortran: yes (versus CTSM)
Interpolated LAI and SAI are used as given (not below 0).
Reason: the cutoff guards CLM's big-leaf canopy against tiny plant area, which entry 49 does
here, and its jump blocks gradients with respect to LAI (values below 0.05 in ~8 % of
patch-months).
Source: plan §0 31; G9; issue 17a.

**46. Soil water and initial soil state in global runs**
2026-10-29 · Phase 6b · implementation (copy CLM) · in force · Departure from Fortran: yes (CLM would start from a spun-up restart)
`H2OSOI` (20 levels) from monthly history files, linear between their stamps at the step end;
initial `TSOI`/`H2OSOI` from the same interpolation at the end of the first step, then CLM-ml's
`SoilInit` (capped at porosity).
Reason: copies CTSM's soil-moisture stream; no spun-up restart exists for CLM-ml.
Source: G10; G11.

**47. Forcing prepared on the host, a chunk ahead**
2026-10-31 · Phase 7 · maintainer · in force (supersedes "meteorology built inside the compiled chunk", plan §0 35 / D3 first form) · Departure from Fortran: no
A background thread reads and builds the next chunk's forcing while the current chunk runs;
the chunk's inputs are 0-based arrays: meteorology per gridcell (`MetInputs`), land data per
patch (`LandInputs`).
Reason: measured on the global run: GPU time is the canopy physics, host forcing work is seconds
per chunk, and forcing memory is not the limit; moving it to the GPU would add GPU work and
change round-off.
Source: plan §0 36; D3 (revised); C24.

**102. Gridded streams: CDEPS's time axis (cycle over the data years)**
2026-11-01 · Phase 8a · maintainer · in force · Departure from CTSM/CDEPS: no (copies CDEPS)
The gridded meteorology and the monthly soil history each have `years: {taxmode, year_first,
year_last, year_align}` in run.yaml (CLM: `DATM_YR_START/END/ALIGN`,
`stream_year_first/last_soilm`, `model_year_align_soilm`; `taxmode = cycle`, as CLM runs
these streams): model year Y reads data year `year_first + (Y - year_align + 2n) mod n`, its
stamps moved to year Y, so a run in the first data year interpolates its first steps from the
last record of `year_last` (CDEPS `shr_stream_findBounds`: the greatest valid date shifted
back by the loop length), and runs past `year_last` start the loop again; `limit` stops
outside the data. Fixes "a grid run cannot start in the first forcing year" (open item O30).
Unchanged for runs inside the data years with `year_align: year_first` (bit for bit).
Reason: copy CLM's rule (maintainer).
Source: open_items O30; CDEPS `dshr_stream_mod.F90`; CTSM `stream_definition_datm.xml`,
`SoilMoistureStreamMod.F90`.

**109. Grid runs use the no-leap calendar by default**
2026-11-05 · after Phase 8 · maintainer · in force (revises 42's default for grid runs) · Departure from Fortran: no
The compset `grid_1deg_2000` sets `time.calendar: noleap`, so grid experiments made from it run
the no-leap calendar; the option's default stays `gregorian`, which tower runs need (their
forcing is a real time series).
Reason: CLM's TRENDY runs, the CRUJRA forcing and the CTSM run giving the soil moisture are all
no-leap.
Source: maintainer, 2026-11-05.

## E. Science changes and departures from Fortran

**48. Acclimation temperature: 10-day running mean**
2026-10-27, done 2026-10-28 · Phase 4b · maintainer · in force · Departure from Fortran: yes
The acclimation temperature is a 10-day running mean of air temperature carried in the state
(CLM's `runmean` accumulation); `physics.acclimation_temperature: run_mean` keeps Fortran's
whole-run mean for validation only.
Reason: the whole-run mean needs all forcing before the run starts, impossible for long runs.
Source: plan §0 15; Phase 4b (May 2007: max 0.95 % from Fortran, within the 2.5 % tolerance).

**49. Minimum plant area in every leafy layer**
2026-10-29 · Phase 6b · maintainer · in force · Departure from Fortran: yes (0 reproduces CLM-ml v2)
Every leafy layer keeps at least 0.01 m² m⁻² of plant area at every step, the missing part added
as stem area (`canopy_layers.min_layer_plant_area`).
Reason: CLM-ml's diffuse transmittance exceeds 1 in layers below ~0.005 m² m⁻², making the
longwave solution unstable (runaway leaf temperatures, energy created in winter); stems add no
photosynthesis or transpiration; the value is CLM-ml v1's `dpai_min`. CHATS7 unchanged.
Source: plan §0 30; G20; issue 14.

**50. Canopy-top latent heat averaged per sub-step**
2026-10-29 · Phase 6b · maintainer · in force · Departure from Fortran: yes (a diagnostic only)
The reported canopy latent heat is the sub-step average of `LatVap(tref)·etair(ncan)`, not the
averaged vapor flux times the last sub-step's latent heat.
Reason: the two differ when the air crosses 0 °C (vaporization vs sublimation), which failed the
canopy energy check; the state and budget were consistent. CHATS7 round-off only.
Source: plan §0 32; issue 12.

**51. Runge-Kutta update covers the top canopy-air layer**
2026-10-28 · after Phase 4 · maintainer · in force · Departure from Fortran: no (restores Fortran)
The translation skipped the top layer (`ncan`) and updated an unused level 0; fixed to Fortran's
layers 1..ncan.
Reason: a translation error; no effect on CHATS7 (no process reads that layer mid-step).
Source: numerics §4.4.

**52. Secant solvers take no step when two residuals are equal**
2026-10-27/28 · Phase 3 · implementation (bug fix) · in force · Departure from Fortran: no (Fortran's hybrid solver stops before this can happen)
In the fixed-step secant iterations (Obukhov length, intercellular CO₂), no step is taken when
`f1 == f0`.
Reason: dividing by a floor of 1e-30 threw converged values ~100 m away; this was the apparent
4.5 % round-off sensitivity; afterwards the noise twins differ by ≤ 4e-11.
Source: numerics §5.

**53. Known solver and translation deviations kept (not changed)**
2026-10-27/28, 2026-10-30 · Phase 3, 7.2 · implementation, recorded · in force (fidelity step not scheduled) · Departure from Fortran: yes
Pre-existing in the validated translation and kept: Obukhov length by 25 fixed secant steps
(Fortran: hybrid secant/Brent, tol 0.1 m); WUE stomatal conductance by 20 halvings plus one
Newton step (Fortran: bisection to 0.001); intercellular CO₂ by 40 secant steps plus a Newton step
(Fortran: hybrid, tol 0.1); the CO₂ profile reset covers levels 0..ncan−1 (no effect while CO₂ is
constant); the tridiagonal solver guards pivots with 1e-10 in solar radiation and soil
temperature (not in Fortran).
Reason: a refactor must not change results; changing them is a fidelity step with its own
validation (CHATS7 May 2007 within 2.5 % of Fortran, r ≥ 0.9999).
Source: numerics §2, §4.1-4.3, §4.5; C9.

**54. Conservation checks inside the compiled step**
2026-10-28, revised 2026-10-31 · Phase 5a, 7.2 · implementation · in force (issue 20 open) · Departure from Fortran: yes (when a run stops; which checks run)
Fortran's conservation checks (leaf and soil energy, longwave, nitrogen integral, ...) record
their largest error in the step (`Diagnostics.balance_error`); the host checks after each chunk,
so a run stops at the end of the chunk, not at the failing step; each step's errors start from
zero. Fortran's 15 checks of the implicit flux-profile solution (`ErrorCheck01/02`) exist as a
function but are not run in the step.
Reason: host callbacks made compiled programs uncacheable and paused the device every step;
adding the implicit-solution checks changes the recorded state (a decision for the maintainer).
Source: plan §6.8; C17; C21; open issue 20.

**55. Legacy translation quirks kept bit for bit, pending decisions**
2026-10-31 · Phase 7.2, 7.4 · implementation, recorded · closed (21 fixed by 92, 22 by 93, 19 by 95) · Departure from Fortran: yes
Kept unchanged in refactors and listed as open issues: soil ice read one level too deep in
`SoilResistance` (19; no effect while ice is 0); soil texture classes looked up one class too far
(21); a missing `organic_matter` read as −999, giving a negative organic fraction (22).
Reason: fixing them is a science change, not part of a refactor.
Source: C14; C27; `global_input.md` issues 19, 21, 22.

**92. Soil texture classes take their own values (issue 21 fixed)**
2026-11-01 · Phase 8a · maintainer · in force · Departure from Fortran: no (restores Fortran)
A tower given a USDA texture class (no sand and clay percentages) gets that class's sand,
clay and Clapp-Hornberger values (`SoilTexMod.F90`); the legacy translation indexed the
0-based tables with Fortran's 1-based class number, so each class got the next class's values
and "clay" failed. Changes only the soils of towers given a class (CHATS7 and grid runs give
percentages: unchanged, bit for bit).
Reason: a translation error; the model must compute what CLM-ml v2 computes.
Source: `global_input.md` issue 21; open_items O1; C27.

**93. One way to say each setting: `processes`, choice keys, named fallbacks**
2026-11-01 · Phase 8a · maintainer · in force · Departure from Fortran: no
model.yaml's scheme choices are `processes`, grouped by process (photosynthesis, stomatal
conductance, radiation, turbulence, canopy air), not "physics" (many are biological); the code
package `clm_ml/physics` is `clm_ml/processes`. Alternatives are one choice key whose
settings appear only with that choice (`yaml_schema.option(when=...)`; the loader refuses the
settings of a choice not made, the files show only those in use): run length is one key
(`time.length: 1 day | 48 steps | until <date-time>`), soil texture is `given_as:
percentages | class`, land-data files and the restart file appear only with their source,
grid CO2 is `source: file | constant`, the model's layer spacing only with `layer_counts:
from_spacing`, `wue_solver` only with the WUE scheme, `acclimation_days` only with the running
mean (`temperature_acclimation: running_mean | run_mean | none` replaces two keys). Nulls that
meant "taken from elsewhere" are named words (`canopy_height: pft_default`, `area_profiles:
pft_default`, `canopy_layers: from_model_yaml`, `nitrogen_profile: computed`, `region:
whole_globe`, `restart_every: end_of_run`) or plain defaults (forcing height 30 m); root
biomass and organic matter are required. The two thin-layer settings are named by when they
act (`merge_thin_layers_below`, at setup; `minimum_leafy_layer_plant_area`, every step).
`clm-ml check` prints each value the run uses and where it comes from (e.g. layer counts);
`clm-ml upgrade` moves earlier files to these forms.
Found while doing it (documentation only, no change to results): a missing canopy height took
the PFT's default height (CLM's `htop_pft`), not the CLM history file as the docs said; a
missing root biomass was an error at setup (as Fortran's `endrun`), not a PFT default; a leaf
or stem profile given alone was not used (Fortran needs both). Requiring organic matter
closes open issue 22 for the files (a missing value can no longer reach the -999 path).
Reason: one place and one form for each setting, so a user cannot set two things that
silently override each other (maintainer).
Source: maintainer, 2026-11-01; open_items O2.

**94. Group C code tidying: `checkpoint_steps` works, PFT optics 0-based, Euler weight**
2026-11-01 · Phase 8a · implementation · in force · Departure from Fortran: no
model.yaml `numerics.checkpoint_steps` now switches `jax.checkpoint` on the step (it was
ignored: the driver used a module constant; default true as before); a model with it off
does not share its compiled program with one with it on. The PFT optics of `CanopyParams`
(leaf/stem reflectance and transmittance) are `[vis, nir]`, 0-based as the rest of the model
(were `[unused, vis, nir]`). The Euler tableau's weight is `b = [1]` (was `[0]`, a trap: an
update would have reset the state; the canopy sub-step still skips the update for Euler, as
Fortran). Bit for bit; gradient tier passes.
Reason: a setting that does nothing, a leftover 1-based layout and a wrong constant are traps
for users and future code.
Source: open_items O33, O65, O67.

**95. Soil ice of the level itself stops its root uptake (issue 19 fixed)**
2026-11-01 · Phase 8a · maintainer · in force · Departure from Fortran: no (restores Fortran)
Root uptake from a soil level stops when that level holds ice (Fortran `SoilResistance`,
`h2osoi_ice(c,j)`); the translation read the level below it (`step.soil_level_ice`). Bit for
bit today: the prescribed soil water is all liquid (open issue 3).
Reason: a translation error; matters once frozen soil is represented.
Source: `global_input.md` issue 19; open_items O3.

**56. Soil colour class 0 refused**
2026-11-01 · Phase 7.4 review · implementation · in force · Departure from Fortran: (unclear: the legacy JAX code wrapped class 0 to class 20; whether Fortran does was not checked)
A soil colour class of 0 is an error.
Reason: a class outside 1-20 has no albedo values.
Source: C31.

**57. Soil thermal properties computed once per step**
2026-10-30 · Phase 7.2 · implementation · in force · Departure from Fortran: no (same values)
`SoilThermProp` is computed once and reused by the soil temperature solution (Fortran calls it
twice with identical inputs).
Reason: half the work, bit-for-bit results.
Source: C11.

## F. Numerics, solvers and gradients

**58. One traceable implementation per process**
2026-10-27/28 · Phase 3 · implementation · in force · Departure from Fortran: no
Each process has one implementation that JAX can trace, computing what the validated model
computed; a traceable copy replaced the original only when equal on identical input to
round-off.
Reason: the gradient path and the forward run must be the same model.
Source: numerics §1.

**59. Implicit-function gradients never change forward values**
2026-10-28 · Phase 3 · implementation · in force (not yet for gs and ci: entry 53) · Departure from Fortran: no
Iterative solvers return the solver's own answer; gradients come from the implicit function
theorem applied as `x0 − (step − stop_gradient(step))` (forward exactly `x0`).
Reason: differentiating through iterations gives wrong or exploding gradients; a Newton step
would change the forward value.
Source: numerics §2; CLAUDE.md "Iterative solvers".

**60. Roughness length with Fortran's bisection stopping rule**
2026-10-28 · Phase 3 · implementation · in force · Departure from Fortran: no
The traceable bisection stops as Fortran's (bracket < 0.001 m, last midpoint), as a fixed-length
loop that stops updating.
Reason: 20 full iterations differed from CLM-ml v2 by up to 4e-4 relative.
Source: numerics §3.

**61. Gradients with respect to parameters, forcing and initial state**
2026-10-27 · plan · maintainer · in force · Departure from Fortran: no
Reason: calibration, sensitivity analysis and data assimilation.
Source: plan §0 7.

**62. psihat tables are data passed into the compiled step**
2026-10-31 · Phase 7.2 · maintainer · in force · Departure from Fortran: no
The roughness-sublayer ψ̂ tables (`psihat.nc`, package data) are read once and passed as an input
of every compiled chunk (not a module global, not a compiled-in constant).
Reason: another table file needs no recompilation, and the tables can be swapped or
differentiated.
Source: C15; C22.

**63. Patch constants are inputs of the compiled chunk**
2026-10-28, 2026-10-31 · Phase 5b, 7.4 · maintainer (C23 b) · in force · Departure from Fortran: no
The per-patch constants (layer grid, soil column, roots, location: `PatchConstants`, computed by
the setup) are an input of the compiled chunk.
Reason: other layer grids, soils or towers reuse the same program; data never trigger
compilation.
Source: C23 (b); C26; C28; plan Phase 5b.

**64. Rules for compiled code**
2026-10-28 · Phase 5a, 6 · implementation · in force · Departure from Fortran: no
No host callbacks (`jax.debug.callback`/`print`) in compiled code; every compiled program is
cached on disk; never scatter into a scan's carry with array indices (patch groups are
contiguous slices); `--xla_cpu_parallel_codegen_split_count=1` set on import (CPU compile crash);
stored values rounded as Fortran stores them where XLA would otherwise refold expressions
(`optimization_barrier`).
Reason: cacheable programs, and avoiding JAX 0.9.2 CPU defects that gave wrong states or crashes.
Source: numerics §6; plan §6.8; C15.

**65. 64-bit precision**
2026-10-27 · plan · maintainer · in force · Departure from Fortran: no
The model runs in float64 (data-center GPUs handle it; consumer GPUs are 30-60× slower).
Reason: Fortran double precision; a lower-precision option would need its own validation.
Source: plan §8 7.

**104. The coupled flux-profile solve differentiated as a linear solve (forward and reverse mode)**
2026-11-02 · after Phase 8 · maintainer (option C of three) · in force · Departure from Fortran: no
`tridiag_2eq` (the implicit air temperature and vapor solve) is written with
`jax.lax.custom_linear_solve`: the solve is Fortran's block Thomas algorithm (forward values
bit for bit), the transpose solve is the same algorithm on the transposed blocks (the former
custom-VJP backward rule), and the derivatives with respect to the coefficients come from the
operator `A x`. JAX then applies the implicit function theorem in either mode, so `jax.jvp`
and `jax.jacfwd` run through a chunk as well as `jax.grad`.
Reason: its reverse-only `custom_vjp` rule made forward-mode derivatives fail (O71); removing
the rule would differentiate the elimination again (huge coefficients, NaN gradients), and a
hand-written `custom_jvp` would change the reverse gradients' round-off and keep hand-derived
calculus. Reverse gradients are unchanged (identical on a unit system and on the CHATS7 chunk);
every other solver was already forward-mode safe (decision 59).
Source: O71; maintainer, 2026-11-02.

## G. Regression and testing policy

**66. Bit for bit for refactors, noise twin for round-off changes**
2026-10-27 · Phase 3 · maintainer · in force (supersedes the plan's first `rtol = 1e-6`) · Departure from Fortran: no
Refactors must reproduce the reference bit for bit; every change must stay within 2× the model's
own round-off sensitivity, measured by two twin runs perturbed by 1e-12 (twins differ by
≤ 4e-11). A round-off-only change larger than ~1e-10 is investigated as a defect, not met with
wider tolerances.
Reason: the model's own sensitivity sets the only meaningful tolerance.
Source: plan §8 3; numerics §6. (Conflict: CLAUDE.md "Decisions" still says "Regression
tolerance is `rtol = 1e-6`".)

**67. Round-off-only reference re-recordings are allowed**
2026-10-27, 2026-10-30 · Phase 3, 7 · maintainer · in force · Departure from Fortran: no
A change that only alters round-off (compilation, operation order, vectorization) is its own
commit: it must pass the noise twin, the reference and twins are re-recorded, and the commit says
so. Done for: Phase 6 steps 2a and 3a; Phase 7 C10, C14, C16, C17, C19, C20, C21, C28 (and C33
re-recorded the reference in the new names with identical values). Science changes also
re-record, explicitly (48, 50).
Reason: compiled code cannot reproduce the last bits after legitimate restructuring.
Source: D7; numerics §6 (workflow); the C entries listed.

**68. Bit-for-bit tests and the reference run on the CPU**
2026-10-30 · Phase 7.1 · implementation · in force · Departure from Fortran: no
Tests and reference recordings force the CPU; on the GPU two fresh compiles of the same program
can differ by ~1e-11, so GPU results are checked with the noise twin.
Reason: CPU compiles are reproducible; the bit-for-bit gate stays meaningful.
Source: C4; numerics §6.

**69. Testing tiers**
2026-10-28, revised 2026-10-30 · Phase 5a, 7 · maintainer · in force (Phase 7 minimum ended with Phase 7) · Departure from Fortran: no
After each edit: unit and fast golden tests; before each commit: bit-for-bit + noise twin and the
changed feature's end-to-end test; at the end of a phase: all regression, slow and gradient
tests and the Fortran comparison. Phase 7 used a smaller per-commit set (lint, bit for bit +
noise twin, moved routines' golden tests).
Reason: end-to-end runs cost ~40-60 s each; the bit-for-bit test checks every field at every
step.
Source: numerics §6.1; C0.

**70. Gradient testing**
2026-10-30 · Phase 7 · maintainer · in force, revised by 105 (LLVM optimization on since JAX 0.10.2) · Departure from Fortran: no
CPU gradient tests run with LLVM optimization off (`tests/gradient_tier.py`; the optimized CPU
gradient segfaults, issue 16), after each move of a hand-written gradient and at package ends;
a GPU gradient job (`jobs/grad_gpu.pbs`) at the end of a phase.
Reason: gradients can change without forward values changing only where gradients are
hand-written.
Source: D12; C0; C3; plan §0 35.

**71. Fortran reference data are fixed**
2026-10-27 · Phase 0 · maintainer · in force · Departure from Fortran: no
The Fortran golden JSON files (rtol 1e-9) are never edited to make a test pass; the end-to-end
Fortran comparison (May 2007) must stay within 2.5 % per column with r ≥ 0.9999 (run with
`acclimation_temperature: run_mean`).
Reason: they are the link to CLM-ml v2.
Source: CLAUDE.md "Rules for changes"; Phase 4b.

**105. JAX 0.10.2**
2026-11-02 · after Phase 8 · maintainer · in force · Departure from Fortran: no
The model is developed and tested with JAX 0.10.2 (the environment rebuilt on 2026-11-01).
The regression reference and its noise twins are re-recorded with it (CPU, Derecho). Rechecked
under 0.10.2 with cold compilation caches: CPU gradients run with LLVM optimization on (9 of 9
gradient tests; O25 closed), so the gradient tier no longer turns it off; the mixed C3/C4
grid-box chunk compiles without `--xla_cpu_parallel_codegen_split_count=1` (5 of 5; O26 closed),
but the flag stays on import as a cheap precaution. The scatter rule of decision 64 is kept
(not rechecked).
Reason: the maintainer's choice (2026-11-02) over pinning 0.9.2; the bit-for-bit test needs a
reference from the JAX version in use.
Source: O72; maintainer, 2026-11-02.

## H. Outputs, restarts and runs

**72. Configurable output streams, accumulated inside the compiled chunk**
2026-10-27 · plan v4, Phase 4 · maintainer · in force · Departure from Fortran: no
Output streams with their own variables, frequency, statistic (mean, instantaneous, min, max),
level (patch, gridcell) and file split, accumulated inside the compiled scan (nested scan over
records and steps, `jax.checkpoint` on the step), so memory scales with output records, not
steps.
Reason: memory independent of run length (36); output independent of forcing and model step.
Source: plan §0 16; §6.9.

**73. Restarts as netCDF**
2026-10-30, done 2026-11-01 · Phase 6 review, 7.5 · maintainer · in force · Departure from Fortran: no
One netCDF file per run (format 6): the state with units, `alive` per patch, the output records
in progress, layout and calendar; written beside the old one and renamed, the previous kept as
`.prev`; resume checks time, layout, calendar and stream identity; restarts hold only the state,
the record in progress and bookkeeping.
Reason: readable with standard tools, no pickled Python objects; restart size must not grow with
run length (review R6).
Source: D2; C32; R6; R3 (its `chunk_anchor` became unnecessary with chunks from step 0, C28).

**74. Legacy `.out` files only for the Fortran comparison**
2026-10-27, revised 2026-11-01 · plan, 7.4f · maintainer · in force · Departure from Fortran: no
The Fortran-format `.out` files are kept only while the Fortran comparison needs them; the run
writes them itself (`output.legacy_ascii`).
Reason: netCDF is the output format; the comparison test still reads `.out`.
Source: plan §8 6; C30.

**75. Output variables computed from the step's fields; internal names stay CLM-ml's**
2026-11-01 · Phase 7.6 · implementation · in force (supersedes the 7.1 plan to rename diagnostics, C6; the legacy view C2/C7) · Departure from Fortran: no
Output variables have readable names and are computed from a step's fields
(`diagnostics.<CLM-ml name>`, `state.<name>`, `layers.<name>`; 0-based), which validation
streams can also name; diagnostics keep their CLM-ml names.
Reason: internal names stay traceable to the Fortran; users see readable output names.
Source: C33.

**76. Diagnostics only as outputs and checks need**
2026-10-30 · Phase 7 · maintainer · in force · Departure from Fortran: no (values unchanged; fewer computed)
The step returns only diagnostics that outputs and checks read; every removed field is listed
(name, meaning, routine) in `phase7_removed_diagnostics.md` so it can be added back.
Reason: smaller compiled programs and memory.
Source: D6; C6; C8 (diffuse transmittance kept for the issue-14 test); plan §0 35.

**77. The first step is an ordinary step**
2026-10-31 · Phase 7.4 · implementation · in force · Departure from Fortran: no (round-off only)
The setup gives the state before the first step (acclimation count 0), and the first step runs
as step 0 of the first chunk: no separate first-step program. `Model` runs the first step at
setup, so its chunks start at step 1.
Reason: one compiled program per run, no special case.
Source: C25; C27; C28; C31.

**78. Health check by layer masks**
2026-10-31 · Phase 7.4 · implementation · in force · Departure from Fortran: no
The state health check (after each chunk) covers the layers in use, every soil level and scalar,
skipping temperatures of exactly 0 in leafless layers.
Reason: replaces a comparison with a template state that no longer exists.
Source: C28.

**108. A grid run's gridcell output on the latitude-longitude grid**
2026-11-05 · after Phase 8 · maintainer · in force · Departure from Fortran: no
For a grid run, the gridcell level (the `*_gridcell.nc` files) is written on the surface
data's latitude-longitude grid, variables `(time, ..., lat, lon)`, instead of a flat
`gridcell` dimension: the whole grid for `region: global`, the grid's rows and columns inside
the box for a region (longitudes in the box's convention, increasing eastward), NaN where no
gridcell was run, and `run_mask` (1 where one was). Patch files stay per patch, and tower runs
keep the `gridcell` dimension. No separate level and no remapping command (runs are repeated).
The file also holds `area` (km2) and `landfrac`, from the CLM history file of `grid.land_mask`
(as in CLM's own history files), and `vegetated_fraction`: the fraction of the land covered by
PFTs 1-16 in the surface data, which the patches represent. Gridcell values are per m2 of
vegetated land (decision 28), not per m2 of land as in CLM: a total is
`value * area * landfrac * vegetated_fraction`, and `value * vegetated_fraction` is CLM's
per-m2-of-land basis for fluxes that bare ground does not have (e.g. GPP, transpiration).
Reason: maps are how gridded results are read and compared with CLM history files; totals and
the comparison with CLM need the vegetated fraction, which differs from CLM's land basis.
Source: maintainer, 2026-11-05.

**112. Derecho batch job with a new case; the global example only sets the case up**
2026-11-05 · after Phase 8 · maintainer · in force (revises 110) · Departure from Fortran: no
`new_experiment` (`clm-ml new`) takes the machine, the device (gpu or cpu), the number of
devices (written to `run.execution.devices`), the account and the walltime. On machine
`derecho` the account is required and the folder also gets `submit.pbs` (`clm_ml.batch`):
a PBS job on the `main` queue (1-4 GPUs of one node, or a CPU node) running `clm-ml run` on
the folder with the Python environment that created it, log in `run.log`. Other machines get
no job script. `examples/run_global.py` only sets the case up (no `RUN_NOW`): the user submits
it from the command line (`qsub CASE_DIR/submit.pbs`, or `clm-ml run CASE_DIR`).
Reason: a global run needs a GPU batch job; writing it with the case, as CESM's case setup
does, saves writing one by hand and keeps setup and running separate.
Source: maintainer, 2026-11-05.

## I. Code architecture

**79. 0-based indexing everywhere**
2026-10-30 · Phase 7 · maintainer · in force · Departure from Fortran: no (layout only)
Layers from the ground, `level` = ground + layer tops, leaf 0 sunlit / 1 shaded, band 0 visible /
1 near-infrared, soil levels from the top, Runge-Kutta stages from 0; unused slots hold `spval`
(outputs show NaN). Only the `.out` writer builds the legacy 1-based layout.
Reason: Python convention; removes a mixed 0/1-based code base.
Source: D1; plan §0 35.

**80. Physics organized by process, variants by name**
2026-10-28 (decided), 2026-10-30/31 (done) · Phase 5 plan, 7.2 · maintainer · completed · Departure from Fortran: no
Phase 5 made the existing modules mask-aware in place; Phase 7 moved the physics to
`clm_ml/processes/` (one module per process, pure functions on 0-based arrays, `Fortran
reference:` in each docstring), with variants in name registries chosen by model.yaml and
declared reads/writes (a lighter form of jax-gcm's composable terms); the soil routines moved
too (§8 5).
Reason: files by physical process, not by the Fortran module they came from.
Source: plan §0 18; §0 21; §6.6; §8 5; C9-C20.

**81. One pure step function**
2026-10-31 · Phase 7.2 · implementation · in force · Departure from Fortran: no
`clm_ml.step.step(state, inputs, constants, params, tables; options, c3) -> StepState` computes
one patch's step in the order of `clm_drv`.
Reason: everything the step depends on is an argument; `vmap`, `grad` and sharding apply
directly.
Source: C19; C21.

**82. The prognostic state**
2026-10-30 · Phase 7.1 · implementation · in force · Departure from Fortran: no
`State` = canopy (air temperature, vapor pressure, leaf temperature, leaf water potential,
intercepted water, ground temperature, acclimation mean and count, ...) and soil (temperature)
per patch, 0-based, no patch axis; batched runs add one leading patch axis.
Reason: the plan's layout (§6.7 keeps soil separate for option B).
Source: C5.

**83. Setup as pure functions**
2026-10-31 · Phase 7.4 · implementation · in force · Departure from Fortran: no (bit for bit with the legacy setup)
`clm_ml/setup/` computes every patch's constants and initial state from the `Domain`, `Params`
and `ModelConfig` objects (tower and grid runs alike).
Reason: no module globals; one setup path for towers and grids.
Source: C27; C29.

**84. Chunk inputs split by owner**
2026-10-31 · Phase 7.3 · implementation · in force · Departure from Fortran: no
`MetInputs` per gridcell (calendar days, meteorology, previous/next records), `LandInputs` per
patch (LAI, SAI, soil water, adjustment factor = 1 when absent), joined per patch inside the
chunk.
Reason: meteorology is sent once per gridcell, not once per patch.
Source: C24.

**85. Only `clm_ml` in `src/`**
2026-10-30 to 11-01 · Phase 7 · implementation (plan) · completed · Departure from Fortran: no
The Fortran-mirroring packages, the legacy driver and dead code were deleted; package data in
`clm_ml/data` (towers, params, psihat); CHATS7 forcing and namelists in `data/` (not
`examples/data/` as the plan said).
Reason: one package organized by process and role.
Source: C1; C30; C33; plan Phase 7b.

**86. No new unrolled Python loops in compiled code**
2026-10-30 · Phase 7 · maintainer · in force · Departure from Fortran: no
Loops over layers or levels are array operations or `lax.scan`, not Python loops unrolled into
the program.
Reason: program size and compile time (the GPU program went from 2.32 GB to 4 MB).
Source: D5; plan §0 35.

**100. One lock for netCDF reads and writes**
2026-11-01 · Phase 8a · implementation (maintainer approved) · in force · Departure from Fortran: no
Every netCDF file is opened through `clm_ml.io.netcdf.dataset` (and the output writers write
under its lock), so the forcing reader thread and the output writers never use HDF5 at the
same time (the HDF5 under netCDF4 here is built without thread safety). The device still runs
the chunk while the next chunk is read. Results unchanged.
Reason: a possible crash when output records complete while the next chunk is read.
Source: open_items O31.

**101. Padding slots on their copied patch's gridcell, and not checked**
2026-11-01 · Phase 8a · implementation (maintainer approved) · in force · Departure from Fortran: no
In multi-device layouts an empty slot copies a patch of the same pathway; it is now placed on
that patch's own gridcell when the gridcell is in the block (it sees its own forcing), and
padding slots are left out of the conservation and health checks (they have weight 0 and are
outside every output and gradient). A padding slot can no longer stop a run. Results
unchanged.
Reason: padding combined one gridcell's weather with another patch's data and was checked as
if real.
Source: open_items O29; Phase 6 review R5.

**106. The compilation cache on scratch by default**
2026-11-02 · after Phase 8 · maintainer · in force · Departure from Fortran: no
Without `JAX_COMPILATION_CACHE_DIR`, compiled programs are cached in `$SCRATCH/jax_compile_cache`
when the system defines `$SCRATCH` (NCAR's machines), and in `~/.cache/jax_compile_cache`
otherwise (`clm_ml.driver._default_cache_dir`).
Reason: the cache grows to gigabytes (2.1 GB in the home directory by 2026-11-02) and filled
the home quota; caches belong on scratch.
Source: maintainer, 2026-11-02.

**111. "Compset", not "preset"**
2026-11-05 · after Phase 8 · maintainer · in force · Departure from Fortran: no
What `clm-ml new DIR --from NAME` starts a folder from is called a compset (component set,
as in CLM/CESM): the package folder `clm_ml/compsets/` (experiments, params, towers),
`compset_names()`, `clm-ml new --from COMPSET`, and every message and document. The old word
is not accepted. Unlike a CESM compset, ours also fixes the grid (or tower) and input files.
Reason: CLM users' terminology.
Source: maintainer, 2026-11-05.

## J. Performance

**87. GPU optimization after the modernization; no science-adjacent speed-ups**
2026-10-28/30 · Phase 6, 6b · maintainer · in force · Departure from Fortran: no
"Modernize first, optimize the GPU later": the GPU work (kernels per step, scaling over GPUs,
the C3/C4 groups) is a separate effort after Phase 8; only arithmetic-preserving changes are
allowed for speed (no fewer solver iterations, fewer sub-steps or another integrator).
Reason: the science must not change for performance; structure first makes optimization easier.
Source: plan Phase 6 step 1c; Phase 6b step 7; "After Phase 8: GPU performance".

## K. Scope deferred

**88. Work deferred to after Phase 8**
2026-10-30/31 · Phase 6b, 7 · maintainer · open · Departure from Fortran: n/a
After Phase 8: issue 10 (round-off amplification in real 1° patches), issue 13 (bare-ground
patches, new code), GPU performance (87), the unsupported variants (17), option B shared soil
columns (23).
Reason: the modernization first; each is its own science or performance effort.
Source: plan "After Phase 8" sections.
