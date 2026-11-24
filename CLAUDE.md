# CLAUDE.md

Guidance for Claude (and other contributors) working in this repository.

## Working with the maintainer

- The maintainer is a **land-surface scientist, not a software engineer**. Explain software
  concepts (pytrees, static vs traced, adapters, etc.) in plain language, with the scientific
  reason they matter. Don't assume familiarity with software-engineering jargon.
- **Think before coding.** State assumptions. If there are several reasonable interpretations,
  present them instead of picking one silently. Push back when something conflicts with the
  science or with an earlier decision (for example, a setting that would reject a supported tower).
- **Every meaningful decision** (maintainer's or implementation's) is recorded in
  `docs/design/decisions.md` (one register, stable numbers; a new one gets the next free
  number, in the same commit). Don't re-open them without a new reason. Every open task, issue
  and pending decision is in `docs/design/open_items.md` (O1, O2, ...). The development logs
  were deleted in Phase 9.0 and are in git history (`git show 4f33cd7:docs/design/<file>`).
  Decision numbers 126-127 are used on branch `chats_demos` (not merged): the next is 129.
- **Standing preferences:**
  - Commit and push when the tests required by the testing tiers pass (authorized); never
    open PRs; one branch per phase. Commit footer:
    `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (or the model in use).
  - Don't modify the maintainer's Python environments (no `pip install` without asking).
  - Stop before starting a new phase to discuss it; during a phase, a measured slowdown of
    more than ~1.5× (timing test or gradient) is discussed before going on.
  - Explain plainly, with the scientific reason; keep reports short.
- **Bugs and science:** diagnose first with probes in scratch, *no code changes*, and bring
  the diagnosis to the maintainer before changing code; no shortcuts, hacks or dropping
  patches/things that fail; the model is a scientific tool (accurate and precise). Where a
  modelling choice comes up, copy CLM (CTSM 5.4 source in `/glade/work/linnia/ctsm5.4.004`;
  CLM-ml v2 Fortran: [gbonan/CLM-ml_v2.CHATS](https://github.com/gbonan/CLM-ml_v2.CHATS)).
  Science changes are the maintainer's decision; record each in `decisions.md` and every open
  issue in `open_items.md`.
- **Comments and docstrings** (decision 128) describe what the code computes and why: no
  phase, decision, issue or refactor references, no pointers to `docs/design/`. Cite Fortran as
  `Fortran reference: Module.F90::Subroutine` (no line numbers). A departure from CLM-ml v2 gets
  one plain line, "Not in CLM-ml v2: ..." or "Differs from CLM-ml v2: ...", with the reason.
- Terminology: say **"timing test"** for wall-clock performance measurements, not "benchmark".
  (In the papers, "benchmark" means the reference multilayer simulation; keep that meaning.)

## Status and where to resume

- **Phase 9** (documentation and release; branch `phase9/docs`; `docs/design/phase9_scope.md`).
  9.0 is a clean-up first (decision 128): 9.0a code comments (done), 9.0b documents and other
  files, 9.0c a discussion of what documentation to write, before any is written.
- **Done:** Phases 0-8 (the modernization), GPU optimization (the 1° globe 2.85× faster on one
  A100), gradient accuracy work (decisions 118-125), all on `main`.
- **Branch `chats_demos`** (not merged, kept apart by the maintainer): gradient experiments at
  CHATS (`experiments/chats_gradients/`), open items O82-O85 and decisions 126-127 recorded
  there only.
- **Gradients, what to know:** a solver's gradient needs an exact slope and fallback values
  with their own gradient; a power with a parameter as exponent needs a base that cannot turn
  negative; check new derivatives against finite differences. Two lagged feedbacks (water
  stress, Obukhov length) have derivative-only rules (`numerics.water_stress_derivative`,
  `numerics.obukhov_derivative`): with them a gradient is not the exact derivative of the
  forward values. Time gradients from the third call. A tower is 4-5× faster on the CPU than on
  a GPU.
- **Device-dependent hidden settings:** `numerics.table_search`, `numerics.solver_loops` and
  `numerics.checkpoint_substeps` follow the device (GPU: written out / checkpointed; CPU: loops,
  the regression reference).

## Where the work is

- **NCAR Derecho** (CPU login node; GPUs through PBS, account UCLB0069, develop queue
  `gpudev`). The regression reference is recorded there (x86).
- **The maintainer's Mac** (arm64): conda environment `clm-ml-jax` (JAX 0.10.2, CPU, dev extras;
  created for Phase 9). Results differ from Derecho in the last bits, so the committed
  bit-for-bit test fails there; the noise twin passes. For bit for bit on the Mac, record a
  reference from the unchanged code in a scratch worktree and compare against that.

## What this model is

**CLM-ML-JAX** is a JAX translation of **CLM-ml v2**, the multilayer canopy model developed for
the Community Land Model (CLM) by Gordon Bonan (NSF NCAR). The Fortran reference is
[gbonan/CLM-ml_v2.CHATS](https://github.com/gbonan/CLM-ml_v2.CHATS). The JAX version adds GPU
execution, automatic differentiation (for calibration and sensitivity analysis), and `vmap`
ensembles, while reproducing the Fortran results. It runs at flux towers (prescribed 30-min
meteorology; leaf and stem area, soil moisture and initial soil temperature from a CLM history
file) and on grids up to global (many gridcells, no lateral exchange).

### Scientific background (key papers)

1. **Bonan, Patton, Finnigan, Baldocchi & Harman (2021)**, *Moving beyond the incorrect but
   useful paradigm: reevaluating big-leaf and multilayer plant canopies to model
   biosphere-atmosphere fluxes – a review*. Agric. For. Meteorol. 306, 108435.
   doi:10.1016/j.agrformet.2021.108435. Introduces **CLM-ml v1** (1 to N layers) and compares
   one-layer with multilayer canopies at seven AmeriFlux forest sites. Multilayer reduces
   biases (friction velocity, sensible heat, nighttime fluxes), mainly through the vertical
   profiles of air temperature, humidity, wind and leaf water potential. **5–10 within-canopy
   layers reproduce the benchmark** (50 cm resolution) for the dense forests tested.
2. **Bonan, Burns & Patton (2026)**, *Beyond surface fluxes: Observational and computational
   needs of multilayer canopy models – A walnut orchard test case*. Agric. For. Meteorol. 378,
   110960. doi:10.1016/j.agrformet.2025.110960. Data: doi:10.5281/zenodo.17426258. Introduces
   **CLM-ml v2** and the **CHATS** test case. **This is the configuration this repo reproduces.**

Earlier model papers: Bonan et al. (2014) for leaf gas exchange, stomatal optimization and plant
hydraulics; Bonan et al. (2018) for roughness-sublayer turbulence and the flux–profile solution,
using Harman & Finnigan (2007, 2008).

### Model physics (CLM-ml v2)

- **Vertical structure.** Canopy air space from the ground to the forcing (reference) height,
  in layers of `dz_tall = 0.5 m` (canopies over 2 m) or `dz_short = 0.1 m`, or explicit counts
  per patch (`canopy_layers: {within: N, above: M}`). Leaf and stem area profiles follow beta
  distributions. Sunlit and shaded leaves in each layer.
- **Flux–profile solution.** Implicit coupled tridiagonal equations for air temperature, water
  vapor and leaf temperature at every layer, plus ground temperature.
- **Turbulence.** First-order closure with the **Harman & Finnigan roughness sublayer (RSL)**:
  exponential wind profile within the canopy, RSL-modified similarity functions above it
  (ψ̂ from the lookup table `src/clm_ml/data/psihat.nc`, passed to the physics as an argument).
  β = u*/U(h) uses the Harman (2012) form with CHATS fit a₁=0.89, a₂=−0.07, a₃=2.19, β_N=0.35.
  Bounds −2 ≤ L_c/L ≤ 1. Not in CLM-ml v2: in stable conditions also (z_ref − d)/L ≤
  `zeta_max_stable` (PFT parameter; 0.5 for shrubs, grasses and crops, no limit for trees).
- **Leaf processes.** Leaf energy balance with heat storage; Farquhar photosynthesis with an
  exponential Vcmax profile; **water-use-efficiency stomatal optimization** (ι) constrained by
  **plant hydraulics**, f(ψ_l) = [1 + (ψ_l/ψ₅₀)^α]⁻¹, solved by bisection; Medlyn and
  Ball–Berry as alternatives. Leaf boundary-layer conductance minimum 0.2 mol m⁻² s⁻¹.
- **Radiation.** Two-stream shortwave and Norman longwave.
- **Time stepping.** The 30-min step is split into six **5-min sub-steps**; forcing uses
  three-point time interpolation; **4th-order Runge–Kutta** for air temperature, vapor
  pressure, leaf temperature, leaf water potential, intercepted water and ground temperature;
  output averaged to the step. Wind minimum 0.5 m s⁻¹; within-layer aerodynamic resistance
  capped at 500 s m⁻¹. The model step is configurable (`numerics.step_seconds`).
- **Soil.** Soil moisture is a **prescribed boundary condition**; soil temperature from a
  heat-diffusion model specific to the multilayer canopy (CLM5 layer structure only).
- **Acclimation temperature:** a 10-day running mean (differs from CLM-ml v2's whole-run
  mean, kept as an option for validation).

### The CHATS test case (CHATS7, May 2007)

- Walnut orchard (*Juglans regia* 'Chandler'), Dixon, California (38.49°N, 121.84°W).
  Canopy height 10 m; forcing at 23 m (radiation at 16 m). LAI 2.04, SAI 0.74. Beta parameters
  (2.6, 1.3) for leaves and (1.8, 1.3) for stems. c_d = 0.25, L_c ≈ 14.8 m.
- Soil: silty clay loam (10% sand, 35% clay), soil color 15, bedrock at 2 m. CO₂ 383 ppm.
  PFT 7 (`broadleaf_deciduous_temperate_tree`). **46 canopy-air layers** with the default
  spacing (20 within the canopy, 26 above).
- **CHATS parameters** (Table 2 of the 2026 paper; compset `chats`):

  | Parameter | Default | CHATS |
  |---|---|---|
  | Vcmax25 | 57.7 | 125 |
  | ι | 750 | 375 |
  | ψ₅₀ | −2.3 | −1.6 |
  | α | 40 | 40 |
  | k_p | 4 | 7 |
  | R_r* | 25 | 14 |
  | Leaf reflectance (vis, nir) | 0.10, 0.45 | 0.06, 0.42 |
  | Leaf transmittance (vis, nir) | 0.05, 0.25 | 0.04, 0.43 |
  | Stem reflectance (vis, nir) | 0.16, 0.39 | 0.21, 0.49 |
  | χ_l (leaf angle) | 0.25 | 0.53 |

- **Known model limitations** (science questions; don't "fix" without discussion): near-ground
  air too warm in strongly stable conditions (first-order closure); daytime latent heat biased
  high (~70 W m⁻²); soil heat flux biased high; soil moisture prescribed.

## Repository layout

```
src/clm_ml/
  processes/      the physics by process, pure functions on 0-based arrays (soil, atm_forcing,
                  solar_radiation, longwave, turbulence, leaf_boundary_layer, photosynthesis,
                  flux_profile, canopy_water, plant_hydraulics, nitrogen_profile, integrators,
                  canopy (the sub-step), constants, orbit, math, layers)
  step.py         one time step of one patch: State + StepInputs + PatchConstants -> StepState
  state.py        State (prognostic), Diagnostics (CLM-ml names, 0-based), PatchLayers
  setup/          every patch's constants and initial state (pure functions, no globals)
  driver.py       the compiled chunk (vmap over patches, C3/C4 groups, shard_map), run_streams
  chunk_inputs.py, tower_forcing.py, grid_forcing.py, forcing.py, forcing_interpolation.py
  io/             tower and CLM history readers, restarts (netCDF), `.out` writer (legacy.py),
                  namelist converter, psihat tables, YAML schema
  outputs.py, output_streams.py, checks.py, calendar.py, options.py
  config.py (model.yaml), run.py (run.yaml), domain.py (domain.yaml), params.py (params.yaml)
  experiment.py   experiment folders: load, new, upgrade; Experiment.execute
  model.py        Model: chunks and gradients
  cli.py          clm-ml new | check | run | upgrade | convert | params
  compsets/       towers/, params/, experiments/: what `clm-ml new` copies
  data/           psihat.nc
configs/          the compsets written out as experiment folders
data/             CHATS7 forcing (input_files/) and the CLM-ml namelists
tests/
  clm_ml/              unit tests (fast)
  fortran_validation/  160+ routine-level tests against Fortran golden JSON (rtol 1e-9)
  regression/          end-to-end runs against committed references
  test_fortran_python_comparison.py  end-to-end vs Fortran .out files (slow; 2.5% / r >= 0.9999)
  test_differentiability.py, test_gradient_accuracy*.py   gradient tier
timing/                timing tests and their results (*.jsonl)
examples/              run_tower.py, run_global.py, plot_global_diag.py
jobs/                  PBS scripts for Derecho GPU jobs
docs/how_the_model_works.md   for users: organization, a run's skeleton, differences from Fortran
docs/design/           decisions.md, open_items.md, model_overview.md,
                       numerics_and_regression.md, phase9_scope.md
```

**Experiment folders:** an experiment is a folder with four complete files: `run.yaml` (when
and with what), `domain.yaml` (where: towers or a grid), `model.yaml` (processes, numerics,
canopy layers, soil; static: a change recompiles), `params.yaml` (every parameter of every PFT
1-16). Strict: missing, unknown or misplaced keys are errors; `clm-ml upgrade` adds new keys.
Every option is declared once in the code with `yaml_schema.option`; the files are generated
from it. Public API: `load_experiment`, `new_experiment`, `upgrade_experiment`, `Experiment`,
`Model`.

**Vocabulary:** **gridcell** (a location with its own atmospheric forcing); **patch** (one PFT's
canopy on a gridcell, with an area weight; the unit of computation); **soil column** (the soil
under a patch; one per patch); **tower** (a named flux-tower location in `domain.yaml`). Avoid
"site" except for observation locations.

### Gotchas in the current code

- **Login nodes have GPUs** (derecho3, derecho7) and JAX uses them by default: tests force the
  CPU (`tests/conftest.py`); probes must set `JAX_PLATFORMS=cpu`. Two fresh GPU compiles can
  differ by ~1e-11, so bit for bit is a CPU test. On CPU-only nodes set `JAX_PLATFORMS=cpu`
  (the CUDA plugin otherwise prints a harmless traceback).
- **Runs are in-process** with no module globals; a process can hold several models. netCDF is
  not thread-safe: never run two runs in parallel threads of one process. The device count is
  fixed when JAX starts (`test_devices` uses subprocesses).
- **The first step is an ordinary step** of the first chunk. Fixed-length chunks count from
  step 0 (a short last chunk padded with inactive steps). `Model` runs the first step at
  setup; its `chunks()` start at step 1.
- **Constants and psihat tables are inputs of the compiled chunk**: other layer grids, soils or
  tables reuse the program. Data never trigger compilation.
- **Indexing is 0-based everywhere** (layers from the ground, `level` = ground + layer tops,
  leaf 0 sunlit / 1 shaded, band 0 visible / 1 near-infrared, soil levels from the top). Slots
  above the top layer hold `spval` (1e36); outputs mask them with NaN. Only `io/legacy.py`
  builds the 1-based `.out` layout.
- **JAX 0.10.2**: the environment and the regression reference. Another version changes the
  last bits: bit for bit fails, the noise twin must pass. `clm_ml/__init__.py` sets
  `XLA_FLAGS=--xla_cpu_parallel_codegen_split_count=1` (a precaution against an old LLVM
  compile crash).
- **Never scatter into a scan's carry with array indices** (`.at[idx].set` inside the chunk's
  `lax.scan`): it gave wrong states and segfaults on CPU. Patch groups are contiguous slices.
- **Never add `jax.debug.callback`/`print` to compiled code**: it makes the program
  uncacheable. Conservation checks are recorded (`Diagnostics.balance_error`) and checked on
  the host after each chunk.
- **Compilation cache** on disk (`JAX_COMPILATION_CACHE_DIR`, default
  `$SCRATCH/jax_compile_cache`, else `~/.cache/jax_compile_cache`). Never delete anything in the
  maintainer's home directory.
- **C3/C4 is a static choice per group of patches**, not per-patch data.
- **Iterative solvers and padding:** compute on every layer slot, give padding the top layer's
  inputs (`processes.layers.pad_with_top`), keep results with masks; no Python loops over layers
  in the time step. Padded layers must not change results or give NaN gradients (safe `where`).
- **Real 1° patches amplify round-off** (1e-15 in LAI -> ~8 W m-2 after sunrise): grid
  regression tests compare inputs exactly and outputs only through the night.
- **Memory must not depend on run length**: forcing streams read in chunks, output accumulated
  inside the compiled scan. Never design something that needs the whole run's forcing or
  per-step output in memory.

## Commands

```bash
pip install -e ".[dev]"

clm-ml new my_site --from chats7_1day      # an experiment folder from a compset
clm-ml check my_site                        # validate + summary, no run
clm-ml run my_site [--set 'run.time.length=3 days']
clm-ml upgrade my_site                      # rewrite with every current setting
clm-ml convert data/namelists/nl.CHATS7.1day -o chats7/
clm-ml params ctsm60_params.nc -o p.yaml   # CLM parameter file -> complete params.yaml
clm-ml new configs/chats7_1day --from chats7_1day --force   # regenerate configs/ after changing options

python -m pytest tests/clm_ml tests/fortran_validation -m "not slow" -q   # fast
python -m pytest tests/regression/test_end_to_end_regression.py tests/regression/test_clm_ml_equivalence.py -m regression -q   # bit for bit + noise twin
python -m pytest tests/regression -m regression -q                         # all end to end (~20 min CPU)
CLM_ML_COMPARE_DIR=<dir> python -m pytest tests/test_fortran_python_comparison.py -m slow -q   # vs Fortran output
python tests/gradient_tier.py [--focused]                  # CPU gradient tier (~7 min)
qsub -v REV=<commit>,TAG=<name> jobs/grad_gpu.pbs          # GPU gradient tier (Derecho)

python tests/regression/record_run.py --update-reference   # only after intended or round-off-only changes; say so
python timing/time_offline.py                              # wall-clock time of a run
python timing/time_model.py [--steps 8] [--no-grad]        # chunk + gradient timing/memory

black --check --line-length 100 src tests timing examples && ruff check src tests timing examples   # CI lint
sphinx-build -W --keep-going -b html docs docs/_build/html   # documentation (warnings are errors; CI publishes main on GitHub Pages)
```

If `pytest-cov` is missing, run pytest with `-p no:cacheprovider -o addopts=""`.
Python >= 3.10; CI tests 3.10 and 3.12.

## Rules for changes

- **Regression policy** (`docs/design/numerics_and_regression.md`): changes that should not
  change the arithmetic pass **bit for bit**; every change passes the **noise twin** (within
  the model's round-off sensitivity, twins perturbed by 1e-12, differing by ≤ 4e-11). A change
  that legitimately alters round-off regenerates the reference with `--update-reference`, said
  so in the commit. Science changes also regenerate it, explicitly.
- **Test in tiers** (end-to-end tests cost ~40–60 s per model run):
  - after each edit: `tests/clm_ml` + `tests/fortran_validation -m "not slow"`, plus the test
    being written;
  - before each commit: the bit-for-bit and noise-twin tests, plus the changed feature's own
    end-to-end test;
  - at the end of a phase: all of `tests/regression` and the slow tests (Phase 9 adds the docs
    build).
  New end-to-end tests simulate as little as needed (usually one day) in as few runs as possible.
- If a round-off-only change moves results by more than ~1e-10, look for a discrete defect
  (compare routines on identical input) instead of widening tolerances. Fixed-step secants take
  no step when f1 == f0.
- **Iterative solvers:** the forward value is the solver's own answer (Fortran algorithm and
  stopping rule). Implicit-function gradients are applied as `x0 - (step - stop_gradient(step))`
  so they never change forward values. Known deviation: the Obukhov solver is 25 fixed secant
  steps, not Fortran's hybrid secant/Brent with tol = 0.1 m.
- Keep the Fortran golden tests passing; never edit their JSON files to make a test pass.
- **Don't change the science while restructuring code.** Keep code-only and science changes in
  separate commits.
- Every function translated from Fortran keeps a `Fortran reference:` line in its docstring.
- Record design decisions in `docs/design/decisions.md` in the same change.

## Design guidance: jax-gcm

Use **[jax-gcm](https://github.com/climate-analytics-lab/jax-gcm)** (docs at
[jax-gcm.readthedocs.io](https://jax-gcm.readthedocs.io/en/latest/)) as the reference for
software design: a differentiable JAX Earth-system model that grew from a Fortran translation.

- **Pure functions and immutable pytree state.** No hidden globals; no Python `if` on traced
  values (`jnp.where`, `lax.cond`); static shapes.
- **Static configuration vs differentiable parameters.** Flags that choose code paths are
  static; **every numeric tunable is differentiable**.
- **Column physics written once**, run over patches with `vmap`.
- **Files organized by physical process and scheme name.**
- **Configuration through files, one run command**; labelled output (xarray with coordinates
  and units; select by coordinate, never by an assumed index).
- **Gradient hygiene.** Test gradients against finite differences. Watch for NaN gradients
  from `jnp.where`/`jnp.clip` guards (both branches are differentiated), `sqrt` at 0, and
  overflow in derivatives.
- **Finish the job.** Complete, faithful fixes that match the reference formulation.
