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
  `docs/design/decisions.md` (one register, stable numbers; new ones get the next number, in
  the same commit). Don't re-open them without a new reason. Every open task, issue and
  pending decision is in `docs/design/open_items.md` (O1, O2, ...). The older documents
  (plan §0/§8, Phase 7 D/C numbers, `global_input.md`) are the detailed history.
- **Standing preferences** (from the working sessions):
  - Commit and push when the tests required by the testing tiers pass (authorized); never
    open PRs; one branch per phase (`phase9/docs` next). Commit footer:
    `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (or the model in use).
  - Don't modify the maintainer's Python environment (no `pip install` without asking).
  - Stop before starting a new phase to discuss it; during a phase, a measured slowdown of
    more than ~1.5× (timing test or gradient) is discussed before going on.
  - Explain plainly, with the scientific reason; keep reports short.
  - Lint with `black --check --line-length 100 src tests timing examples && ruff check src tests timing examples`
    (dev extras). If `pytest-cov` is missing, run pytest with `-p no:cacheprovider -o addopts=""`.
- **Bugs and science (maintainer, Phase 6b):** diagnose first with probes in scratch, *no code
  changes*, and bring the diagnosis to the maintainer before changing code; no shortcuts,
  hacks or dropping patches/things that fail; the model is a scientific tool (accurate and
  precise). Where a modelling choice comes up, copy CLM (CTSM 5.4 source in
  `/glade/work/linnia/ctsm5.4.004`; CLM-ml v1/v2 Fortran sources: see the progress log).
  Science changes are the maintainer's decision; record each in `docs/design/decisions.md`
  and every open issue in `docs/design/open_items.md`.
- **Resume here:** `docs/design/phase9_scope.md` (Phase 8 closed 2026-11-01 with a short 8b:
  README, `docs/how_the_model_works.md`, `examples/run_tower.py` and `examples/run_global.py`,
  decision 103). Phase 9, the formal documentation and release, starts by discussing its §2
  decisions with the maintainer, on a new branch `phase9/docs` from `main`. Open items:
  `docs/design/open_items.md`; how the model works now: `docs/how_the_model_works.md` (users)
  and `docs/design/model_overview.md` (detail).
- **Testing:** the tiers under "Rules for changes" (Phase 9 adds the docs build; phase9_scope §5).
- **Where the work is:** NCAR Derecho (CPU login node; GPUs through PBS, account UCLB0069,
  develop queue `gpudev`). Phase 6 history: `docs/design/phase6_progress.md`.
- Terminology: say **"timing test"** for wall-clock performance measurements, not "benchmark".
  (In the papers, "benchmark" means the reference multilayer simulation; keep that meaning.)

## What this model is

**CLM-ML-JAX** is a JAX translation of **CLM-ml v2**, the multilayer canopy model developed for
the Community Land Model (CLM) by Gordon Bonan (NSF NCAR). The Fortran reference is
[gbonan/CLM-ml_v2.CHATS](https://github.com/gbonan/CLM-ml_v2.CHATS). The JAX version adds GPU
execution, automatic differentiation (for calibration and sensitivity analysis), and `vmap`
ensembles, while reproducing the Fortran results.

It runs **offline at flux towers**: prescribed 30-min meteorology drives the canopy, with leaf
and stem area, soil moisture and initial soil temperature read from a CLM history file.

### Scientific background (key papers)

1. **Bonan, Patton, Finnigan, Baldocchi & Harman (2021)**, *Moving beyond the incorrect but
   useful paradigm: reevaluating big-leaf and multilayer plant canopies to model
   biosphere-atmosphere fluxes – a review*. Agric. For. Meteorol. 306, 108435.
   doi:10.1016/j.agrformet.2021.108435. Introduces **CLM-ml v1**, generalized to any number of
   layers (1 to N), and compares one-layer (1L) with multilayer (ML) canopies at seven AmeriFlux
   forest sites (56 site-years, e.g. US-UMB with 42 layers, US-Me2 with 28).
   - ML reduces biases relative to 1L, especially in friction velocity, sensible heat and
     nighttime fluxes. The main reasons are the vertical profiles of air temperature, humidity
     and wind (the "well-mixed assumption" tests), and of leaf water potential (upper-canopy
     water stress).
   - **5–10 within-canopy layers reproduce the benchmark ML canopy**, with little improvement
     beyond 10 layers, for the dense forests tested. The benchmark uses 50 cm resolution
     (28–54 within-canopy layers).
2. **Bonan, Burns & Patton (2026)**, *Beyond surface fluxes: Observational and computational
   needs of multilayer canopy models – A walnut orchard test case*. Agric. For. Meteorol. 378,
   110960. doi:10.1016/j.agrformet.2025.110960. Data: github.com/gbonan,
   doi:10.5281/zenodo.17426258. Introduces **CLM-ml v2** and the **CHATS** test case. **This is
   the configuration this repo reproduces.**

Earlier model papers: Bonan et al. (2014) for leaf gas exchange, stomatal optimization and plant
hydraulics; Bonan et al. (2018) for roughness-sublayer turbulence and the flux–profile solution,
using Harman & Finnigan (2007, 2008).

### Model physics (CLM-ml v2)

- **Vertical structure.** Canopy air space from the ground to the forcing (reference) height,
  divided into layers of `dz_tall = 0.5 m` (canopies over 2 m) or `dz_short = 0.1 m`. Leaf and
  stem area profiles follow beta distributions. There are separate sunlit and shaded leaves in
  each layer.
- **Flux–profile solution.** Implicit coupled tridiagonal equations for air temperature, water
  vapor and leaf temperature at every layer, plus ground temperature. This avoids CLM5's
  iterative solver and its ad-hoc "fixers".
- **Turbulence.** First-order closure with the **Harman & Finnigan roughness sublayer (RSL)**:
  an exponential wind profile within the canopy, and RSL-modified similarity functions above it
  (ψ̂ from the lookup table `src/clm_ml/data/psihat.nc`, passed to the physics as an argument).
  - β = u*/U(h) uses the Harman (2012) form, Eq. 2 of the 2026 paper, with CHATS fit a₁=0.89,
    a₂=−0.07, a₃=2.19, β_N=0.35.
  - Applicability bounds −2 ≤ L_c/L ≤ 1 (L_c = (c_d a)⁻¹, canopy length scale).
  - Log-profile aerodynamic conductance in the lowest layer.
- **Leaf processes.**
  - Leaf energy balance, including a heat-storage term.
  - Farquhar photosynthesis with an exponential leaf nitrogen (Vcmax) profile.
  - **Water-use-efficiency stomatal optimization** (parameter ι), constrained by **plant
    hydraulics** (SPA-style). Stomata close as leaf water potential ψ_l falls, through
    f(ψ_l) = [1 + (ψ_l/ψ₅₀)^α]⁻¹. v2 solves for stomatal conductance by bisection.
  - Medlyn and Ball–Berry stomatal conductance are alternatives.
  - Leaf boundary-layer conductance has a minimum of 0.2 mol m⁻² s⁻¹; there is no minimum wind
    speed.
- **Radiation.** Two-stream shortwave (Norman as an alternative), integrated over each layer,
  and Norman longwave.
- **Time stepping.**
  - The 30-min forcing step is split into six **5-min sub-steps** (`dtime_ml = 300 s`).
  - Forcing is **interpolated with three-point time interpolation** (previous, current and next
    records; `met_type = 3`), centered in each interval.
  - State advances with **4th-order Runge–Kutta** (`runge_kutta_type = 41`); v1 used Euler.
  - The variables advanced are air temperature, vapor pressure, leaf temperature (sunlit and
    shaded), leaf water potential, intercepted water and ground temperature.
  - Output is averaged over the sub-steps to 30 min.
  - Forcing wind speed has a minimum of 0.5 m s⁻¹, and aerodynamic resistance within a layer is
    capped at 500 s m⁻¹.
- **Soil.** Soil moisture is a **prescribed boundary condition** (from CLM, adjusted to
  observations). Soil temperature is computed with a heat-diffusion model specific to the
  multilayer canopy, which differs from standard CLM.

### The CHATS test case (CHATS7, May 2007)

- Walnut orchard (*Juglans regia* 'Chandler'), Dixon, California (38.49°N, 121.84°W), part of
  the Canopy Horizontal Array Turbulence Study (CHATS).
- Canopy height 10 m. Forcing at 23 m (radiation at 16 m). LAI 2.04, SAI 0.74. Beta parameters
  are (2.6, 1.3) for leaves and (1.8, 1.3) for stems. Drag coefficient c_d = 0.25, giving
  L_c ≈ 14.8 m.
- Soil: silty clay loam (10% sand, 35% clay), soil color class 15, depth to bedrock 2 m.
  CO₂ is 383 ppm. PFT 7 (`broadleaf_deciduous_temperate_tree`).
- **With the default spacing, CHATS7 has 46 canopy-air layers**: 20 within the canopy and 26
  above it.
- **CHATS parameter overrides** (`pftcon_val = 1`; Table 2 of the 2026 paper; in code in
  `MLpftconMod.py` and `pftconMod.py`):

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

- **Known model limitations** (don't "fix" these without discussion; they are science
  questions):
  - Biases in strongly stable conditions: near-ground air several degrees too warm. First-order
    closure cannot represent non-local within-canopy mixing.
  - Daytime latent heat flux biased high (by about 70 W m⁻²).
  - Soil heat flux biased high (but the point measurement is likely unrepresentative).
  - Soil moisture is prescribed rather than predicted.

## Repository layout (current, after Phase 7)

Organized by process and role (Phase 7: the Fortran-mirroring packages were deleted;
`src/` holds only `clm_ml`).

```
src/clm_ml/
  physics/        the processes, pure functions on 0-based arrays (soil, atm_forcing,
                  solar_radiation, longwave, turbulence, leaf_boundary_layer, photosynthesis,
                  flux_profile, canopy_water, plant_hydraulics, nitrogen_profile, integrators,
                  canopy (the sub-step), constants (Python floats), orbit, math, layers)
  step.py         one time step of one patch: State + StepInputs + PatchConstants -> StepState
  state.py        State (prognostic), Diagnostics (CLM-ml names, 0-based), PatchLayers
  setup/          every patch's constants and initial state (soil, canopy, patches, towers,
                  grid, tables): pure functions, no globals
  driver.py       Simulation, the compiled chunk (vmap over patches, C3/C4 groups,
                  shard_map), run_streams (chunks, checks, restarts)
  chunk_inputs.py MetInputs (per gridcell) / LandInputs (per patch) from the forcing
  tower_forcing.py, grid_forcing.py, forcing.py, forcing_interpolation.py   forcing streams
  io/             tower and CLM history readers, restart (netCDF), legacy .out writer,
                  namelist converter (clm-ml convert), psihat tables, YAML schema
  outputs.py      named output variables from a step's fields; netCDF writers
  output_streams.py, checks.py, calendar.py, options.py, config.py, params.py, domain.py,
  run.py (run.yaml schema), domain.py (domain.yaml), config.py (model.yaml), params.py,
  experiment.py (experiment folders: load, new, upgrade; Experiment.execute),
  model.py (Model: chunks and gradients), cli.py
  compsets/       towers/, params/, experiments/: what `clm-ml new` copies (package data)
  data/           psihat.nc (package data)
configs/          the compsets written out as experiment folders (run, domain, model, params)
data/             CHATS7 forcing (input_files/) and the CLM-ml namelists (namelists/nl.*)
tests/
  clm_ml/              unit tests (fast)
  fortran_validation/  160+ routine-level tests against Fortran golden JSON (rtol 1e-9)
  regression/          end-to-end runs against committed references
  test_fortran_python_comparison.py  end-to-end vs Fortran .out files (slow; 2.5% / r >= 0.9999)
  test_differentiability.py          jax.grad through one step (gradient tier)
timing/                time_offline.py, time_model.py, time_vmap.py, time_global.py, time_program.py
examples/              run_tower.py (any tower, CHATS7 default), run_global.py (1° grid)
docs/how_the_model_works.md   organization, skeleton of a run, differences from Fortran
docs/design/           plan, decisions.md, open_items.md, model_overview.md, phase logs
```

### Gotchas in the current code

- **Login nodes have GPUs** (derecho3, derecho7: two A100s each) and JAX uses them by
  default: tests and the reference recording force the CPU (`tests/conftest.py`); probes
  must set `JAX_PLATFORMS=cpu` too. On the GPU, two fresh compiles of the same program can
  differ by ~1e-11 (decision C4 in `phase7_decisions.md`), so bit for bit is a CPU test.
- **Runs are in-process** (Phase 7.4): `Run.execute` and `Model` set the run up with
  `clm_ml.setup` and run `clm_ml.driver`. No module globals hold model state; a process can
  hold several models (any towers, soil structures). netCDF is not thread-safe: do not run
  two runs in parallel threads of one process (tests run them one after the other or in
  subprocesses); the device count is fixed when JAX starts (`test_devices` uses
  subprocesses).
- **The first step is an ordinary step** of the first chunk: the setup gives the state
  before it (acclimation count 0). Fixed-length chunks count from step 0 (decision 33; a
  short last chunk padded with inactive steps, one program per run). `Model` runs the first
  step at setup; its `chunks()` start at step 1.
- **Constants are an input of the compiled chunk** (`PatchConstants`, per patch), as are the
  psihat tables (`tables=`, C22): other layer grids, soils or tables reuse the program.
- **Indexing is 0-based everywhere** (layers from the ground, `level` = ground + layer tops,
  leaf 0 sunlit / 1 shaded, band 0 visible / 1 near-infrared, soil levels from the top).
  Slots above the top layer in use hold `spval` (1e36); outputs mask them with NaN. Only
  `clm_ml/io/legacy.py` (the `.out` writer) builds the legacy 1-based layout.
- **Output variables and the reference** are computed from a step's fields
  (`outputs.step_fields`: `diagnostics.<CLM-ml name>`, `state.<name>`, `layers.<name>`);
  validation streams can name these raw fields.
- **JAX 0.10.2** (decision 105): the environment, the regression reference and the
  rechecks below. Another JAX version changes the last bits: bit for bit fails, the noise twin
  must pass. The environment has the CUDA 12 plugin (`environment.yml`): on CPU-only nodes
  (derecho1) set `JAX_PLATFORMS=cpu`, or JAX prints a harmless CUDA_ERROR_NO_DEVICE traceback.
- **CPU gradients** (issue 16, O25): with JAX 0.9.2 they segfaulted in LLVM-optimized code
  and needed `--xla_backend_optimization_level=0`; with 0.10.2 they run with the default
  optimization (`tests/gradient_tier.py` only forces the CPU; the regression tests still
  refuse that flag, which changes round-off).
- **CPU compile crash (JAX 0.9.2, O26):** large batched programs segfaulted intermittently while
  compiling (LLVM JIT linker recursion); `clm_ml/__init__.py` sets
  `XLA_FLAGS=--xla_cpu_parallel_codegen_split_count=1` on import unless already chosen (kept as a
  precaution: 5 of 5 cold compiles clean without it under 0.10.2).
- **Never scatter into a scan's carry with array indices** (`.at[idx].set` inside the chunk's
  `lax.scan`): with JAX 0.9.2 on CPU it gave wrong states and intermittent segfaults. Patch
  groups are contiguous slices joined by concatenation.
- **Never add `jax.debug.callback`/`print` to compiled code:** it makes the program
  uncacheable. Conservation checks are recorded in the step (`Diagnostics.balance_error`,
  `clm_ml.checks.BALANCE_CHECKS`) and checked on the host after each chunk.
- **Compilation cache:** every compiled program is cached on disk
  (`JAX_COMPILATION_CACHE_DIR`, default `$SCRATCH/jax_compile_cache`, or
  `~/.cache/jax_compile_cache` without `$SCRATCH`; set up by `clm_ml.driver`; decision 106).
  Never delete anything in the maintainer's home directory (the old cache there included).
- **C3/C4 is a static choice per group of patches** (decision 22), not per-patch data.
- **Iterative solvers and padding:** compute on every layer slot, give padding the top
  layer's inputs (`processes.layers.pad_with_top`), keep results with masks; no Python loops
  over layers in the time step.
- **Real 1° patches amplify round-off** (1e-15 in LAI -> ~8 W m-2 after sunrise): grid
  regression tests compare inputs exactly and outputs only through the night.
- **CLM soil only** (decision 37, C35): one soil layer structure, CLM5's (`nlevsoi` 20,
  `nlevgrnd` 25; `clm_ml.setup.soil.NLEVSOI`/`NLEVGRND`), called "CLM"; CLM4.5 was dropped.
  CLM histories must have 20 `H2OSOI` levels (checked). The CHATS7 data directory
  `data/input_files/clm5_0/` keeps its Fortran name (the namelists point there).

## Commands

```bash
pip install -e ".[dev]"                     # dev extras include pytest-cov, which pytest.ini needs (--no-cov)

clm-ml new my_site --from chats7_1day      # an experiment folder from a compset (configs/README.md)
clm-ml check my_site                        # validate + summary, no run
clm-ml run my_site [--set 'run.time.length=3 days']  # run (netCDF output, config copy in OUTPUT/config)
clm-ml upgrade my_site                      # rewrite with every current setting (also old layout)
clm-ml convert data/namelists/nl.CHATS7.1day -o chats7/   # namelist -> experiment folder
clm-ml params ctsm60_params.nc -o p.yaml   # CLM parameter file -> complete params.yaml (compset clm6 = CTSM 6.0)
clm-ml new configs/chats7_1day --from chats7_1day --force   # regenerate configs/ after changing options

python -m pytest tests/clm_ml tests/fortran_validation -m "not slow" -q   # unit + golden tests (fast)
python -m pytest tests/regression -m regression -q                    # all end-to-end regression tests (~20 min CPU)
python -m pytest tests/regression/test_end_to_end_regression.py tests/regression/test_clm_ml_equivalence.py -m regression -q   # bit-for-bit + noise twin (~40 s)
CLM_ML_COMPARE_DIR=<dir> python -m pytest tests/test_fortran_python_comparison.py -m slow -q   # vs Fortran output of configs/chats7_may2007_fortran
python tests/gradient_tier.py [--focused]                  # CPU gradient tier (tests marked `gradient`, ~7 min)
qsub -v REV=<commit>,TAG=<name> jobs/grad_gpu.pbs          # GPU gradient tier (develop queue, warm GPU compile cache)

python tests/regression/record_run.py --update-reference   # reference + 2 noise twins; only after intended or round-off-only changes; say so in the PR
python timing/time_offline.py                              # wall-clock time of a whole run; timing/results.jsonl
python timing/time_model.py [--steps 8] [--no-grad]        # compiled chunk + gradient timing/memory; results_model.jsonl

black --check --line-length 100 src tests timing examples && ruff check src tests timing examples   # what CI runs
```

Python >= 3.10. CI tests 3.10 and 3.12.

## Rules for changes

- **Regression policy** (`docs/design/numerics_and_regression.md` §6):
  - refactors that should not change the arithmetic must pass the **bit-for-bit** test;
  - every change must pass the **noise-twin** test, i.e. stay within the model's own round-off
    sensitivity, measured by two twin runs perturbed by 1e-12.
  A change that legitimately alters round-off (traceable code, compilation, GPU) is expected
  to fail bit-for-bit. Regenerate the reference with `--update-reference` and say so.
  Science changes also regenerate it, explicitly.
- **Test in tiers** (numerics doc §6.1); end-to-end tests cost ~40–60 s per model run, so don't
  run the whole suite after every change:
  - after each edit: `tests/clm_ml` + `tests/fortran_validation -m "not slow"` (~10 s), plus the
    test being written;
  - before each commit: the bit-for-bit and noise-twin tests (one shared run, ~1 min), plus the
    changed feature's own end-to-end test;
  - at the end of a phase: all of `tests/regression` (~8 min) and the slow tests (~20 min).
  Every compiled program goes into JAX's on-disk cache (`$SCRATCH/jax_compile_cache`), so a
  repeated run in a new process takes ~20 s; never add host callbacks (`jax.debug.callback`)
  to compiled code: they make the program uncacheable (record errors instead, §6.8 of the plan).
  New end-to-end tests simulate as little as needed (usually one day) in as few runs as possible.
- **Noise twins differ by ≤ 4e-11**, so the noise-twin test is tight. An apparent
  evening-transition sensitivity (up to 4.5%) was a solver bug: the fixed-step secant took a
  ~100 m step when two converged residuals were exactly equal (fixed: no step when f1 == f0;
  numerics doc §5). If a round-off-only change moves results by more than ~1e-10, look for a
  similar discrete defect (compare routines on identical input, §1) instead of widening tolerances.
- **Iterative solvers:** the forward value must be the solver's own answer (Fortran algorithm
  and stopping rule). Implicit-function gradients are applied with
  `x0 - (step - stop_gradient(step))` so they never change forward values. Known deviation:
  the Obukhov solver is 25 fixed secant steps, not Fortran's hybrid secant/Brent with
  tol = 0.1 m (recorded, not changed). Fixed-step secants must take no step when f1 == f0.
- Keep the Fortran golden tests passing. Their JSON files are fixed reference data; never edit
  them to make a test pass.
- **Don't change the science while refactoring.** Keep refactors and science changes in
  separate commits and PRs.
- Every refactored function keeps a `Fortran reference:` line in its docstring (module and
  subroutine) so it can be traced back to CLM-ml v2.
- Record design decisions in `docs/design/` in the same change. Don't leave documentation for
  later.
- Commit and push when the required tests pass (see standing preferences). Work on a branch per
  phase (`phase0/safety-net`, ...).

## Modernization plan

The full plan is in **`docs/design/modernization_plan.md`**. The proposed API is in
`docs/design/api_sketch.py` (historical), and runnable config files are in `configs/`.

**Decisions:**
- **Tower compsets: CHATS7 only** (`src/clm_ml/compsets/towers/chats7.yaml`). Other towers
  go in an experiment's `domain.yaml`, with explicit layer counts per patch
  (`canopy_layers: {within: N, above: M}`) where default spacing gives too many layers.
- **Names.** Import as `import clm_ml`. The install name is `clm-ml-jax`. The command-line tool
  is `clm-ml new | check | run | upgrade | convert | params`.
- **Experiment folders** (decision 91, `docs/design/experiment_folders.md`): an experiment is
  a folder with four complete files (fixed names), and a run reads nothing else but data:
  - `run.yaml`: when and with what (dates, calendar, forcing and land-data files and the
    tower forcing interval, initial conditions, output, chunks, restarts). Replaces the namelist.
  - `domain.yaml`: where (towers or a grid).
  - `model.yaml`: `processes` (each process's parameterization: photosynthesis, stomatal
    conductance, radiation, turbulence, canopy air; not "physics"), numerics (including the
    model step), canopy layers, soil. Static: changing them recompiles.
  - One way to say each setting (decision 93): a choice key, with its settings shown and
    allowed only for that choice (`option(when=...)`); named words instead of nulls that
    meant "taken from elsewhere"; `clm-ml check` prints where each value comes from.
  - `params.yaml`: parameters **per PFT**, every parameter of every PFT 1-16 (no `extends`).
  - Strict: missing, unknown or misplaced keys are errors; `clm-ml upgrade` adds new keys at
    their defaults. Every option is declared once in the code with `yaml_schema.option`
    (explanation, Fortran name, `applies`, `hidden`, `path`); the files are generated from
    it (`clm-ml new --from COMPSET`, compsets in `src/clm_ml/compsets/`). `configs/` holds the
    compsets written out; regenerate after changing an option
    (`clm-ml new configs/NAME --from NAME --force`; `tests/clm_ml/test_configs.py`).
- YAML loads into typed dataclass pytrees. Validation is plain Python, with errors that name the
  file and key. No Hydra, no Flax.
- **Readable option names** (`stomatal_conductance: wue`). Fortran integer codes are accepted
  only by `clm-ml convert`.
- **Vocabulary** (use these terms precisely; avoid "site" except for observation locations
  in the scientific sense):
  - **gridcell**: a location with its own atmospheric forcing.
  - **patch**: one PFT's canopy on a gridcell, with an area weight. **The unit of computation.**
  - **soil column**: the soil under a patch.
  - **tower**: a named flux-tower location in `domain.yaml`, which defines a gridcell and
    its patches.
- **Spatial scope:** from one tower up to global (many gridcells, no lateral exchange).
  - Patches are a flat list with `gridcell_index`/`column_index`.
  - The physics is written for one patch, and `vmap` runs it over the patch axis.
  - Forcing is gathered from gridcells and parameters from the PFT table. Gridcell output is the
    area-weighted `segment_sum` of patches.
- **Soil under patches.** Option A (default, `soil.columns: per_patch`): each patch has its own
  soil column, identical to today's p == c. Option B (`shared`) is a later, separately validated
  science change.
- Every patch gets a **fixed `max_layers` (default 50) with masks**. Padded layers must not
  change results, and must not give NaN gradients (safe-`where`).
- **Memory must not depend on run length** (target: global, 150 years of 6-hourly forcing).
  Forcing comes from **streams** (meteorology, vegetation, soil water, each with its own time
  axis) read in **chunks** with one halo record at each end. A run is a loop over chunks that
  carries the state, reads the next chunk in the background, and writes output per chunk. Never
  design something that needs the whole run's forcing or per-step output in memory (plan §6.9).
- **Chunks vs output intervals** (as jax-gcm): output is accumulated *inside* the compiled scan
  (one scan over steps, each step adding into its output record's slot, `jax.checkpoint` on
  the step), so memory scales with output records, not steps. The chunk length is host
  bookkeeping (forcing in memory, restarts with `.prev`, health checks, file writes). A
  compiled chunk is reused for all chunks of the same shape; data never trigger compilation.
- **Model time step is configurable** (`numerics.step_seconds`); forcing streams are
  interpolated to it (linear, `coszen` shortwave, constant precipitation, midpoint
  interpolation of daily/monthly LAI, SAI and soil water).
- **Acclimation temperature: 10-day running mean** carried in the state (science change,
  Phase 4b). The Fortran whole-run mean stays as an option for validation only.
- **LAI, SAI and soil water** come from user-supplied daily or monthly files (CLM history at the
  forcing step still supported for CHATS7).
- **Output is configurable and independent** of forcing and model step: output streams with
  their own variables, frequency, statistic, level and file split, accumulated during the run.
- **Gradients** with respect to parameters, meteorological forcing and initial state.
- Regression: bit for bit for refactors, the noise twin for round-off changes (see "Rules for
  changes"). The legacy `.out` output is only kept until validation
  moves to netCDF.

**Phases:**

| Phase | Work |
|---|---|
| 0 | Safety net: regression test, packaging, timing test (done) |
| 1 | YAML config and `clm_ml` API as a wrapper around the existing code (done) |
| 2 | Forcing and time become data (done) |
| 3 | Explicit state and a pure `step` (done) |
| 4 | Chunked `jit(scan)` runs, forcing streams, output streams, restarts (done, except GPU timing) |
| 4b | Science changes: 10-day acclimation, configurable model step with time interpolation, daily/monthly land data (done; plus forcing split in time and the §4.1 forcing layout) |
| 5a | Explicit configuration: `MLOptions`, cacheable compiled programs, several same-tower models per process (done) |
| 5b | Mask-aware layers **in place** (same files, 1-based): one compiled step for any layer count (done; gradient padding test and GPU timing after Phase 6) |
| 6 | Many patches and gridcells, **on Derecho (GPU)**: steps 0-6 (per-site data, batched patches, C3/C4 groups, devices, vectorized setup) done 2026-10-29; GPU optimization (step 1c) deferred; progress log `docs/design/phase6_progress.md` |
| 6b | Real global input before Phase 7 (decisions 26-32): steps 1-6 done (grid loader, gridded forcing, setup, regional test), done 2026-10-30 (step 7 closed; review fixes, calendar decision 34); fixes: per-layer minimum plant area (30), no 0.05 LAI cutoff (31), canopy-top latent heat per sub-step (32); open issues in `global_input.md` |
| 7 | Restructure (decision 35; `phase7_scope.md`): 7.0 preparation, 7.1 step state (0-based), 7.2 physics by process (`clm_ml/processes/`), 7.3 chunk forcing as 0-based arrays (forcing prepared on the host: decision 36), 7.4 setup without globals (in-process runs), 7.5 netCDF restarts, 7.6 cleanup. **Done 2026-11-01** (`phase7_progress.md`; decisions C0-C34) |
| 8 | 8a housekeeping before the docs; 8b README, one docs page, two examples (decision 103). **Done 2026-11-01** |
| 9 | Documentation and release: docs site, settings reference, guides, examples, Fortran-users page, API reference, formatting clean-up, 1.0.0 (`phase9_scope.md`) |

Decisions 18–21 (after Phase 4b): Phase 5 converts to masks in place; masks rather than grouping
patches by layer structure; configuration passed explicitly (5a); process interface in Phase 7.
Decision 22 (Phase 5b, measured): C3/C4 stays a static choice per group of patches; Phase 6
groups patches by pathway inside one compiled program (computing both pathways per patch
doubles photosynthesis work on GPU). Masks for layer counts (many values), groups for
few-valued code switches.

Every phase must give the same answer as the previous one. Use adapters first and rewrite
internals later.

**Layer counts.** With the default spacing, short canopies under tall towers get very many
layers: above-canopy spacing equals within-canopy spacing, so a 0.5 m grass under a 30 m tower
needs 300 layers. Bonan et al. (2021) found 5–10 within-canopy layers sufficient for fluxes;
use explicit layer counts (`canopy_layers: {within: N, above: M}`) for such towers.

## Design guidance: jax-gcm

Use **[jax-gcm](https://github.com/climate-analytics-lab/jax-gcm)** (JCM; docs at
[jax-gcm.readthedocs.io](https://jax-gcm.readthedocs.io/en/latest/)) as the reference for software
design. Like this repo, it is a differentiable JAX Earth-system model that grew from a Fortran
translation (SPEEDY). Its design philosophy:

- **Pure functions and immutable pytree state.** No hidden globals. Everything a function
  depends on is an argument, and everything it changes is returned. No Python `if` on traced
  values: use `jnp.where` or `lax.cond`. Array shapes are known statically.
- **Separate static configuration from differentiable parameters.** Flags and enums that choose
  code paths are static. **Every numeric tunable must be differentiable**; never hide a
  parameter from `jax.grad` by making it static.
- **Composable physics.** Each parameterization is a `PhysicsTerm` with one uniform signature
  `(state, diagnostics, forcing, terrain) -> (tendency, diagnostics)`. Each term declares what it
  `requires` and `provides`, and the container validates the ordering (single writer, upstream
  requirements met) when the model is built. Schemes are swapped by name (`replace`, `remove`,
  `+`). Our plan uses a lighter registry with the same declare-reads/writes idea (plan §6.6).
- **Column physics is written once and runs anywhere.** jax-gcm writes columns
  "broadcasting-native" (vertical on axis 0, horizontal axes broadcast) or wraps them with
  `vmap` (`vectorize_columns`). The same code runs on one column or a whole grid.
- **Wrap translated Fortran, don't rewrite it.** The SPEEDY numerics were left untouched behind
  thin term wrappers that own unit conversions at the boundary, then modernized gradually.
- **Files are organized by physical process and scheme name**, not by the model they were ported
  from.
- **Configuration through files, one run command.** Validated configurations are named config
  files with comments explaining why each setting is what it is. There are no ad-hoc driver
  scripts. (jax-gcm uses Hydra; we use plain YAML.)
- **Labelled output.** Results go to xarray with coordinates and units. Select by coordinate
  value, never by an assumed index (for example, which end of the vertical axis is the ground).
- **Documentation lives with the change.** Design decisions go in `docs/.../design/*.md`, and a
  science choice is stated with its reference provenance (Fortran `file::routine` or paper).
- **Gradient hygiene.** Test gradients against finite differences. Watch for NaN gradients from
  `jnp.where`/`jnp.clip` guards (both branches are differentiated), `sqrt` at 0, and float32
  overflow in derivatives. See jax-gcm's `JAX_gotchas.md`.
- **Finish the job.** Deliver complete, faithful fixes that match the reference formulation, not
  partial workarounds presented as done.
