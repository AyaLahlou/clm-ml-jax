# Iterative solvers, differentiability, and regression testing

Decisions made during Phase 3 (making the physics traceable for `jax.grad`, and merging the
"normal" and "differentiable" code paths). Each states the decision, why, and the evidence.
Fortran references are to CLM-ml v2 (`gbonan/CLM-ml_v2.CHATS`).

---

## 1. One implementation per process, matching the running model

**Context.** Before Phase 3, several physics routines existed twice: a "normal" version
(Python floats and loops, used for runs) and a traceable `_jax` / `grid`-branch version (used
only by `make_clm_ml_forward` for gradients). The normal versions were validated against
Fortran; the traceable ones were not, and some had drifted.

**Decision.** Each process gets **one implementation**, written so JAX can trace it. That
implementation computes **what the running (validated) model computes today**. A traceable copy
is accepted only when it reproduces the running model's routine **on identical input** to
round-off (about 1e-15 relative). Then the duplicate is deleted.

Where the running model itself differs from Fortran, the merge keeps the running model's
behavior and the difference is recorded in §4. Changing it is a fidelity change: separate,
validated, with a regenerated reference. Using Fortran directly in a refactor would change
results in a step that is meant not to.

**How it is checked.** `tests/regression/process_io.py --record` captures the real inputs and
outputs of each such routine at dusk, night and midday of the CHATS7 reference day, from the
code before the merge (`reference/process_io.pkl.gz`, 2.4 MB). `test_process_io.py` replays
the inputs through the current routines and requires every output field to match within 1e-12
relative. (The recording must be loaded *after* the model is initialized: unpickling imports
model modules, which copy run options such as `met_type` at import.) During the merge, a
scratch harness also called both versions on the same captured inputs. As of Phase 3c, every routine with a `grid` branch
(`GetAtmForcing`, `SolarRadiation`, `LongwaveRadiation`, `CanopyTurbulence`,
`LeafPhotosynthesis`, `FluxProfileSolution`, `RungeKuttaUpdate`) agrees to within 7e-16 on every
captured call of a 2-step run, after the two fixes in §2 and §3.

**Result (Phase 3c).** The duplicates are deleted. Each routine takes a `GridInfo` (patch index
and canopy layer indices as Python ints), which `MLCanopyFluxes` builds once per CLM step (or
receives from `make_clm_ml_forward` when the state is traced). Checks that need concrete numbers
(energy conservation, the PAI check, the HF2008 `eta` warning) run only when the values are
concrete (`MLMathToolsMod.is_concrete`), so they still run in a normal run but do not break
tracing. Flux averaging, canopy diagnostics and the sun/shade merge now also run when
differentiating; before, the traceable path skipped them.

**Sub-step driver.** One function, `_substep` (physics, then flux accumulation), is the
sub-step. A normal run drives it with a Python loop. When the state is traced (`jax.grad`), it is
driven by `lax.scan` with `jax.checkpoint`, so that backward memory does not grow with the number
of sub-steps. A `lax.scan` in a normal run would recompile at every CLM step (the forcing it
closes over changes); compiling the whole run once is plan Phase 4. The two drivers run the same
code; they can differ only in round-off (XLA may fuse operations differently).

**Speed: compile the turbulence routine.** The deleted "normal" turbulence code used Python
`math` on floats. Run operation by operation, the traceable version dispatches thousands of tiny
JAX operations per call (`_GetPsiRSL` about 90 times per call, in per-layer loops), and one CLM
step went from 4.5 s to 25 s. Two fixes restore the speed (5.7 s per step under the profiler):
- `_HF2008` is compiled with `jax.jit`, the `GridInfo` layer indices being static. Its warning
  (`eta ≥ eta_max`) is checked from the outputs in `CanopyTurbulence`, because compiled code
  cannot print.
- `bisection_traceable` is compiled once per residual function. The residual is a module-level
  function with explicit parameters (a closure would be a new function at every call and
  recompile every time).

Compiling changes round-off only: the per-routine replay agrees to 3e-14.

**Gradients through a full step: program size.** `jax.grad` of one CLM step (6 sub-steps × 5
Runge–Kutta stages of the full physics) could not compile on a 16 GB machine: XLA needed
~87 GB (before 3c the traced forward pass alone peaked at 87 GB). Compile memory grows much
faster than the size of the program, so the fix was to make the program small:
- **Per-layer Python loops vectorized** (array slices or `vmap` over layers, same formula per
  layer): turbulence (`_WindProfile`, `_AerodynamicConductance`, momentum flux; ψ functions
  evaluated once per height), the flux–profile solution (leaf coefficients, tridiagonal
  coefficients, leaf and source fluxes), solar radiation (optical properties; the two-stream
  sweeps are `lax.scan` over layers with both wavebands at once). Element-by-element `.at[].set`
  writes were the worst case: their gradients compiled very slowly (solar radiation alone:
  233 s and 7.9 GB, now 1.9 s and 1.4 GB).
- **Runge–Kutta stages as one function** (`_rk_stage`), driven by a Python loop in normal runs
  and by `lax.scan` when traced, so the physics is traced once instead of five times.
  `RungeKuttaUpdate` takes the stage number as data (masked sums instead of slices).

Result: one step's traced program went from ~840k to ~150k operations; `jax.grad` of a step
compiles in ~4 min with ~6.5 GB; gradients with respect to the state and to a parameter
(`vcmax25`) agree with finite differences (`tests/test_differentiability.py`). Normal runs are
also faster (2.7 s per step, from 4.5 s). All changes alter round-off only: per-routine replay
≤ 3e-14, whole run ≤ 4.5e-11 (within the noise twins).

## 2. Root solvers: forward values and implicit-function gradients

**Context.** Three quantities are found by iterative root solvers: the Obukhov length
(`clm_ml.processes.turbulence.obukhov_length`), leaf intercellular CO₂ (`clm_ml.processes.photosynthesis.solve_ci`) and stomatal conductance under
water-use-efficiency optimization (`clm_ml.processes.photosynthesis.solve_gs_bisection`). Differentiating *through* the iterations
gives wrong or exploding gradients (observed ~1e144 for the Obukhov solver), so the traceable
code uses the implicit function theorem (IFT). If the solver finds `x*` with `F(x*; θ) = 0`, then
`dx*/dθ = −(∂F/∂θ) / (∂F/∂x)`.

**Problem found.** The IFT was implemented as one Newton step,
`x = x0 − F(x0)/stop_gradient(F′(x0))`. That gives the right gradient, but **it also changes the
forward value** whenever the solver is not fully converged (`F(x0) ≠ 0`). The traceable path
therefore computed slightly different physics from the model that runs (Obukhov length, CO₂,
stomatal conductance), and gradients described that slightly different model.

**Decision, applied to the Obukhov length.** The running model uses the solver's value with no
Newton step. Only the traceable copy added one, so the traceable copy is fixed to apply the
correction to the gradient only:

```python
step = F(x0) / stop_gradient(F'(x0))          # zero where the IFT is ill-conditioned
x = x0 - (step - stop_gradient(step))         # forward: exactly x0; gradient: the IFT
```

The forward value is exactly the solver's root, as in the running model. The gradient is
unchanged.

**The slope `F′(x0)` must be accurate where the gradient is used (decision 119, 2026-11-14).**
For the Obukhov length it was a central difference with a fixed step of 1 m: good when |L|
is hundreds of metres, up to 49 % off when |L| is ~10 m (strongly stable or unstable; the
residual bends within a metre, and has a corner at the limit L_c/L = 1 of the
roughness-sublayer functions). It is now the exact derivative at the solution, by
forward-mode differentiation of the residual with the inputs held constant. Likewise a
solver's fallback value must keep its own gradient: where the WUE optimization has no root,
`solve_gs_bisection` returns `gs_min` with the gradient of `gs_min` (it was passed through
`stop_gradient`, which halved the sensitivity of the fluxes to `gs_min`). Neither changes a
forward value. The slopes of `solve_ci` and `solve_gs_bisection` are still central
differences (step 1e-4 in CO₂ and in conductance); the WUE one agrees with finite
differences of the model to 6e-5 over a CHATS7 day (`gradient_progress.md` round 2).

**Linear solves.** The coupled tridiagonal solve of the flux-profile solution (`tridiag_2eq`)
is a `jax.lax.custom_linear_solve` (decision 104): the forward value is Fortran's block Thomas
elimination; derivatives come from the implicit function theorem for `A x = d` (forward mode:
`A dx = dd − dA x`; reverse mode: the transpose system `Aᵀ z = g`, solved by the same
elimination, and `dL/dA = −z xᵀ`), never from differentiating the elimination (whose
intermediate coefficients can be huge). Both `jax.grad` and `jax.jvp` therefore work. The
other tridiagonal solves (longwave radiation, soil temperature) are differentiated directly.

**Not applied (yet) to stomatal conductance and intercellular CO₂.** For these two, the running
model itself uses the IFT helpers (`solve_gs_bisection` for WUE; `solve_ci` for
Medlyn/Ball–Berry). So the Newton-shifted forward value *is* the validated
behavior. Removing the shift changes results (about 1e-5 relative in stomatal conductance), so
it is listed in §4 as a deviation from Fortran to fix in a fidelity step.

## 3. Roughness length: the Fortran bisection stopping rule

(Here the running model already matched Fortran; only the traceable copy differed.)

**Problem found.** Fortran `RoughnessLength` finds `z0m` by bisection on `[0, h_top]`, stopping
when the bracket is narrower than `err = 0.001 m` (at most `nmax = 20` iterations), and returns
the last midpoint. The normal Python version matches this. The traceable version always ran 20
iterations and returned the midpoint of the final bracket. That is more refined, but it is not
what CLM-ml v2 computes: `z0m` differed by up to 4e-4 relative.

**Decision.** The traceable version uses the same stopping rule, through a shared helper,
`MLMathToolsMod.bisection_traceable`. That helper mirrors Fortran `bisection`: halve while
`|b − a| > tol`, keep `[a, c]` when `f(a)·f(c) < 0`, and return the last midpoint.
`lax.while_loop` cannot be reverse-differentiated, so it is written as a fixed-length
`fori_loop` whose updates stop once the tolerance is met. The iterations and result are identical to Fortran's, and it remains
differentiable. (The derivative of a bisection result is zero almost everywhere. This is the
Fortran algorithm's property, not something introduced here.)

## 4. Known deviations of the running model from Fortran (kept; see `open_items.md` O5)

Each is pre-existing behavior of the validated translation (CHATS7 May 2007 within 2.5% of
Fortran per output column). Each would change results, so each is a fidelity change for a later,
validated step (plan Phase 4b or later).

### 4.1 Obukhov-length solver

Fortran `GetObu` (MLCanopyTurbulenceMod.F90, lines 217–260) calls the hybrid solver
`MLMathToolsMod::hybrid`:
- secant iterations from two guesses, `obu0 = 100 m` and `obu1 = −100 m`;
- stops when the step is smaller than `tol = 0.1 m`;
- switches to Brent's method (`zbrent`) if a sign change brackets the root;
- at most 40 iterations.

The model state is left as computed by the *last* function evaluation (the returned root is a
dummy).

The JAX translation, in both its normal and traceable paths, instead runs **exactly 25 secant
steps** from a single start (`clm_ml.processes.turbulence.obukhov_length`, Phase 7.2; before:
`_obu_fixed_iter`) and then recomputes the derived quantities from
that value. This predates the modernization. It is within the validated agreement with Fortran
(CHATS7 May 2007: ≤ 2.5% per output column, r ≥ 0.9999), but it is not the Fortran algorithm.

**Bug fixed (Phase 3).** The fixed-step secant iteration took a huge step when two consecutive
residuals were exactly equal after convergence; it now takes no step (§5). The same fix is in
the intercellular-CO₂ solver (§4.3).

### 4.2 Stomatal conductance (WUE optimization, the default `gs_type = 2`)

Fortran `StomataOptimization` (MLLeafPhotosynthesisMod.F90, lines 798–850) finds gs with
`bisection` between `gsmin` and 2 mol m⁻² s⁻¹, stopping when the bracket is narrower than
`tol = 0.001` mol m⁻² s⁻¹. It returns the last midpoint.

The running JAX model (`clm_ml.processes.photosynthesis.solve_gs_bisection`; before Phase 7.2
`_bisect_gs_ift`) instead:
- does **20 halvings** and takes the final bracket's midpoint;
- then applies one **Newton step** (part of its implicit-function gradient).

So the result is closer to the exact root than Fortran's. Switching to Fortran's rule changed
gs by up to 1e-2 relative in a test at dusk and midday (the size of the tolerance).

A tolerance-based `_StomataOptimization` that matched Fortran existed in the same module, but no
run used it; it was deleted in Phase 3d with the other unused scalar copies. The fidelity step
can use `clm_ml.processes.math.bisection_traceable` (§3), which has Fortran's stopping rule.

### 4.3 Intercellular CO₂ (Medlyn and Ball–Berry, `gs_type = 0, 1`; not the CHATS default)

Fortran uses `hybrid` with `tol = 0.1` (mol/mol scaled). The running JAX model uses
`clm_ml.processes.photosynthesis.solve_ci` (before Phase 7.2 `_ci_solver_scan_ift`): 40 fixed
secant steps followed by a Newton step.

### 4.4 Runge–Kutta update skipped the top canopy-air layer (fixed)

Fortran `RungeKuttaUpdate` (MLRungeKuttaMod.F90, line 90) updates layers `ic = 1, ncan(p)`. The
JAX translation sliced `[p, :n]`, i.e. levels 0 … ncan−1: the top canopy-air layer (`ncan`,
just below the reference height) never received the intermediate Runge–Kutta states, and the
unused level 0 was updated. Found by poisoning each state field with NaN before a step
(Phase 3e): `tair_bef(0)` reached the results.

**Fixed** (after Phase 4, at the maintainer's request): the slices are now `1 : ncan + 1`.

**Effect on results: none for CHATS7.** A full-month (May 2007) run gives byte-identical
output files with and without the fix, and the regression run is bit for bit unchanged. The
reason: within a Runge–Kutta stage no process reads the top layer's current air
temperature, vapor pressure or water. The flux–profile solution starts from the values at
the beginning of the sub-step, turbulence uses the canopy-top layer (`ntop`, inside the
canopy), and leaf and radiation processes use the foliage layers; the final pass of each
sub-step recomputes the state. The fix matters only if a process reads that layer
mid-step. The per-routine reference (`process_io.pkl.gz`) was re-recorded, because
`RungeKuttaUpdate` itself now returns different values for level 0 and the top layer.

### 4.5 CO₂ profile reset covers levels 0 … ncan−1 (not changed)

Fortran `FluxProfileSolution` sets `cair(p,ic) = co2ref(p)` for `ic = 1, ncan(p)`. The JAX
translation sets levels 0 … ncan−1 (the same slicing pattern as §4.4): level 0 is set and the
top layer `ncan` is not. It has no effect today, because `cair` is never computed (no CO₂
profile: every layer is initialized to the reference value and stays there), so the top
layer keeps the reference value anyway. Found in Phase 5b; kept as translated (a refactor
does not change behaviour); to fix together with any change that makes `cair` vary.

## 5. The apparent evening-transition sensitivity was a solver bug (fixed)

**What was seen.** Two runs differing only by `leaf_emissivity × (1 ± 1e-12)` (the noise twins)
differed by up to 4.5% in turbulence fields (`mflx_profile`, `gac_profile`, wind at the leaves,
u*), with jumps at single steps (largest at step 7, the evening transition). A refactor that
changed only round-off produced jumps of the same kind. This was first read as a property of
the model.

**Cause.** The secant iteration of the Obukhov-length solver (`_obu_body_pure`; the same code is
in the intercellular-CO₂ solver, now `clm_ml.processes.photosynthesis._secant`) runs a fixed number of steps and does not stop
once converged. After about five steps the residual is at round-off level (±1e-14). When two
consecutive residuals come out *exactly equal*, the secant denominator `f1 − f0` is zero, and the
code divided by a floor of 1e-30 instead:

```
dx = −f1 · (x1 − x0) / max(|f1 − f0|, 1e-30)  ≈  (1e-14 · 1e-14) / 1e-30  ≈  100 m
```

The converged iterate was thrown far away (in one case from L = 73.6 m, stable, to −128 m,
unstable). The secant then pulled it back over the following steps, in a cycle, and the value
returned after step 25 depended on where in that cycle the iteration ended. Whether this
happened depended on round-off in the last digits, hence the "sensitivity".

**Fix.** When `f1 == f0` the secant step is undefined, so no step is taken
(`dx = where(f1 == f0, 0, …)`). Converged values are unchanged; Fortran's `hybrid`, which stops
when the step is below 0.1 m, never does this.

**Effect on the reference day** (CHATS7, 50 steps):
- The old reference itself was distorted by the bug at several steps: the fix changes it by up
  to 2.4% (relative to the field's magnitude) at step 32, 0.8% at step 7 and 0.4% at steps 5 and
  23, and by ≤ 1e-4 elsewhere.
- **The twins now differ from the reference by at most 4e-11** (any field, any step), against
  4.5e-2 before. The model is well conditioned: a perturbation in the 12th digit stays in the
  11th digit.

| Field | Twin difference before | after |
|---|---|---|
| momentum flux profile (`mflx_profile`) | 4.5e-2 | 7.9e-13 |
| aerodynamic conductance profile (`gac_profile`) | 4.4e-2 | 3.8e-12 |
| wind speed at shaded leaves | 4.4e-2 | 2.6e-12 |
| canopy air heat storage (`stflx_air_canopy`) | 3.2e-2 | 3.1e-12 |
| u* | 3.1e-2 | 7.7e-13 |
| sensible heat flux | 1.2e-2 | 3.0e-11 |

The bug may account for part of the 2.5% tolerance needed in the JAX–Fortran comparison of the
full month (`tests/test_fortran_python_comparison.py`, not yet re-run with the fix).

## 6. Regression testing policy

A change that alters only round-off (compiling code, a different order of operations, GPU)
cannot reproduce the reference bit for bit, so a tolerance is needed. The model's own
sensitivity to round-off, measured by noise twins, sets it.

The policy has two checks:

1. **Bit-for-bit** (`test_bit_identical_to_reference`). Refactors that should not change the
   arithmetic (moving code, explicit arguments instead of globals) must reproduce the reference
   exactly. All of Phases 0–3b met this. Fill values (|x| ≥ 1e30: `spval` and sums of it in
   unused level 0 or leaf index 0) must sit in the same places, but their last bits are not
   compared; they carry no information (Phase 3e: rebuilding each step from a template changed
   only those bits).
2. **Noise twin** (`test_within_noise_of_reference`). For changes that legitimately alter
   round-off (a traceable implementation, whole-run compilation, GPU), the comparison is against
   the model's own sensitivity. The reference is stored with **twin runs** that differ only by
   a 1e-12 relative perturbation (`leaf_emissivity × (1 ± 1e-12)`). For each output field, the
   noise is the largest difference between the reference and either twin over all steps. A
   change passes if, for every field, its largest difference from the reference is at most
   **2 × the noise** (plus 1e-12 × the field's magnitude, for fields the twins leave unchanged).

**Workflow for a round-off-changing step:**
1. Run the regression tests. The noise-twin test must pass. The bit-for-bit test is expected to
   fail.
2. Regenerate the reference and twins with `python tests/regression/record_run.py
   --update-reference`. Say in the commit that the step changes round-off only, and why.
3. The bit-for-bit test then guards the next refactors again.

**Where the reference is recorded.** The last bits depend on the machine and the JAX version.
Since Phase 6 (2026-10-28) the reference is recorded on NCAR Derecho (CPU), with JAX 0.9.2
until 2026-11-02 and JAX 0.10.2 since (decision 105: 67 of 96 fields changed, by at most
1.9e-9 relative; noise-twin test passed); moving there from the laptop (JAX 0.6.2) changed 65
fields by ~1e-16 and passed the noise-twin test. On another machine or JAX version, rely on the noise-twin test and expect the
bit-for-bit test to fail.

**CPU only (Phase 7, decision C4 in `phase7_decisions.md`).** The Derecho login nodes have
GPUs, which JAX uses by default; `tests/conftest.py` and `record_run.py` select the CPU. On the
GPU, compiling the same program in two processes gives a different program in about one case
in five (~1e-11 from the first step; within the noise twins), so the bit-for-bit test is a
CPU test; on the CPU fresh compiles are identical to each other and to the reference.

**Phase 6 round-off changes** (each passed the noise-twin test; reference regenerated):
- Per-site data as inputs of the compiled chunk (step 2a): XLA no longer folds divisions by
  the constant soil layer thickness (identical-input check of `SoilWater`: 1e-15; run
  ≤ 1.4e-11).
- The step batched over patches with `vmap` (step 3a), also for one patch: XLA fuses the
  batched step differently (single values differ in the last digit; run ≤ 2.7e-11).
- The identical-input replay (`test_process_io.py`) of `FluxProfileSolution` gives
  `shair_profile` (sensible heat between air layers, a difference of nearly equal
  temperatures) 1.1e-10 relative with JAX 0.9.2 on Derecho against the JAX 0.6.2 recording,
  before any Phase 6 change; the solved temperatures agree to < 1e-13. That field alone is
  allowed 3e-10.
- **Do not scatter into a scan's carry with array indices** (observed with JAX 0.9.2 on CPU,
  Phase 6 step 4/5, not reduced to a minimal reproducer): gathering patch groups with
  `jnp.take` and writing their results back into the carried state with `.at[idx].set` inside
  the chunk's `lax.scan` gave wrong states for groups of two or more patches and intermittent
  segmentation faults, while the identical operations outside a scan were right. The patch
  groups are contiguous slot ranges taken by slicing and joined by concatenation instead.
- **Intermittent segmentation faults while compiling large CPU programs** (JAX 0.9.2 on
  Derecho; the programs with C3 and C4 groups in one chunk): `gdb` shows a deep recursion in
  LLVM's JIT linker (`llvm::orc::RTDyldObjectLinkingLayer::emit` → `ExecutionSession::lookup`
  → `jitLinkForORC` …, 245 nested levels) overflowing a compile thread's stack. XLA splits
  a CPU program into pieces compiled in parallel; `--xla_cpu_parallel_codegen_split_count=1`
  (one piece) avoids it (5 of 5 compilations clean, against about half crashing). `clm_ml`
  sets it on import unless `XLA_FLAGS` already choose it. Separate from the scatter problem
  above (which gave wrong values, not only crashes).

**How tight it is.** Since the solver fix (§5), the twin noise is at most about 4e-11 relative,
so the noise-twin test detects any change larger than about 1e-10. (Before the fix it allowed
up to ~9% at a single step.) The other checks remain:
- the bit-for-bit test, for refactors;
- the per-routine identical-input checks (§1), for changes that alter round-off;
- the Fortran golden tests (`rtol = 1e-9`).

A change that alters round-off is accepted only when it passes **both** the noise-twin test and
an identical-input check of every routine it touches.

### 6.1 When to run which tests

Each end-to-end test runs the whole CHATS7 model in a fresh process. Before Phase 5a that
cost ~40–60 s per run, mostly compiling ~540 programs that the on-disk compilation cache did
not keep; since Phase 5a every program is cached, so a repeated run takes ~20 s (the first
run after a change to the compiled code still compiles). The policy above needs only one of
the runs, so the suites are run in three tiers (times measured after Phase 5a, cache warm):

| When | Tests | Time |
|---|---|---|
| After each edit | `tests/clm_ml` and `tests/fortran_validation -m "not slow"`, plus the test being written | ~10 s |
| Before each commit | the bit-for-bit and noise-twin tests (`test_end_to_end_regression.py`, `test_clm_ml_equivalence.py`; one shared run), plus the changed feature's own end-to-end test | ~1 min (with tier 1: 41 s) |
| At the end of a phase, before merging to `main` | all of `tests/regression -m regression` (~8 min), and the slow tests: `test_differentiability.py` and `test_model_api.py` (~20 min, gradient compilation), `test_fortran_python_comparison.py` | ~30 min |

**Phase 7** (structure only, maintainer 2026-10-30) uses a smaller set: lint, the bit-for-bit
and noise-twin test and the moved routines' golden tests per commit, gradient and feature tests
at package ends, the rest at the phase end (`phase7_progress.md`, "Testing in Phase 7").

The feature tests (restarts, ensembles, output streams, coarse forcing, land-data files,
parameters from YAML, per-routine process IO) check behaviour that a change elsewhere rarely
touches; a change that does touch it runs that test before committing. New end-to-end tests
simulate as little as they need (usually one day) with as few separate runs as possible.
