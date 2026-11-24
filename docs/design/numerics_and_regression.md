# Iterative solvers, differentiability, and regression testing

How the model's iterative solvers, gradients and round-off are handled, and the regression
policy that follows from them. Each section states the rule, why, and the evidence. Fortran
references are to CLM-ml v2 (`gbonan/CLM-ml_v2.CHATS`).

---

## 1. One implementation per process, matching the validated model

**Rule.** Each process has **one implementation**, written so JAX can trace it, used both for
runs and for gradients. It computes **what the validated translation computes**. A new
implementation of a routine replaces the old one only when it reproduces it **on identical
input** to round-off (about 1e-15 relative): both are called on the same inputs, captured
from a run (for example at dusk, night and midday of the CHATS7 reference day), and every
output field is compared.

Where the model differs from Fortran, the difference is recorded in §4. Changing it is a
fidelity change: separate, validated, with a regenerated reference. Using Fortran directly in
a change meant to keep results would change them.

**Checks need concrete values.** The conservation checks are recorded in the step
(`Diagnostics.balance_error`) and tested on the host after each chunk; under `jax.grad`
(`Model.run_chunk`) the errors are traced values and the host test is skipped
(`processes.math.is_concrete`).

**Compiled helpers.** An iterative helper compiled once per residual function
(`processes.math.bisection_traceable`) takes a module-level residual with explicit
parameters: a closure would be a new function at every call and recompile every time.

**Small programs for gradients.** The memory XLA needs to compile a gradient grows much faster
than the size of the program, so the step is kept small:
- **No per-layer Python loops**: array slices or `vmap` over layers, same formula per layer
  (turbulence, the flux–profile coefficients and fluxes, solar radiation optics; the
  two-stream sweeps are `lax.scan` over layers with both wavebands at once).
  Element-by-element `.at[].set` writes are the worst case: their gradients compile very
  slowly (solar radiation alone: 233 s and 7.9 GB, against 1.9 s and 1.4 GB vectorized).
- **Runge–Kutta stages as one function** (`canopy._stage`) driven by `lax.scan`, so the
  physics is traced once instead of once per stage; `integrators.runge_kutta_update` takes
  the stage number as data (masked sums instead of slices). The sub-steps are a `lax.scan` too.
- **Recomputation** (`jax.checkpoint`) on each step, and optionally on each sub-step
  (`numerics.checkpoint_substeps`), so backward memory does not grow with the number of
  steps or sub-steps.

Unvectorized, `jax.grad` of one step (6 sub-steps × the Runge–Kutta stages of the full physics)
needed ~87 GB to compile; vectorized, a step's traced program has ~150k operations instead of
~840k. Gradients with respect to the state and to a parameter (`vcmax25`) agree with finite
differences (`tests/test_differentiability.py`).

## 2. Root solvers: forward values and implicit-function gradients

**Context.** Three quantities are found by iterative root solvers: the Obukhov length
(`clm_ml.processes.turbulence.obukhov_length`), leaf intercellular CO₂
(`clm_ml.processes.photosynthesis.solve_ci`) and stomatal conductance under
water-use-efficiency optimization (`clm_ml.processes.photosynthesis.solve_gs_bisection`).
Differentiating *through* the iterations gives wrong or exploding gradients (observed ~1e144
for the Obukhov solver), so the gradients come from the implicit function theorem (IFT). If
the solver finds `x*` with `F(x*; θ) = 0`, then `dx*/dθ = −(∂F/∂θ) / (∂F/∂x)`.

**Pitfall.** Applying the IFT as one Newton step, `x = x0 − F(x0)/stop_gradient(F′(x0))`,
gives the right gradient, but **it also changes the forward value** whenever the solver is not
fully converged (`F(x0) ≠ 0`): the gradient then describes a slightly different model from the
one that runs.

**Rule, applied to the Obukhov length.** The forward value is the solver's own answer, with no
Newton step; the correction is applied to the gradient only:

```python
step = F(x0) / stop_gradient(F'(x0))          # zero where the IFT is ill-conditioned
x = x0 - (step - stop_gradient(step))         # forward: exactly x0; gradient: the IFT
```

The forward value is exactly the solver's root; the gradient is the IFT one.

**The slope `F′(x0)` must be accurate where the gradient is used (decisions.md 119).**
For the Obukhov length it is the exact derivative at the solution, by forward-mode
differentiation of the residual with the inputs held constant. A central difference with a
fixed step of 1 m is good when |L| is hundreds of metres, but up to 49 % off when |L| is
~10 m (strongly stable or unstable; the residual bends within a metre, and has a corner at
the limit L_c/L = 1 of the roughness-sublayer functions). Likewise a solver's fallback value
keeps its own gradient: where the WUE optimization has no root, `solve_gs_bisection` returns
`gs_min` with the gradient of `gs_min` (through `stop_gradient`, the sensitivity of the
fluxes to `gs_min` would be halved). Neither changes a forward value. The slopes of
`solve_ci` and `solve_gs_bisection` are central differences (step 1e-4 in CO₂ and in
conductance); the WUE one agrees with finite differences of the model to 6e-5 over a CHATS7
day.

**Linear solves.** The coupled tridiagonal solve of the flux-profile solution (`tridiag_2eq`)
is a `jax.lax.custom_linear_solve` (decisions.md 104): the forward value is Fortran's block
elimination; derivatives come from the implicit function theorem for `A x = d` (forward mode:
`A dx = dd − dA x`; reverse mode: the transpose system `Aᵀ z = g`, solved by the same
elimination, and `dL/dA = −z xᵀ`), never from differentiating the elimination (whose
intermediate coefficients can be huge). Both `jax.grad` and `jax.jvp` therefore work. The
other tridiagonal solves (longwave radiation, soil temperature) are differentiated directly.

**Not applied to stomatal conductance and intercellular CO₂.** For these two, the forward
value of the IFT helpers (`solve_gs_bisection` for WUE; `solve_ci` for Medlyn/Ball–Berry)
includes the Newton step, and that Newton-shifted value *is* the validated behavior.
Removing the shift changes results (about 1e-5 relative in stomatal conductance), so it is
listed in §4 as a deviation from Fortran to fix in a fidelity step.

## 3. Roughness length: the Fortran bisection stopping rule

Fortran `RoughnessLength` finds `z0m` by bisection on `[0, h_top]`, stopping when the bracket
is narrower than `err = 0.001 m` (at most `nmax = 20` iterations), and returns the last
midpoint. Always running 20 iterations and returning the midpoint of the final bracket is more
refined, but it is not what CLM-ml v2 computes: `z0m` differs by up to 4e-4 relative.

**Rule.** `turbulence.roughness_length` uses the same stopping rule, through a shared helper,
`processes.math.bisection_traceable`. That helper mirrors Fortran `bisection`: halve while
`|b − a| > tol`, keep `[a, c]` when `f(a)·f(c) < 0`, and return the last midpoint.
`lax.while_loop` cannot be reverse-differentiated, so it is written as a fixed-length
`fori_loop` whose updates stop once the tolerance is met. The iterations and result are
identical to Fortran's, and it remains differentiable. (The derivative of a bisection result
is zero almost everywhere. This is the Fortran algorithm's property, not something introduced
here.)

## 4. Known deviations of the model from Fortran (kept; see `open_items.md` O5)

Each is within the validated agreement with Fortran (CHATS7 May 2007 within 2.5% per output
column). Changing one changes results, so each is a fidelity change with its own validation.

### 4.1 Obukhov-length solver

Fortran `GetObu` (MLCanopyTurbulenceMod.F90) calls the hybrid solver
`MLMathToolsMod::hybrid`:
- secant iterations from two guesses, `obu0 = 100 m` and `obu1 = −100 m`;
- stops when the step is smaller than `tol = 0.1 m`;
- switches to Brent's method (`zbrent`) if a sign change brackets the root;
- at most 40 iterations.

The model state is left as computed by the *last* function evaluation (the returned root is a
dummy).

The JAX model instead runs **exactly 25 secant steps** from the same two guesses
(`clm_ml.processes.turbulence.obukhov_length`) and then recomputes the derived quantities
from that value. It is within the validated agreement with Fortran (CHATS7 May 2007: ≤ 2.5%
per output column, r ≥ 0.9999), but it is not the Fortran algorithm.

The fixed-step secant takes no step when two consecutive residuals are exactly equal (§5);
so does the intercellular-CO₂ solver (§4.3).

### 4.2 Stomatal conductance (WUE optimization, the default `gs_type = 2`)

Fortran `StomataOptimization` (MLLeafPhotosynthesisMod.F90) finds gs with
`bisection` between `gsmin` and 2 mol m⁻² s⁻¹, stopping when the bracket is narrower than
`tol = 0.001` mol m⁻² s⁻¹. It returns the last midpoint.

The JAX model (`clm_ml.processes.photosynthesis.solve_gs_bisection`) instead:
- does **20 halvings** and takes the final bracket's midpoint;
- then applies one **Newton step** (part of its implicit-function gradient).

So the result is closer to the exact root than Fortran's. Fortran's rule gives gs different
by up to 1e-2 relative (a test at dusk and midday; the size of the tolerance). A fidelity
change can use `clm_ml.processes.math.bisection_traceable` (§3), which has Fortran's stopping
rule.

### 4.3 Intercellular CO₂ (Medlyn and Ball–Berry, `gs_type = 0, 1`; not the CHATS default)

Fortran uses `hybrid` with `tol = 0.1` (mol/mol scaled). The JAX model uses
`clm_ml.processes.photosynthesis.solve_ci`: 40 fixed secant steps followed by a Newton step.

### 4.4 Runge–Kutta update of the top canopy-air layer (as Fortran)

Fortran `RungeKuttaUpdate` (MLRungeKuttaMod.F90) updates layers `ic = 1, ncan(p)`; the JAX
update (`integrators.runge_kutta_update`) covers the same layers, 0-based, including the top
canopy-air layer just below the reference height (decisions.md 51). Within a Runge–Kutta stage
no process reads the top layer's current air temperature, vapor pressure or water: the
flux–profile solution starts from the values at the beginning of the sub-step, turbulence
uses the canopy-top layer (`ntop`, inside the canopy), and leaf and radiation processes use
the foliage layers. Whether that layer receives the intermediate stage states therefore does
not change CHATS7 results; it matters only if a process reads that layer mid-step.

### 4.5 CO₂ profile reset leaves out the top layer (not changed)

Fortran `FluxProfileSolution` sets `cair(p,ic) = co2ref(p)` for `ic = 1, ncan(p)`, every layer.
The JAX model sets the layers below the top one (0-based `0 … n_layers − 2`; `canopy._fluxes`):
the top layer is not set. It has no effect, because `cair` is never computed (no CO₂ profile:
every layer is initialized to the reference value and stays there), so the top layer keeps
the reference value anyway. To fix together with any change that makes `cair` vary.

## 5. Secant solvers take no step when two residuals are equal

The secant iterations of the Obukhov-length solver (`turbulence._secant_step`) and of the
intercellular-CO₂ solver (`photosynthesis._secant`) run a fixed number of steps and do not stop
once converged. After about five steps the residual is at round-off level (±1e-14). When two
consecutive residuals come out *exactly equal*, the secant denominator `f1 − f0` is zero.
Dividing by a floor of 1e-30 instead would throw the converged iterate far away:

```
dx = −f1 · (x1 − x0) / max(|f1 − f0|, 1e-30)  ≈  (1e-14 · 1e-14) / 1e-30  ≈  100 m
```

(in one case from L = 73.6 m, stable, to −128 m, unstable). The secant would then pull it
back over the following steps, in a cycle, and the value returned after step 25 would depend
on where in that cycle the iteration ended, that is on round-off in the last digits.

**Rule.** When `f1 == f0` the secant step is undefined, so no step is taken
(`dx = where(f1 == f0, 0, …)`; decisions.md 52). Converged values are unchanged; Fortran's
`hybrid`, which stops when the step is below 0.1 m, never meets this case.

**Evidence** (CHATS7 reference day, 50 steps). With the 1e-30 floor, two runs differing only
by `leaf_emissivity × (1 ± 1e-12)` (the noise twins) differ by up to 4.5% in turbulence fields,
with jumps at single steps (largest at the evening transition), which looks like a
sensitivity of the model. With no step taken, **the twins differ from the reference by at most
4e-11** (any field, any step): the model is well conditioned, a perturbation in the 12th digit
stays in the 11th digit.

| Field | Twin difference with the 1e-30 floor | with no step |
|---|---|---|
| momentum flux profile (`mflx_profile`) | 4.5e-2 | 7.9e-13 |
| aerodynamic conductance profile (`gac_profile`) | 4.4e-2 | 3.8e-12 |
| wind speed at shaded leaves | 4.4e-2 | 2.6e-12 |
| canopy air heat storage (`stflx_air_canopy`) | 3.2e-2 | 3.1e-12 |
| u* | 3.1e-2 | 7.7e-13 |
| sensible heat flux | 1.2e-2 | 3.0e-11 |

## 6. Regression testing policy

A change that alters only round-off (compiling code, a different order of operations, GPU)
cannot reproduce the reference bit for bit, so a tolerance is needed. The model's own
sensitivity to round-off, measured by noise twins, sets it.

The policy has two checks:

1. **Bit-for-bit** (`test_bit_identical_to_reference`). Code changes that should not change
   the arithmetic (restructuring or moving code, explicit arguments instead of globals) must
   reproduce the reference exactly. Fill values (|x| ≥ 1e30: `spval` and sums of it) must sit
   in the same places, but their last bits are not compared; they carry no information.
2. **Noise twin** (`test_within_noise_of_reference`). For changes that legitimately alter
   round-off (compiling code differently, a different order of operations, GPU), the
   comparison is against the model's own sensitivity. The reference is stored with **twin
   runs** that differ only by a 1e-12 relative perturbation (`leaf_emissivity × (1 ± 1e-12)`).
   For each output field, the noise is the largest difference between the reference and
   either twin over all steps. A change passes if, for every field, its largest difference
   from the reference is at most **2 × the noise** (plus 1e-12 × the field's magnitude, for
   fields the twins leave unchanged).

**Workflow for a change that alters round-off:**
1. Run the regression tests. The noise-twin test must pass. The bit-for-bit test is expected to
   fail.
2. Regenerate the reference and twins with `python tests/regression/record_run.py
   --update-reference`. Say in the commit that the change alters round-off only, and why.
3. The bit-for-bit test then guards the next changes again.

**Where the reference is recorded.** The last bits depend on the machine and the JAX version.
The reference is recorded on NCAR Derecho (CPU) with JAX 0.10.2 (decisions.md 105). On another
machine or JAX version, rely on the noise-twin test and expect the bit-for-bit test to fail.

**CPU only** (decisions.md 68). The Derecho login nodes have GPUs, which JAX uses by default;
`tests/conftest.py` and `record_run.py` select the CPU. On the GPU, compiling the same program
in two processes gives a different program in about one case in five (~1e-11 from the first
step; within the noise twins), so the bit-for-bit test is a CPU test; on the CPU fresh compiles
are identical to each other and to the reference.

**Rules for compiled code from CPU defects** (JAX 0.9.2 on Derecho; decisions.md 64):
- **Do not scatter into a scan's carry with array indices** (not reduced to a minimal
  reproducer): gathering patch groups with `jnp.take` and writing their results back into the
  carried state with `.at[idx].set` inside the chunk's `lax.scan` gave wrong states for groups
  of two or more patches and intermittent segmentation faults, while the identical operations
  outside a scan were right. The patch groups are contiguous slot ranges taken by slicing and
  joined by concatenation instead.
- **Intermittent segmentation faults while compiling large CPU programs** (the programs with C3
  and C4 groups in one chunk): `gdb` shows a deep recursion in LLVM's JIT linker
  (`llvm::orc::RTDyldObjectLinkingLayer::emit` → `ExecutionSession::lookup` → `jitLinkForORC`
  …, 245 nested levels) overflowing a compile thread's stack. XLA splits a CPU program into
  pieces compiled in parallel; `--xla_cpu_parallel_codegen_split_count=1` (one piece) avoids
  it (5 of 5 compilations clean, against about half crashing). `clm_ml` sets it on import
  unless `XLA_FLAGS` already choose it. With JAX 0.10.2 these programs compile cleanly without
  it; it stays as a precaution. Separate from the scatter problem above (which gave wrong
  values, not only crashes).

**How tight it is.** With the secant rule of §5, the twin noise is at most about 4e-11
relative, so the noise-twin test detects any change larger than about 1e-10. The other checks
remain:
- the bit-for-bit test, for changes that should not alter the arithmetic;
- the per-routine identical-input checks (§1), for changes that alter round-off;
- the Fortran golden tests (`rtol = 1e-9`).

A change that alters round-off is accepted only when it passes **both** the noise-twin test and
an identical-input check of every routine it touches.

### 6.1 When to run which tests

Each end-to-end test runs the whole CHATS7 model in a fresh process. Every compiled program
is cached on disk, so a repeated run takes ~20 s; the first run after a change to the compiled
code compiles again (~40–60 s). The policy above needs only one of the runs, so the suites are
run in three tiers (times with the cache warm):

| When | Tests | Time |
|---|---|---|
| After each edit | `tests/clm_ml` and `tests/fortran_validation -m "not slow"`, plus the test being written | ~10 s |
| Before each commit | the bit-for-bit and noise-twin tests (`test_end_to_end_regression.py`, `test_clm_ml_equivalence.py`; one shared run), plus the changed feature's own end-to-end test | ~1 min (with tier 1: 41 s) |
| At the end of a phase, before merging to `main` | all of `tests/regression -m regression` (~8 min), and the slow tests: `test_differentiability.py` and `test_model_api.py` (~20 min, gradient compilation), `test_fortran_python_comparison.py` | ~30 min |

The feature tests (restarts, ensembles, output streams, coarse forcing, land-data files,
parameters from YAML, per-routine process IO) check behaviour that a change elsewhere rarely
touches; a change that does touch it runs that test before committing. New end-to-end tests
simulate as little as they need (usually one day) with as few separate runs as possible.
