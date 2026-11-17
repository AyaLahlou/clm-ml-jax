# Gradients: progress log

Open item O76 (maintainer, 2026-11-14). The use: **sensitivity experiments** of the model to
parameters, states and forcings, at a tower first and globally later. Order of work: speed
first, then the quality and accuracy of the gradients (zeros, NaNs, agreement with finite
differences). Branch `gpu_grad` (from `gpu_opt`).

Words used here:
- **forward-mode derivative** (`jax.jvp`, `jax.jacfwd`): one pass gives the response of
  *every output* to *one input direction*; its cost grows with the number of inputs.
- **reverse-mode gradient** (`jax.grad`): one pass gives the sensitivity of *one number* to
  *every input*; a second output needs a second pass.
- **checkpoint** (`jax.checkpoint`): the backward pass of a gradient needs the intermediate
  values of the forward run; a checkpoint recomputes them when needed instead of keeping
  them, trading time for memory.

## Summary (2026-11-16)

**What works.** Gradients are fast enough for tower experiments (CPU: forward 10 ms per
step, gradient 52 ms). At CHATS7 every parameter derivative agrees with finite differences
(rounds 2-3). On real 1° patches there are no NaNs, and after decisions 119-123 daily-mean
derivatives agree within 2.3 % on the test days and to three or four digits over 30 days
on the patches without the two problems below.

**What is left: two defects of the same kind.** A quantity the fluxes react to strongly is
taken from the start of a Runge-Kutta stage instead of being solved with the stage. The
forward model then overshoots from stage to stage, and AD returns the true, huge and
useless local slope.

| | Water stress | Obukhov length |
|---|---|---|
| lagged quantity | leaf water potential in `1/(1+(psi/psi50)^40)` | L from the canopy-top air |
| where | patches that reach stress (Great Plains July: 7 of 21, from day 4) | stability transitions, afternoon and evening; rare in the Great Plains (2 of 21 patches), frequent in the wet-season cerrado (10 of 18) |
| effect on a 30-day mean derivative | 3 to 375 % off | up to 5e4 times too large |
| consistent forward form | round 9: within 2.3 %; forward results change by 0.1 W m-2 in the mean; forward 1.45x, gradient 1.75x | not prototyped |
| derivative-only form (forward bit for bit) | round 10: within 6 %, no measurable cost | round 11: Great Plains within 0.5 %; cerrado within 12 %, a third of daily means still > 5 % off; cost not yet meaningful |
| derivative-only form, as decided (rounds 12-13) | round 10's form, on | a derivative rule, blended by the loop's gain: forward bit for bit and at its speed, gradient 1.7x; Great Plains within 0.5 %, cerrado within 12 % |
| status | decision 124, **on hold** (2026-11-16): not built | decision 124, **on hold**: not built |

**Both together over 30 days (round 13):** Great Plains July 19 of 21 patches within 1.3 %
(worst 6.6 %), cerrado January 14 of 18 within 5 % (worst 11.6 %; a third of the daily
means more than 5 % off); no step derivative above 1.3e3 (repository: 1.6e9). Cutting the
link from the canopy-top air to L instead (free) is biased by 5-20 % (round 12).

**What it means for gradient-based science.**

- Usable now: CHATS7-like towers, and 1° patches over a few days or without stress and
  without frequent stability flips. Check any new case against a finite change of the
  mean (the forward model is reproducible, so finite differences of means are sound).
- With decision 124 (on hold) built: month-long derivatives on water-stressed patches (within ~6 %)
  and, with the Obukhov rule on, in regimes like the wet-season cerrado (monthly means
  within ~12 %, single days tens of percent). These are approximations: the gradient is
  then not the derivative of the forward values, and a finite difference at a single bad
  step disagrees by design.
- The consistent forward forms (round 9; not prototyped for L) give exact derivatives
  because they remove the overshoot from the model; they change forward results and cost
  forward time. Not chosen (maintainer, 2026-11-16: approximate, linearized gradients are
  acceptable; the forward model stays fast).

**Not yet looked at:** state and forcing derivatives on 1° patches, other parameters than
`iota` and other outputs than latent heat, other seasons and regions, and the open points
of round 1 (mode, checkpoints, GPU).

## Resume here

**State (2026-11-16).** Rounds 1 (speed) and 2-4 (accuracy) are measured. In the code: the
two gradient corrections of round 2 (decision 119; forward values unchanged) and the `avmu`
fix of round 3 (decision 120), and no water stress at or above zero leaf water potential
(decision 121; round 4's NaN gradients with dew, O79, closed). Round 5 (2026-11-15, no code
change): the corners are kinks, not jumps, but short canopies amplify round-off ~1e11 at the
evening transition and their daily-mean latent heat gradient can be 10-40 % off a finite
change. Round 6 (2026-11-15, prototype in scratch): a stable limit on zeta of 0.5 for the
short PFTs removes that alternation. Round 7 (2026-11-15, prototype in scratch): the
exploding derivatives (1e35 in a day, 1e220 in five) are the water on the leaves, whose
wetted fraction is taken from the water before the sub-step; with the wetted fraction of
the water left at the end of the sub-step they are gone. **Both are in the code (maintainer 2026-11-15): the implicit wetted fraction (decision
122; `numerics.wetted_fraction`, `explicit` is Fortran's) and the stable zeta limit as a
PFT parameter `zeta_max_stable` (decision 123; 0.5 for shrubs, grasses and crops, 1e30 = no
limit for trees).** With them the daily-mean gradients agree with finite differences
within 2.3 % on the four test days. **Not yet over 30 days (round 8, at the end): good
for the first three days, then single patches go wrong.** Next: the leaf water potential
(water stress) treated like the leaf water, and the diagnosis of what is left.
Round 9 (2026-11-15, prototype in scratch): the stress taken from the leaf water potential
of the end of the sub-step, solved with its own transpiration, removes the water-stress
part (Great Plains, 30 days: 97 -> 27 patch-days off, as with water stress off) at a
forward cost of 1.45x on a CHATS7 day as written (50 halvings). The cerrado is not water
stress. Round 10 (prototype, not adopted): a derivative-only form of the same, forward
unchanged, approximate. Round 11 (2026-11-16, prototype, not adopted): what was left in
both boxes is the Obukhov length, solved from the canopy-top air of the stage's start; a
derivative-only prototype removes the explosions but is only good to tens of percent on
bad cerrado days. Round 12 (2026-11-16): the derivative of L without the canopy-top air
of the stage's start (free) is biased by 5-20 %: not used. Round 13 (2026-11-16): round
11's treatment as a derivative rule, blended by the loop's gain; with round 10's water
stress, 30 days in both boxes; forward unchanged, gradient 1.8 times. Decision 124 (maintainer 2026-11-16: the
derivative-only water stress adopted, the Obukhov rule a setting, off by default) was put
**on hold the same day**: more such feedbacks are expected and one patch per feedback and
regime is not wanted; a general solution is looked for first (round 14). Round 14 (2026-11-16): other Runge-Kutta schemes
do not solve it. Round 15 (2026-11-16): an implicit derivative over the stages (general,
bounded, the exact gradient's cost, but biased in the morning transition and worse than
rounds 10 + 13 over 30 cerrado days), and **the checkpoint on the sub-step costs a factor 6
in gradient time on a box of 18 patches**. Round 16 (2026-11-16): on a GPU that checkpoint
costs no time and saves memory, so it is now the setting `numerics.checkpoint_substeps`,
which follows the device (decision 125, in the code); the candidate of round 15 (the exact
derivative where the time stepping is stable, the implicit one where it is not) is exact
on CHATS7 and bounded in the cerrado, but leaves three Great Plains patches exploding: not
yet the general solution. Next (maintainer, 2026-11-16): re-evaluate the direction. **Nothing of
rounds 9-16 but that setting is in the code** (prototype: `$SCRATCH/clm-ml-jax/trees/obu_rule`, its difference from
round 11's tree in `probe_cut/obu_rule.diff`; settings, tests and a reliability flag as
listed at the end of round 13). The overview is the summary just above.

**Open from round 1** (O76): forward or reverse mode as the tuned path; the checkpoint
layout as a setting; GPU measurements of checkpointing and of batched forward mode; a
jitted derivative as the first call on a model fails; the CPU gradient with written-out
loops ended without a result.

**Open from rounds 2-4** (O76): the 1° check in other seasons and regions; state and
forcing derivatives on 1° patches.

**How to measure** (probes in `jobs/gpu_probes/`, results in `$SCRATCH/clm-ml-jax/probe_grad0/`):

```bash
# CPU (login node): forward, one forward-mode derivative, gradient; any setting as an override
JAX_PLATFORMS=cpu python jobs/gpu_probes/grad0.py . [--wrt params|state|forcing] \
    [--modes fwd,jvp,grad,jgrad,jjvp,jacfwd] [--steps 8] [--stop MODULE:FUNCTION] [overrides]
# GPU (one A100, develop queue; submit from scratch, time a fixed snapshot of the code)
qsub -v TAG=name,TREE=$SCRATCH/clm-ml-jax/trees/NAME,SL=unrolled jobs/gpu_probes/grad0.pbs
# what the forward and gradient programs contain (static; CPU)
JAX_PLATFORMS=cpu python jobs/gpu_probes/census_grad.py .   # operations by source function
JAX_PLATFORMS=cpu python jobs/gpu_probes/loops_grad.py .    # every loop, with its trip count
# accuracy: a day of outputs against finite differences (CPU; params 5 min, forcing 5, state 20)
JAX_PLATFORMS=cpu python jobs/gpu_probes/gradcheck0.py . --wrt params|state|forcing [--h 1e-3,1e-5]
python jobs/gpu_probes/gradcheck_report.py gradcheck_params.npz [-v]
# the same for every patch of a 1° box (parameters; ~15 min for 20 patches)
JAX_PLATFORMS=cpu python jobs/gpu_probes/gradcheck_grid.py . "{lat: [35, 37], lon: [-100, -98]}"
```

Modes of `grad0.py`: `grad`/`jvp` are called as a user would (not jitted), `jgrad`/`jjvp`
jitted, `jacfwd` is the jitted forward-mode Jacobian over all inputs of `--wrt`. Each is
called four times; the time reported is the fastest of the last three. `--stop` removes the
derivative of one process (its results pass through `stop_gradient`): an ablation.

**Things learned that are easy to trip over**
- **Time a gradient from its third call.** The second call of an un-jitted `jax.grad` of
  `Model.run_chunk` is several times slower than the calls after it (decision 118).
- **A tower belongs on the CPU:** forward and gradient are 4–5 times faster there than on an
  A100 (one patch has nothing for a GPU to do in parallel).
- **Call the model once plainly before a jitted derivative** (`jax.jit(jax.grad(...))` as
  the first call fails: O76).
- The 1-day CHATS7 case has 48 steps and `Model` chunks start at step 1: at most 47 steps.
- **A derivative with respect to one parameter can hide a NaN in another:** differentiate
  with respect to the whole parameter set (as `jax.grad` over `params` does) when looking
  for NaN; a NaN partial derivative times a zero tangent is NaN in every parameter.
- **Run each gradient test file in its own process** (`tests/gradient_tier.py` does): O78.
- **AD is the slope at a point: on short canopies (0.5 m crops and grasses) check it against
  a finite change** at the sunrise and sunset steps before trusting a daily mean (round 5).

## Round 1: where a gradient's time goes (2026-11-14). No change to the model

CHATS7, one patch, a chunk of 8 steps from step 1, the derivative of the chunk-mean latent
heat flux. CPU: a Derecho login node (one core). GPU: one A100, code `c7f4a20`.

### The earlier gradient timings were of the wrong call

`timing/time_model.py` and `timing/time_vmap.py` timed the second gradient call. Times of
four successive calls (s, 8 steps, one patch):

| | Call 1 (compiles) | Call 2 | Calls 3, 4 |
|---|---|---|---|
| CPU, `jax.grad` | 40.2 | 1.70 | 0.39 |
| CPU, `jax.jit(jax.grad)` | 39.6 | 0.40 | 0.40 |
| A100, loops, `jax.grad` | 46.8 | 8.21 | 5.28 |
| A100, loops written out, `jax.grad` | 83.5 | 11.49 | 1.75 |

What the second call does was not tracked down. Its consequences: every gradient time
recorded before today is too high, and the step-2a conclusion that written-out loops slow
the gradient was wrong (`gpu_progress.md`, corrected; decision 118).

### Steady state

| ms per step | CPU | A100, loops | A100, written out |
|---|---|---|---|
| Forward run | 9.3 | 150 | 41 |
| One forward-mode derivative (all parameters moved together) | 17 (1.9×) | 215 | 55 (1.4×) |
| Reverse-mode gradient, parameters (27 values) | 48 (5.2×) | 659 | 219 (5.4×) |
| Reverse-mode gradient, initial state (428 values) | 45 | | |
| Reverse-mode gradient, forcing (464 values) | 54 | | |
| Forward-mode Jacobian, all 27 parameters | 88 | | |
| Forward-mode Jacobian, all 428 state values | 615 | | |

With 4 096 patches on the A100: forward 129, one forward-mode derivative 226, gradient 787
ms per step (loops: 269, 421, 1294).

- The cost per step does not depend on the chunk length (24 steps: 50 ms per step) and
  hardly on what is differentiated.
- **A reverse-mode gradient costs about 5 forward runs.** A forward-mode derivative costs
  under 2, and batched over inputs about 3 ms per step and input on the CPU: for the 27
  parameters, forward mode gives every output's sensitivity for less than two reverse-mode
  gradients.

### Why 5 forward runs

- **The step is computed three times.** The gradient program has the stomatal bisection and
  the Obukhov loop three times each (`loops_grad.py`): the forward sweep, then again for
  the checkpoint of each step, then again for the checkpoint of each sub-step. The
  checkpoints keep memory to the intermediate values of one sub-step (~2 MB per patch; a
  whole step would need ~12 MB per patch, a chunk that times its steps).
- **Checkpoints cost little time on the CPU:** without the step checkpoint 40.3 ms, without
  the sub-step checkpoint 41.1, without both 40.4 (with both: 47.7). Not measured on the GPU.
- **The solvers are not differentiated through their iterations** (stomatal bisection,
  Obukhov, roughness length): their loops carry no derivative, as intended.
- **No process stands out.** The gradient without the derivative of one process
  (`--stop`; 48.4 ms with all of them):

  | Derivative removed | Gradient, ms per step | Forward-mode derivative (16.3 with all) |
  |---|---|---|
  | Photosynthesis | 39.9 | 13.1 |
  | Flux-profile solution (cuts most of the paths through the state) | 40.5 | 12.1 |
  | Shortwave (optics and two-stream) | 42.9 | 15.4 |
  | Turbulence | 45.1 | 15.4 |
  | Leaf boundary layer | 46.9 | 14.6 |
  | Longwave, leaf water potential, nitrogen profile | 48.0–48.8 | 16.0–16.4 |

- The gradient program has 2.2 times the operations of the forward program
  (`census_grad.py`), and its loops stack 41 times as many values for the backward pass.

So a faster reverse mode is a matter of structure (fewer recomputations, cheaper backward
loops), worth perhaps 1.5×, not of one slow process.

### Not done

- GPU: checkpoint variants, the batched forward-mode Jacobian, a kernel profile of the
  gradient program.
- The CPU gradient with `numerics.solver_loops: unrolled` (forward: 17.8 ms per step, twice
  the loops' time) ended without a result or an error message.

## Round 2: are the gradients right? (2026-11-14). No change to the model

`jobs/gpu_probes/gradcheck0.py`: CHATS7, the 47 steps after the first (2007-05-01, a day
with a stable night, a morning transition and an unstable afternoon; no rain). Seven
outputs at every step (latent, sensible and ground heat flux, net radiation, GPP, canopy-top
air temperature, friction velocity). Each derivative by automatic differentiation (AD;
forward mode, with reverse mode checked against it) and by central finite differences (FD)
at three to five step sizes. Inputs: each of the 27 parameter values; each of the 396
state values in use; each forcing field (the field scaled at all steps together).
Results in `$SCRATCH/clm-ml-jax/probe_grad1/`.

### What is sound

- **No NaN or infinite derivative** anywhere in the day, for any input.
- **Reverse and forward mode agree** to 1e-9 (parameters, forcing) and 6e-10 (state).
- **Nothing that matters has a zero derivative by mistake** (but see gs_min and rain below).
  Zero where it should be: the Medlyn and Ball-Berry parameters (the run uses the WUE
  optimization), the leaf state of layers without leaves, the CO2 state outside the canopy.

### What is wrong: AD and FD disagree

FD is steady across its step sizes at the steps in question, so the model is smooth there
and it is the AD value that is off.

| Input | Pairs (input, output) within 0.1 % of FD | Worst |
|---|---|---|
| Parameters (161 pairs with a dependence) | 20 | `gs_min`: half the true value at every step; the others 0.1–9 % of the day's largest value, at a few steps much more (vcmax25 on latent heat at sunrise: 3.6 times too small) |
| Forcing fields (203 pairs, without the calendar day) | 10 | 4–20 % |
| State (1 200 pairs above the FD noise) | 912 | 21 pairs beyond 5 % |

The errors sit at particular steps: 17–18 and 20–21 (night), 25–28 (sunrise), 30–38
(afternoon, mostly friction velocity). Two causes, both in how the gradient of an iterative
solver is formed (decision 59: the forward value is the solver's answer, the gradient comes
from the implicit-function rule, i.e. from the slope of the solved equation at the answer):

1. **The Obukhov length's slope is a finite difference with a fixed 1 m step**
   (`turbulence.py::obukhov_length`). Right when |L| is hundreds of metres, wrong when it is
   ~10 m: at those steps L is 4–15 m at night and −7 to −14 m in the afternoon, and the
   solved function bends within a metre (in the afternoon), or has a corner at the limit of
   the roughness-sublayer functions, L = L_c ≈ 14.8 m (L_c/L ≤ 1; the step straddles it at
   night and at sunrise). Printed from the solver for all 1 410 solves of the day: the 1 m
   slope differs from the exact one by up to 49 %, at exactly the steps above; the solver
   itself converges everywhere (residual below 1e-10 m).
2. **`gs_min` loses half of its effect** (`photosynthesis.py::solve_gs_bisection`). Where
   the WUE optimization has no root (dark or shaded leaves) the conductance is `gs_min`,
   but returned through `stop_gradient`, so without its dependence on `gs_min`. The
   water-stress step then takes `max(gspot f(psi), gs_min)` with both arguments equal, of
   which JAX gives each half the derivative: half survives.

### Prototype (scratch copy only: `$SCRATCH/clm-ml-jax/trees/grad_diag`)

1. the slope of the Obukhov residual by forward-mode AD at the solution (exact; one
   derivative evaluation instead of two residual evaluations);
2. `gs_min` returned with its gradient where there is no root.

Both change gradients only: **the forward values of the day are bit for bit the same.**

| Input | Before | With the two corrections |
|---|---|---|
| Parameters | 20 of 161 pairs within 0.1 %; worst 50 % | **161 of 161; worst 6e-5** |
| Forcing fields (without the calendar day) | 10 of 203; worst 20 % | 187 of 203; worst 1.9 % (air temperature only, steps 34, 35, 40) |
| State | 912 of 1 200; 21 beyond 5 % | 1 171 of 1 200; 2 beyond 5 % (dependences near the FD noise) |

### Left over, not errors of AD as far as seen

- **Rain and snow at zero:** AD gives 0, FD grows without bound as its step shrinks. The
  wetted fraction is (water / capacity)^(2/3): its slope at a dry canopy is infinite, so no
  finite sensitivity to rain exists at rain = 0. A day with rain is needed to check rain.
- **Air temperature** (the field shifted by a fraction at all steps): 0.1–2 % off at steps
  34, 35 and 40 after the corrections. Not diagnosed.
- **The calendar day** as an input is differentiated but is not a meaningful sensitivity.
- **Roughness length `z0m`** (an output only): its bisection returns a midpoint of the fixed
  bracket, so AD gives it no dependence on stability. Nothing in the model reads it.
- Not covered by this day: rain, snow, Medlyn and Ball-Berry (their `solve_ci` slope is a
  finite difference too, with a step of 1e-4), C4, other towers, padded layers beyond
  CHATS7's 4, sparse 1° patches (O7: the plant-area floor).

### Done (decision 119, maintainer 2026-11-14)

The two corrections are in `src/` as prototyped. Tests: unit tests of the Obukhov gradient
at six stabilities (to 1e-6; the 1 m slope was 12 % off at L = −8 m) and of the `gs_min`
gradient without a root (exactly 1; it was 0 before and 0.5 after the water-stress step);
`tests/test_gradient_accuracy.py` in the gradient tier (four parameters, four fluxes, the
four steps at sunrise, within 1e-3 of finite differences; fails on the old code by 5–50 %).

## Round 3: the cases a clear CHATS7 day does not cover (2026-11-14)

The check of round 2 (`gradcheck0.py`, parameters and forcing fields, 47 steps, 7 outputs)
on the code with decision 119, for variants of the CHATS7 patch. No NaN and reverse equal
to forward mode in every case (except the C4 case, below).

| Case | Parameters within 0.1 % of FD | Forcing fields | Verdict |
|---|---|---|---|
| CHATS7, FD steps down to 1e-8 (the air-temperature leftover of round 2) | | 217 of 217 | sound: FD needed smaller steps |
| Medlyn stomatal conductance | 168 of 168 | 204 of 217 | sound (the rest: air temperature, as above) |
| Ball-Berry | 112 of 112 (49 negligible) | 200 of 217 | sound |
| Coarse layers: 8 within and 6 above the canopy (36 of 50 slots padding) | 161 of 161 | 183 of 217 | sound; padding does not disturb the derivatives |
| Invented showers (rain 0.36 mm/h in 3 of every 12 steps, snow in 2) | 152 of 161 (worst 0.5 %) | 169 of 231 | sound where the canopy is wet or dry; while it dries out (steps 8-10) FD is erratic and approaches AD only at its smallest step |
| The patch as `c4_grass` | 110 of 133; **leaf angle index 75–99 % off at every step** | 189 of 210 | **a forward-model error: O77** |

- **Air temperature** (and, less, wind and the layer count's effect on them): at a few
  afternoon steps FD changes erratically with its step size and reaches AD only below a
  relative step of 1e-6 (0.3 mK). The model has corners in air temperature there (the limits
  of the roughness-sublayer functions switching within the step); AD gives the local slope.
- **Rain:** with rain falling, AD and FD agree (1e-4). At zero rain no finite sensitivity
  exists (round 2). As intercepted water runs out, the wetted fraction (water^(2/3)) makes
  the fluxes non-smooth: sensitivities at those steps are large and local.

### O77: `avmu` for a negative leaf angle index (forward model; science)

For the C4 grass the finite differences with respect to the leaf angle index do not
converge: they grow as the step shrinks. The forward model is a sawtooth in the leaf angle
index (diffuse shortwave ±0.1 %, GPP ±0.25 %, period 7e-6 in χ_l), which is rounding noise
amplified.

- **Cause:** `solar_radiation.py::canopy_optics` computes the two-stream average inverse
  diffuse optical depth `avmu = (1 − φ1/φ2 ln((φ1+φ2)/φ1)) / φ2` with φ2 replaced by
  `max(φ2, 1e-10)`, a guard against dividing by zero. φ2 = 0.877 (1 − 2 φ1) is **negative
  whenever χ_l < 0** (−0.281 for χ_l = −0.3), so for those PFTs the formula is evaluated at
  φ2 = 1e-10: the result is the limit 1/(2 φ1) = 0.757 instead of 1.077, computed as a
  difference of nearly equal numbers (hence the noise).
- **Fortran and CTSM** use φ2 as it is (`MLSolarRadiationMod.F90` line 239;
  `SurfaceAlbedoMod.F90` line 1446); the leaf angle index is kept away from 0 (|χ_l| ≥ 0.01)
  so φ2 is never 0. The guard is the translation's.
- **Who is affected:** the five PFTs with χ_l = −0.3 (`c3_arctic_grass`, `c3_non-arctic_grass`,
  `c4_grass`, `c3_crop`, `c3_irrigated`), i.e. grass and crop patches of global runs. Not
  CHATS7 (0.53) nor any tree or shrub (χ_l ≥ 0.01), so no validation against Fortran saw it.
- **Size** (prototype with a signed guard, scratch copy `trees/grad_avmu`; the CHATS7 patch
  run as `c4_grass`, one day): day-mean GPP 5.9 → 10.5 µmol m⁻² s⁻¹, latent heat 61 → 77
  W m⁻², sensible heat 18 → 27, ground heat flux 30 → 5; net radiation unchanged in the mean
  (up to 16 W m⁻² at a step). With the prototype all 158 parameter derivatives of the C4
  case are within 0.1 % of FD, and reverse equals forward mode to 7e-10 (1e-3 before).
- **Fixed (decision 120, maintainer 2026-11-14):** the guard keeps the sign of φ2 and only
  acts at |φ2| < 1e-10, which never occurs: the Fortran formula for every PFT. Bit for bit
  and all of `tests/regression` (216) pass without a new reference; unit tests for eight
  leaf angle indices. Not compared with a Fortran run of a grass patch (none exists).

## Round 4: real 1° patches (2026-11-14)

`jobs/gpu_probes/gradcheck_grid.py`: the check of round 2 for every patch of a box of
`configs/grid_1deg_2000`, 1 January 2000, 47 steps, 7 outputs per patch, the 27 parameters
(each scaled in all PFT rows together), on the code with decisions 119 and 120. FD steps
1e-4, 1e-5, 1e-6.

| Box | Patches | Result |
|---|---|---|
| US Great Plains, 35-37°N (winter; LAI 0.00-0.85; needleleaf, deciduous tree, shrub, C3 and C4 grass, crop) | 21 | no NaN; 2 289 of 2 323 pairs within 0.1 % of FD, the rest within 1.3 % (mostly `gs_min`, and a patch with LAI 0.00); 26 pairs "lost": the leaf angle index of the four needleleaf patches (below) |
| Brazilian cerrado, 10-12°S (wet season; LAI 1.6-4.4; tropical trees, crops, C4 grass) | 18 | 9 patches: within 0.5 % of FD; **9 patches: every parameter derivative NaN from step 17 on** (O79) |

- **Leaf angle index of needleleaf trees:** their χ_l is 0.01, exactly where CLM's rule
  "|χ_l| ≤ 0.01 becomes 0.01" starts, so AD gives no sensitivity (FD sees the half above).
  A property of the rule, not an error.
- Sparse canopies (LAI down to 0.00, the plant-area floor of decision 30) gave no NaN and
  no wrong derivative for the parameters.

### O79: NaN parameter gradients where leaf water potential is positive

- **What happens:** in 9 of the 18 cerrado patches the derivative of every output with
  respect to every parameter is NaN from step 17 (pre-dawn) to the end of the day, in
  forward and in reverse mode. The forward run is finite.
- **Cause:** the water-stress factor `fpsi = 1 / (1 + (psi_l/psi50)^shape_gs)`
  (`photosynthesis.py::leaf_photosynthesis`; Fortran `MLLeafPhotosynthesisMod.F90` line
  438). At that step dew forms (leaf latent heat −0.3 to −2.9 W m⁻²), the transpiration term
  is negative and leaf water potential rises above zero (up to +0.03 MPa in the step mean)
  in exactly those patches. `psi_l/psi50` is then negative. Its 40th power is still a number
  (40 is a whole number), but its derivative with respect to the exponent `shape_gs`
  contains ln(negative) = NaN; and a NaN partial derivative makes the gradient of every
  parameter NaN (NaN × 0 = NaN), which is how `nan_by_parameter.py` found it: only a
  derivative taken with respect to `shape_gs` alone, or to the whole parameter set, is NaN.
- **Also a forward risk:** a calibrated, non-whole `shape_gs` (39.5, say) would make the
  forward value NaN wherever psi_l > 0; Fortran's `**` with a real exponent and a negative
  base is undefined there as well. CTSM's own vulnerability curve has the same form without
  a guard.
- **Proposal (a modelling choice; accepted, see "Done" below):** no water stress at psi_l ≥ 0:
  `(max(psi_l/psi50, 0))^shape_gs`. For the default whole-number `shape_gs` the forward
  value does not change (`fpsi` is exactly 1 for |psi_l/psi50| < 0.4 either way: to be
  confirmed bit for bit), and the derivatives are 0 there instead of NaN.
- **Behind it, not proposed to change now:** that leaf water potential can exceed zero at
  all (dew entering the plant-hydraulics balance as negative transpiration,
  `plant_hydraulics.py::leaf_water_potential`, as Fortran).

### Done (decision 121, maintainer 2026-11-14)

`fpsi = 1 / (1 + max(psi_l/psi50, 0)^shape_gs)`. Fast tier 468 pass; bit for bit and noise
twin pass (forward values unchanged). The cerrado box again
(`probe_grad1/r5_cerrado.npz`, `log_round5.txt`): **no NaN in any of the 18 patches**.

- The nine patches that were NaN: three (tropical trees, C4 grass) within 0.3 % of FD
  everywhere; the six crop patches have pairs that the report calls "bad" or "fair". None
  is an error of AD as far as looked at:
  - *Negligible derivatives* (patches 3 and 12; water-stress and root parameters): AD is
    1e-10 of the output or less, and FD is rounding noise that grows as the step shrinks.
  - *A corner at the sunset step* (patches 7 and 8, step 41; `iota`, leaf dimension,
    emissivity, clumping, near-infrared optics): FD changes with its step. A scan in `iota`
    (`probe_grad1/scan_r5.py`): the secants for relative shifts from −1e-5 to +3e-7 are
    2.94-3.04 and AD is 2.959; from +1e-6 up they jump to 6-9. The model has a corner
    within 1e-6 of the default value at that step, and AD is the slope on the side where
    the default lies. Which line of the model makes the corner was not looked for.
- A sensitivity study should know this: at sunrise and sunset steps a derivative can be
  exact and still say little about a finite change of the parameter.

### O78: XLA aborts while writing the compilation cache in the gradient tier

After decision 120 the CPU gradient tier aborted three times ("Fatal Python error:
Aborted" in JAX's `compilation_cache.put_executable_and_time`, inside
`executable.serialize()`), each time in `tests/test_gradient_accuracy.py` when it compiled a
new chunk in the pytest process that had already run `tests/test_differentiability.py`.
The same test alone passes, and so does the tier once the program is in the cache. Not
diagnosed. `tests/gradient_tier.py` and `jobs/grad_gpu.pbs` now run each test file in its
own process: 13 pass with an empty compilation cache.

## Round 5: jumps, kinks and round-off amplification (2026-11-15). No change to the model

Question (maintainer): are the corners at sunrise and sunset steps jumps (the output itself
steps, which AD cannot see) or kinks (only the slope changes), and what do they do to the
gradient of a daily mean? Probes and results: `$SCRATCH/clm-ml-jax/probe_jump/`
(`scan_lin.py`: a dense linear scan of a day's outputs in one parameter; `find_jumps.py`;
`refine.py`: 40 halvings of an interval towards the larger change; `obu_trace.py`,
`obu_roots.py`: every Obukhov solve of a day, by a host callback in the probe only). Only
`iota` was scanned (CHATS7 ±2 %, 801 points; cerrado box ±0.2 %, 401 points, and ±5 %, 201).

### Not jumps

- **The sunset corner of round 4 is a kink** (cerrado patch 7, step 41): latent heat is
  continuous, its slope changes from about +125 to −39.5 at a relative shift of −3.5e-4 in
  `iota`.
- **Two rough features refined to an interval of 1e-15 are continuous** (patch 3, steps 40
  and 41): the change shrinks with the interval, down to a floor that is round-off.
- **Jumps exist but are negligible:** isolated steps of at most ~1e-3 W m-2 in every scan
  (CHATS7 too); not traced (the stomatal bisection is the likely source).

### Round-off is amplified ~1e11 at the evening transition of short canopies

Two runs of the cerrado box whose `iota` differs by 4e-15 (relative), patch 3 (a crop,
canopy height 0.5 m, LAI 3.15):

| Steps | Largest relative difference of any field |
|---|---|
| 0-16 (night) | 0 |
| 17-39 (day) | 1e-12 to 1e-10 |
| 40 | 3e-7 |
| 41 | 3e-3 (latent heat), 6e-5 (friction velocity) |
| 42-46 | 1e-5, falling (GPP, near zero, order 1) |

Inside that noise the response to `iota` is rough at every scale: slopes of 1 300-1 800
W m-2 per unit relative change over intervals of 1e-6 (AD at the default: −290; the daily
mean: about −10), and latent heat of step 40 wandering between 31.5 and 39.3 W m-2 over ±5 %.
This is the amplification known from the 1° grid tests (1e-15 in LAI, ~8 W m-2 after
sunrise). CHATS7 does not have it (noise twins within 4e-11).

**What happens in those steps** (the 30 Obukhov solves of each step, patch 3):

- The solver is not at fault: all 25 380 solves of the day (18 patches) end with a
  residual below 1e-6 m (the ones listed: below 1e-12), and the residual has one root, or none where the answer is at
  the stable limit.
- Through the afternoon the canopy top is weakly unstable (L = −2 to −25 m, u* 0.04-0.05
  m/s; the canopy-top air is slightly colder than the forcing, humidity makes it buoyant).
- From the 24th solve of step 40 on, successive solves alternate between that state and
  the stable limit: u* 0.046, 0.0008, 0.040, 0.0008, 0.057, 0.0097, ..., with the
  canopy-top air temperature moving 0.3-0.6 K from one solve to the next. By step 42 the
  patch stays at the limit.
- The stable limit is L >= Lc (`LcL_max = 1`). For this patch Lc = 0.62 m and the forcing
  height is 30.4 m, 49 Lc above the ground: at the limit u* is 0.0008 m/s for a wind of 0.5
  m/s. Between neutral and the limit, u* falls by a factor 50 over ~0.1 K of canopy-top
  air temperature. For the forest patches the forcing height is 2-3 Lc, for CHATS7 1.6.
- So the turbulence is computed from the state before the solve, switches the canopy
  between coupled and decoupled, and the state answers by crossing the switch again: an
  alternation whose timing depends on the last digits. Not shown that this alone makes
  the amplification, but the alternation and the growth are in the same two steps.

Alternations (u* changing by more than 3× between successive solves) per patch: none to 16
for the forests (canopy 10-20 m), 4 to 55 for the 0.5 m crops and grasses, at steps 19-21
and 39-41. More than half of all solves of every patch (the night) are at the stable limit.

### What it does to the gradient of a daily mean (`iota`)

| | AD against the response to ±1 % |
|---|---|
| CHATS7: latent heat, sensible heat, GPP | within 0.3-0.8 % (±2 %: the same) |
| Cerrado, GPP, 18 patches | within ~1 % |
| Cerrado, latent heat, 14 patches | within ~1 % |
| Cerrado, latent heat, patch 3 | AD −11.9, response −8.5 (±5 %: −5.9) |
| Cerrado, latent heat, patches 7 and 8 | AD −6.6, response −5.9 |
| Cerrado, latent heat, patch 12 | AD −5.3, response −5.5 |

The gradient of a mean is the mean of the step gradients, so one step with an extreme
local slope (−290 at patch 3, step 40) moves the daily value. A sensitivity study on short
canopies under a high forcing level should compare AD with a finite change of the size it
cares about, or leave the transition steps out.

### Other places where the output can step (read from the code; none changed)

| Where | What | Seen |
|---|---|---|
| `photosynthesis.py::solve_gs_bisection` | no sign change because opening always pays (both ends positive): `gs_min` is returned, next to solutions near the upper end 2 mol m-2 s-1. As Fortran | not in the cerrado day (largest conductance 0.84) |
| `water_vapor.py` (latent heat) | vaporization above 0 °C, sublimation below: a 13 % step in temperature | not tested (warm cases) |
| `soil.py` (thermal conductivity) | frozen and unfrozen Kersten number differ at 0 °C | not tested |
| `water_vapor.py::sat_vap` | water and ice polynomials meet at 0 °C | not tested; expected tiny |
| `plant_hydraulics.py` (`psis`) | the minimum potential when no layer takes up water | not tested; rare |
| `solar_radiation.py` (leaf angle index) | values within ±0.01 become +0.01: a step at −0.01 | negligible |

Everything else read is a kink (limits, minima and maxima, the drip cap, the wetted
fraction, the stability limits of L).

### Open (maintainer to decide)

- The stable limit for short canopies under a high forcing level: L >= Lc allows a height
  of 49 Lc, and u* of 0.0008 m/s. Friction velocity is below 0.0024 m/s in a quarter of
  the cerrado patch-steps. A science question (CLM limits z/L instead: `zetamaxstable`).
- The alternation at the transitions: whether the lagged turbulence should be damped or
  iterated. Also a science and numerics question; nothing tried.
- Scans in other parameters, and the freezing-point steps on the Great Plains winter box.

## Round 6: a stable limit on zeta per PFT, prototype (2026-11-15). No change to the model

Maintainer's idea for the first open point of round 5: CLM's `zetamaxstable` as a PFT
parameter. Prototype in a scratch copy only (`$SCRATCH/clm-ml-jax/trees/zeta`): in
`turbulence.py::obukhov_terms`, after the limit L >= Lc, in stable conditions also
`L >= (zref - zdisp) / zeta_max`, with the displacement height of the Lc-limited length
(CLM: `zeta = zldis/obu <= zetamaxstable`, 0.5 in CTSM 5.4, `FrictionVelocityMod.F90`);
beta and the displacement height are then recomputed. The value is a per-PFT input (it
rides in an unused parameter in the prototype), so one compiled program runs every value.
Probes and results: `$SCRATCH/clm-ml-jax/probe_zeta/` (`eval_zeta.py`, `report.py`,
`scan_lin_zeta.py`, `gradcheck_grid_zeta.py`). "Short": grasses, crops, shrubs.

### It removes the alternation of round 5, if it is tight

Cerrado box, 1 January (47 steps, 18 patches); AD against a ±1 % change of `iota`:

| Limit | Alternations | Smallest u* (m/s) | Daily-mean LE, AD off by (worst patch) |
|---|---|---|---|
| off (the repository) | 307 | 0.0008 | 39 % |
| 0.5, all PFTs | 1 | 0.022 | 0.5 % |
| 1, all | 91 | 0.017 | 1.4 % |
| 2, all | 305 | 0.012 | 12 % |
| 5, all | 414 | 0.006 | 52 % |
| 10, all | 354 | 0.004 | 358 % |
| 0.5 short, trees off | 53 | 0.017 | 1.2 % |

- Only values up to about 1 help; 2 and more leave the alternation. A value calibrated
  upwards brings the roughness back.
- Dense scan of `iota` (±5 %, 201 points) with 0.5 for the short PFTs: the worst step's
  departure of latent heat from a smooth curve falls from 1.33 to 0.33 W m-2 (patch 3),
  0.18 to 0.002 (patches 7, 8), 0.70 to 0.07 (patch 12); the daily-mean slope of AD
  against the fitted one from +92 %, +26 %, −8 % off to −2.9 %, +0.2 %, −2.6 %. Trees
  are unchanged (one tree patch is 3 % off with or without).
- All 27 parameters against FD with 0.5 for the short PFTs: no NaN; 16 of 18 patches have
  no "bad" pair (worst 5e-3); two crop patches (16, 17) keep 12 "bad" pairs each (not
  looked at).
- The limit's own derivative: no NaN; AD agrees with FD at a relative step of 1e-4 (within
  1e-6 to 3e-3 of the largest sensitivity for 0.5; cerrado, plains, CHATS7).
- With the limit off the prototype is bit for bit the repository code (CHATS7 day).

### What it changes in the forward model (daily means, change from "off")

| Case | Limit | LE (W m-2) | H (W m-2) | Canopy-top air, night (K) |
|---|---|---|---|---|
| CHATS7 (largest zeta of the day 0.95) | 1 and more, or trees off | 0 | 0 | 0 |
| CHATS7 | 0.5 | +0.8 (a step: 17) | −1.2 | 0 |
| Cerrado, January | 0.5 short | +1.1 (patch: +3.4) | −1.1 (−2.8) | +0.5 |
| Cerrado, July | 0.5 short | +2.6 (+7.4) | −3.4 (−7.8) | +0.4 |
| Great Plains, January | 0.5 short | +2.3 (+5.0) | −20 (−35) | +3.7 |
| Great Plains, July | 0.5 short | +5.6 (+11.9) | −9.7 (−19) | +1.8 |

Night friction velocity of the short canopies rises from 0.002 to ~0.03 m/s (cerrado) and
from 0.07 to 0.30 m/s (Great Plains, January). The Great Plains January box had no
gradient problem (2 alternations): there the limit is purely a change of the science.

### It is not the whole problem: derivatives explode on the July days (new)

With and without the limit:

- **Great Plains, 1 July:** from step 27 (mid-morning) the derivative of latent heat with
  respect to `iota` grows 10 to 1000 times per step, to 1e35 at the end of the day, on 15
  of 21 patches (trees included). It is real: two runs whose `iota` differs by 1e-14 are
  5-11 W m-2 apart in the afternoon. The ±1 % finite differences of the daily means stay
  plausible (−35 to −5 W m-2). The day is windy (u* 0.5-1.0 m/s) and near neutral; the
  Obukhov solves of the five stages of a sub-step differ strongly (u* 0.74, 0.69, 0.73,
  0.63, 0.68 within one sub-step), also on patches that do not explode. This is very likely the
  amplification the 1° grid tests know (1e-15 in LAI, ~8 W m-2 after sunrise), and not the
  evening alternation of round 5. Not diagnosed.
- **Cerrado, 1 July:** 2-3 patches with derivatives of 1e4 to 1e7 in the afternoon; the
  others agree with FD where the sensitivity is not small.
- A single pair of runs 1e-14 apart at the default parameters is not a measure of the
  amplification (the cerrado January pair stays within 5e-9 W m-2 with the limit off): use
  a scan, or the growth of AD through the day.

### Done (decision 123, maintainer 2026-11-15)

`zeta_max_stable` is a PFT parameter (params.yaml; `turbulence.py::obukhov_terms`): 0.5 for
shrubs, grasses and crops, 1e30 (no limit) for trees in the three parameter compsets;
`clm-ml upgrade` adds it to older params files. CHATS7 is bit for bit unchanged (the
reference was not recorded again). The repository code differs from the prototype runs
above only in the first step of a run, which the prototype's probe ran without the limit
(the difference decays through the day). With decision 122, one day, daily-mean latent heat,
AD against ±1 % of `iota`, worst patch: cerrado 1 January 2.3 %, cerrado 1 July 1.6 %,
Great Plains 1 July 0.7 %, 1 January 0.0 %; no derivative above 1e3. Forward changes: as
the table above (decision 123 has the numbers with decision 122). New gradient-tier test
`tests/test_gradient_accuracy_grid.py` (a day of each box). All of `tests/regression` and
the gradient tier pass.

### Open at the time (decided: see "Done" above)

- Whether to add the parameter, and its values: 0.5 for the short PFTs and no limit for
  trees keeps CHATS7 and removes the alternation; 0.5 for all is CLM but changes CHATS7.
  The definition of zeta (with `zref - zdisp`, as above) is part of the decision.
- The exploding derivatives of windy days (above): to be diagnosed first; they matter more
  for gradients than the alternation.

## Round 7: the exploding derivatives are the water on the leaves (2026-11-15). No change to the model

Probes and results: `$SCRATCH/clm-ml-jax/probe_expl/` (`cfg.py`: a day or five with any
setting, a twin, AD and ±1 % of `iota`; `eig.py`: the growth of a small difference of the
state over one step, by power iteration on the step's Jacobian; `dump_h2o.py`). Prototype:
`$SCRATCH/clm-ml-jax/trees/wet` (switch `CLM_WET=implicit`; off, CHATS7 is bit for bit the
repository code) and `trees/both` (with round 6's zeta limit).

### Diagnosis

- **It is not the time stepping as such.** Great Plains, 1 July: Euler, the midpoint rule
  and 60 s sub-steps all explode (1e22, 1e22, 1e154): the growth is per sub-step, whatever
  its length.
- **Each daytime step multiplies a small difference of the state by 100 to 36 000**
  (`eig.py`; 1.0 at night). Restricted to one variable at a time, only the **intercepted
  water** grows (the same factors alone; everything else without it: damped).
- **With leaves that never get wet** (probe: maximum wetted fraction 0) nothing explodes:
  twins 1e-14 apart stay within 3e-10 W m-2 (11 W m-2 before), the largest derivative is
  190 (1e35 before). Water stress off changes nothing.
- **The mechanism.** The wetted fraction is `(water / capacity)^0.67`, capped at 0.05, and
  is computed from the water *before* the stage (Fortran the same:
  `MLCanopyWaterMod::CanopyWettedFraction` in each Runge-Kutta stage; no floor at zero,
  commented out). Below 1.1 % of the capacity the cap no longer holds and one 5-minute
  sub-step would evaporate more water than there is: the water overshoots below zero, the
  next stage sees a dry leaf and evaporates nothing, and so on. The slope of
  `x^0.67` grows without bound as the water goes to zero. The canopy never quite dries
  (0.01-0.02 kg m-2 stay through the afternoon), layers reach −0.003 kg m-2, and latent
  heat flickers by several W m-2.
- **Why all day.** The day starts with rain (0.35 mm per step until dawn), then the
  6-hourly forcing gives a trace of drizzle (0.0001-0.015 mm per step) for the rest of the
  day: the leaves are always almost dry and never dry. Dew after sunrise does the same:
  this is the round-off amplification the 1° regression tests know (CLAUDE.md: 1e-15 in
  LAI, ~8 W m-2 after sunrise).

### Prototype: the wetted fraction of the water left at the end of the sub-step

Backward Euler for the intercepted water, as the temperatures (implicit) and the leaf
water potential (analytic) already are: the wetted fraction `w` solves

    dewmx dpai w^(1/0.67) + dt E_wet w = water at the start + interception,   w <= 0.05

with `E_wet` the evaporation of a fully wet layer from the stage's leaf temperature and
vapour pressure (12 Newton steps from above, monotone; gradient by the implicit function,
decision 119). The water budget is unchanged (`evaporation` as it is). The leaves cannot
lose more than they hold, and the slope of the result is bounded (between 0 and 1).
CLM does it differently (`CanopyFluxesMod`: `rpp = min(fwet, (h2ocan/dtime)/efpot)`, and
the excess moved to sensible heat afterwards). Not prototyped. (Later note: applied before
the flux solve it would probably also stop the overshoot, with a new corner; its two
branches are the limits the implicit solution joins smoothly.)

| Great Plains, 1 July | repository | prototype |
|---|---|---|
| patches with a derivative above 1e4 | 15 of 21 (1e35) | 0 (190) |
| twins 1e-14 apart, latent heat | 11 W m-2 | 4e-10 W m-2 |
| daily-mean latent heat, AD against ±1 % of `iota` | meaningless | within 0.7 % |
| lowest water of a layer | −3e-3 kg m-2 | −2e-9 kg m-2 |
| mean latent / sensible heat | 86.2 / 24.3 | 84.0 / 25.7 W m-2 |
| the same with 60 s sub-steps; Euler | 85.6; 84.3 (explode) | 84.0; 84.2 (0.3 %; 0.1 %) |

- Forward change of the daily means: Great Plains July −2.2 W m-2 latent heat (worst
  patch −7.1), cerrado July −0.35 (−1.3), cerrado January 0.00 (the leaves stay at the
  cap all day), Great Plains January 0.00 (0.06). CHATS7 (dry leaves): 3e-10 W m-2.
- All 27 parameters, Great Plains July, both prototypes: no NaN; every pair that
  disagrees with finite differences is a sensitivity of ~1e-6 W m-2 (root and capacitance
  parameters), where the finite difference is round-off.

### The two prototypes together (daily-mean latent heat, AD against ±1 % of `iota`, worst patch)

| | repository | zeta limit only | wet leaves only | both |
|---|---|---|---|---|
| Cerrado, 1 January | 40 % | 1.2 % | 40 % | 1.2 % |
| Cerrado, 1 July | 1e5 % | 5e6 % | 123 % | 3.9 % |
| Great Plains, 1 July | 6e34 % | (explodes) | 0.7 % | 0.7 % |
| Great Plains, 1 January | 0.0 % | | 0.0 % | 0.0 % |

Each is needed; they are the same kind of defect in two places: a quantity the fluxes
react to strongly (the water on the leaves; the Obukhov length) is taken from before the
solve instead of being part of it.

### What is left (five days, both prototypes)

- **Great Plains, 1-5 July:** no explosion (repository: 1e220, twins 39 W m-2 apart, all
  21 patches), but crops and grasses are 3-15 % off at midday of days 4-5. With water
  stress off: 0.2 %. A third lagged feedback: stomatal conductance is reduced by the leaf
  water potential of before the stage with a very steep function (`shape_gs` 40). Not
  prototyped; the same treatment would apply (the leaf water potential solved together
  with its stress).
- **Cerrado, 1-5 January:** on the evening of day 4 three patches have a local slope of
  latent heat of −23 000 where ±1 % gives −164. AD is right there (a finite difference
  with a 1e-6 step gives −22 804 against −22 805): the response itself has a narrow steep
  place at that evening transition. Not diagnosed.

### Done (decision 122, maintainer 2026-11-15)

The implicit wetted fraction is the default (`canopy_water.py::wetted_fraction_implicit`,
`wet_leaf_demand`; `numerics.wetted_fraction: explicit` is Fortran's form and bit for bit
the code before, used by the Fortran comparison compset). The repository code gives the
prototype's numbers on the test days (Great Plains 1 July: 83.98 W m-2, no exploding patch,
twins 3e-10 W m-2, AD within 0.7 % of ±1 %; cerrado 1 July 123 % and 1 January 40 % as the
prototype without the zeta limit). The regression reference was recorded again (CHATS7
changes by 3e-10 W m-2, dew). All of `tests/regression` and the gradient tier pass; the
forward run is about 7-12 % slower on a CHATS7 day (CPU), the gradient 4-12 %.

What it does not change in earlier fixes: leaf water potential above zero under dew is the
same (+0.031 MPa on the cerrado day, 7 of 18 patches, with either form: dew enters through
the transpiration term, not the wetted fraction), so decision 121 is still needed. The tiny
floor in the explicit `wetted_fraction` (against the infinite slope of 0^0.67) is not
needed by the implicit form, which gives a dry layer a zero gradient.

### Open (maintainer to decide)

- The zeta limit of round 6 (unchanged proposal).
- Whether the 1° regression tests can now compare daytime outputs too (not tried: the
  evening alternation of round 5 is still there without the zeta limit).
- Whether to prototype the leaf water potential the same way, and to look at the steep
  evening of the cerrado.

## Round 8: thirty days (2026-11-15). No change to the model

With decisions 122 and 123 in the code. `$SCRATCH/clm-ml-jax/probe_expl/` (`run_30d.sh`,
`run_30d_b.sh`, `cfg.py` with `NSTEP=1439`; results `repo2_*_30d*.npz`): 30 days from the
1st, forward-mode derivative of latent heat with respect to `iota` against ±1 %; a
"patch-day" is one patch's daily mean, "off" more than 5 % from the finite change. One run
takes 8-12 minutes on a login node; two at once were killed (memory).

| Box, 30 days | Patches with the 30-day mean within 1 % | Patches with a derivative above 1e4 | Patch-days off | Twins (1e-14) apart at most |
|---|---|---|---|---|
| Great Plains, July | 12 of 21 (worst: 37 times too large) | 5 | 97 of 609 | 1e-5 W m-2 |
| the same, water stress off | 19 of 21 (the same worst patch) | 2 | 24 of 609 | 1e-5 |
| Cerrado, January | 3 of 18 (8 within 5 %; worst 5e4 times) | 7 | 211 of 522 | 4e-3 |

- **The first three days are clean in both boxes** (no patch-day off); the first bad ones
  are on day 4. The patches that stay good agree to three or four digits over the 30 days.
- **Water stress (the third lagged feedback of round 7) is confirmed:** with
  `water_stress: false` 19 of 21 Great Plains patches are within 0.5 % over 30 days and
  nothing is off before day 13. Stomata take the leaf water potential of the stage's
  start through `1/(1 + (psi/psi50)^40)`; once a patch reaches stress, single steps have
  derivatives of 1e4 to 1e7. Not prototyped: the same idea as decision 122 (the leaf
  water potential the stress uses solved with the transpiration of the stage).
- **Something else is left:** two Great Plains patches from day 13 without water stress
  (one 37 times off in the 30-day mean, from a few steps around step 601-622), and the
  cerrado, where 10 of 18 patches are more than 5 % off (from the steep evening of day 4,
  round 7). Not diagnosed; whether the cerrado is water stress too is not tested.
- The forward model is now reproducible over 30 days (twins 1e-5 to 4e-3 W m-2 apart; before
  decision 122: 39 W m-2 after five days), so **finite differences of means are sound**
  where AD is not yet.

### Open (maintainer to decide)

- Prototype the leaf water potential treatment (scratch first, as rounds 6-7).
- Diagnose the two Great Plains patches and the cerrado (power iteration by state block,
  as round 7).
- After those: state and forcing derivatives on 1° patches, other seasons and regions, and
  a longer window in the gradient tier (a 30-day test costs ~10 minutes).

## Round 9: water stress from the leaf water potential of the end of the sub-step, prototype (2026-11-15). No change to the model

Prototype: `$SCRATCH/clm-ml-jax/trees/lwp` (commit 3f9e0f9 plus `lwp_prototype.diff`;
switch `CLM_LWP=implicit`, off: the repository code). Probes and results:
`$SCRATCH/clm-ml-jax/probe_lwp/` (`cfg.py` as round 7, `cmp.py`, `run5.sh`, `run30.sh`,
`newton_test.py`).

### The defect and the prototype

In each stage the stomata take the stress `f = 1 / (1 + (psi / psi50)^shape)` of the leaf
water potential at the start of the stage, transpiration follows from them, and the
potential is then updated from that transpiration (Fortran the same: `LeafPhotosynthesis`
before `LeafWaterPotential`). Near the threshold the function is very steep (`shape_gs`
40): a leaf that looks wet opens, transpires and falls below the threshold; in the next
stage it looks dry and closes. The derivative of one step with respect to the potential
is then far above 1 (round 8: 1e4 to 1e7).

The prototype solves the potential the stress uses together with its transpiration, per
leaf (as decision 122 for the water on the leaves):

    psi = psi_bef + (psis - HEAD zs - 1000 tr(psi) / lsc - psi_bef) (1 - exp(-dt lsc / capac))
    tr(psi) = fdry gleaf(max(gspot f(psi), gs_min)) max(esat(tleaf) - eair, 0) / pref

`plant_hydraulics.leaf_water_potential` with the transpiration of
`flux_profile.leaf_fluxes` at the stage's leaf temperature and vapour pressure. The left
side minus the right rises with `psi` with a slope of at least 1: exactly one root,
between the potentials of the unstressed and of the minimum conductance. 50 halvings of
that interval; the gradient by the implicit function (decision 119), its slope bounded.
The potential carried in the state is still updated after the flux solution from the
transpiration it gives: the water budget is unchanged. Unstressed leaves (`f` = 1) are
not changed.

### Results (latent heat, AD against ±1 % of `iota`)

| | repository | prototype | repository, water stress off |
|---|---|---|---|
| Great Plains, 1-5 July: patches with the mean within 1 % | 16 of 21 (worst 15.8 %) | 21 of 21 (worst 0.15 %) | |
| Great Plains, 30 days: patches within 1 % | 12 of 21 | 18 of 21 | 19 of 21 |
| the same: patch-days off | 97 of 609 | 27 of 609 | 24 of 609 |
| the same: first day with a patch off | 4 | 13 | 13 |
| the same: patches with a derivative above 1e4 | 5 | 2 | 2 |
| the same: twins (1e-14) apart at most | 1e-5 W m-2 | 3.5e-6 | 1e-5 |
| Cerrado, January, 30 days: patch-days off | 211 of 522 | 211 of 522 | |
| CHATS7, one day | 0.3 % | 0.3 % | |

- **The water-stress part is removed:** with the prototype the Great Plains are where
  they are with water stress switched off. Left: the two patches of round 8 from day 13
  (37 and 2 times too large in the 30-day mean) and one patch 2.3 % off; the other 18
  within 0.3 %.
- **The cerrado is not water stress:** the prototype changes its forward run by 3e-6
  W m-2 and nothing in the gradients; the prototype with water stress off gives the same
  211 patch-days.
- **Forward change** (a science change): Great Plains 30 days −0.10 W m-2 latent heat
  (worst patch −0.58; single steps up to 29 W m-2 where the stress sets in), five days
  −0.04 (−0.30); CHATS7 −0.013 W m-2 (largest step 0.5).

### Cost (`timing/time_model.py --steps 47`, CHATS7, CPU, two runs each)

| | switch off | prototype |
|---|---|---|
| forward, per step | 9.9-10.4 ms | 14.5-14.8 ms (1.45 times) |
| gradient, per step | 52-57 ms | 91-97 ms (1.75 times) |

The cost is the 50 evaluations of the power `(psi / psi50)^shape` per leaf and stage.
Fewer evaluations were tried on random leaves only (`newton_test.py`, not in the model):
with the two corners taken out exactly (unstressed; minimum conductance), 8 halvings and
8 Newton steps are within 1e-13 MPa for all but ~5 of 400 000 leaves (those within 1e-7),
which would be about a third of the evaluations; Newton alone crawls (the function is
almost a step). A solver that does not reach round-off on every leaf is a discrete
defect of the kind the noise-twin test finds, so this is not a finished alternative.

### Open (maintainer to decide)

- Whether the implicit stress potential goes into the code (a science change; a setting
  with Fortran's form as the other choice, as decision 122), and at which cost: as
  written (1.45 times forward), or after work on a cheaper solver that still reaches
  round-off.
- The diagnosis of what is left (unchanged from round 8): the two Great Plains patches
  from day 13, and the cerrado (now known not to be water stress).

## Round 10: a derivative-only treatment of the water stress, prototype (2026-11-15). Documented, not adopted

The maintainer's question after round 9: is there a patch for the gradients that does not
touch the forward model? Prototype: `$SCRATCH/clm-ml-jax/trees/lwp_grad` (commit 3f9e0f9
plus `probe_lwp/lwp_damped.diff`; switch `CLM_LWP=damped`). Results in
`$SCRATCH/clm-ml-jax/probe_lwp/` (`run5d.sh`, `run_damped2.sh`, `*_damped*.npz`).
**Not adopted (maintainer, 2026-11-15: "not convinced this is the right solution").**

### What it is

The exploding derivative of round 8 is, as far as diagnosed, the true local slope of the
forward model, which overshoots at the stress threshold. A derivative-only patch therefore
computes the slope of a smoother model than the one that runs: an approximation by design.

- `_stress_residual(psi, ...)`: `psi` minus the leaf water potential at the end of the
  sub-step that the stress of `psi` gives (round 9's equation; 0 where `psi` is consistent
  with its own transpiration).
- `stress_potential_damped(lwp, ...)` (`jax.custom_jvp`): returns `lwp`, the potential of
  the stage's start, unchanged (the forward program is the repository's). Its derivative
  rule is the implicit function of the residual at `lwp`:
  `d psi = -(d residual at fixed psi) / max(d residual / d psi, 1)`. The derivative with
  respect to the stage's potential (the link that grows) is dropped; the one with respect
  to `lwp_bef` and the other inputs is divided by a slope that is large where the stress
  function is steep. No solver.
- `leaf_photosynthesis` passes the potential through it before the stress line.

A cut of the link (the stress factor as a constant in the derivative) was not built: it
drops how a parameter changes the stress itself (maintainer: unacceptable).

### Results (Great Plains July, latent heat, AD against ±1 % of `iota`)

| | repository | derivative only (this round) | implicit forward (round 9) |
|---|---|---|---|
| forward results | reference | identical, bit for bit (5 and 30 days) | changed |
| five days: patches with the mean within 1 % | 16 of 21 | 18 of 21 | 21 of 21 |
| five days: worst patch | 15.8 % | 4.1 % | 0.15 % |
| five days: largest derivative of a step | 1300 | 270 | 270 |
| 30 days: patch-days off | 97 of 609 | 80 of 609 | 27 of 609 |
| 30 days, the 19 patches without round 8's two: worst 30-day mean | 361 % | 6.2 % | 2.3 % |
| the same: patches more than 1 % / 5 % off | 7 / 5 | 5 / 1 | 1 / 0 |
| forward, per step (CHATS7, CPU) | 10.1 ms | 10.0-10.4 ms | 14.5-14.8 ms |
| gradient, per step | 52 ms | 53-58 ms | 91-97 ms |

- Means over 30 days come from "meaningless on the stressed patches" (35-360 %) to
  within 0.1-6 %; daily means are often still more than 5 % off.
- **Part of what is left is the reference, not the prototype:** with the forward model
  unchanged it still overshoots at the threshold, and its response to a finite change
  depends on the size of the change. Five days, ±0.3 % / ±1 % / ±3 % of `iota`: −27.96 /
  −29.12 / −29.02 and −12.46 / −13.16 / −13.40 W m-2 on two stressed patches (4-7 %); the
  prototype gives −27.93 and −13.03. The other part is the approximation (the slope is
  taken at the stage's potential, not at the consistent one).
- CHATS7: forward identical, gradient check unchanged (0.3 %).

### What it does not do

The forward overshoot stays (single steps differ by up to 29 W m-2 from the consistent
form, round 9), and a finite difference with a very small step at a stressed step
disagrees with this derivative by design.

## Round 11: the rest is the Obukhov length taken from the stage's start; a derivative-only prototype (2026-11-16). Documented, not adopted

Round 8 left two things undiagnosed: two Great Plains patches from day 13 ("issue 2") and
the cerrado ("issue 3"). **They are one defect.** No change to the model; probes and results
in `$SCRATCH/clm-ml-jax/probe_rest/`, prototype in `$SCRATCH/clm-ml-jax/trees/obu` (commit
d73e558 plus `probe_rest/obu_damped.diff`; switch `CLM_OBU=damped`).

### Diagnosis

Each Runge-Kutta stage solves the Obukhov length L from the canopy-top air temperature and
humidity of the stage's *start*, then computes new canopy-top air with that L. The loop
"air -> L -> turbulence -> fluxes -> air" is therefore closed with a lag of one stage, like
the wetted fraction (round 7) and the water stress (round 8). Near neutral, in a
transition, a small change of the air changes L a lot, and the new air answers with the
opposite sign and a larger size: an overshoot that alternates from stage to stage.

Evidence (`eig.py`: growth of a small state difference over one step, by power iteration;
`obu_probe.py`: every Obukhov solve of a step recorded):

- **Great Plains.** The two patches are needleleaf evergreen temperate trees. Growth per
  step 2.9 at step 601 and 2600 at step 602 (the other one: 2.4, 9, 26, 33, 3.1 at steps
  619-623). Only the canopy air and leaf temperatures amplify; not the soil, the
  intercepted water or the leaf water potential.
- **Switching off the derivative of the Obukhov length alone** (`stop_gradient` on
  `turbulence.obukhov_length`) gives growth 1.0 at every one of these steps, in both boxes.
- **The record of one bad step:** L goes 110, 117, 111, 132, 116, 194, 142, 256, 74, 256,
  100, 530, 63, -31 m from stage to stage and then flips between the stable limit
  (Lc/L = 1, u* 0.06 m s-1) and the unstable limit (Lc/L = -2, u* 0.24). A difference of
  1e-6 K in the air changes sign at every stage and grows to 7500 times its size.
- **It is not the solver:** at every call the equation for L has one root and an ordinary
  slope (no fold, no second solution). The gain of the lagged loop is about -2 to -3 per
  stage; anything below -1 grows.
- **Cerrado.** The bad steps are in the local afternoon and evening (18-21 UTC). Growth per
  step 8850 (a C3 crop), 85 (a tree), 35 (C4 grass); 1 with the Obukhov derivative off.
  Short PFTs do it too: the stable zeta limit of decision 123 removes the stable-night
  alternation (round 6), not this one.
- The forward model really does this (the flips are in the forward values), so the large
  derivative is again the true local slope of a model that overshoots.

### Prototype: the derivative of L damped by the loop's gain

Forward values unchanged (checked bit for bit against the repository, one cerrado day,
switch off and on). Only the derivative of L is replaced, in `canopy.py::_stage`:

1. the stage's solve once with L's derivative cut: the new canopy-top air and its
   derivative without L;
2. the response of that air to L (one forward-mode pass), times the response of L to the
   air: the **gain** of the loop;
3. L's derivative is taken as if L had been solved with the new air, divided by
   (1 - gain): the slope of the consistent solution instead of the lagged one. Applied
   only where gain < -G; elsewhere the repository's derivative.

Three forms were compared on five cerrado days (reference: ±1 % of `iota`; daily means of
latent heat, 4 complete days x 18 patches):

| | largest step derivative | patch-days > 5 % off | worst day-4 patches (repository: 58, 4750, 242, 172 %) |
|---|---|---|---|
| repository | 2.3e4 | 6 | as given |
| every step, divisor at least 1 | 100 | 12 | 8, 30, 29, 30 % |
| every step, divisor at least 0.2 | 420 | 12 | 8, 30, 29, 29 % |
| only where gain < -0.8 | 100 | 11 | 8, 30, 29, 30 % |
| only where gain < -2 (used below) | 100 | 10 | similar |
| only where gain < -4 | 1400 | - | one patch back to 176 % |

### Thirty days (gain < -2)

| | repository | prototype |
|---|---|---|
| **Great Plains July**: the two patches, 30-day mean off by | 3764 % and 112 % | 0.1 % and 0.5 % |
| their days off | 17 and 5 | 0 and 2 |
| the other 19 patches | unchanged (what is off there is water stress, rounds 9-10) | |
| **Cerrado January**: largest step derivative | 1.6e9 | 1.7e3 |
| worst 30-day mean | 5.6e6 % | 11.7 % |
| patches with the 30-day mean within 1 % / 5 % | 3 / 10 of 18 | 5 / 14 of 18 |
| patch-days more than 5 % off | 211 of 522 | 170 of 522 |
| patch-days more than 1 % off | 279 | 333 |
| forward, per step (CHATS7, CPU), as prototyped | 9.9 ms | 46-47 ms |
| gradient, per step, as prototyped | 52-53 ms | 176-192 ms |
| memory | 3.3-3.6 GB | 8-9 GB |

What this says:

- **The explosions are gone**, and where the defect is rare (Great Plains) the result is
  good: 0.1 and 0.5 % over 30 days.
- **Where the defect is frequent (cerrado) the result is mediocre.** Monthly means are
  within 12 %, most within 5 %, but a third of the daily means are still more than 5 %
  off, and on days where the repository's derivative was right the prototype can be a
  few percent worse (up to 12 % on two crop patches in the five-day test).
- Why: where the forward model flips between the stable and the unstable limit from
  stage to stage, its mean responds to a parameter through *how often* it flips. Neither
  the true local slope nor the slope of a consistent solution describes that. The
  ±1 % reference may itself depend on the size of the change there (measured for water
  stress in round 10, not here).
- **The cost is the prototype's, not the method's.** The extra passes were written as
  ordinary code, so the forward run pays for them too (4.7x), and the gradient 3.5x. A
  production form would be a derivative rule (as round 10), leaving the forward run
  untouched; its gradient cost was not measured and would be about two extra derivative
  passes of the stage.
- Not run: both derivative-only prototypes (this one and round 10) together.

### The forward alternative (not prototyped)

As in rounds 7 and 9, the consistent form would solve L together with the canopy-top air
of the stage (an outer iteration around the flux-profile solve), which removes the flips
in the forward model. It changes forward results and costs several flux-profile solves
per stage. It is a science change and the maintainer's decision.

### Open (maintainer to decide)

- Whether any derivative-only treatment is wanted (rounds 10 and 11: bounded, approximate,
  forward unchanged), or the consistent forward forms (rounds 9 and this one's
  alternative: exact derivatives of a model that no longer overshoots, at a forward
  cost), or neither (use finite differences of means where AD is unreliable).
- If derivative-only: write the Obukhov one as a derivative rule and measure its cost.

## Round 12: the derivative of L without the canopy-top air of the stage's start (2026-11-16). Not used

The maintainer's question after round 11: approximate, linearized gradients are acceptable
if the forward model stays fast and the gradients are smooth and cheap. The cheapest such
form for the Obukhov length is a cut: L keeps its value and its derivative with respect to
its other inputs, but none through the canopy-top air (`taf`, `qaf`) of the stage's start,
the lagged link of round 11. Tree `$SCRATCH/clm-ml-jax/trees/obu_cut` (round 11's tree
plus a `stop_gradient` on `taf` and `qaf` in `harman_finnigan`'s Obukhov inputs; switch
`CLM_OBU=cut`; round 10's `CLM_LWP=damped` is in this tree too). Probes and results:
`$SCRATCH/clm-ml-jax/probe_cut/` (`run_cut.sh`, `cmp_modes.py`, `*_cut*.npz`). Forward
values unchanged.

Latent heat, AD against ±1 % of `iota`, 30 days:

| | repository | round 11 (switch at -2) | cut | cut + round 10 |
|---|---|---|---|---|
| **Cerrado January**: largest step derivative | 1.6e9 | 1.7e3 | 1.6e2 | 1.6e2 |
| worst 30-day mean | 5.6e6 % | 11.7 % | 20.5 % | 20.5 % |
| median 30-day mean error | 3.3 % | 2.0 % | 4.1 % | 4.1 % |
| patches within 1 % / 5 % (of 18) | 3 / 10 | 5 / 14 | 0 / 10 | 0 / 10 |
| patch-days more than 5 % off (of 522) | 211 | 170 | 350 | 350 |
| **Great Plains July**: largest step derivative | 8.9e6 | 1.9e5 | 1.8e5 | 3.1e2 |
| worst 30-day mean | 3764 % | 376 % | 388 % | 10.5 % |
| median 30-day mean error | 0.08 % | 0.09 % | 2.8 % | 2.7 % |
| patches within 1 % / 5 % (of 21) | 12 / 14 | 14 / 16 | 6 / 13 | 6 / 16 |
| patch-days more than 5 % off (of 609) | 97 | 76 | 177 | 182 |

- **The stability feedback is a real part of the sensitivity.** Without it the derivative
  is too large in magnitude on most patches, by 3-20 % in the monthly mean, on the days
  that were right before too (five cerrado days: 49 of 72 patch-days more than 5 % off;
  repository 6, round 11 10).
- It is bounded, smooth and cheap (a 30-day run in 4 minutes).
- `iota` reaches L only through the air, so this test does not show what keeping L's
  direct dependence on parameters and forcing is worth.
- The cut and round 10 together run without trouble (the first run of an Obukhov and a
  water-stress treatment together).

## Round 13: the Obukhov treatment as a derivative rule, blended by the loop's gain; with the water stress, thirty days (2026-11-16). Decision 124

Tree `$SCRATCH/clm-ml-jax/trees/obu_rule` (the round 12 tree plus the rule; its
difference from round 11's tree: `probe_cut/obu_rule.diff`; switch `CLM_OBU=rule`, form
`CLM_OBU_MODE=clip|switch|blend`). Probes and results in `probe_cut/` (`run_rule5.sh`,
`run_rule30.sh`, `*_rule*.npz`, timing lines "r13" of the tree's
`timing/results_model.jsonl`).

### The rule

`canopy.py::_consistent_obukhov_rule`: the stage from the turbulence to the flux-profile
solution as one function with a `jax.custom_jvp`. Its value is the repository's (L solved
from the stage's start, the stage evaluated once). Its derivative:

1. the stage with L held: the outputs' derivative without L's, among them the new
   canopy-top air's (`d air1`);
2. the stage's response to L (one more forward-mode pass): `d out / dL`, among it the new
   air's (`air_L`); with the solver's `L_air`, the **gain** of the loop, `L_air . air_L`;
3. two derivatives of L: the repository's, `dL_model = dL_direct + L_air . d air0` (the
   air of the stage's start), and the consistent one,
   `dL_cons = (dL_direct + L_air . d air1) / max(1 - gain, 1)`;
4. `dL = (1 - w) dL_model + w dL_cons` and `d out = (d out at fixed L) + (d out / dL) dL`.

The weight `w` is the form: `clip`: 1 everywhere; `switch`: 1 where gain < -2, else 0
(round 11); **`blend`: 0 for gain >= -1 (the lagged loop settles; the model's own
derivative is right), 1 for gain <= -2, linear in between** (continuous, no jump). The
rule is linear in the tangents, so reverse mode works. An output that does not depend on L
keeps its tangent (a constant output with a tangent that varies from patch to patch fails
under `vmap`).

Checks: forward bit for bit against the same tree with the switch off (CHATS7, one day;
five cerrado days); with `switch` the derivatives equal round 11's prototype to 2e-7
(largest 109).

### The three forms (latent heat, AD against ±1 % of `iota`)

| | repository | switch at -2 | blend (-1 to -2) | clip |
|---|---|---|---|---|
| CHATS7, one day: daily mean off by | 0.3 % | 0.2 % | < 0.05 % | 2.0 % |
| cerrado, five days: patch-days > 5 % / > 1 % off (of 72) | 6 / 11 | 10 / 25 | 10 / 26 | 12 / 30 |
| cerrado: worst five-day mean | 1000 % | 11.9 % | 12.1 % | 12.2 % |
| cerrado: largest step derivative | 2.3e4 | 1.1e2 | 1.1e2 | 1.0e2 |

The consistent derivative at every step is the derivative of another model: 2 % off on a
clean day. The blend is as good as the switch and continuous: used below.

### Thirty days: the rule (blend) and round 10's water stress together

| | repository | both |
|---|---|---|
| **Great Plains July**: largest step derivative | 8.9e6 | 1.3e3 |
| worst 30-day mean | 3764 % | 6.6 % |
| patches within 1 % / 5 % (of 21) | 12 / 14 | 17 / 20 |
| patch-days more than 5 % / 1 % off (of 609) | 97 / 143 | 59 / 105 |
| **Cerrado January**: largest step derivative | 1.6e9 | 8.9e2 |
| worst 30-day mean | 5.6e6 % | 11.6 % |
| median 30-day mean error | 3.3 % | 1.9 % |
| patches within 1 % / 5 % (of 18) | 3 / 10 | 5 / 14 |
| patch-days more than 5 % / 1 % off (of 522) | 211 / 279 | 168 / 336 |

- Great Plains: 19 of 21 patches within 1.3 % (the others 4.3 and 6.6 %, stressed
  patches). Cerrado: as round 11 alone (it is not water stress); a third of the daily
  means stay more than 5 % off, where the forward model flips between the stable and the
  unstable limit (round 11).
- The two treatments do not interfere; no NaN.
- The 30-day forward values differ from the stored references by 1e-5 (Great Plains) and
  2e-3 W m-2 (cerrado), the size of the twins of round 8 (another compiled program).

### Cost (`timing/time_model.py --steps 47`, CHATS7, CPU, two runs each)

| | switch off | rule (blend) | rule + water stress |
|---|---|---|---|
| forward, per step | 10.0-10.3 ms | 9.6-10.0 ms | 9.9-10.1 ms |
| gradient, per step | 56-57 ms | 94-102 ms (1.7 times) | 100-110 ms (1.8 times) |
| peak memory | 3.6-4.1 GB | 5.1-6.2 GB | 5.2-6.2 GB |
| first gradient call (compile) | 47-80 s | 112-181 s | 118-179 s |

### Decided (decision 124, maintainer 2026-11-16)

**Put on hold the same day (maintainer):** see decision 124 and round 14. The list below is
what building it would take.

Of three options (both on; the water stress on and the Obukhov rule a setting, off by
default; the water stress only, finite differences of means elsewhere), the second:
the default gradient keeps today's cost, and a sensitivity experiment switches the rule on
where stability flips. Not in the code yet. To build:

- the water-stress derivative (round 10's form, written as one named function's
  derivative) and the Obukhov rule (blend), each with a setting; names to be chosen;
- tests: forward bit for bit with the settings on and off; the CHATS7 gradient check
  unchanged; bounded derivatives over several days in both 1° boxes; agreement with finite
  differences of means within stated tolerances;
- a reliability flag in the output (per patch and record: L changed sign between stages;
  water stress active), so a sensitivity map shows where a derivative is approximate;
- documentation that with these settings the gradient is not the derivative of the forward
  values.

### Open

- Tested for one parameter (`iota`), one output (latent heat), two boxes, one month each;
  the -1 to -2 ramp of the blend was not tuned (round 11 compared switches at -0.8, -2
  and -4).
- A cheaper Obukhov rule (the second pass is most of the 1.7 times).

## Round 14: other Runge-Kutta schemes for the sub-step (2026-11-16). Not pursued

The maintainer's question: the feedbacks arise from the stages; is the scheme the cause?
The stages take every diagnosed quantity from the stage's start, so the scheme steps
explicitly over the loop "state -> diagnosed quantity -> fluxes -> state". For a loop of
constant gain g, the tangent grows per sub-step by the scheme's stability polynomial R(g):
Euler is stable for g > -2, RK4 for g > -2.78 (round 11 measured gains of -2 to -3 and
beyond). Stabilized Runge-Kutta-Chebyshev schemes with the same four stages are stable to
g = -9.9 (order 2) and -31 (order 1). Tree `$SCRATCH/clm-ml-jax/trees/rkc` (tableaus
`rkc2_4`, `rkc1_4`, `rkc2_5` added to `integrators.TABLEAUS`); probes
`$SCRATCH/clm-ml-jax/probe_rkc/` (`rkc_tableau.py`, `run5.sh`).

Five cerrado days, latent heat, AD against each scheme's own ±1 % of `iota`:

| | rk4 | euler | rkc1_4 | rkc2_4 |
|---|---|---|---|---|
| largest step derivative | 1.3e5 | 1.1e4 | 3.0e3 | 1.9e4 |
| patch-days more than 5 % off (of 72) | 6 | 8 | 2 | 18 |
| patches with the mean within 1 % (of 18) | 11 | 9 | 16 | 8 |
| worst mean | 6085 % | 331 % | 72 % | 502 % |
| mean latent heat, change from rk4 (W m-2) | | -0.00 | +0.63 | -0.14 |
| largest single-step change (W m-2) | | 65 | 80 | 72 |

The first-order stabilized scheme helps but does not solve it and moves the forward model
most; the second-order one is worse than RK4. The gain is not constant from stage to stage
and the limits on L intervene: the constant-gain picture holds only in part.

## Round 15: an implicit derivative over the stages; the checkpoint on the sub-step (2026-11-16)

The maintainer's idea: do not differentiate through the Runge-Kutta stages at all. Trees
`$SCRATCH/clm-ml-jax/trees/impl` and `impl2` (the repository plus
`canopy.py::_implicit_stages`; switch `CLM_IMPL=fwd|rev`, `CLM_IMPL_K` GMRES steps,
`impl2`: `CLM_IMPL_FORM=fix|exp`). Probes and results: `$SCRATCH/clm-ml-jax/probe_impl/`
(`cfg.py`, `cfg2.py` with `WRT=iota|lw|tsoil`, `gradtime.py`, `run_*.sh`).

### The rule

The last pass of a sub-step is `out = P(z, q)`: the canopy processes from the state `z`
the stages gave, with everything else `q` (the state at the sub-step's start, radiation,
forcing, parameters). Forward values: the repository's (bit for bit, CHATS7). Derivative:
that of a sub-step whose state agrees with the state its own pass gives (`z = G(z, q)`,
`G` the state part of `P`), at the forward values:

    (I - G_z) y = G_q dq,   d out = P_q dq + P_z y

One linearization of the last pass, applied k + 1 times in one compiled loop (GMRES;
6 steps are converged). No quantity is named. A forward-mode rule (`jax.custom_jvp`) and
its transpose as a reverse-mode rule (`jax.custom_vjp`), which agree (CHATS7: -27.7237
both). With the library's `gmres` the program took 9.5 GB to compile (box runs killed at
~10 GB on the login node); with the pass in one loop 5.9 GB (CHATS7) and 8.1 GB (box).

A second form (`exp`: the stages as the equation they integrate, with the pass linearized
once, `d z = d z_start + phi(A) u`) was no better on CHATS7 and gave 1e38 at a cerrado
patch (a strongly reinforcing feedback in the linearized pass): dropped.

### Accuracy (AD against ±1 % of `iota`, unless said)

| | exact (repository) | implicit | rounds 10 + 13 |
|---|---|---|---|
| CHATS7, one day: LE / H / GPP daily mean off by | 0.3 / 0.3 / 0.3 % | 1.2 / 3.2 / 6.6 % | |
| the same without water stress | 0.2 / 0.2 / 0.1 % | 1.2 / 3.3 / 7.2 % | |
| cerrado, 5 days, LE: worst patch; patch-days > 5 % (of 72) | 6085 %; 6 | 10.5 %; 13 | 12.1 %; 10 (round 13) |
| **Great Plains July, 30 days**, LE: worst; patches within 5 % (of 21) | 3764 %; 14 | 10.9 %; 19 | 6.6 %; 20 |
| H: worst; within 5 % | 3000 %; 14 | 6.4 %; 20 | 3.4 %; 21 |
| GPP: worst; within 5 % | 4570 %; 13 | 24.4 %; 17 | 17.3 %; 17 |
| **Cerrado January, 30 days**, LE: worst; within 5 % / 10 % (of 18) | 5.6e6 %; 10 / 10 | 60.3 %; 11 / 12 | 11.6 %; 14 / 17 |
| H: worst; within 5 % / 10 % | 1.7e6 %; 9 / 11 | 25.6 %; 11 / 13 | 15.8 %; 16 / 16 |
| GPP: worst; within 5 % / 10 % | 1.2e6 %; 10 / 10 | 4.5 %; 18 / 18 | 4.3 %; 18 / 18 |
| largest step derivative, 30 days (LE) | 1.6e9 | 2.1e4 | 1.3e3 |
| cerrado, 5 days, downward longwave x (1 + e): LE worst; median | 7000 %; 3.1 % | 34.8 %; 0.6 % | |
| cerrado, 5 days, initial soil temperature + e: LE worst; median | 3280 %; 0.8 % | 19.6 %; 1.2 % | |
| the same, GPP: worst; median; patches within 5 % | 12300 %; 2.0 %; 13 | 46.3 %; 4.9 %; 9 | |

- It removes the explosions for a parameter, a forcing and a state, without naming a
  quantity, and where the exact derivative is meaningless it is far closer.
- It is biased where the exact derivative is right: on CHATS7 the error sits in the
  morning steps (GPP derivative 20-30 % low from sunrise to late morning, the rest of the
  day within 1 %), with or without water stress. Both forms give it, so it is not the
  function applied to the gain: the pass is linearized once, at the last pass, and in the
  morning transition its slopes differ much from stage to stage.
- Over 30 days in the cerrado it is worse than the treatments of rounds 10 and 13 (six
  patches more than 10 % off in latent heat, the worst 60 %): a reinforcing feedback in
  the linearized pass makes `I - G_z` nearly singular at single steps, and nothing limits
  the amplification (round 13 divides by at least 1).
- States with memory: the derivative with respect to the initial soil temperature is
  bounded but GPP is 5 % off in the median.

### Cost, and the checkpoint on the sub-step (`gradtime.py`: reverse mode, mean latent heat, all parameters, one day)

| | CHATS7 (1 patch), ms per step, two runs | cerrado (18 patches), ms per step | peak memory, cerrado |
|---|---|---|---|
| forward | 10 | 46-51 | 2.2 GB |
| exact, checkpoint on the sub-step (the repository) | 53.3 / 55.7 | 1166 / 1150 | 8.1 GB (6.3 GB from the cache) |
| exact, no checkpoint on the sub-step (`CLM_ML_NO_CHECKPOINT=1`) | 48.6 / 43.8 | 195 | 6.5 GB with the compilation |
| implicit, 6 steps, checkpoint | 54.8 / 49.1 | 545 | 8.4 GB |
| implicit, 6 steps, no checkpoint | 48.2 / 45.9 | 235 | 6.4 GB |
| implicit, 3 steps, no checkpoint | 44.6 / 39.9 | 198 | 6.4 GB |

- **The checkpoint on the sub-step costs a factor 6 on the box** (10-20 % on one patch)
  and saves no memory that these runs show; the checkpoint on the model step stays. Not
  measured: memory per patch at run time for many patches, and a GPU.
- The implicit gradient costs what the exact one costs (within 20 %).

### Open (maintainer to decide)

- The checkpoint on the sub-step as a setting, off by default (the gradient's values do
  not change).
- The implicit rule is not good enough as the gradient (the 30-day cerrado, the morning
  bias). A candidate that follows from these runs, not built: the exact derivative of the
  sub-step where its linearized pass is within the scheme's stability limit, the implicit
  one where it is not, weighted by one measure of the pass (its spectral radius, by power
  iteration): round 13's blend without naming a quantity. Its cost, both derivatives of a
  sub-step, is affordable without the checkpoint (about 430 against 1150 ms per step
  today on the box).

## Round 16: the sub-step checkpoint on a GPU (decision 125); the exact and implicit derivatives combined (2026-11-16)

### The checkpoint on the sub-step: memory and time on a GPU

`$SCRATCH/clm-ml-jax/probe_ckpt/` (`grad_mem.py`: `jobs/gpu_probes/grad0.py` with the peak
device memory; `mem.pbs`; one A100 of 40 GB, loops written out, reverse mode with respect
to the parameters, 8 steps, steady state; the patches are copies of the CHATS7 patch).

| patches | checkpoint on the sub-step | gradient, ms per step | peak GPU memory | forward, ms per step |
|---|---|---|---|---|
| 1 | yes | 240 | 0.005 GB | 43 |
| 1 | no | 209 | 0.10 GB | 43 |
| 1 024 | yes | 333 | 2.2 GB | 60 |
| 1 024 | no | 348 | 11.5 GB | 60 |
| 4 096 | yes | 791 | 8.3 GB | 132 |
| 4 096 | no | out of memory (an allocation of 42.7 GB) | | 132 |

- On a GPU the checkpoint costs no time with many patches (13 % on one patch) and saves a
  factor 5 in memory at 1 024 patches (2 against 11 MB per patch); without it 4 096
  patches do not fit. On the CPU it cost a factor 6 on 18 patches (round 15).
- **Decision 125 (maintainer, 2026-11-16):** the setting `numerics.checkpoint_substeps`
  (`auto`, `always`, `never`); `auto` is `always` on a GPU and `never` on the CPU. In the
  code, with a test that the gradient is the same either way.
- Not measured: CPU gradients of thousands of patches.

### The combined rule (prototype, not adopted)

The candidate at the end of round 15. Tree `$SCRATCH/clm-ml-jax/trees/hyb` (`impl2` plus
the blend in `canopy.py::_implicit_stages`, switch `CLM_IMPL=hyb`); probes and results in
`$SCRATCH/clm-ml-jax/probe_hyb/` (`hyb.pbs`, tag `a`). The derivative of a sub-step is
(1 - w) times the exact one plus w times the implicit one. The weight w comes from the
forward values only: g, the most negative real part of the eigenvalues of the pass's
feedback on its own state, estimated with 6 Arnoldi steps. Runge-Kutta 4 is stable for
g > -2.78; w is 0 for g >= -2.0, 1 for g <= -2.78 and linear in between. Forward mode,
latent heat against ±1 % of `iota`, means over the run per patch:

| | exact (repository) | implicit (round 15) | rounds 10 + 13 | combined |
|---|---|---|---|---|
| CHATS7, one day: worst | 0.3 % | 1.2 % | | 0.3 % |
| cerrado, 5 days: worst; patches within 5 % (of 18) | 6085 % | 10.5 % | 12.1 % | 7.5 %; 16 |
| cerrado January, 30 days: worst; within 5 % / 10 % (of 18) | 5.6e6 %; 10 / 10 | 60.3 %; 11 / 12 | 11.6 %; 14 / 17 | 10.2 %; 12 / 16 |
| the same: largest step derivative | 1.6e9 | | 8.9e2 | 1.8e3 |
| Great Plains July, 30 days: worst; within 5 % (of 21) | 3764 %; 14 | 10.9 %; 19 | 6.6 %; 20 | 274 %; 18 |
| the same: largest step derivative; patches above 1e4 | 8.9e6 | | 1.3e3; 0 | 1.6e5; 3 |

- CHATS7: the morning bias of the implicit rule is gone (GPP and sensible heat 0.3 and
  0.4 % off, against 6.6 and 3.2 %): where the time stepping is stable the rule is the
  exact derivative.
- Cerrado: bounded over 30 days, the worst patch as rounds 10 + 13, fewer patches within
  5 %.
- **Great Plains: not solved.** Three of 21 patches still have step derivatives above 1e4
  (largest 1.6e5) and one 30-day mean is 274 % off (-130 against -35); the other 20 are
  within 6.4 % (16 within 1 %). Not diagnosed. The Great Plains defect was the water
  stress (round 9), a feedback through the leaf water potential from sub-step to
  sub-step: a measure of one sub-step's stages may not see it (a guess, to be checked).
- Cost: not measured as a gradient time. The program with both derivatives needs more
  than the login node's 10 GB to compile (the cerrado run was killed there; run on a
  compute node with 100 GB: 1004 s for the 30 cerrado days in forward mode).
- One threshold pair, one parameter, one output; rounds 10 + 13 remain better in both
  boxes over 30 days.

### Open (maintainer, 2026-11-16: re-evaluate)

- The direction for the lagged feedbacks: diagnose the three Great Plains patches under
  the combined rule, return to decision 124 (rounds 10 + 13), or something else.
