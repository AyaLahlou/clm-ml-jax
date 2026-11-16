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

## Resume here

**State (2026-11-14).** Rounds 1 (speed) and 2-4 (accuracy) are measured. In the code: the
two gradient corrections of round 2 (decision 119; forward values unchanged) and the `avmu`
fix of round 3 (decision 120), and no water stress at or above zero leaf water potential
(decision 121; round 4's NaN gradients with dew, O79, closed). Round 5 (2026-11-15, no code
change): the corners are kinks, not jumps, but short canopies amplify round-off ~1e11 at the
evening transition and their daily-mean latent heat gradient can be 10-40 % off a finite
change. Round 6 (2026-11-15, prototype in scratch): a stable limit on zeta of 0.5 for the
short PFTs removes that alternation. Round 7 (2026-11-15, prototype in scratch): the
exploding derivatives (1e35 in a day, 1e220 in five) are the water on the leaves, whose
wetted fraction is taken from the water before the sub-step; with the wetted fraction of
the water left at the end of the sub-step they are gone. **That implicit wetted fraction is
in the code (decision 122, maintainer 2026-11-15; `numerics.wetted_fraction`, `explicit`
is Fortran's).** With it and round 6's zeta limit (still a prototype) the daily-mean
gradients agree with finite differences within 4 % on the four test days.
**Waiting for the maintainer:** the open points at the ends of rounds 5, 6 and 7.

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

### Open (maintainer to decide)

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

