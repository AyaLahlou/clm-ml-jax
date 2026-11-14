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

**State (2026-11-14).** Round 1 (speed) and round 2 (accuracy) are measured. The two
gradient corrections of round 2 are in the code (decision 119; forward values unchanged),
with tests. Next: round 3, the same check for the cases a clear CHATS7 day does not cover.

**Open from round 1** (O76): forward or reverse mode as the tuned path; the checkpoint
layout as a setting; GPU measurements of checkpointing and of batched forward mode; a
jitted derivative as the first call on a model fails; the CPU gradient with written-out
loops ended without a result.

**Open from round 2** (O76): the check for what this day does not cover (the remaining
air-temperature mismatch, Medlyn and Ball-Berry, rain and snow, C4, other towers and padded
layers, 1° patches with sparse canopies).

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
