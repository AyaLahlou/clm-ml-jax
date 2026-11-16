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

**State (2026-11-14).** Round 1 of the speed work is measured (below); no model code has
changed. The timing scripts are corrected (decision 118). Next, by the maintainer's choice:
the quality and accuracy of the gradients.

**Open from round 1** (O76): forward or reverse mode as the tuned path; the checkpoint
layout as a setting; GPU measurements of checkpointing and of batched forward mode; a
jitted derivative as the first call on a model fails; the CPU gradient with written-out
loops ended without a result.

**How to measure** (the probes are not in the repository: put them back with
`git checkout b954677 -- jobs/gpu_probes`, see `gpu_progress.md`; results in
`$SCRATCH/clm-ml-jax/probe_grad0/`):

```bash
# CPU (login node): forward, one forward-mode derivative, gradient; any setting as an override
JAX_PLATFORMS=cpu python jobs/gpu_probes/grad0.py . [--wrt params|state|forcing] \
    [--modes fwd,jvp,grad,jgrad,jjvp,jacfwd] [--steps 8] [--stop MODULE:FUNCTION] [overrides]
# GPU (one A100, develop queue; submit from scratch, time a fixed snapshot of the code)
qsub -v TAG=name,TREE=$SCRATCH/clm-ml-jax/trees/NAME,SL=unrolled jobs/gpu_probes/grad0.pbs
# what the forward and gradient programs contain (static; CPU)
JAX_PLATFORMS=cpu python jobs/gpu_probes/census_grad.py .   # operations by source function
JAX_PLATFORMS=cpu python jobs/gpu_probes/loops_grad.py .    # every loop, with its trip count
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
