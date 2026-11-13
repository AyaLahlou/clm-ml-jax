# GPU optimization plan (open item O55)

Status: **step 0 (measurements) done 2026-11-13; stopped for the maintainer before step 1**
(branch `gpu_opt`). Results and the revised order of work: [`gpu_progress.md`](gpu_progress.md).
In short: the fixed cost is 1.2 s of the 2.0 s step (not 0.4 s), 94 % of the GPU commands
come from the psihat table search, and a prototype that removes that loop reaches 1.86×. §2
below is the picture before step 0 and is kept as written. Written 2026-11-13; reviewed and
revised the same day (§9 lists what the review changed). The deferred work it picks up: Phase 6 step 1c (kernels per step),
"After Phase 8: GPU performance" in the modernization plan, O55. Target and scope: decision 113.

## 1. Goal, targets and how they are measured

**Goal:** more simulated time per GPU-hour for a **global 1° forward run** (the maintainer's
target, §7), with **forward values unchanged** (bit for bit where the arithmetic is
untouched, otherwise within the noise twin). The same code is used for gradients, so
gradients must keep working, and their time and memory are watched at every step.

**Primary measure:** steady patch-steps per second on one A100 for the 1° global domain
(72 575 patches, 14 346 gridcells), warm compilation cache, compile time excluded.

**Target (maintainer, 2026-11-13): a 2× speed-up.** For the 1° globe on one A100: **1.0 s per
30-min step** (today 2.0 s), ~74 000 patch-steps/s, a simulated year in ~4.9 h instead of
~9.6 h. Measured as the run's wall time after setup and compile, divided by its steps, so
padded steps count against it. (The maintainer first asked for 0.5 s per step, 4×, and then
chose 2× for now.)

| Milestone | Seconds per step | Simulated year | How |
|---|---|---|---|
| Today | 2.0 (2.6 averaged over the January run, with its padding) | ~9.6 h | |
| 1 | 2.0 on average | ~9.6 h | padded steps cost nothing (step 1a) |
| 2 | ~1.3 (1.5×) | ~6.4 h | flags, first unrolling |
| **Target** | **1.0 (2×)** | **~4.9 h** | unrolling and hot-spot rewrites (steps 2–3) |

**Is 2× realistic?** Probably, but not known before step 0:
- the arithmetic is not the obstacle: the GPU computes at a few percent of its capacity (§2);
- a step is ~0.4 s of fixed cost plus ~1.6 s that grows with the patches. 1.0 s needs, for
  example, 0.2 s + 0.8 s: both parts halved. Loop unrolling attacks both.

Step 0 ends with an estimate of how far each step can go, and the target is confirmed or
revised with the maintainer then.

**Limits that must hold at every step:**
- forward compile time and code size within the budget (§7, item 6);
- peak GPU memory under ~30 GB of the A100's 40 GB (today 7.2 GB, so there is room to trade
  memory for speed);
- gradient time, memory and compile time not worse than ~1.5× today's without discussion.

**When to stop:** when the target is met, or when the remaining candidates each promise
less than ~5 %, or when the only candidate left is the invasive one of step 4 and the
maintainer decides it is not worth its cost. The effort ends with a report, not when
the list is empty.

**Where we start** (run `global_1mon_test`, 2026-11-05, JAX 0.10.2):

| Item | Value |
|---|---|
| Steady chunk | ~945 s per 480 steps = **~36 900 patch-steps/s**, ~2.0 s per step |
| One simulated year (estimate) | ~9.6 h on one A100 |
| Setup | 47 s |
| First chunk | +45 s over steady (compile) |
| Peak memory | 7.2 GB of 40 GB |
| Earlier measurements | Phase 6b: 42 700 patch-steps/s; end of Phase 7: 38 400 (JAX 0.9.2) |

The model has become about 14 % slower per chunk since Phase 6b. Step 0 finds out why
(§3, 0c): it may be the cheapest gain of all.

## 2. What we know, and the hypotheses to test

One 30-min step is 6 sub-steps × 4 Runge–Kutta stages = **24 evaluations of the canopy
physics**. Each evaluation contains loops that the GPU runs one iteration at a time:
- the Obukhov length solver (25 secant steps);
- the WUE stomatal bisection (20 halvings plus 5 more evaluations);
- the tridiagonal and 2×2 block-tridiagonal solves (loops over the layers, forward and back);
- the two-stream radiation sweeps (loops over the layers);
- some sums written as loops.

That is several thousand loop iterations per step, each made of a few small GPU programs
("kernels").

### Evidence already on disk

**Time against number of patches** (Phase 6 step 1b, `timing/results_vmap.jsonl`, one A100,
CHATS7 copies with 46 layers):

| Patches | 1 | 4 096 | 16 384 | 65 536 |
|---|---|---|---|---|
| ms per step | 365 | 508 | 912 | 2 303 |

So a step costs a **fixed ~0.4 s** however few patches there are, plus **~30 µs per patch**.
For today's global run (72 575 patches, 20 layers, 2.0 s per step) that is roughly **20 %
fixed cost and 80 % cost that grows with the number of patches**.

**The Phase 6 profiles** (`$SCRATCH/clm-ml-jax/prof/base_N{1,4096}*`; old code, so
indicative only):
- at 1 and at 4 096 patches the GPU is mostly idle (busy 16 % at one patch). The time goes
  to **one CPU thread issuing commands to the GPU**: per step about 57 000 kernel launches
  and 55 000 copies of data from one place in GPU memory to another, at 3–4 µs each. That is
  the fixed ~0.4 s;
- the copies are the values a loop carries from one iteration to the next;
- the kernels themselves are tiny (1.6–2.7 µs at 4 096 patches). One kernel
  (`loop_select_fusion_2`) takes 22 % of kernel time in few calls: most likely the body of
  the stomatal bisection;
- kernels that reorder array axes (transposes) exist but are small (~2 %).

### Hypotheses for the 80 % that grows with patches

The old profiles do not reach the global size, so these are hypotheses. Step 0 tests each.

| | Hypothesis | Test in step 0 | Fix if true |
|---|---|---|---|
| H1 | **Carried-value copies.** Every loop iteration copies its carried arrays; at global size each copy is 0.6–23 MB instead of a few kB | copies per step × bytes, from the profile | unroll loops (2a), carry less (2b) |
| H2 | **Unfused arithmetic.** Each kernel reads its inputs from GPU memory and writes its result back. Many small kernels in a row move the same arrays many times; the GPU's memory speed, not its arithmetic, is the limit | kernel time against bytes moved; arithmetic rate against the A100's peak | unroll so XLA merges kernels (2a); rewrite hot bodies (3) |
| H3 | **A few heavy kernels** (the bisection body, the Obukhov body) dominate | top kernels by time | targeted rewrites (3) |
| H4 | **C3 and C4 groups** run the whole step twice (decision 22), doubling the fixed cost for 9 % of the patches | time a mixed domain against an all-C3 one of the same size | maintainer's decision (§6) |
| H5 | **Array layout.** Arrays are (patch, layer); loops over layers read every 20th number | time in transpose and slice kernels | layout change (4), only with strong evidence |

A rough estimate supports H1/H2: the model does a few million arithmetic operations per
patch-step in 22 µs, a few percent of what an A100 can compute. The GPU is waiting on memory
and on commands, not on arithmetic.

### Two costs that are certain

- **Padded steps are fully computed.** A short last chunk is padded to the full length
  (decision 33), and the padded steps are discarded with `jnp.where` (`driver.py`), so they
  cost as much as real steps. In the January run the 1-day last chunk cost a whole 10-day
  chunk: about **25 % of the run**.
- **The fixed ~0.4 s per step**, about 20 % of a global step.

### What the review ruled out

- **Gradient checkpointing in forward runs costs nothing.** The first version of this plan
  suspected `jax.checkpoint` of slowing forward runs. JAX's source says otherwise: the barrier
  it adds is only inserted when the program is differentiated
  (`jax/_src/ad_checkpoint.py::_remat_lowering`). A forward run compiles the plain step. So
  this is now a 10-minute check in step 0, not a step.

## 3. Step 0: measure (no changes to the model; probes in scratch)

Nothing in `src/` changes in step 0. The existing timing tools are reused, not rewritten.

- **0a. The yardstick.** `timing/time_program.py` through `jobs/program_gpu.pbs` already
  times the global chunk on one A100 for any commit (`qsub -v REV=...,TAG=...`). Add a short
  variant with a 1-day chunk (48 steps, `run.execution.chunk=1 day`), ~5–10 min per job, for
  quick comparisons; the 10-day chunk stays the reference number.
  - Run the baseline **three times** to get the run-to-run spread. A change counts only if
    it exceeds three times that spread.
  - Each result records commit, JAX version, XLA flags, CPU cores, and GPU model. The
    command thread runs on the CPU, so CPU contention on the node can change the result
    (the January log shows a failed CPU-binding call); ask for the same cores every time.
- **0b. The scaling curve today.** `timing/time_vmap.py` at 1, 4 096, 16 384 and 65 536
  patches with 20 layers: the fixed cost and the cost per patch of the current code.
- **0c. Why 14 % slower than Phase 6b?** Time a few commits between Phase 6b and now with
  0a (the job takes `REV`). Known suspects: patch constants became inputs of the program
  in Phase 7 (it made the program small and cacheable, but XLA can no longer pre-compute with
  them); the JAX upgrade (decision 105); output streams. Result: which change cost what.
- **0d. A profile of the global chunk** (nsys, as Phase 6 step 1c; its job and probe are in
  `$SCRATCH/clm-ml-jax/jobs/gpu_1c.pbs` and `probe/profile_step.py`, to be brought up to the
  current code; two steps are enough): kernels and copies per
  step, bytes copied, GPU busy fraction, top kernels by time and the physics they belong to,
  transposes. This decides H1, H2, H3 and H5.
- **0e. Each process alone.** Time every process as its own compiled function on
  global-size inputs (72 575 patches × 20 layers): radiation, turbulence with the Obukhov
  solver, photosynthesis with the bisection, flux profile, soil, canopy water. Weight by
  calls per step. Result: a ranking, and small test programs that compile in seconds. These
  are the test beds for steps 2 and 3.
- **0f. C3/C4 groups** (H4): mixed against all-C3.
- **0g. Quick checks:** the forward run with and without `checkpoint_steps` (expected: no
  difference); time between chunks when the GPU waits for the host (forcing, output,
  health checks).

**Result:** `docs/design/gpu_progress.md` with a table of where the 2.0 s per step goes and
which hypotheses hold. **Stop and agree the order and targets of steps 1–4 with the
maintainer.**

## 4. Steps 1–5

Each step ends with the yardstick (results appended to `timing/results_*.jsonl` with a
label), the tests of §5, and a short entry in `gpu_progress.md` with the gain measured. Each
change is its own commit, so any one can be dropped. A change that gains less than the
measurement spread is **not kept** (less code to maintain).

**Method for steps 2 and 3: prototype first.** Try each idea on the single process from 0e,
in scratch, where a compile takes seconds. Only ideas that win there are ported to `src/`
and timed on the whole model. This keeps failed experiments out of the repository and saves
GPU hours.

### Step 1: certain wins (arithmetic unchanged; bit for bit on CPU)

- **1a. Skip padded steps.** Replace the `jnp.where` on `active` with `lax.cond`, so a padded
  step does no work. One program per run is kept (decision 33).
  - Checks: real steps bit for bit; the padded-chunk and restart tests; gradients through a
    padded chunk; several devices (`shard_map`).
  - Expected: a run's padding costs ~0 instead of up to one chunk.
- **1b. XLA GPU flags.** Re-measure at global size what Phase 6 measured at 4 096 patches:
  command buffers with unrolled loops (`--xla_gpu_command_buffer_unroll_loops`, 10 % faster
  then), while-loop double buffering, autotuning level. Flags that help are set in
  `clm_ml.driver` unless the user has chosen their own, each recorded as a decision. A new
  flag means a new compiled program: the cache is refilled once.
- **1c. Whatever 0c finds**, if it can be undone without giving up cacheable programs.

### Step 2: fewer, larger loop iterations (same operations; code size watched)

The likely main lever: it attacks both the fixed cost (fewer commands) and the cost per
patch (fewer copies of carried values, and XLA can merge the kernels of several iterations
into one, so intermediate values never go to memory).

- **2a. Unroll the fixed-length loops.** `lax.scan` and `lax.fori_loop` take `unroll=k`: the
  same iterations in the same order, but k of them become one stretch of straight-line code.
  Order of attack, by expected gain per risk:
  1. the solver loops, whose bodies are pure array arithmetic: bisection (20), Obukhov (25),
     Ci secant (40, Medlyn/Ball–Berry only);
  2. the loops over layers: tridiagonal and block-tridiagonal solves, radiation sweeps;
  3. the Runge–Kutta stages and the sub-step loop (largest code growth; last, if at all).
- **Unrolling is a static setting, not a constant** (a hidden `numerics` option, one
  factor per loop family). Reasons: the best factor is found by measurement; a gradient
  program can keep loops if unrolling makes its compile too slow; the CPU can stay as it is.
- **Gradients:** the solver loops are behind `stop_gradient` (their gradients come from the
  implicit-function step), so unrolling them adds nothing to gradient memory. The layer
  loops are differentiated: their gradient compile time and memory are measured.
- **Risks:** code size and compile time (Phase 6b: 2.3 GB programs that could not be cached;
  the gradient program already compiles in 6–26 min). Measured for every factor, against the
  budget in §7.
- **Bit for bit:** unrolling keeps the operations and their order, so the CPU result should
  be identical. If the compiler still changes the last bits, the change is treated as a
  round-off change (noise twin, reference regenerated, said so).
- **2b. Carry less.** If the profile shows large arrays carried through loops that do not
  change them, take them out of the carry.

### Step 3: round-off-level rewrites of the hot spots (noise-twin test)

Guided by the 0e ranking; candidates:
- **Sums written as loops** (`soil.py`, the radiation sum) → `jnp.sum`. Parallel instead of
  in order; the order of addition changes, so round-off does.
- **Hot kernel bodies** (H3): for example evaluating the bisection's first two and last three
  function values in one stacked call instead of five.
- **Tridiagonal solves.** Preferred: the Thomas algorithm unrolled (2a), which keeps
  Fortran's arithmetic. JAX's library solver (`jax.lax.linalg.tridiagonal_solve`, cuSPARSE on
  the GPU) is **only a candidate to measure**: it uses a different algorithm (pivoting), and
  its speed for 72 575 small systems at once is unknown. Not adopted without a clear win in
  the 0e test bed.
- **Recomputed quantities.** Values identical in every RK stage or sub-step are computed
  once, where that matches the Fortran order of operations. Where it would not match, it is
  a science question for the maintainer and is not done here.

### Step 4: array layout (only with strong evidence, and only if steps 1–3 miss the target)

Putting the patch axis last for the loops over layers touches every process. It is the most
invasive change in this plan and the easiest to get subtly wrong. Only if H5 holds; after a
discussion; starting with one process as a trial.

### Step 5: runs, scaling and gradients

- Gaps between chunks where the GPU waits for the host.
- Multi-GPU scaling re-measured (Phase 6: one GPU twice as efficient per GPU-hour as four).
- More patches per GPU: the fixed cost is shared by more patches, so 0.5° or an ensemble may
  cost less per patch than 1°. Worth one measurement with the memory now free.
- Gradient time and memory on the GPU with all changes in place.

## 5. Testing for every change

- **CPU tiers as usual** (CLAUDE.md): fast tier after each edit; bit-for-bit and noise twin
  before each commit. Steps 1–2 are expected to pass bit for bit. Steps 3–4 are expected to
  fail bit for bit, must pass the noise twin, and the reference is regenerated with
  `--update-reference`, saying so.
- **The Fortran tests stay the anchor:** the golden routine tests after every edit; the
  end-to-end Fortran comparison (`tests/test_fortran_python_comparison.py`) after step 3 and
  at the end. Golden JSON files are never edited.
- **If a "round-off" change moves results by more than ~1e-10,** it is not round-off: look
  for a discrete defect (CLAUDE.md, "Noise twins"), do not widen tolerances.
- **A GPU comparison run.** The CPU tests prove the code; this proves the GPU program. One
  day of global output on the GPU, compared with the step-0 baseline output:
  - two compiles of the same GPU program already differ by ~1e-11 (O27), and real 1° patches
    amplify round-off after sunrise. So, as in the grid regression tests, the comparison is
    tight through the night and statistical in the day (distribution of differences, global
    means, no new excluded patches);
  - the baseline is run twice, so that the GPU's own compile-to-compile spread is known
    before any change is judged against it.
- **Gradients:** the CPU gradient tier (`tests/gradient_tier.py`) after any change to a
  differentiated loop; the GPU gradient job (`jobs/grad_gpu.pbs`) after steps 2 and 3 and at
  the end, with its time, memory and compile time recorded.
- **Standing rule:** a slowdown of more than ~1.5× in anything measured (forward, gradient,
  compile) is discussed before going on.

## 6. Not in this plan (each its own decision)

- **Fewer solver iterations** (Obukhov 25, bisection 20, Ci 40), fewer sub-steps, another
  integrator: numerics choices tied to Fortran (decision 87). Not optimizations.
- **float32** (O60): needs its own validation; float64 is fast on the A100.
- **O27** (GPU compile-to-compile variation): affects how GPU results are compared, not speed.
- **C3/C4 groups** (decision 22): step 0f measures their cost. If it is more than ~10 %, the
  number goes to the maintainer as the new reason to revisit that decision; nothing changes
  before.
- **Custom CUDA kernels: ruled out for now** (maintainer, 2026-11-13). JAX can call
  hand-written C++/CUDA kernels (`jax.ffi.ffi_call`), and one kernel could do a whole solver
  (bisection, Obukhov, Ci) per leaf in one pass over memory. The solvers are outside the
  gradient path, so gradients would still work. Not done because the science would be written
  twice (C++ next to the JAX version that the CPU and gradients need), the package would stop
  being pure Python, and CI has no GPU. To be reconsidered only if a later, more ambitious
  target (the 0.5 s per step first asked for) is taken up and the solver loops still dominate.
- **Skipping work where nothing happens** (photosynthesis in the polar night, for example):
  it would need patches reordered by time of day. Complex, and only worth discussing if
  photosynthesis dominates the profile.

## 7. Decisions

Answered by the maintainer (2026-11-13; decision 113):

1. **Target: global forward runs.** Gradients will be used too. So no change may break
   them, and every step reports gradient time and memory alongside the forward ones. A
   forward gain that makes gradients much slower or bigger in memory is discussed before it
   is kept.
2. **Round-off changes are part of this effort** (steps 3–4), under the regression policy:
   noise-twin test, reference regenerated and said so.

3. **Target: 2×** (1.0 s per step; §1). Confirmed or revised after step 0.
4. **Timing jobs run on the `develop` queue** (account `UCLB0069`, as the repository's job
   scripts).
5. **Custom CUDA kernels are ruled out for now** (§6).

Open (the maintainer has no preference yet; proposals, to be settled with the step-0 numbers):

6. **Budget for compile time and code size when unrolling.** Proposal:
   - forward chunk: compile under ~5 min (today ~1 min), generated code under ~100 MB (today
     4 MB), far below the 2 GB that broke the cache in Phase 6b;
   - gradient program: compile within ~2× today's.

   The reason for a limit: a large program costs time at the start of every run that misses
   the cache, and above 2 GB it can never be cached at all. Because the unroll factors are
   settings (step 2a), the budget can be met separately for forward and gradient programs.
7. **Padded chunks (1a).** Proposal: `lax.cond` inside the one program. It keeps decision 33
   (one program per run, nothing new to compile or cache); padded steps do no work; real
   steps give identical results. The alternative, a second program for a short last chunk,
   costs a second compile and its memory, which decision 33 avoided. Step 1a measures the
   `lax.cond` version before this is settled.
Also to note:

8. **Base branch:** `gpu_opt` was made from `feature/latlon_output` (it has the lat-lon output
   and the plotting script), not from `main`.

## 8. Risks

| Risk | How it is contained |
|---|---|
| Optimizing the wrong thing | Step 0 before any change; hypotheses with a test each; stop to agree the order |
| Gains too small to tell from noise | Baseline run three times; a change counts only above 3× the spread |
| Unrolling makes programs huge or slow to compile (Phase 6b) | Budget (§7); factors are settings; measured per factor |
| Forward gets faster, gradients slower | Gradient time, memory and compile recorded at every step; 1.5× rule |
| A round-off change hides a real error | Noise twin, golden tests, Fortran comparison; >1e-10 means a defect |
| GPU results cannot be compared exactly (O27) | CPU tests for exactness; GPU comparison tight at night, statistical by day |
| A library routine behaves differently from Fortran's algorithm | Prefer the same algorithm unrolled; library solver only with a measured win |
| Wasted GPU hours | 1-day yardstick; prototypes on single processes; full 10-day run only to confirm |
| Invasive change with little gain (layout) | Step 4 only with evidence, after steps 1–3, as a trial on one process, by the maintainer's decision |
| The 2× target is out of reach | Milestones (§1); step 0 gives an estimate; target confirmed or revised then |

## 9. What the review changed (2026-11-13)

- **Removed a step that could not help:** turning off checkpointing in forward runs. JAX
  only adds its barrier when differentiating (checked in the installed JAX 0.10.2 source).
- **Used the evidence already on disk:** the Phase 6 scaling curve and profiles give the
  20 % fixed / 80 % per-patch split, and show the fixed part is the CPU issuing ~110 000
  commands per step, half of them copies of loop-carried values.
- **Hypotheses with a test each** (§2) instead of a general "probably memory".
- **Targets, limits and a stop rule** (§1); a risk table (§8).
- **Step 0 reuses the existing timing tools** and adds: run-to-run spread, the scaling
  curve, the 14 % slowdown since Phase 6b, each process alone, the C3/C4 cost.
- **Unrolling moved up** as the likely main lever, made a setting, with gradients and
  compile budget handled separately for forward and gradient programs.
- **The library tridiagonal solver demoted** to a candidate: different algorithm, unknown
  speed for many small systems; the unrolled Thomas solve is preferred.
- **Layout change moved to its own, last step** with a condition.
- **Testing strengthened:** Fortran comparison named, the GPU baseline run twice, gradient
  tiers placed, the >1e-10 rule restated.
