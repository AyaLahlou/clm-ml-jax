# GPU optimization: progress log

Plan: [`gpu_optimization_plan.md`](gpu_optimization_plan.md) (O55, decision 113). Target: 2×,
1.0 s per 30-min step for the 1° globe on one A100.

Where things are:
- Probes, job script and logs: `$SCRATCH/clm-ml-jax/probe_gpu0/` (`gpu0.py`, `gpu0.pbs`,
  `census.py`, `attribute.py`, `results_gpu0.jsonl`, `log_*.txt`).
- Profiles and compiler dumps: `$SCRATCH/clm-ml-jax/prof0/`.
- Code snapshot the jobs ran: `$SCRATCH/clm-ml-jax/trees/gpu0_base` (commit `8f44da5`).
- Jobs: `develop` queue, account `UCLB0069`, one A100 (40 GB), 16 cores, JAX 0.10.2.

**Status (2026-11-13, evening):** steps 0, 1 and 2a done. The 1° globe runs at **0.70 s per
step, 2.85×** (target 2×, 0.99 s: met); padded steps cost nothing. Open for the maintainer:
gradient programs are slower with the loops written out (step 2 below, "Gradients").

## Resume here

**State.** Branch `gpu_opt` (made from `feature/latlon_output`, not merged anywhere), pushed.
Steps 0, 1 and 2a of the plan are done, tested and committed. Nothing is half-finished: no
uncommitted model changes, no jobs running. The work was paused by the maintainer after
step 1 and resumed the same evening for step 2a.

| Commit | What |
|---|---|
| `8f44da5` | the plan (decision 113) |
| `644e3be` | step 0: measurements |
| `8af77c8` | table search without a loop on GPUs (decision 114) |
| `a191453` | one search per grid in `psi_rsl` (decision 115) |
| `9c8a741` | padded steps not computed; regression reference regenerated (decision 116) |
| `2143fbb` | step-1 gradient timings, GPU gradient tier, 10-day reference |
| `c7f4a20` | step 2a: layer and turbulence loops written out on GPUs (decision 117) |

**Numbers to beat** (1° globe, 72 575 patches, one A100, JAX 0.10.2):
- 1-day chunk: **696 ms per step** (104 300 patch-steps/s); step 1: 1057–1065; before: 1975–1985.
- 10-day chunk (`timing/time_program.py`): 334.0 s, 104 300 patch-steps/s; step 1: 508.5 s;
  before: 935.5 s.
- 21-patch box (the fixed cost): 69 ms per step; step 1: 175; before: 1212.
- Compile (forward, cold): ~110 s; step 1: 35–40 s. Generated code 0.01 MB; peak 4.7 GB.
- Gradient (`timing/time_vmap.py --grad`, CHATS7 copies): **slower than step 1**: 1519 ms per
  step for 1 patch (step 1: 1037), 2197 for 4 096 patches (step 1: 1690; 7.9 GB); compile
  ~420 s (step 1: ~105–115 s). With `numerics.solver_loops: loop` the gradient program is
  step 1's.
- Target: 2× = 0.99 s per step. Reached: 2.85×.
- Measurement spread: 0.2 % within a job, 0.6 % between jobs. Believe changes above ~2 %.

**Next step** (nothing below is started; each needs the maintainer's go-ahead):
1. **Gradients and `solver_loops`** (open decision below): which loop group slows the
   gradient (the layer loops are differentiated; the Obukhov and roughness-length loops are
   behind `stop_gradient`), with `gpu0.py`-style prototypes on `timing/time_vmap.py --grad`.
2. **A profile of the unrolled program** (`MODE=prof` with `XF=nocb` on `trees/gpu2_unroll`):
   why the layer loops gave 0.34 s (the estimate from the loop bookkeeping was 0.05–0.1 s),
   and what is left. Before step 2a the stage physics was 0.43 s in kernels that ran 4–8
   times slower per operation than the stomatal bisection kernel.
3. **The stomatal bisection** (0.25 s per step before step 2a, one fused kernel of 357 µs run
   600 times per step and group). It is already one efficient kernel: stacking its five extra
   evaluations would not reduce the arithmetic, and unrolling it does not compile in
   reasonable time (step 2). A gain here needs less arithmetic per halving (invariants
   computed once outside the loop), a round-off-level rewrite.

**Open decisions**
- **Gradient programs and `solver_loops`** (plan §7 item 1: a forward gain that slows
  gradients is discussed before it is kept). With `auto`, a GPU gradient is 1.3–1.5× slower
  and compiles 4× longer than at step 1, while its forward pass is 2–4× faster. Choices:
  (a) keep `auto` for everything and set `numerics.solver_loops: loop` in the `model.yaml`
  of gradient work; (b) make `auto` mean loops for `Model` (the gradient interface) and
  written-out loops for `clm-ml run`; (c) write out only the loops that do not slow the
  gradient, once next step 1 has found them.
- The compile-time and code-size budget for unrolling (plan §7 item 6): settled for the
  forward program by decision 117 (~110 s accepted); open for the gradient program.
- Whether to keep change 2 (shared searches, decision 115): 0.7 % at the global size, 9 % of
  a small domain's fixed cost. Kept for now.
- Whether the round-off amplification counted in "GPU results against the baseline" below
  (5.8 % of patches more than 1e-6 apart after a day between two compiles of the same code)
  gets its own investigation (O27; issue 10 of decision 88).
- Merging: `gpu_opt` holds `feature/latlon_output` plus this work. Not merged to `main`.

**How to measure** (probe scripts are in the repository: `jobs/gpu_probes/`; they are not
linted and not part of the package):

```bash
# 1. a fixed snapshot of the code to time (never time the working tree: jobs wait in the queue)
git worktree add --detach $SCRATCH/clm-ml-jax/trees/NAME COMMIT
# 2. submit from scratch (PBS drops its own .o file where qsub is run)
cd $SCRATCH/clm-ml-jax/probe_gpu0
J=/glade/work/linnia/clm-ml-jax/jobs/gpu_probes/gpu0.pbs
T=$SCRATCH/clm-ml-jax/trees/NAME
qsub -v MODE=base,TAG=name,TREE=$T $J                    # yardstick: globe, 1 day, 3 repeats
qsub -v MODE=save,TAG=name_2,TREE=$T,STEPS=2 $J          # saved output after 2 steps (fresh compile)
qsub -v MODE=save,TAG=name_48,TREE=$T,REP=3,SMALL=1,REAL=24 $J   # 1 day saved + small box + padded chunk
qsub -v MODE=grad,TAG=name,TREE=$T $J                    # gradient timing, 1 and 4 096 patches
qsub -v MODE=prof,TAG=name,TREE=$T[,XF=nocb] $J          # nsys profile + XLA dump (globe and small box)
qsub -v MODE=scale,TAG=name,TREE=$T $J                   # step time against number of patches
qsub -v MODE=unroll,TAG=name,TREE=$T,U=layers+obukhov+z0m $J   # prototype: loop groups written out by the probe (add STEPS=2,SAVE=1 to save)
qsub -v REV=COMMIT,TAG=name jobs/program_gpu.pbs         # the 10-day reference chunk
qsub -v REV=COMMIT,TAG=name jobs/grad_gpu.pbs            # GPU gradient tier (12 min)
# 3. read the results (in $SCRATCH/clm-ml-jax/probe_gpu0: results_gpu0.jsonl, log_TAG.txt)
python jobs/gpu_probes/compare.py save_A.npz save_B.npz                 # two saved outputs
python jobs/gpu_probes/attribute.py $SCRATCH/clm-ml-jax/prof0/TAG_global $SCRATCH/clm-ml-jax/prof0/hlo_TAG_global
JAX_PLATFORMS=cpu python jobs/gpu_probes/census.py TREE                 # loops per step (CPU)
JAX_PLATFORMS=cpu python jobs/gpu_probes/noise_cpu.py unrolled          # a table search on the CPU against the noise twins
JAX_PLATFORMS=cpu python jobs/gpu_probes/noise_cpu.py solver_loops      # the solver loops written out on the CPU, likewise
```

Queue `develop`, account `UCLB0069`; jobs started within minutes on 2026-11-13. The copies
of the scripts in `jobs/gpu_probes/` were smoke-tested on the CPU (`gpu0.py`) and syntax-checked
(`gpu0.pbs`) after being moved there; the jobs of this log ran the same scripts from scratch.

**Things learned that are easy to trip over**
- **Compare GPU outputs after 2 steps, and against a baseline of the same compile variant.**
  Fresh compiles of unchanged code come in two variants (O27), 5e-14 apart after 2 steps and
  far apart in some patches after a day. A change is fine if it is ~1e-16 from *one* of the two
  baselines after 2 steps (`save_base_a2.npz`, `save_base_b2.npz`). A fresh compile needs an
  empty cache directory (the `save` and `prof` modes set one).
- **nsys does not show kernels inside XLA's command buffers** (on by default). For a profile
  with every kernel, use `XF=nocb`; it runs ~7 % slower, so time with the defaults.
- **Kernel names map to the source** through the `op_name` metadata in
  `*gpu_after_optimizations.txt` of the XLA dump (`attribute.py`).
- **`jax.checkpoint` costs nothing in forward runs** (JAX adds its barrier only when
  differentiating), and no XLA flag helped. Do not retry these.
- **Unrolling is not bit for bit on the CPU** even when the operations are the same (the
  unrolled table search: last bits; one field at 1.2 times the noise-twin allowance). So is
  moving code into a `lax.cond` branch (decision 116).
- **`qsub -v` splits on commas:** flag sets with commas are presets in `gpu0.pbs` (`XF=`);
  loop groups are joined with `+` (`U=`).
- **The probe jobs print nothing until a run ends** (their output goes through `grep`): an
  empty log means still compiling or running, not a failure.
- **Submit `jobs/program_gpu.pbs` and `jobs/grad_gpu.pbs` from scratch too:** PBS drops its
  `.o` file where `qsub` is run.
- **Do not write out the stomatal bisection** (20 halvings of one large kernel): the GPU
  program was still compiling after 34 minutes.
- **A loop written out is slower on the CPU** (the 21-patch box: 4 times) and slower to
  differentiate on the GPU (step 2).
- One step is 6 sub-steps × **5** physics evaluations (the plan's §2 says 4).
- `timing/time_vmap.py` needed `gridcell_count` for padded patches (fixed in `8af77c8`).

**Scratch** (Derecho purges unused scratch files; everything needed to resume is in the
repository, the rest is evidence): `$SCRATCH/clm-ml-jax/probe_gpu0/` (results, logs, saved
outputs, ~500 MB), `prof0/` (profiles, XLA dumps, per-tag caches; several GB, can be deleted),
`trees/gpu0_base`, `gpu1_search`, `gpu1_shared`, `gpu1_cond`, `gpu1_done`, `gpu2_unroll` (git worktrees: remove
with `git worktree remove PATH`, or `git worktree prune` after a purge).

## Step 0: measurements (2026-11-13). No change to the model

The maintainer agreed to the three proposals at the end of this section (decisions 114-116).

### Summary

1. **A global step takes 1.98 s: 1.21 s fixed cost and 0.76 s that grows with the patches.**
   The fixed cost is 61 % of the step, not the 20 % estimated from the Phase 6 numbers.
2. **The fixed cost is almost entirely the psihat table lookup.** The roughness-sublayer
   lookup (`turbulence.py::lookup_psihat`) finds its place in the table with
   `jnp.searchsorted`, which JAX runs as a small loop. These searches are 89 % of all loop
   iterations in a step and 94 % of all commands sent to the GPU.
3. **A prototype that only changes how the search is done runs the globe at 1.06 s per step
   (1.86×)**, with no other change. It was applied from the probe script; `src/` is untouched.
4. **After that change the run is limited by the GPU's own arithmetic** (0.99 s of kernel
   time in a 1.14 s step), and the remaining candidates are known and smaller.
5. The 2× target looks reachable with the table search fixed, the searches shared, and
   the padded steps skipped.

### 0a. The yardstick and its spread

Global domain (72 575 patches, 14 346 gridcells, layout 66 097 C3 + 6 478 C4 slots), eight
daily-mean output variables at patch and gridcell level, compile time excluded.

| Run | ms per step | patch-steps/s |
|---|---|---|
| 1-day chunk, job 1 (3 repeats) | 1990, 1987, 1985 | 36 560 |
| 1-day chunk, job 2 | 1975, 1975, 1974 | 36 770 |
| 1-day chunk, job 3 | 1982, 1979, 1978 | 36 690 |
| 10-day chunk (`timing/time_program.py`, `results_program.jsonl` "gpu0_ref") | 1949 | 37 240 |

- **Spread:** 0.1 % between repeats in a job, 0.6 % between jobs. A change of more than ~2 %
  is real.
- The 1-day chunk is a fair stand-in for the 10-day one (1.5 % apart).
- Compile 40 s (cold), generated code 0.01 MB, peak GPU memory 3.5 GB (1 day), 5.1 GB (10 days).
- The cost does not depend on the data: all solver loops have fixed lengths.

### 0b. Step time against the number of patches

Regional boxes of the same 1° domain, 8-step chunks.

| Patches (C3 + C4 slots) | ms per step |
|---|---|
| 21 (17 + 4) | 1212 |
| 801 (658 + 143) | 1205 |
| 4 740 (4 266 + 474) | 1219 |
| 18 313 | 1323 |
| 72 575 (66 097 + 6 478) | 1975 |
| 64, C3 only | 611 |

- **Fixed cost: 1.21 s per step** with both photosynthesis pathways present.
- **Cost per patch: ~10.5 µs** per patch-step (0.76 s at the global size).

### 0f. C3 and C4 groups (hypothesis H4: holds)

A box with only C3 patches costs 611 ms per step; one with both costs 1212 ms. The step is
traced once per pathway (decision 22), so **the C4 group doubles the fixed cost**, 0.6 s per
step, for 9 % of the patches. Once the table search is fixed this falls to ~0.1 s (below),
so decision 22 does not need to be revisited for this target.

### 0g. Checkpointing in forward runs (as predicted: no cost)

Global, without `checkpoint_steps` and without the sub-step checkpoint: 1975 ms per step,
against 1974–1985 with them. No difference.

### Loop census: which loops a step runs

Counted from the traced program (`census.py`; static, both patch groups together). One step
is 6 sub-steps × **5** evaluations of the canopy physics (the four Runge–Kutta stages and one
more pass) = 30 evaluations, not 24 as the plan assumed.

| Loop | Loop runs per step | Iterations per step | Share |
|---|---|---|---|
| **psihat table search** (`jnp.searchsorted`, 6 or 9 iterations) | 16 320 | **122 400** | **89 %** |
| Longwave tridiagonal solve (42 levels, forward and back) | 120 | 4 980 | 3.6 % |
| Flux-profile block-tridiagonal solves (20 layers) | 240 | 4 680 | 3.4 % |
| Obukhov solver (25 iterations; each calls the table lookup) | 60 | 1 500 | 1.1 % |
| Stomatal bisection (20 halvings) | 60 | 1 200 | 0.9 % |
| Second bisection (`math.py::bisection_traceable`) | 60 | 1 200 | 0.9 % |
| Radiation sweeps, soil, sub-steps, stages | | ~1 200 | 0.9 % |
| **All loops** | | **137 141** | |

Why so many searches: one call of the RSL functions (`turbulence.py::rsl_psi`) looks up the
table four times (two heights × momentum and scalars), and each lookup runs two searches (one
per table axis). That call sits inside the Obukhov solver (25 iterations per evaluation) and
is made about nine more times per evaluation.

Three of every four of these searches repeat work already done:
- the momentum and scalar tables have **identical grids** (`psihat.nc`), so their searches
  give the same answer;
- one of the two heights is always the canopy top, `(z - h)/(h - d) = 0`: a constant.

So 2 searches per call would do instead of 8.

### 0d. Profiles (nsys, two global steps)

**Today's program** (`prof0/prof1_global`):

| Per step | Count | Time |
|---|---|---|
| Kernel launches | 189 000 | |
| Copies of loop-carried values (device to device) | 189 000 | |
| CPU time issuing these commands | | ~2.0 s (the whole step) |
| GPU time in kernels | | 0.66 s |

By kind (kernel names matched to the source through XLA's dump, `attribute.py`):

| Kind | Launches per step | GPU s per step |
|---|---|---|
| Loop counters (one tiny kernel per loop iteration) | 92 100 (49 %) | 0.17 |
| Table-search loop bodies | 86 300 (46 %) | 0.22 |
| Physics | 10 600 (6 %) | 0.27 |

The run is **limited by the CPU thread issuing commands** (~5 µs each), and 94 % of the
commands belong to the table searches. Hypotheses H1 (copies of carried values) and H2 (many
small kernels) hold, but for one reason only: the search loops. The copies are small
(26 GB per step in 189 000 copies): the count hurts, not the bytes.

**With the search loop removed** (prototype `scan_unrolled`, command buffers off so every
kernel is visible, `prof0/prof3_global`; 1.14 s per step in this configuration):

| Per step | Value |
|---|---|
| Kernel launches | 57 900 (was 189 000 with bundling on; see note) |
| Copies | 28 900, 21 GB |
| GPU time in kernels | **0.99 s** of the 1.14 s step |

Note: XLA bundles kernels into "command buffers" by default; nsys does not show kernels
inside a bundle, so the launch counts of the two profiles are not comparable one to one.

Where the 0.99 s of GPU time goes:

| What | GPU s per step | Share |
|---|---|---|
| Canopy physics of the Runge–Kutta stages (30 evaluations; kernels of 1–3 ms on patch × layer × leaf arrays) | 0.48 | 49 % |
| Solver loops inside a stage; of this the **stomatal bisection body: 0.23 s** (one fused kernel, 357 µs, 600 times per step) | 0.31 | 31 % |
| Table searches (still 16 200 launches) | 0.10 | 10 % |
| Outside the sub-steps (soil, output, forcing) | 0.08 | 8 % |

So after the fix the GPU is busy 87 % of the time with real arithmetic. H3 (a few heavy
kernels) holds for what is left: the bisection body is 22 % of it. H5 (layout): the stage
kernels work on arrays with the layer axis first (`[20, 66097, 2]`), as XLA arranged them
for the loops over layers; transposes are not among the top kernels. No evidence for a layout
change.

### Prototype: other search methods (plan "prototype first")

`jnp.searchsorted` has a `method` argument. The probe replaced the default (`scan`, a loop)
for the whole program, with no change in `src/`.

| Method | Global, ms per step | Speed-up | 21 patches, ms per step | CPU result (1 day, small box) | CPU speed |
|---|---|---|---|---|---|
| `scan` (today) | 1975 | 1.00× | 1212 | reference | 52 ms per step |
| `compare_all` (compare with every table entry, no loop) | 1156 | 1.71× | 171 | **bit for bit** | 56 (8 % slower) |
| `scan_unrolled` (the same binary search, written out) | **1062** | **1.86×** | 193 | differs in the last bits (largest absolute difference 1.4e-8 over the output sums and the state; noise twin not yet run) | 64 (24 % slower) |

- The fixed cost falls from 1.21 s to ~0.19 s (C3 only: 0.11 s).
- The cost per patch is unchanged (~12 µs): it is the physics.
- `scan_unrolled` returns the same table indices as `scan`. That its CPU result differs in
  the last bits means the compiler arranged the surrounding arithmetic differently. It must
  pass the noise-twin test before it is accepted.

### Step 1b in advance: XLA flags

With the `scan_unrolled` prototype, global:

| Flags | ms per step |
|---|---|
| default | 1062 |
| command buffers off | 1142 (7 % slower) |
| command buffers for loops, unrolled (Phase 6's "CBU") | 1076 (no gain) |

**No flag helps. The defaults stay.** Step 1b is closed.

### Not done in step 0

- **0c (why slower than Phase 6b):** not pursued. Today's reference (37 240 patch-steps/s) is
  3 % below the end of Phase 7 (38 400, JAX 0.9.2, two output variables instead of eight). The
  Phase 6b figure (42 700) was a different program. The table search dwarfs this.
- **0e (each process alone):** replaced by the profile matched to the source, which ranks the
  real program. Test beds for single processes are built when step 3 needs them.
- **Gradient baseline on the GPU:** not yet measured. It is the first job of step 1, before
  any change, so that every later step can report gradient time and memory against it.

### What this means for the plan

| Step | Change | Expected global step | Kind |
|---|---|---|---|
| | today | 1.98 s | |
| 1 | table search without a loop | ~1.06 s (measured in the prototype) | bit for bit (`compare_all`) or round-off (`scan_unrolled`) |
| 2 | one search per distinct question: 2 instead of 8 per RSL call | ~0.95 s (estimate: removes most of the 0.10 s of search time and ~12 000 launches) | same table indices |
| 3 | padded steps skipped (`lax.cond`) | no padding cost: 25 % of the January run | bit for bit |
| later | stomatal bisection body (0.23 s), stage physics (0.48 s), other loops unrolled | to be measured | round-off |

Steps 1–2 alone should meet the 2× target for steady running. The rest of the plan (unrolling
the layer loops, hot-spot rewrites) becomes optional: it is what a later, more ambitious
target would need.

### For the maintainer

1. **Which search method?**
   - `scan_unrolled`: fastest on the GPU (1.86×). Round-off-level change (noise twin,
     reference regenerated). 24 % slower on the CPU.
   - `compare_all`: 1.71× on the GPU. Bit for bit, no new reference. 8 % slower on the CPU.
   - Either can be a hidden static setting chosen by device (the plan's "unrolling is a
     setting"), so the CPU keeps today's loop and its bit-for-bit reference.

   Proposal: a setting; `scan_unrolled` on the GPU, today's `scan` on the CPU.
2. **Sharing the searches** needs the code to know that the momentum and scalar tables use
   the same grids. The tables are an input of the compiled program (other tables can be
   supplied). Proposal: store the two grids once in the table object and check at setup that
   a supplied table has one pair of grids (true of `psihat.nc`).
3. **Order:** the gradient baseline, then steps 1, 2, 3 above, each with its tests; then
   measure and decide whether to go on.

## Step 1: the table search, shared searches, padded steps (2026-11-13)

Code snapshots in `$SCRATCH/clm-ml-jax/trees/` (`gpu1_search`, `gpu1_shared`, `gpu1_cond`);
saved outputs `probe_gpu0/save_*.npz`; comparisons with `probe_gpu0/compare.py`.

### Results (1° globe, one A100, 1-day chunk)

| Code | ms per step | patch-steps/s | Speed-up | 21 patches, ms per step |
|---|---|---|---|---|
| Baseline (`8f44da5`) | 1975–1985 | 36 700 | 1.00× | 1212 |
| 1. Table search without a loop on GPUs (decision 114, `8af77c8`) | 1065 | 68 170 | 1.86× | 192 |
| 2. One search per grid in `psi_rsl` (decision 115, `a191453`) | 1057 | 68 670 | 1.87× | 175 |
| 3. Padded steps skipped (decision 116) | 1061 | 68 400 | 1.87× | |

- **Change 1** gives what the prototype gave.
- **Change 2 gave much less than estimated** (0.7 % at the global size, where 5–10 % was
  expected; 9 % of the fixed cost of a small domain). Probably the compiler already merged
  the repeated searches once they were no longer loops. Its global gain is at the edge of the
  measurement spread (0.6 %). It is kept: it helps small domains, changes no result, and the
  maintainer agreed to its condition on the tables.
- **Change 3:** a 48-step chunk with 24 real steps takes 25.4 s instead of 50.6 s; a full
  chunk costs the same (+0.4 %, within the spread). The January run (31 days in 10-day
  chunks) would no longer pay for 9 padded days: 25 % of its time.
- CPU speed is unchanged by all three (50 ms per step for a 21-patch box).

**Against the target:** steady running is at 1.87× (1.06 s per step; 2× is 0.99 s). For the
January run as it was run, with its padding, the average step goes from 2.57 s to about
1.06 s (2.4×).

### Tests

| Test | 1. search | 2. shared | 3. padded steps |
|---|---|---|---|
| Fast tier (unit + Fortran golden) | pass (445) | pass (447) | pass (447) |
| CPU bit for bit | pass | pass | **fails by design (73 of 96 fields, last bits); reference regenerated**, then passes |
| CPU noise twin | pass | pass | pass (before and after regenerating) |
| CPU, small 1° box, one day | identical | identical | |
| Restart, padded chunks equal one chunk, output streams, devices | | | pass |
| CPU gradient tier (`tests/gradient_tier.py`) | | | pass (9, 6 min; all three changes in) |
| All of `tests/regression` (CPU) | | | pass (216, 14 min; all three changes in) |

On the CPU the search stays a loop (`auto`), so changes 1 and 2 leave the CPU arithmetic as it
was. With `unrolled` forced on the CPU, the regression case differs from the reference at
round-off level (largest relative difference 2.8e-11; one field, `stflx_air_canopy`, at 1.2
times the noise-twin allowance; decision 114).

### GPU results against the baseline

Two fresh compiles of the **same, unchanged** program on the GPU come in two variants (open
item O27). Saved global outputs and state, largest relative difference:

| Compared | After 2 steps | After 1 day |
|---|---|---|
| Baseline A against baseline B (same code, two compiles) | 5e-14 | 0.21 in the worst patch; 99.9 % of values below 4e-3 |
| Change 1 against the baseline of its variant | 4e-16 | 1.5e-13 |
| Change 2 against the baseline of its variant | 2e-16 | 99.9 % of values below 4e-16; worst patch 1.6e-2 |
| Change 3 against the baseline of its variant | 2e-16 | 1.4e-13 |

So each change moves the GPU results by one unit in the last place after two steps, far less
than the difference between two compiles of the unchanged code.

**For the maintainer (not caused by this work, but measured here):** the two compile variants
of the unchanged model, 5e-14 apart after two steps, are more than 1e-6 apart after one day
in 4 176 of the 72 575 patches (5.8 %), more than 1e-3 in 1 085 and more than 1 % in 189 (leaf
or air state). This is the round-off amplification of real 1° patches already recorded
(O27 and the "issue 10" item of decision 88), now counted on the globe. It limits how GPU runs
can be compared with each other, and it may deserve its own look.

### Gradients (GPU, `timing/time_vmap.py --grad`, CHATS7 copies, 8-step chunk)

| | Forward ms per step | Gradient ms per step | Gradient compile | Peak GB |
|---|---|---|---|---|
| Baseline, 1 patch | 616 | 1957 | 124 s | 0.004 |
| Baseline, 4 096 patches | 724 | 2577 | 117 s | 8.2 |
| After step 1 (`9c8a741`), 1 patch | 152 | 1037 | 103 s | 0.004 |
| After step 1, 4 096 patches | 268 | 1690 | 117 s | 8.2 |

Gradients are 1.9× faster for one patch and 1.5× for 4 096 patches, with the same memory and
compile time. (This test case has one patch group and 46 layers, so its forward gain, 4.1× and
2.7×, is larger than the globe's.) The GPU gradient tier (`jobs/grad_gpu.pbs`, `9c8a741`):
11 pass, 12 min.

### The 10-day reference chunk (`timing/time_program.py`, `results_program.jsonl`)

| | Chunk (480 steps) | patch-steps/s | Compile | Peak GB |
|---|---|---|---|---|
| Baseline ("gpu0_ref") | 935.5 s | 37 240 | 41 s | 5.1 |
| After step 1 ("gpu1_ref") | 508.5 s | 68 510 | 34 s | 5.1 |

A simulated year: about 5.2 h on one A100 (was 9.5 h).

### Where the step time is now, and what could come next

After step 1 a global step is about 0.18 s of fixed cost and 0.88 s of GPU arithmetic (from
the step-0 profile of the prototype): the canopy physics of the Runge–Kutta stages 0.48 s,
the stomatal bisection 0.23 s, the remaining table lookups 0.10 s, the rest 0.08 s.

To go from 1.87× to 2× the step must lose another 0.07 s. Candidates, by expected gain per
risk:

1. **Unroll the remaining loops** (plan step 2a: tridiagonal solves, flux-profile solves,
   Obukhov and bisection loops; ~15 000 loop iterations per step are left). Attacks the
   0.18 s of fixed cost. Same operations; code size and compile time to be watched.
2. **The stomatal bisection** (0.23 s): its five extra function evaluations stacked, or its
   body rearranged. Round-off-level.
3. **A fresh profile first** (the step-0 profile of the prototype had command buffers off):
   cheap, and it says which of 1 and 2 is worth more.

**Stopped here for the maintainer** (paused; resumed the same evening with step 2 below).

## Step 2a: the remaining loops (2026-11-13, evening)

Code snapshots `trees/gpu1_done` (`9c8a741`, profiles and prototypes) and `trees/gpu2_unroll`
(`c7f4a20`, the change); logs `probe_gpu0/log_prof4*.txt`, `log_u_*.txt`, `log_u2_*.txt`,
`log_s4*.txt`, `log_grad_step2.txt`; profiles `prof0/prof4_*`, `prof4n_*`.

### A fresh profile of the step-1 code

1068 ms per step with the default flags, 1141 ms with every kernel visible (`XF=nocb`). GPU
time per step, kernels matched to the source by loop depth and by how often they run:

| What | GPU s per step |
|---|---|
| Canopy physics of the Runge–Kutta stages (174 kernels, each run 30 times per step and group) | 0.43 |
| Stomatal bisection loop (one kernel of 357 µs for the C3 group, 600 times per step) | 0.25 |
| Obukhov loop, with its table lookups | 0.07 |
| Other table lookups | 0.07 |
| Tridiagonal and flux-profile loops | 0.02 |
| Loop counters (12 400 per step) | 0.02 |
| Outside the sub-steps (soil, output, forcing) | 0.07 |
| Copies of loop-carried values (29 000 per step, 21 GB) | 0.09 |

- **The copies are per-patch vectors** (25 600 of the 29 000 are 66 097 or 6 478 values of
  8 bytes): the carries of the Obukhov and layer loops, not of the stomatal bisection.
- **Five stage kernels take 0.26 s and are slow for their size:** 150–750 operations on
  patch × layer (× leaf) arrays in 1.0–2.8 ms, where the bisection kernel does 241 operations
  on the same number of values in 0.36 ms. They hold the slices, concatenations and updates
  that assemble and read the layer solves.
- The picture is the step-0 prototype's: nothing new moved in.

### Prototype: loop groups written out by the probe (`gpu0.py --unroll`, `src/` untouched)

"Written out" (unrolled): the compiled program holds the loop's iterations one after another,
without a counter, an end test and copies of the carried values; the Python source keeps one
loop (`unroll=` of `lax.scan` and `lax.fori_loop`). Globe, 1-day chunk; 21-patch box.

| Written out | Globe, ms per step | Speed-up | 21 patches, ms per step | Compile |
|---|---|---|---|---|
| Nothing (step 1) | 1062 | 1.87× | 189 | 38 s |
| Layer loops (tridiagonal and flux-profile solves, radiation sweeps, soil) | 727 | 2.72× | 98 | 72 s |
| Obukhov and roughness-length iterations | 1018 | 1.95× | 142 | 77 s |
| Layer loops, Obukhov and roughness length | **695** | **2.85×** | 67 | 113 s |
| Stomatal bisection, alone or with the others | not obtained | | | no result after 34 min (jobs cancelled) |

- **The layer loops gave 0.34 s**, far more than the 0.05–0.1 s expected from the copies and
  counters. Probably the compiler now merges the layer sweeps with the physics around them
  (the slow stage kernels above); a profile of the unrolled program would show it. Not done.
- On the CPU the fully written-out 21-patch box compiled in 60 s instead of 23 s and ran at
  190 ms per step instead of 50: the CPU keeps its loops.

### The change (decision 117, `c7f4a20`)

`numerics.solver_loops` (hidden; `auto`, `loop`, `unrolled`; `auto`: `unrolled` on a GPU,
`loop` on the CPU), passed down as a static keyword to the shortwave sweeps
(`solar_radiation.py`), the longwave solve (`math.py::tridiag`), the flux-profile solve
(`math.py::tridiag_2eq`), the Obukhov iteration and the roughness-length bisection
(`turbulence.py`). The stomatal solvers, sub-steps, stages, steps and the soil temperature
solve (once per step; written out in the prototype, a loop in the change) stay loops.

| 1° globe, one A100 | ms per step | patch-steps/s | Speed-up | Compile | Peak GB |
|---|---|---|---|---|---|
| Baseline (`8f44da5`) | 1975–1985 | 36 700 | 1.00× | 40 s | 3.5 |
| Step 1 (`9c8a741`) | 1057–1065 | 68 400 | 1.87× | 35–40 s | |
| Step 2a (`c7f4a20`), 1-day chunk | **696** | **104 300** | **2.85×** | 107–112 s | |
| Step 2a, 10-day chunk ("gpu2_ref") | 696 (334.0 s) | 104 300 | 2.80× of 935.5 s | 110 s | 4.7 |

- A 21-patch box: 69 ms per step (step 1: 175; baseline: 1212).
- A 48-step chunk with 24 real steps: 16.7 s (a full one: 33.4 s).
- A simulated year: about 3.4 h on one A100 (step 1: 5.2 h; baseline: 9.5 h).

### Tests

| Test | Result |
|---|---|
| Fast tier (unit + Fortran golden) | pass (452; new: written-out solves and iterations equal the loops) |
| CPU bit for bit, noise twin | pass (the CPU program is unchanged: `auto` is `loop`) |
| CPU with `unrolled` forced (`noise_cpu.py solver_loops`) | passes the noise-twin criterion: largest difference 0.76 times the allowance, largest relative difference 5e-11, 32 of 96 fields bit for bit |
| CPU gradient tier | pass (9) |
| All of `tests/regression` (CPU) | pass (216) |
| GPU gradient tier (`jobs/grad_gpu.pbs`, `c7f4a20`) | pass (11), but 46 min instead of 12: the gradient programs compile about four times longer |

### GPU results against the baselines

Largest relative difference of the saved global outputs and state:

| Compared | After 2 steps | After 1 day |
|---|---|---|
| Prototype, all three groups, against baseline A | 3.7e-16 | |
| Step 2a against step 1 (`save_s3_48`) | | 5.1e-16 |
| Step 2a against baseline B | | 1.4e-13 |
| Step 2a, another compile (`save_s4_2`), against baselines A and B | 4.4e-8 in the worst value; 99.9 % of values below 1.3e-13 | |
| Prototype, layer loops only, against baselines A and B | 4.4e-8; 99.9 % below 1.0e-13 | |

So the written-out program has compile variants too (O27). One agrees with the step-1
code to one unit in the last place after a whole day. Another (met twice: `save_s4_2` and
the layers-only prototype, 4.6e-11 apart) differs after two steps in 199 of the 72 575
patches by more than 1e-12 (90 values above 1e-10, the largest 4.4e-8, in a leaf state):
larger than the 5e-14 between the two variants of the baseline. It is round-off amplified
within two steps in a few patches, as O27 describes over a day; not investigated further.

### Gradients (GPU, `timing/time_vmap.py --grad`, CHATS7 copies, 8-step chunk)

| | Forward ms per step | Gradient ms per step | Gradient compile | Peak GB |
|---|---|---|---|---|
| Step 1, 1 patch | 152 | 1037 | 103 s | 0.004 |
| Step 1, 4 096 patches | 268 | 1690 | 117 s | 8.2 |
| Step 2a, 1 patch | 41 | **1519** | **419 s** | 0.004 |
| Step 2a, 4 096 patches | 128 | **2197** | **428 s** | 7.9 |

**The forward pass is 2–4× faster, the gradient 1.3–1.5× slower, and the gradient compile
4× longer.** The layer loops are differentiated (the Obukhov and roughness-length loops are
behind `stop_gradient`), and the derivative of a written-out loop is a larger program than
the derivative of a loop. Against the baseline before the GPU work (1957 and 2577 ms) the
gradient is still faster. With `numerics.solver_loops: loop` in `model.yaml` a gradient
program is step 1's. What `auto` should mean for gradient programs is open ("Resume here").

**Stopped here for the maintainer.**
