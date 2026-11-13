# GPU optimization: progress log

Plan: [`gpu_optimization_plan.md`](gpu_optimization_plan.md) (O55, decision 113). Target: 2×,
1.0 s per 30-min step for the 1° globe on one A100.

Where things are:
- Probes, job script and logs: `$SCRATCH/clm-ml-jax/probe_gpu0/` (`gpu0.py`, `gpu0.pbs`,
  `census.py`, `attribute.py`, `results_gpu0.jsonl`, `log_*.txt`).
- Profiles and compiler dumps: `$SCRATCH/clm-ml-jax/prof0/`.
- Code snapshot the jobs ran: `$SCRATCH/clm-ml-jax/trees/gpu0_base` (commit `8f44da5`).
- Jobs: `develop` queue, account `UCLB0069`, one A100 (40 GB), 16 cores, JAX 0.10.2.

## Step 0: measurements (2026-11-13). No change to the model

**Stopped here for the maintainer:** the order of steps 1–3 and the choices in "For the
maintainer" below.

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
