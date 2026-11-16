# Timing tests

Baseline for the modernization plan (Phase 0). Later results are not copied here: each timing
script appends one JSON line per measurement to its file in [`timing/`](../../timing/)
(`results.jsonl`: `time_offline.py`, whole runs; `results_model.jsonl`: `time_model.py`, the
compiled chunk and its gradient; `results_vmap.jsonl`: `time_vmap.py`, batched patches;
`results_program.jsonl`: `time_program.py`, GPU program size and compile time), with a label
naming the phase or commit; the progress logs quote the comparisons that mattered.

**Gradient times recorded before 2026-11-14 are too high** (decision 118): the scripts timed
the second gradient call, which is several times slower than the calls after it. From that
date `gradient_steady_*` is the fastest of the third and fourth call, and the second call is
recorded as `gradient_second_s`.

A timing test measures wall-clock run time only. It does not check results (that is the
regression test's job), runs on demand rather than in CI, and is only comparable between runs
on the same machine.

## Phase 0 baseline (before any refactoring)

Case: CHATS7, 2007-05-01 00Z, 50 half-hour steps (`nl.CHATS7.50steps`), warm JAX
compilation cache.

| Machine | Device | Startup | First step | Steady step (median) | Total | Per simulated day |
|---|---|---|---|---|---|---|
| Apple Silicon Mac (Darwin arm64), Python 3.10.9, JAX 0.6.2 | CPU | 2.3 s | 23.3 s | 4.46 s (4.39–4.74) | 246 s | **214 s** |
| GPU | — | — | — | — | — | *not yet measured* |

A separate timing of the same case via `clm-ml-offline` gave 251 s total, with user CPU time
roughly equal to wall time (256 s vs 251 s). The model is effectively running on **one core**.

### What the numbers say

- **Almost all the time is inside `ModelAdvance`** (4.4 of 4.5 s per step). Reading forcing takes
  0.01 s and writing output 0.06 s per step, so I/O is not the bottleneck today.
- **Each step costs about the same as every other step.** The per-step work is dispatched from
  Python sub-step by sub-step, rather than compiled as a whole. That is what Phase 4
  (`jit` of a `lax.scan` over the whole run) is meant to fix.
- **The first step costs about 5 steady steps** (23 s), mostly JIT compilation of the jitted
  helpers. `main.py` notes that a cold cache costs ~290 s. Measure that with
  `--cold-cache` when comparing compile times in Phase 4.

## How to add a row

```bash
python timing/time_offline.py                # warm cache
python timing/time_offline.py --cold-cache   # includes full compilation
```

Then add a row to the table above, with the phase it was measured at.
