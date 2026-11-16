# GPU probes

Measurement scripts of the GPU optimization (O55, decisions 113-118). They are **not part of
the model and not tested**: they record how the numbers in
[`docs/design/gpu_progress.md`](../../docs/design/gpu_progress.md) and
[`docs/design/gradient_progress.md`](../../docs/design/gradient_progress.md) were measured, so
they can be measured again. How to run them, step by step: `gpu_progress.md`, "Resume here".

| File | What it measures | Where it runs |
|---|---|---|
| `gpu0.pbs`, `gpu0.py` | time per step of the compiled chunk (1° globe), saved outputs, profiles, prototypes of written-out loops | one A100 (PBS job) |
| `grad0.pbs`, `grad0.py` | forward chunk, forward-mode derivative and reverse-mode gradient of a CHATS7 chunk | one A100 (PBS job) |
| `compare.py` | the differences between two saved outputs of `gpu0.py` | login node |
| `attribute.py` | GPU time of a profile by what each kernel does | login node |
| `census.py` | every loop in one model step: how often it runs and what it carries | CPU |
| `census_grad.py`, `loops_grad.py` | the operations of the forward and gradient programs, by source file and by loop | CPU |
| `noise_cpu.py` | the regression case with another table search or written-out loops, against the committed reference (noise-twin criterion) | CPU |

They assume NCAR Derecho: the PBS account and queue at the top of the `.pbs` files, the
checkout in `/glade/work/$USER/clm-ml-jax` (or `REPO=...`), the environment in
`/glade/work/$USER/conda-envs/clm-ml-jax`, results in `$SCRATCH/clm-ml-jax/`. The jobs time
a fixed snapshot of the code (a git worktree in scratch, `TREE=...`), never the working
tree. The snapshot names in the logs (`trees/gpu0_base`, `gpu1_done`, `gpu2_unroll`, ...)
are scratch worktrees of the commits listed in `gpu_progress.md`; make them again with
`git worktree add --detach $SCRATCH/clm-ml-jax/trees/NAME COMMIT`.
