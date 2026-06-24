# GPU Profiling Tutorial — JAX + NVTX + Nsight


---

## What is a "GPU gap"?

A **GPU gap** is a stretch of time where the GPU sits completely idle while the CPU is still running. On a timeline it looks like a white space between green CUDA kernel bars. Every millisecond of gap is wasted throughput.

Common causes in JAX/Python code:

| Cause | What it looks like in Nsight |
|---|---|
| NumPy ↔ JAX array round-trip | `cudaMemcpyDeviceToHost` + `cudaMemcpyHostToDevice` pair |
| Python `for` loop over GPU work | Many tiny kernels with gaps in between |
| `jax.debug.print` / `.item()` inside JIT | Forced synchronization barrier |
| First-run XLA JIT compilation | A single multi-second gap at start |
| Insufficient parallelism (too-small batch) | GPU utilization < 50% throughout |

---

## Setup

```bash
# Install the nvtx Python bindings
pip install nvtx

# Install Nsight Systems (comes with CUDA toolkit, or download separately)
# https://developer.nvidia.com/nsight-systems

# Create the reports directory
mkdir -p profiling_tutorial/reports
```

---

## Step-by-step workflow

### 1. Annotate your code with NVTX ranges

NVTX ranges appear as colored bars in the Nsight timeline. They let you map kernel activity back to named Python functions.

```python
import nvtx

# Context-manager style (recommended)
with nvtx.annotate("solar_radiation", color="yellow"):
    result = solar_radiation(par)

# Push/pop style (useful when start/end are in different call frames)
nvtx.push_range("outer_loop", color="green")
...
nvtx.pop_range()
```

The script `01_profile_jax_kernels.py` annotates every major CLM sub-kernel so you can see exactly which phase owns each gap.

### 2. Capture a timeline with Nsight Systems

```bash
# Run from the repo root
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=profiling_tutorial/reports/clm_profile \
    python profiling_tutorial/01_profile_jax_kernels.py
```

Key flags:
- `--trace=cuda,nvtx,osrt` — capture CUDA API calls, NVTX annotations, and OS runtime (thread scheduling)
- `--output` — path for the `.nsys-rep` report file

### 3. Open in Nsight Systems GUI



**What to look for:**

1. **NVTX row** — find the `run_with_gaps` range (red). Zoom in to see the white gaps.
2. **CUDA row** — compare the density of green kernel bars inside `optimized_run` vs `run_with_gaps`.
3. **cudaMemcpy rows** — look for `D→H` / `H→D` pairs inside `gap: numpy_roundtrip`. These are the cause of the gap.
4. **Warmup** — the first `JIT compile` range will be very long. Everything after warmup should be much shorter.

### 4. Drill into a kernel with Nsight Compute

Once Nsight Systems tells you *which* kernel is slow, use Nsight Compute to find out *why*:

```bash
ncu --set full \
    --kernel-name-base function \
    --launch-skip 10 --launch-count 1 \
    --output profiling_tutorial/reports/clm_kernel \
    python profiling_tutorial/01_profile_jax_kernels.py

ncu-ui profiling_tutorial/reports/clm_kernel.ncu-rep
```

Key metrics in the Nsight Compute report:

| Section | What it tells you |
|---|---|
| GPU Speed Of Light | % of theoretical peak bandwidth/compute achieved |
| Memory Workload | L1/L2/HBM traffic — are you bandwidth-bound? |
| Compute Workload | SM utilization — are you compute-bound? |
| Warp State Statistics | % of warp cycles spent stalled (ideal: < 30%) |
| Source Counters | Per-line roofline — exact hotspot in the kernel |

---

## The two patterns demonstrated

### Bad pattern — NumPy round-trip

```python
# This creates a D→H copy, CPU work, then an H→D copy.
# The GPU is completely idle during the CPU work.
cpu_array = np.array(gpu_array)     # GPU stalls here
cpu_array = np.clip(cpu_array, 0, None)
gpu_array = jnp.array(cpu_array)    # re-upload
```

**Fix:** use `jnp.clip` instead — it stays on the GPU.

### Bad pattern — Python loop over columns

```python
# Each iteration dispatches one tiny kernel.
# The Python interpreter overhead between dispatches leaves the GPU idle.
results = []
for i in range(N_cols):
    results.append(my_jit_fn(data[i:i+1]))
```

**Fix:** use `jax.vmap` to vectorize over the batch dimension. All columns run in a single kernel launch.

```python
batched_fn = jax.vmap(my_fn)
result = batched_fn(data)   # one kernel, no loop, no gap
```

---

## Profiling the real CLM-ML code

To profile an actual simulation:

```bash
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=profiling_tutorial/reports/full_sim \
    python src/offline_executable/main.py < path/to/namelist.nml
```

Then add NVTX annotations around the calls you care about in:

- [src/clm_src_cpl/lnd_comp_nuopc.py](../src/clm_src_cpl/lnd_comp_nuopc.py) — `ModelAdvance` (top-level time step)
- [src/multilayer_canopy/MLCanopyFluxesMod.py](../src/multilayer_canopy/MLCanopyFluxesMod.py) — canopy flux driver
- [src/multilayer_canopy/MLLeafPhotosynthesisMod.py](../src/multilayer_canopy/MLLeafPhotosynthesisMod.py) — Farquhar model

---

## Quick reference

```bash
# Profile timeline
nsys profile --trace=cuda,nvtx --output=reports/out python script.py

# Open timeline
nsys-ui reports/out.nsys-rep

# Profile a single kernel (skip first 10 launches, capture 1)
ncu --set full --launch-skip 10 --launch-count 1 --output reports/kern python script.py

# Open kernel report
ncu-ui reports/kern.ncu-rep

# Check GPU utilization live (while script runs)
watch -n 0.5 nvidia-smi
```
