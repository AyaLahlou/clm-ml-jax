"""Run the tests on CPU unless ``JAX_PLATFORMS`` chooses otherwise.

The regression reference was recorded on CPU, and the test tiers' timings assume CPU. Derecho's
login nodes have GPUs (derecho7: two A100s), where JAX would otherwise default to the GPU: the
bit-for-bit test then fails by round-off (75 of 88 fields) and compiling for the
GPU makes the fast tier ~70x slower. GPU test jobs set ``JAX_PLATFORMS=cuda``
(``jobs/grad_gpu.pbs``). Set here, before any test initializes a JAX backend; the model's
worker subprocesses inherit it.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
