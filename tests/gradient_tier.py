"""Run the gradient tier (tests marked ``gradient``) on CPU.

With JAX 0.9.2, compiled CPU gradients of a CHATS7 chunk segfaulted in LLVM-optimized code
(issue 16, O25), and this script turned LLVM optimization off for them. JAX 0.10.2 computes
them with the default optimization (rechecked 2026-11-02, cold compilation cache; decision
105), so the script now only forces the CPU (a login node's GPUs would otherwise be used)::

    python tests/gradient_tier.py            # all gradient tests (19, ~15 min on a login node)
    python tests/gradient_tier.py --focused  # the chunk derivative tests only (~6 min), after
                                             # moving a routine with a hand-written gradient

Extra arguments go to pytest. On GPU, run the same tests without this script
(``jobs/grad_gpu.pbs``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOCUSED = "tests/test_model_api.py"  # the chunk derivative tests (reverse and forward mode)
# Files holding the tests marked ``gradient`` (named, not collected from all of tests/: other
# modules set legacy module globals at import)
GRADIENT_FILES = [
    "tests/test_model_api.py",
    "tests/test_differentiability.py",
    "tests/test_gradient_accuracy.py",
    "tests/test_gradient_accuracy_grid.py",
]


def main(argv: list[str]) -> int:
    focused = "--focused" in argv
    extra = [a for a in argv if a != "--focused"]
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    targets = [FOCUSED] if focused else GRADIENT_FILES
    # One pytest process per file: with a cold compilation cache, XLA aborted while
    # serializing a newly compiled chunk in a process that had run another file's tests
    # (test_gradient_accuracy.py after test_differentiability.py; O78)
    code = 0
    for target in targets:
        cmd = [sys.executable, "-m", "pytest", target, "-m", "gradient"]
        cmd += ["-p", "no:cacheprovider", "-o", "addopts=", "-q", *extra]
        print("JAX_PLATFORMS=cpu", " ".join(cmd), flush=True)
        code = max(code, abs(subprocess.run(cmd, cwd=ROOT, env=env).returncode))
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
