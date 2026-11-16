"""Kernel launches and GPU kernel time of an nsys profile, by what the kernel does (from the
op_name metadata of XLA's optimized HLO dump). Usage: attribute.py PROFILE_PREFIX HLO_DIR"""

import collections
import csv
import glob
import re
import sys

prefix, hlo = sys.argv[1], sys.argv[2]
f = max(glob.glob(hlo + "/*jit_run_chunk*gpu_after_optimizations.txt"), key=lambda p: len(p))
meta = {}
pat = re.compile(r"^\s*(?:ROOT )?%?([\w.\-]+) = (\S+).*?metadata=\{op_name=\"([^\"]*)\"")
for line in open(f):
    m = pat.match(line)
    if m:
        meta[m.group(1).replace(".", "_")] = (m.group(3), m.group(2))


def group(name):
    op, shape = meta.get(name, ("?", "?"))
    depth = op.count("while/body")
    if "searchsorted" in op:
        return "table search (binary search body)"
    if op.endswith("/add") and shape.startswith("s64[]"):
        return "loop counters"
    if shape.startswith(("s64[]", "s32[]", "pred[]", "u32[]")):
        return "other scalars (loop control)"
    return f"physics, loop depth {depth}"


rows = list(csv.DictReader(open(prefix + "_cuda_gpu_kern_sum.csv")))
agg = collections.defaultdict(lambda: [0, 0.0])
for r in rows:
    g = group(r["Name"])
    agg[g][0] += int(r["Instances"])
    agg[g][1] += int(r["Total Time (ns)"]) / 1e9
n = sum(a[0] for a in agg.values())
t = sum(a[1] for a in agg.values())
print(f"{'launches/step':>13} {'%':>5} {'GPU s/step':>10} {'%':>5}  group   (2 profiled steps)")
for g, a in sorted(agg.items(), key=lambda x: -x[1][0]):
    print(f"{a[0] // 2:>13} {100 * a[0] / n:5.1f} {a[1] / 2:10.3f} {100 * a[1] / t:5.1f}  {g}")
print("unmapped kernels:", sum(1 for r in rows if r["Name"] not in meta), "of", len(rows))
