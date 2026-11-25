"""Inspect historical CHATS7 output, with matched keys and -999 mask."""
import argparse
import json
from pathlib import Path

import numpy as np

SCHEMAS = {
    "flux": (18, 1, 1),
    "aux": (6, 0, 1),
    "fsun": (32, 0, 1),
    "soiltemp": (21, 1, 1),
    "profile": (28, 2, 46),
    "fluxprofile": (13, 2, 46),
}


def compare_files(ref_file, jax_file, columns, key_columns, layers, steps, ref_steps):
    ref = np.loadtxt(ref_file, ndmin=2, comments=None)
    jax = np.loadtxt(jax_file, ndmin=2, comments=None)
    if ref.shape != (ref_steps * layers, columns) or jax.shape != (steps * layers, columns):
        raise ValueError(f"Invalid dimensions for {ref_file.name}: {ref.shape=} {jax.shape=}")
    if not np.isfinite(ref).all() or not np.isfinite(jax).all():
        raise ValueError(f"Non-finite values in {ref_file.name}")
    ref = ref[: len(jax)]
    if key_columns and not np.array_equal(ref[:, :key_columns], jax[:, :key_columns]):
        raise ValueError(f"Timestamp/height ordering mismatch in {ref_file.name}")
    if key_columns:
        times = ref[:, 0].reshape(steps, layers)
        if not np.all(times == times[:, :1]) or not np.all(np.diff(times[:, 0]) > 0):
            raise ValueError(f"Invalid time blocks in {ref_file.name}")
    if key_columns == 2:
        heights = ref[:, 1].reshape(steps, layers)
        if not np.all(np.diff(heights, axis=1) < 0):
            raise ValueError(f"Invalid layer ordering in {ref_file.name}")

    metrics = {}
    for index in range(key_columns, columns):
        ref_values = ref[:, index]
        jax_values = jax[:, index]
        mask = ref_values != -999.0
        if not np.array_equal(mask, jax_values != -999.0):
            raise ValueError(f"Missing-value mask mismatch in {ref_file.name} column {index}")
        if not mask.any():
            metrics[index] = {"compared": 0}
            continue
        ref_values = ref_values[mask]
        jax_values = jax_values[mask]
        error = np.abs(ref_values - jax_values)
        if np.ptp(ref_values) == 0.0 or np.ptp(jax_values) == 0.0:
            correlation = None
        else:
            correlation = float(np.corrcoef(ref_values, jax_values)[0, 1])
        metrics[index] = {
            "compared": int(mask.sum()),
            "max_absolute_error": float(error.max()),
            "mean_absolute_error": float(error.mean()),
            "root_mean_squared_error": float(np.sqrt(np.mean(np.square(error)))),
            "mean_absolute_error_over_peak": float(error.mean() / max(np.abs(ref_values).max(), 1e-6)),
            "pearson_r": correlation,
        }
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_directory", type=Path)
    parser.add_argument("jax_directory", type=Path)
    parser.add_argument("--reference-steps", required=True, type=int)
    parser.add_argument("--jax-steps", required=True, type=int)
    args = parser.parse_args()
    if not 0 < args.jax_steps <= args.reference_steps:
        parser.error("Expected positive lengths with reference steps >= JAX steps")
    result = {
        "reference_steps": args.reference_steps,
        "jax_steps": args.jax_steps,
        "prefix_comparison": args.reference_steps != args.jax_steps,
        "metrics": {},
    }
    for tag, (columns, key_columns, layers) in SCHEMAS.items():
        filename = f"CHATS7_2007-05_{tag}.out"
        result["metrics"][tag] = compare_files(
            args.reference_directory / filename,
            args.jax_directory / filename,
            columns,
            key_columns,
            layers,
            args.jax_steps,
            args.reference_steps,
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
