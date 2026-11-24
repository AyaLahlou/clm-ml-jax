"""
End-to-end regression: a 50-step CHATS7 run compared with the committed reference.

Two checks:

- ``test_bit_identical_to_reference``: code changes that should not change the arithmetic must
  reproduce the reference exactly.
- ``test_within_noise_of_reference``: every change must stay within the model's own
  sensitivity to round-off, measured by two noise twins (``leaf_emissivity × (1 ± 1e-12)``).
  A field passes if its largest difference from the reference is at most ``NOISE_FACTOR``
  times the twins' largest difference (plus a 1e-12 relative floor).

A change that legitimately alters round-off (a traceable implementation, compilation, GPU)
passes the noise test and fails the bit-for-bit test; regenerate the reference with
``python tests/regression/record_run.py --update-reference`` and say so in the commit.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.regression.record_run import REFERENCE_FILE

pytestmark = pytest.mark.regression

NOISE_FACTOR = 2.0
FLOOR = 1e-12  # relative to the field's magnitude, for fields the twins leave unchanged


def _fields():
    with np.load(REFERENCE_FILE) as data:
        return [k for k in data.files if not k.startswith("meta.")]


FIELDS = _fields() if REFERENCE_FILE.exists() else []


def _valid(values):
    return np.isfinite(values) & (np.abs(values) < 1e30)  # skip spval (1e36) fill values


def test_reference_holds_every_step_field(current_fields):
    """The recorded fields are exactly the reference's: every step field
    (``clm_ml.outputs.step_fields``: diagnostics, prognostic state, layer grid), none missing,
    none new."""
    assert sorted(current_fields) == sorted(FIELDS)


@pytest.mark.parametrize("field", FIELDS)
def test_bit_identical_to_reference(field, reference, current_fields):
    """Every real value is bit-identical. Fill values (|x| >= 1e30, e.g. spval sums in unused
    level 0 or leaf index 0) must be in the same places, but their last bits are meaningless."""
    ref, new = reference[field], current_fields[field]
    if not np.issubdtype(ref.dtype, np.floating):
        np.testing.assert_array_equal(new, ref, err_msg=field)
        return
    np.testing.assert_array_equal(_valid(new), _valid(ref), err_msg=f"{field}: fill pattern")
    valid = _valid(ref)
    np.testing.assert_array_equal(new[valid], ref[valid], err_msg=field)


@pytest.mark.parametrize("field", FIELDS)
def test_within_noise_of_reference(field, reference, twins, current_fields):
    ref = reference[field].astype(float)
    new = current_fields[field].astype(float)
    assert new.shape == ref.shape, f"{field}: shape {new.shape} != reference {ref.shape}"
    valid = _valid(ref)
    if np.issubdtype(reference[field].dtype, np.integer):
        np.testing.assert_array_equal(new, ref, err_msg=field)
        return
    noise = max(
        float(np.max(np.abs(twin[field].astype(float)[valid] - ref[valid]), initial=0.0))
        for twin in twins.values()
    )
    scale = float(np.max(np.abs(ref[valid]), initial=0.0))
    allowed = NOISE_FACTOR * noise + FLOOR * scale
    diff = float(np.max(np.abs(new[valid] - ref[valid]), initial=0.0))
    nonfinite_mismatch = not np.array_equal(np.isfinite(new), np.isfinite(ref))
    assert not nonfinite_mismatch, f"{field}: NaN/inf pattern differs from reference"
    assert diff <= allowed, (
        f"{field}: max difference {diff:.3e} exceeds {NOISE_FACTOR} x twin noise "
        f"{noise:.3e} (+ floor {FLOOR * scale:.3e})"
    )
