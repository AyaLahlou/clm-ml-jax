"""Fortran's conservation checks: recorded in the step instead of a host callback
inside compiled code, and checked on the host; a run stops when a tolerance is exceeded."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from clm_ml.checks import BALANCE_CHECKS, LONGWAVE, NITROGEN, ModelError, check_balance  # noqa
from clm_ml.step import _record  # noqa: E402


def test_records_the_largest_error_per_check():
    errors = jnp.zeros(len(BALANCE_CHECKS))
    errors = _record(errors, LONGWAVE, jnp.array([1e-5, -3e-4]))
    errors = _record(errors, LONGWAVE, 2e-4)  # smaller: maximum kept
    assert float(errors[LONGWAVE]) == pytest.approx(3e-4)
    assert float(errors.sum()) == pytest.approx(3e-4)


def test_check_stops_above_the_tolerance_only():
    errors = jnp.zeros((5, 2, len(BALANCE_CHECKS)))  # e.g. (step, patch, check)
    check_balance(errors.at[3, 1, LONGWAVE].set(9e-4), "test")  # below 1e-3
    with pytest.raises(ModelError, match="canopy nitrogen"):
        check_balance(errors.at[3, 1, NITROGEN].set(2e-6), "test")  # above 1e-6
