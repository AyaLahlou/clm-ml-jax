"""
Pytest test configuration file.

Pytest fixtures that may need to be shared across multiple test files should
be defined here. See:

https://docs.pytest.org/en/stable/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files)
"""

# IMPORTANT: Enable JAX 64-bit precision BEFORE any imports that use JAX
import jax
jax.config.update("jax_enable_x64", True)

from pathlib import Path

import numpy as np
import pytest
import jax.numpy as jnp
import xarray as xr


@pytest.fixture(autouse=True)
def fix_random_seeds():
    """Fixture to fix the random seeds for reproducibility."""
    seed = 7
    np.random.seed(seed)


@pytest.fixture(scope="session")
def bomex_dataset():
    """Provides the BOMEX NetCDF file to tests."""
    tests_dir = Path(__file__).parent
    path = tests_dir / "data" / "BOMEX_pruned_thinned.nc"

    if not path.exists():
        raise FileNotFoundError(f"BOMEX dataset not found at '{path}'")

    return xr.open_dataset(path)


@pytest.fixture(scope="session")
def bomex_mixing_length():
    """
    Provides reference `compute_mixing_length` evaluations from Fortran code.

    The data was obtained by running CLUBB standalone calculation and logging
    input/outputs on each invocation of the Fortran function from:

    https://github.com/m2lines/clubb_ML/blob/master/src/CLUBB_core/mixing_length.F90

    Each input/output is stored under a key that matches its Fortran variable
    name. 'samples' dimension refers to individual, logged invocations of the
    function.

    The modified CLUBB with data logging is available here:

    https://github.com/m2lines/clubb_ML/tree/log-mixing-length

    """
    path_prefix = Path(__file__).parent / "sample_data"
    path = path_prefix / "bomex_mixing_length_calculation_samples.nc"

    if not path.exists():
        raise FileNotFoundError(
            f"BOMEX mixing length reference input not found at '{path}'"
        )

    return xr.open_dataset(path)


@pytest.fixture(scope="session")
def test_files_dir():
    """Provides the path to the local test files directory."""
    return Path(__file__).parent / "data"
