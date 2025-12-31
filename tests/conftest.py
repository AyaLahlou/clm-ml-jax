"""
Pytest configuration and shared fixtures for CLM-JAX tests.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import sys

# Add src directory to Python path so tests can import modules
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(autouse=True)
def jax_config():
    """Configure JAX for testing."""
    # Use CPU for tests by default
    jax.config.update("jax_platform_name", "cpu")
    # Enable double precision for more accurate tests
    jax.config.update("jax_enable_x64", True)
    yield
    # Reset config after test
    jax.config.update("jax_enable_x64", False)


@pytest.fixture
def sample_grid():
    """Provide a sample grid for testing."""
    return {
        'begp': 1,
        'endp': 10,
        'begc': 1, 
        'endc': 5,
        'begg': 1,
        'endg': 2,
        'maxpatch_pft': 17,
        'nlevgrnd': 25,
        'nlevsoi': 10,
        'nlevlak': 10
    }


@pytest.fixture
def sample_arrays():
    """Provide sample arrays for testing."""
    key = jax.random.PRNGKey(42)
    return {
        'temperature': jax.random.normal(key, (25,)) * 10 + 273.15,
        'moisture': jax.random.uniform(key, (10,)) * 0.5,
        'pressure': jax.random.uniform(key, (25,)) * 1000 + 101325,
    }


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return PROJECT_ROOT / "tests" / "data"


# Custom markers for different test types
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning"),
]