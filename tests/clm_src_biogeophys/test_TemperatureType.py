"""
Comprehensive pytest suite for TemperatureType module.

This test suite covers:
- Initialization functions (init_temperature_state, init_allocate_temperature, init_temperature)
- Temperature extraction functions (get_soil_temperature, get_snow_temperature)
- State update functions (update_temperature)
- Edge cases (minimum domains, extreme temperatures, boundary conditions)
- Physical realism (temperatures > 0K, realistic gradients)
- Array shapes and data types
"""

import pytest
import jax.numpy as jnp
import numpy as np
from collections import namedtuple
from typing import Optional

# Define namedtuples matching the module specification
BoundsType = namedtuple('BoundsType', ['begp', 'endp', 'begc', 'endc', 'begg', 'endg'])
TemperatureState = namedtuple('TemperatureState', ['t_soisno_col', 't_a10_patch', 't_ref2m_patch'])

# Constants from module specification
DEFAULT_INIT_TEMP = 273.15
DEFAULT_NLEVSNO = 5
DEFAULT_NLEVGRND = 15


# ============================================================================
# Mock Implementation (Replace with actual module imports)
# ============================================================================

def init_temperature_state(
    n_columns: int,
    n_patches: int,
    n_levtot: int,
    initial_temp: float = DEFAULT_INIT_TEMP
) -> TemperatureState:
    """Initialize temperature state with specified dimensions and initial temperature."""
    t_soisno_col = jnp.full((n_columns, n_levtot), initial_temp, dtype=jnp.float32)
    t_a10_patch = jnp.full((n_patches,), initial_temp, dtype=jnp.float32)
    t_ref2m_patch = jnp.full((n_patches,), initial_temp, dtype=jnp.float32)
    return TemperatureState(t_soisno_col, t_a10_patch, t_ref2m_patch)


def init_allocate_temperature(
    bounds: BoundsType,
    nlevsno: int = DEFAULT_NLEVSNO,
    nlevgrnd: int = DEFAULT_NLEVGRND
) -> TemperatureState:
    """Allocate temperature arrays filled with NaN."""
    n_columns = bounds.endc - bounds.begc + 1
    n_patches = bounds.endp - bounds.begp + 1
    n_levtot = nlevsno + nlevgrnd
    
    t_soisno_col = jnp.full((n_columns, n_levtot), jnp.nan, dtype=jnp.float32)
    t_a10_patch = jnp.full((n_patches,), jnp.nan, dtype=jnp.float32)
    t_ref2m_patch = jnp.full((n_patches,), jnp.nan, dtype=jnp.float32)
    return TemperatureState(t_soisno_col, t_a10_patch, t_ref2m_patch)


def init_temperature(
    bounds: BoundsType,
    nlevsno: int = DEFAULT_NLEVSNO,
    nlevgrnd: int = DEFAULT_NLEVGRND
) -> TemperatureState:
    """Initialize temperature state from bounds."""
    n_columns = bounds.endc - bounds.begc + 1
    n_patches = bounds.endp - bounds.begp + 1
    n_levtot = nlevsno + nlevgrnd
    return init_temperature_state(n_columns, n_patches, n_levtot)


def get_soil_temperature(
    temp_state: TemperatureState,
    column_idx: int,
    nlevsno: int = DEFAULT_NLEVSNO
) -> jnp.ndarray:
    """Extract soil temperatures (skip snow layers)."""
    return temp_state.t_soisno_col[column_idx, nlevsno:]


def get_snow_temperature(
    temp_state: TemperatureState,
    column_idx: int,
    nlevsno: int = DEFAULT_NLEVSNO
) -> jnp.ndarray:
    """Extract snow temperatures."""
    return temp_state.t_soisno_col[column_idx, :nlevsno]


def update_temperature(
    temp_state: TemperatureState,
    t_soisno_col: Optional[jnp.ndarray] = None,
    t_a10_patch: Optional[jnp.ndarray] = None,
    t_ref2m_patch: Optional[jnp.ndarray] = None
) -> TemperatureState:
    """Update temperature state with new values (immutable)."""
    new_t_soisno_col = t_soisno_col if t_soisno_col is not None else temp_state.t_soisno_col
    new_t_a10_patch = t_a10_patch if t_a10_patch is not None else temp_state.t_a10_patch
    new_t_ref2m_patch = t_ref2m_patch if t_ref2m_patch is not None else temp_state.t_ref2m_patch
    return TemperatureState(new_t_soisno_col, new_t_a10_patch, new_t_ref2m_patch)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_data():
    """Load test data from specification."""
    return {
        "nominal_cases": [
            {
                "name": "default_temp",
                "n_columns": 10,
                "n_patches": 25,
                "n_levtot": 20,
                "initial_temp": 273.15,
            },
            {
                "name": "warm_climate",
                "n_columns": 50,
                "n_patches": 100,
                "n_levtot": 20,
                "initial_temp": 298.15,
            },
            {
                "name": "cold_climate",
                "n_columns": 30,
                "n_patches": 60,
                "n_levtot": 20,
                "initial_temp": 253.15,
            },
        ],
        "edge_cases": [
            {
                "name": "minimum_domain",
                "n_columns": 1,
                "n_patches": 1,
                "n_levtot": 1,
                "initial_temp": 273.15,
            },
            {
                "name": "extreme_cold",
                "n_columns": 15,
                "n_patches": 30,
                "n_levtot": 20,
                "initial_temp": 173.15,
            },
            {
                "name": "extreme_hot",
                "n_columns": 15,
                "n_patches": 30,
                "n_levtot": 20,
                "initial_temp": 373.15,
            },
        ],
        "bounds_variations": [
            {
                "name": "small_domain",
                "bounds": BoundsType(0, 9, 0, 4, 0, 1),
                "nlevsno": 5,
                "nlevgrnd": 15,
            },
            {
                "name": "large_domain",
                "bounds": BoundsType(0, 999, 0, 499, 0, 99),
                "nlevsno": 5,
                "nlevgrnd": 15,
            },
            {
                "name": "offset_indices",
                "bounds": BoundsType(100, 149, 50, 74, 20, 29),
                "nlevsno": 5,
                "nlevgrnd": 15,
            },
        ],
        "layer_configurations": [
            {"name": "no_snow", "nlevsno": 0, "nlevgrnd": 15, "n_levtot": 15},
            {"name": "maximum_snow", "nlevsno": 5, "nlevgrnd": 15, "n_levtot": 20},
            {"name": "deep_soil", "nlevsno": 5, "nlevgrnd": 30, "n_levtot": 35},
            {"name": "minimal_layers", "nlevsno": 1, "nlevgrnd": 1, "n_levtot": 2},
        ],
    }


@pytest.fixture
def sample_bounds():
    """Standard bounds for testing."""
    return BoundsType(begp=0, endp=49, begc=0, endc=19, begg=0, endg=9)


@pytest.fixture
def sample_temp_state():
    """Sample temperature state with realistic vertical gradient."""
    # Create realistic temperature profile: cold at surface, warmer at depth
    n_columns = 3
    n_levtot = 20
    n_patches = 5
    
    # Column 0: Cold surface to moderate depth
    col0 = jnp.array([250.0, 252.0, 255.0, 258.0, 261.0, 265.0, 268.0, 270.0, 272.0, 273.0,
                      274.0, 275.0, 276.0, 277.0, 278.0, 279.0, 280.0, 281.0, 282.0, 283.0])
    
    # Column 1: Very cold surface to moderate depth
    col1 = jnp.array([245.0, 248.0, 251.0, 254.0, 257.0, 260.0, 263.0, 266.0, 269.0, 271.0,
                      273.0, 274.5, 276.0, 277.5, 279.0, 280.5, 282.0, 283.5, 285.0, 286.5])
    
    # Column 2: Moderate surface to warm depth
    col2 = jnp.array([255.0, 257.0, 259.0, 262.0, 265.0, 268.0, 271.0, 273.5, 275.0, 276.0,
                      277.0, 278.0, 279.0, 280.0, 281.0, 282.0, 283.0, 284.0, 285.0, 286.0])
    
    t_soisno_col = jnp.stack([col0, col1, col2])
    t_a10_patch = jnp.array([273.15, 275.0, 270.0, 268.0, 280.0])
    t_ref2m_patch = jnp.array([272.0, 274.5, 269.5, 267.0, 279.0])
    
    return TemperatureState(t_soisno_col, t_a10_patch, t_ref2m_patch)


# ============================================================================
# Test init_temperature_state
# ============================================================================

@pytest.mark.parametrize("case", [
    {"n_columns": 10, "n_patches": 25, "n_levtot": 20, "initial_temp": 273.15},
    {"n_columns": 50, "n_patches": 100, "n_levtot": 20, "initial_temp": 298.15},
    {"n_columns": 30, "n_patches": 60, "n_levtot": 20, "initial_temp": 253.15},
])
def test_init_temperature_state_shapes_nominal(case):
    """Test that init_temperature_state returns correct shapes for nominal cases."""
    state = init_temperature_state(
        case["n_columns"],
        case["n_patches"],
        case["n_levtot"],
        case["initial_temp"]
    )
    
    assert state.t_soisno_col.shape == (case["n_columns"], case["n_levtot"]), \
        f"t_soisno_col shape mismatch: expected {(case['n_columns'], case['n_levtot'])}, got {state.t_soisno_col.shape}"
    assert state.t_a10_patch.shape == (case["n_patches"],), \
        f"t_a10_patch shape mismatch: expected {(case['n_patches'],)}, got {state.t_a10_patch.shape}"
    assert state.t_ref2m_patch.shape == (case["n_patches"],), \
        f"t_ref2m_patch shape mismatch: expected {(case['n_patches'],)}, got {state.t_ref2m_patch.shape}"


@pytest.mark.parametrize("case", [
    {"n_columns": 10, "n_patches": 25, "n_levtot": 20, "initial_temp": 273.15},
    {"n_columns": 50, "n_patches": 100, "n_levtot": 20, "initial_temp": 298.15},
    {"n_columns": 30, "n_patches": 60, "n_levtot": 20, "initial_temp": 253.15},
])
def test_init_temperature_state_values_nominal(case):
    """Test that init_temperature_state initializes with correct temperature values."""
    state = init_temperature_state(
        case["n_columns"],
        case["n_patches"],
        case["n_levtot"],
        case["initial_temp"]
    )
    
    expected_temp = case["initial_temp"]
    
    assert jnp.allclose(state.t_soisno_col, expected_temp, atol=1e-6, rtol=1e-6), \
        f"t_soisno_col values incorrect: expected all {expected_temp}"
    assert jnp.allclose(state.t_a10_patch, expected_temp, atol=1e-6, rtol=1e-6), \
        f"t_a10_patch values incorrect: expected all {expected_temp}"
    assert jnp.allclose(state.t_ref2m_patch, expected_temp, atol=1e-6, rtol=1e-6), \
        f"t_ref2m_patch values incorrect: expected all {expected_temp}"


def test_init_temperature_state_edge_minimum_domain():
    """Test initialization with minimum possible domain size (1x1x1)."""
    state = init_temperature_state(n_columns=1, n_patches=1, n_levtot=1, initial_temp=273.15)
    
    assert state.t_soisno_col.shape == (1, 1), "Minimum domain t_soisno_col shape incorrect"
    assert state.t_a10_patch.shape == (1,), "Minimum domain t_a10_patch shape incorrect"
    assert state.t_ref2m_patch.shape == (1,), "Minimum domain t_ref2m_patch shape incorrect"
    
    assert jnp.allclose(state.t_soisno_col, 273.15, atol=1e-6), "Minimum domain values incorrect"


def test_init_temperature_state_edge_extreme_cold():
    """Test initialization with extreme cold temperature (173.15K = -100°C)."""
    state = init_temperature_state(n_columns=15, n_patches=30, n_levtot=20, initial_temp=173.15)
    
    assert jnp.all(state.t_soisno_col >= 0.0), "Temperature below absolute zero"
    assert jnp.allclose(state.t_soisno_col, 173.15, atol=1e-6), "Extreme cold temperature incorrect"
    assert jnp.allclose(state.t_a10_patch, 173.15, atol=1e-6), "Extreme cold t_a10_patch incorrect"


def test_init_temperature_state_edge_extreme_hot():
    """Test initialization with extreme hot temperature (373.15K = 100°C)."""
    state = init_temperature_state(n_columns=15, n_patches=30, n_levtot=20, initial_temp=373.15)
    
    assert jnp.allclose(state.t_soisno_col, 373.15, atol=1e-6), "Extreme hot temperature incorrect"
    assert jnp.allclose(state.t_a10_patch, 373.15, atol=1e-6), "Extreme hot t_a10_patch incorrect"
    assert jnp.allclose(state.t_ref2m_patch, 373.15, atol=1e-6), "Extreme hot t_ref2m_patch incorrect"


def test_init_temperature_state_dtypes():
    """Test that init_temperature_state returns correct data types."""
    state = init_temperature_state(n_columns=10, n_patches=25, n_levtot=20)
    
    assert state.t_soisno_col.dtype == jnp.float32, f"t_soisno_col dtype incorrect: {state.t_soisno_col.dtype}"
    assert state.t_a10_patch.dtype == jnp.float32, f"t_a10_patch dtype incorrect: {state.t_a10_patch.dtype}"
    assert state.t_ref2m_patch.dtype == jnp.float32, f"t_ref2m_patch dtype incorrect: {state.t_ref2m_patch.dtype}"


def test_init_temperature_state_physical_realism():
    """Test that all temperatures are physically realistic (> 0K)."""
    test_temps = [173.15, 273.15, 298.15, 373.15]
    
    for temp in test_temps:
        state = init_temperature_state(n_columns=10, n_patches=20, n_levtot=15, initial_temp=temp)
        
        assert jnp.all(state.t_soisno_col > 0.0), f"Temperature {temp}K: t_soisno_col below absolute zero"
        assert jnp.all(state.t_a10_patch > 0.0), f"Temperature {temp}K: t_a10_patch below absolute zero"
        assert jnp.all(state.t_ref2m_patch > 0.0), f"Temperature {temp}K: t_ref2m_patch below absolute zero"


# ============================================================================
# Test init_allocate_temperature
# ============================================================================

@pytest.mark.parametrize("bounds_case", [
    BoundsType(0, 49, 0, 19, 0, 9),
    BoundsType(0, 99, 0, 39, 0, 19),
    BoundsType(100, 149, 50, 74, 20, 29),
])
def test_init_allocate_temperature_shapes(bounds_case):
    """Test that init_allocate_temperature returns correct shapes."""
    state = init_allocate_temperature(bounds_case, nlevsno=5, nlevgrnd=15)
    
    expected_n_columns = bounds_case.endc - bounds_case.begc + 1
    expected_n_patches = bounds_case.endp - bounds_case.begp + 1
    expected_n_levtot = 20  # 5 + 15
    
    assert state.t_soisno_col.shape == (expected_n_columns, expected_n_levtot), \
        f"t_soisno_col shape mismatch for bounds {bounds_case}"
    assert state.t_a10_patch.shape == (expected_n_patches,), \
        f"t_a10_patch shape mismatch for bounds {bounds_case}"
    assert state.t_ref2m_patch.shape == (expected_n_patches,), \
        f"t_ref2m_patch shape mismatch for bounds {bounds_case}"


def test_init_allocate_temperature_nan_values():
    """Test that init_allocate_temperature fills arrays with NaN."""
    bounds = BoundsType(0, 49, 0, 19, 0, 9)
    state = init_allocate_temperature(bounds, nlevsno=5, nlevgrnd=15)
    
    assert jnp.all(jnp.isnan(state.t_soisno_col)), "t_soisno_col should be filled with NaN"
    assert jnp.all(jnp.isnan(state.t_a10_patch)), "t_a10_patch should be filled with NaN"
    assert jnp.all(jnp.isnan(state.t_ref2m_patch)), "t_ref2m_patch should be filled with NaN"


def test_init_allocate_temperature_layer_configurations(test_data):
    """Test init_allocate_temperature with various layer configurations."""
    bounds = BoundsType(0, 49, 0, 19, 0, 9)
    
    for config in test_data["layer_configurations"]:
        state = init_allocate_temperature(bounds, nlevsno=config["nlevsno"], nlevgrnd=config["nlevgrnd"])
        
        expected_n_levtot = config["nlevsno"] + config["nlevgrnd"]
        assert state.t_soisno_col.shape[1] == expected_n_levtot, \
            f"Layer config {config['name']}: expected {expected_n_levtot} layers, got {state.t_soisno_col.shape[1]}"


def test_init_allocate_temperature_edge_no_snow():
    """Test allocation with no snow layers (nlevsno=0)."""
    bounds = BoundsType(0, 9, 0, 4, 0, 1)
    state = init_allocate_temperature(bounds, nlevsno=0, nlevgrnd=15)
    
    assert state.t_soisno_col.shape == (5, 15), "No snow configuration shape incorrect"
    assert jnp.all(jnp.isnan(state.t_soisno_col)), "No snow arrays should be NaN"


def test_init_allocate_temperature_edge_minimal_layers():
    """Test allocation with minimal layer configuration (1 snow + 1 soil)."""
    bounds = BoundsType(0, 9, 0, 4, 0, 1)
    state = init_allocate_temperature(bounds, nlevsno=1, nlevgrnd=1)
    
    assert state.t_soisno_col.shape == (5, 2), "Minimal layers shape incorrect"


# ============================================================================
# Test init_temperature
# ============================================================================

def test_init_temperature_nominal(sample_bounds):
    """Test init_temperature with standard bounds."""
    state = init_temperature(sample_bounds, nlevsno=5, nlevgrnd=15)
    
    expected_n_columns = sample_bounds.endc - sample_bounds.begc + 1
    expected_n_patches = sample_bounds.endp - sample_bounds.begp + 1
    
    assert state.t_soisno_col.shape == (expected_n_columns, 20), "init_temperature shape incorrect"
    assert state.t_a10_patch.shape == (expected_n_patches,), "init_temperature patch shape incorrect"


def test_init_temperature_default_values(sample_bounds):
    """Test that init_temperature uses default temperature (273.15K)."""
    state = init_temperature(sample_bounds)
    
    # Should initialize with DEFAULT_INIT_TEMP
    assert jnp.allclose(state.t_soisno_col, DEFAULT_INIT_TEMP, atol=1e-6), \
        "init_temperature should use default temperature"


def test_init_temperature_bounds_variations(test_data):
    """Test init_temperature with various bounds configurations."""
    for bounds_case in test_data["bounds_variations"]:
        bounds = bounds_case["bounds"]
        state = init_temperature(bounds, nlevsno=bounds_case["nlevsno"], nlevgrnd=bounds_case["nlevgrnd"])
        
        expected_n_columns = bounds.endc - bounds.begc + 1
        expected_n_patches = bounds.endp - bounds.begp + 1
        expected_n_levtot = bounds_case["nlevsno"] + bounds_case["nlevgrnd"]
        
        assert state.t_soisno_col.shape == (expected_n_columns, expected_n_levtot), \
            f"Bounds case {bounds_case['name']}: incorrect shape"


# ============================================================================
# Test get_soil_temperature
# ============================================================================

def test_get_soil_temperature_extraction(sample_temp_state):
    """Test extraction of soil temperatures from column."""
    soil_temps = get_soil_temperature(sample_temp_state, column_idx=1, nlevsno=5)
    
    # Should extract layers 5-19 (15 soil layers)
    expected = sample_temp_state.t_soisno_col[1, 5:]
    
    assert soil_temps.shape == (15,), f"Soil temperature shape incorrect: {soil_temps.shape}"
    assert jnp.allclose(soil_temps, expected, atol=1e-6), "Soil temperature values incorrect"


def test_get_soil_temperature_realistic_gradient(sample_temp_state):
    """Test that extracted soil temperatures show realistic vertical gradient."""
    soil_temps = get_soil_temperature(sample_temp_state, column_idx=1, nlevsno=5)
    
    # Expected values from test data (column 1, layers 5-19)
    expected = jnp.array([260.0, 263.0, 266.0, 269.0, 271.0, 273.0, 274.5, 276.0, 
                          277.5, 279.0, 280.5, 282.0, 283.5, 285.0, 286.5])
    
    assert jnp.allclose(soil_temps, expected, atol=1e-6), \
        f"Soil temperature gradient incorrect.\nExpected: {expected}\nGot: {soil_temps}"


def test_get_soil_temperature_multiple_columns(sample_temp_state):
    """Test soil temperature extraction from multiple columns."""
    for col_idx in range(3):
        soil_temps = get_soil_temperature(sample_temp_state, column_idx=col_idx, nlevsno=5)
        
        assert soil_temps.shape == (15,), f"Column {col_idx}: incorrect shape"
        assert jnp.all(soil_temps > 0.0), f"Column {col_idx}: temperatures below absolute zero"


def test_get_soil_temperature_no_snow_layers():
    """Test soil temperature extraction when nlevsno=0 (no snow)."""
    state = init_temperature_state(n_columns=5, n_patches=10, n_levtot=15, initial_temp=280.0)
    soil_temps = get_soil_temperature(state, column_idx=2, nlevsno=0)
    
    # Should extract all layers
    assert soil_temps.shape == (15,), "No snow: should extract all layers"
    assert jnp.allclose(soil_temps, 280.0, atol=1e-6), "No snow: values incorrect"


# ============================================================================
# Test get_snow_temperature
# ============================================================================

def test_get_snow_temperature_extraction(sample_temp_state):
    """Test extraction of snow temperatures from column."""
    snow_temps = get_snow_temperature(sample_temp_state, column_idx=1, nlevsno=5)
    
    # Should extract layers 0-4 (5 snow layers)
    expected = sample_temp_state.t_soisno_col[1, :5]
    
    assert snow_temps.shape == (5,), f"Snow temperature shape incorrect: {snow_temps.shape}"
    assert jnp.allclose(snow_temps, expected, atol=1e-6), "Snow temperature values incorrect"


def test_get_snow_temperature_realistic_values(sample_temp_state):
    """Test that extracted snow temperatures are realistic (coldest at surface)."""
    snow_temps = get_snow_temperature(sample_temp_state, column_idx=1, nlevsno=5)
    
    # Expected values from test data (column 1, layers 0-4)
    expected = jnp.array([245.0, 248.0, 251.0, 254.0, 257.0])
    
    assert jnp.allclose(snow_temps, expected, atol=1e-6), \
        f"Snow temperature values incorrect.\nExpected: {expected}\nGot: {snow_temps}"
    
    # Snow should generally warm with depth (closer to soil)
    assert snow_temps[0] <= snow_temps[-1], "Snow should warm with depth"


def test_get_snow_temperature_multiple_columns(sample_temp_state):
    """Test snow temperature extraction from multiple columns."""
    for col_idx in range(3):
        snow_temps = get_snow_temperature(sample_temp_state, column_idx=col_idx, nlevsno=5)
        
        assert snow_temps.shape == (5,), f"Column {col_idx}: incorrect shape"
        assert jnp.all(snow_temps > 0.0), f"Column {col_idx}: temperatures below absolute zero"


def test_get_snow_temperature_single_layer():
    """Test snow temperature extraction with single snow layer."""
    state = init_temperature_state(n_columns=5, n_patches=10, n_levtot=16, initial_temp=265.0)
    snow_temps = get_snow_temperature(state, column_idx=2, nlevsno=1)
    
    assert snow_temps.shape == (1,), "Single snow layer: incorrect shape"
    assert jnp.allclose(snow_temps, 265.0, atol=1e-6), "Single snow layer: value incorrect"


# ============================================================================
# Test update_temperature
# ============================================================================

def test_update_temperature_full_update():
    """Test updating all fields in temperature state."""
    initial_state = init_temperature_state(n_columns=2, n_patches=3, n_levtot=5, initial_temp=273.15)
    
    new_t_soisno = jnp.full((2, 5), 280.0)
    new_t_a10 = jnp.full((3,), 285.0)
    new_t_ref2m = jnp.full((3,), 282.0)
    
    updated_state = update_temperature(initial_state, new_t_soisno, new_t_a10, new_t_ref2m)
    
    assert jnp.allclose(updated_state.t_soisno_col, 280.0, atol=1e-6), "Full update: t_soisno_col incorrect"
    assert jnp.allclose(updated_state.t_a10_patch, 285.0, atol=1e-6), "Full update: t_a10_patch incorrect"
    assert jnp.allclose(updated_state.t_ref2m_patch, 282.0, atol=1e-6), "Full update: t_ref2m_patch incorrect"


def test_update_temperature_partial_update():
    """Test partial update (only some fields modified)."""
    initial_state = init_temperature_state(n_columns=2, n_patches=3, n_levtot=5, initial_temp=273.15)
    
    new_t_a10 = jnp.array([280.0, 285.0, 275.0])
    
    updated_state = update_temperature(initial_state, t_a10_patch=new_t_a10)
    
    # t_a10_patch should be updated
    assert jnp.allclose(updated_state.t_a10_patch, new_t_a10, atol=1e-6), \
        "Partial update: t_a10_patch not updated correctly"
    
    # Other fields should remain unchanged
    assert jnp.allclose(updated_state.t_soisno_col, initial_state.t_soisno_col, atol=1e-6), \
        "Partial update: t_soisno_col should remain unchanged"
    assert jnp.allclose(updated_state.t_ref2m_patch, initial_state.t_ref2m_patch, atol=1e-6), \
        "Partial update: t_ref2m_patch should remain unchanged"


def test_update_temperature_immutability():
    """Test that update_temperature returns new state (immutable)."""
    initial_state = init_temperature_state(n_columns=2, n_patches=3, n_levtot=5, initial_temp=273.15)
    
    new_t_a10 = jnp.array([280.0, 285.0, 275.0])
    updated_state = update_temperature(initial_state, t_a10_patch=new_t_a10)
    
    # Original state should be unchanged
    assert jnp.allclose(initial_state.t_a10_patch, 273.15, atol=1e-6), \
        "Original state was modified (not immutable)"
    
    # Updated state should have new values
    assert jnp.allclose(updated_state.t_a10_patch, new_t_a10, atol=1e-6), \
        "Updated state incorrect"


def test_update_temperature_no_update():
    """Test update_temperature with no new values (should return equivalent state)."""
    initial_state = init_temperature_state(n_columns=2, n_patches=3, n_levtot=5, initial_temp=273.15)
    
    updated_state = update_temperature(initial_state)
    
    assert jnp.allclose(updated_state.t_soisno_col, initial_state.t_soisno_col, atol=1e-6), \
        "No update: t_soisno_col changed"
    assert jnp.allclose(updated_state.t_a10_patch, initial_state.t_a10_patch, atol=1e-6), \
        "No update: t_a10_patch changed"
    assert jnp.allclose(updated_state.t_ref2m_patch, initial_state.t_ref2m_patch, atol=1e-6), \
        "No update: t_ref2m_patch changed"


def test_update_temperature_shape_preservation():
    """Test that update_temperature preserves array shapes."""
    initial_state = init_temperature_state(n_columns=10, n_patches=25, n_levtot=20, initial_temp=273.15)
    
    new_t_soisno = jnp.full((10, 20), 280.0)
    updated_state = update_temperature(initial_state, t_soisno_col=new_t_soisno)
    
    assert updated_state.t_soisno_col.shape == initial_state.t_soisno_col.shape, \
        "Shape changed after update"
    assert updated_state.t_a10_patch.shape == initial_state.t_a10_patch.shape, \
        "Patch shape changed after update"


# ============================================================================
# Integration Tests
# ============================================================================

def test_integration_init_and_extract():
    """Integration test: initialize state and extract temperatures."""
    # Initialize
    state = init_temperature_state(n_columns=5, n_patches=10, n_levtot=20, initial_temp=275.0)
    
    # Extract soil temperatures
    soil_temps = get_soil_temperature(state, column_idx=2, nlevsno=5)
    assert jnp.allclose(soil_temps, 275.0, atol=1e-6), "Integration: soil extraction failed"
    
    # Extract snow temperatures
    snow_temps = get_snow_temperature(state, column_idx=2, nlevsno=5)
    assert jnp.allclose(snow_temps, 275.0, atol=1e-6), "Integration: snow extraction failed"


def test_integration_init_update_extract():
    """Integration test: initialize, update, and extract temperatures."""
    # Initialize
    state = init_temperature_state(n_columns=3, n_patches=5, n_levtot=10, initial_temp=273.15)
    
    # Update with new profile
    new_profile = jnp.array([[260, 265, 270, 275, 280, 282, 284, 286, 288, 290],
                             [255, 260, 265, 270, 275, 278, 281, 284, 287, 290],
                             [265, 268, 271, 274, 277, 280, 283, 286, 289, 292]], dtype=jnp.float32)
    
    updated_state = update_temperature(state, t_soisno_col=new_profile)
    
    # Extract and verify
    soil_temps = get_soil_temperature(updated_state, column_idx=1, nlevsno=3)
    expected_soil = new_profile[1, 3:]
    
    assert jnp.allclose(soil_temps, expected_soil, atol=1e-6), \
        "Integration: update and extract failed"


def test_integration_allocate_vs_initialize():
    """Integration test: compare allocate (NaN) vs initialize (values)."""
    bounds = BoundsType(0, 49, 0, 19, 0, 9)
    
    # Allocate with NaN
    allocated_state = init_allocate_temperature(bounds, nlevsno=5, nlevgrnd=15)
    assert jnp.all(jnp.isnan(allocated_state.t_soisno_col)), "Allocated state should have NaN"
    
    # Initialize with values
    initialized_state = init_temperature(bounds, nlevsno=5, nlevgrnd=15)
    assert not jnp.any(jnp.isnan(initialized_state.t_soisno_col)), "Initialized state should not have NaN"
    assert jnp.allclose(initialized_state.t_soisno_col, DEFAULT_INIT_TEMP, atol=1e-6), \
        "Initialized state should have default temperature"


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_edge_case_large_domain():
    """Test with large domain to check memory and performance."""
    state = init_temperature_state(n_columns=1000, n_patches=2000, n_levtot=35, initial_temp=273.15)
    
    assert state.t_soisno_col.shape == (1000, 35), "Large domain: incorrect shape"
    assert state.t_a10_patch.shape == (2000,), "Large domain: incorrect patch shape"


def test_edge_case_deep_soil_profile():
    """Test with deep soil profile (30 layers)."""
    state = init_temperature_state(n_columns=10, n_patches=20, n_levtot=35, initial_temp=280.0)
    
    soil_temps = get_soil_temperature(state, column_idx=5, nlevsno=5)
    assert soil_temps.shape == (30,), "Deep soil: incorrect shape"


def test_edge_case_bounds_consistency():
    """Test that bounds indices are handled consistently."""
    # Test with offset indices
    bounds = BoundsType(100, 149, 50, 74, 20, 29)
    state = init_temperature(bounds, nlevsno=5, nlevgrnd=15)
    
    expected_n_columns = 74 - 50 + 1  # 25
    expected_n_patches = 149 - 100 + 1  # 50
    
    assert state.t_soisno_col.shape[0] == expected_n_columns, \
        f"Offset bounds: expected {expected_n_columns} columns, got {state.t_soisno_col.shape[0]}"
    assert state.t_a10_patch.shape[0] == expected_n_patches, \
        f"Offset bounds: expected {expected_n_patches} patches, got {state.t_a10_patch.shape[0]}"


# ============================================================================
# Physical Realism Tests
# ============================================================================

def test_physical_realism_temperature_bounds():
    """Test that all temperatures remain within physical bounds."""
    test_cases = [
        (173.15, "extreme_cold"),
        (273.15, "freezing"),
        (298.15, "room_temp"),
        (373.15, "boiling"),
    ]
    
    for temp, label in test_cases:
        state = init_temperature_state(n_columns=10, n_patches=20, n_levtot=15, initial_temp=temp)
        
        assert jnp.all(state.t_soisno_col > 0.0), f"{label}: temperatures below absolute zero"
        assert jnp.all(jnp.isfinite(state.t_soisno_col)), f"{label}: non-finite temperatures"


def test_physical_realism_vertical_gradient():
    """Test that realistic vertical temperature gradients are maintained."""
    # Create state with cold surface, warm depth
    n_columns = 1
    n_levtot = 20
    
    # Create realistic gradient
    temps = jnp.linspace(250.0, 280.0, n_levtot)
    t_soisno_col = temps.reshape(1, -1)
    
    state = TemperatureState(
        t_soisno_col=t_soisno_col,
        t_a10_patch=jnp.array([265.0]),
        t_ref2m_patch=jnp.array([263.0])
    )
    
    # Extract and verify gradient
    soil_temps = get_soil_temperature(state, column_idx=0, nlevsno=5)
    
    # Soil should warm with depth
    assert jnp.all(jnp.diff(soil_temps) >= 0), "Soil should warm with depth"


def test_physical_realism_snow_colder_than_soil():
    """Test that snow layers are typically colder than soil layers."""
    # Create realistic profile
    snow_temps = jnp.array([245.0, 250.0, 255.0, 260.0, 265.0])
    soil_temps = jnp.array([268.0, 270.0, 272.0, 274.0, 276.0, 278.0, 280.0, 282.0, 
                            284.0, 285.0, 286.0, 287.0, 288.0, 289.0, 290.0])
    
    t_soisno_col = jnp.concatenate([snow_temps, soil_temps]).reshape(1, -1)
    
    state = TemperatureState(
        t_soisno_col=t_soisno_col,
        t_a10_patch=jnp.array([265.0]),
        t_ref2m_patch=jnp.array([263.0])
    )
    
    extracted_snow = get_snow_temperature(state, column_idx=0, nlevsno=5)
    extracted_soil = get_soil_temperature(state, column_idx=0, nlevsno=5)
    
    # Coldest snow should be colder than warmest soil (typically)
    assert jnp.min(extracted_snow) < jnp.max(extracted_soil), \
        "Snow should generally be colder than deep soil"


# ============================================================================
# Documentation Tests
# ============================================================================

def test_namedtuple_structure():
    """Test that namedtuples have correct structure and field names."""
    # Test BoundsType
    bounds = BoundsType(0, 10, 0, 5, 0, 2)
    assert hasattr(bounds, 'begp'), "BoundsType missing begp field"
    assert hasattr(bounds, 'endp'), "BoundsType missing endp field"
    assert hasattr(bounds, 'begc'), "BoundsType missing begc field"
    assert hasattr(bounds, 'endc'), "BoundsType missing endc field"
    assert hasattr(bounds, 'begg'), "BoundsType missing begg field"
    assert hasattr(bounds, 'endg'), "BoundsType missing endg field"
    
    # Test TemperatureState
    state = init_temperature_state(5, 10, 15)
    assert hasattr(state, 't_soisno_col'), "TemperatureState missing t_soisno_col field"
    assert hasattr(state, 't_a10_patch'), "TemperatureState missing t_a10_patch field"
    assert hasattr(state, 't_ref2m_patch'), "TemperatureState missing t_ref2m_patch field"


def test_constants_defined():
    """Test that module constants are correctly defined."""
    assert DEFAULT_INIT_TEMP == 273.15, f"DEFAULT_INIT_TEMP incorrect: {DEFAULT_INIT_TEMP}"
    assert DEFAULT_NLEVSNO == 5, f"DEFAULT_NLEVSNO incorrect: {DEFAULT_NLEVSNO}"
    assert DEFAULT_NLEVGRND == 15, f"DEFAULT_NLEVGRND incorrect: {DEFAULT_NLEVGRND}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])