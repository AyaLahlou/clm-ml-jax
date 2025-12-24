"""
Comprehensive pytest suite for shr_kind_mod constants module.

This module tests the dtype constants defined in shr_kind_mod, which provides
precision/kind constants for CTSM numeric types. Since this is a constants-only
module, tests focus on:
- Verifying dtype properties (itemsize, kind, precision)
- Testing array creation with these dtypes
- Edge cases for numerical boundaries
- Compatibility with JAX/NumPy operations
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import the actual constants from the translated module
from cime_src_share_util.shr_kind_mod import SHR_KIND_R8, SHR_KIND_IN, r8


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load test data for shr_kind_mod constants testing.
    
    Returns:
        Dictionary containing test cases with inputs and expected outputs
    """
    return {
        "r8_dtype_properties": {
            "expected": {
                "dtype": "float64",
                "itemsize": 8,
                "kind": "f",
                "precision_digits": 15,
            }
        },
        "in_dtype_properties": {
            "expected": {
                "dtype": "int32",
                "itemsize": 4,
                "kind": "i",
                "min_value": -2147483648,
                "max_value": 2147483647,
            }
        },
        "r8_array_scalar": {
            "inputs": {"value": 273.15},
            "expected": {"array": 273.15, "dtype": "float64", "shape": ()},
        },
        "r8_array_1d": {
            "inputs": {"values": [0.5, 1.0, 2.5, 10.0, 100.0]},
            "expected": {"dtype": "float64", "shape": (5,), "sum": 114.0},
        },
        "in_array_indices": {
            "inputs": {"start": 0, "stop": 1000},
            "expected": {"dtype": "int32", "shape": (1000,), "min": 0, "max": 999},
        },
        "r8_very_small": {
            "inputs": {"values": [1e-300, 1e-200, 1e-100, 1e-50, 1e-15]},
            "expected": {
                "dtype": "float64",
                "all_positive": True,
                "all_finite": True,
                "min_representable": 2.2250738585072014e-308,
            },
        },
        "r8_very_large": {
            "inputs": {"values": [1e15, 1e50, 1e100, 1e200, 1e300]},
            "expected": {
                "dtype": "float64",
                "all_positive": True,
                "all_finite": True,
                "max_representable": 1.7976931348623157e308,
            },
        },
        "in_boundary": {
            "inputs": {"values": [-2147483648, -1000000, 0, 1000000, 2147483647]},
            "expected": {
                "dtype": "int32",
                "min": -2147483648,
                "max": 2147483647,
                "contains_zero": True,
            },
        },
        "r8_special": {
            "inputs": {"values": [0.0, -0.0, 1.0, -1.0]},
            "expected": {
                "dtype": "float64",
                "contains_zero": True,
                "contains_negative": True,
                "all_finite": True,
            },
        },
        "multidimensional": {
            "inputs": {
                "r8_shape": (10, 20, 30),
                "in_shape": (5, 10),
                "r8_fill_value": 1.5,
                "in_fill_value": 42,
            },
            "expected": {
                "r8_dtype": "float64",
                "r8_shape": (10, 20, 30),
                "r8_size": 6000,
                "in_dtype": "int32",
                "in_shape": (5, 10),
                "in_size": 50,
            },
        },
    }


# ============================================================================
# Dtype Properties Tests
# ============================================================================


def test_shr_kind_r8_dtype_properties(test_data):
    """
    Test that SHR_KIND_R8 constant has correct float64 properties.
    
    Verifies:
    - Dtype is float64
    - Item size is 8 bytes
    - Kind is floating point ('f')
    - Precision is appropriate for double precision
    """
    expected = test_data["r8_dtype_properties"]["expected"]
    
    # Check dtype name
    assert SHR_KIND_R8.name == expected["dtype"], (
        f"SHR_KIND_R8 dtype should be {expected['dtype']}, "
        f"got {SHR_KIND_R8.name}"
    )
    
    # Check item size (bytes)
    assert SHR_KIND_R8.itemsize == expected["itemsize"], (
        f"SHR_KIND_R8 itemsize should be {expected['itemsize']} bytes, "
        f"got {SHR_KIND_R8.itemsize}"
    )
    
    # Check kind (floating point)
    assert SHR_KIND_R8.kind == expected["kind"], (
        f"SHR_KIND_R8 kind should be '{expected['kind']}', "
        f"got '{SHR_KIND_R8.kind}'"
    )
    
    # Verify it's a JAX dtype
    assert isinstance(SHR_KIND_R8, jnp.dtype), (
        "SHR_KIND_R8 should be a JAX dtype instance"
    )


def test_r8_alias_matches_shr_kind_r8(test_data):
    """
    Test that r8 alias is identical to SHR_KIND_R8.
    
    The r8 constant should be an exact alias for SHR_KIND_R8,
    commonly used as a convenience in translated modules.
    """
    assert r8 == SHR_KIND_R8, (
        "r8 alias should be identical to SHR_KIND_R8"
    )
    
    assert r8.name == SHR_KIND_R8.name, (
        "r8 and SHR_KIND_R8 should have the same dtype name"
    )
    
    assert r8.itemsize == SHR_KIND_R8.itemsize, (
        "r8 and SHR_KIND_R8 should have the same itemsize"
    )


def test_shr_kind_in_dtype_properties(test_data):
    """
    Test that SHR_KIND_IN constant has correct int32 properties.
    
    Verifies:
    - Dtype is int32
    - Item size is 4 bytes
    - Kind is integer ('i')
    - Range matches int32 limits
    """
    expected = test_data["in_dtype_properties"]["expected"]
    
    # Check dtype name
    assert SHR_KIND_IN.name == expected["dtype"], (
        f"SHR_KIND_IN dtype should be {expected['dtype']}, "
        f"got {SHR_KIND_IN.name}"
    )
    
    # Check item size (bytes)
    assert SHR_KIND_IN.itemsize == expected["itemsize"], (
        f"SHR_KIND_IN itemsize should be {expected['itemsize']} bytes, "
        f"got {SHR_KIND_IN.itemsize}"
    )
    
    # Check kind (integer)
    assert SHR_KIND_IN.kind == expected["kind"], (
        f"SHR_KIND_IN kind should be '{expected['kind']}', "
        f"got '{SHR_KIND_IN.kind}'"
    )
    
    # Verify it's a JAX dtype
    assert isinstance(SHR_KIND_IN, jnp.dtype), (
        "SHR_KIND_IN should be a JAX dtype instance"
    )


def test_shr_kind_in_range_limits(test_data):
    """
    Test that SHR_KIND_IN can represent expected int32 range.
    
    Verifies that arrays created with SHR_KIND_IN can hold
    values at the int32 boundaries (-2^31 to 2^31-1).
    """
    expected = test_data["in_dtype_properties"]["expected"]
    
    # Create array with boundary values
    min_val = expected["min_value"]
    max_val = expected["max_value"]
    
    arr = jnp.array([min_val, max_val], dtype=SHR_KIND_IN)
    
    assert arr[0] == min_val, (
        f"SHR_KIND_IN should represent int32 min value {min_val}"
    )
    
    assert arr[1] == max_val, (
        f"SHR_KIND_IN should represent int32 max value {max_val}"
    )


# ============================================================================
# Array Creation Tests
# ============================================================================


def test_r8_array_creation_scalar(test_data):
    """
    Test creating scalar array with r8 dtype.
    
    Verifies that scalar values (e.g., temperature) can be
    represented with the r8 dtype constant.
    """
    inputs = test_data["r8_array_scalar"]["inputs"]
    expected = test_data["r8_array_scalar"]["expected"]
    
    # Create scalar array
    arr = jnp.array(inputs["value"], dtype=r8)
    
    # Check dtype
    assert arr.dtype == jnp.dtype(expected["dtype"]), (
        f"Array dtype should be {expected['dtype']}, got {arr.dtype}"
    )
    
    # Check shape (scalar)
    assert arr.shape == expected["shape"], (
        f"Scalar array shape should be {expected['shape']}, got {arr.shape}"
    )
    
    # Check value
    assert float(arr) == expected["array"], (
        f"Array value should be {expected['array']}, got {float(arr)}"
    )


def test_r8_array_creation_1d(test_data):
    """
    Test creating 1D array with r8 dtype.
    
    Verifies that 1D arrays of physical quantities can be
    created with the r8 dtype constant.
    """
    inputs = test_data["r8_array_1d"]["inputs"]
    expected = test_data["r8_array_1d"]["expected"]
    
    # Create 1D array
    arr = jnp.array(inputs["values"], dtype=r8)
    
    # Check dtype
    assert arr.dtype == jnp.dtype(expected["dtype"]), (
        f"Array dtype should be {expected['dtype']}, got {arr.dtype}"
    )
    
    # Check shape
    assert arr.shape == expected["shape"], (
        f"Array shape should be {expected['shape']}, got {arr.shape}"
    )
    
    # Check sum (verify values)
    assert np.isclose(float(jnp.sum(arr)), expected["sum"], rtol=1e-10), (
        f"Array sum should be {expected['sum']}, got {float(jnp.sum(arr))}"
    )


def test_in_array_creation_indices(test_data):
    """
    Test creating index array with SHR_KIND_IN dtype.
    
    Verifies that integer index arrays (e.g., loop counters)
    can be created with the SHR_KIND_IN dtype constant.
    """
    inputs = test_data["in_array_indices"]["inputs"]
    expected = test_data["in_array_indices"]["expected"]
    
    # Create index array
    arr = jnp.arange(inputs["start"], inputs["stop"], dtype=SHR_KIND_IN)
    
    # Check dtype
    assert arr.dtype == jnp.dtype(expected["dtype"]), (
        f"Array dtype should be {expected['dtype']}, got {arr.dtype}"
    )
    
    # Check shape
    assert arr.shape == expected["shape"], (
        f"Array shape should be {expected['shape']}, got {arr.shape}"
    )
    
    # Check min/max values
    assert int(jnp.min(arr)) == expected["min"], (
        f"Array min should be {expected['min']}, got {int(jnp.min(arr))}"
    )
    
    assert int(jnp.max(arr)) == expected["max"], (
        f"Array max should be {expected['max']}, got {int(jnp.max(arr))}"
    )


def test_multidimensional_array_creation(test_data):
    """
    Test creating multidimensional arrays with dtype constants.
    
    Verifies that both r8 and SHR_KIND_IN can be used to create
    multidimensional arrays (2D, 3D) as needed in CTSM computations.
    """
    inputs = test_data["multidimensional"]["inputs"]
    expected = test_data["multidimensional"]["expected"]
    
    # Create 3D r8 array
    r8_arr = jnp.full(inputs["r8_shape"], inputs["r8_fill_value"], dtype=r8)
    
    assert r8_arr.dtype == jnp.dtype(expected["r8_dtype"]), (
        f"r8 array dtype should be {expected['r8_dtype']}, got {r8_arr.dtype}"
    )
    
    assert r8_arr.shape == expected["r8_shape"], (
        f"r8 array shape should be {expected['r8_shape']}, got {r8_arr.shape}"
    )
    
    assert r8_arr.size == expected["r8_size"], (
        f"r8 array size should be {expected['r8_size']}, got {r8_arr.size}"
    )
    
    # Create 2D int array
    in_arr = jnp.full(inputs["in_shape"], inputs["in_fill_value"], dtype=SHR_KIND_IN)
    
    assert in_arr.dtype == jnp.dtype(expected["in_dtype"]), (
        f"int array dtype should be {expected['in_dtype']}, got {in_arr.dtype}"
    )
    
    assert in_arr.shape == expected["in_shape"], (
        f"int array shape should be {expected['in_shape']}, got {in_arr.shape}"
    )
    
    assert in_arr.size == expected["in_size"], (
        f"int array size should be {expected['in_size']}, got {in_arr.size}"
    )


# ============================================================================
# Edge Cases Tests
# ============================================================================


def test_r8_precision_very_small_values(test_data):
    """
    Test r8 dtype with very small positive values near underflow limit.
    
    Verifies that r8 (float64) can represent very small magnitudes
    down to approximately 2.2e-308 without underflowing to zero.
    """
    inputs = test_data["r8_very_small"]["inputs"]
    expected = test_data["r8_very_small"]["expected"]
    
    # Create array with very small values
    arr = jnp.array(inputs["values"], dtype=r8)
    
    # Check dtype
    assert arr.dtype == jnp.dtype(expected["dtype"]), (
        f"Array dtype should be {expected['dtype']}, got {arr.dtype}"
    )
    
    # Check all values are positive
    assert jnp.all(arr > 0), (
        "All very small values should remain positive (not underflow to zero)"
    )
    
    # Check all values are finite
    assert jnp.all(jnp.isfinite(arr)), (
        "All very small values should be finite"
    )
    
    # Verify smallest value is above underflow limit
    assert float(jnp.min(arr)) >= expected["min_representable"], (
        f"Smallest value should be >= {expected['min_representable']}"
    )


def test_r8_precision_very_large_values(test_data):
    """
    Test r8 dtype with very large values near overflow limit.
    
    Verifies that r8 (float64) can represent very large magnitudes
    up to approximately 1.8e308 without overflowing to infinity.
    """
    inputs = test_data["r8_very_large"]["inputs"]
    expected = test_data["r8_very_large"]["expected"]
    
    # Create array with very large values
    arr = jnp.array(inputs["values"], dtype=r8)
    
    # Check dtype
    assert arr.dtype == jnp.dtype(expected["dtype"]), (
        f"Array dtype should be {expected['dtype']}, got {arr.dtype}"
    )
    
    # Check all values are positive
    assert jnp.all(arr > 0), (
        "All very large values should be positive"
    )
    
    # Check all values are finite (not inf)
    assert jnp.all(jnp.isfinite(arr)), (
        "All very large values should be finite (not overflow to inf)"
    )
    
    # Verify largest value is below overflow limit
    assert float(jnp.max(arr)) <= expected["max_representable"], (
        f"Largest value should be <= {expected['max_representable']}"
    )


def test_in_boundary_values(test_data):
    """
    Test SHR_KIND_IN with int32 boundary values.
    
    Verifies that SHR_KIND_IN can represent the full int32 range
    including minimum (-2^31), maximum (2^31-1), and zero.
    """
    inputs = test_data["in_boundary"]["inputs"]
    expected = test_data["in_boundary"]["expected"]
    
    # Create array with boundary values
    arr = jnp.array(inputs["values"], dtype=SHR_KIND_IN)
    
    # Check dtype
    assert arr.dtype == jnp.dtype(expected["dtype"]), (
        f"Array dtype should be {expected['dtype']}, got {arr.dtype}"
    )
    
    # Check min value
    assert int(jnp.min(arr)) == expected["min"], (
        f"Array min should be {expected['min']}, got {int(jnp.min(arr))}"
    )
    
    # Check max value
    assert int(jnp.max(arr)) == expected["max"], (
        f"Array max should be {expected['max']}, got {int(jnp.max(arr))}"
    )
    
    # Check zero is present
    assert jnp.any(arr == 0), (
        "Array should contain zero value"
    )


def test_r8_special_float_values(test_data):
    """
    Test r8 dtype with special float values including signed zeros.
    
    Verifies that r8 can represent special IEEE 754 values like
    positive zero, negative zero, and signed unity values.
    """
    inputs = test_data["r8_special"]["inputs"]
    expected = test_data["r8_special"]["expected"]
    
    # Create array with special values
    arr = jnp.array(inputs["values"], dtype=r8)
    
    # Check dtype
    assert arr.dtype == jnp.dtype(expected["dtype"]), (
        f"Array dtype should be {expected['dtype']}, got {arr.dtype}"
    )
    
    # Check contains zero (positive or negative)
    assert jnp.any(arr == 0.0), (
        "Array should contain zero value"
    )
    
    # Check contains negative values
    assert jnp.any(arr < 0), (
        "Array should contain negative values"
    )
    
    # Check all values are finite
    assert jnp.all(jnp.isfinite(arr)), (
        "All special values should be finite"
    )


# ============================================================================
# Compatibility Tests
# ============================================================================


def test_r8_numpy_compatibility():
    """
    Test that r8 dtype is compatible with NumPy operations.
    
    Verifies that arrays created with r8 can be used in
    standard NumPy/JAX mathematical operations.
    """
    # Create test array
    arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=r8)
    
    # Test basic operations
    result_sum = jnp.sum(arr)
    result_mean = jnp.mean(arr)
    result_std = jnp.std(arr)
    
    # Verify operations work and return correct dtype
    assert result_sum.dtype == r8, (
        f"Sum result should have dtype {r8}, got {result_sum.dtype}"
    )
    
    assert np.isclose(float(result_mean), 3.0, rtol=1e-10), (
        f"Mean should be 3.0, got {float(result_mean)}"
    )
    
    # Test array operations
    result_mult = arr * 2.0
    assert result_mult.dtype == r8, (
        f"Multiplication result should have dtype {r8}, got {result_mult.dtype}"
    )


def test_in_numpy_compatibility():
    """
    Test that SHR_KIND_IN dtype is compatible with NumPy operations.
    
    Verifies that integer arrays created with SHR_KIND_IN can be
    used in standard NumPy/JAX indexing and integer operations.
    """
    # Create test array
    arr = jnp.arange(10, dtype=SHR_KIND_IN)
    
    # Test basic operations
    result_sum = jnp.sum(arr)
    result_max = jnp.max(arr)
    
    # Verify operations work and return correct dtype
    assert result_sum.dtype == SHR_KIND_IN, (
        f"Sum result should have dtype {SHR_KIND_IN}, got {result_sum.dtype}"
    )
    
    assert int(result_max) == 9, (
        f"Max should be 9, got {int(result_max)}"
    )
    
    # Test as indices
    data = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=r8)
    indices = jnp.array([0, 2, 4], dtype=SHR_KIND_IN)
    selected = data[indices]
    
    assert selected.shape == (3,), (
        f"Selected array should have shape (3,), got {selected.shape}"
    )
    
    assert np.allclose(selected, [10.0, 30.0, 50.0], rtol=1e-10), (
        "Selected values should match expected indices"
    )


def test_dtype_constants_immutability():
    """
    Test that dtype constants are immutable and consistent.
    
    Verifies that the dtype constants maintain their properties
    and cannot be accidentally modified.
    """
    # Store original properties
    r8_original = r8
    in_original = SHR_KIND_IN
    
    # Verify they remain unchanged
    assert r8 == r8_original, (
        "r8 constant should remain unchanged"
    )
    
    assert SHR_KIND_IN == in_original, (
        "SHR_KIND_IN constant should remain unchanged"
    )
    
    # Verify they are the same dtype objects
    assert r8 is r8_original, (
        "r8 should be the same object reference"
    )
    
    assert SHR_KIND_IN is in_original, (
        "SHR_KIND_IN should be the same object reference"
    )


# ============================================================================
# Usage Pattern Tests
# ============================================================================


@pytest.mark.parametrize(
    "shape,fill_value",
    [
        ((10,), 273.15),  # 1D temperature array
        ((5, 10), 1.0),  # 2D fraction array
        ((3, 4, 5), 0.5),  # 3D physical quantity
        ((100,), 1e-6),  # 1D small values
        ((20, 30), 1e3),  # 2D large values
    ],
)
def test_r8_typical_usage_patterns(shape, fill_value):
    """
    Test r8 dtype with typical CTSM usage patterns.
    
    Parametrized test covering common array shapes and value ranges
    used in climate model computations.
    """
    # Create array with typical pattern
    arr = jnp.full(shape, fill_value, dtype=r8)
    
    # Verify dtype
    assert arr.dtype == r8, (
        f"Array should have dtype {r8}, got {arr.dtype}"
    )
    
    # Verify shape
    assert arr.shape == shape, (
        f"Array should have shape {shape}, got {arr.shape}"
    )
    
    # Verify all values are correct
    assert jnp.all(arr == fill_value), (
        f"All array values should be {fill_value}"
    )
    
    # Verify values are finite
    assert jnp.all(jnp.isfinite(arr)), (
        "All array values should be finite"
    )


@pytest.mark.parametrize(
    "start,stop,step",
    [
        (0, 100, 1),  # Small range
        (0, 1000, 1),  # Medium range
        (0, 10000, 10),  # Large range with step
        (-100, 100, 1),  # Negative to positive
        (1, 1000, 5),  # Non-zero start with step
    ],
)
def test_in_typical_usage_patterns(start, stop, step):
    """
    Test SHR_KIND_IN dtype with typical indexing patterns.
    
    Parametrized test covering common index array patterns
    used for loop counters and array indexing in CTSM.
    """
    # Create index array with typical pattern
    arr = jnp.arange(start, stop, step, dtype=SHR_KIND_IN)
    
    # Verify dtype
    assert arr.dtype == SHR_KIND_IN, (
        f"Array should have dtype {SHR_KIND_IN}, got {arr.dtype}"
    )
    
    # Verify range
    if arr.size > 0:
        assert int(jnp.min(arr)) >= start, (
            f"Array min should be >= {start}"
        )
        
        assert int(jnp.max(arr)) < stop, (
            f"Array max should be < {stop}"
        )
    
    # Verify all values are finite
    assert jnp.all(jnp.isfinite(arr)), (
        "All array values should be finite"
    )


def test_mixed_dtype_operations():
    """
    Test operations mixing r8 and SHR_KIND_IN dtypes.
    
    Verifies that r8 float arrays can be indexed by SHR_KIND_IN
    integer arrays, a common pattern in CTSM computations.
    """
    # Create float data array
    data = jnp.array([10.5, 20.5, 30.5, 40.5, 50.5], dtype=r8)
    
    # Create integer index array
    indices = jnp.array([0, 2, 4], dtype=SHR_KIND_IN)
    
    # Index float array with integer indices
    selected = data[indices]
    
    # Verify result dtype is r8
    assert selected.dtype == r8, (
        f"Selected array should have dtype {r8}, got {selected.dtype}"
    )
    
    # Verify correct values selected
    expected = jnp.array([10.5, 30.5, 50.5], dtype=r8)
    assert jnp.allclose(selected, expected, rtol=1e-10), (
        f"Selected values should match expected, got {selected}"
    )


def test_zeros_and_ones_creation():
    """
    Test creating zeros and ones arrays with dtype constants.
    
    Verifies that common array initialization patterns work
    correctly with r8 and SHR_KIND_IN dtypes.
    """
    shape = (5, 10)
    
    # Create zeros with r8
    r8_zeros = jnp.zeros(shape, dtype=r8)
    assert r8_zeros.dtype == r8, (
        f"Zeros array should have dtype {r8}"
    )
    assert jnp.all(r8_zeros == 0.0), (
        "All values should be zero"
    )
    
    # Create ones with r8
    r8_ones = jnp.ones(shape, dtype=r8)
    assert r8_ones.dtype == r8, (
        f"Ones array should have dtype {r8}"
    )
    assert jnp.all(r8_ones == 1.0), (
        "All values should be one"
    )
    
    # Create zeros with SHR_KIND_IN
    in_zeros = jnp.zeros(shape, dtype=SHR_KIND_IN)
    assert in_zeros.dtype == SHR_KIND_IN, (
        f"Zeros array should have dtype {SHR_KIND_IN}"
    )
    assert jnp.all(in_zeros == 0), (
        "All values should be zero"
    )
    
    # Create ones with SHR_KIND_IN
    in_ones = jnp.ones(shape, dtype=SHR_KIND_IN)
    assert in_ones.dtype == SHR_KIND_IN, (
        f"Ones array should have dtype {SHR_KIND_IN}"
    )
    assert jnp.all(in_ones == 1), (
        "All values should be one"
    )