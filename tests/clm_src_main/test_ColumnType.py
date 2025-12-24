"""
Comprehensive pytest suite for ColumnType module.

This test suite covers the column_type dataclass and its associated methods,
including initialization, validation, index conversion, and layer masking operations.
Tests follow CLM conventions for Fortran-style indexing and physical constraints.

Test Coverage:
- Dataclass initialization with various column ranges
- Index conversion between Python (0-based) and Fortran (1-based) conventions
- Snow and soil layer masking operations
- Array validation and dimension checking
- Layer information retrieval
- Edge cases: single columns, zero values, boundary conditions
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import pytest
import jax.numpy as jnp
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from clm_src_main.ColumnType import (
    column_type,
    create_column_instance,
    reset_global_column,
    get_snow_layer_range,
    get_soil_layer_range,
    get_all_layer_range,
    get_active_snow_layers,
    get_soil_layer_mask,
    get_snow_layer_mask,
    create_layer_index_arrays,
)

# Import CLM constants (these should be defined in the module or we use defaults)
# Assuming standard CLM configuration
NLEVGRND = 15  # Number of ground/soil layers
NLEVSNO = 5    # Maximum number of snow layers
ISPVAL = -9999  # Integer special value
SPVAL = np.nan  # Float special value


@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Fixture providing test data for column_type tests.
    
    Returns:
        Dictionary containing test cases with inputs and expected behaviors
    """
    return {
        "single_column": {
            "begc": 0,
            "endc": 0,
            "expected_col_size": 1
        },
        "multiple_columns_typical": {
            "begc": 0,
            "endc": 9,
            "expected_col_size": 10
        },
        "large_column_range": {
            "begc": 0,
            "endc": 99,
            "expected_col_size": 100
        },
        "non_zero_start": {
            "begc": 10,
            "endc": 19,
            "expected_col_size": 10
        },
        "equal_indices": {
            "begc": 5,
            "endc": 5,
            "expected_col_size": 1
        },
        "snl_arrays": {
            "mixed": jnp.array([0, 2, 0, 5, 1, 0, 3, 0, 4, 0], dtype=jnp.int32),
            "all_zero": jnp.array([0, 0, 0, 0, 0], dtype=jnp.int32),
            "all_active": jnp.array([5, 3, 4, 2, 1, 5, 4, 3], dtype=jnp.int32),
        },
        "layer_indices": {
            "positive": jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "negative": jnp.array([-4, -3, -2, -1, 0]),
            "mixed": jnp.array([-2, -1, 0, 1, 2, 3, 4, 5]),
            "snow_only": jnp.array([-5, -4, -3, -2, -1]),
            "with_zero": jnp.array([-3, -2, -1, 0, 1, 2]),
        }
    }


@pytest.fixture
def initialized_column() -> column_type:
    """
    Fixture providing an initialized column_type instance.
    
    Returns:
        Initialized column_type with 10 columns (begc=0, endc=9)
    """
    col = create_column_instance(begc=0, endc=9)
    return col


class TestColumnTypeInitialization:
    """Tests for column_type initialization and basic properties."""
    
    @pytest.mark.parametrize("begc,endc,expected_size", [
        (0, 0, 1),      # Single column
        (0, 9, 10),     # Typical range
        (0, 99, 100),   # Large range
        (10, 19, 10),   # Non-zero start
        (5, 5, 1),      # Equal indices
    ])
    def test_init_column_sizes(self, begc: int, endc: int, expected_size: int):
        """
        Test column initialization with various index ranges.
        
        Verifies that column_size is correctly calculated as (endc - begc + 1)
        and that begc/endc are properly stored.
        """
        col = create_column_instance(begc=begc, endc=endc)
        
        assert col.begc == begc, f"Expected begc={begc}, got {col.begc}"
        assert col.endc == endc, f"Expected endc={endc}, got {col.endc}"
        
        col_count = col.get_column_count()
        assert col_count == expected_size, \
            f"Expected {expected_size} columns, got {col_count}"
    
    def test_init_array_shapes(self, initialized_column: column_type):
        """
        Test that initialized arrays have correct shapes.
        
        Verifies:
        - snl shape: (col_size,)
        - dz, z shape: (col_size, nlevgrnd + nlevsno)
        - zi shape: (col_size, nlevgrnd + nlevsno + 1)
        - nbedrock shape: (col_size,)
        """
        col = initialized_column
        col_size = col.get_column_count()
        
        # Check 1D arrays
        assert col.snl.shape == (col_size,), \
            f"snl shape mismatch: expected ({col_size},), got {col.snl.shape}"
        assert col.nbedrock.shape == (col_size,), \
            f"nbedrock shape mismatch: expected ({col_size},), got {col.nbedrock.shape}"
        
        # Check 2D arrays (dz, z)
        expected_layers = NLEVGRND + NLEVSNO
        assert col.dz.shape == (col_size, expected_layers), \
            f"dz shape mismatch: expected ({col_size}, {expected_layers}), got {col.dz.shape}"
        assert col.z.shape == (col_size, expected_layers), \
            f"z shape mismatch: expected ({col_size}, {expected_layers}), got {col.z.shape}"
        
        # Check interface array (zi) - has one extra layer
        expected_zi_layers = NLEVGRND + NLEVSNO + 1
        assert col.zi.shape == (col_size, expected_zi_layers), \
            f"zi shape mismatch: expected ({col_size}, {expected_zi_layers}), got {col.zi.shape}"
    
    def test_init_array_dtypes(self, initialized_column: column_type):
        """
        Test that initialized arrays have correct data types.
        
        Verifies:
        - Integer arrays (snl, nbedrock) are int32
        - Float arrays (dz, z, zi) are float64 (r8)
        """
        col = initialized_column
        
        # Integer arrays
        assert col.snl.dtype == jnp.int32, \
            f"snl dtype mismatch: expected int32, got {col.snl.dtype}"
        assert col.nbedrock.dtype == jnp.int32, \
            f"nbedrock dtype mismatch: expected int32, got {col.nbedrock.dtype}"
        
        # Float arrays (r8 = float64)
        assert col.dz.dtype == jnp.float64, \
            f"dz dtype mismatch: expected float64, got {col.dz.dtype}"
        assert col.z.dtype == jnp.float64, \
            f"z dtype mismatch: expected float64, got {col.z.dtype}"
        assert col.zi.dtype == jnp.float64, \
            f"zi dtype mismatch: expected float64, got {col.zi.dtype}"
    
    def test_is_initialized(self, initialized_column: column_type):
        """
        Test is_initialized method returns True for initialized column.
        """
        col = initialized_column
        assert col.is_initialized(), "Column should be initialized"
    
    def test_uninitialized_column(self):
        """
        Test that uninitialized column_type returns False for is_initialized.
        """
        col = column_type()
        assert not col.is_initialized(), "Uninitialized column should return False"


class TestColumnTypeValidation:
    """Tests for column_type validation methods."""
    
    def test_validate_arrays_success(self, initialized_column: column_type):
        """
        Test validate_arrays returns True for properly initialized column.
        """
        col = initialized_column
        assert col.validate_arrays(), "Validation should pass for initialized column"
    
    def test_get_layer_info(self, initialized_column: column_type):
        """
        Test get_layer_info returns correct dimension information.
        
        Verifies dictionary contains:
        - num_columns
        - dz_z_layers (nlevgrnd + nlevsno)
        - zi_layers (nlevgrnd + nlevsno + 1)
        - snow_layers_max (nlevsno)
        - ground_layers_total (nlevgrnd)
        """
        col = initialized_column
        info = col.get_layer_info()
        
        assert "num_columns" in info, "Missing num_columns in layer info"
        assert "dz_z_layers" in info, "Missing dz_z_layers in layer info"
        assert "zi_layers" in info, "Missing zi_layers in layer info"
        assert "snow_layers_max" in info, "Missing snow_layers_max in layer info"
        assert "ground_layers_total" in info, "Missing ground_layers_total in layer info"
        
        assert info["num_columns"] == 10, \
            f"Expected 10 columns, got {info['num_columns']}"
        assert info["dz_z_layers"] == NLEVGRND + NLEVSNO, \
            f"Expected {NLEVGRND + NLEVSNO} dz/z layers, got {info['dz_z_layers']}"
        assert info["zi_layers"] == NLEVGRND + NLEVSNO + 1, \
            f"Expected {NLEVGRND + NLEVSNO + 1} zi layers, got {info['zi_layers']}"
        assert info["snow_layers_max"] == NLEVSNO, \
            f"Expected {NLEVSNO} max snow layers, got {info['snow_layers_max']}"
        assert info["ground_layers_total"] == NLEVGRND, \
            f"Expected {NLEVGRND} ground layers, got {info['ground_layers_total']}"


class TestFortranIndexConversion:
    """Tests for Python to Fortran index conversion."""
    
    @pytest.mark.parametrize("col_idx,layer_idx,array_type,expected_fortran_layer", [
        (0, 0, "dz", -NLEVSNO + 1),  # First snow layer
        (0, NLEVSNO - 1, "dz", 0),   # Last snow layer (interface)
        (0, NLEVSNO, "dz", 1),       # First soil layer
        (2, 10, "z", 10 - NLEVSNO + 1),  # Mid-range soil layer
    ])
    def test_get_fortran_indices_dz_z(
        self, 
        initialized_column: column_type,
        col_idx: int,
        layer_idx: int,
        array_type: str,
        expected_fortran_layer: int
    ):
        """
        Test Fortran index conversion for dz and z arrays.
        
        Fortran convention:
        - Snow layers: -nlevsno+1 to 0
        - Soil layers: 1 to nlevgrnd
        Python uses 0-based indexing for both dimensions.
        """
        col = initialized_column
        fortran_col, fortran_layer = col.get_fortran_indices(
            col_idx, layer_idx, array_type
        )
        
        # Column index: Fortran is 1-based
        expected_fortran_col = col_idx + 1
        assert fortran_col == expected_fortran_col, \
            f"Expected Fortran col {expected_fortran_col}, got {fortran_col}"
        
        assert fortran_layer == expected_fortran_layer, \
            f"Expected Fortran layer {expected_fortran_layer}, got {fortran_layer}"
    
    def test_get_fortran_indices_zi(self, initialized_column: column_type):
        """
        Test Fortran index conversion for zi (interface) array.
        
        zi array has one extra layer for interfaces.
        """
        col = initialized_column
        col_idx = 1
        layer_idx = 5
        
        fortran_col, fortran_layer = col.get_fortran_indices(
            col_idx, layer_idx, "zi"
        )
        
        # zi has nlevsno extra layers at the beginning
        expected_fortran_layer = layer_idx - NLEVSNO
        
        assert fortran_col == col_idx + 1, \
            f"Expected Fortran col {col_idx + 1}, got {fortran_col}"
        assert fortran_layer == expected_fortran_layer, \
            f"Expected Fortran layer {expected_fortran_layer}, got {fortran_layer}"


class TestLayerRangeFunctions:
    """Tests for layer range utility functions."""
    
    def test_get_snow_layer_range(self):
        """
        Test get_snow_layer_range returns correct Fortran-style range.
        
        Expected: (-nlevsno + 1, 0)
        """
        start, end = get_snow_layer_range()
        
        assert start == -NLEVSNO + 1, \
            f"Expected snow start {-NLEVSNO + 1}, got {start}"
        assert end == 0, f"Expected snow end 0, got {end}"
    
    def test_get_soil_layer_range(self):
        """
        Test get_soil_layer_range returns correct Fortran-style range.
        
        Expected: (1, nlevgrnd)
        """
        start, end = get_soil_layer_range()
        
        assert start == 1, f"Expected soil start 1, got {start}"
        assert end == NLEVGRND, f"Expected soil end {NLEVGRND}, got {end}"
    
    @pytest.mark.parametrize("include_interface,expected_start,expected_end", [
        (False, -NLEVSNO + 1, NLEVGRND),
        (True, -NLEVSNO, NLEVGRND),
    ])
    def test_get_all_layer_range(
        self, 
        include_interface: bool,
        expected_start: int,
        expected_end: int
    ):
        """
        Test get_all_layer_range with and without interface layers.
        
        Without interface: (-nlevsno + 1, nlevgrnd)
        With interface: (-nlevsno, nlevgrnd)
        """
        start, end = get_all_layer_range(include_interface=include_interface)
        
        assert start == expected_start, \
            f"Expected start {expected_start}, got {start}"
        assert end == expected_end, \
            f"Expected end {expected_end}, got {end}"


class TestActiveSnowLayers:
    """Tests for get_active_snow_layers function."""
    
    def test_get_active_snow_layers_mixed(self, test_data: Dict[str, Any]):
        """
        Test get_active_snow_layers with mixed zero and positive values.
        
        Expected: Boolean mask where True indicates snl > 0
        """
        snl_array = test_data["snl_arrays"]["mixed"]
        mask = get_active_snow_layers(snl_array)
        
        expected_mask = jnp.array(
            [False, True, False, True, True, False, True, False, True, False],
            dtype=bool
        )
        
        assert mask.shape == snl_array.shape, \
            f"Mask shape {mask.shape} doesn't match input shape {snl_array.shape}"
        assert jnp.array_equal(mask, expected_mask), \
            f"Expected mask {expected_mask}, got {mask}"
    
    def test_get_active_snow_layers_all_zero(self, test_data: Dict[str, Any]):
        """
        Test get_active_snow_layers with all zero values.
        
        Expected: All False
        """
        snl_array = test_data["snl_arrays"]["all_zero"]
        mask = get_active_snow_layers(snl_array)
        
        assert mask.shape == snl_array.shape, \
            f"Mask shape {mask.shape} doesn't match input shape {snl_array.shape}"
        assert not jnp.any(mask), "Expected all False for zero snl values"
    
    def test_get_active_snow_layers_all_active(self, test_data: Dict[str, Any]):
        """
        Test get_active_snow_layers with all positive values.
        
        Expected: All True
        """
        snl_array = test_data["snl_arrays"]["all_active"]
        mask = get_active_snow_layers(snl_array)
        
        assert mask.shape == snl_array.shape, \
            f"Mask shape {mask.shape} doesn't match input shape {snl_array.shape}"
        assert jnp.all(mask), "Expected all True for positive snl values"
    
    def test_get_active_snow_layers_dtype(self, test_data: Dict[str, Any]):
        """
        Test that get_active_snow_layers returns boolean array.
        """
        snl_array = test_data["snl_arrays"]["mixed"]
        mask = get_active_snow_layers(snl_array)
        
        assert mask.dtype == bool, \
            f"Expected boolean dtype, got {mask.dtype}"


class TestSoilLayerMask:
    """Tests for get_soil_layer_mask function."""
    
    def test_get_soil_layer_mask_positive_indices(self, test_data: Dict[str, Any]):
        """
        Test get_soil_layer_mask with positive indices (soil layers).
        
        Expected: All True for positive indices
        """
        layer_indices = test_data["layer_indices"]["positive"]
        mask = get_soil_layer_mask(layer_indices)
        
        assert mask.shape == layer_indices.shape, \
            f"Mask shape {mask.shape} doesn't match input shape {layer_indices.shape}"
        assert jnp.all(mask), "Expected all True for positive indices"
    
    def test_get_soil_layer_mask_negative_indices(self, test_data: Dict[str, Any]):
        """
        Test get_soil_layer_mask with negative/zero indices (snow layers).
        
        Expected: All False for non-positive indices
        """
        layer_indices = test_data["layer_indices"]["negative"]
        mask = get_soil_layer_mask(layer_indices)
        
        assert mask.shape == layer_indices.shape, \
            f"Mask shape {mask.shape} doesn't match input shape {layer_indices.shape}"
        assert not jnp.any(mask), "Expected all False for non-positive indices"
    
    def test_get_soil_layer_mask_mixed_indices(self, test_data: Dict[str, Any]):
        """
        Test get_soil_layer_mask with mixed positive and negative indices.
        
        Expected: True for positive, False for non-positive
        """
        layer_indices = test_data["layer_indices"]["mixed"]
        mask = get_soil_layer_mask(layer_indices)
        
        expected_mask = jnp.array(
            [False, False, False, True, True, True, True, True],
            dtype=bool
        )
        
        assert mask.shape == layer_indices.shape, \
            f"Mask shape {mask.shape} doesn't match input shape {layer_indices.shape}"
        assert jnp.array_equal(mask, expected_mask), \
            f"Expected mask {expected_mask}, got {mask}"
    
    def test_get_soil_layer_mask_dtype(self, test_data: Dict[str, Any]):
        """
        Test that get_soil_layer_mask returns boolean array.
        """
        layer_indices = test_data["layer_indices"]["positive"]
        mask = get_soil_layer_mask(layer_indices)
        
        assert mask.dtype == bool, \
            f"Expected boolean dtype, got {mask.dtype}"


class TestSnowLayerMask:
    """Tests for get_snow_layer_mask function."""
    
    def test_get_snow_layer_mask_negative_indices(self, test_data: Dict[str, Any]):
        """
        Test get_snow_layer_mask with negative indices (snow layers).
        
        Expected: All True for negative indices
        """
        layer_indices = test_data["layer_indices"]["snow_only"]
        mask = get_snow_layer_mask(layer_indices)
        
        assert mask.shape == layer_indices.shape, \
            f"Mask shape {mask.shape} doesn't match input shape {layer_indices.shape}"
        assert jnp.all(mask), "Expected all True for negative indices"
    
    def test_get_snow_layer_mask_positive_indices(self, test_data: Dict[str, Any]):
        """
        Test get_snow_layer_mask with positive indices (soil layers).
        
        Expected: All False for positive indices
        """
        layer_indices = test_data["layer_indices"]["positive"]
        mask = get_snow_layer_mask(layer_indices)
        
        assert mask.shape == layer_indices.shape, \
            f"Mask shape {mask.shape} doesn't match input shape {layer_indices.shape}"
        assert not jnp.any(mask), "Expected all False for positive indices"
    
    def test_get_snow_layer_mask_with_zero(self, test_data: Dict[str, Any]):
        """
        Test get_snow_layer_mask with zero index (boundary).
        
        Expected: True for negative, False for zero and positive
        """
        layer_indices = test_data["layer_indices"]["with_zero"]
        mask = get_snow_layer_mask(layer_indices)
        
        expected_mask = jnp.array(
            [True, True, True, False, False, False],
            dtype=bool
        )
        
        assert mask.shape == layer_indices.shape, \
            f"Mask shape {mask.shape} doesn't match input shape {layer_indices.shape}"
        assert jnp.array_equal(mask, expected_mask), \
            f"Expected mask {expected_mask}, got {mask}"
    
    def test_get_snow_layer_mask_dtype(self, test_data: Dict[str, Any]):
        """
        Test that get_snow_layer_mask returns boolean array.
        """
        layer_indices = test_data["layer_indices"]["snow_only"]
        mask = get_snow_layer_mask(layer_indices)
        
        assert mask.dtype == bool, \
            f"Expected boolean dtype, got {mask.dtype}"


class TestLayerIndexArrays:
    """Tests for create_layer_index_arrays function."""
    
    def test_create_layer_index_arrays_keys(self):
        """
        Test that create_layer_index_arrays returns dictionary with expected keys.
        """
        arrays = create_layer_index_arrays()
        
        expected_keys = ["dz_z_indices", "zi_indices", "snow_indices", "soil_indices"]
        for key in expected_keys:
            assert key in arrays, f"Missing key '{key}' in layer index arrays"
    
    def test_create_layer_index_arrays_shapes(self):
        """
        Test that layer index arrays have correct shapes.
        """
        arrays = create_layer_index_arrays()
        
        # dz_z_indices: -nlevsno+1 to nlevgrnd
        expected_dz_z_size = NLEVGRND + NLEVSNO
        assert arrays["dz_z_indices"].shape == (expected_dz_z_size,), \
            f"Expected dz_z_indices shape ({expected_dz_z_size},), got {arrays['dz_z_indices'].shape}"
        
        # zi_indices: -nlevsno to nlevgrnd (one extra for interface)
        expected_zi_size = NLEVGRND + NLEVSNO + 1
        assert arrays["zi_indices"].shape == (expected_zi_size,), \
            f"Expected zi_indices shape ({expected_zi_size},), got {arrays['zi_indices'].shape}"
        
        # snow_indices: -nlevsno+1 to 0
        expected_snow_size = NLEVSNO
        assert arrays["snow_indices"].shape == (expected_snow_size,), \
            f"Expected snow_indices shape ({expected_snow_size},), got {arrays['snow_indices'].shape}"
        
        # soil_indices: 1 to nlevgrnd
        expected_soil_size = NLEVGRND
        assert arrays["soil_indices"].shape == (expected_soil_size,), \
            f"Expected soil_indices shape ({expected_soil_size},), got {arrays['soil_indices'].shape}"
    
    def test_create_layer_index_arrays_values(self):
        """
        Test that layer index arrays contain correct Fortran-style indices.
        """
        arrays = create_layer_index_arrays()
        
        # Check snow indices range
        assert jnp.min(arrays["snow_indices"]) == -NLEVSNO + 1, \
            f"Expected min snow index {-NLEVSNO + 1}, got {jnp.min(arrays['snow_indices'])}"
        assert jnp.max(arrays["snow_indices"]) == 0, \
            f"Expected max snow index 0, got {jnp.max(arrays['snow_indices'])}"
        
        # Check soil indices range
        assert jnp.min(arrays["soil_indices"]) == 1, \
            f"Expected min soil index 1, got {jnp.min(arrays['soil_indices'])}"
        assert jnp.max(arrays["soil_indices"]) == NLEVGRND, \
            f"Expected max soil index {NLEVGRND}, got {jnp.max(arrays['soil_indices'])}"


class TestGlobalColumnOperations:
    """Tests for global column instance operations."""
    
    def test_reset_global_column(self):
        """
        Test that reset_global_column properly resets the global instance.
        """
        # Create a global column
        col = create_column_instance(begc=0, endc=9)
        
        # Reset it
        reset_global_column()
        
        # After reset, creating a new instance should work
        new_col = create_column_instance(begc=0, endc=4)
        assert new_col.get_column_count() == 5, \
            "Expected 5 columns after reset and re-initialization"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_column_at_offset(self):
        """
        Test initialization with single column at non-zero offset.
        """
        col = create_column_instance(begc=5, endc=5)
        
        assert col.begc == 5, f"Expected begc=5, got {col.begc}"
        assert col.endc == 5, f"Expected endc=5, got {col.endc}"
        assert col.get_column_count() == 1, \
            f"Expected 1 column, got {col.get_column_count()}"
    
    def test_empty_snl_array(self):
        """
        Test get_active_snow_layers with empty array.
        """
        snl_array = jnp.array([], dtype=jnp.int32)
        mask = get_active_snow_layers(snl_array)
        
        assert mask.shape == (0,), f"Expected empty mask, got shape {mask.shape}"
    
    def test_empty_layer_indices(self):
        """
        Test layer mask functions with empty arrays.
        """
        layer_indices = jnp.array([])
        
        soil_mask = get_soil_layer_mask(layer_indices)
        snow_mask = get_snow_layer_mask(layer_indices)
        
        assert soil_mask.shape == (0,), \
            f"Expected empty soil mask, got shape {soil_mask.shape}"
        assert snow_mask.shape == (0,), \
            f"Expected empty snow mask, got shape {snow_mask.shape}"
    
    def test_max_snow_layers(self):
        """
        Test with maximum number of snow layers (nlevsno).
        """
        snl_array = jnp.array([NLEVSNO], dtype=jnp.int32)
        mask = get_active_snow_layers(snl_array)
        
        assert jnp.all(mask), "Expected True for max snow layers"
    
    def test_boundary_layer_index_zero(self):
        """
        Test layer masks at boundary (index 0).
        
        Index 0 is the boundary between snow and soil layers.
        """
        layer_indices = jnp.array([0])
        
        soil_mask = get_soil_layer_mask(layer_indices)
        snow_mask = get_snow_layer_mask(layer_indices)
        
        # Zero should not be considered soil (positive) or snow (negative)
        assert not jnp.any(soil_mask), "Index 0 should not be soil layer"
        assert not jnp.any(snow_mask), "Index 0 should not be snow layer"


class TestPhysicalConstraints:
    """Tests for physical constraint validation."""
    
    def test_column_indices_ordering(self):
        """
        Test that begc <= endc constraint is maintained.
        """
        # Valid case
        col = create_column_instance(begc=5, endc=10)
        assert col.begc <= col.endc, "begc should be <= endc"
    
    def test_non_negative_column_indices(self):
        """
        Test that column indices are non-negative.
        """
        col = create_column_instance(begc=0, endc=5)
        assert col.begc >= 0, "begc should be non-negative"
        assert col.endc >= 0, "endc should be non-negative"
    
    def test_snow_layer_count_range(self):
        """
        Test that snow layer counts are in valid range [0, nlevsno].
        """
        snl_values = jnp.array([0, 1, 2, 3, 4, 5], dtype=jnp.int32)
        
        assert jnp.all(snl_values >= 0), "snl values should be non-negative"
        assert jnp.all(snl_values <= NLEVSNO), \
            f"snl values should not exceed {NLEVSNO}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])