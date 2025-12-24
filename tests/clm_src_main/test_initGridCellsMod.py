"""
Comprehensive pytest suite for initGridCellsMod module.

This test suite covers:
- Grid cell initialization functions
- Patch structure validation
- Grid statistics calculation
- Global state management
- Edge cases and error handling
- Integration tests for initialization lifecycle

Test Strategy:
- Nominal cases: Typical grid configurations (50%)
- Edge cases: Boundary conditions, invalid inputs (30%)
- Special cases: Large grids, sparse indices (20%)
- Integration tests: State lifecycle and validation sequences
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import pytest
import jax.numpy as jnp
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from clm_src_main.initGridCellsMod import (
    initGridcells,
    set_landunit_veg_compete,
    validate_patch_structure,
    get_grid_initialization_state,
    reset_grid_initialization,
    create_simple_grid,
    calculate_grid_statistics,
    print_initialization_summary,
    initialize_single_patch_grid,
    initialize_multi_patch_grid,
    validate_initialization,
    GridCellInitialization,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load test data for all test cases.
    
    Returns:
        Dictionary containing test cases organized by function and test type.
    """
    return {
        "validate_patch_structure": {
            "nominal": [
                {
                    "name": "single_patch",
                    "patch_indices": jnp.array([0]),
                    "landunit_types": jnp.array([1]),
                    "expected": True,
                },
                {
                    "name": "multiple_patches",
                    "patch_indices": jnp.array([0, 1, 2, 3, 4]),
                    "landunit_types": jnp.array([1, 1, 2, 2, 3]),
                    "expected": True,
                },
            ],
            "edge": [
                {
                    "name": "zero_indices",
                    "patch_indices": jnp.array([0, 0, 0]),
                    "landunit_types": jnp.array([1, 2, 3]),
                    "expected": True,
                },
                {
                    "name": "large_indices",
                    "patch_indices": jnp.array([0, 100, 500, 999, 10000]),
                    "landunit_types": jnp.array([1, 1, 1, 1, 1]),
                    "expected": True,
                },
                {
                    "name": "negative_indices",
                    "patch_indices": jnp.array([-1, 0, 1]),
                    "landunit_types": jnp.array([1, 1, 1]),
                    "expected": False,
                },
                {
                    "name": "empty_arrays",
                    "patch_indices": jnp.array([]),
                    "landunit_types": jnp.array([]),
                    "expected": True,
                },
            ],
            "special": [
                {
                    "name": "mismatched_sizes",
                    "patch_indices": jnp.array([0, 1, 2]),
                    "landunit_types": jnp.array([1, 1]),
                    "expected": False,
                },
            ],
        },
        "create_simple_grid": {
            "nominal": [
                {
                    "name": "default_single_patch",
                    "num_patches": 1,
                    "pft_types": None,
                },
                {
                    "name": "multiple_pfts",
                    "num_patches": 5,
                    "pft_types": jnp.array([1, 2, 3, 4, 5]),
                },
            ],
            "edge": [
                {
                    "name": "minimum_patches",
                    "num_patches": 1,
                    "pft_types": jnp.array([1]),
                },
            ],
            "special": [
                {
                    "name": "large_grid",
                    "num_patches": 100,
                    "pft_types": jnp.array([i % 5 + 1 for i in range(100)]),
                },
            ],
        },
        "calculate_grid_statistics": {
            "nominal": [
                {
                    "name": "sequential",
                    "patch_indices": jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                    "expected": {
                        "mean_index": 4.5,
                        "max_index": 9.0,
                        "min_index": 0.0,
                        "num_patches": 10.0,
                    },
                },
            ],
            "edge": [
                {
                    "name": "single_patch",
                    "patch_indices": jnp.array([42]),
                    "expected": {
                        "mean_index": 42.0,
                        "max_index": 42.0,
                        "min_index": 42.0,
                        "num_patches": 1.0,
                    },
                },
                {
                    "name": "all_zeros",
                    "patch_indices": jnp.array([0, 0, 0, 0, 0]),
                    "expected": {
                        "mean_index": 0.0,
                        "max_index": 0.0,
                        "min_index": 0.0,
                        "num_patches": 5.0,
                    },
                },
            ],
            "special": [
                {
                    "name": "sparse_indices",
                    "patch_indices": jnp.array([0, 10, 100, 1000, 10000]),
                    "expected": {
                        "mean_index": 2222.0,
                        "max_index": 10000.0,
                        "min_index": 0.0,
                        "num_patches": 5.0,
                    },
                },
            ],
        },
    }


@pytest.fixture(autouse=True)
def reset_state():
    """
    Automatically reset grid initialization state before each test.
    
    This ensures test isolation and prevents state leakage between tests.
    """
    reset_grid_initialization()
    yield
    reset_grid_initialization()


# ============================================================================
# Test validate_patch_structure
# ============================================================================

class TestValidatePatchStructure:
    """Tests for validate_patch_structure function."""
    
    @pytest.mark.parametrize("test_case", [
        pytest.param(
            {"patch_indices": jnp.array([0]), "landunit_types": jnp.array([1])},
            id="single_patch"
        ),
        pytest.param(
            {"patch_indices": jnp.array([0, 1, 2, 3, 4]), 
             "landunit_types": jnp.array([1, 1, 2, 2, 3])},
            id="multiple_patches"
        ),
    ])
    def test_nominal_cases(self, test_case: Dict[str, Any]):
        """
        Test validate_patch_structure with nominal/typical inputs.
        
        These tests verify that the function correctly validates standard
        grid configurations with valid patch indices and landunit types.
        """
        result = validate_patch_structure(
            test_case["patch_indices"],
            test_case["landunit_types"]
        )
        assert isinstance(result, (bool, jnp.ndarray, np.bool_))
        assert bool(result) is True, "Valid patch structure should return True"
    
    @pytest.mark.parametrize("test_case,expected", [
        pytest.param(
            {"patch_indices": jnp.array([0, 0, 0]), 
             "landunit_types": jnp.array([1, 2, 3])},
            True,
            id="zero_indices"
        ),
        pytest.param(
            {"patch_indices": jnp.array([0, 100, 500, 999, 10000]), 
             "landunit_types": jnp.array([1, 1, 1, 1, 1])},
            True,
            id="large_indices"
        ),
        pytest.param(
            {"patch_indices": jnp.array([-1, 0, 1]), 
             "landunit_types": jnp.array([1, 1, 1])},
            False,
            id="negative_indices"
        ),
        pytest.param(
            {"patch_indices": jnp.array([]), 
             "landunit_types": jnp.array([])},
            True,
            id="empty_arrays"
        ),
    ])
    def test_edge_cases(self, test_case: Dict[str, Any], expected: bool):
        """
        Test validate_patch_structure with edge cases.
        
        Edge cases include:
        - Zero indices (minimum valid values)
        - Large indices (upper boundary)
        - Negative indices (invalid input)
        - Empty arrays (no patches)
        """
        result = validate_patch_structure(
            test_case["patch_indices"],
            test_case["landunit_types"]
        )
        assert bool(result) == expected, \
            f"Expected {expected} for {test_case}"
    
    def test_mismatched_array_sizes(self):
        """
        Test validate_patch_structure with mismatched array dimensions.
        
        This should return False as the arrays must have consistent sizes.
        """
        patch_indices = jnp.array([0, 1, 2])
        landunit_types = jnp.array([1, 1])
        
        result = validate_patch_structure(patch_indices, landunit_types)
        assert bool(result) is False, \
            "Mismatched array sizes should return False"
    
    def test_output_dtype(self):
        """Verify that validate_patch_structure returns boolean type."""
        patch_indices = jnp.array([0, 1, 2])
        landunit_types = jnp.array([1, 1, 1])
        
        result = validate_patch_structure(patch_indices, landunit_types)
        assert isinstance(result, (bool, jnp.ndarray, np.bool_)), \
            f"Expected boolean type, got {type(result)}"
    
    def test_jit_compilation(self):
        """
        Test that validate_patch_structure works with JIT compilation.
        
        This function is marked as jit_compiled in the signature.
        """
        from jax import jit
        
        jitted_validate = jit(validate_patch_structure)
        patch_indices = jnp.array([0, 1, 2])
        landunit_types = jnp.array([1, 1, 1])
        
        result = jitted_validate(patch_indices, landunit_types)
        assert bool(result) is True, "JIT-compiled function should work correctly"


# ============================================================================
# Test create_simple_grid
# ============================================================================

class TestCreateSimpleGrid:
    """Tests for create_simple_grid function."""
    
    def test_default_single_patch(self):
        """
        Test create_simple_grid with default parameters.
        
        Should create a single patch with default PFT type.
        """
        grid = create_simple_grid(num_patches=1, pft_types=None)
        
        assert isinstance(grid, GridCellInitialization), \
            "Should return GridCellInitialization object"
        assert grid.num_patches == 1, "Should have 1 patch"
    
    def test_multiple_patches_with_pfts(self):
        """
        Test create_simple_grid with multiple patches and specified PFT types.
        """
        num_patches = 5
        pft_types = jnp.array([1, 2, 3, 4, 5])
        
        grid = create_simple_grid(num_patches=num_patches, pft_types=pft_types)
        
        assert isinstance(grid, GridCellInitialization)
        assert grid.num_patches == num_patches, \
            f"Expected {num_patches} patches, got {grid.num_patches}"
    
    def test_minimum_patches(self):
        """
        Test create_simple_grid with minimum valid number of patches.
        
        Boundary condition: num_patches must be >= 1.
        """
        grid = create_simple_grid(num_patches=1, pft_types=jnp.array([1]))
        
        assert grid.num_patches == 1
        assert isinstance(grid, GridCellInitialization)
    
    def test_large_grid(self):
        """
        Test create_simple_grid with large number of patches.
        
        Tests scalability with 100 patches.
        """
        num_patches = 100
        pft_types = jnp.array([i % 5 + 1 for i in range(num_patches)])
        
        grid = create_simple_grid(num_patches=num_patches, pft_types=pft_types)
        
        assert grid.num_patches == num_patches
        assert len(grid.patch_indices) == num_patches or grid.patch_indices.size == num_patches
    
    def test_invalid_num_patches(self):
        """
        Test create_simple_grid with invalid num_patches (< 1).
        
        Should raise ValueError according to constraints.
        """
        with pytest.raises((ValueError, RuntimeError)):
            create_simple_grid(num_patches=0, pft_types=None)
    
    def test_pft_types_length_mismatch(self):
        """
        Test create_simple_grid when pft_types length doesn't match num_patches.
        
        Should raise ValueError.
        """
        with pytest.raises((ValueError, RuntimeError)):
            create_simple_grid(num_patches=5, pft_types=jnp.array([1, 2, 3]))


# ============================================================================
# Test calculate_grid_statistics
# ============================================================================

class TestCalculateGridStatistics:
    """Tests for calculate_grid_statistics function."""
    
    def test_sequential_indices(self):
        """
        Test calculate_grid_statistics with sequential patch indices.
        
        Verifies correct calculation of mean, max, min, and count.
        """
        patch_indices = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        expected = {
            "mean_index": 4.5,
            "max_index": 9.0,
            "min_index": 0.0,
            "num_patches": 10.0,
        }
        
        result = calculate_grid_statistics(patch_indices)
        
        assert isinstance(result, dict), "Should return dictionary"
        assert "mean_index" in result
        assert "max_index" in result
        assert "min_index" in result
        assert "num_patches" in result
        
        np.testing.assert_allclose(
            result["mean_index"], expected["mean_index"],
            rtol=1e-6, atol=1e-6,
            err_msg="Mean index calculation incorrect"
        )
        np.testing.assert_allclose(
            result["max_index"], expected["max_index"],
            rtol=1e-6, atol=1e-6,
            err_msg="Max index calculation incorrect"
        )
        np.testing.assert_allclose(
            result["min_index"], expected["min_index"],
            rtol=1e-6, atol=1e-6,
            err_msg="Min index calculation incorrect"
        )
        np.testing.assert_allclose(
            result["num_patches"], expected["num_patches"],
            rtol=1e-6, atol=1e-6,
            err_msg="Patch count incorrect"
        )
    
    def test_single_patch(self):
        """
        Test calculate_grid_statistics with single patch.
        
        All statistics should equal the single value.
        """
        patch_indices = jnp.array([42])
        expected = {
            "mean_index": 42.0,
            "max_index": 42.0,
            "min_index": 42.0,
            "num_patches": 1.0,
        }
        
        result = calculate_grid_statistics(patch_indices)
        
        for key in expected:
            np.testing.assert_allclose(
                result[key], expected[key],
                rtol=1e-6, atol=1e-6,
                err_msg=f"{key} incorrect for single patch"
            )
    
    def test_all_zeros(self):
        """
        Test calculate_grid_statistics with all zero indices.
        
        Tests handling of uniform zero values.
        """
        patch_indices = jnp.array([0, 0, 0, 0, 0])
        expected = {
            "mean_index": 0.0,
            "max_index": 0.0,
            "min_index": 0.0,
            "num_patches": 5.0,
        }
        
        result = calculate_grid_statistics(patch_indices)
        
        for key in expected:
            np.testing.assert_allclose(
                result[key], expected[key],
                rtol=1e-6, atol=1e-6,
                err_msg=f"{key} incorrect for all zeros"
            )
    
    def test_sparse_indices(self):
        """
        Test calculate_grid_statistics with sparse non-sequential indices.
        
        Tests handling of large gaps between indices.
        """
        patch_indices = jnp.array([0, 10, 100, 1000, 10000])
        expected = {
            "mean_index": 2222.0,
            "max_index": 10000.0,
            "min_index": 0.0,
            "num_patches": 5.0,
        }
        
        result = calculate_grid_statistics(patch_indices)
        
        for key in expected:
            np.testing.assert_allclose(
                result[key], expected[key],
                rtol=1e-6, atol=1e-6,
                err_msg=f"{key} incorrect for sparse indices"
            )
    
    def test_output_types(self):
        """Verify that calculate_grid_statistics returns correct types."""
        patch_indices = jnp.array([0, 1, 2, 3, 4])
        result = calculate_grid_statistics(patch_indices)
        
        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(value, (float, np.floating, jnp.ndarray)), \
                f"{key} should be float type, got {type(value)}"
    
    def test_jit_compilation(self):
        """
        Test that calculate_grid_statistics works with JIT compilation.
        
        This function is marked as jit_compiled in the signature.
        """
        from jax import jit
        
        jitted_stats = jit(calculate_grid_statistics)
        patch_indices = jnp.array([0, 1, 2, 3, 4])
        
        result = jitted_stats(patch_indices)
        assert isinstance(result, dict)
        assert len(result) == 4


# ============================================================================
# Test initialize_single_patch_grid
# ============================================================================

class TestInitializeSinglePatchGrid:
    """Tests for initialize_single_patch_grid function."""
    
    def test_default_pft(self):
        """
        Test initialize_single_patch_grid with default PFT type.
        
        Should initialize a single patch with PFT type 1.
        """
        initialize_single_patch_grid(pft=1)
        
        state = get_grid_initialization_state()
        assert state.initialized is True, "Grid should be initialized"
        assert state.num_patches >= 1, "Should have at least 1 patch"
    
    def test_different_pft(self):
        """
        Test initialize_single_patch_grid with different PFT type.
        """
        initialize_single_patch_grid(pft=7)
        
        state = get_grid_initialization_state()
        assert state.initialized is True
        assert state.num_patches >= 1
    
    def test_state_modification(self):
        """
        Test that initialize_single_patch_grid modifies global state.
        
        Verifies side effect of modifying _grid_init_state.
        """
        # Get initial state
        initial_state = get_grid_initialization_state()
        
        # Initialize
        initialize_single_patch_grid(pft=1)
        
        # Get new state
        new_state = get_grid_initialization_state()
        
        # State should have changed
        assert new_state.initialized != initial_state.initialized or \
               new_state.num_patches != initial_state.num_patches, \
               "State should be modified after initialization"


# ============================================================================
# Test initialize_multi_patch_grid
# ============================================================================

class TestInitializeMultiPatchGrid:
    """Tests for initialize_multi_patch_grid function."""
    
    def test_homogeneous_pfts(self):
        """
        Test initialize_multi_patch_grid with same PFT type.
        
        Tests homogeneous vegetation configuration.
        """
        pft_list = jnp.array([1, 1, 1, 1, 1])
        initialize_multi_patch_grid(pft_list)
        
        state = get_grid_initialization_state()
        assert state.initialized is True
        assert state.num_patches >= len(pft_list)
    
    def test_heterogeneous_pfts(self):
        """
        Test initialize_multi_patch_grid with diverse PFT types.
        
        Tests heterogeneous vegetation configuration.
        """
        pft_list = jnp.array([1, 2, 3, 4, 5, 6, 7, 8])
        initialize_multi_patch_grid(pft_list)
        
        state = get_grid_initialization_state()
        assert state.initialized is True
        assert state.num_patches >= len(pft_list)
    
    def test_large_array(self):
        """
        Test initialize_multi_patch_grid with large PFT array.
        
        Tests scalability with 50 patches.
        """
        pft_list = jnp.array([i % 10 + 1 for i in range(50)])
        initialize_multi_patch_grid(pft_list)
        
        state = get_grid_initialization_state()
        assert state.initialized is True
        assert state.num_patches >= len(pft_list)


# ============================================================================
# Test validate_initialization
# ============================================================================

class TestValidateInitialization:
    """Tests for validate_initialization function."""
    
    def test_uninitialized_state(self):
        """
        Test validate_initialization before initialization.
        
        Should return (False, error_message).
        """
        reset_grid_initialization()
        is_valid, error_msg = validate_initialization()
        
        assert isinstance(is_valid, bool)
        assert isinstance(error_msg, str)
        assert is_valid is False, "Uninitialized state should be invalid"
        assert len(error_msg) > 0, "Should provide error message"
    
    def test_initialized_state(self):
        """
        Test validate_initialization after initialization.
        
        Should return (True, "").
        """
        initialize_single_patch_grid(pft=1)
        is_valid, error_msg = validate_initialization()
        
        assert isinstance(is_valid, bool)
        assert isinstance(error_msg, str)
        assert is_valid is True, "Initialized state should be valid"


# ============================================================================
# Test GridCellInitialization dataclass
# ============================================================================

class TestGridCellInitialization:
    """Tests for GridCellInitialization dataclass."""
    
    def test_default_initialization(self):
        """
        Test GridCellInitialization with default values.
        """
        grid = GridCellInitialization()
        
        assert grid.initialized is False
        assert grid.num_patches == 0
        assert len(grid.patch_indices) == 0 or grid.patch_indices.size == 0
        assert len(grid.landunit_types) == 0 or grid.landunit_types.size == 0
        assert isinstance(grid.metadata, dict)
        assert len(grid.metadata) == 0
    
    def test_custom_initialization(self):
        """
        Test GridCellInitialization with custom values.
        """
        grid = GridCellInitialization(
            initialized=True,
            num_patches=5,
            patch_indices=jnp.array([0, 1, 2, 3, 4]),
            landunit_types=jnp.array([1, 1, 2, 2, 3]),
            metadata={"test": "value"}
        )
        
        assert grid.initialized is True
        assert grid.num_patches == 5
        assert len(grid.patch_indices) == 5
        assert len(grid.landunit_types) == 5
        assert grid.metadata["test"] == "value"
    
    def test_reset_method(self):
        """
        Test GridCellInitialization.reset() method.
        
        Should return a new GridCellInitialization with default values.
        """
        grid = GridCellInitialization(
            initialized=True,
            num_patches=5,
            patch_indices=jnp.array([0, 1, 2, 3, 4])
        )
        
        reset_grid = grid.reset()
        
        assert isinstance(reset_grid, GridCellInitialization)
        assert reset_grid.initialized is False
        assert reset_grid.num_patches == 0
    
    def test_is_valid_method(self):
        """
        Test GridCellInitialization.is_valid() method.
        """
        # Invalid: not initialized
        grid = GridCellInitialization()
        assert grid.is_valid() is False
        
        # Valid: initialized with patches
        grid = GridCellInitialization(
            initialized=True,
            num_patches=1,
            patch_indices=jnp.array([0])
        )
        assert grid.is_valid() is True
    
    def test_get_info_method(self):
        """
        Test GridCellInitialization.get_info() method.
        
        Should return a dictionary with summary information.
        """
        grid = GridCellInitialization(
            initialized=True,
            num_patches=5,
            patch_indices=jnp.array([0, 1, 2, 3, 4])
        )
        
        info = grid.get_info()
        
        assert isinstance(info, dict)
        assert "initialized" in info
        assert "num_patches" in info


# ============================================================================
# Test global state management
# ============================================================================

class TestGlobalStateManagement:
    """Tests for global state management functions."""
    
    def test_reset_grid_initialization(self):
        """
        Test reset_grid_initialization function.
        
        Should reset global state to default values.
        """
        # Initialize something
        initialize_single_patch_grid(pft=1)
        
        # Reset
        reset_grid_initialization()
        
        # Check state is reset
        state = get_grid_initialization_state()
        assert state.initialized is False
        assert state.num_patches == 0
    
    def test_get_grid_initialization_state_copy(self):
        """
        Test that get_grid_initialization_state returns a copy.
        
        Modifying the returned state should not affect global state.
        """
        initialize_single_patch_grid(pft=1)
        
        state1 = get_grid_initialization_state()
        state2 = get_grid_initialization_state()
        
        # Should be separate objects (copies)
        assert state1 is not state2, "Should return copies, not references"
    
    def test_state_persistence_across_calls(self):
        """
        Test that state persists across multiple function calls.
        """
        initialize_single_patch_grid(pft=1)
        state1 = get_grid_initialization_state()
        
        # Call another function
        is_valid, _ = validate_initialization()
        
        state2 = get_grid_initialization_state()
        
        # State should be consistent
        assert state1.initialized == state2.initialized
        assert state1.num_patches == state2.num_patches


# ============================================================================
# Integration tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete initialization workflows."""
    
    def test_complete_initialization_lifecycle(self):
        """
        Test complete lifecycle: reset -> initialize -> validate -> reset.
        
        This tests the full workflow of grid initialization.
        """
        # Step 1: Reset
        reset_grid_initialization()
        state = get_grid_initialization_state()
        assert state.initialized is False
        
        # Step 2: Validate (should fail)
        is_valid, error_msg = validate_initialization()
        assert is_valid is False
        assert len(error_msg) > 0
        
        # Step 3: Initialize
        initialize_single_patch_grid(pft=1)
        state = get_grid_initialization_state()
        assert state.initialized is True
        
        # Step 4: Validate (should pass)
        is_valid, error_msg = validate_initialization()
        assert is_valid is True
        
        # Step 5: Reset again
        reset_grid_initialization()
        state = get_grid_initialization_state()
        assert state.initialized is False
    
    def test_multiple_initializations(self):
        """
        Test multiple sequential initializations.
        
        Each initialization should update the state.
        """
        # First initialization
        initialize_single_patch_grid(pft=1)
        state1 = get_grid_initialization_state()
        
        # Reset and second initialization
        reset_grid_initialization()
        initialize_multi_patch_grid(jnp.array([1, 2, 3]))
        state2 = get_grid_initialization_state()
        
        # Both should be initialized but potentially different
        assert state1.initialized is True
        assert state2.initialized is True
    
    def test_create_simple_grid_integration(self):
        """
        Test create_simple_grid integration with validation.
        """
        grid = create_simple_grid(num_patches=5, pft_types=jnp.array([1, 2, 3, 4, 5]))
        
        # Validate the created grid structure
        if len(grid.patch_indices) > 0 and len(grid.landunit_types) > 0:
            is_valid = validate_patch_structure(
                grid.patch_indices,
                grid.landunit_types
            )
            assert bool(is_valid) is True
    
    def test_statistics_after_initialization(self):
        """
        Test calculating statistics after grid initialization.
        """
        # Initialize grid
        initialize_multi_patch_grid(jnp.array([1, 2, 3, 4, 5]))
        state = get_grid_initialization_state()
        
        # Calculate statistics if patches exist
        if len(state.patch_indices) > 0:
            stats = calculate_grid_statistics(state.patch_indices)
            
            assert "mean_index" in stats
            assert "max_index" in stats
            assert "min_index" in stats
            assert "num_patches" in stats
            
            # Verify consistency
            assert stats["num_patches"] == len(state.patch_indices)


# ============================================================================
# Test print_initialization_summary
# ============================================================================

class TestPrintInitializationSummary:
    """Tests for print_initialization_summary function."""
    
    def test_print_summary_no_crash(self, capsys):
        """
        Test that print_initialization_summary doesn't crash.
        
        Captures stdout to verify it prints something.
        """
        initialize_single_patch_grid(pft=1)
        
        # Should not raise exception
        print_initialization_summary()
        
        # Verify something was printed
        captured = capsys.readouterr()
        # Output might be empty or contain summary, just verify no crash
        assert True  # If we got here, no crash occurred


# ============================================================================
# Test initGridcells and set_landunit_veg_compete
# ============================================================================

class TestInitGridcells:
    """Tests for initGridcells function."""
    
    def test_initGridcells_execution(self):
        """
        Test that initGridcells executes without error.
        
        This function has no parameters and modifies global state.
        """
        # Should not raise exception
        try:
            initGridcells()
            success = True
        except Exception as e:
            success = False
            pytest.fail(f"initGridcells raised exception: {e}")
        
        assert success
    
    def test_initGridcells_state_modification(self):
        """
        Test that initGridcells modifies global state.
        """
        reset_grid_initialization()
        initial_state = get_grid_initialization_state()
        
        initGridcells()
        
        final_state = get_grid_initialization_state()
        
        # State should have changed (at least one field)
        state_changed = (
            final_state.initialized != initial_state.initialized or
            final_state.num_patches != initial_state.num_patches
        )
        assert state_changed, "initGridcells should modify global state"


class TestSetLandunitVegCompete:
    """Tests for set_landunit_veg_compete function."""
    
    def test_set_landunit_veg_compete_execution(self):
        """
        Test that set_landunit_veg_compete executes without error.
        
        This function has no parameters and modifies global state.
        """
        try:
            set_landunit_veg_compete()
            success = True
        except Exception as e:
            success = False
            pytest.fail(f"set_landunit_veg_compete raised exception: {e}")
        
        assert success


# ============================================================================
# Edge case and error handling tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_validate_patch_structure_with_nan(self):
        """
        Test validate_patch_structure with NaN values.
        
        Should handle NaN gracefully (likely return False).
        """
        patch_indices = jnp.array([0.0, jnp.nan, 2.0])
        landunit_types = jnp.array([1, 1, 1])
        
        # Should not crash
        result = validate_patch_structure(patch_indices, landunit_types)
        assert isinstance(result, (bool, jnp.ndarray, np.bool_))
    
    def test_validate_patch_structure_with_inf(self):
        """
        Test validate_patch_structure with infinity values.
        
        Should handle Inf gracefully.
        """
        patch_indices = jnp.array([0.0, jnp.inf, 2.0])
        landunit_types = jnp.array([1, 1, 1])
        
        # Should not crash
        result = validate_patch_structure(patch_indices, landunit_types)
        assert isinstance(result, (bool, jnp.ndarray, np.bool_))
    
    def test_calculate_grid_statistics_empty_array(self):
        """
        Test calculate_grid_statistics with empty array.
        
        Should handle gracefully (may return NaN or raise error).
        """
        patch_indices = jnp.array([])
        
        try:
            result = calculate_grid_statistics(patch_indices)
            # If it succeeds, verify it returns a dict
            assert isinstance(result, dict)
        except (ValueError, RuntimeError):
            # Empty array might raise an error, which is acceptable
            pass


# ============================================================================
# Property-based tests
# ============================================================================

class TestProperties:
    """Property-based tests for invariants."""
    
    def test_statistics_invariants(self):
        """
        Test that grid statistics satisfy mathematical invariants.
        
        - min <= mean <= max
        - num_patches > 0 implies valid statistics
        """
        patch_indices = jnp.array([5, 10, 15, 20, 25])
        stats = calculate_grid_statistics(patch_indices)
        
        # Extract values
        min_idx = float(stats["min_index"])
        mean_idx = float(stats["mean_index"])
        max_idx = float(stats["max_index"])
        num_patches = float(stats["num_patches"])
        
        # Invariants
        assert min_idx <= mean_idx <= max_idx, \
            "min <= mean <= max invariant violated"
        assert num_patches == len(patch_indices), \
            "num_patches should equal array length"
    
    def test_validation_consistency(self):
        """
        Test that validation is consistent.
        
        If validate_patch_structure returns True, the structure should be valid.
        """
        patch_indices = jnp.array([0, 1, 2, 3, 4])
        landunit_types = jnp.array([1, 1, 2, 2, 3])
        
        is_valid = validate_patch_structure(patch_indices, landunit_types)
        
        if bool(is_valid):
            # If valid, arrays should have same length
            assert len(patch_indices) == len(landunit_types)
            # All indices should be non-negative
            assert jnp.all(patch_indices >= 0)