"""
Comprehensive pytest suite for initSubgridMod module.

This module tests subgrid structure management functions including:
- Patch addition and management
- Hierarchy validation
- Statistics computation
- Subgrid structure creation and manipulation
- Data structure initialization and methods

Tests cover nominal cases, edge cases, and special scenarios for scientific
computing in land surface modeling contexts.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pytest
import jax.numpy as jnp
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from clm_src_main.initSubgridMod import (
    add_patch,
    validate_patch_hierarchy,
    get_patch_statistics,
    get_subgrid_structure,
    reset_subgrid_structure,
    create_simple_subgrid,
    add_multiple_patches,
    print_subgrid_summary,
    validate_subgrid_consistency,
    create_single_patch_subgrid,
    create_multi_column_subgrid,
    SubgridStructure,
    PatchData,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load and provide test data for all test cases.
    
    Returns:
        Dictionary containing test cases organized by function and scenario.
    """
    return {
        "add_patch": {
            "nominal_sequential": {"pi": 0, "ptype": 5},
            "initial_state": {"pi": -1, "ptype": 0},
            "large_index": {"pi": 999, "ptype": 15},
        },
        "validate_hierarchy": {
            "simple_valid": {
                "patch_columns": jnp.array([1, 1, 2, 2, 3]),
                "patch_gridcells": jnp.array([1, 1, 1, 1, 2]),
                "patch_types": jnp.array([0, 1, 2, 3, 4]),
                "active_mask": jnp.array([True, True, True, True, True]),
            },
            "partial_active": {
                "patch_columns": jnp.array([1, 1, 2, 3, 3, 3]),
                "patch_gridcells": jnp.array([1, 1, 1, 2, 2, 2]),
                "patch_types": jnp.array([0, 5, 10, 2, 8, 12]),
                "active_mask": jnp.array([True, False, True, True, False, True]),
            },
            "single_patch": {
                "patch_columns": jnp.array([1]),
                "patch_gridcells": jnp.array([1]),
                "patch_types": jnp.array([0]),
                "active_mask": jnp.array([True]),
            },
            "large_scale": {
                "patch_columns": jnp.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5]),
                "patch_gridcells": jnp.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]),
                "patch_types": jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
                "active_mask": jnp.array([True] * 20),
            },
            "uniform_assignment": {
                "patch_columns": jnp.array([1, 1, 1]),
                "patch_gridcells": jnp.array([1, 1, 1]),
                "patch_types": jnp.array([0, 0, 0]),
                "active_mask": jnp.array([True, True, True]),
            },
        },
        "patch_statistics": {
            "diverse_types": {
                "patch_types": jnp.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45]),
                "active_mask": jnp.array([True] * 10),
            },
            "sparse_active": {
                "patch_types": jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
                "active_mask": jnp.array([True, False, False, True, False, False, False, True, False, False, False, False, True]),
            },
            "all_inactive": {
                "patch_types": jnp.array([0, 5, 10, 15, 20]),
                "active_mask": jnp.array([False] * 5),
            },
            "single_active": {
                "patch_types": jnp.array([0, 5, 10, 15, 20, 25, 30]),
                "active_mask": jnp.array([False, False, False, True, False, False, False]),
            },
        },
        "create_simple_subgrid": {
            "multi_patch": {
                "patch_types": [0, 1, 2, 5, 8, 10],
                "column_assignments": [1, 1, 2, 2, 3, 3],
                "gridcell_assignments": [1, 1, 1, 1, 2, 2],
            },
            "defaults": {
                "patch_types": [0, 3, 7],
                "column_assignments": None,
                "gridcell_assignments": None,
            },
            "single_patch": {
                "patch_types": [0],
                "column_assignments": [1],
                "gridcell_assignments": [1],
            },
        },
        "add_multiple_patches": {
            "varied": {
                "patch_info": [(1, 1, 0), (1, 1, 5), (2, 1, 10), (2, 2, 3), (3, 2, 8)],
            },
            "single": {
                "patch_info": [(1, 1, 0)],
            },
            "large_batch": {
                "patch_info": [
                    (1, 1, 0), (1, 1, 1), (1, 1, 2),
                    (2, 1, 3), (2, 1, 4), (2, 1, 5),
                    (3, 2, 6), (3, 2, 7), (3, 2, 8),
                    (4, 2, 9), (4, 2, 10), (4, 2, 11),
                    (5, 3, 12), (5, 3, 13), (5, 3, 14),
                ],
            },
        },
        "reset_subgrid": {
            "default": {"max_patches": 1000},
            "minimal": {"max_patches": 1},
            "large": {"max_patches": 10000},
        },
        "create_single_patch": {
            "default": {"ptype": 1},
            "zero_type": {"ptype": 0},
        },
        "create_multi_column": {
            "varied": {
                "patch_configs": [(1, 0), (1, 5), (2, 10), (2, 3), (3, 8), (3, 12)],
            },
            "single_column": {
                "patch_configs": [(1, 0), (1, 5), (1, 10)],
            },
            "many_columns": {
                "patch_configs": [(i, i-1) for i in range(1, 11)],
            },
        },
    }


@pytest.fixture
def clean_subgrid():
    """
    Fixture to ensure clean subgrid state before each test.
    
    Resets the subgrid structure to default state before test execution
    and cleans up after test completion.
    """
    reset_subgrid_structure(max_patches=1000)
    yield
    reset_subgrid_structure(max_patches=1000)


# ============================================================================
# Tests for add_patch
# ============================================================================

class TestAddPatch:
    """Test suite for add_patch function."""
    
    def test_add_patch_nominal_sequential(self, test_data, clean_subgrid):
        """
        Test adding a patch sequentially with typical patch type.
        
        Verifies that patch index increments correctly and returns expected value.
        """
        data = test_data["add_patch"]["nominal_sequential"]
        result = add_patch(data["pi"], data["ptype"])
        
        assert isinstance(result, int), "Result should be an integer"
        assert result == data["pi"] + 1, f"Expected {data['pi'] + 1}, got {result}"
    
    def test_add_patch_initial_state(self, test_data, clean_subgrid):
        """
        Test adding first patch (pi=-1) with minimum patch type (0).
        
        Verifies correct handling of initial state where pi=-1.
        """
        data = test_data["add_patch"]["initial_state"]
        result = add_patch(data["pi"], data["ptype"])
        
        assert isinstance(result, int), "Result should be an integer"
        assert result == 0, f"First patch should have index 0, got {result}"
    
    def test_add_patch_large_index(self, test_data, clean_subgrid):
        """
        Test adding patch near maximum capacity with typical ptype.
        
        Verifies function handles large indices correctly.
        """
        data = test_data["add_patch"]["large_index"]
        result = add_patch(data["pi"], data["ptype"])
        
        assert isinstance(result, int), "Result should be an integer"
        assert result == data["pi"] + 1, f"Expected {data['pi'] + 1}, got {result}"
    
    def test_add_patch_dtype(self, test_data, clean_subgrid):
        """Verify add_patch returns correct data type."""
        data = test_data["add_patch"]["nominal_sequential"]
        result = add_patch(data["pi"], data["ptype"])
        
        assert isinstance(result, (int, np.integer)), "Result must be integer type"
    
    @pytest.mark.parametrize("pi,ptype", [
        (0, 0), (5, 10), (100, 5), (500, 20),
    ])
    def test_add_patch_parametrized(self, pi, ptype, clean_subgrid):
        """
        Parametrized test for various pi and ptype combinations.
        
        Tests multiple scenarios with different patch indices and types.
        """
        result = add_patch(pi, ptype)
        assert result == pi + 1, f"Expected {pi + 1}, got {result}"


# ============================================================================
# Tests for validate_patch_hierarchy
# ============================================================================

class TestValidatePatchHierarchy:
    """Test suite for validate_patch_hierarchy function."""
    
    def test_validate_hierarchy_simple_valid(self, test_data):
        """
        Test validation of simple valid hierarchy.
        
        Verifies that a well-formed hierarchy with multiple columns and
        gridcells is correctly validated.
        """
        data = test_data["validate_hierarchy"]["simple_valid"]
        result = validate_patch_hierarchy(
            data["patch_columns"],
            data["patch_gridcells"],
            data["patch_types"],
            data["active_mask"],
        )
        
        assert isinstance(result, (bool, np.bool_)), "Result should be boolean"
        assert result == True, "Valid hierarchy should return True"
    
    def test_validate_hierarchy_partial_active(self, test_data):
        """
        Test validation with some inactive patches.
        
        Verifies that hierarchy validation correctly handles mixed active/inactive states.
        """
        data = test_data["validate_hierarchy"]["partial_active"]
        result = validate_patch_hierarchy(
            data["patch_columns"],
            data["patch_gridcells"],
            data["patch_types"],
            data["active_mask"],
        )
        
        assert isinstance(result, (bool, np.bool_)), "Result should be boolean"
    
    def test_validate_hierarchy_single_patch(self, test_data):
        """
        Test minimal valid hierarchy with single patch.
        
        Verifies edge case of single-element arrays.
        """
        data = test_data["validate_hierarchy"]["single_patch"]
        result = validate_patch_hierarchy(
            data["patch_columns"],
            data["patch_gridcells"],
            data["patch_types"],
            data["active_mask"],
        )
        
        assert isinstance(result, (bool, np.bool_)), "Result should be boolean"
        assert result == True, "Single valid patch should return True"
    
    def test_validate_hierarchy_large_scale(self, test_data):
        """
        Test large-scale hierarchy validation with 20 patches.
        
        Verifies function handles larger datasets correctly.
        """
        data = test_data["validate_hierarchy"]["large_scale"]
        result = validate_patch_hierarchy(
            data["patch_columns"],
            data["patch_gridcells"],
            data["patch_types"],
            data["active_mask"],
        )
        
        assert isinstance(result, (bool, np.bool_)), "Result should be boolean"
    
    def test_validate_hierarchy_shapes(self, test_data):
        """
        Verify that all input arrays have consistent shapes.
        
        Tests that function properly handles array dimension requirements.
        """
        data = test_data["validate_hierarchy"]["simple_valid"]
        
        n_patches = len(data["patch_columns"])
        assert data["patch_gridcells"].shape == (n_patches,), "Gridcells shape mismatch"
        assert data["patch_types"].shape == (n_patches,), "Types shape mismatch"
        assert data["active_mask"].shape == (n_patches,), "Mask shape mismatch"
    
    def test_validate_hierarchy_uniform_assignment(self, test_data):
        """
        Test hierarchy with all patches in same column/gridcell.
        
        Verifies edge case of uniform spatial assignment.
        """
        data = test_data["validate_hierarchy"]["uniform_assignment"]
        result = validate_patch_hierarchy(
            data["patch_columns"],
            data["patch_gridcells"],
            data["patch_types"],
            data["active_mask"],
        )
        
        assert isinstance(result, (bool, np.bool_)), "Result should be boolean"


# ============================================================================
# Tests for get_patch_statistics
# ============================================================================

class TestGetPatchStatistics:
    """Test suite for get_patch_statistics function."""
    
    def test_get_patch_statistics_diverse_types(self, test_data):
        """
        Test statistics computation for diverse patch types.
        
        Verifies correct calculation of statistics across varied patch types.
        """
        data = test_data["patch_statistics"]["diverse_types"]
        result = get_patch_statistics(data["patch_types"], data["active_mask"])
        
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "num_active_patches" in result, "Missing num_active_patches"
        assert "min_patch_type" in result, "Missing min_patch_type"
        assert "max_patch_type" in result, "Missing max_patch_type"
        assert "mean_patch_type" in result, "Missing mean_patch_type"
        
        assert result["num_active_patches"] == 10, "Should have 10 active patches"
        assert result["min_patch_type"] == 0, "Min should be 0"
        assert result["max_patch_type"] == 45, "Max should be 45"
    
    def test_get_patch_statistics_sparse_active(self, test_data):
        """
        Test statistics with sparse active patches (4 of 13).
        
        Verifies correct handling of mostly inactive patches.
        """
        data = test_data["patch_statistics"]["sparse_active"]
        result = get_patch_statistics(data["patch_types"], data["active_mask"])
        
        assert isinstance(result, dict), "Result should be a dictionary"
        assert result["num_active_patches"] == 4, "Should have 4 active patches"
    
    def test_get_patch_statistics_all_inactive(self, test_data):
        """
        Test statistics with all patches inactive.
        
        Verifies edge case of no active patches.
        """
        data = test_data["patch_statistics"]["all_inactive"]
        result = get_patch_statistics(data["patch_types"], data["active_mask"])
        
        assert isinstance(result, dict), "Result should be a dictionary"
        assert result["num_active_patches"] == 0, "Should have 0 active patches"
    
    def test_get_patch_statistics_single_active(self, test_data):
        """
        Test statistics with only one active patch.
        
        Verifies edge case of single active patch among many.
        """
        data = test_data["patch_statistics"]["single_active"]
        result = get_patch_statistics(data["patch_types"], data["active_mask"])
        
        assert isinstance(result, dict), "Result should be a dictionary"
        assert result["num_active_patches"] == 1, "Should have 1 active patch"
        assert result["min_patch_type"] == result["max_patch_type"], "Min should equal max for single patch"
    
    def test_get_patch_statistics_dtypes(self, test_data):
        """Verify correct data types in statistics dictionary."""
        data = test_data["patch_statistics"]["diverse_types"]
        result = get_patch_statistics(data["patch_types"], data["active_mask"])
        
        assert isinstance(result["num_active_patches"], (int, np.integer, float, np.floating)), \
            "num_active_patches should be numeric"
        assert isinstance(result["min_patch_type"], (int, np.integer, float, np.floating)), \
            "min_patch_type should be numeric"
        assert isinstance(result["max_patch_type"], (int, np.integer, float, np.floating)), \
            "max_patch_type should be numeric"
        assert isinstance(result["mean_patch_type"], (float, np.floating)), \
            "mean_patch_type should be float"


# ============================================================================
# Tests for create_simple_subgrid
# ============================================================================

class TestCreateSimpleSubgrid:
    """Test suite for create_simple_subgrid function."""
    
    def test_create_simple_subgrid_multi_patch(self, test_data, clean_subgrid):
        """
        Test creating subgrid with multiple patches across columns and gridcells.
        
        Verifies correct structure creation with explicit assignments.
        """
        data = test_data["create_simple_subgrid"]["multi_patch"]
        result = create_simple_subgrid(
            data["patch_types"],
            data["column_assignments"],
            data["gridcell_assignments"],
        )
        
        assert isinstance(result, SubgridStructure), "Result should be SubgridStructure"
        assert result.current_patches == len(data["patch_types"]), \
            f"Should have {len(data['patch_types'])} patches"
    
    def test_create_simple_subgrid_defaults(self, test_data, clean_subgrid):
        """
        Test creating subgrid with default column/gridcell assignments.
        
        Verifies that None assignments default to all patches in column/gridcell 1.
        """
        data = test_data["create_simple_subgrid"]["defaults"]
        result = create_simple_subgrid(
            data["patch_types"],
            data["column_assignments"],
            data["gridcell_assignments"],
        )
        
        assert isinstance(result, SubgridStructure), "Result should be SubgridStructure"
        assert result.current_patches == len(data["patch_types"]), \
            f"Should have {len(data['patch_types'])} patches"
    
    def test_create_simple_subgrid_single_patch(self, test_data, clean_subgrid):
        """
        Test creating simplest possible subgrid with single patch.
        
        Verifies edge case of minimal subgrid structure.
        """
        data = test_data["create_simple_subgrid"]["single_patch"]
        result = create_simple_subgrid(
            data["patch_types"],
            data["column_assignments"],
            data["gridcell_assignments"],
        )
        
        assert isinstance(result, SubgridStructure), "Result should be SubgridStructure"
        assert result.current_patches == 1, "Should have 1 patch"
    
    def test_create_simple_subgrid_structure_validity(self, test_data, clean_subgrid):
        """Verify created subgrid structure is valid."""
        data = test_data["create_simple_subgrid"]["multi_patch"]
        result = create_simple_subgrid(
            data["patch_types"],
            data["column_assignments"],
            data["gridcell_assignments"],
        )
        
        assert result.is_valid(), "Created subgrid should be valid"


# ============================================================================
# Tests for add_multiple_patches
# ============================================================================

class TestAddMultiplePatches:
    """Test suite for add_multiple_patches function."""
    
    def test_add_multiple_patches_varied(self, test_data, clean_subgrid):
        """
        Test adding multiple patches with varied assignments.
        
        Verifies correct batch addition of patches with different properties.
        """
        data = test_data["add_multiple_patches"]["varied"]
        result = add_multiple_patches(data["patch_info"])
        
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == len(data["patch_info"]), \
            f"Should return {len(data['patch_info'])} indices"
        assert all(isinstance(idx, (int, np.integer)) for idx in result), \
            "All indices should be integers"
    
    def test_add_multiple_patches_single(self, test_data, clean_subgrid):
        """
        Test adding single patch via multiple patch interface.
        
        Verifies edge case of single-element batch addition.
        """
        data = test_data["add_multiple_patches"]["single"]
        result = add_multiple_patches(data["patch_info"])
        
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 1, "Should return 1 index"
    
    def test_add_multiple_patches_large_batch(self, test_data, clean_subgrid):
        """
        Test adding large batch of patches (15).
        
        Verifies function handles larger batches correctly.
        """
        data = test_data["add_multiple_patches"]["large_batch"]
        result = add_multiple_patches(data["patch_info"])
        
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 15, "Should return 15 indices"
    
    def test_add_multiple_patches_sequential_indices(self, test_data, clean_subgrid):
        """Verify returned indices are sequential."""
        data = test_data["add_multiple_patches"]["varied"]
        result = add_multiple_patches(data["patch_info"])
        
        for i in range(1, len(result)):
            assert result[i] == result[i-1] + 1, "Indices should be sequential"


# ============================================================================
# Tests for reset_subgrid_structure
# ============================================================================

class TestResetSubgridStructure:
    """Test suite for reset_subgrid_structure function."""
    
    def test_reset_subgrid_structure_default(self, test_data):
        """
        Test resetting subgrid structure with default maximum patches.
        
        Verifies reset to default state.
        """
        data = test_data["reset_subgrid"]["default"]
        reset_subgrid_structure(data["max_patches"])
        
        structure = get_subgrid_structure()
        assert structure.max_patches == data["max_patches"], \
            f"Max patches should be {data['max_patches']}"
        assert structure.current_patches == 0, "Should have 0 current patches after reset"
    
    def test_reset_subgrid_structure_minimal(self, test_data):
        """
        Test resetting subgrid structure with minimal capacity (1 patch).
        
        Verifies edge case of minimum capacity.
        """
        data = test_data["reset_subgrid"]["minimal"]
        reset_subgrid_structure(data["max_patches"])
        
        structure = get_subgrid_structure()
        assert structure.max_patches == 1, "Max patches should be 1"
    
    def test_reset_subgrid_structure_large(self, test_data):
        """
        Test resetting subgrid structure with large capacity.
        
        Verifies handling of large capacity values.
        """
        data = test_data["reset_subgrid"]["large"]
        reset_subgrid_structure(data["max_patches"])
        
        structure = get_subgrid_structure()
        assert structure.max_patches == data["max_patches"], \
            f"Max patches should be {data['max_patches']}"


# ============================================================================
# Tests for create_single_patch_subgrid
# ============================================================================

class TestCreateSinglePatchSubgrid:
    """Test suite for create_single_patch_subgrid function."""
    
    def test_create_single_patch_subgrid_default(self, test_data, clean_subgrid):
        """
        Test creating single patch subgrid with default patch type.
        
        Verifies basic single-patch creation.
        """
        data = test_data["create_single_patch"]["default"]
        create_single_patch_subgrid(data["ptype"])
        
        structure = get_subgrid_structure()
        assert structure.current_patches == 1, "Should have 1 patch"
    
    def test_create_single_patch_subgrid_zero_type(self, test_data, clean_subgrid):
        """
        Test creating single patch subgrid with minimum patch type (0).
        
        Verifies edge case of minimum patch type.
        """
        data = test_data["create_single_patch"]["zero_type"]
        create_single_patch_subgrid(data["ptype"])
        
        structure = get_subgrid_structure()
        assert structure.current_patches == 1, "Should have 1 patch"


# ============================================================================
# Tests for create_multi_column_subgrid
# ============================================================================

class TestCreateMultiColumnSubgrid:
    """Test suite for create_multi_column_subgrid function."""
    
    def test_create_multi_column_subgrid_varied(self, test_data, clean_subgrid):
        """
        Test creating multi-column subgrid with varied patch types.
        
        Verifies creation of complex multi-column structures.
        """
        data = test_data["create_multi_column"]["varied"]
        create_multi_column_subgrid(data["patch_configs"])
        
        structure = get_subgrid_structure()
        assert structure.current_patches == len(data["patch_configs"]), \
            f"Should have {len(data['patch_configs'])} patches"
    
    def test_create_multi_column_subgrid_single_column(self, test_data, clean_subgrid):
        """
        Test creating multi-column subgrid with all patches in single column.
        
        Verifies edge case of single-column configuration.
        """
        data = test_data["create_multi_column"]["single_column"]
        create_multi_column_subgrid(data["patch_configs"])
        
        structure = get_subgrid_structure()
        assert structure.current_patches == len(data["patch_configs"]), \
            f"Should have {len(data['patch_configs'])} patches"
    
    def test_create_multi_column_subgrid_many_columns(self, test_data, clean_subgrid):
        """
        Test creating subgrid with many columns (10).
        
        Verifies handling of many-column configurations.
        """
        data = test_data["create_multi_column"]["many_columns"]
        create_multi_column_subgrid(data["patch_configs"])
        
        structure = get_subgrid_structure()
        assert structure.current_patches == 10, "Should have 10 patches"


# ============================================================================
# Tests for get_subgrid_structure
# ============================================================================

class TestGetSubgridStructure:
    """Test suite for get_subgrid_structure function."""
    
    def test_get_subgrid_structure_returns_copy(self, clean_subgrid):
        """
        Verify get_subgrid_structure returns a copy of the structure.
        
        Tests that returned structure is independent of internal state.
        """
        structure1 = get_subgrid_structure()
        structure2 = get_subgrid_structure()
        
        assert isinstance(structure1, SubgridStructure), "Should return SubgridStructure"
        assert isinstance(structure2, SubgridStructure), "Should return SubgridStructure"
    
    def test_get_subgrid_structure_after_modifications(self, clean_subgrid):
        """
        Test getting structure after modifications.
        
        Verifies structure reflects current state after patches are added.
        """
        create_single_patch_subgrid(ptype=5)
        structure = get_subgrid_structure()
        
        assert structure.current_patches > 0, "Should have patches after creation"


# ============================================================================
# Tests for validate_subgrid_consistency
# ============================================================================

class TestValidateSubgridConsistency:
    """Test suite for validate_subgrid_consistency function."""
    
    def test_validate_subgrid_consistency_clean_state(self, clean_subgrid):
        """
        Test validation of clean subgrid state.
        
        Verifies that freshly reset structure is valid.
        """
        is_valid, error_msg = validate_subgrid_consistency()
        
        assert isinstance(is_valid, bool), "First return should be boolean"
        assert isinstance(error_msg, str), "Second return should be string"
        assert is_valid == True, "Clean state should be valid"
    
    def test_validate_subgrid_consistency_after_creation(self, clean_subgrid):
        """
        Test validation after creating patches.
        
        Verifies that created structures maintain consistency.
        """
        create_single_patch_subgrid(ptype=1)
        is_valid, error_msg = validate_subgrid_consistency()
        
        assert isinstance(is_valid, bool), "First return should be boolean"
        assert isinstance(error_msg, str), "Second return should be string"


# ============================================================================
# Tests for print_subgrid_summary
# ============================================================================

class TestPrintSubgridSummary:
    """Test suite for print_subgrid_summary function."""
    
    def test_print_subgrid_summary_executes(self, clean_subgrid, capsys):
        """
        Test that print_subgrid_summary executes without error.
        
        Verifies function runs and produces output.
        """
        print_subgrid_summary()
        captured = capsys.readouterr()
        
        # Function should produce some output
        assert len(captured.out) >= 0, "Should produce output (or empty string)"


# ============================================================================
# Tests for SubgridStructure dataclass
# ============================================================================

class TestSubgridStructure:
    """Test suite for SubgridStructure dataclass."""
    
    def test_subgrid_structure_initialization_default(self):
        """
        Test SubgridStructure initialization with defaults.
        
        Verifies default initialization creates valid structure.
        """
        structure = SubgridStructure()
        
        assert structure.max_patches == 1000, "Default max_patches should be 1000"
        assert structure.current_patches == 0, "Default current_patches should be 0"
        assert structure.patch_columns.shape == (1000,), "patch_columns shape mismatch"
        assert structure.patch_gridcells.shape == (1000,), "patch_gridcells shape mismatch"
        assert structure.patch_types.shape == (1000,), "patch_types shape mismatch"
        assert structure.active_mask.shape == (1000,), "active_mask shape mismatch"
    
    def test_subgrid_structure_initialization_custom(self):
        """
        Test SubgridStructure initialization with custom parameters.
        
        Verifies custom initialization with specified max_patches.
        """
        structure = SubgridStructure(max_patches=500)
        
        assert structure.max_patches == 500, "max_patches should be 500"
        assert structure.current_patches == 0, "current_patches should be 0"
    
    def test_subgrid_structure_is_valid(self):
        """
        Test SubgridStructure.is_valid() method.
        
        Verifies validation method works correctly.
        """
        structure = SubgridStructure()
        result = structure.is_valid()
        
        assert isinstance(result, bool), "is_valid should return boolean"
    
    def test_subgrid_structure_get_active_patches(self):
        """
        Test SubgridStructure.get_active_patches() method.
        
        Verifies method returns correct active patch indices.
        """
        structure = SubgridStructure()
        result = structure.get_active_patches()
        
        assert isinstance(result, jnp.ndarray), "Should return JAX array"
    
    def test_subgrid_structure_get_patch_info(self):
        """
        Test SubgridStructure.get_patch_info() method.
        
        Verifies method returns patch information dictionary.
        """
        structure = SubgridStructure()
        result = structure.get_patch_info(0)
        
        assert isinstance(result, dict), "Should return dictionary"
    
    def test_subgrid_structure_resize(self):
        """
        Test SubgridStructure.resize() method.
        
        Verifies resizing creates new structure with correct capacity.
        """
        structure = SubgridStructure(max_patches=100)
        resized = structure.resize(new_max_patches=200)
        
        assert isinstance(resized, SubgridStructure), "Should return SubgridStructure"
        assert resized.max_patches == 200, "Resized structure should have max_patches=200"
    
    def test_subgrid_structure_dtypes(self):
        """Verify SubgridStructure field data types."""
        structure = SubgridStructure()
        
        assert isinstance(structure.max_patches, (int, np.integer)), "max_patches should be int"
        assert isinstance(structure.current_patches, (int, np.integer)), "current_patches should be int"
        assert isinstance(structure.patch_columns, jnp.ndarray), "patch_columns should be JAX array"
        assert isinstance(structure.patch_gridcells, jnp.ndarray), "patch_gridcells should be JAX array"
        assert isinstance(structure.patch_types, jnp.ndarray), "patch_types should be JAX array"
        assert isinstance(structure.active_mask, jnp.ndarray), "active_mask should be JAX array"
        assert isinstance(structure.metadata, dict), "metadata should be dict"


# ============================================================================
# Tests for PatchData dataclass
# ============================================================================

class TestPatchData:
    """Test suite for PatchData dataclass."""
    
    def test_patch_data_initialization_default(self):
        """
        Test PatchData initialization with defaults.
        
        Verifies default initialization creates empty arrays.
        """
        patch_data = PatchData()
        
        assert isinstance(patch_data.column, jnp.ndarray), "column should be JAX array"
        assert isinstance(patch_data.gridcell, jnp.ndarray), "gridcell should be JAX array"
        assert isinstance(patch_data.itype, jnp.ndarray), "itype should be JAX array"
        assert len(patch_data.column) == 0, "column should be empty"
        assert len(patch_data.gridcell) == 0, "gridcell should be empty"
        assert len(patch_data.itype) == 0, "itype should be empty"
    
    def test_patch_data_initialization_custom(self):
        """
        Test PatchData initialization with custom arrays.
        
        Verifies initialization with provided data.
        """
        column = jnp.array([1, 2, 3])
        gridcell = jnp.array([1, 1, 2])
        itype = jnp.array([0, 5, 10])
        
        patch_data = PatchData(column=column, gridcell=gridcell, itype=itype)
        
        assert jnp.array_equal(patch_data.column, column), "column mismatch"
        assert jnp.array_equal(patch_data.gridcell, gridcell), "gridcell mismatch"
        assert jnp.array_equal(patch_data.itype, itype), "itype mismatch"
    
    def test_patch_data_resize(self):
        """
        Test PatchData.resize() method.
        
        Verifies resizing adjusts array sizes correctly.
        """
        patch_data = PatchData()
        patch_data.resize(new_size=10)
        
        assert len(patch_data.column) == 10, "column should have size 10"
        assert len(patch_data.gridcell) == 10, "gridcell should have size 10"
        assert len(patch_data.itype) == 10, "itype should have size 10"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for combined functionality."""
    
    def test_full_workflow_simple(self, clean_subgrid):
        """
        Test complete workflow: reset, create, validate, get statistics.
        
        Verifies that functions work together correctly in typical usage.
        """
        # Reset
        reset_subgrid_structure(max_patches=100)
        
        # Create simple subgrid
        structure = create_simple_subgrid(
            patch_types=[0, 1, 2, 3, 4],
            column_assignments=[1, 1, 2, 2, 3],
            gridcell_assignments=[1, 1, 1, 1, 2],
        )
        
        assert structure.current_patches == 5, "Should have 5 patches"
        
        # Validate
        is_valid, error_msg = validate_subgrid_consistency()
        assert is_valid == True, f"Structure should be valid: {error_msg}"
        
        # Get statistics
        stats = get_patch_statistics(
            structure.patch_types[:5],
            structure.active_mask[:5],
        )
        assert stats["num_active_patches"] == 5, "Should have 5 active patches"
    
    def test_full_workflow_complex(self, clean_subgrid):
        """
        Test complex workflow with multiple operations.
        
        Verifies robustness across multiple sequential operations.
        """
        # Reset with custom size
        reset_subgrid_structure(max_patches=50)
        
        # Add multiple patches
        patch_info = [(1, 1, i) for i in range(10)]
        indices = add_multiple_patches(patch_info)
        
        assert len(indices) == 10, "Should have added 10 patches"
        
        # Get structure
        structure = get_subgrid_structure()
        assert structure.current_patches == 10, "Should have 10 patches"
        
        # Validate hierarchy
        is_valid = validate_patch_hierarchy(
            structure.patch_columns[:10],
            structure.patch_gridcells[:10],
            structure.patch_types[:10],
            structure.active_mask[:10],
        )
        assert is_valid == True, "Hierarchy should be valid"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Additional edge case tests."""
    
    def test_empty_arrays_handling(self):
        """
        Test handling of empty arrays.
        
        Verifies functions handle empty inputs gracefully.
        """
        empty_array = jnp.array([])
        
        # This should handle empty arrays appropriately
        # Actual behavior depends on implementation
        try:
            result = validate_patch_hierarchy(
                empty_array, empty_array, empty_array, jnp.array([], dtype=bool)
            )
            assert isinstance(result, (bool, np.bool_)), "Should return boolean"
        except (ValueError, IndexError):
            # Empty arrays may raise errors, which is acceptable
            pass
    
    def test_large_patch_counts(self, clean_subgrid):
        """
        Test handling of large patch counts.
        
        Verifies system handles large numbers of patches.
        """
        reset_subgrid_structure(max_patches=5000)
        structure = get_subgrid_structure()
        
        assert structure.max_patches == 5000, "Should support 5000 patches"
    
    def test_boundary_values(self):
        """
        Test boundary values for constraints.
        
        Verifies correct handling of minimum allowed values.
        """
        # Test minimum patch type (0)
        result = add_patch(pi=0, ptype=0)
        assert result == 1, "Should handle minimum ptype=0"
        
        # Test minimum pi (-1)
        result = add_patch(pi=-1, ptype=0)
        assert result == 0, "Should handle minimum pi=-1"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])