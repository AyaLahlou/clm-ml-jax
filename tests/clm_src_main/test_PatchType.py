"""
Comprehensive pytest suite for PatchType module.

Tests the patch_type dataclass and associated functions for managing
patch-level data in CLM, including vegetation types, hierarchical relationships,
and PFT classification functions.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clm_src_main.PatchType import (
    PFTType,
    PhenologyType,
    PhotosynthesisType,
    VegetationCategory,
    create_mixed_vegetation_patches,
    create_patch_instance,
    create_single_pft_patches,
    get_pft_name,
    get_pft_statistics,
    get_phenology_type,
    get_photosynthesis_type,
    get_vegetation_category,
    is_c3_plant,
    is_c4_plant,
    is_crop,
    is_grass,
    is_irrigated,
    is_shrub,
    is_tree,
    is_vegetated,
    patch_type,
    print_patch_summary,
    reset_global_patch,
    validate_patch_structure,
)

# Constants from module
ISPVAL = -9999
NAN = -999.0


@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load comprehensive test data for PatchType module.
    
    Returns:
        Dictionary containing all test cases with inputs and expected outputs.
    """
    return {
        "test_cases": [
            {
                "name": "test_single_patch_basic_tree",
                "inputs": {
                    "begp": 0,
                    "endp": 0,
                    "max_patches": 10,
                    "column": jnp.array([0, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999], dtype=int),
                    "gridcell": jnp.array([0, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999], dtype=int),
                    "itype": jnp.array([1, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999], dtype=int),
                },
                "expected": {
                    "is_vegetated": True,
                    "is_tree": True,
                    "is_crop": False,
                    "photosynthesis_type": PhotosynthesisType.C3,
                    "phenology_type": PhenologyType.EVERGREEN,
                },
            },
            {
                "name": "test_multiple_patches_mixed_vegetation",
                "inputs": {
                    "begp": 0,
                    "endp": 4,
                    "max_patches": 10,
                    "column": jnp.array([0, 0, 1, 1, 2, -9999, -9999, -9999, -9999, -9999], dtype=int),
                    "gridcell": jnp.array([0, 0, 0, 0, 1, -9999, -9999, -9999, -9999, -9999], dtype=int),
                    "itype": jnp.array([1, 4, 12, 14, 15, -9999, -9999, -9999, -9999, -9999], dtype=int),
                },
                "expected": {
                    "num_patches": 5,
                    "num_vegetated": 5,
                    "num_trees": 2,
                    "num_crops": 1,
                    "num_c3": 3,
                    "num_c4": 1,
                },
            },
            {
                "name": "test_not_vegetated_patch",
                "inputs": {
                    "begp": 0,
                    "endp": 2,
                    "max_patches": 5,
                    "column": jnp.array([0, 1, 2, -9999, -9999], dtype=int),
                    "gridcell": jnp.array([0, 0, 0, -9999, -9999], dtype=int),
                    "itype": jnp.array([0, 0, 0, -9999, -9999], dtype=int),
                },
                "expected": {
                    "is_vegetated": False,
                    "is_tree": False,
                    "is_crop": False,
                    "num_vegetated": 0,
                },
            },
            {
                "name": "test_maximum_pft_code",
                "inputs": {
                    "begp": 0,
                    "endp": 2,
                    "max_patches": 5,
                    "column": jnp.array([0, 1, 2, -9999, -9999], dtype=int),
                    "gridcell": jnp.array([0, 0, 0, -9999, -9999], dtype=int),
                    "itype": jnp.array([76, 77, 78, -9999, -9999], dtype=int),
                },
                "expected": {
                    "is_vegetated": True,
                    "max_pft_code": 78,
                },
            },
            {
                "name": "test_single_patch_zero_index",
                "inputs": {
                    "begp": 0,
                    "endp": 0,
                    "max_patches": 1,
                    "column": jnp.array([0], dtype=int),
                    "gridcell": jnp.array([0], dtype=int),
                    "itype": jnp.array([13], dtype=int),
                },
                "expected": {
                    "num_patches": 1,
                    "is_grass": True,
                    "is_c3_plant": True,
                },
            },
            {
                "name": "test_all_crop_types",
                "inputs": {
                    "begp": 0,
                    "endp": 3,
                    "max_patches": 10,
                    "column": jnp.array([0, 1, 2, 3, -9999, -9999, -9999, -9999, -9999, -9999], dtype=int),
                    "gridcell": jnp.array([0, 0, 0, 0, -9999, -9999, -9999, -9999, -9999, -9999], dtype=int),
                    "itype": jnp.array([15, 16, 15, 16, -9999, -9999, -9999, -9999, -9999, -9999], dtype=int),
                },
                "expected": {
                    "num_crops": 4,
                    "num_irrigated": 2,
                    "is_crop": True,
                    "is_c3_plant": True,
                },
            },
            {
                "name": "test_all_shrub_types",
                "inputs": {
                    "begp": 0,
                    "endp": 2,
                    "max_patches": 5,
                    "column": jnp.array([0, 1, 2, -9999, -9999], dtype=int),
                    "gridcell": jnp.array([0, 0, 0, -9999, -9999], dtype=int),
                    "itype": jnp.array([9, 10, 11, -9999, -9999], dtype=int),
                },
                "expected": {
                    "num_patches": 3,
                    "is_shrub": True,
                },
            },
            {
                "name": "test_sparse_patch_distribution",
                "inputs": {
                    "begp": 10,
                    "endp": 14,
                    "max_patches": 20,
                    "column": jnp.array(
                        [-9999] * 10 + [5, 6, 7, 8, 9] + [-9999] * 5, dtype=int
                    ),
                    "gridcell": jnp.array(
                        [-9999] * 10 + [2, 2, 3, 3, 4] + [-9999] * 5, dtype=int
                    ),
                    "itype": jnp.array(
                        [-9999] * 10 + [1, 4, 12, 13, 14] + [-9999] * 5, dtype=int
                    ),
                },
                "expected": {
                    "num_patches": 5,
                    "begp": 10,
                    "endp": 14,
                },
            },
            {
                "name": "test_c3_vs_c4_photosynthesis",
                "inputs": {
                    "begp": 0,
                    "endp": 5,
                    "max_patches": 10,
                    "column": jnp.array([0, 0, 1, 1, 2, 2, -9999, -9999, -9999, -9999], dtype=int),
                    "gridcell": jnp.array([0, 0, 0, 0, 0, 0, -9999, -9999, -9999, -9999], dtype=int),
                    "itype": jnp.array([12, 13, 14, 15, 16, 1, -9999, -9999, -9999, -9999], dtype=int),
                },
                "expected": {
                    "num_c3": 5,
                    "num_c4": 1,
                    "num_patches": 6,
                },
            },
        ]
    }


# ============================================================================
# Dataclass Tests
# ============================================================================


def test_patch_type_initialization():
    """Test basic patch_type dataclass initialization with default values."""
    patch = patch_type()
    
    assert patch.begp == 0, "Default begp should be 0"
    assert patch.endp == 0, "Default endp should be 0"
    assert patch.max_patches == 1000, "Default max_patches should be 1000"
    assert len(patch.column) == 0, "Default column array should be empty"
    assert len(patch.gridcell) == 0, "Default gridcell array should be empty"
    assert len(patch.itype) == 0, "Default itype array should be empty"
    assert isinstance(patch.metadata, dict), "Metadata should be a dictionary"


def test_patch_type_custom_initialization():
    """Test patch_type initialization with custom values."""
    column = jnp.array([0, 1, 2], dtype=int)
    gridcell = jnp.array([0, 0, 1], dtype=int)
    itype = jnp.array([1, 4, 12], dtype=int)
    
    patch = patch_type(
        column=column,
        gridcell=gridcell,
        itype=itype,
        begp=0,
        endp=2,
        max_patches=3,
    )
    
    assert patch.begp == 0
    assert patch.endp == 2
    assert patch.max_patches == 3
    assert jnp.array_equal(patch.column, column)
    assert jnp.array_equal(patch.gridcell, gridcell)
    assert jnp.array_equal(patch.itype, itype)


def test_patch_type_array_shapes(test_data):
    """Test that patch arrays have consistent shapes."""
    for test_case in test_data["test_cases"]:
        inputs = test_case["inputs"]
        max_patches = inputs["max_patches"]
        
        patch = patch_type(
            column=inputs["column"],
            gridcell=inputs["gridcell"],
            itype=inputs["itype"],
            begp=inputs["begp"],
            endp=inputs["endp"],
            max_patches=max_patches,
        )
        
        assert patch.column.shape == (max_patches,), \
            f"Column array shape mismatch in {test_case['name']}"
        assert patch.gridcell.shape == (max_patches,), \
            f"Gridcell array shape mismatch in {test_case['name']}"
        assert patch.itype.shape == (max_patches,), \
            f"Itype array shape mismatch in {test_case['name']}"


def test_patch_type_array_dtypes(test_data):
    """Test that patch arrays have correct data types."""
    for test_case in test_data["test_cases"]:
        inputs = test_case["inputs"]
        
        patch = patch_type(
            column=inputs["column"],
            gridcell=inputs["gridcell"],
            itype=inputs["itype"],
            begp=inputs["begp"],
            endp=inputs["endp"],
            max_patches=inputs["max_patches"],
        )
        
        assert jnp.issubdtype(patch.column.dtype, jnp.integer), \
            f"Column dtype should be integer in {test_case['name']}"
        assert jnp.issubdtype(patch.gridcell.dtype, jnp.integer), \
            f"Gridcell dtype should be integer in {test_case['name']}"
        assert jnp.issubdtype(patch.itype.dtype, jnp.integer), \
            f"Itype dtype should be integer in {test_case['name']}"


# ============================================================================
# PFT Classification Function Tests
# ============================================================================


@pytest.mark.parametrize("pft_code,expected", [
    (0, False),  # Not vegetated
    (1, True),   # Needleleaf evergreen temperate tree
    (4, True),   # Broadleaf evergreen tropical tree
    (12, True),  # C3 arctic grass
    (15, True),  # C3 crop
    (78, True),  # Maximum PFT code
])
def test_is_vegetated(pft_code, expected):
    """Test is_vegetated function for various PFT codes."""
    result = is_vegetated(pft_code)
    assert result == expected, \
        f"is_vegetated({pft_code}) should return {expected}"


@pytest.mark.parametrize("pft_code,expected", [
    (0, False),  # Not vegetated
    (1, True),   # Needleleaf evergreen temperate tree
    (2, True),   # Needleleaf evergreen boreal tree
    (8, True),   # Broadleaf deciduous boreal tree
    (9, False),  # Broadleaf evergreen shrub
    (12, False), # C3 arctic grass
    (15, False), # C3 crop
])
def test_is_tree(pft_code, expected):
    """Test is_tree function for various PFT codes."""
    result = is_tree(pft_code)
    assert result == expected, \
        f"is_tree({pft_code}) should return {expected}"


@pytest.mark.parametrize("pft_code,expected", [
    (0, False),  # Not vegetated
    (1, False),  # Tree
    (9, True),   # Broadleaf evergreen shrub
    (10, True),  # Broadleaf deciduous temperate shrub
    (11, True),  # Broadleaf deciduous boreal shrub
    (12, False), # Grass
])
def test_is_shrub(pft_code, expected):
    """Test is_shrub function for various PFT codes."""
    result = is_shrub(pft_code)
    assert result == expected, \
        f"is_shrub({pft_code}) should return {expected}"


@pytest.mark.parametrize("pft_code,expected", [
    (0, False),  # Not vegetated
    (1, False),  # Tree
    (9, False),  # Shrub
    (12, True),  # C3 arctic grass
    (13, True),  # C3 non-arctic grass
    (14, True),  # C4 grass
    (15, False), # Crop
])
def test_is_grass(pft_code, expected):
    """Test is_grass function for various PFT codes."""
    result = is_grass(pft_code)
    assert result == expected, \
        f"is_grass({pft_code}) should return {expected}"


@pytest.mark.parametrize("pft_code,expected", [
    (0, False),  # Not vegetated
    (1, False),  # Tree
    (12, False), # Grass
    (15, True),  # C3 crop
    (16, True),  # C3 irrigated
])
def test_is_crop(pft_code, expected):
    """Test is_crop function for various PFT codes."""
    result = is_crop(pft_code)
    assert result == expected, \
        f"is_crop({pft_code}) should return {expected}"


@pytest.mark.parametrize("pft_code,expected", [
    (0, False),  # Not vegetated
    (15, False), # C3 crop (not irrigated)
    (16, True),  # C3 irrigated
])
def test_is_irrigated(pft_code, expected):
    """Test is_irrigated function for various PFT codes."""
    result = is_irrigated(pft_code)
    assert result == expected, \
        f"is_irrigated({pft_code}) should return {expected}"


@pytest.mark.parametrize("pft_code,expected", [
    (0, False),  # Not vegetated
    (1, True),   # Needleleaf tree (C3)
    (12, True),  # C3 arctic grass
    (13, True),  # C3 non-arctic grass
    (14, False), # C4 grass
    (15, True),  # C3 crop
])
def test_is_c3_plant(pft_code, expected):
    """Test is_c3_plant function for various PFT codes."""
    result = is_c3_plant(pft_code)
    assert result == expected, \
        f"is_c3_plant({pft_code}) should return {expected}"


@pytest.mark.parametrize("pft_code,expected", [
    (0, False),  # Not vegetated
    (1, False),  # C3 tree
    (12, False), # C3 grass
    (14, True),  # C4 grass
    (15, False), # C3 crop
])
def test_is_c4_plant(pft_code, expected):
    """Test is_c4_plant function for various PFT codes."""
    result = is_c4_plant(pft_code)
    assert result == expected, \
        f"is_c4_plant({pft_code}) should return {expected}"


# ============================================================================
# PFT Information Function Tests
# ============================================================================


def test_get_pft_name_valid_codes():
    """Test get_pft_name returns valid strings for known PFT codes."""
    test_codes = [0, 1, 4, 9, 12, 15, 16]
    
    for code in test_codes:
        name = get_pft_name(code)
        assert isinstance(name, str), f"PFT name for code {code} should be a string"
        assert len(name) > 0, f"PFT name for code {code} should not be empty"


def test_get_vegetation_category():
    """Test get_vegetation_category returns correct enum values."""
    assert get_vegetation_category(0) == VegetationCategory.NOT_VEGETATED
    assert get_vegetation_category(1) == VegetationCategory.TREE
    assert get_vegetation_category(9) == VegetationCategory.SHRUB
    assert get_vegetation_category(12) == VegetationCategory.GRASS
    assert get_vegetation_category(15) == VegetationCategory.CROP


def test_get_photosynthesis_type():
    """Test get_photosynthesis_type returns correct enum values."""
    assert get_photosynthesis_type(0) == PhotosynthesisType.NONE
    assert get_photosynthesis_type(1) == PhotosynthesisType.C3
    assert get_photosynthesis_type(12) == PhotosynthesisType.C3
    assert get_photosynthesis_type(14) == PhotosynthesisType.C4


def test_get_phenology_type():
    """Test get_phenology_type returns correct enum values."""
    assert get_phenology_type(0) == PhenologyType.NOT_VEGETATED
    assert get_phenology_type(1) == PhenologyType.EVERGREEN
    assert get_phenology_type(3) == PhenologyType.DECIDUOUS


# ============================================================================
# Patch Instance Creation Tests
# ============================================================================


def test_create_patch_instance_default():
    """Test create_patch_instance with default parameters."""
    patch = create_patch_instance()
    
    assert patch.begp == 0
    assert patch.endp == 0
    assert patch.max_patches == 1000
    assert isinstance(patch, patch_type)


def test_create_patch_instance_custom():
    """Test create_patch_instance with custom parameters."""
    patch = create_patch_instance(begp=5, endp=10, max_patches=20)
    
    assert patch.begp == 5
    assert patch.endp == 10
    assert patch.max_patches == 20


def test_create_single_pft_patches():
    """Test create_single_pft_patches creates patches with specified PFT."""
    pft_code = 1  # Needleleaf evergreen temperate tree
    num_patches = 5
    
    patch = create_single_pft_patches(pft_code, num_patches)
    
    assert patch.endp - patch.begp + 1 == num_patches
    
    # Check that all active patches have the correct PFT
    for i in range(patch.begp, patch.endp + 1):
        assert patch.itype[i] == pft_code, \
            f"Patch {i} should have PFT code {pft_code}"


def test_create_mixed_vegetation_patches():
    """Test create_mixed_vegetation_patches with multiple PFT codes."""
    pft_codes = [1, 4, 12, 14, 15]
    
    patch = create_mixed_vegetation_patches(pft_codes)
    
    assert patch.endp - patch.begp + 1 == len(pft_codes)
    
    # Check that patches have the specified PFT codes
    for i, expected_pft in enumerate(pft_codes):
        assert patch.itype[patch.begp + i] == expected_pft, \
            f"Patch {i} should have PFT code {expected_pft}"


# ============================================================================
# Patch Statistics Tests
# ============================================================================


def test_get_pft_statistics_single_patch(test_data):
    """Test get_pft_statistics with single patch."""
    test_case = test_data["test_cases"][0]  # Single patch basic tree
    inputs = test_case["inputs"]
    
    patch = patch_type(
        column=inputs["column"],
        gridcell=inputs["gridcell"],
        itype=inputs["itype"],
        begp=inputs["begp"],
        endp=inputs["endp"],
        max_patches=inputs["max_patches"],
    )
    
    stats = get_pft_statistics(patch)
    
    assert stats["num_patches"] == 1
    assert stats["num_vegetated"] == 1
    assert stats["num_trees"] == 1
    assert stats["num_crops"] == 0


def test_get_pft_statistics_mixed_vegetation(test_data):
    """Test get_pft_statistics with mixed vegetation."""
    test_case = test_data["test_cases"][1]  # Multiple patches mixed vegetation
    inputs = test_case["inputs"]
    expected = test_case["expected"]
    
    patch = patch_type(
        column=inputs["column"],
        gridcell=inputs["gridcell"],
        itype=inputs["itype"],
        begp=inputs["begp"],
        endp=inputs["endp"],
        max_patches=inputs["max_patches"],
    )
    
    stats = get_pft_statistics(patch)
    
    assert stats["num_patches"] == expected["num_patches"]
    assert stats["num_vegetated"] == expected["num_vegetated"]
    assert stats["num_trees"] == expected["num_trees"]
    assert stats["num_crops"] == expected["num_crops"]
    assert stats["num_c3"] == expected["num_c3"]
    assert stats["num_c4"] == expected["num_c4"]


def test_get_pft_statistics_not_vegetated(test_data):
    """Test get_pft_statistics with non-vegetated patches."""
    test_case = test_data["test_cases"][2]  # Not vegetated patch
    inputs = test_case["inputs"]
    
    patch = patch_type(
        column=inputs["column"],
        gridcell=inputs["gridcell"],
        itype=inputs["itype"],
        begp=inputs["begp"],
        endp=inputs["endp"],
        max_patches=inputs["max_patches"],
    )
    
    stats = get_pft_statistics(patch)
    
    assert stats["num_vegetated"] == 0
    assert stats["num_trees"] == 0
    assert stats["num_crops"] == 0


# ============================================================================
# Patch Validation Tests
# ============================================================================


def test_validate_patch_structure_valid(test_data):
    """Test validate_patch_structure with valid patch structures."""
    for test_case in test_data["test_cases"]:
        inputs = test_case["inputs"]
        
        patch = patch_type(
            column=inputs["column"],
            gridcell=inputs["gridcell"],
            itype=inputs["itype"],
            begp=inputs["begp"],
            endp=inputs["endp"],
            max_patches=inputs["max_patches"],
        )
        
        is_valid, error_msg = validate_patch_structure(patch)
        
        assert is_valid, \
            f"Patch structure should be valid for {test_case['name']}: {error_msg}"


def test_validate_patch_structure_invalid_indices():
    """Test validate_patch_structure with invalid indices."""
    # Create patch with endp < begp
    patch = patch_type(
        column=jnp.array([0], dtype=int),
        gridcell=jnp.array([0], dtype=int),
        itype=jnp.array([1], dtype=int),
        begp=5,
        endp=3,
        max_patches=10,
    )
    
    is_valid, error_msg = validate_patch_structure(patch)
    
    assert not is_valid, "Patch with endp < begp should be invalid"
    assert len(error_msg) > 0, "Error message should be provided"


def test_validate_patch_structure_out_of_bounds():
    """Test validate_patch_structure with out-of-bounds indices."""
    # Create patch with endp >= max_patches
    patch = patch_type(
        column=jnp.array([0, 1, 2], dtype=int),
        gridcell=jnp.array([0, 0, 0], dtype=int),
        itype=jnp.array([1, 2, 3], dtype=int),
        begp=0,
        endp=5,
        max_patches=3,
    )
    
    is_valid, error_msg = validate_patch_structure(patch)
    
    assert not is_valid, "Patch with endp >= max_patches should be invalid"


# ============================================================================
# Patch Method Tests
# ============================================================================


def test_patch_is_valid(test_data):
    """Test patch_type.is_valid() method."""
    test_case = test_data["test_cases"][0]
    inputs = test_case["inputs"]
    
    patch = patch_type(
        column=inputs["column"],
        gridcell=inputs["gridcell"],
        itype=inputs["itype"],
        begp=inputs["begp"],
        endp=inputs["endp"],
        max_patches=inputs["max_patches"],
    )
    
    assert patch.is_valid(), "Valid patch should return True from is_valid()"


def test_patch_get_active_patches(test_data):
    """Test patch_type.get_active_patches() method."""
    test_case = test_data["test_cases"][1]  # Multiple patches
    inputs = test_case["inputs"]
    expected = test_case["expected"]
    
    patch = patch_type(
        column=inputs["column"],
        gridcell=inputs["gridcell"],
        itype=inputs["itype"],
        begp=inputs["begp"],
        endp=inputs["endp"],
        max_patches=inputs["max_patches"],
    )
    
    active_patches = patch.get_active_patches()
    
    assert len(active_patches) == expected["num_patches"], \
        f"Should have {expected['num_patches']} active patches"
    assert jnp.all(active_patches >= inputs["begp"]), \
        "All active patches should be >= begp"
    assert jnp.all(active_patches <= inputs["endp"]), \
        "All active patches should be <= endp"


def test_patch_get_patch_info(test_data):
    """Test patch_type.get_patch_info() method."""
    test_case = test_data["test_cases"][0]  # Single patch
    inputs = test_case["inputs"]
    expected = test_case["expected"]
    
    patch = patch_type(
        column=inputs["column"],
        gridcell=inputs["gridcell"],
        itype=inputs["itype"],
        begp=inputs["begp"],
        endp=inputs["endp"],
        max_patches=inputs["max_patches"],
    )
    
    patch_info = patch.get_patch_info(0)
    
    assert "patch_index" in patch_info
    assert "pft_code" in patch_info
    assert "is_vegetated" in patch_info
    assert "is_tree" in patch_info
    assert patch_info["is_vegetated"] == expected["is_vegetated"]
    assert patch_info["is_tree"] == expected["is_tree"]


def test_patch_resize():
    """Test patch_type.resize() method."""
    patch = patch_type(
        column=jnp.array([0, 1], dtype=int),
        gridcell=jnp.array([0, 0], dtype=int),
        itype=jnp.array([1, 4], dtype=int),
        begp=0,
        endp=1,
        max_patches=2,
    )
    
    new_max = 5
    patch.resize(new_max)
    
    assert patch.max_patches == new_max
    assert patch.column.shape[0] == new_max
    assert patch.gridcell.shape[0] == new_max
    assert patch.itype.shape[0] == new_max


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_edge_case_zero_patches():
    """Test handling of zero patches (begp == endp, but no valid data)."""
    patch = patch_type(
        column=jnp.array([ISPVAL], dtype=int),
        gridcell=jnp.array([ISPVAL], dtype=int),
        itype=jnp.array([ISPVAL], dtype=int),
        begp=0,
        endp=0,
        max_patches=1,
    )
    
    stats = get_pft_statistics(patch)
    assert stats["num_patches"] >= 0


def test_edge_case_maximum_pft_boundary(test_data):
    """Test patches with maximum valid PFT codes (boundary test)."""
    test_case = test_data["test_cases"][3]  # Maximum PFT code
    inputs = test_case["inputs"]
    
    patch = patch_type(
        column=inputs["column"],
        gridcell=inputs["gridcell"],
        itype=inputs["itype"],
        begp=inputs["begp"],
        endp=inputs["endp"],
        max_patches=inputs["max_patches"],
    )
    
    # Should not raise errors
    stats = get_pft_statistics(patch)
    assert stats["num_vegetated"] > 0


def test_edge_case_sparse_distribution(test_data):
    """Test patches with non-zero begp (sparse distribution)."""
    test_case = test_data["test_cases"][7]  # Sparse patch distribution
    inputs = test_case["inputs"]
    expected = test_case["expected"]
    
    patch = patch_type(
        column=inputs["column"],
        gridcell=inputs["gridcell"],
        itype=inputs["itype"],
        begp=inputs["begp"],
        endp=inputs["endp"],
        max_patches=inputs["max_patches"],
    )
    
    assert patch.begp == expected["begp"]
    assert patch.endp == expected["endp"]
    
    active_patches = patch.get_active_patches()
    assert len(active_patches) == expected["num_patches"]


def test_edge_case_single_patch_minimal(test_data):
    """Test minimal configuration with single patch."""
    test_case = test_data["test_cases"][4]  # Single patch zero index
    inputs = test_case["inputs"]
    
    patch = patch_type(
        column=inputs["column"],
        gridcell=inputs["gridcell"],
        itype=inputs["itype"],
        begp=inputs["begp"],
        endp=inputs["endp"],
        max_patches=inputs["max_patches"],
    )
    
    assert patch.max_patches == 1
    assert patch.begp == 0
    assert patch.endp == 0


# ============================================================================
# Integration Tests
# ============================================================================


def test_integration_full_workflow(test_data):
    """Test complete workflow: create, validate, query, and analyze patches."""
    test_case = test_data["test_cases"][1]  # Mixed vegetation
    inputs = test_case["inputs"]
    expected = test_case["expected"]
    
    # Create patch instance
    patch = patch_type(
        column=inputs["column"],
        gridcell=inputs["gridcell"],
        itype=inputs["itype"],
        begp=inputs["begp"],
        endp=inputs["endp"],
        max_patches=inputs["max_patches"],
    )
    
    # Validate structure
    is_valid, _ = validate_patch_structure(patch)
    assert is_valid
    
    # Get statistics
    stats = get_pft_statistics(patch)
    assert stats["num_patches"] == expected["num_patches"]
    assert stats["num_vegetated"] == expected["num_vegetated"]
    
    # Query individual patches
    for i in range(patch.begp, patch.endp + 1):
        info = patch.get_patch_info(i)
        assert "pft_code" in info
        assert info["pft_code"] == patch.itype[i]


def test_integration_pft_classification_consistency():
    """Test that PFT classification functions are mutually consistent."""
    # Test various PFT codes
    test_codes = [0, 1, 4, 9, 10, 12, 13, 14, 15, 16]
    
    for code in test_codes:
        # Get all classifications
        veg = is_vegetated(code)
        tree = is_tree(code)
        shrub = is_shrub(code)
        grass = is_grass(code)
        crop = is_crop(code)
        c3 = is_c3_plant(code)
        c4 = is_c4_plant(code)
        
        # Check mutual exclusivity where appropriate
        if not veg:
            assert not tree and not shrub and not grass and not crop, \
                f"Non-vegetated PFT {code} should not be tree/shrub/grass/crop"
        
        # Only one vegetation category should be true
        veg_categories = sum([tree, shrub, grass, crop])
        if veg:
            assert veg_categories <= 1, \
                f"PFT {code} should belong to at most one vegetation category"
        
        # C3 and C4 should be mutually exclusive
        assert not (c3 and c4), \
            f"PFT {code} cannot be both C3 and C4"


def test_integration_enum_consistency():
    """Test that enum values are consistent with classification functions."""
    test_codes = [0, 1, 9, 12, 15]
    
    for code in test_codes:
        veg_cat = get_vegetation_category(code)
        photo_type = get_photosynthesis_type(code)
        pheno_type = get_phenology_type(code)
        
        # Check consistency with boolean functions
        if veg_cat == VegetationCategory.NOT_VEGETATED:
            assert not is_vegetated(code)
            assert photo_type == PhotosynthesisType.NONE
            assert pheno_type == PhenologyType.NOT_VEGETATED
        
        if veg_cat == VegetationCategory.TREE:
            assert is_tree(code)
        
        if photo_type == PhotosynthesisType.C3:
            assert is_c3_plant(code)
        
        if photo_type == PhotosynthesisType.C4:
            assert is_c4_plant(code)


# ============================================================================
# Global Instance Tests
# ============================================================================


def test_reset_global_patch():
    """Test reset_global_patch function."""
    # Reset should not raise errors
    reset_global_patch()
    
    # After reset, global patch should be accessible
    # (Implementation-dependent, but should not crash)


def test_print_patch_summary(test_data):
    """Test print_patch_summary function (output test)."""
    test_case = test_data["test_cases"][0]
    inputs = test_case["inputs"]
    
    patch = patch_type(
        column=inputs["column"],
        gridcell=inputs["gridcell"],
        itype=inputs["itype"],
        begp=inputs["begp"],
        endp=inputs["endp"],
        max_patches=inputs["max_patches"],
    )
    
    # Should not raise errors
    print_patch_summary(patch)


# ============================================================================
# Parametrized Comprehensive Tests
# ============================================================================


@pytest.mark.parametrize("test_case_idx", range(10))
def test_all_test_cases_validation(test_data, test_case_idx):
    """Parametrized test running validation on all test cases."""
    test_case = test_data["test_cases"][test_case_idx]
    inputs = test_case["inputs"]
    
    patch = patch_type(
        column=inputs["column"],
        gridcell=inputs["gridcell"],
        itype=inputs["itype"],
        begp=inputs["begp"],
        endp=inputs["endp"],
        max_patches=inputs["max_patches"],
    )
    
    is_valid, error_msg = validate_patch_structure(patch)
    assert is_valid, f"Test case {test_case['name']} failed validation: {error_msg}"


@pytest.mark.parametrize("test_case_idx", range(10))
def test_all_test_cases_statistics(test_data, test_case_idx):
    """Parametrized test checking statistics for all test cases."""
    test_case = test_data["test_cases"][test_case_idx]
    inputs = test_case["inputs"]
    expected = test_case.get("expected", {})
    
    patch = patch_type(
        column=inputs["column"],
        gridcell=inputs["gridcell"],
        itype=inputs["itype"],
        begp=inputs["begp"],
        endp=inputs["endp"],
        max_patches=inputs["max_patches"],
    )
    
    stats = get_pft_statistics(patch)
    
    # Check expected properties if provided
    if "num_patches" in expected:
        assert stats["num_patches"] == expected["num_patches"], \
            f"Patch count mismatch in {test_case['name']}"
    
    if "num_vegetated" in expected:
        assert stats["num_vegetated"] == expected["num_vegetated"], \
            f"Vegetated count mismatch in {test_case['name']}"
    
    if "num_trees" in expected:
        assert stats["num_trees"] == expected["num_trees"], \
            f"Tree count mismatch in {test_case['name']}"
    
    if "num_crops" in expected:
        assert stats["num_crops"] == expected["num_crops"], \
            f"Crop count mismatch in {test_case['name']}"


# ============================================================================
# Property-Based Tests
# ============================================================================


def test_property_pft_code_range():
    """Property test: All valid PFT codes should be in range [0, 78]."""
    for code in range(79):
        # Should not raise errors for valid codes
        name = get_pft_name(code)
        assert isinstance(name, str)
        
        veg_cat = get_vegetation_category(code)
        assert isinstance(veg_cat, VegetationCategory)


def test_property_patch_indices_consistency():
    """Property test: Active patches should always be within [begp, endp]."""
    test_configs = [
        (0, 0, 1),
        (0, 5, 10),
        (10, 15, 20),
        (0, 99, 100),
    ]
    
    for begp, endp, max_patches in test_configs:
        patch = create_patch_instance(begp, endp, max_patches)
        active = patch.get_active_patches()
        
        assert jnp.all(active >= begp), \
            f"Active patches should be >= begp ({begp})"
        assert jnp.all(active <= endp), \
            f"Active patches should be <= endp ({endp})"


def test_property_statistics_non_negative():
    """Property test: All statistics should be non-negative."""
    pft_codes = [1, 4, 12, 14, 15]
    patch = create_mixed_vegetation_patches(pft_codes)
    
    stats = get_pft_statistics(patch)
    
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            assert value >= 0, f"Statistic {key} should be non-negative"


def test_property_vegetation_categories_mutually_exclusive():
    """Property test: Vegetation categories should be mutually exclusive."""
    for code in range(79):
        tree = is_tree(code)
        shrub = is_shrub(code)
        grass = is_grass(code)
        crop = is_crop(code)
        
        # Count how many categories are True
        category_count = sum([tree, shrub, grass, crop])
        
        # Should be at most 1 (could be 0 for non-vegetated)
        assert category_count <= 1, \
            f"PFT {code} belongs to multiple vegetation categories"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])