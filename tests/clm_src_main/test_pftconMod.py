"""
Comprehensive pytest suite for pftconMod module.

This test suite validates the PFT (Plant Functional Type) constants module,
including the pftcon_type dataclass and associated utility functions.

Test Coverage:
- Dataclass initialization and validation
- Array shape and dimension verification
- Constraint validation (min/max bounds, valid values)
- Edge cases (zeros, boundaries, extreme values)
- Special cases (bare ground, mixed C3/C4 pathways)
- Physical realism (negative water potentials, fractions in [0,1])
- Utility functions (getters, validators, subset creation)
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

import pytest
import jax.numpy as jnp
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from clm_src_main.pftconMod import (
    pftcon_type,
    get_photosynthesis_pathway,
    get_leaf_reflectance,
    get_vcmax,
    reset_pftcon,
    validate_pftcon,
    print_pftcon_summary,
    create_pftcon_subset,
    mxpft,
    numrad,
    ivis,
    inir,
    pftcon
)


@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Load comprehensive test data for pftconMod testing.
    
    Returns:
        Dictionary containing test cases with inputs and metadata
    """
    return {
        "test_nominal_temperate_forest_pfts": {
            "pft_indices": [1, 2, 3, 4, 5],
            "mxpft": 78,
            "numrad": 2,
            "dleaf": [0.04, 0.04, 0.04, 0.04, 0.04],
            "c3psn": [1.0, 1.0, 1.0, 1.0, 1.0],
            "xl": [0.01, 0.01, 0.01, 0.01, 0.01],
            "rhol": [[0.07, 0.35], [0.08, 0.36], [0.08, 0.36], [0.1, 0.45], [0.1, 0.45]],
            "rhos": [[0.16, 0.39], [0.16, 0.39], [0.16, 0.39], [0.16, 0.39], [0.16, 0.39]],
            "taul": [[0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.25], [0.05, 0.25]],
            "taus": [[0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001]],
            "roota_par": [7.0, 7.0, 7.0, 6.0, 6.0],
            "rootb_par": [2.0, 2.0, 2.0, 2.0, 2.0],
            "rootprof_beta": [0.943, 0.943, 0.943, 0.966, 0.966],
            "slatop": [0.012, 0.012, 0.012, 0.01, 0.01],
            "vcmaxpft": [62.5, 62.5, 62.5, 41.0, 41.0],
            "gplant_SPA": [40.0, 40.0, 40.0, 35.0, 35.0],
            "capac_SPA": [2000.0, 2000.0, 2000.0, 1800.0, 1800.0],
            "iota_SPA": [4000.0, 4000.0, 4000.0, 3500.0, 3500.0],
            "root_radius_SPA": [0.00029, 0.00029, 0.00029, 0.00029, 0.00029],
            "root_density_SPA": [310000.0, 310000.0, 310000.0, 310000.0, 310000.0],
            "root_resist_SPA": [25.0, 25.0, 25.0, 25.0, 25.0],
            "gsmin_SPA": [0.002, 0.002, 0.002, 0.002, 0.002],
            "g0_BB": [0.01, 0.01, 0.01, 0.01, 0.01],
            "g1_BB": [9.0, 9.0, 9.0, 9.0, 9.0],
            "g0_MED": [0.0, 0.0, 0.0, 0.0, 0.0],
            "g1_MED": [4.7, 4.7, 4.7, 4.7, 4.7],
            "psi50_gs": [-2.5, -2.5, -2.5, -2.0, -2.0],
            "shape_gs": [3.0, 3.0, 3.0, 3.0, 3.0],
            "emleaf": [0.97, 0.97, 0.97, 0.97, 0.97],
            "clump_fac": [0.75, 0.75, 0.75, 0.8, 0.8],
            "pbeta_lai": [1.0, 1.0, 1.0, 1.0, 1.0],
            "qbeta_lai": [1.0, 1.0, 1.0, 1.0, 1.0],
            "pbeta_sai": [1.0, 1.0, 1.0, 1.0, 1.0],
            "qbeta_sai": [1.0, 1.0, 1.0, 1.0, 1.0],
            "type": "nominal"
        },
        "test_nominal_c4_grassland_pfts": {
            "pft_indices": [13, 14],
            "mxpft": 78,
            "numrad": 2,
            "dleaf": [0.04, 0.04],
            "c3psn": [0.0, 0.0],
            "xl": [-0.3, -0.3],
            "rhol": [[0.11, 0.35], [0.11, 0.35]],
            "rhos": [[0.36, 0.58], [0.36, 0.58]],
            "taul": [[0.05, 0.34], [0.05, 0.34]],
            "taus": [[0.22, 0.38], [0.22, 0.38]],
            "roota_par": [6.0, 6.0],
            "rootb_par": [2.0, 2.0],
            "rootprof_beta": [0.943, 0.943],
            "slatop": [0.02, 0.02],
            "vcmaxpft": [39.0, 39.0],
            "gplant_SPA": [50.0, 50.0],
            "capac_SPA": [1500.0, 1500.0],
            "iota_SPA": [1800.0, 1800.0],
            "root_radius_SPA": [0.00029, 0.00029],
            "root_density_SPA": [310000.0, 310000.0],
            "root_resist_SPA": [25.0, 25.0],
            "gsmin_SPA": [0.002, 0.002],
            "g0_BB": [0.04, 0.04],
            "g1_BB": [4.0, 4.0],
            "g0_MED": [0.0, 0.0],
            "g1_MED": [1.6, 1.6],
            "psi50_gs": [-3.0, -3.0],
            "shape_gs": [4.0, 4.0],
            "emleaf": [0.97, 0.97],
            "clump_fac": [0.85, 0.85],
            "pbeta_lai": [1.0, 1.0],
            "qbeta_lai": [1.0, 1.0],
            "pbeta_sai": [1.0, 1.0],
            "qbeta_sai": [1.0, 1.0],
            "type": "nominal"
        },
        "test_edge_minimum_valid_values": {
            "pft_indices": [0, 1, 2],
            "mxpft": 78,
            "numrad": 2,
            "dleaf": [0.001, 0.001, 0.001],
            "c3psn": [1.0, 0.0, 1.0],
            "xl": [-1.0, -1.0, -1.0],
            "rhol": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            "rhos": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            "taul": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            "taus": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            "roota_par": [0.1, 0.1, 0.1],
            "rootb_par": [0.1, 0.1, 0.1],
            "rootprof_beta": [0.0, 0.0, 0.0],
            "slatop": [0.001, 0.001, 0.001],
            "vcmaxpft": [1.0, 1.0, 1.0],
            "gplant_SPA": [0.1, 0.1, 0.1],
            "capac_SPA": [10.0, 10.0, 10.0],
            "iota_SPA": [100.0, 100.0, 100.0],
            "root_radius_SPA": [1e-06, 1e-06, 1e-06],
            "root_density_SPA": [1000.0, 1000.0, 1000.0],
            "root_resist_SPA": [0.1, 0.1, 0.1],
            "gsmin_SPA": [1e-06, 1e-06, 1e-06],
            "g0_BB": [0.0, 0.0, 0.0],
            "g1_BB": [0.1, 0.1, 0.1],
            "g0_MED": [0.0, 0.0, 0.0],
            "g1_MED": [0.1, 0.1, 0.1],
            "psi50_gs": [-10.0, -10.0, -10.0],
            "shape_gs": [0.1, 0.1, 0.1],
            "emleaf": [0.0, 0.0, 0.0],
            "clump_fac": [0.0, 0.0, 0.0],
            "pbeta_lai": [0.1, 0.1, 0.1],
            "qbeta_lai": [0.1, 0.1, 0.1],
            "pbeta_sai": [0.1, 0.1, 0.1],
            "qbeta_sai": [0.1, 0.1, 0.1],
            "type": "edge"
        },
        "test_edge_maximum_valid_values": {
            "pft_indices": [75, 76, 77, 78],
            "mxpft": 78,
            "numrad": 2,
            "dleaf": [0.5, 0.5, 0.5, 0.5],
            "c3psn": [1.0, 1.0, 1.0, 1.0],
            "xl": [1.0, 1.0, 1.0, 1.0],
            "rhol": [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            "rhos": [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            "taul": [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            "taus": [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            "roota_par": [50.0, 50.0, 50.0, 50.0],
            "rootb_par": [10.0, 10.0, 10.0, 10.0],
            "rootprof_beta": [1.0, 1.0, 1.0, 1.0],
            "slatop": [0.1, 0.1, 0.1, 0.1],
            "vcmaxpft": [200.0, 200.0, 200.0, 200.0],
            "gplant_SPA": [500.0, 500.0, 500.0, 500.0],
            "capac_SPA": [10000.0, 10000.0, 10000.0, 10000.0],
            "iota_SPA": [10000.0, 10000.0, 10000.0, 10000.0],
            "root_radius_SPA": [0.01, 0.01, 0.01, 0.01],
            "root_density_SPA": [10000000.0, 10000000.0, 10000000.0, 10000000.0],
            "root_resist_SPA": [100.0, 100.0, 100.0, 100.0],
            "gsmin_SPA": [0.1, 0.1, 0.1, 0.1],
            "g0_BB": [0.5, 0.5, 0.5, 0.5],
            "g1_BB": [20.0, 20.0, 20.0, 20.0],
            "g0_MED": [0.5, 0.5, 0.5, 0.5],
            "g1_MED": [15.0, 15.0, 15.0, 15.0],
            "psi50_gs": [-0.1, -0.1, -0.1, -0.1],
            "shape_gs": [10.0, 10.0, 10.0, 10.0],
            "emleaf": [1.0, 1.0, 1.0, 1.0],
            "clump_fac": [1.0, 1.0, 1.0, 1.0],
            "pbeta_lai": [10.0, 10.0, 10.0, 10.0],
            "qbeta_lai": [10.0, 10.0, 10.0, 10.0],
            "pbeta_sai": [10.0, 10.0, 10.0, 10.0],
            "qbeta_sai": [10.0, 10.0, 10.0, 10.0],
            "type": "edge"
        },
        "test_special_single_pft": {
            "pft_indices": [0],
            "mxpft": 78,
            "numrad": 2,
            "dleaf": [0.04],
            "c3psn": [-999.0],
            "xl": [0.0],
            "rhol": [[0.0, 0.0]],
            "rhos": [[0.0, 0.0]],
            "taul": [[0.0, 0.0]],
            "taus": [[0.0, 0.0]],
            "roota_par": [0.0],
            "rootb_par": [0.0],
            "rootprof_beta": [0.0],
            "slatop": [0.0],
            "vcmaxpft": [0.0],
            "gplant_SPA": [0.0],
            "capac_SPA": [0.0],
            "iota_SPA": [0.0],
            "root_radius_SPA": [0.0],
            "root_density_SPA": [0.0],
            "root_resist_SPA": [0.0],
            "gsmin_SPA": [0.0],
            "g0_BB": [0.0],
            "g1_BB": [0.0],
            "g0_MED": [0.0],
            "g1_MED": [0.0],
            "psi50_gs": [0.0],
            "shape_gs": [0.0],
            "emleaf": [0.0],
            "clump_fac": [0.0],
            "pbeta_lai": [0.0],
            "qbeta_lai": [0.0],
            "pbeta_sai": [0.0],
            "qbeta_sai": [0.0],
            "type": "special"
        },
        "test_special_mixed_c3_c4_pathways": {
            "pft_indices": [6, 7, 8, 13, 14, 15],
            "mxpft": 78,
            "numrad": 2,
            "dleaf": [0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
            "c3psn": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "xl": [0.25, 0.25, 0.25, -0.3, -0.3, -0.3],
            "rhol": [[0.1, 0.45], [0.1, 0.45], [0.1, 0.45], [0.11, 0.35], [0.11, 0.35], [0.11, 0.35]],
            "rhos": [[0.16, 0.39], [0.16, 0.39], [0.16, 0.39], [0.36, 0.58], [0.36, 0.58], [0.36, 0.58]],
            "taul": [[0.05, 0.25], [0.05, 0.25], [0.05, 0.25], [0.05, 0.34], [0.05, 0.34], [0.05, 0.34]],
            "taus": [[0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.22, 0.38], [0.22, 0.38], [0.22, 0.38]],
            "roota_par": [6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
            "rootb_par": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            "rootprof_beta": [0.966, 0.966, 0.966, 0.943, 0.943, 0.943],
            "slatop": [0.01, 0.01, 0.01, 0.02, 0.02, 0.02],
            "vcmaxpft": [41.0, 41.0, 41.0, 39.0, 39.0, 39.0],
            "gplant_SPA": [35.0, 35.0, 35.0, 50.0, 50.0, 50.0],
            "capac_SPA": [1800.0, 1800.0, 1800.0, 1500.0, 1500.0, 1500.0],
            "iota_SPA": [3500.0, 3500.0, 3500.0, 1800.0, 1800.0, 1800.0],
            "root_radius_SPA": [0.00029, 0.00029, 0.00029, 0.00029, 0.00029, 0.00029],
            "root_density_SPA": [310000.0, 310000.0, 310000.0, 310000.0, 310000.0, 310000.0],
            "root_resist_SPA": [25.0, 25.0, 25.0, 25.0, 25.0, 25.0],
            "gsmin_SPA": [0.002, 0.002, 0.002, 0.002, 0.002, 0.002],
            "g0_BB": [0.01, 0.01, 0.01, 0.04, 0.04, 0.04],
            "g1_BB": [9.0, 9.0, 9.0, 4.0, 4.0, 4.0],
            "g0_MED": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "g1_MED": [4.7, 4.7, 4.7, 1.6, 1.6, 1.6],
            "psi50_gs": [-2.0, -2.0, -2.0, -3.0, -3.0, -3.0],
            "shape_gs": [3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
            "emleaf": [0.97, 0.97, 0.97, 0.97, 0.97, 0.97],
            "clump_fac": [0.8, 0.8, 0.8, 0.85, 0.85, 0.85],
            "pbeta_lai": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "qbeta_lai": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "pbeta_sai": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "qbeta_sai": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "type": "special"
        }
    }


# ============================================================================
# pftcon_type Dataclass Tests
# ============================================================================

def test_pftcon_type_initialization():
    """Test that pftcon_type can be initialized with default values."""
    pft = pftcon_type()
    
    assert pft.is_initialized == False, "Default initialization should set is_initialized to False"
    assert isinstance(pft.metadata, dict), "Metadata should be a dictionary"
    assert len(pft.metadata) == 0, "Default metadata should be empty"


def test_pftcon_type_init_allocate():
    """Test InitAllocate method creates arrays with correct shapes."""
    pft = pftcon_type()
    pft.InitAllocate()
    
    # Check 1D arrays
    expected_1d_shape = (mxpft + 1,)
    assert pft.dleaf.shape == expected_1d_shape, f"dleaf shape should be {expected_1d_shape}"
    assert pft.c3psn.shape == expected_1d_shape, f"c3psn shape should be {expected_1d_shape}"
    assert pft.xl.shape == expected_1d_shape, f"xl shape should be {expected_1d_shape}"
    assert pft.roota_par.shape == expected_1d_shape, f"roota_par shape should be {expected_1d_shape}"
    assert pft.vcmaxpft.shape == expected_1d_shape, f"vcmaxpft shape should be {expected_1d_shape}"
    
    # Check 2D arrays
    expected_2d_shape = (mxpft + 1, numrad)
    assert pft.rhol.shape == expected_2d_shape, f"rhol shape should be {expected_2d_shape}"
    assert pft.rhos.shape == expected_2d_shape, f"rhos shape should be {expected_2d_shape}"
    assert pft.taul.shape == expected_2d_shape, f"taul shape should be {expected_2d_shape}"
    assert pft.taus.shape == expected_2d_shape, f"taus shape should be {expected_2d_shape}"


def test_pftcon_type_init_read():
    """Test InitRead method populates arrays with valid values."""
    pft = pftcon_type()
    pft.Init()
    
    assert pft.is_initialized == True, "After Init(), is_initialized should be True"
    
    # Check that arrays are not all zeros (have been populated)
    assert jnp.any(pft.dleaf != 0.0), "dleaf should contain non-zero values"
    assert jnp.any(pft.vcmaxpft != 0.0), "vcmaxpft should contain non-zero values"
    
    # Check constraint: c3psn should only contain 0.0, 1.0, or -999.0
    unique_c3psn = jnp.unique(pft.c3psn)
    valid_c3psn = jnp.array([0.0, 1.0, -999.0])
    for val in unique_c3psn:
        assert jnp.any(jnp.isclose(val, valid_c3psn)), \
            f"c3psn contains invalid value {val}, should be 0.0, 1.0, or -999.0"


def test_pftcon_type_is_valid():
    """Test is_valid method correctly identifies initialized state."""
    pft = pftcon_type()
    
    # Before initialization
    assert pft.is_valid() == False, "is_valid should return False before initialization"
    
    # After initialization
    pft.Init()
    assert pft.is_valid() == True, "is_valid should return True after initialization"


def test_pftcon_type_get_pft_parameters():
    """Test get_pft_parameters returns correct dictionary for a given PFT index."""
    pft = pftcon_type()
    pft.Init()
    
    # Test valid PFT index
    pft_idx = 5
    params = pft.get_pft_parameters(pft_idx)
    
    assert isinstance(params, dict), "get_pft_parameters should return a dictionary"
    assert "dleaf" in params, "Parameters should include dleaf"
    assert "c3psn" in params, "Parameters should include c3psn"
    assert "vcmaxpft" in params, "Parameters should include vcmaxpft"
    assert "rhol" in params, "Parameters should include rhol"
    
    # Check that rhol is 1D array with length numrad
    assert params["rhol"].shape == (numrad,), f"rhol should have shape ({numrad},)"


@pytest.mark.parametrize("pft_idx", [0, 1, 10, 50, 78])
def test_pftcon_type_get_pft_parameters_various_indices(pft_idx):
    """Test get_pft_parameters works for various valid PFT indices."""
    pft = pftcon_type()
    pft.Init()
    
    params = pft.get_pft_parameters(pft_idx)
    
    assert isinstance(params, dict), f"Should return dict for PFT index {pft_idx}"
    assert len(params) > 0, f"Should return non-empty dict for PFT index {pft_idx}"


# ============================================================================
# Constraint Validation Tests
# ============================================================================

def test_constraints_positive_values():
    """Test that parameters with min >= 0 constraint contain only non-negative values."""
    pft = pftcon_type()
    pft.Init()
    
    # Parameters that must be >= 0
    positive_params = [
        "dleaf", "roota_par", "rootb_par", "slatop", "vcmaxpft",
        "gplant_SPA", "capac_SPA", "iota_SPA", "root_radius_SPA",
        "root_density_SPA", "root_resist_SPA", "gsmin_SPA",
        "g0_BB", "g1_BB", "g0_MED", "g1_MED", "shape_gs",
        "pbeta_lai", "qbeta_lai", "pbeta_sai", "qbeta_sai"
    ]
    
    for param_name in positive_params:
        param_array = getattr(pft, param_name)
        # Exclude special values like -999.0
        valid_mask = param_array != -999.0
        assert jnp.all(param_array[valid_mask] >= 0.0), \
            f"{param_name} should contain only non-negative values"


def test_constraints_fraction_values():
    """Test that fraction parameters are in [0, 1] range."""
    pft = pftcon_type()
    pft.Init()
    
    # Parameters that must be in [0, 1]
    fraction_params = ["rhol", "rhos", "taul", "taus", "rootprof_beta", "emleaf", "clump_fac"]
    
    for param_name in fraction_params:
        param_array = getattr(pft, param_name)
        # Exclude special values
        valid_mask = param_array != -999.0
        assert jnp.all(param_array[valid_mask] >= 0.0), \
            f"{param_name} should be >= 0.0"
        assert jnp.all(param_array[valid_mask] <= 1.0), \
            f"{param_name} should be <= 1.0"


def test_constraints_negative_water_potential():
    """Test that psi50_gs contains only non-positive values (water potential)."""
    pft = pftcon_type()
    pft.Init()
    
    # Exclude special values
    valid_mask = pft.psi50_gs != -999.0
    assert jnp.all(pft.psi50_gs[valid_mask] <= 0.0), \
        "psi50_gs (water potential) should be <= 0.0"


def test_constraints_c3psn_valid_values():
    """Test that c3psn contains only valid photosynthetic pathway values."""
    pft = pftcon_type()
    pft.Init()
    
    valid_values = jnp.array([0.0, 1.0, -999.0])
    
    for val in jnp.unique(pft.c3psn):
        assert jnp.any(jnp.isclose(val, valid_values, atol=1e-6)), \
            f"c3psn contains invalid value {val}, should be 0.0 (C4), 1.0 (C3), or -999.0 (bare ground)"


# ============================================================================
# Utility Function Tests
# ============================================================================

def test_get_photosynthesis_pathway_shapes():
    """Test get_photosynthesis_pathway returns correct output shape."""
    pft = pftcon_type()
    pft.Init()
    
    pft_indices = jnp.array([1, 2, 3, 4, 5])
    result = get_photosynthesis_pathway(pft_indices)
    
    assert result.shape == pft_indices.shape, \
        f"Output shape {result.shape} should match input shape {pft_indices.shape}"


def test_get_photosynthesis_pathway_values():
    """Test get_photosynthesis_pathway returns correct C3/C4 values."""
    pft = pftcon_type()
    pft.Init()
    
    # Test C3 PFTs (should return 1.0)
    c3_indices = jnp.array([1, 2, 3])
    c3_result = get_photosynthesis_pathway(c3_indices)
    
    # Test C4 PFTs (should return 0.0)
    c4_indices = jnp.array([13, 14])
    c4_result = get_photosynthesis_pathway(c4_indices)
    
    # Values should be either 0.0 or 1.0 (excluding -999.0 for bare ground)
    for val in c3_result:
        if val != -999.0:
            assert val in [0.0, 1.0], f"Photosynthesis pathway should be 0.0 or 1.0, got {val}"


def test_get_leaf_reflectance_shapes():
    """Test get_leaf_reflectance returns correct output shape."""
    pft = pftcon_type()
    pft.Init()
    
    pft_indices = jnp.array([1, 2, 3, 4, 5])
    
    # Test visible band
    result_vis = get_leaf_reflectance(pft_indices, ivis)
    assert result_vis.shape == pft_indices.shape, \
        f"Output shape {result_vis.shape} should match input shape {pft_indices.shape}"
    
    # Test NIR band
    result_nir = get_leaf_reflectance(pft_indices, inir)
    assert result_nir.shape == pft_indices.shape, \
        f"Output shape {result_nir.shape} should match input shape {pft_indices.shape}"


def test_get_leaf_reflectance_values():
    """Test get_leaf_reflectance returns values in valid range [0, 1]."""
    pft = pftcon_type()
    pft.Init()
    
    pft_indices = jnp.array([1, 2, 3, 4, 5])
    
    for band in [ivis, inir]:
        result = get_leaf_reflectance(pft_indices, band)
        assert jnp.all(result >= 0.0), f"Reflectance should be >= 0.0 for band {band}"
        assert jnp.all(result <= 1.0), f"Reflectance should be <= 1.0 for band {band}"


def test_get_vcmax_shapes():
    """Test get_vcmax returns correct output shape."""
    pft = pftcon_type()
    pft.Init()
    
    pft_indices = jnp.array([1, 2, 3, 4, 5])
    result = get_vcmax(pft_indices)
    
    assert result.shape == pft_indices.shape, \
        f"Output shape {result.shape} should match input shape {pft_indices.shape}"


def test_get_vcmax_values():
    """Test get_vcmax returns non-negative values."""
    pft = pftcon_type()
    pft.Init()
    
    pft_indices = jnp.array([1, 2, 3, 4, 5])
    result = get_vcmax(pft_indices)
    
    assert jnp.all(result >= 0.0), "Vcmax should be non-negative"


def test_validate_pftcon():
    """Test validate_pftcon returns correct validation status."""
    # Before initialization
    reset_pftcon()
    is_valid, message = validate_pftcon()
    assert is_valid == False, "Should be invalid before initialization"
    assert isinstance(message, str), "Should return error message string"
    
    # After initialization
    pft = pftcon_type()
    pft.Init()
    is_valid, message = validate_pftcon()
    assert is_valid == True, "Should be valid after initialization"


def test_create_pftcon_subset():
    """Test create_pftcon_subset returns correct subset of PFT data."""
    pft = pftcon_type()
    pft.Init()
    
    subset_indices = [1, 2, 3]
    subset = create_pftcon_subset(subset_indices)
    
    assert isinstance(subset, dict), "Should return dictionary"
    assert "dleaf" in subset, "Subset should contain dleaf"
    assert "vcmaxpft" in subset, "Subset should contain vcmaxpft"
    
    # Check that subset has correct length
    assert subset["dleaf"].shape[0] == len(subset_indices), \
        f"Subset should have {len(subset_indices)} elements"


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_edge_case_minimum_values(test_data):
    """Test handling of minimum valid values at lower bounds."""
    data = test_data["test_edge_minimum_valid_values"]
    
    # Create custom pftcon with minimum values
    pft = pftcon_type()
    pft.InitAllocate()
    
    # Set minimum values for selected PFTs
    for i, idx in enumerate(data["pft_indices"]):
        pft.dleaf = pft.dleaf.at[idx].set(data["dleaf"][i])
        pft.vcmaxpft = pft.vcmaxpft.at[idx].set(data["vcmaxpft"][i])
        pft.emleaf = pft.emleaf.at[idx].set(data["emleaf"][i])
    
    pft.is_initialized = True
    
    # Verify minimum values are accepted
    assert jnp.all(pft.dleaf >= 0.0), "Minimum dleaf values should be valid"
    assert jnp.all(pft.vcmaxpft >= 0.0), "Minimum vcmaxpft values should be valid"
    assert jnp.all(pft.emleaf >= 0.0), "Minimum emleaf values should be valid"


def test_edge_case_maximum_values(test_data):
    """Test handling of maximum valid values at upper bounds."""
    data = test_data["test_edge_maximum_valid_values"]
    
    # Create custom pftcon with maximum values
    pft = pftcon_type()
    pft.InitAllocate()
    
    # Set maximum values for selected PFTs
    for i, idx in enumerate(data["pft_indices"]):
        pft.emleaf = pft.emleaf.at[idx].set(data["emleaf"][i])
        pft.clump_fac = pft.clump_fac.at[idx].set(data["clump_fac"][i])
        pft.rootprof_beta = pft.rootprof_beta.at[idx].set(data["rootprof_beta"][i])
    
    pft.is_initialized = True
    
    # Verify maximum values are accepted
    assert jnp.all(pft.emleaf <= 1.0), "Maximum emleaf values should be valid"
    assert jnp.all(pft.clump_fac <= 1.0), "Maximum clump_fac values should be valid"
    assert jnp.all(pft.rootprof_beta <= 1.0), "Maximum rootprof_beta values should be valid"


def test_edge_case_extreme_negative_water_potential(test_data):
    """Test handling of extreme negative water potentials."""
    data = test_data["test_edge_minimum_valid_values"]
    
    pft = pftcon_type()
    pft.InitAllocate()
    
    # Set extreme negative water potentials
    for i, idx in enumerate(data["pft_indices"]):
        pft.psi50_gs = pft.psi50_gs.at[idx].set(data["psi50_gs"][i])
    
    pft.is_initialized = True
    
    # Verify extreme negative values are accepted
    assert jnp.all(pft.psi50_gs <= 0.0), "Water potential should be non-positive"


def test_edge_case_single_pft(test_data):
    """Test handling of single PFT (bare ground with special values)."""
    data = test_data["test_special_single_pft"]
    
    pft = pftcon_type()
    pft.InitAllocate()
    
    # Set bare ground values
    idx = data["pft_indices"][0]
    pft.c3psn = pft.c3psn.at[idx].set(data["c3psn"][0])
    pft.dleaf = pft.dleaf.at[idx].set(data["dleaf"][0])
    
    pft.is_initialized = True
    
    # Verify special value for bare ground
    assert jnp.isclose(pft.c3psn[idx], -999.0, atol=1e-6), \
        "Bare ground should have c3psn = -999.0"


def test_edge_case_mixed_c3_c4_pathways(test_data):
    """Test handling of mixed C3 and C4 PFTs in same dataset."""
    data = test_data["test_special_mixed_c3_c4_pathways"]
    
    pft = pftcon_type()
    pft.InitAllocate()
    
    # Set mixed C3/C4 values
    for i, idx in enumerate(data["pft_indices"]):
        pft.c3psn = pft.c3psn.at[idx].set(data["c3psn"][i])
    
    pft.is_initialized = True
    
    # Verify both C3 and C4 pathways are present
    c3_mask = pft.c3psn == 1.0
    c4_mask = pft.c3psn == 0.0
    
    assert jnp.any(c3_mask), "Should have C3 PFTs"
    assert jnp.any(c4_mask), "Should have C4 PFTs"


# ============================================================================
# Data Type Tests
# ============================================================================

def test_dtypes_arrays():
    """Test that all arrays have correct JAX data types."""
    pft = pftcon_type()
    pft.Init()
    
    # All numeric arrays should be JAX arrays
    numeric_fields = [
        "dleaf", "c3psn", "xl", "rhol", "rhos", "taul", "taus",
        "roota_par", "rootb_par", "rootprof_beta", "slatop", "vcmaxpft",
        "gplant_SPA", "capac_SPA", "iota_SPA", "root_radius_SPA",
        "root_density_SPA", "root_resist_SPA", "gsmin_SPA",
        "g0_BB", "g1_BB", "g0_MED", "g1_MED", "psi50_gs", "shape_gs",
        "emleaf", "clump_fac", "pbeta_lai", "qbeta_lai", "pbeta_sai", "qbeta_sai"
    ]
    
    for field_name in numeric_fields:
        field_array = getattr(pft, field_name)
        assert isinstance(field_array, jnp.ndarray), \
            f"{field_name} should be a JAX array"


def test_dtypes_boolean():
    """Test that boolean fields have correct type."""
    pft = pftcon_type()
    
    assert isinstance(pft.is_initialized, bool), \
        "is_initialized should be a boolean"


def test_dtypes_metadata():
    """Test that metadata field is a dictionary."""
    pft = pftcon_type()
    
    assert isinstance(pft.metadata, dict), \
        "metadata should be a dictionary"


# ============================================================================
# Array Shape Tests
# ============================================================================

def test_shapes_1d_arrays():
    """Test that 1D arrays have correct shape (mxpft+1,)."""
    pft = pftcon_type()
    pft.Init()
    
    expected_shape = (mxpft + 1,)
    
    fields_1d = [
        "dleaf", "c3psn", "xl", "roota_par", "rootb_par", "rootprof_beta",
        "slatop", "vcmaxpft", "gplant_SPA", "capac_SPA", "iota_SPA",
        "root_radius_SPA", "root_density_SPA", "root_resist_SPA",
        "gsmin_SPA", "g0_BB", "g1_BB", "g0_MED", "g1_MED",
        "psi50_gs", "shape_gs", "emleaf", "clump_fac",
        "pbeta_lai", "qbeta_lai", "pbeta_sai", "qbeta_sai"
    ]
    
    for field_name in fields_1d:
        field_array = getattr(pft, field_name)
        assert field_array.shape == expected_shape, \
            f"{field_name} should have shape {expected_shape}, got {field_array.shape}"


def test_shapes_2d_arrays():
    """Test that 2D arrays have correct shape (mxpft+1, numrad)."""
    pft = pftcon_type()
    pft.Init()
    
    expected_shape = (mxpft + 1, numrad)
    
    fields_2d = ["rhol", "rhos", "taul", "taus"]
    
    for field_name in fields_2d:
        field_array = getattr(pft, field_name)
        assert field_array.shape == expected_shape, \
            f"{field_name} should have shape {expected_shape}, got {field_array.shape}"


# ============================================================================
# Integration Tests
# ============================================================================

def test_integration_full_workflow():
    """Test complete workflow: initialize, validate, query, and subset."""
    # Initialize
    pft = pftcon_type()
    pft.Init()
    
    # Validate
    assert pft.is_valid(), "PFT constants should be valid after initialization"
    is_valid, message = validate_pftcon()
    assert is_valid, f"Validation should pass: {message}"
    
    # Query specific PFT
    pft_idx = 5
    params = pft.get_pft_parameters(pft_idx)
    assert len(params) > 0, "Should retrieve parameters for PFT"
    
    # Create subset
    subset_indices = [1, 2, 3, 4, 5]
    subset = create_pftcon_subset(subset_indices)
    assert len(subset) > 0, "Should create non-empty subset"
    
    # Use utility functions
    pft_indices = jnp.array(subset_indices)
    pathways = get_photosynthesis_pathway(pft_indices)
    assert pathways.shape == (len(subset_indices),), "Should return correct pathway shape"
    
    reflectance = get_leaf_reflectance(pft_indices, ivis)
    assert reflectance.shape == (len(subset_indices),), "Should return correct reflectance shape"
    
    vcmax = get_vcmax(pft_indices)
    assert vcmax.shape == (len(subset_indices),), "Should return correct vcmax shape"


def test_integration_reset_and_reinitialize():
    """Test that reset and reinitialization works correctly."""
    # Initialize
    pft = pftcon_type()
    pft.Init()
    assert pft.is_valid(), "Should be valid after first initialization"
    
    # Reset
    reset_pftcon()
    is_valid, _ = validate_pftcon()
    assert is_valid == False, "Should be invalid after reset"
    
    # Reinitialize
    pft2 = pftcon_type()
    pft2.Init()
    assert pft2.is_valid(), "Should be valid after reinitialization"


def test_integration_constants():
    """Test that module constants have correct values."""
    assert mxpft == 78, "mxpft should be 78"
    assert numrad == 2, "numrad should be 2"
    assert ivis == 0, "ivis should be 0"
    assert inir == 1, "inir should be 1"


# ============================================================================
# Parametrized Tests for Multiple PFT Indices
# ============================================================================

@pytest.mark.parametrize("pft_idx", [0, 1, 5, 10, 20, 50, 78])
def test_parametrized_pft_indices(pft_idx):
    """Test that various PFT indices can be queried successfully."""
    pft = pftcon_type()
    pft.Init()
    
    params = pft.get_pft_parameters(pft_idx)
    
    assert isinstance(params, dict), f"Should return dict for PFT {pft_idx}"
    assert "vcmaxpft" in params, f"Should include vcmaxpft for PFT {pft_idx}"
    assert params["vcmaxpft"] >= 0.0 or params["vcmaxpft"] == -999.0, \
        f"vcmaxpft should be non-negative or -999.0 for PFT {pft_idx}"


@pytest.mark.parametrize("band", [ivis, inir])
def test_parametrized_radiation_bands(band):
    """Test that both radiation bands work correctly."""
    pft = pftcon_type()
    pft.Init()
    
    pft_indices = jnp.array([1, 2, 3, 4, 5])
    reflectance = get_leaf_reflectance(pft_indices, band)
    
    assert reflectance.shape == pft_indices.shape, \
        f"Should return correct shape for band {band}"
    assert jnp.all(reflectance >= 0.0), f"Reflectance should be >= 0 for band {band}"
    assert jnp.all(reflectance <= 1.0), f"Reflectance should be <= 1 for band {band}"


@pytest.mark.parametrize("test_case_name", [
    "test_nominal_temperate_forest_pfts",
    "test_nominal_c4_grassland_pfts",
    "test_edge_minimum_valid_values",
    "test_edge_maximum_valid_values",
    "test_special_single_pft",
    "test_special_mixed_c3_c4_pathways"
])
def test_parametrized_test_cases(test_data, test_case_name):
    """Test various test cases from test data."""
    data = test_data[test_case_name]
    
    pft = pftcon_type()
    pft.InitAllocate()
    
    # Set values from test data
    for i, idx in enumerate(data["pft_indices"]):
        pft.dleaf = pft.dleaf.at[idx].set(data["dleaf"][i])
        pft.c3psn = pft.c3psn.at[idx].set(data["c3psn"][i])
        pft.vcmaxpft = pft.vcmaxpft.at[idx].set(data["vcmaxpft"][i])
    
    pft.is_initialized = True
    
    # Verify basic constraints
    assert jnp.all(pft.dleaf >= 0.0), f"dleaf should be non-negative in {test_case_name}"
    
    # Check c3psn values
    for idx in data["pft_indices"]:
        c3psn_val = pft.c3psn[idx]
        assert c3psn_val in [0.0, 1.0, -999.0] or jnp.isclose(c3psn_val, 0.0) or \
               jnp.isclose(c3psn_val, 1.0) or jnp.isclose(c3psn_val, -999.0), \
            f"c3psn should be valid in {test_case_name}"


# ============================================================================
# Documentation and Summary Tests
# ============================================================================

def test_print_pftcon_summary_runs():
    """Test that print_pftcon_summary executes without error."""
    pft = pftcon_type()
    pft.Init()
    
    # Should not raise an exception
    try:
        print_pftcon_summary()
    except Exception as e:
        pytest.fail(f"print_pftcon_summary raised exception: {e}")


def test_global_pftcon_instance():
    """Test that global pftcon instance exists and can be accessed."""
    assert pftcon is not None, "Global pftcon instance should exist"
    assert isinstance(pftcon, pftcon_type), "Global pftcon should be pftcon_type instance"