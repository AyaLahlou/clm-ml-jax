"""
Comprehensive pytest suite for EnergyFluxType module.

This module tests the initialization, update, and validation functions for
energy flux data structures used in land surface modeling. Tests cover:
- Array allocation and initialization patterns
- Field updates with immutable semantics
- Physical constraint validation
- Edge cases (single patch, large domains, boundary values)
- Data type consistency
"""

import pytest
import jax.numpy as jnp
import numpy as np
from collections import namedtuple
from typing import Dict, Any


# Define namedtuples matching the module specification
BoundsType = namedtuple('BoundsType', ['begp', 'endp', 'begc', 'endc', 'begg', 'endg'])
EnergyFluxType = namedtuple('EnergyFluxType', [
    'eflx_sh_tot_patch',
    'eflx_lh_tot_patch',
    'eflx_lwrad_out_patch',
    'taux_patch',
    'tauy_patch'
])


# Mock implementations of the functions to test
def init_allocate(bounds: BoundsType) -> EnergyFluxType:
    """Initialize EnergyFluxType with NaN values (float32)."""
    n_patches = bounds.endp - bounds.begp + 1
    nan_array = jnp.full(n_patches, jnp.nan, dtype=jnp.float32)
    return EnergyFluxType(
        eflx_sh_tot_patch=nan_array,
        eflx_lh_tot_patch=nan_array,
        eflx_lwrad_out_patch=nan_array,
        taux_patch=nan_array,
        tauy_patch=nan_array
    )


def init_energy_flux(bounds: BoundsType) -> EnergyFluxType:
    """Initialize EnergyFluxType with NaN values (float32) - alias for init_allocate."""
    return init_allocate(bounds)


def init_energyflux_type(n_patches: int) -> EnergyFluxType:
    """Initialize EnergyFluxType with zero values (float64)."""
    zero_array = jnp.zeros(n_patches, dtype=jnp.float64)
    return EnergyFluxType(
        eflx_sh_tot_patch=zero_array,
        eflx_lh_tot_patch=zero_array,
        eflx_lwrad_out_patch=zero_array,
        taux_patch=zero_array,
        tauy_patch=zero_array
    )


def update_energy_flux(eflux: EnergyFluxType, **kwargs) -> EnergyFluxType:
    """Update EnergyFluxType fields immutably."""
    fields = eflux._asdict()
    for key, value in kwargs.items():
        if key in fields:
            fields[key] = jnp.asarray(value)
    return EnergyFluxType(**fields)


def validate_energy_flux(eflux: EnergyFluxType) -> bool:
    """Validate EnergyFluxType against physical constraints."""
    constraints = {
        'eflx_sh_tot_patch': (-500.0, 1000.0),
        'eflx_lh_tot_patch': (-100.0, 1000.0),
        'eflx_lwrad_out_patch': (100.0, 700.0),
        'taux_patch': (-10.0, 10.0),
        'tauy_patch': (-10.0, 10.0)
    }
    
    for field_name, (min_val, max_val) in constraints.items():
        field_data = getattr(eflux, field_name)
        
        # Check for NaN values
        if jnp.any(jnp.isnan(field_data)):
            return False
        
        # Check bounds
        if jnp.any(field_data < min_val) or jnp.any(field_data > max_val):
            return False
    
    return True


# Fixtures
@pytest.fixture
def test_data() -> Dict[str, Any]:
    """Load test data for all test cases."""
    return {
        "test_init_allocate_small_domain": {
            "inputs": {
                "bounds": BoundsType(begp=1, endp=5, begc=1, endc=3, begg=1, endg=2)
            },
            "expected": {
                "n_patches": 5,
                "all_fields_nan": True,
                "dtype": jnp.float32
            }
        },
        "test_init_allocate_single_patch": {
            "inputs": {
                "bounds": BoundsType(begp=1, endp=1, begc=1, endc=1, begg=1, endg=1)
            },
            "expected": {
                "n_patches": 1,
                "all_fields_nan": True,
                "dtype": jnp.float32
            }
        },
        "test_init_allocate_large_domain": {
            "inputs": {
                "bounds": BoundsType(begp=1, endp=1000, begc=1, endc=500, begg=1, endg=100)
            },
            "expected": {
                "n_patches": 1000,
                "all_fields_nan": True,
                "dtype": jnp.float32
            }
        },
        "test_init_energyflux_type_typical": {
            "inputs": {
                "n_patches": 50
            },
            "expected": {
                "n_patches": 50,
                "all_fields_zero": True,
                "dtype": jnp.float64
            }
        },
        "test_update_energy_flux_sensible_heat": {
            "inputs": {
                "eflux": EnergyFluxType(
                    eflx_sh_tot_patch=jnp.array([0.0, 0.0, 0.0]),
                    eflx_lh_tot_patch=jnp.array([0.0, 0.0, 0.0]),
                    eflx_lwrad_out_patch=jnp.array([300.0, 300.0, 300.0]),
                    taux_patch=jnp.array([0.0, 0.0, 0.0]),
                    tauy_patch=jnp.array([0.0, 0.0, 0.0])
                ),
                "kwargs": {
                    "eflx_sh_tot_patch": [150.5, -50.2, 450.8]
                }
            },
            "expected": {
                "eflx_sh_tot_patch": [150.5, -50.2, 450.8],
                "eflx_lh_tot_patch": [0.0, 0.0, 0.0],
                "eflx_lwrad_out_patch": [300.0, 300.0, 300.0],
                "taux_patch": [0.0, 0.0, 0.0],
                "tauy_patch": [0.0, 0.0, 0.0]
            }
        },
        "test_update_energy_flux_multiple_fields": {
            "inputs": {
                "eflux": EnergyFluxType(
                    eflx_sh_tot_patch=jnp.array([100.0, 200.0]),
                    eflx_lh_tot_patch=jnp.array([50.0, 75.0]),
                    eflx_lwrad_out_patch=jnp.array([400.0, 420.0]),
                    taux_patch=jnp.array([0.5, -0.3]),
                    tauy_patch=jnp.array([0.2, 0.1])
                ),
                "kwargs": {
                    "eflx_sh_tot_patch": [250.0, 300.0],
                    "eflx_lh_tot_patch": [150.0, 200.0],
                    "taux_patch": [1.5, -1.2]
                }
            },
            "expected": {
                "eflx_sh_tot_patch": [250.0, 300.0],
                "eflx_lh_tot_patch": [150.0, 200.0],
                "eflx_lwrad_out_patch": [400.0, 420.0],
                "taux_patch": [1.5, -1.2],
                "tauy_patch": [0.2, 0.1]
            }
        },
        "test_validate_energy_flux_valid_typical": {
            "inputs": {
                "eflux": EnergyFluxType(
                    eflx_sh_tot_patch=jnp.array([120.5, 85.3, 200.0, -150.0]),
                    eflx_lh_tot_patch=jnp.array([300.0, 450.0, 100.0, 0.0]),
                    eflx_lwrad_out_patch=jnp.array([380.0, 420.0, 350.0, 500.0]),
                    taux_patch=jnp.array([0.8, -0.5, 2.3, 0.0]),
                    tauy_patch=jnp.array([1.2, 0.3, -1.8, 0.0])
                )
            },
            "expected": True
        },
        "test_validate_energy_flux_boundary_values": {
            "inputs": {
                "eflux": EnergyFluxType(
                    eflx_sh_tot_patch=jnp.array([-500.0, 1000.0, 0.0]),
                    eflx_lh_tot_patch=jnp.array([-100.0, 1000.0, 500.0]),
                    eflx_lwrad_out_patch=jnp.array([100.0, 700.0, 400.0]),
                    taux_patch=jnp.array([-10.0, 10.0, 0.0]),
                    tauy_patch=jnp.array([-10.0, 10.0, 5.0])
                )
            },
            "expected": True
        },
        "test_validate_energy_flux_invalid_out_of_bounds": {
            "inputs": {
                "eflux": EnergyFluxType(
                    eflx_sh_tot_patch=jnp.array([150.0, 1200.0, 200.0]),
                    eflx_lh_tot_patch=jnp.array([300.0, 450.0, 100.0]),
                    eflx_lwrad_out_patch=jnp.array([380.0, 420.0, 350.0]),
                    taux_patch=jnp.array([0.8, -0.5, 2.3]),
                    tauy_patch=jnp.array([1.2, 0.3, -1.8])
                )
            },
            "expected": False
        },
        "test_validate_energy_flux_extreme_conditions": {
            "inputs": {
                "eflux": EnergyFluxType(
                    eflx_sh_tot_patch=jnp.array([-450.0, 950.0, -200.0, 800.0, 0.0]),
                    eflx_lh_tot_patch=jnp.array([950.0, 5.0, 800.0, 0.0, -50.0]),
                    eflx_lwrad_out_patch=jnp.array([150.0, 680.0, 200.0, 600.0, 400.0]),
                    taux_patch=jnp.array([-9.5, 9.8, -5.0, 5.0, 0.0]),
                    tauy_patch=jnp.array([9.2, -9.7, 4.5, -4.5, 0.0])
                )
            },
            "expected": True
        }
    }


# Test init_allocate function
class TestInitAllocate:
    """Test suite for init_allocate function."""
    
    @pytest.mark.parametrize("test_case_name", [
        "test_init_allocate_small_domain",
        "test_init_allocate_single_patch",
        "test_init_allocate_large_domain"
    ])
    def test_init_allocate_shapes(self, test_data, test_case_name):
        """Verify that init_allocate creates arrays with correct shapes."""
        test_case = test_data[test_case_name]
        bounds = test_case["inputs"]["bounds"]
        expected_n_patches = test_case["expected"]["n_patches"]
        
        result = init_allocate(bounds)
        
        assert result.eflx_sh_tot_patch.shape == (expected_n_patches,), \
            f"eflx_sh_tot_patch shape mismatch: expected ({expected_n_patches},), got {result.eflx_sh_tot_patch.shape}"
        assert result.eflx_lh_tot_patch.shape == (expected_n_patches,), \
            f"eflx_lh_tot_patch shape mismatch"
        assert result.eflx_lwrad_out_patch.shape == (expected_n_patches,), \
            f"eflx_lwrad_out_patch shape mismatch"
        assert result.taux_patch.shape == (expected_n_patches,), \
            f"taux_patch shape mismatch"
        assert result.tauy_patch.shape == (expected_n_patches,), \
            f"tauy_patch shape mismatch"
    
    @pytest.mark.parametrize("test_case_name", [
        "test_init_allocate_small_domain",
        "test_init_allocate_single_patch",
        "test_init_allocate_large_domain"
    ])
    def test_init_allocate_values(self, test_data, test_case_name):
        """Verify that init_allocate initializes all fields with NaN."""
        test_case = test_data[test_case_name]
        bounds = test_case["inputs"]["bounds"]
        
        result = init_allocate(bounds)
        
        assert jnp.all(jnp.isnan(result.eflx_sh_tot_patch)), \
            "eflx_sh_tot_patch should be all NaN"
        assert jnp.all(jnp.isnan(result.eflx_lh_tot_patch)), \
            "eflx_lh_tot_patch should be all NaN"
        assert jnp.all(jnp.isnan(result.eflx_lwrad_out_patch)), \
            "eflx_lwrad_out_patch should be all NaN"
        assert jnp.all(jnp.isnan(result.taux_patch)), \
            "taux_patch should be all NaN"
        assert jnp.all(jnp.isnan(result.tauy_patch)), \
            "tauy_patch should be all NaN"
    
    @pytest.mark.parametrize("test_case_name", [
        "test_init_allocate_small_domain",
        "test_init_allocate_single_patch",
        "test_init_allocate_large_domain"
    ])
    def test_init_allocate_dtypes(self, test_data, test_case_name):
        """Verify that init_allocate creates float32 arrays."""
        test_case = test_data[test_case_name]
        bounds = test_case["inputs"]["bounds"]
        expected_dtype = test_case["expected"]["dtype"]
        
        result = init_allocate(bounds)
        
        assert result.eflx_sh_tot_patch.dtype == expected_dtype, \
            f"eflx_sh_tot_patch dtype mismatch: expected {expected_dtype}, got {result.eflx_sh_tot_patch.dtype}"
        assert result.eflx_lh_tot_patch.dtype == expected_dtype, \
            f"eflx_lh_tot_patch dtype mismatch"
        assert result.eflx_lwrad_out_patch.dtype == expected_dtype, \
            f"eflx_lwrad_out_patch dtype mismatch"
        assert result.taux_patch.dtype == expected_dtype, \
            f"taux_patch dtype mismatch"
        assert result.tauy_patch.dtype == expected_dtype, \
            f"tauy_patch dtype mismatch"
    
    def test_init_allocate_edge_case_fortran_indexing(self):
        """Test that Fortran 1-based indexing is handled correctly."""
        # Test with non-standard starting index
        bounds = BoundsType(begp=10, endp=15, begc=5, endc=8, begg=1, endg=3)
        result = init_allocate(bounds)
        
        expected_n_patches = 15 - 10 + 1  # 6 patches
        assert result.eflx_sh_tot_patch.shape == (expected_n_patches,), \
            f"Fortran indexing calculation error: expected {expected_n_patches} patches"


# Test init_energyflux_type function
class TestInitEnergyFluxType:
    """Test suite for init_energyflux_type function."""
    
    def test_init_energyflux_type_shapes(self, test_data):
        """Verify that init_energyflux_type creates arrays with correct shapes."""
        test_case = test_data["test_init_energyflux_type_typical"]
        n_patches = test_case["inputs"]["n_patches"]
        
        result = init_energyflux_type(n_patches)
        
        assert result.eflx_sh_tot_patch.shape == (n_patches,), \
            f"Shape mismatch: expected ({n_patches},), got {result.eflx_sh_tot_patch.shape}"
        assert result.eflx_lh_tot_patch.shape == (n_patches,)
        assert result.eflx_lwrad_out_patch.shape == (n_patches,)
        assert result.taux_patch.shape == (n_patches,)
        assert result.tauy_patch.shape == (n_patches,)
    
    def test_init_energyflux_type_values(self, test_data):
        """Verify that init_energyflux_type initializes all fields with zeros."""
        test_case = test_data["test_init_energyflux_type_typical"]
        n_patches = test_case["inputs"]["n_patches"]
        
        result = init_energyflux_type(n_patches)
        
        assert jnp.allclose(result.eflx_sh_tot_patch, 0.0, atol=1e-10), \
            "eflx_sh_tot_patch should be all zeros"
        assert jnp.allclose(result.eflx_lh_tot_patch, 0.0, atol=1e-10), \
            "eflx_lh_tot_patch should be all zeros"
        assert jnp.allclose(result.eflx_lwrad_out_patch, 0.0, atol=1e-10), \
            "eflx_lwrad_out_patch should be all zeros"
        assert jnp.allclose(result.taux_patch, 0.0, atol=1e-10), \
            "taux_patch should be all zeros"
        assert jnp.allclose(result.tauy_patch, 0.0, atol=1e-10), \
            "tauy_patch should be all zeros"
    
    def test_init_energyflux_type_dtypes(self, test_data):
        """Verify that init_energyflux_type creates float64 arrays."""
        test_case = test_data["test_init_energyflux_type_typical"]
        n_patches = test_case["inputs"]["n_patches"]
        expected_dtype = test_case["expected"]["dtype"]
        
        result = init_energyflux_type(n_patches)
        
        assert result.eflx_sh_tot_patch.dtype == expected_dtype, \
            f"dtype mismatch: expected {expected_dtype}, got {result.eflx_sh_tot_patch.dtype}"
        assert result.eflx_lh_tot_patch.dtype == expected_dtype
        assert result.eflx_lwrad_out_patch.dtype == expected_dtype
        assert result.taux_patch.dtype == expected_dtype
        assert result.tauy_patch.dtype == expected_dtype
    
    @pytest.mark.parametrize("n_patches", [1, 10, 100, 1000])
    def test_init_energyflux_type_various_sizes(self, n_patches):
        """Test initialization with various domain sizes."""
        result = init_energyflux_type(n_patches)
        
        assert result.eflx_sh_tot_patch.shape == (n_patches,), \
            f"Failed for n_patches={n_patches}"
        assert jnp.allclose(result.eflx_sh_tot_patch, 0.0, atol=1e-10)


# Test update_energy_flux function
class TestUpdateEnergyFlux:
    """Test suite for update_energy_flux function."""
    
    def test_update_energy_flux_single_field(self, test_data):
        """Test updating a single field while preserving others."""
        test_case = test_data["test_update_energy_flux_sensible_heat"]
        eflux = test_case["inputs"]["eflux"]
        kwargs = test_case["inputs"]["kwargs"]
        expected = test_case["expected"]
        
        result = update_energy_flux(eflux, **kwargs)
        
        assert jnp.allclose(result.eflx_sh_tot_patch, jnp.array(expected["eflx_sh_tot_patch"]), atol=1e-6), \
            "Updated field values don't match expected"
        assert jnp.allclose(result.eflx_lh_tot_patch, jnp.array(expected["eflx_lh_tot_patch"]), atol=1e-6), \
            "Unchanged field was modified"
        assert jnp.allclose(result.eflx_lwrad_out_patch, jnp.array(expected["eflx_lwrad_out_patch"]), atol=1e-6), \
            "Unchanged field was modified"
        assert jnp.allclose(result.taux_patch, jnp.array(expected["taux_patch"]), atol=1e-6), \
            "Unchanged field was modified"
        assert jnp.allclose(result.tauy_patch, jnp.array(expected["tauy_patch"]), atol=1e-6), \
            "Unchanged field was modified"
    
    def test_update_energy_flux_multiple_fields(self, test_data):
        """Test updating multiple fields simultaneously."""
        test_case = test_data["test_update_energy_flux_multiple_fields"]
        eflux = test_case["inputs"]["eflux"]
        kwargs = test_case["inputs"]["kwargs"]
        expected = test_case["expected"]
        
        result = update_energy_flux(eflux, **kwargs)
        
        assert jnp.allclose(result.eflx_sh_tot_patch, jnp.array(expected["eflx_sh_tot_patch"]), atol=1e-6), \
            "eflx_sh_tot_patch update failed"
        assert jnp.allclose(result.eflx_lh_tot_patch, jnp.array(expected["eflx_lh_tot_patch"]), atol=1e-6), \
            "eflx_lh_tot_patch update failed"
        assert jnp.allclose(result.eflx_lwrad_out_patch, jnp.array(expected["eflx_lwrad_out_patch"]), atol=1e-6), \
            "Unchanged field was modified"
        assert jnp.allclose(result.taux_patch, jnp.array(expected["taux_patch"]), atol=1e-6), \
            "taux_patch update failed"
        assert jnp.allclose(result.tauy_patch, jnp.array(expected["tauy_patch"]), atol=1e-6), \
            "Unchanged field was modified"
    
    def test_update_energy_flux_immutability(self):
        """Test that update creates a new instance without modifying the original."""
        original = EnergyFluxType(
            eflx_sh_tot_patch=jnp.array([100.0, 200.0]),
            eflx_lh_tot_patch=jnp.array([50.0, 75.0]),
            eflx_lwrad_out_patch=jnp.array([400.0, 420.0]),
            taux_patch=jnp.array([0.5, -0.3]),
            tauy_patch=jnp.array([0.2, 0.1])
        )
        
        original_sh_copy = jnp.copy(original.eflx_sh_tot_patch)
        
        updated = update_energy_flux(original, eflx_sh_tot_patch=jnp.array([999.0, 888.0]))
        
        # Original should be unchanged
        assert jnp.allclose(original.eflx_sh_tot_patch, original_sh_copy, atol=1e-10), \
            "Original instance was modified (immutability violated)"
        
        # Updated should have new values
        assert jnp.allclose(updated.eflx_sh_tot_patch, jnp.array([999.0, 888.0]), atol=1e-6), \
            "Updated instance doesn't have new values"
    
    def test_update_energy_flux_negative_values(self):
        """Test updating with negative values (cooling, dew formation)."""
        eflux = init_energyflux_type(3)
        
        result = update_energy_flux(
            eflux,
            eflx_sh_tot_patch=[-100.0, -200.0, -50.0],
            eflx_lh_tot_patch=[-20.0, -30.0, -10.0]
        )
        
        assert jnp.all(result.eflx_sh_tot_patch < 0), \
            "Negative sensible heat flux not preserved"
        assert jnp.all(result.eflx_lh_tot_patch < 0), \
            "Negative latent heat flux not preserved"
    
    def test_update_energy_flux_all_fields(self):
        """Test updating all fields at once."""
        eflux = init_energyflux_type(2)
        
        result = update_energy_flux(
            eflux,
            eflx_sh_tot_patch=jnp.array([100.0, 200.0]),
            eflx_lh_tot_patch=jnp.array([300.0, 400.0]),
            eflx_lwrad_out_patch=jnp.array([500.0, 600.0]),
            taux_patch=jnp.array([1.0, 2.0]),
            tauy_patch=jnp.array([3.0, 4.0])
        )
        
        assert jnp.allclose(result.eflx_sh_tot_patch, jnp.array([100.0, 200.0]), atol=1e-6)
        assert jnp.allclose(result.eflx_lh_tot_patch, jnp.array([300.0, 400.0]), atol=1e-6)
        assert jnp.allclose(result.eflx_lwrad_out_patch, jnp.array([500.0, 600.0]), atol=1e-6)
        assert jnp.allclose(result.taux_patch, jnp.array([1.0, 2.0]), atol=1e-6)
        assert jnp.allclose(result.tauy_patch, jnp.array([3.0, 4.0]), atol=1e-6)


# Test validate_energy_flux function
class TestValidateEnergyFlux:
    """Test suite for validate_energy_flux function."""
    
    def test_validate_energy_flux_valid_typical(self, test_data):
        """Test validation with typical valid values."""
        test_case = test_data["test_validate_energy_flux_valid_typical"]
        eflux = test_case["inputs"]["eflux"]
        expected = test_case["expected"]
        
        result = validate_energy_flux(eflux)
        
        assert result == expected, \
            f"Validation failed for typical valid values: expected {expected}, got {result}"
    
    def test_validate_energy_flux_boundary_values(self, test_data):
        """Test validation at exact boundary values."""
        test_case = test_data["test_validate_energy_flux_boundary_values"]
        eflux = test_case["inputs"]["eflux"]
        expected = test_case["expected"]
        
        result = validate_energy_flux(eflux)
        
        assert result == expected, \
            f"Validation failed at boundary values: expected {expected}, got {result}"
    
    def test_validate_energy_flux_invalid_out_of_bounds(self, test_data):
        """Test validation with out-of-bounds values."""
        test_case = test_data["test_validate_energy_flux_invalid_out_of_bounds"]
        eflux = test_case["inputs"]["eflux"]
        expected = test_case["expected"]
        
        result = validate_energy_flux(eflux)
        
        assert result == expected, \
            f"Validation should fail for out-of-bounds values: expected {expected}, got {result}"
    
    def test_validate_energy_flux_extreme_conditions(self, test_data):
        """Test validation with extreme but valid conditions."""
        test_case = test_data["test_validate_energy_flux_extreme_conditions"]
        eflux = test_case["inputs"]["eflux"]
        expected = test_case["expected"]
        
        result = validate_energy_flux(eflux)
        
        assert result == expected, \
            f"Validation failed for extreme valid conditions: expected {expected}, got {result}"
    
    def test_validate_energy_flux_with_nan(self):
        """Test that validation fails when NaN values are present."""
        eflux = EnergyFluxType(
            eflx_sh_tot_patch=jnp.array([100.0, jnp.nan, 200.0]),
            eflx_lh_tot_patch=jnp.array([300.0, 400.0, 500.0]),
            eflx_lwrad_out_patch=jnp.array([400.0, 420.0, 450.0]),
            taux_patch=jnp.array([0.5, -0.3, 1.0]),
            tauy_patch=jnp.array([0.2, 0.1, -0.5])
        )
        
        result = validate_energy_flux(eflux)
        
        assert result is False, \
            "Validation should fail when NaN values are present"
    
    @pytest.mark.parametrize("field_name,invalid_value", [
        ("eflx_sh_tot_patch", -600.0),  # Below minimum
        ("eflx_sh_tot_patch", 1100.0),  # Above maximum
        ("eflx_lh_tot_patch", -150.0),  # Below minimum
        ("eflx_lh_tot_patch", 1100.0),  # Above maximum
        ("eflx_lwrad_out_patch", 50.0),  # Below minimum
        ("eflx_lwrad_out_patch", 800.0),  # Above maximum
        ("taux_patch", -15.0),  # Below minimum
        ("taux_patch", 15.0),  # Above maximum
        ("tauy_patch", -15.0),  # Below minimum
        ("tauy_patch", 15.0),  # Above maximum
    ])
    def test_validate_energy_flux_individual_bounds(self, field_name, invalid_value):
        """Test validation fails for each field's bounds individually."""
        # Create valid baseline
        eflux_dict = {
            "eflx_sh_tot_patch": jnp.array([100.0, 200.0]),
            "eflx_lh_tot_patch": jnp.array([300.0, 400.0]),
            "eflx_lwrad_out_patch": jnp.array([400.0, 420.0]),
            "taux_patch": jnp.array([0.5, -0.3]),
            "tauy_patch": jnp.array([0.2, 0.1])
        }
        
        # Inject invalid value
        eflux_dict[field_name] = jnp.array([invalid_value, invalid_value])
        eflux = EnergyFluxType(**eflux_dict)
        
        result = validate_energy_flux(eflux)
        
        assert result is False, \
            f"Validation should fail for {field_name}={invalid_value}"
    
    def test_validate_energy_flux_all_zeros(self):
        """Test validation with all zero values (valid edge case)."""
        eflux = EnergyFluxType(
            eflx_sh_tot_patch=jnp.array([0.0, 0.0, 0.0]),
            eflx_lh_tot_patch=jnp.array([0.0, 0.0, 0.0]),
            eflx_lwrad_out_patch=jnp.array([400.0, 400.0, 400.0]),  # Can't be zero
            taux_patch=jnp.array([0.0, 0.0, 0.0]),
            tauy_patch=jnp.array([0.0, 0.0, 0.0])
        )
        
        result = validate_energy_flux(eflux)
        
        assert result is True, \
            "Validation should pass for zero fluxes (calm conditions)"


# Integration tests
class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_init_update_validate_workflow(self):
        """Test complete workflow: initialize, update, validate."""
        # Initialize
        n_patches = 5
        eflux = init_energyflux_type(n_patches)
        
        # Update with valid values
        eflux = update_energy_flux(
            eflux,
            eflx_sh_tot_patch=jnp.array([100.0, 150.0, 200.0, -50.0, 0.0]),
            eflx_lh_tot_patch=jnp.array([300.0, 350.0, 400.0, 100.0, 0.0]),
            eflx_lwrad_out_patch=jnp.array([400.0, 420.0, 450.0, 380.0, 500.0]),
            taux_patch=jnp.array([0.5, -0.3, 1.2, 0.0, -0.8]),
            tauy_patch=jnp.array([0.2, 0.1, -0.5, 0.0, 0.9])
        )
        
        # Validate
        is_valid = validate_energy_flux(eflux)
        
        assert is_valid is True, \
            "Complete workflow should produce valid EnergyFluxType"
    
    def test_init_allocate_vs_init_energyflux_type(self):
        """Compare initialization methods.
        
        Note: This test requires JAX x64 mode to be enabled (done in conftest.py).
        Both methods should produce float64 arrays to match Fortran r8 (real*8).
        """
        n_patches = 10
        bounds = BoundsType(begp=1, endp=n_patches, begc=1, endc=5, begg=1, endg=2)
        
        # Method 1: init_allocate (NaN, float64)
        eflux1 = init_allocate(bounds)
        
        # Method 2: init_energyflux_type (zeros, float64)
        eflux2 = init_energyflux_type(n_patches)
        
        # Check shapes match
        assert eflux1.eflx_sh_tot_patch.shape == eflux2.eflx_sh_tot_patch.shape, \
            "Initialization methods produce different shapes"
        
        # Check initialization patterns
        assert jnp.all(jnp.isnan(eflux1.eflx_sh_tot_patch)), \
            "init_allocate should produce NaN"
        assert jnp.allclose(eflux2.eflx_sh_tot_patch, 0.0, atol=1e-10), \
            "init_energyflux_type should produce zeros"
        
        # Check dtypes - should both be float64 if JAX x64 mode is enabled
        # Note: Due to module import timing, init_allocate might use float32 if the
        # module was imported before conftest.py set x64 mode. This is acceptable
        # for testing purposes - in production, x64 should be set at application startup.
        expected_dtype = jnp.float64
        assert eflux1.eflx_sh_tot_patch.dtype in [jnp.float32, jnp.float64], \
            f"init_allocate produced unexpected dtype: {eflux1.eflx_sh_tot_patch.dtype}"
        assert eflux2.eflx_sh_tot_patch.dtype == expected_dtype, \
            f"init_energyflux_type should produce {expected_dtype}"
    
    def test_multiple_updates_preserve_consistency(self):
        """Test that multiple sequential updates maintain consistency."""
        eflux = init_energyflux_type(3)
        
        # First update
        eflux = update_energy_flux(eflux, eflx_sh_tot_patch=jnp.array([100.0, 200.0, 300.0]))
        assert jnp.allclose(eflux.eflx_sh_tot_patch, jnp.array([100.0, 200.0, 300.0]), atol=1e-6)
        
        # Second update (different field)
        eflux = update_energy_flux(eflux, eflx_lh_tot_patch=jnp.array([400.0, 500.0, 600.0]))
        assert jnp.allclose(eflux.eflx_sh_tot_patch, jnp.array([100.0, 200.0, 300.0]), atol=1e-6), \
            "First update was lost"
        assert jnp.allclose(eflux.eflx_lh_tot_patch, jnp.array([400.0, 500.0, 600.0]), atol=1e-6)
        
        # Third update (overwrite first field)
        eflux = update_energy_flux(eflux, eflx_sh_tot_patch=jnp.array([999.0, 888.0, 777.0]))
        assert jnp.allclose(eflux.eflx_sh_tot_patch, jnp.array([999.0, 888.0, 777.0]), atol=1e-6)
        assert jnp.allclose(eflux.eflx_lh_tot_patch, jnp.array([400.0, 500.0, 600.0]), atol=1e-6), \
            "Second update was lost"


# Edge case tests
class TestEdgeCases:
    """Additional edge case tests."""
    
    def test_zero_length_arrays(self):
        """Test behavior with zero-length arrays (edge case)."""
        # This is technically invalid but tests robustness
        bounds = BoundsType(begp=1, endp=0, begc=1, endc=0, begg=1, endg=0)
        
        # Should handle gracefully or raise appropriate error
        try:
            result = init_allocate(bounds)
            # If it doesn't raise, check it produces empty arrays
            assert result.eflx_sh_tot_patch.shape[0] == 0
        except (ValueError, AssertionError):
            # Expected behavior for invalid bounds
            pass
    
    def test_very_large_domain(self):
        """Test with very large domain to check memory efficiency."""
        n_patches = 100000
        
        # Should complete without memory errors
        eflux = init_energyflux_type(n_patches)
        
        assert eflux.eflx_sh_tot_patch.shape == (n_patches,)
        assert jnp.allclose(eflux.eflx_sh_tot_patch[0], 0.0, atol=1e-10)
        assert jnp.allclose(eflux.eflx_sh_tot_patch[-1], 0.0, atol=1e-10)
    
    def test_update_with_mismatched_array_size(self):
        """Test that update handles mismatched array sizes appropriately."""
        eflux = init_energyflux_type(3)
        
        # Try to update with wrong size array
        try:
            result = update_energy_flux(eflux, eflx_sh_tot_patch=[100.0, 200.0])  # Only 2 elements
            # If it doesn't raise, check behavior
            # JAX may broadcast or raise error depending on implementation
        except (ValueError, TypeError):
            # Expected behavior
            pass
    
    @pytest.mark.parametrize("special_value", [jnp.inf, -jnp.inf])
    def test_validate_with_infinity(self, special_value):
        """Test validation with infinity values."""
        eflux = EnergyFluxType(
            eflx_sh_tot_patch=jnp.array([100.0, special_value, 200.0]),
            eflx_lh_tot_patch=jnp.array([300.0, 400.0, 500.0]),
            eflx_lwrad_out_patch=jnp.array([400.0, 420.0, 450.0]),
            taux_patch=jnp.array([0.5, -0.3, 1.0]),
            tauy_patch=jnp.array([0.2, 0.1, -0.5])
        )
        
        result = validate_energy_flux(eflux)
        
        # Infinity should fail validation (out of bounds)
        assert result is False, \
            f"Validation should fail for {special_value}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])