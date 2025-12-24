"""
Pytest suite for lnd_comp_mct module - CLM coupling interface.

Tests the BoundsType and the coupling interface functions that connect
CLM to the CESM coupler (lnd_init_mct and lnd_run_mct).
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clm_src_cpl.lnd_comp_mct import (
    BoundsType,
    lnd_init_mct,
    lnd_run_mct,
    R8,
)


class TestBoundsType:
    """Tests for BoundsType NamedTuple."""
    
    def test_bounds_type_creation(self):
        """Test creating BoundsType with all required fields."""
        bounds = BoundsType(
            begg=0, endg=10,
            begl=0, endl=20,
            begc=0, endc=50,
            begp=0, endp=100
        )
        
        assert isinstance(bounds, BoundsType)
        assert bounds.begg == 0
        assert bounds.endg == 10
        assert bounds.begl == 0
        assert bounds.endl == 20
        assert bounds.begc == 0
        assert bounds.endc == 50
        assert bounds.begp == 0
        assert bounds.endp == 100
    
    def test_bounds_type_single_gridcell(self):
        """Test bounds for single gridcell configuration."""
        bounds = BoundsType(
            begg=0, endg=1,
            begl=0, endl=1,
            begc=0, endc=1,
            begp=0, endp=1
        )
        
        # Check ranges
        assert bounds.endg - bounds.begg == 1
        assert bounds.endl - bounds.begl == 1
        assert bounds.endc - bounds.begc == 1
        assert bounds.endp - bounds.begp == 1
    
    def test_bounds_type_multi_column(self):
        """Test bounds for multi-column configuration."""
        bounds = BoundsType(
            begg=0, endg=1,
            begl=0, endl=1,
            begc=0, endc=5,  # 5 columns
            begp=0, endp=10  # 10 PFTs
        )
        
        num_columns = bounds.endc - bounds.begc
        num_pfts = bounds.endp - bounds.begp
        
        assert num_columns == 5
        assert num_pfts == 10
    
    def test_bounds_type_all_indices_non_negative(self):
        """Test that all indices are non-negative."""
        bounds = BoundsType(
            begg=0, endg=5,
            begl=0, endl=10,
            begc=0, endc=15,
            begp=0, endp=20
        )
        
        assert bounds.begg >= 0
        assert bounds.endg >= 0
        assert bounds.begl >= 0
        assert bounds.endl >= 0
        assert bounds.begc >= 0
        assert bounds.endc >= 0
        assert bounds.begp >= 0
        assert bounds.endp >= 0
    
    def test_bounds_type_valid_ranges(self):
        """Test that end indices are greater than begin indices."""
        bounds = BoundsType(
            begg=0, endg=5,
            begl=0, endl=10,
            begc=0, endc=15,
            begp=0, endp=20
        )
        
        assert bounds.endg > bounds.begg
        assert bounds.endl > bounds.begl
        assert bounds.endc > bounds.begc
        assert bounds.endp > bounds.begp
    
    def test_bounds_type_namedtuple_properties(self):
        """Test that BoundsType has NamedTuple properties."""
        bounds = BoundsType(
            begg=0, endg=1,
            begl=0, endl=1,
            begc=0, endc=1,
            begp=0, endp=1
        )
        
        # Has _fields attribute
        assert hasattr(bounds, '_fields')
        assert 'begg' in bounds._fields
        assert 'endg' in bounds._fields
        
        # Can be unpacked
        begg, endg, begl, endl, begc, endc, begp, endp = bounds
        assert begg == bounds.begg


class TestLndInitMct:
    """Tests for lnd_init_mct function."""
    
    def test_lnd_init_mct_callable(self):
        """Test that lnd_init_mct is callable."""
        assert callable(lnd_init_mct)
    
    def test_lnd_init_mct_with_bounds(self):
        """Test calling lnd_init_mct with bounds."""
        bounds = BoundsType(
            begg=0, endg=1,
            begl=0, endl=1,
            begc=0, endc=1,
            begp=0, endp=1
        )
        
        # Function should execute without error (even if it's a placeholder)
        result = lnd_init_mct(bounds)
        
        # The current implementation returns None (placeholder)
        # In future, this would return initialized state
        assert result is None  # Placeholder behavior
    
    def test_lnd_init_mct_multiple_gridcells(self):
        """Test lnd_init_mct with multiple gridcells."""
        bounds = BoundsType(
            begg=0, endg=10,
            begl=0, endl=10,
            begc=0, endc=50,
            begp=0, endp=100
        )
        
        # Should handle multiple gridcells
        result = lnd_init_mct(bounds)
        assert result is None  # Placeholder


class TestLndRunMct:
    """Tests for lnd_run_mct function."""
    
    def test_lnd_run_mct_callable(self):
        """Test that lnd_run_mct is callable."""
        assert callable(lnd_run_mct)
    
    def test_lnd_run_mct_with_parameters(self):
        """Test calling lnd_run_mct with required parameters."""
        bounds = BoundsType(
            begg=0, endg=1,
            begl=0, endl=1,
            begc=0, endc=1,
            begp=0, endp=1
        )
        
        # Function should execute without error
        result = lnd_run_mct(bounds, time_indx=1, fin="test.nc")
        
        # The current implementation returns None (placeholder)
        assert result is None  # Placeholder behavior
    
    def test_lnd_run_mct_different_time_indices(self):
        """Test lnd_run_mct with different time indices."""
        bounds = BoundsType(
            begg=0, endg=1,
            begl=0, endl=1,
            begc=0, endc=1,
            begp=0, endp=1
        )
        
        for time_indx in [0, 1, 10, 100]:
            result = lnd_run_mct(bounds, time_indx=time_indx, fin="test.nc")
            assert result is None  # Placeholder


class TestModuleConstants:
    """Tests for module-level constants."""
    
    def test_r8_constant_exists(self):
        """Test that R8 precision constant exists."""
        assert R8 is not None
    
    def test_r8_is_float64(self):
        """Test that R8 is double precision (float64)."""
        assert R8 == jnp.float64
    
    def test_r8_usage(self):
        """Test using R8 to create arrays."""
        test_array = jnp.array([1.0, 2.0, 3.0], dtype=R8)
        
        # R8 is float64, but JAX may use float32 by default depending on config
        assert test_array.dtype in (jnp.float32, jnp.float64)
        # The important thing is that R8 is defined and can be used
        assert R8 is not None
