"""
Pytest suite for clm_varctl module - CLM run control variables.

Tests the ClmVarCtl NamedTuple and its factory/update functions for
managing CLM run control parameters and configuration flags.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clm_src_main.clm_varctl import (
    ClmVarCtl,
    create_clm_varctl,
    update_clm_varctl,
    DEFAULT_CLM_VARCTL,
)


class TestClmVarCtl:
    """Tests for ClmVarCtl NamedTuple."""
    
    def test_clm_varctl_creation_with_defaults(self):
        """Test creating ClmVarCtl with default values."""
        ctl = create_clm_varctl()
        
        assert isinstance(ctl, ClmVarCtl)
        assert hasattr(ctl, 'iulog')
    
    def test_default_clm_varctl_exists(self):
        """Test that DEFAULT_CLM_VARCTL is available."""
        assert isinstance(DEFAULT_CLM_VARCTL, ClmVarCtl)
        assert hasattr(DEFAULT_CLM_VARCTL, 'iulog')
    
    def test_clm_varctl_default_iulog_value(self):
        """Test that default iulog is 6 (stdout)."""
        ctl = create_clm_varctl()
        
        # Default Fortran stdout unit number is 6
        assert ctl.iulog == 6
    
    def test_clm_varctl_custom_iulog(self):
        """Test creating ClmVarCtl with custom iulog value."""
        custom_log = 10
        ctl = create_clm_varctl(iulog=custom_log)
        
        assert ctl.iulog == custom_log
    
    def test_update_clm_varctl_immutability(self):
        """Test that update_clm_varctl returns new instance."""
        original = create_clm_varctl(iulog=6)
        updated = update_clm_varctl(original, iulog=7)
        
        # Original should be unchanged
        assert original.iulog == 6
        # Updated should have new value
        assert updated.iulog == 7
        # They should be different instances
        assert original is not updated
    
    def test_update_clm_varctl_preserves_type(self):
        """Test that updated instance is still ClmVarCtl."""
        original = create_clm_varctl()
        updated = update_clm_varctl(original, iulog=10)
        
        assert isinstance(updated, ClmVarCtl)
        assert type(updated) == type(original)
    
    def test_clm_varctl_valid_log_unit_range(self):
        """Test that log units are within valid Fortran I/O range."""
        # Fortran I/O units are typically 1-99, with 5-6 reserved for stdin/stdout
        for log_unit in [6, 10, 20, 50, 99]:
            ctl = create_clm_varctl(iulog=log_unit)
            assert 0 < ctl.iulog < 100
    
    def test_clm_varctl_namedtuple_properties(self):
        """Test that ClmVarCtl has NamedTuple properties."""
        ctl = create_clm_varctl(iulog=6)
        
        # Has _fields attribute
        assert hasattr(ctl, '_fields')
        assert 'iulog' in ctl._fields
        
        # Can access by index
        assert ctl[0] == ctl.iulog
    
    def test_multiple_updates_chain(self):
        """Test chaining multiple updates."""
        ctl1 = create_clm_varctl(iulog=6)
        ctl2 = update_clm_varctl(ctl1, iulog=7)
        ctl3 = update_clm_varctl(ctl2, iulog=8)
        
        # Each should have its own value
        assert ctl1.iulog == 6
        assert ctl2.iulog == 7
        assert ctl3.iulog == 8
    
    def test_default_varctl_is_immutable(self):
        """Test that DEFAULT_CLM_VARCTL cannot be modified in-place."""
        original_value = DEFAULT_CLM_VARCTL.iulog
        
        # Update should return new instance
        updated = update_clm_varctl(DEFAULT_CLM_VARCTL, iulog=99)
        
        # Default should be unchanged
        assert DEFAULT_CLM_VARCTL.iulog == original_value
        assert updated.iulog == 99


class TestClmVarCtlUsagePatterns:
    """Tests for common usage patterns of ClmVarCtl."""
    
    def test_use_default_configuration(self):
        """Test using default configuration."""
        ctl = DEFAULT_CLM_VARCTL
        
        assert ctl.iulog == 6
        assert isinstance(ctl, ClmVarCtl)
    
    def test_create_custom_configuration(self):
        """Test creating custom configuration for specific simulation."""
        custom_ctl = create_clm_varctl(iulog=15)
        
        assert custom_ctl.iulog == 15
        assert custom_ctl.iulog != DEFAULT_CLM_VARCTL.iulog
    
    def test_update_for_runtime_changes(self):
        """Test updating configuration at runtime."""
        ctl = DEFAULT_CLM_VARCTL
        
        # Switch logging during run
        new_ctl = update_clm_varctl(ctl, iulog=20)
        
        assert new_ctl.iulog == 20
        assert ctl.iulog == DEFAULT_CLM_VARCTL.iulog  # Original unchanged
    
    def test_configuration_in_jax_context(self):
        """Test that configuration works in JAX transformations."""
        ctl = create_clm_varctl(iulog=6)
        
        # Can be used in JAX operations (though iulog is just an int)
        log_unit_array = jnp.array(ctl.iulog)
        assert log_unit_array == 6
