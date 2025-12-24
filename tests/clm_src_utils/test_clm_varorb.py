"""
Pytest suite for clm_varorb module - orbital parameters for CTSM.

Tests the OrbitalParams NamedTuple and its factory/update functions for
managing Earth's orbital parameters used in solar radiation calculations.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clm_src_utils.clm_varorb import (
    OrbitalParams,
    create_orbital_params,
    update_orbital_params,
)


class TestOrbitalParams:
    """Tests for OrbitalParams NamedTuple."""
    
    def test_orbital_params_creation_with_defaults(self):
        """Test creating OrbitalParams with default values."""
        params = create_orbital_params()
        
        assert isinstance(params, OrbitalParams)
        assert hasattr(params, 'eccen')
        assert hasattr(params, 'obliqr')
        assert hasattr(params, 'lambm0')
        assert hasattr(params, 'mvelpp')
    
    def test_orbital_params_eccen_range(self):
        """Test eccentricity is within valid range [0, 1)."""
        params = create_orbital_params(eccen=0.0167)
        
        assert 0.0 <= params.eccen < 1.0
        # Check it's a numeric type (float or JAX array)
        assert isinstance(params.eccen, (float, int)) or hasattr(params.eccen, 'dtype')
    
    def test_orbital_params_obliquity_range(self):
        """Test obliquity is within valid range for Earth (~0.4 rad = 23°)."""
        params = create_orbital_params(obliqr=0.409)
        
        # Obliquity should be positive and less than pi/4 (45°)
        assert 0.0 < params.obliqr < jnp.pi / 4
    
    def test_orbital_params_custom_values(self):
        """Test creating OrbitalParams with custom values."""
        eccen = 0.0167
        obliqr = 0.409
        lambm0 = 1.5
        mvelpp = 4.7
        
        params = create_orbital_params(
            eccen=eccen,
            obliqr=obliqr,
            lambm0=lambm0,
            mvelpp=mvelpp
        )
        
        assert params.eccen == eccen
        assert params.obliqr == obliqr
        assert params.lambm0 == lambm0
        assert params.mvelpp == mvelpp
    
    def test_update_orbital_params_immutability(self):
        """Test that update_orbital_params returns new instance."""
        original = create_orbital_params(eccen=0.01)
        updated = update_orbital_params(original, eccen=0.02)
        
        # Original should be unchanged
        assert original.eccen == 0.01
        # Updated should have new value
        assert updated.eccen == 0.02
        # They should be different instances
        assert original is not updated
    
    def test_update_orbital_params_partial_update(self):
        """Test updating only some parameters."""
        original = create_orbital_params(eccen=0.01, obliqr=0.40)
        updated = update_orbital_params(original, eccen=0.02)
        
        # Updated parameter
        assert updated.eccen == 0.02
        # Unchanged parameter
        assert updated.obliqr == original.obliqr
    
    def test_update_orbital_params_multiple_fields(self):
        """Test updating multiple parameters at once."""
        original = create_orbital_params()
        updated = update_orbital_params(
            original,
            eccen=0.03,
            obliqr=0.42,
            lambm0=2.0
        )
        
        assert updated.eccen == 0.03
        assert updated.obliqr == 0.42
        assert updated.lambm0 == 2.0
        # mvelpp should be unchanged
        assert updated.mvelpp == original.mvelpp
    
    def test_orbital_params_namedtuple_properties(self):
        """Test that OrbitalParams has NamedTuple properties."""
        params = create_orbital_params(eccen=0.0167, obliqr=0.409)
        
        # Can be unpacked
        eccen, obliqr, lambm0, mvelpp = params
        assert eccen == params.eccen
        
        # Has _fields attribute
        assert hasattr(params, '_fields')
        assert 'eccen' in params._fields
        assert 'obliqr' in params._fields
    
    def test_orbital_params_jax_compatibility(self):
        """Test that orbital params work in JAX context."""
        params = create_orbital_params(eccen=0.0167)
        
        # Should work with JAX operations
        scaled_eccen = params.eccen * jnp.array(2.0)
        assert jnp.allclose(scaled_eccen, 0.0334)


class TestOrbitalParamsPhysicalRealism:
    """Tests for physical realism of orbital parameters."""
    
    def test_typical_earth_values(self):
        """Test creating params with typical Earth orbital values."""
        # Current Earth orbital parameters
        params = create_orbital_params(
            eccen=0.0167,  # Current Earth eccentricity
            obliqr=0.4091  # Current Earth obliquity (~23.44°)
        )
        
        assert params.eccen == pytest.approx(0.0167, abs=0.0001)
        assert params.obliqr == pytest.approx(0.4091, abs=0.001)
    
    def test_extreme_but_valid_eccentricity(self):
        """Test with high but physically valid eccentricity."""
        # Mars has eccentricity ~0.093
        params = create_orbital_params(eccen=0.09)
        assert 0.0 <= params.eccen < 1.0
    
    def test_different_obliquity_scenarios(self):
        """Test various obliquity values within Earth's range."""
        # Earth's obliquity varies between ~22.1° and 24.5° over 41,000 years
        min_obliq = jnp.radians(22.1)
        max_obliq = jnp.radians(24.5)
        
        params_min = create_orbital_params(obliqr=float(min_obliq))
        params_max = create_orbital_params(obliqr=float(max_obliq))
        
        assert params_min.obliqr >= 0
        assert params_max.obliqr >= 0
        assert params_min.obliqr < params_max.obliqr
