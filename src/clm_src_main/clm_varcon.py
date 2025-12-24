"""
CLM variable constants module

This module contains various model constants used throughout CLM including
mathematical constants, physical constants, and special value flags.
Translated from Fortran CLM code to Python JAX.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Final

# Import dependencies
try:
    from ..cime_src_share_util.shr_kind_mod import r8
except ImportError:
    # Fallback for when running outside package context
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cime_src_share_util.shr_kind_mod import r8


# Mathematical constants used in CLM
rpi: Final[float] = 3.141592654  # pi


# Physical constants used in CLM
class PhysicalConstants:
    """Physical constants used in CLM calculations"""
    
    # Temperature constants
    tfrz: Final[float] = 273.15  # Freezing point of water (K)
    
    # Fundamental physical constants
    sb: Final[float] = 5.67e-08  # Stefan-Boltzmann constant (W/m2/K4)
    grav: Final[float] = 9.80665  # Gravitational acceleration (m/s2)
    vkc: Final[float] = 0.4  # von Karman constant
    
    # Density constants
    denh2o: Final[float] = 1000.0  # Density of liquid water (kg/m3)
    denice: Final[float] = 917.0   # Density of ice (kg/m3)
    
    # Thermal conductivity constants
    tkwat: Final[float] = 0.57   # Thermal conductivity of water (W/m/K)
    tkice: Final[float] = 2.29   # Thermal conductivity of ice (W/m/K)
    tkair: Final[float] = 0.023  # Thermal conductivity of air (W/m/K)
    
    # Latent heat constants
    hfus: Final[float] = 0.3337e6   # Latent heat of fusion for water at 0 C (J/kg)
    hvap: Final[float] = 2.5010e6   # Latent heat of evaporation (J/kg)
    hsub: Final[float] = 2.8347e6   # Latent heat of sublimation (J/kg)
    
    # Specific heat constants
    cpice: Final[float] = 2.11727e3  # Specific heat of ice (J/kg/K)
    cpliq: Final[float] = 4.188e3    # Specific heat of water (J/kg/K)


# Constants used in CLM for bedrock
class BedrockConstants:
    """Constants related to bedrock properties"""
    
    thk_bedrock: Final[float] = 3.0      # Thermal conductivity of saturated granitic rock (W/m/K)
    csol_bedrock: Final[float] = 2.0e6   # Vol. heat capacity of granite/sandstone (J/m3/K)
    zmin_bedrock: Final[float] = 0.4     # Minimum depth to bedrock (soil depth) [m]


# Special value flags used in CLM
class SpecialValues:
    """Special value flags for missing or invalid data"""
    
    spval: Final[float] = 1.e36    # Special value for real data
    ispval: Final[int] = -9999     # Special value for integer data


# For backwards compatibility and direct access (matching Fortran module variables)
# Mathematical constants
rpi = PhysicalConstants.tfrz / PhysicalConstants.tfrz * 3.141592654  # pi

# Physical constants
tfrz = PhysicalConstants.tfrz
sb = PhysicalConstants.sb
grav = PhysicalConstants.grav
vkc = PhysicalConstants.vkc
denh2o = PhysicalConstants.denh2o
denice = PhysicalConstants.denice
tkwat = PhysicalConstants.tkwat
tkice = PhysicalConstants.tkice
tkair = PhysicalConstants.tkair
hfus = PhysicalConstants.hfus
hvap = PhysicalConstants.hvap
hsub = PhysicalConstants.hsub
cpice = PhysicalConstants.cpice
cpliq = PhysicalConstants.cpliq

# Bedrock constants
thk_bedrock = BedrockConstants.thk_bedrock
csol_bedrock = BedrockConstants.csol_bedrock
zmin_bedrock = BedrockConstants.zmin_bedrock

# Special values
spval = SpecialValues.spval
ispval = SpecialValues.ispval


# JAX-compatible versions of constants as arrays
def get_jax_constants() -> dict:
    """
    Get JAX-compatible versions of all constants
    
    Returns:
        Dictionary of constants as JAX arrays with appropriate dtypes
    """
    return {
        # Mathematical constants
        'rpi': jnp.array(rpi, dtype=r8),
        
        # Physical constants
        'tfrz': jnp.array(tfrz, dtype=r8),
        'sb': jnp.array(sb, dtype=r8),
        'grav': jnp.array(grav, dtype=r8),
        'vkc': jnp.array(vkc, dtype=r8),
        'denh2o': jnp.array(denh2o, dtype=r8),
        'denice': jnp.array(denice, dtype=r8),
        'tkwat': jnp.array(tkwat, dtype=r8),
        'tkice': jnp.array(tkice, dtype=r8),
        'tkair': jnp.array(tkair, dtype=r8),
        'hfus': jnp.array(hfus, dtype=r8),
        'hvap': jnp.array(hvap, dtype=r8),
        'hsub': jnp.array(hsub, dtype=r8),
        'cpice': jnp.array(cpice, dtype=r8),
        'cpliq': jnp.array(cpliq, dtype=r8),
        
        # Bedrock constants
        'thk_bedrock': jnp.array(thk_bedrock, dtype=r8),
        'csol_bedrock': jnp.array(csol_bedrock, dtype=r8),
        'zmin_bedrock': jnp.array(zmin_bedrock, dtype=r8),
        
        # Special values
        'spval': jnp.array(spval, dtype=r8),
        'ispval': jnp.array(ispval, dtype=jnp.int32),
    }


# Pre-computed JAX constants for performance
_jax_constants = get_jax_constants()


def get_constant(name: str, as_jax: bool = False):
    """
    Get a constant by name, optionally as a JAX array
    
    Args:
        name: Name of the constant
        as_jax: If True, return as JAX array; if False, return as Python scalar
        
    Returns:
        The requested constant value
    """
    if as_jax:
        return _jax_constants.get(name)
    else:
        return globals().get(name)


def is_special_value(value: float, tolerance: float = 1e-10) -> bool:
    """
    Check if a value is a special value (missing data indicator)
    
    Args:
        value: Value to check
        tolerance: Tolerance for comparison
        
    Returns:
        True if value is considered a special value
    """
    return abs(value - spval) < tolerance


def is_special_int_value(value: int) -> bool:
    """
    Check if an integer value is a special value (missing data indicator)
    
    Args:
        value: Integer value to check
        
    Returns:
        True if value is considered a special integer value
    """
    return value == ispval


# Utility functions for common calculations
@jax.jit
def celsius_to_kelvin(celsius: jnp.ndarray) -> jnp.ndarray:
    """Convert temperature from Celsius to Kelvin"""
    return celsius + tfrz


@jax.jit
def kelvin_to_celsius(kelvin: jnp.ndarray) -> jnp.ndarray:
    """Convert temperature from Kelvin to Celsius"""
    return kelvin - tfrz


@jax.jit
def stefan_boltzmann_flux(temperature: jnp.ndarray) -> jnp.ndarray:
    """Calculate Stefan-Boltzmann radiation flux"""
    return sb * jnp.power(temperature, 4)


# Public interface
__all__ = [
    # Mathematical constants
    'rpi',
    
    # Physical constants
    'tfrz', 'sb', 'grav', 'vkc', 'denh2o', 'denice',
    'tkwat', 'tkice', 'tkair', 'hfus', 'hvap', 'hsub',
    'cpice', 'cpliq',
    
    # Bedrock constants
    'thk_bedrock', 'csol_bedrock', 'zmin_bedrock',
    
    # Special values
    'spval', 'ispval',
    
    # Classes
    'PhysicalConstants', 'BedrockConstants', 'SpecialValues',
    
    # Utility functions
    'get_jax_constants', 'get_constant', 'is_special_value', 'is_special_int_value',
    'celsius_to_kelvin', 'kelvin_to_celsius', 'stefan_boltzmann_flux'
]