"""
JAX translation of clm_varorb module.

This module contains orbital parameters used in CTSM calculations for Earth's
orbital mechanics. These parameters are used to compute solar radiation and
seasonal variations.

Translated from clm_varorb.F90, lines 1-21.

The module provides:
- OrbitalParams: NamedTuple containing orbital parameters
- create_orbital_params: Factory function for initialization
- update_orbital_params: Immutable update function

Reference:
    Fortran source: clm_varorb.F90
"""

from typing import NamedTuple
import jax.numpy as jnp
from jax import Array


# ============================================================================
# Type Definitions
# ============================================================================

class OrbitalParams(NamedTuple):
    """
    Orbital parameters for Earth's orbit calculations.
    
    These parameters define Earth's orbital characteristics used in solar
    radiation calculations and seasonal cycle computations.
    
    Attributes:
        eccen: Orbital eccentricity factor (dimensionless).
               Input to orbit_parms. Typical value ~0.0167.
               Reference: Line 13
        obliqr: Earth's obliquity in radians.
                The tilt of Earth's axis relative to orbital plane.
                Output from orbit_params. Typical value ~0.409 rad (23.44°).
                Reference: Line 17
        lambm0: Mean longitude of perihelion at the vernal equinox (radians).
                Defines the position of perihelion relative to vernal equinox.
                Output from orbit_params.
                Reference: Line 18
        mvelpp: Earth's moving vernal equinox longitude of perihelion 
                plus pi (radians).
                Used in solar declination calculations.
                Output from orbit_params.
                Reference: Line 19
    
    Note:
        All angular quantities are in radians for computational efficiency.
        These parameters vary on Milankovitch timescales (10^4-10^5 years).
    
    Reference:
        Fortran source: clm_varorb.F90, lines 1-21
    """
    eccen: Array   # float64 scalar
    obliqr: Array  # float64 scalar
    lambm0: Array  # float64 scalar
    mvelpp: Array  # float64 scalar


# ============================================================================
# Factory and Update Functions
# ============================================================================

def create_orbital_params(
    eccen: float = 0.0,
    obliqr: float = 0.0,
    lambm0: float = 0.0,
    mvelpp: float = 0.0
) -> OrbitalParams:
    """
    Create an OrbitalParams instance with default or specified values.
    
    This factory function initializes orbital parameters, converting Python
    floats to JAX arrays with float64 precision for consistency with the
    original Fortran code.
    
    Args:
        eccen: Orbital eccentricity factor (default: 0.0).
               Valid range: [0.0, ~0.06] for Earth.
        obliqr: Earth's obliquity in radians (default: 0.0).
                Valid range: [~0.37, ~0.44] rad (~21-25°).
        lambm0: Mean longitude of perihelion at vernal equinox in radians 
                (default: 0.0).
                Valid range: [0, 2π].
        mvelpp: Moving vernal equinox longitude of perihelion plus pi 
                in radians (default: 0.0).
                Valid range: [0, 2π].
    
    Returns:
        OrbitalParams instance with the specified values as float64 JAX arrays.
    
    Example:
        >>> params = create_orbital_params(eccen=0.0167, obliqr=0.4091)
        >>> print(params.eccen)
        Array(0.0167, dtype=float64)
    
    Reference:
        Fortran source: clm_varorb.F90, lines 1-21
    """
    return OrbitalParams(
        eccen=jnp.asarray(eccen, dtype=jnp.float64),
        obliqr=jnp.asarray(obliqr, dtype=jnp.float64),
        lambm0=jnp.asarray(lambm0, dtype=jnp.float64),
        mvelpp=jnp.asarray(mvelpp, dtype=jnp.float64)
    )


def update_orbital_params(
    params: OrbitalParams,
    eccen: float | None = None,
    obliqr: float | None = None,
    lambm0: float | None = None,
    mvelpp: float | None = None
) -> OrbitalParams:
    """
    Update orbital parameters with new values (immutable update).
    
    Creates a new OrbitalParams instance with updated values while preserving
    immutability. Only specified parameters are updated; others retain their
    original values.
    
    Args:
        params: Current OrbitalParams instance to update.
        eccen: New orbital eccentricity factor (optional).
               If None, retains current value.
        obliqr: New Earth's obliquity in radians (optional).
                If None, retains current value.
        lambm0: New mean longitude of perihelion at vernal equinox (optional).
                If None, retains current value.
        mvelpp: New moving vernal equinox longitude of perihelion plus pi 
                (optional).
                If None, retains current value.
    
    Returns:
        New OrbitalParams instance with updated values.
    
    Example:
        >>> params = create_orbital_params(eccen=0.0167)
        >>> updated = update_orbital_params(params, obliqr=0.4091)
        >>> print(updated.eccen, updated.obliqr)
        Array(0.0167, dtype=float64) Array(0.4091, dtype=float64)
    
    Note:
        This function follows JAX's functional programming paradigm by
        returning a new instance rather than modifying the input.
    
    Reference:
        Fortran source: clm_varorb.F90, lines 1-21
    """
    return OrbitalParams(
        eccen=jnp.asarray(eccen, dtype=jnp.float64) if eccen is not None else params.eccen,
        obliqr=jnp.asarray(obliqr, dtype=jnp.float64) if obliqr is not None else params.obliqr,
        lambm0=jnp.asarray(lambm0, dtype=jnp.float64) if lambm0 is not None else params.lambm0,
        mvelpp=jnp.asarray(mvelpp, dtype=jnp.float64) if mvelpp is not None else params.mvelpp
    )