"""
Soil Texture Parameters Module.

Translated from CTSM's SoilTexMod.F90 (lines 1-48)

This module provides soil texture class parameters based on:
- Cosby et al. (1984). Water Resources Research 20:682-690 (texture fractions)
- Clapp and Hornberger (1978). Water Resources Research 14:601-604 (hydraulic parameters)

The module defines 11 standard soil texture classes with their physical and
hydraulic properties used throughout the soil physics calculations.

Key Parameters:
    - Texture fractions (sand, silt, clay) for each class
    - Hydraulic properties (porosity, matric potential, conductivity)
    - Clapp-Hornberger "b" parameter for water retention curves

Texture Classes:
    0: sand
    1: loamy sand
    2: sandy loam
    3: silty loam
    4: loam
    5: sandy clay loam
    6: silty clay loam
    7: clay loam
    8: sandy clay
    9: silty clay
    10: clay

Usage:
    from jax_ctsm.soil.texture import DEFAULT_SOIL_TEXTURE_PARAMS
    
    # Access parameters
    params = DEFAULT_SOIL_TEXTURE_PARAMS
    sand_fractions = params.sand_tex
    porosity = params.watsat_tex
    
    # Or create custom instance
    custom_params = create_soil_texture_params()

References:
    Cosby, B.J., et al. (1984). A statistical exploration of the relationships
        of soil moisture characteristics to the physical properties of soils.
        Water Resources Research, 20(6), 682-690.
    
    Clapp, R.B., & Hornberger, G.M. (1978). Empirical equations for some soil
        hydraulic properties. Water Resources Research, 14(4), 601-604.
"""

from typing import NamedTuple
import jax.numpy as jnp


# =============================================================================
# Type Definitions
# =============================================================================


class SoilTextureParams(NamedTuple):
    """Soil texture class parameters.
    
    All arrays have shape [ntex=11] corresponding to the 11 texture classes.
    
    Attributes:
        ntex: Number of soil texture classes (11)
        soil_tex: Soil texture class names [ntex]
        sand_tex: Sand fraction [dimensionless, 0-1] [ntex]
        silt_tex: Silt fraction [dimensionless, 0-1] [ntex]
        clay_tex: Clay fraction [dimensionless, 0-1] [ntex]
        watsat_tex: Volumetric soil water at saturation (porosity) [m3/m3] [ntex]
        smpsat_tex: Soil matric potential at saturation [mm] [ntex]
        hksat_tex: Hydraulic conductivity at saturation [mm H2O/min] [ntex]
        bsw_tex: Clapp and Hornberger "b" parameter [dimensionless] [ntex]
    
    Note:
        All texture fractions (sand, silt, clay) sum to 1.0 for each class.
        Negative values for smpsat_tex indicate tension (suction).
    
    Reference:
        Lines 14-46 of SoilTexMod.F90
    """
    ntex: int
    soil_tex: tuple
    sand_tex: jnp.ndarray
    silt_tex: jnp.ndarray
    clay_tex: jnp.ndarray
    watsat_tex: jnp.ndarray
    smpsat_tex: jnp.ndarray
    hksat_tex: jnp.ndarray
    bsw_tex: jnp.ndarray


# =============================================================================
# Parameter Creation Functions
# =============================================================================


def create_soil_texture_params() -> SoilTextureParams:
    """Create soil texture parameters from Cosby et al. (1984) and Clapp & Hornberger (1978).
    
    This function initializes all soil texture class parameters with empirically
    derived values from the literature. The parameters are used throughout the
    soil physics calculations for water movement, retention, and thermal properties.
    
    Returns:
        SoilTextureParams: Immutable container with all soil texture parameters
        
    Note:
        The hydraulic conductivity values (hksat_tex) are in mm H2O/min and may
        need conversion to mm/s (multiply by 1/60) for some calculations.
        
        The matric potential values (smpsat_tex) are negative, indicating
        tension/suction in the soil matrix.
    
    Reference:
        Lines 14-46 of SoilTexMod.F90
    """
    # Line 14: Number of soil texture classes
    ntex = 11
    
    # Lines 26-27: Soil texture class names
    # Order matches the array indices for all parameter arrays
    soil_tex = (
        'sand',              # 0
        'loamy sand',        # 1
        'sandy loam',        # 2
        'silty loam',        # 3
        'loam',              # 4
        'sandy clay loam',   # 5
        'silty clay loam',   # 6
        'clay loam',         # 7
        'sandy clay',        # 8
        'silty clay',        # 9
        'clay'               # 10
    )
    
    # Line 29: Sand fraction (Cosby et al. 1984)
    # Dimensionless fraction [0-1]
    sand_tex = jnp.array([
        0.92,  # sand
        0.82,  # loamy sand
        0.58,  # sandy loam
        0.17,  # silty loam
        0.43,  # loam
        0.58,  # sandy clay loam
        0.10,  # silty clay loam
        0.32,  # clay loam
        0.52,  # sandy clay
        0.06,  # silty clay
        0.22   # clay
    ], dtype=jnp.float64)
    
    # Line 30: Silt fraction (Cosby et al. 1984)
    # Dimensionless fraction [0-1]
    silt_tex = jnp.array([
        0.05,  # sand
        0.12,  # loamy sand
        0.32,  # sandy loam
        0.70,  # silty loam
        0.39,  # loam
        0.15,  # sandy clay loam
        0.56,  # silty clay loam
        0.34,  # clay loam
        0.06,  # sandy clay
        0.47,  # silty clay
        0.20   # clay
    ], dtype=jnp.float64)
    
    # Line 31: Clay fraction (Cosby et al. 1984)
    # Dimensionless fraction [0-1]
    # Note: sand + silt + clay = 1.0 for each texture class
    clay_tex = jnp.array([
        0.03,  # sand
        0.06,  # loamy sand
        0.10,  # sandy loam
        0.13,  # silty loam
        0.18,  # loam
        0.27,  # sandy clay loam
        0.34,  # silty clay loam
        0.34,  # clay loam
        0.42,  # sandy clay
        0.47,  # silty clay
        0.58   # clay
    ], dtype=jnp.float64)
    
    # Lines 35-36: Volumetric soil water at saturation (Clapp & Hornberger 1978)
    # Also known as porosity [m3/m3]
    # Represents the maximum water content when all pore space is filled
    watsat_tex = jnp.array([
        0.395,  # sand
        0.410,  # loamy sand
        0.435,  # sandy loam
        0.485,  # silty loam
        0.451,  # loam
        0.420,  # sandy clay loam
        0.477,  # silty clay loam
        0.476,  # clay loam
        0.426,  # sandy clay
        0.492,  # silty clay
        0.482   # clay
    ], dtype=jnp.float64)
    
    # Lines 38-39: Soil matric potential at saturation [mm] (Clapp & Hornberger 1978)
    # Negative values indicate tension (suction)
    # Used in water retention curve: psi = psi_sat * (theta/theta_sat)^(-b)
    smpsat_tex = jnp.array([
        -121.,  # sand
        -90.,   # loamy sand
        -218.,  # sandy loam
        -786.,  # silty loam
        -478.,  # loam
        -299.,  # sandy clay loam
        -356.,  # silty clay loam
        -630.,  # clay loam
        -153.,  # sandy clay
        -490.,  # silty clay
        -405.   # clay
    ], dtype=jnp.float64)
    
    # Lines 41-42: Hydraulic conductivity at saturation [mm H2O/min] (Clapp & Hornberger 1978)
    # Maximum rate of water movement through saturated soil
    # Used in unsaturated conductivity: K = K_sat * (theta/theta_sat)^(2b+3)
    # Note: Convert to mm/s by dividing by 60 if needed
    hksat_tex = jnp.array([
        10.560,  # sand - highest conductivity
        9.380,   # loamy sand
        2.080,   # sandy loam
        0.432,   # silty loam
        0.417,   # loam
        0.378,   # sandy clay loam
        0.102,   # silty clay loam
        0.147,   # clay loam
        0.130,   # sandy clay
        0.062,   # silty clay
        0.077    # clay - lowest conductivity
    ], dtype=jnp.float64)
    
    # Lines 44-45: Clapp and Hornberger "b" parameter (Clapp & Hornberger 1978)
    # Dimensionless exponent in water retention and conductivity curves
    # Higher values indicate steeper curves (more rapid change with moisture)
    # Used in: psi = psi_sat * (theta/theta_sat)^(-b)
    #          K = K_sat * (theta/theta_sat)^(2b+3)
    bsw_tex = jnp.array([
        4.05,   # sand - lowest b (gradual curves)
        4.38,   # loamy sand
        4.90,   # sandy loam
        5.30,   # silty loam
        5.39,   # loam
        7.12,   # sandy clay loam
        7.75,   # silty clay loam
        8.52,   # clay loam
        10.40,  # sandy clay
        10.40,  # silty clay
        11.40   # clay - highest b (steep curves)
    ], dtype=jnp.float64)
    
    return SoilTextureParams(
        ntex=ntex,
        soil_tex=soil_tex,
        sand_tex=sand_tex,
        silt_tex=silt_tex,
        clay_tex=clay_tex,
        watsat_tex=watsat_tex,
        smpsat_tex=smpsat_tex,
        hksat_tex=hksat_tex,
        bsw_tex=bsw_tex,
    )


# =============================================================================
# Utility Functions
# =============================================================================


def get_texture_class_index(texture_name: str) -> int:
    """Get the array index for a given texture class name.
    
    Args:
        texture_name: Name of texture class (e.g., 'sand', 'loam', 'clay')
        
    Returns:
        Index [0-10] corresponding to the texture class
        
    Raises:
        ValueError: If texture_name is not recognized
        
    Example:
        >>> idx = get_texture_class_index('loam')
        >>> params = DEFAULT_SOIL_TEXTURE_PARAMS
        >>> loam_porosity = params.watsat_tex[idx]
    """
    params = DEFAULT_SOIL_TEXTURE_PARAMS
    try:
        return params.soil_tex.index(texture_name.lower())
    except ValueError:
        valid_names = ', '.join(params.soil_tex)
        raise ValueError(
            f"Unknown texture class '{texture_name}'. "
            f"Valid options: {valid_names}"
        )


def interpolate_texture_params(
    sand_frac: jnp.ndarray,
    clay_frac: jnp.ndarray,
    params: SoilTextureParams,
) -> dict:
    """Interpolate soil parameters from sand and clay fractions.
    
    This function provides a simple linear interpolation of soil parameters
    based on sand and clay fractions. For more accurate results, consider
    using pedotransfer functions.
    
    Args:
        sand_frac: Sand fraction [0-1] [n_points]
        clay_frac: Clay fraction [0-1] [n_points]
        params: Soil texture parameters
        
    Returns:
        Dictionary with interpolated parameters:
            - watsat: Porosity [m3/m3]
            - smpsat: Matric potential at saturation [mm]
            - hksat: Hydraulic conductivity [mm/min]
            - bsw: Clapp-Hornberger b parameter
            
    Note:
        This is a simplified interpolation. The original CTSM may use
        more sophisticated pedotransfer functions for continuous texture.
        
    Reference:
        Conceptually based on texture-parameter relationships in SoilTexMod.F90
    """
    # Calculate silt fraction
    silt_frac = 1.0 - sand_frac - clay_frac
    
    # Simple weighted average based on texture fractions
    # This is a placeholder - actual CTSM may use different interpolation
    watsat = (
        sand_frac * params.watsat_tex[0] +  # sand
        silt_frac * params.watsat_tex[3] +  # silty loam (silt-dominated)
        clay_frac * params.watsat_tex[10]   # clay
    )
    
    smpsat = (
        sand_frac * params.smpsat_tex[0] +
        silt_frac * params.smpsat_tex[3] +
        clay_frac * params.smpsat_tex[10]
    )
    
    hksat = (
        sand_frac * params.hksat_tex[0] +
        silt_frac * params.hksat_tex[3] +
        clay_frac * params.hksat_tex[10]
    )
    
    bsw = (
        sand_frac * params.bsw_tex[0] +
        silt_frac * params.bsw_tex[3] +
        clay_frac * params.bsw_tex[10]
    )
    
    return {
        'watsat': watsat,
        'smpsat': smpsat,
        'hksat': hksat,
        'bsw': bsw,
    }


# =============================================================================
# Module-Level Constants
# =============================================================================


# Create default instance for module-level access
# This is the primary way to access soil texture parameters
DEFAULT_SOIL_TEXTURE_PARAMS = create_soil_texture_params()


# Convenience constants for common texture classes
SAND_INDEX = 0
LOAMY_SAND_INDEX = 1
SANDY_LOAM_INDEX = 2
SILTY_LOAM_INDEX = 3
LOAM_INDEX = 4
SANDY_CLAY_LOAM_INDEX = 5
SILTY_CLAY_LOAM_INDEX = 6
CLAY_LOAM_INDEX = 7
SANDY_CLAY_INDEX = 8
SILTY_CLAY_INDEX = 9
CLAY_INDEX = 10