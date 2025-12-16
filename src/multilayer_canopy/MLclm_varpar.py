"""
Multilayer Canopy Model Parameters.

Translated from CTSM's MLclm_varpar.F90

This module defines fundamental parameters for the multilayer canopy model,
which represents vertical structure within the plant canopy. The multilayer
approach resolves vertical gradients in:
    - Radiation absorption (sunlit vs shaded leaves)
    - Turbulent transport (momentum, heat, water vapor)
    - Leaf temperature and photosynthesis
    - Stomatal conductance

The canopy is discretized into vertical layers, with separate treatment of
sunlit and shaded leaf fractions in each layer. This allows accurate
representation of:
    - Light extinction through the canopy
    - Within-canopy wind profiles
    - Vertical gradients in leaf physiology
    - Canopy-atmosphere coupling

Key Design:
    - 100 vertical layers provide fine resolution of canopy profiles
    - Sunlit/shaded distinction captures light environment effects
    - Parameters are immutable for JIT compilation
    - Used to dimension arrays throughout multilayer canopy physics

Reference:
    MLclm_varpar.F90, lines 1-21
    Bonan et al. (2018) Agricultural and Forest Meteorology
"""

from typing import NamedTuple
import jax.numpy as jnp


class MLCanopyParams(NamedTuple):
    """Parameters for multilayer canopy model.
    
    These parameters define the vertical discretization and leaf type
    classification used throughout the multilayer canopy physics.
    
    Attributes:
        nlevmlcan: Number of vertical layers in canopy model [dimensionless]
            - 100 layers provides fine resolution of vertical profiles
            - Layers are typically equally spaced in cumulative LAI
            - Used to dimension radiation, turbulence, and flux arrays
            
        nleaf: Number of leaf types [dimensionless]
            - Always 2: sunlit and shaded leaves
            - Sunlit leaves receive direct beam radiation
            - Shaded leaves receive only diffuse radiation
            - Critical for accurate photosynthesis and conductance
            
        isun: Array index for sunlit leaves [dimensionless]
            - Index 1 in leaf-type dimension
            - Used to access sunlit leaf properties
            - Convention: sunlit = 1, shaded = 2
            
        isha: Array index for shaded leaves [dimensionless]
            - Index 2 in leaf-type dimension
            - Used to access shaded leaf properties
            - Shaded leaves typically have lower photosynthesis
        
    Physical Context:
        The sunlit/shaded distinction is fundamental to canopy radiative
        transfer. At any height in the canopy:
            - Sunlit fraction = exp(-K_beam * LAI_above)
            - Shaded fraction = 1 - sunlit fraction
        where K_beam is the beam extinction coefficient.
        
        Sunlit leaves receive both direct beam and diffuse radiation,
        while shaded leaves receive only diffuse. This creates large
        differences in:
            - Absorbed PAR (photosynthetically active radiation)
            - Leaf temperature
            - Photosynthetic rate
            - Stomatal conductance
            
    Usage:
        params = get_mlcanopy_params()
        n_layers = params.nlevmlcan  # 100
        n_leaf_types = params.nleaf  # 2
        
        # Dimension arrays
        radiation = jnp.zeros((n_patches, n_layers, n_leaf_types))
        
        # Index into arrays
        rad_sunlit = radiation[:, :, params.isun - 1]  # -1 for 0-based indexing
        rad_shaded = radiation[:, :, params.isha - 1]
        
    Reference:
        MLclm_varpar.F90, lines 14-18
    """
    nlevmlcan: int = 100  # Line 14: Number of layers in multilayer canopy
    nleaf: int = 2        # Line 15: Number of leaf types (sunlit and shaded)
    isun: int = 1         # Line 16: Sunlit leaf index (Fortran 1-based)
    isha: int = 2         # Line 17: Shaded leaf index (Fortran 1-based)


# Default instance for global use
DEFAULT_MLCANOPY_PARAMS = MLCanopyParams()


def get_mlcanopy_params() -> MLCanopyParams:
    """Get default multilayer canopy parameters.
    
    Returns immutable parameter set for multilayer canopy model.
    This function provides a consistent interface for accessing
    parameters across the codebase.
    
    Returns:
        MLCanopyParams: Default parameter values
            - nlevmlcan = 100 (vertical layers)
            - nleaf = 2 (sunlit and shaded)
            - isun = 1 (sunlit index)
            - isha = 2 (shaded index)
            
    Example:
        >>> params = get_mlcanopy_params()
        >>> print(f"Canopy has {params.nlevmlcan} layers")
        Canopy has 100 layers
        >>> print(f"Sunlit index: {params.isun}, Shaded index: {params.isha}")
        Sunlit index: 1, Shaded index: 2
        
    Note:
        Indices are 1-based (Fortran convention). Subtract 1 when
        indexing into JAX arrays:
            sunlit_data = array[:, params.isun - 1]
            shaded_data = array[:, params.isha - 1]
        
    Reference:
        MLclm_varpar.F90, lines 14-18
    """
    return DEFAULT_MLCANOPY_PARAMS


def validate_mlcanopy_params(params: MLCanopyParams) -> bool:
    """Validate multilayer canopy parameters.
    
    Checks that parameters have physically reasonable values:
        - nlevmlcan > 0 (need at least one layer)
        - nleaf == 2 (sunlit and shaded only)
        - isun == 1 and isha == 2 (standard indexing)
    
    Args:
        params: MLCanopyParams to validate
        
    Returns:
        True if parameters are valid, False otherwise
        
    Example:
        >>> params = get_mlcanopy_params()
        >>> assert validate_mlcanopy_params(params)
        >>> 
        >>> bad_params = MLCanopyParams(nlevmlcan=0, nleaf=2, isun=1, isha=2)
        >>> assert not validate_mlcanopy_params(bad_params)
    """
    if params.nlevmlcan <= 0:
        return False
    if params.nleaf != 2:
        return False
    if params.isun != 1 or params.isha != 2:
        return False
    return True


# Convenience constants for direct access
# (Use get_mlcanopy_params() for full parameter set)
NLEVMLCAN = DEFAULT_MLCANOPY_PARAMS.nlevmlcan  # 100 layers
NLEAF = DEFAULT_MLCANOPY_PARAMS.nleaf          # 2 leaf types
ISUN = DEFAULT_MLCANOPY_PARAMS.isun            # Sunlit index = 1
ISHA = DEFAULT_MLCANOPY_PARAMS.isha            # Shaded index = 2