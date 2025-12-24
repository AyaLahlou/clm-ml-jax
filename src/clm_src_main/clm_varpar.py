"""
CLM variable parameters module

This module contains various model parameters used throughout CLM including
layer definitions, radiation parameters, and plant functional type settings.
Translated from Fortran CLM code to Python JAX.
"""

import jax
import jax.numpy as jnp
from typing import Final, Optional, Dict, Any
from dataclasses import dataclass

# Import dependencies
try:
    from ..cime_src_share_util.shr_kind_mod import r8
    from ..multilayer_canopy.MLclm_varctl import MLCanopyConfig, create_default_config
except ImportError:
    # Fallback for when running outside package context
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cime_src_share_util.shr_kind_mod import r8
    from multilayer_canopy.MLclm_varctl import MLCanopyConfig, create_default_config

# Get default config for clm_phys value
_default_ml_config = create_default_config()
clm_phys = _default_ml_config.clm_phys


@dataclass
class CLMParameters:
    """
    Container for CLM model parameters
    
    This class holds all the model parameters and provides methods
    for initialization and validation.
    """
    
    # Layer parameters (initialized to -1, set during initialization)
    nlevsno: int = -1      # Maximum number of snow layers
    nlevsoi: int = -1      # Number of hydrologically active soil layers
    nlevgrnd: int = -1     # Number of ground layers (including hydrologically inactive)
    
    # Radiation parameters (constants)
    numrad: Final[int] = 2     # Number of radiation wavebands
    ivis: Final[int] = 1       # Visible waveband index
    inir: Final[int] = 2       # Near-infrared waveband index
    
    # Plant functional type parameters
    mxpft: Final[int] = 78     # Maximum number of plant functional types
    
    def is_initialized(self) -> bool:
        """Check if parameters have been initialized"""
        return (self.nlevsno > 0 and 
                self.nlevsoi > 0 and 
                self.nlevgrnd > 0)
    
    def validate(self) -> bool:
        """Validate parameter consistency"""
        if not self.is_initialized():
            return False
        
        # Check that ground layers >= soil layers
        if self.nlevgrnd < self.nlevsoi:
            return False
            
        # Check reasonable ranges
        if (self.nlevsno < 1 or self.nlevsno > 50 or
            self.nlevsoi < 1 or self.nlevsoi > 100 or
            self.nlevgrnd < 1 or self.nlevgrnd > 100):
            return False
            
        return True
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get parameters as dictionary"""
        return {
            'nlevsno': self.nlevsno,
            'nlevsoi': self.nlevsoi,
            'nlevgrnd': self.nlevgrnd,
            'numrad': self.numrad,
            'ivis': self.ivis,
            'inir': self.inir,
            'mxpft': self.mxpft
        }


# Global parameter instance
_clm_params = CLMParameters()

# Module-level variables for backward compatibility with Fortran
nlevsno: int = _clm_params.nlevsno
nlevsoi: int = _clm_params.nlevsoi
nlevgrnd: int = _clm_params.nlevgrnd
numrad: Final[int] = _clm_params.numrad
ivis: Final[int] = _clm_params.ivis  
inir: Final[int] = _clm_params.inir
mxpft: Final[int] = _clm_params.mxpft


def clm_varpar_init() -> None:
    """
    Initialize module variables
    
    Sets the number of snow, soil, and ground layers based on the
    CLM physics version (CLM4.5 or CLM5.0).
    """
    global _clm_params, nlevsno, nlevsoi, nlevgrnd
    
    # CLM4.5 and CLM5 have different snow/soil layers
    if clm_phys == 'CLM5_0':
        _clm_params.nlevsno = 12
        _clm_params.nlevsoi = 20
        _clm_params.nlevgrnd = _clm_params.nlevsoi + 5  # 25
    elif clm_phys == 'CLM4_5':
        _clm_params.nlevsno = 5
        _clm_params.nlevsoi = 10
        _clm_params.nlevgrnd = 15
    else:
        # Default to CLM4.5 values if physics version not recognized
        _clm_params.nlevsno = 5
        _clm_params.nlevsoi = 10
        _clm_params.nlevgrnd = 15
    
    # Update module-level variables
    nlevsno = _clm_params.nlevsno
    nlevsoi = _clm_params.nlevsoi
    nlevgrnd = _clm_params.nlevgrnd
    
    # Validate parameters
    if not _clm_params.validate():
        raise ValueError(f"Invalid CLM parameters after initialization: {_clm_params.get_config_dict()}")


def get_clm_parameters() -> CLMParameters:
    """
    Get the current CLM parameters
    
    Returns:
        Current CLM parameters instance
    """
    return _clm_params


def set_custom_parameters(nlevsno_val: int, nlevsoi_val: int, nlevgrnd_val: int) -> None:
    """
    Set custom parameter values (for testing or special configurations)
    
    Args:
        nlevsno_val: Number of snow layers
        nlevsoi_val: Number of soil layers  
        nlevgrnd_val: Number of ground layers
    """
    global _clm_params, nlevsno, nlevsoi, nlevgrnd
    
    _clm_params.nlevsno = nlevsno_val
    _clm_params.nlevsoi = nlevsoi_val
    _clm_params.nlevgrnd = nlevgrnd_val
    
    # Update module-level variables
    nlevsno = _clm_params.nlevsno
    nlevsoi = _clm_params.nlevsoi
    nlevgrnd = _clm_params.nlevgrnd
    
    # Validate parameters
    if not _clm_params.validate():
        raise ValueError(f"Invalid custom CLM parameters: {_clm_params.get_config_dict()}")


def get_layer_info() -> Dict[str, int]:
    """
    Get information about layer configuration
    
    Returns:
        Dictionary with layer information
    """
    return {
        'max_snow_layers': nlevsno,
        'active_soil_layers': nlevsoi,
        'total_ground_layers': nlevgrnd,
        'inactive_soil_layers': nlevgrnd - nlevsoi,
        'total_subsurface_layers': nlevsno + nlevgrnd
    }


def get_radiation_info() -> Dict[str, int]:
    """
    Get information about radiation configuration
    
    Returns:
        Dictionary with radiation band information
    """
    return {
        'total_bands': numrad,
        'visible_band_index': ivis,
        'near_infrared_band_index': inir
    }


def get_pft_info() -> Dict[str, int]:
    """
    Get information about plant functional types
    
    Returns:
        Dictionary with PFT information
    """
    return {
        'max_pft_types': mxpft
    }


def create_layer_arrays() -> Dict[str, jnp.ndarray]:
    """
    Create JAX arrays for layer indexing
    
    Returns:
        Dictionary of JAX arrays for different layer types
    """
    if not _clm_params.is_initialized():
        raise RuntimeError("CLM parameters must be initialized before creating layer arrays")
    
    return {
        'snow_layers': jnp.arange(-nlevsno + 1, 1, dtype=jnp.int32),  # Snow layers (negative indices)
        'soil_layers': jnp.arange(1, nlevsoi + 1, dtype=jnp.int32),   # Active soil layers
        'ground_layers': jnp.arange(1, nlevgrnd + 1, dtype=jnp.int32), # All ground layers
        'radiation_bands': jnp.arange(1, numrad + 1, dtype=jnp.int32), # Radiation bands
        'pft_indices': jnp.arange(0, mxpft, dtype=jnp.int32)           # PFT indices
    }


def reset_parameters() -> None:
    """Reset parameters to uninitialized state"""
    global _clm_params, nlevsno, nlevsoi, nlevgrnd
    
    _clm_params = CLMParameters()
    nlevsno = _clm_params.nlevsno
    nlevsoi = _clm_params.nlevsoi  
    nlevgrnd = _clm_params.nlevgrnd


# Configuration presets for different CLM versions
CLM_PRESETS = {
    'CLM5_0': {
        'nlevsno': 12,
        'nlevsoi': 20,
        'nlevgrnd': 25
    },
    'CLM4_5': {
        'nlevsno': 5,
        'nlevsoi': 10,
        'nlevgrnd': 15
    }
}


def load_preset(preset_name: str) -> None:
    """
    Load a predefined parameter preset
    
    Args:
        preset_name: Name of the preset ('CLM4_5' or 'CLM5_0')
    """
    if preset_name not in CLM_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(CLM_PRESETS.keys())}")
    
    preset = CLM_PRESETS[preset_name]
    set_custom_parameters(preset['nlevsno'], preset['nlevsoi'], preset['nlevgrnd'])


# JIT-compiled utility functions
@jax.jit
def snow_layer_indices(nlevsno_val: int) -> jnp.ndarray:
    """Get snow layer indices as JAX array"""
    return jnp.arange(-nlevsno_val + 1, 1, dtype=jnp.int32)


@jax.jit  
def soil_layer_indices(nlevsoi_val: int) -> jnp.ndarray:
    """Get soil layer indices as JAX array"""
    return jnp.arange(1, nlevsoi_val + 1, dtype=jnp.int32)


@jax.jit
def ground_layer_indices(nlevgrnd_val: int) -> jnp.ndarray:
    """Get ground layer indices as JAX array"""
    return jnp.arange(1, nlevgrnd_val + 1, dtype=jnp.int32)


# Public interface
__all__ = [
    # Module variables (Fortran compatibility)
    'nlevsno', 'nlevsoi', 'nlevgrnd', 'numrad', 'ivis', 'inir', 'mxpft',
    
    # Classes and functions
    'CLMParameters', 'clm_varpar_init', 'get_clm_parameters', 'set_custom_parameters',
    'get_layer_info', 'get_radiation_info', 'get_pft_info', 'create_layer_arrays',
    'reset_parameters', 'load_preset', 'CLM_PRESETS',
    
    # JAX utilities
    'snow_layer_indices', 'soil_layer_indices', 'ground_layer_indices'
]