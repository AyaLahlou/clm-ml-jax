"""
Atmosphere to land variables type

This module defines the atm2lnd_type class for handling atmospheric forcing
data that gets passed to the land model. Translated from Fortran CLM code to Python JAX.
"""

import jax
import jax.numpy as jnp
from typing import Optional
from dataclasses import dataclass, field

# Import dependencies
try:
    from ..cime_src_share_util.shr_kind_mod import r8
    from .clm_varpar import numrad
    from .decompMod import BoundsType
except ImportError:
    # Fallback for when running outside package context
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cime_src_share_util.shr_kind_mod import r8
    from clm_src_main.clm_varpar import numrad
    from clm_src_main.decompMod import BoundsType

# Alias for backward compatibility
bounds_type = BoundsType


@dataclass
class atm2lnd_type:
    """
    Atmosphere to land variables type
    
    This class contains atmospheric forcing variables that are passed to the land model.
    Some variables are defined on the gridcell level (not downscaled) while others
    are downscaled to the column level.
    """
    
    # atm -> land: not downscaled (gridcell level)
    forc_u_grc: Optional[jnp.ndarray] = None                # Atmospheric wind speed in east direction (m/s)
    forc_v_grc: Optional[jnp.ndarray] = None                # Atmospheric wind speed in north direction (m/s)  
    forc_pco2_grc: Optional[jnp.ndarray] = None             # Atmospheric CO2 partial pressure (Pa)
    forc_po2_grc: Optional[jnp.ndarray] = None              # Atmospheric O2 partial pressure (Pa)
    forc_solad_grc: Optional[jnp.ndarray] = None            # Atmospheric direct beam radiation (W/m2)
    forc_solai_grc: Optional[jnp.ndarray] = None            # Atmospheric diffuse radiation (W/m2)
    
    # atm -> land: downscaled to column
    forc_t_downscaled_col: Optional[jnp.ndarray] = None     # Atmospheric temperature (K)
    forc_q_downscaled_col: Optional[jnp.ndarray] = None     # Atmospheric specific humidity (kg/kg)
    forc_pbot_downscaled_col: Optional[jnp.ndarray] = None  # Atmospheric pressure (Pa)
    forc_lwrad_downscaled_col: Optional[jnp.ndarray] = None # Atmospheric longwave radiation (W/m2)
    forc_rain_downscaled_col: Optional[jnp.ndarray] = None  # Rainfall rate (mm/s)
    forc_snow_downscaled_col: Optional[jnp.ndarray] = None  # Snowfall rate (mm/s)
    
    def Init(self, bounds: bounds_type) -> None:
        """
        Initialize the atm2lnd_type instance
        
        Args:
            bounds: CLM bounds structure containing grid indices
        """
        self.InitAllocate(bounds)
    
    def InitAllocate(self, bounds: bounds_type) -> None:
        """
        Initialize module data structure by allocating arrays
        
        Args:
            bounds: CLM bounds structure containing grid indices
        """
        # Initial value for all arrays
        ival = 0.0
        
        # Extract bounds for readability
        begg = bounds.begg
        endg = bounds.endg
        begc = bounds.begc  
        endc = bounds.endc
        
        # Calculate array sizes
        grid_size = endg - begg + 1
        col_size = endc - begc + 1
        
        # Allocate and initialize gridcell-level arrays
        self.forc_u_grc = jnp.full((grid_size,), ival, dtype=r8)
        self.forc_v_grc = jnp.full((grid_size,), ival, dtype=r8)
        self.forc_pco2_grc = jnp.full((grid_size,), ival, dtype=r8)
        self.forc_po2_grc = jnp.full((grid_size,), ival, dtype=r8)
        self.forc_solad_grc = jnp.full((grid_size, numrad), ival, dtype=r8)
        self.forc_solai_grc = jnp.full((grid_size, numrad), ival, dtype=r8)
        
        # Allocate and initialize column-level arrays
        self.forc_t_downscaled_col = jnp.full((col_size,), ival, dtype=r8)
        self.forc_q_downscaled_col = jnp.full((col_size,), ival, dtype=r8)
        self.forc_pbot_downscaled_col = jnp.full((col_size,), ival, dtype=r8)
        self.forc_lwrad_downscaled_col = jnp.full((col_size,), ival, dtype=r8)
        self.forc_rain_downscaled_col = jnp.full((col_size,), ival, dtype=r8)
        self.forc_snow_downscaled_col = jnp.full((col_size,), ival, dtype=r8)


# Factory function to create and initialize atm2lnd_type instance
def create_atm2lnd_instance(bounds: bounds_type) -> atm2lnd_type:
    """
    Factory function to create and initialize an atm2lnd_type instance
    
    Args:
        bounds: CLM bounds structure containing grid indices
        
    Returns:
        Initialized atm2lnd_type instance
    """
    instance = atm2lnd_type()
    instance.Init(bounds)
    return instance


# JIT-compiled version of the initialization for performance
@jax.jit
def init_atm2lnd_arrays(bounds_dict: dict) -> dict:
    """
    JIT-compiled function to initialize atm2lnd arrays
    
    Args:
        bounds_dict: Dictionary containing bounds information
        
    Returns:
        Dictionary of initialized arrays
    """
    ival = 0.0
    begg = bounds_dict['begg']
    endg = bounds_dict['endg'] 
    begc = bounds_dict['begc']
    endc = bounds_dict['endc']
    
    grid_size = endg - begg + 1
    col_size = endc - begc + 1
    
    return {
        'forc_u_grc': jnp.full((grid_size,), ival, dtype=r8),
        'forc_v_grc': jnp.full((grid_size,), ival, dtype=r8),
        'forc_pco2_grc': jnp.full((grid_size,), ival, dtype=r8),
        'forc_po2_grc': jnp.full((grid_size,), ival, dtype=r8),
        'forc_solad_grc': jnp.full((grid_size, numrad), ival, dtype=r8),
        'forc_solai_grc': jnp.full((grid_size, numrad), ival, dtype=r8),
        'forc_t_downscaled_col': jnp.full((col_size,), ival, dtype=r8),
        'forc_q_downscaled_col': jnp.full((col_size,), ival, dtype=r8),
        'forc_pbot_downscaled_col': jnp.full((col_size,), ival, dtype=r8),
        'forc_lwrad_downscaled_col': jnp.full((col_size,), ival, dtype=r8),
        'forc_rain_downscaled_col': jnp.full((col_size,), ival, dtype=r8),
        'forc_snow_downscaled_col': jnp.full((col_size,), ival, dtype=r8),
    }


# Public interface
__all__ = ['atm2lnd_type', 'create_atm2lnd_instance', 'init_atm2lnd_arrays']