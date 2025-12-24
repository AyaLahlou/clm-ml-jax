"""
Column data type module

This module defines the column_type class for handling column-level
data structures in CLM including snow layers, soil layers, and bedrock.
Translated from Fortran CLM code to Python JAX.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple
from dataclasses import dataclass

# Import dependencies
try:
    from ..cime_src_share_util.shr_kind_mod import r8
    from .clm_varpar import nlevgrnd, nlevsno
    from .clm_varcon import ispval, spval as nan
except ImportError:
    # Fallback for when running outside package context
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cime_src_share_util.shr_kind_mod import r8
    from clm_src_main.clm_varpar import nlevgrnd, nlevsno
    from clm_src_main.clm_varcon import ispval, spval as nan


@dataclass
class column_type:
    """
    Column data type for CLM
    
    This class contains column-level variables including snow layer information,
    soil layer properties, and bedrock characteristics.
    """
    
    # Column arrays
    snl: Optional[jnp.ndarray] = None          # Number of snow layers [begc:endc]
    dz: Optional[jnp.ndarray] = None           # Soil layer thickness (m) [begc:endc, -nlevsno+1:nlevgrnd]
    z: Optional[jnp.ndarray] = None            # Soil layer depth (m) [begc:endc, -nlevsno+1:nlevgrnd]
    zi: Optional[jnp.ndarray] = None           # Soil layer depth at layer interface (m) [begc:endc, -nlevsno:nlevgrnd]
    nbedrock: Optional[jnp.ndarray] = None     # Variable depth to bedrock index [begc:endc]
    
    # Store bounds for reference
    begc: Optional[int] = None
    endc: Optional[int] = None
    
    def Init(self, begc: int, endc: int) -> None:
        """
        Initialize column data structure
        
        Args:
            begc: Beginning column index
            endc: Ending column index
        """
        self.begc = begc
        self.endc = endc
        
        # Calculate array sizes
        col_size = endc - begc + 1
        
        # Soil/snow layer dimensions
        # Note: Fortran arrays are 1-based and inclusive on both ends
        # Python arrays are 0-based, so we adjust indexing
        layer_size_dz_z = nlevgrnd + nlevsno  # -nlevsno+1 to nlevgrnd inclusive
        layer_size_zi = nlevgrnd + nlevsno + 1  # -nlevsno to nlevgrnd inclusive
        
        # Initialize arrays with appropriate initial values
        self.snl = jnp.full((col_size,), ispval, dtype=jnp.int32)
        self.dz = jnp.full((col_size, layer_size_dz_z), nan, dtype=r8)
        self.z = jnp.full((col_size, layer_size_dz_z), nan, dtype=r8)
        self.zi = jnp.full((col_size, layer_size_zi), nan, dtype=r8)
        self.nbedrock = jnp.full((col_size,), ispval, dtype=jnp.int32)
    
    def is_initialized(self) -> bool:
        """Check if the column type has been initialized"""
        return all([
            self.snl is not None,
            self.dz is not None,
            self.z is not None,
            self.zi is not None,
            self.nbedrock is not None,
            self.begc is not None,
            self.endc is not None
        ])
    
    def get_column_count(self) -> int:
        """Get the number of columns"""
        if self.begc is not None and self.endc is not None:
            return self.endc - self.begc + 1
        return 0
    
    def get_layer_info(self) -> dict:
        """Get information about layer dimensions"""
        if not self.is_initialized():
            return {}
        
        return {
            'num_columns': self.get_column_count(),
            'dz_z_layers': self.dz.shape[1],  # -nlevsno+1 to nlevgrnd
            'zi_layers': self.zi.shape[1],    # -nlevsno to nlevgrnd
            'snow_layers_max': nlevsno,
            'ground_layers_total': nlevgrnd
        }
    
    def validate_arrays(self) -> bool:
        """Validate array dimensions and consistency"""
        if not self.is_initialized():
            return False
        
        try:
            col_size = self.get_column_count()
            
            # Check array shapes
            if (self.snl.shape != (col_size,) or
                self.nbedrock.shape != (col_size,) or
                self.dz.shape[0] != col_size or
                self.z.shape[0] != col_size or
                self.zi.shape[0] != col_size):
                return False
            
            # Check layer dimensions
            expected_dz_z_layers = nlevgrnd + nlevsno
            expected_zi_layers = nlevgrnd + nlevsno + 1
            
            if (self.dz.shape[1] != expected_dz_z_layers or
                self.z.shape[1] != expected_dz_z_layers or
                self.zi.shape[1] != expected_zi_layers):
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_fortran_indices(self, col_idx: int, layer_idx: int, array_type: str = 'dz') -> Tuple[int, int]:
        """
        Convert Python indices to equivalent Fortran indices
        
        Args:
            col_idx: Python column index (0-based)
            layer_idx: Python layer index (0-based)
            array_type: Type of array ('dz', 'z', or 'zi')
            
        Returns:
            Tuple of (fortran_col_idx, fortran_layer_idx)
        """
        fortran_col_idx = self.begc + col_idx
        
        if array_type in ['dz', 'z']:
            # These arrays go from -nlevsno+1 to nlevgrnd
            fortran_layer_idx = -nlevsno + 1 + layer_idx
        elif array_type == 'zi':
            # This array goes from -nlevsno to nlevgrnd  
            fortran_layer_idx = -nlevsno + layer_idx
        else:
            raise ValueError(f"Unknown array type: {array_type}")
        
        return fortran_col_idx, fortran_layer_idx


# Global column instance (equivalent to Fortran module-level variable)
col = column_type()


def create_column_instance(begc: int, endc: int) -> column_type:
    """
    Factory function to create and initialize a column_type instance
    
    Args:
        begc: Beginning column index
        endc: Ending column index
        
    Returns:
        Initialized column_type instance
    """
    instance = column_type()
    instance.Init(begc, endc)
    return instance


def reset_global_column() -> None:
    """Reset the global column instance"""
    global col
    col = column_type()


def get_snow_layer_range() -> Tuple[int, int]:
    """
    Get the range of snow layer indices (Fortran-style)
    
    Returns:
        Tuple of (start_index, end_index) for snow layers
    """
    return (-nlevsno + 1, 0)


def get_soil_layer_range() -> Tuple[int, int]:
    """
    Get the range of soil layer indices (Fortran-style)
    
    Returns:
        Tuple of (start_index, end_index) for soil layers
    """
    return (1, nlevgrnd)


def get_all_layer_range(include_interface: bool = False) -> Tuple[int, int]:
    """
    Get the range of all layer indices (Fortran-style)
    
    Args:
        include_interface: If True, include interface layers (for zi array)
        
    Returns:
        Tuple of (start_index, end_index) for all layers
    """
    if include_interface:
        return (-nlevsno, nlevgrnd)  # For zi array
    else:
        return (-nlevsno + 1, nlevgrnd)  # For dz, z arrays


# JAX utility functions for layer operations
@jax.jit
def get_active_snow_layers(snl_array: jnp.ndarray) -> jnp.ndarray:
    """
    Get mask for columns with active snow layers
    
    Args:
        snl_array: Array of snow layer counts
        
    Returns:
        Boolean mask for columns with snow
    """
    return snl_array > 0


@jax.jit
def get_soil_layer_mask(layer_indices: jnp.ndarray) -> jnp.ndarray:
    """
    Get mask for soil layers (positive indices)
    
    Args:
        layer_indices: Array of layer indices
        
    Returns:
        Boolean mask for soil layers
    """
    return layer_indices > 0


@jax.jit
def get_snow_layer_mask(layer_indices: jnp.ndarray) -> jnp.ndarray:
    """
    Get mask for snow layers (negative indices)
    
    Args:
        layer_indices: Array of layer indices
        
    Returns:
        Boolean mask for snow layers
    """
    return layer_indices < 0


def create_layer_index_arrays() -> dict:
    """
    Create JAX arrays for layer indexing
    
    Returns:
        Dictionary of layer index arrays
    """
    snow_start, soil_end = get_all_layer_range()
    zi_start, zi_end = get_all_layer_range(include_interface=True)
    
    return {
        'dz_z_indices': jnp.arange(snow_start, soil_end + 1, dtype=jnp.int32),
        'zi_indices': jnp.arange(zi_start, zi_end + 1, dtype=jnp.int32),
        'snow_indices': jnp.arange(-nlevsno + 1, 1, dtype=jnp.int32),
        'soil_indices': jnp.arange(1, nlevgrnd + 1, dtype=jnp.int32)
    }


# Public interface
__all__ = [
    'column_type', 'col', 'create_column_instance', 'reset_global_column',
    'get_snow_layer_range', 'get_soil_layer_range', 'get_all_layer_range',
    'get_active_snow_layers', 'get_soil_layer_mask', 'get_snow_layer_mask',
    'create_layer_index_arrays'
]