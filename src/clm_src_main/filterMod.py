"""
Filter module for CLM

This module provides filters used for processing CLM columns and patches.
Filters allow selective processing of subsets of the computational domain.
Translated from Fortran CLM code to Python JAX.
"""

import jax
import jax.numpy as jnp
from typing import Optional, List, Union
from dataclasses import dataclass

# Import dependencies
try:
    from ..cime_src_share_util.shr_kind_mod import r8
    from .clm_varcon import ispval, spval as nan
except ImportError:
    # Fallback for when running outside package context
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cime_src_share_util.shr_kind_mod import r8
    from clm_src_main.clm_varcon import ispval, spval as nan


@dataclass
class clumpfilter:
    """
    CLM filter data structure
    
    This class contains various filters for processing subsets of
    CLM patches and columns based on different criteria.
    """
    
    # Exposed vegetation patch filter
    num_exposedvegp: int = 0                    # number of patches in exposedvegp filter
    exposedvegp: Optional[jnp.ndarray] = None   # patches where frac_veg_nosno is non-zero
    
    # Non-lake, non-urban patch filter  
    num_nolakeurbanp: int = 0                   # number of patches in non-lake, non-urban filter
    nolakeurbanp: Optional[jnp.ndarray] = None  # non-lake, non-urban filter (patches)
    
    # Non-lake column filter
    num_nolakec: int = 0                        # number of columns in non-lake filter
    nolakec: Optional[jnp.ndarray] = None       # non-lake filter (columns)
    
    # Non-urban column filter
    num_nourbanc: int = 0                       # number of columns in non-urban filter
    nourbanc: Optional[jnp.ndarray] = None      # non-urban filter (columns)
    
    # Hydrology column filter
    num_hydrologyc: int = 0                     # number of columns in hydrology filter
    hydrologyc: Optional[jnp.ndarray] = None    # hydrology filter (columns)
    
    # Store bounds for reference
    begp: Optional[int] = None
    endp: Optional[int] = None
    begc: Optional[int] = None
    endc: Optional[int] = None
    
    def is_allocated(self) -> bool:
        """Check if filter arrays have been allocated"""
        return all([
            self.exposedvegp is not None,
            self.nolakeurbanp is not None,
            self.nolakec is not None,
            self.nourbanc is not None,
            self.hydrologyc is not None
        ])
    
    def get_patch_count(self) -> int:
        """Get total number of patches"""
        if self.begp is not None and self.endp is not None:
            return self.endp - self.begp + 1
        return 0
    
    def get_column_count(self) -> int:
        """Get total number of columns"""
        if self.begc is not None and self.endc is not None:
            return self.endc - self.begc + 1
        return 0
    
    def get_filter_info(self) -> dict:
        """Get information about all filters"""
        return {
            'total_patches': self.get_patch_count(),
            'total_columns': self.get_column_count(),
            'num_exposedvegp': self.num_exposedvegp,
            'num_nolakeurbanp': self.num_nolakeurbanp,
            'num_nolakec': self.num_nolakec,
            'num_nourbanc': self.num_nourbanc,
            'num_hydrologyc': self.num_hydrologyc
        }
    
    def validate_filters(self) -> bool:
        """Validate filter consistency and bounds"""
        if not self.is_allocated():
            return False
        
        try:
            # Check that filter counts don't exceed array sizes
            if (self.num_exposedvegp > len(self.exposedvegp) or
                self.num_nolakeurbanp > len(self.nolakeurbanp) or
                self.num_nolakec > len(self.nolakec) or
                self.num_nourbanc > len(self.nourbanc) or
                self.num_hydrologyc > len(self.hydrologyc)):
                return False
            
            # Check that indices are within bounds
            patch_max = self.get_patch_count()
            column_max = self.get_column_count()
            
            if patch_max > 0:
                exposed_indices = self.exposedvegp[:self.num_exposedvegp]
                nolakeurban_indices = self.nolakeurbanp[:self.num_nolakeurbanp]
                
                if (jnp.any(exposed_indices < 1) or jnp.any(exposed_indices > patch_max) or
                    jnp.any(nolakeurban_indices < 1) or jnp.any(nolakeurban_indices > patch_max)):
                    return False
            
            if column_max > 0:
                nolake_indices = self.nolakec[:self.num_nolakec]
                nourban_indices = self.nourbanc[:self.num_nourbanc]
                hydrology_indices = self.hydrologyc[:self.num_hydrologyc]
                
                if (jnp.any(nolake_indices < 1) or jnp.any(nolake_indices > column_max) or
                    jnp.any(nourban_indices < 1) or jnp.any(nourban_indices > column_max) or
                    jnp.any(hydrology_indices < 1) or jnp.any(hydrology_indices > column_max)):
                    return False
            
            return True
            
        except Exception:
            return False


# Global filter instance (equivalent to Fortran module-level variable)
filter = clumpfilter()


def allocFilters(filter_inst: clumpfilter, begp: int, endp: int, begc: int, endc: int) -> None:
    """
    Initialize filter data structure by allocating arrays
    
    Args:
        filter_inst: Filter instance to initialize
        begp: Beginning patch index
        endp: Ending patch index
        begc: Beginning column index
        endc: Ending column index
    """
    # Store bounds
    filter_inst.begp = begp
    filter_inst.endp = endp
    filter_inst.begc = begc
    filter_inst.endc = endc
    
    # Calculate array sizes
    patch_size = endp - begp + 1
    column_size = endc - begc + 1
    
    # Allocate arrays for patch filters
    filter_inst.exposedvegp = jnp.zeros(patch_size, dtype=jnp.int32)
    filter_inst.nolakeurbanp = jnp.zeros(patch_size, dtype=jnp.int32)
    
    # Allocate arrays for column filters
    filter_inst.nolakec = jnp.zeros(column_size, dtype=jnp.int32)
    filter_inst.nourbanc = jnp.zeros(column_size, dtype=jnp.int32)
    filter_inst.hydrologyc = jnp.zeros(column_size, dtype=jnp.int32)
    
    # Initialize counts to 0
    filter_inst.num_exposedvegp = 0
    filter_inst.num_nolakeurbanp = 0
    filter_inst.num_nolakec = 0
    filter_inst.num_nourbanc = 0
    filter_inst.num_hydrologyc = 0


def setFilters(filter_inst: clumpfilter) -> None:
    """
    Set CLM filters to default values
    
    In this simplified implementation, all filters contain a single element
    with index 1, matching the original Fortran behavior.
    
    Args:
        filter_inst: Filter instance to set
    """
    if not filter_inst.is_allocated():
        raise RuntimeError("Filter arrays must be allocated before setting filters")
    
    # Set simple filters (all contain single element with index 1)
    filter_inst.num_nolakeurbanp = 1
    filter_inst.nolakeurbanp = filter_inst.nolakeurbanp.at[0].set(1)
    
    filter_inst.num_nolakec = 1
    filter_inst.nolakec = filter_inst.nolakec.at[0].set(1)
    
    filter_inst.num_nourbanc = 1
    filter_inst.nourbanc = filter_inst.nourbanc.at[0].set(1)
    
    filter_inst.num_hydrologyc = 1
    filter_inst.hydrologyc = filter_inst.hydrologyc.at[0].set(1)


def setExposedvegpFilter(filter_inst: clumpfilter, frac_veg_nosno: jnp.ndarray) -> None:
    """
    Set the exposedvegp patch filter
    
    This filter includes patches for which frac_veg_nosno > 0.
    It does not include urban or lake points.
    
    Args:
        filter_inst: Filter instance to update
        frac_veg_nosno: Fraction of vegetation not covered by snow for each patch
    """
    if not filter_inst.is_allocated():
        raise RuntimeError("Filter arrays must be allocated before setting exposedvegp filter")
    
    # Use vectorized approach for better performance
    fe = 0
    exposed_patches = []
    
    # Iterate through non-lake, non-urban patches
    for fp in range(filter_inst.num_nolakeurbanp):
        p = filter_inst.nolakeurbanp[fp] - 1  # Convert to 0-based indexing
        if p < len(frac_veg_nosno) and frac_veg_nosno[p] > 0:
            exposed_patches.append(filter_inst.nolakeurbanp[fp])  # Keep 1-based for consistency
            fe += 1
    
    # Update the filter
    filter_inst.num_exposedvegp = fe
    if fe > 0:
        exposed_array = jnp.array(exposed_patches, dtype=jnp.int32)
        filter_inst.exposedvegp = filter_inst.exposedvegp.at[:fe].set(exposed_array)


@jax.jit
def setExposedvegpFilter_jax(nolakeurban_indices: jnp.ndarray, 
                            frac_veg_nosno: jnp.ndarray,
                            max_patches: int) -> tuple:
    """
    JAX-compiled version of exposed vegetation filter setting
    
    Args:
        nolakeurban_indices: Indices of non-lake, non-urban patches
        frac_veg_nosno: Fraction of vegetation not covered by snow
        max_patches: Maximum number of patches
        
    Returns:
        Tuple of (exposed_indices, num_exposed)
    """
    # Create mask for patches with exposed vegetation
    valid_mask = (nolakeurban_indices > 0) & (nolakeurban_indices <= len(frac_veg_nosno))
    adjusted_indices = jnp.where(valid_mask, nolakeurban_indices - 1, 0)  # Convert to 0-based
    veg_mask = jnp.where(valid_mask, frac_veg_nosno[adjusted_indices] > 0, False)
    
    # Get exposed patch indices
    exposed_indices = jnp.where(veg_mask, nolakeurban_indices, 0)
    
    # Count non-zero entries
    num_exposed = jnp.sum(exposed_indices > 0)
    
    # Create properly sized output array
    output_indices = jnp.zeros(max_patches, dtype=jnp.int32)
    
    # Compact non-zero indices to the beginning of the array
    valid_positions = jnp.cumsum(exposed_indices > 0) - 1
    output_indices = output_indices.at[valid_positions].set(
        jnp.where(exposed_indices > 0, exposed_indices, 0)
    )
    
    return output_indices, num_exposed


def create_filter_instance(begp: int, endp: int, begc: int, endc: int) -> clumpfilter:
    """
    Factory function to create and initialize a filter instance
    
    Args:
        begp: Beginning patch index
        endp: Ending patch index
        begc: Beginning column index
        endc: Ending column index
        
    Returns:
        Initialized clumpfilter instance
    """
    filter_inst = clumpfilter()
    allocFilters(filter_inst, begp, endp, begc, endc)
    setFilters(filter_inst)
    return filter_inst


def reset_global_filter() -> None:
    """Reset the global filter instance"""
    global filter
    filter = clumpfilter()


def get_filter_indices(filter_inst: clumpfilter, filter_name: str) -> tuple:
    """
    Get indices and count for a specific filter
    
    Args:
        filter_inst: Filter instance
        filter_name: Name of the filter ('exposedvegp', 'nolakeurbanp', etc.)
        
    Returns:
        Tuple of (indices_array, count)
    """
    filter_map = {
        'exposedvegp': (filter_inst.exposedvegp, filter_inst.num_exposedvegp),
        'nolakeurbanp': (filter_inst.nolakeurbanp, filter_inst.num_nolakeurbanp),
        'nolakec': (filter_inst.nolakec, filter_inst.num_nolakec),
        'nourbanc': (filter_inst.nourbanc, filter_inst.num_nourbanc),
        'hydrologyc': (filter_inst.hydrologyc, filter_inst.num_hydrologyc)
    }
    
    if filter_name not in filter_map:
        raise ValueError(f"Unknown filter name: {filter_name}")
    
    indices, count = filter_map[filter_name]
    return indices[:count], count


def apply_patch_filter(data: jnp.ndarray, filter_inst: clumpfilter, filter_name: str) -> jnp.ndarray:
    """
    Apply a patch filter to data array
    
    Args:
        data: Data array to filter (patch-indexed)
        filter_inst: Filter instance
        filter_name: Name of the filter to apply
        
    Returns:
        Filtered data array
    """
    indices, count = get_filter_indices(filter_inst, filter_name)
    if count == 0:
        return jnp.array([])
    
    # Convert to 0-based indexing for array access
    zero_based_indices = indices - 1
    return data[zero_based_indices]


def apply_column_filter(data: jnp.ndarray, filter_inst: clumpfilter, filter_name: str) -> jnp.ndarray:
    """
    Apply a column filter to data array
    
    Args:
        data: Data array to filter (column-indexed)
        filter_inst: Filter instance
        filter_name: Name of the filter to apply
        
    Returns:
        Filtered data array
    """
    indices, count = get_filter_indices(filter_inst, filter_name)
    if count == 0:
        return jnp.array([])
    
    # Convert to 0-based indexing for array access
    zero_based_indices = indices - 1
    return data[zero_based_indices]


# Public interface
__all__ = [
    'clumpfilter', 'filter', 'allocFilters', 'setFilters', 'setExposedvegpFilter',
    'setExposedvegpFilter_jax', 'create_filter_instance', 'reset_global_filter',
    'get_filter_indices', 'apply_patch_filter', 'apply_column_filter'
]