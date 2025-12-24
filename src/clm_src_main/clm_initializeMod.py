"""
CLM initialization module

This module performs land model initialization in two phases.
Translated from Fortran CLM code to Python JAX.
"""

import jax
import jax.numpy as jnp
from typing import Any

# Import dependencies
try:
    from ..cime_src_share_util.shr_kind_mod import r8
    from .decompMod import BoundsType
    from .clm_varpar import clm_varpar_init
    from .pftconMod import pftcon
    from .GridcellType import grc
    from .ColumnType import col
    from .PatchType import patch
    from .initGridCellsMod import initGridCells
    from .filterMod import allocFilters, filter
    from .clm_instMod import clm_instInit
    from ..multilayer_canopy.MLCanopyTurbulenceMod import LookupPsihatINI
except ImportError:
    # Fallback for when running outside package context
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cime_src_share_util.shr_kind_mod import r8
    from clm_src_main.decompMod import BoundsType
    from clm_src_main.clm_varpar import clm_varpar_init
    from clm_src_main.pftconMod import pftcon
    from clm_src_main.GridcellType import grc
    from clm_src_main.ColumnType import col
    from clm_src_main.PatchType import patch
    from clm_src_main.initGridCellsMod import initGridCells
    from clm_src_main.filterMod import allocFilters, filter
    from clm_src_main.clm_instMod import clm_instInit
    from multilayer_canopy.MLCanopyTurbulenceMod import LookupPsihatINI

# Alias for backward compatibility
bounds_type = BoundsType


def initialize1(bounds: bounds_type) -> None:
    """
    CLM initialization - first phase
    
    This function performs the first phase of CLM initialization including:
    - Initializing run control variables
    - Reading PFT parameters
    - Initializing lookup tables
    - Allocating memory for subgrid data structures
    - Building subgrid hierarchy
    - Allocating filters
    
    Args:
        bounds: CLM bounds structure containing grid indices
    """
    
    # Initialize run control variables
    clm_varpar_init()
    
    # Read list of PFTs and their parameter values
    pftcon.Init()
    
    # Initialize the look-up tables needed to calculate the CLMml
    # roughness sublayer psihat functions
    LookupPsihatINI()
    
    # Allocate memory for subgrid data structures
    grc.Init(bounds.begg, bounds.endg)
    col.Init(bounds.begc, bounds.endc)
    patch.Init(bounds.begp, bounds.endp)
    
    # Build subgrid hierarchy of landunit, column, and patch
    initGridCells()
    
    # Allocate filters
    allocFilters(filter, bounds.begp, bounds.endp, bounds.begc, bounds.endc)


def initialize2(bounds: bounds_type) -> None:
    """
    CLM initialization - second phase
    
    This function performs the second phase of CLM initialization including:
    - Initializing instances of all derived types
    - Initializing time constant variables
    
    Args:
        bounds: CLM bounds structure containing grid indices
    """
    
    # Initialize instances of all derived types as well as
    # time constant variables
    clm_instInit(bounds)


# JIT-compiled versions for performance
initialize1_jit = jax.jit(initialize1, static_argnames=['bounds'])
initialize2_jit = jax.jit(initialize2, static_argnames=['bounds'])


def full_initialize(bounds: bounds_type) -> None:
    """
    Complete CLM initialization - both phases
    
    Convenience function that runs both initialization phases in sequence.
    
    Args:
        bounds: CLM bounds structure containing grid indices
    """
    initialize1(bounds)
    initialize2(bounds)


# JIT-compiled version of full initialization
full_initialize_jit = jax.jit(full_initialize, static_argnames=['bounds'])


def validate_initialization(bounds: bounds_type) -> bool:
    """
    Validate that initialization was successful
    
    This function performs basic validation checks to ensure that
    the initialization completed successfully.
    
    Args:
        bounds: CLM bounds structure containing grid indices
        
    Returns:
        True if initialization appears successful, False otherwise
    """
    try:
        # Check that bounds are valid
        if bounds.begg > bounds.endg or bounds.begc > bounds.endc or bounds.begp > bounds.endp:
            return False
            
        # Check that basic structures are initialized
        # Note: In a full implementation, you would check that grc, col, patch
        # and other structures have been properly initialized
        
        return True
        
    except Exception:
        return False


# Public interface
__all__ = [
    'initialize1', 'initialize2', 'full_initialize',
    'initialize1_jit', 'initialize2_jit', 'full_initialize_jit',
    'validate_initialization'
]