"""
CLM initialization module

This module performs land model initialization in two phases.
Translated from Fortran CLM code to Python JAX.
"""

import jax
import jax.numpy as jnp
from typing import Any
from pathlib import Path
import os

# Enable 64-bit floats in JAX (required for CLM precision)
jax.config.update("jax_enable_x64", True)

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

# Default path for RSL lookup table file
def _get_default_rsl_path() -> str:
    """Get default path to RSL lookup table file."""
    # Try to find the rsl_lookup_tables directory
    module_dir = Path(__file__).parent.parent
    
    # Check common locations
    possible_paths = [
        module_dir.parent / "CLM-ml_v1" / "rsl_lookup_tables" / "psihat.nc",
        module_dir.parent / "rsl_lookup_tables" / "psihat.nc",
        Path.cwd() / "CLM-ml_v1" / "rsl_lookup_tables" / "psihat.nc",
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    # If file not found, return a placeholder path
    # The initialize_rsl_tables function will handle missing files gracefully
    return str(module_dir.parent / "CLM-ml_v1" / "rsl_lookup_tables" / "psihat.nc")


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
    rsl_path = _get_default_rsl_path()
    LookupPsihatINI(rsl_path)
    
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


# Note: JIT compilation is not compatible with these initialization functions
# because they modify global state. The _jit versions are provided as aliases
# for API compatibility but do NOT actually use JIT compilation.
initialize1_jit = initialize1
initialize2_jit = initialize2


def full_initialize(bounds: bounds_type) -> None:
    """
    Complete CLM initialization - both phases
    
    Convenience function that runs both initialization phases in sequence.
    
    Args:
        bounds: CLM bounds structure containing grid indices
    """
    initialize1(bounds)
    initialize2(bounds)


# JIT version is an alias (JIT not compatible with side effects)
full_initialize_jit = full_initialize


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