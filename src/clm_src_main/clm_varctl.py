"""
JAX translation of clm_varctl module.

This module contains run control variables for the Community Land Model (CLM).
Provides configuration parameters and control flags for model execution.

Translated from: clm_varctl.F90, lines 1-15

Key Features:
    - Run control variables stored in immutable NamedTuple
    - Pure functional interface for JAX compatibility
    - Default configuration factory function
    - Module-level default instance for convenience

Note:
    In the original Fortran, this module uses I/O unit numbers for logging.
    In JAX translation, these are preserved for compatibility but actual I/O
    operations are handled through Python's logging system or other mechanisms.
"""

from typing import NamedTuple
import jax.numpy as jnp
from jax import Array

# =============================================================================
# Type Aliases
# =============================================================================

# Type alias for double precision (Fortran r8 from shr_kind_mod)
r8 = jnp.float64


# =============================================================================
# Data Structures
# =============================================================================

class ClmVarCtl(NamedTuple):
    """
    Run control variables for CLM.
    
    This immutable structure holds configuration parameters that control
    the execution of CLM simulations. In the original Fortran module,
    these were module-level variables.
    
    Attributes:
        iulog: "stdout" log file unit number (default: 6)
               In Fortran, this represents the logical unit number for
               standard output logging. In JAX translation, this is kept
               for compatibility but actual I/O operations are handled
               through Python's logging mechanisms.
    
    Reference:
        clm_varctl.F90, lines 1-15
        
    Example:
        >>> ctl = create_clm_varctl()
        >>> print(ctl.iulog)
        6
        >>> # Create custom configuration
        >>> custom_ctl = ClmVarCtl(iulog=10)
    """
    iulog: int = 6


# =============================================================================
# Factory Functions
# =============================================================================

def create_clm_varctl() -> ClmVarCtl:
    """
    Create default ClmVarCtl configuration.
    
    Factory function that returns a ClmVarCtl instance with default values.
    This provides a clean interface for initializing the control variables
    and allows for future extension with additional parameters.
    
    Returns:
        ClmVarCtl: Default run control variables with iulog=6
        
    Reference:
        clm_varctl.F90, line 13
        
    Example:
        >>> ctl = create_clm_varctl()
        >>> assert ctl.iulog == 6
    """
    return ClmVarCtl(iulog=6)


# =============================================================================
# Module-Level Constants
# =============================================================================

# Default instance for convenience - can be used throughout the codebase
# without needing to call create_clm_varctl() repeatedly
DEFAULT_CLM_VARCTL = create_clm_varctl()


# =============================================================================
# Utility Functions (for future extension)
# =============================================================================

def update_clm_varctl(ctl: ClmVarCtl, **kwargs) -> ClmVarCtl:
    """
    Update ClmVarCtl with new values.
    
    Since ClmVarCtl is immutable (NamedTuple), this function creates a new
    instance with updated values. This is the functional programming pattern
    for "modifying" immutable data structures.
    
    Args:
        ctl: Existing ClmVarCtl instance
        **kwargs: Keyword arguments matching ClmVarCtl field names
        
    Returns:
        ClmVarCtl: New instance with updated values
        
    Example:
        >>> ctl = create_clm_varctl()
        >>> new_ctl = update_clm_varctl(ctl, iulog=10)
        >>> assert new_ctl.iulog == 10
        >>> assert ctl.iulog == 6  # Original unchanged
    """
    return ctl._replace(**kwargs)


def validate_clm_varctl(ctl: ClmVarCtl) -> bool:
    """
    Validate ClmVarCtl configuration.
    
    Checks that all control variables have valid values. Currently validates
    that iulog is a positive integer.
    
    Args:
        ctl: ClmVarCtl instance to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
        
    Example:
        >>> ctl = create_clm_varctl()
        >>> assert validate_clm_varctl(ctl)
        >>> invalid_ctl = ClmVarCtl(iulog=-1)
        >>> assert not validate_clm_varctl(invalid_ctl)
    """
    # Validate iulog is positive
    if ctl.iulog <= 0:
        return False
    
    return True