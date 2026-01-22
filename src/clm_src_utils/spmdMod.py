"""
SPMD (Single Program Multiple Data) Module.

Translated from CTSM's spmdMod.F90 (lines 1-16)

This module provides SPMD initialization and configuration for parallel processing.
In the JAX translation, we maintain the masterproc flag as a constant for 
compatibility with code that checks for master process status, though JAX's
execution model differs from traditional MPI parallelism.

Original Fortran module provided:
- masterproc: Logical flag indicating master process (proc 0) for printing messages

In JAX context:
- We preserve the masterproc constant for code compatibility
- JAX handles parallelism through its own mechanisms (pmap, sharding, etc.)
- This module serves primarily as a compatibility layer
- For multi-device execution, use jax.process_index() == 0 to identify master

Key differences from Fortran MPI model:
1. JAX uses functional parallelism (pmap, pjit) rather than SPMD message passing
2. No explicit MPI initialization or finalization needed
3. Process identification handled through JAX's distributed runtime
4. This module maintains API compatibility for code that checks masterproc

Usage:
    from jax_ctsm.utils.spmd_mod import MASTERPROC, is_master_proc
    
    if is_master_proc():
        print("Message from master process")
"""

from typing import Final
import jax


# =============================================================================
# Constants
# =============================================================================

# Line 14: logical, parameter :: masterproc = .true.
# Master process flag for printing messages
# In single-process JAX execution, this is always True
MASTERPROC: Final[bool] = True


# =============================================================================
# Functions
# =============================================================================

def is_master_proc() -> bool:
    """Check if current process is the master process.
    
    In the original SPMD model, this would check if the current MPI rank is 0.
    In JAX, we maintain this for compatibility but always return True in
    single-process execution.
    
    For multi-device/multi-host JAX execution:
    - Use jax.process_index() to get the current process ID
    - Master is typically process 0
    - This function can be extended to support distributed execution
    
    Returns:
        True if master process (always True in JAX single-process mode)
        
    Example:
        >>> if is_master_proc():
        ...     print("Initialization message")
        
    Note:
        For multi-device JAX execution using pmap, additional logic would be
        needed to identify the "master" device. Example implementation:
        
        >>> # Multi-device version:
        >>> # return jax.process_index() == 0
    """
    return MASTERPROC