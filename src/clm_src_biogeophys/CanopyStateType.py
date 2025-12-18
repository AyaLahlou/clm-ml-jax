"""
Canopy State Type Module.

Translated from CTSM's CanopyStateType.F90

This module defines the canopy state variables including vegetation fraction,
leaf area index (LAI), stem area index (SAI), and canopy height. These variables
describe the physical structure of the vegetation canopy at the patch level.

Key variables:
    - frac_veg_nosno: Fraction of vegetation not covered by snow (0 or 1)
    - elai: Effective leaf area index (one-sided, with snow burial)
    - esai: Effective stem area index (one-sided, with snow burial)
    - htop: Canopy top height [m]

The canopy state is fundamental to many land surface processes including:
    - Radiation interception and albedo
    - Turbulent exchange with atmosphere
    - Precipitation interception
    - Snow burial effects on vegetation

Physics Notes:
    - LAI and SAI are "effective" values that account for snow burial
    - frac_veg_nosno is binary: 1 when vegetation is exposed, 0 when snow-covered
    - htop determines aerodynamic roughness and displacement height
    - All variables are at patch level (subgrid vegetation type)

Translation Notes:
    - Fortran derived type becomes immutable NamedTuple
    - Pointer arrays become JAX arrays with explicit patch dimension
    - Initialization uses NaN to detect uninitialized values
    - All arrays are float32 for GPU efficiency
"""

from typing import NamedTuple
import jax.numpy as jnp


# =============================================================================
# Type Definitions
# =============================================================================


class CanopyState(NamedTuple):
    """Canopy state variables at patch level.
    
    Translated from CanopyStateType.F90 (lines 1-35).
    
    This immutable structure holds the physical state of the vegetation canopy.
    All variables are defined at the patch level, where each patch represents
    a specific plant functional type (PFT) within a grid cell.
    
    Attributes:
        frac_veg_nosno_patch: Fraction of vegetation not covered by snow.
            Binary values: 0.0 (snow covered) or 1.0 (exposed). This controls
            whether vegetation processes (photosynthesis, transpiration) are
            active. [-] [n_patches]
            
        elai_patch: Effective one-sided leaf area index with snow burial.
            Total leaf area per unit ground area, accounting for snow burial.
            Used in radiation transfer, photosynthesis, and turbulent exchange.
            [m2 leaf/m2 ground] [n_patches]
            
        esai_patch: Effective one-sided stem area index with snow burial.
            Total stem/branch area per unit ground area, accounting for snow
            burial. Important for winter radiation and turbulent exchange.
            [m2 stem/m2 ground] [n_patches]
            
        htop_patch: Canopy top height above ground.
            Maximum height of vegetation canopy, used to calculate aerodynamic
            roughness length and displacement height for turbulent exchange.
            [m] [n_patches]
    
    Note:
        All arrays have shape [n_patches] where n_patches is the number of
        patch-level grid cells in the domain. In the original Fortran,
        frac_veg_nosno is integer, but JAX arrays use float with values
        restricted to {0.0, 1.0}.
        
        Arrays are initialized to NaN to detect uninitialized usage, following
        the Fortran pattern with spval (special value).
    """
    frac_veg_nosno_patch: jnp.ndarray  # [n_patches], values in {0.0, 1.0}
    elai_patch: jnp.ndarray            # [n_patches], m2/m2, >= 0
    esai_patch: jnp.ndarray            # [n_patches], m2/m2, >= 0
    htop_patch: jnp.ndarray            # [n_patches], m, >= 0


class BoundsType(NamedTuple):
    """Domain bounds for array indexing.
    
    Defines the range of valid indices for different subgrid levels.
    Follows Fortran convention of inclusive bounds.
    
    Attributes:
        begp: Beginning patch index (inclusive)
        endp: Ending patch index (inclusive)
        begc: Beginning column index (inclusive)
        endc: Ending column index (inclusive)
        begg: Beginning gridcell index (inclusive)
        endg: Ending gridcell index (inclusive)
    """
    begp: int
    endp: int
    begc: int
    endc: int
    begg: int
    endg: int


# =============================================================================
# Initialization Functions
# =============================================================================


def init_allocate_canopy_state(
    bounds: BoundsType,
) -> CanopyState:
    """Initialize canopy state arrays with NaN values.
    
    Allocates and initializes patch-level canopy state variables to NaN.
    This follows the Fortran pattern of using special values (spval) to
    detect uninitialized array elements during debugging.
    
    Translated from CanopyStateType.F90, lines 46-66 (InitAllocate subroutine).
    
    Args:
        bounds: Domain bounds containing patch/column/gridcell indices.
            Uses begp and endp to determine patch array size.
        
    Returns:
        CanopyState with all arrays initialized to NaN. Arrays have shape
        [n_patches] where n_patches = bounds.endp - bounds.begp + 1.
        
    Note:
        Fortran uses inclusive indexing [begp:endp], so array size is
        endp - begp + 1. All arrays are float32 for GPU efficiency.
        
        Original Fortran code:
            allocate(this%frac_veg_nosno_patch(begp:endp))
            this%frac_veg_nosno_patch(begp:endp) = nan
            
    Example:
        >>> bounds = BoundsType(begp=1, endp=100, begc=1, endc=50, begg=1, endg=10)
        >>> state = init_allocate_canopy_state(bounds)
        >>> state.elai_patch.shape
        (100,)
        >>> jnp.all(jnp.isnan(state.elai_patch))
        True
    """
    # Calculate number of patches (Fortran inclusive indexing)
    # Lines 46-66: allocate arrays from begp to endp
    n_patches = bounds.endp - bounds.begp + 1
    
    # Initialize all arrays to NaN (lines 60-63)
    # Original: this%frac_veg_nosno_patch(begp:endp) = nan
    frac_veg_nosno_patch = jnp.full(n_patches, jnp.nan, dtype=jnp.float32)
    
    # Original: this%elai_patch(begp:endp) = nan
    elai_patch = jnp.full(n_patches, jnp.nan, dtype=jnp.float32)
    
    # Original: this%esai_patch(begp:endp) = nan
    esai_patch = jnp.full(n_patches, jnp.nan, dtype=jnp.float32)
    
    # Original: this%htop_patch(begp:endp) = nan
    htop_patch = jnp.full(n_patches, jnp.nan, dtype=jnp.float32)
    
    return CanopyState(
        frac_veg_nosno_patch=frac_veg_nosno_patch,
        elai_patch=elai_patch,
        esai_patch=esai_patch,
        htop_patch=htop_patch,
    )


def init_canopy_state(
    bounds: BoundsType,
) -> CanopyState:
    """Initialize canopy state variables.
    
    Main initialization routine that sets up all canopy state arrays by
    calling the allocation routine. This is the primary entry point for
    creating a new CanopyState instance.
    
    Translated from CanopyStateType.F90, lines 36-43 (Init subroutine).
    
    In the original Fortran, this is a class method that calls InitAllocate
    to set up the derived type members. In JAX, we return an immutable
    NamedTuple with pre-allocated arrays.
    
    Args:
        bounds: Domain bounds containing patch/column/gridcell indices.
            Passed through to init_allocate_canopy_state.
        
    Returns:
        Initialized CanopyState with all arrays allocated and set to NaN.
        
    Note:
        Original Fortran structure:
            subroutine Init(this, bounds)
                class(canopystate_type) :: this
                type(bounds_type), intent(in) :: bounds
                call this%InitAllocate(bounds)
            end subroutine Init
            
        In JAX, we maintain the two-level structure (Init calls InitAllocate)
        for consistency with the Fortran code, even though they could be
        combined.
        
    Example:
        >>> bounds = BoundsType(begp=1, endp=100, begc=1, endc=50, begg=1, endg=10)
        >>> state = init_canopy_state(bounds)
        >>> state.elai_patch.shape
        (100,)
    """
    # Call the allocation routine (lines 36-43)
    # Original: call this%InitAllocate(bounds)
    return init_allocate_canopy_state(bounds)


def init_canopy_state_zeros(n_patches: int) -> CanopyState:
    """Initialize canopy state with zero values.
    
    Alternative initialization that creates a CanopyState with all arrays
    set to zeros instead of NaN. Useful for testing or when starting from
    a known state.
    
    This is a convenience function not present in the original Fortran,
    but useful for JAX workflows where we may want deterministic initial
    values.
    
    Args:
        n_patches: Number of patch-level grid cells.
        
    Returns:
        Initialized CanopyState with zero values for all arrays.
        
    Note:
        Unlike init_allocate_canopy_state, this does not use bounds and
        directly specifies the array size. All arrays are float32.
        
    Example:
        >>> state = init_canopy_state_zeros(100)
        >>> state.elai_patch.shape
        (100,)
        >>> jnp.all(state.elai_patch == 0.0)
        True
    """
    return CanopyState(
        frac_veg_nosno_patch=jnp.zeros(n_patches, dtype=jnp.float32),
        elai_patch=jnp.zeros(n_patches, dtype=jnp.float32),
        esai_patch=jnp.zeros(n_patches, dtype=jnp.float32),
        htop_patch=jnp.zeros(n_patches, dtype=jnp.float32),
    )


# =============================================================================
# Utility Functions
# =============================================================================


def validate_canopy_state(state: CanopyState) -> bool:
    """Validate canopy state values are physically reasonable.
    
    Checks that all canopy state variables are within valid physical ranges:
        - frac_veg_nosno in {0.0, 1.0}
        - elai >= 0
        - esai >= 0
        - htop >= 0
    
    Args:
        state: CanopyState to validate
        
    Returns:
        True if all values are valid, False otherwise
        
    Note:
        This is a utility function for debugging, not present in original
        Fortran. In production code, consider using assertions or removing
        for performance.
        
    Example:
        >>> state = init_canopy_state_zeros(10)
        >>> validate_canopy_state(state)
        True
    """
    # Check frac_veg_nosno is binary (0 or 1)
    frac_valid = jnp.all(
        (state.frac_veg_nosno_patch == 0.0) | 
        (state.frac_veg_nosno_patch == 1.0) |
        jnp.isnan(state.frac_veg_nosno_patch)
    )
    
    # Check LAI/SAI/height are non-negative (or NaN for uninitialized)
    elai_valid = jnp.all((state.elai_patch >= 0.0) | jnp.isnan(state.elai_patch))
    esai_valid = jnp.all((state.esai_patch >= 0.0) | jnp.isnan(state.esai_patch))
    htop_valid = jnp.all((state.htop_patch >= 0.0) | jnp.isnan(state.htop_patch))
    
    return bool(frac_valid & elai_valid & esai_valid & htop_valid)


def get_total_lai(state: CanopyState) -> jnp.ndarray:
    """Get total leaf + stem area index.
    
    Calculates the total plant area index (LAI + SAI) for each patch.
    This is commonly used in radiation transfer and turbulent exchange
    calculations.
    
    Args:
        state: CanopyState containing LAI and SAI
        
    Returns:
        Total plant area index [m2/m2] [n_patches]
        
    Note:
        Returns NaN for patches where either LAI or SAI is NaN.
        
    Example:
        >>> state = CanopyState(
        ...     frac_veg_nosno_patch=jnp.ones(10),
        ...     elai_patch=jnp.full(10, 3.0),
        ...     esai_patch=jnp.full(10, 1.0),
        ...     htop_patch=jnp.full(10, 20.0)
        ... )
        >>> get_total_lai(state)
        Array([4., 4., 4., 4., 4., 4., 4., 4., 4., 4.], dtype=float32)
    """
    return state.elai_patch + state.esai_patch