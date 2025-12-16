"""
Temperature State Variables and Initialization.

Translated from CTSM's TemperatureType.F90

This module defines the temperature state variables used throughout CTSM,
including soil/snow temperatures and atmospheric reference temperatures.

Key variables:
    - t_soisno: Soil and snow layer temperatures [K]
    - t_a10: 10-day running mean 2m air temperature [K]
    - t_ref2m: 2m reference air temperature [K]

The temperature state is stored in an immutable NamedTuple for JAX compatibility.
All initialization functions are pure and return new state objects.

Fortran source: TemperatureType.F90 (lines 1-68)
"""

from typing import NamedTuple
import jax.numpy as jnp


# =============================================================================
# Type Definitions
# =============================================================================


class BoundsType(NamedTuple):
    """Domain decomposition bounds.
    
    Attributes:
        begp: Beginning patch index
        endp: Ending patch index
        begc: Beginning column index
        endc: Ending column index
        begg: Beginning grid cell index
        endg: Ending grid cell index
    """
    begp: int
    endp: int
    begc: int
    endc: int
    begg: int
    endg: int


class TemperatureState(NamedTuple):
    """Temperature state variables for CTSM.
    
    Fortran source: TemperatureType.F90, lines 20-24
    
    This corresponds to the Fortran type temperature_type, containing
    temperature fields at column and patch levels.
    
    Attributes:
        t_soisno_col: Soil and snow layer temperatures [K]
                      Shape: [n_columns, n_levtot] where n_levtot = nlevsno + nlevgrnd
                      Indexing: layer 0 is top snow layer (if present), 
                               layer nlevsno is top soil layer
        t_a10_patch: 10-day running mean of 2m air temperature [K]
                     Shape: [n_patches]
                     Used for phenology and acclimation calculations
        t_ref2m_patch: 2m height surface air temperature [K]
                       Shape: [n_patches]
                       Diagnostic output and forcing for biogeochemistry
    
    Note:
        In Fortran, t_soisno_col is indexed as (-nlevsno+1:nlevgrnd).
        In JAX, we use 0-based indexing with shape (nlevsno + nlevgrnd).
        Layer mapping:
            Fortran index -nlevsno+1 -> JAX index 0 (top snow)
            Fortran index 0 -> JAX index nlevsno-1 (bottom snow/top soil interface)
            Fortran index 1 -> JAX index nlevsno (top soil)
            Fortran index nlevgrnd -> JAX index nlevsno+nlevgrnd-1 (bottom soil)
    """
    
    t_soisno_col: jnp.ndarray  # [n_columns, n_levtot] in K
    t_a10_patch: jnp.ndarray   # [n_patches] in K
    t_ref2m_patch: jnp.ndarray # [n_patches] in K


# =============================================================================
# Constants
# =============================================================================

# Default initialization temperature [K]
DEFAULT_INIT_TEMP = 273.15  # 0°C

# Special value for uninitialized data
NAN_VALUE = jnp.nan


# =============================================================================
# Initialization Functions
# =============================================================================


def init_temperature_state(
    n_columns: int,
    n_patches: int,
    n_levtot: int,
    initial_temp: float = DEFAULT_INIT_TEMP,
) -> TemperatureState:
    """Initialize temperature state variables.
    
    Fortran source: TemperatureType.F90, lines 26-27 (Init, InitAllocate procedures)
    
    Creates initial temperature state with all temperatures set to a
    reference value (default is 273.15 K = 0°C).
    
    Args:
        n_columns: Number of columns in domain
        n_patches: Number of patches in domain
        n_levtot: Total number of soil + snow layers (nlevsno + nlevgrnd)
        initial_temp: Initial temperature for all fields [K]
                     Default is 273.15 K (0°C)
    
    Returns:
        Initialized TemperatureState with all temperatures set to initial_temp
        
    Note:
        In the Fortran code, Init and InitAllocate are type-bound procedures.
        Here we provide a pure function for initialization that can be used
        in a functional JAX context.
    """
    return TemperatureState(
        t_soisno_col=jnp.full((n_columns, n_levtot), initial_temp, dtype=jnp.float32),
        t_a10_patch=jnp.full(n_patches, initial_temp, dtype=jnp.float32),
        t_ref2m_patch=jnp.full(n_patches, initial_temp, dtype=jnp.float32),
    )


def init_temperature(
    bounds: BoundsType,
    nlevsno: int,
    nlevgrnd: int,
) -> TemperatureState:
    """Initialize temperature state variables from bounds.
    
    This function allocates and initializes all temperature arrays needed
    for the simulation based on the domain decomposition bounds.
    
    Fortran source: TemperatureType.F90, lines 37-44
    
    Args:
        bounds: Domain decomposition bounds containing:
            - begc, endc: Column index bounds
            - begp, endp: Patch index bounds
            - begg, endg: Grid cell index bounds
        nlevsno: Maximum number of snow layers (positive value)
        nlevgrnd: Number of ground layers (soil + bedrock)
            
    Returns:
        TemperatureState: Initialized temperature state with allocated arrays
        
    Note:
        This is a wrapper that calls init_allocate to perform the actual
        allocation. In the JAX translation, we directly return the initialized
        NamedTuple rather than mutating a class instance.
        
    Reference:
        Fortran source lines 37-44 in TemperatureType.F90
    """
    # Call the allocation routine (equivalent to this%InitAllocate(bounds))
    return init_allocate(bounds, nlevsno, nlevgrnd)


def init_allocate(
    bounds: BoundsType,
    nlevsno: int,
    nlevgrnd: int,
) -> TemperatureState:
    """Initialize temperature state structure with allocated arrays.
    
    Allocates memory for all temperature state variables based on spatial
    bounds. All arrays are initialized to NaN to help detect uninitialized
    values during debugging.
    
    Fortran source: TemperatureType.F90, lines 47-68
    
    Args:
        bounds: Spatial bounds containing patch and column indices
        nlevsno: Maximum number of snow layers (positive value)
        nlevgrnd: Number of ground layers (soil + bedrock)
        
    Returns:
        TemperatureState with allocated arrays initialized to NaN
        
    Note:
        The t_soisno_col array spans from -nlevsno+1 (top snow layer) to
        nlevgrnd (bottom soil layer). In JAX, we use 0-based indexing, so
        the array has shape [n_cols, nlevsno + nlevgrnd].
        
        Array dimensions are determined by bounds:
        - Column arrays: (endc - begc + 1,)
        - Patch arrays: (endp - begp + 1,)
        - Soil layer arrays: (endc - begc + 1, nlevgrnd)
        - Snow layer arrays: (endc - begc + 1, nlevsno)
    """
    # Extract bounds (lines 61-62)
    begp = bounds.begp
    endp = bounds.endp
    begc = bounds.begc
    endc = bounds.endc
    
    # Calculate array sizes
    n_patches = endp - begp + 1
    n_cols = endc - begc + 1
    n_layers = nlevsno + nlevgrnd  # Total layers from -nlevsno+1 to nlevgrnd
    
    # Allocate and initialize arrays with NaN (lines 64-66)
    # Using NaN helps detect uninitialized values during debugging
    t_soisno_col = jnp.full((n_cols, n_layers), NAN_VALUE, dtype=jnp.float32)
    t_a10_patch = jnp.full((n_patches,), NAN_VALUE, dtype=jnp.float32)
    t_ref2m_patch = jnp.full((n_patches,), NAN_VALUE, dtype=jnp.float32)
    
    return TemperatureState(
        t_soisno_col=t_soisno_col,
        t_a10_patch=t_a10_patch,
        t_ref2m_patch=t_ref2m_patch,
    )


# =============================================================================
# Utility Functions
# =============================================================================


def get_soil_temperature(
    state: TemperatureState,
    col_idx: int,
    layer_idx: int,
    nlevsno: int,
) -> float:
    """Extract soil temperature for a specific column and layer.
    
    Helper function to access soil/snow temperatures with proper indexing.
    
    Args:
        state: Temperature state
        col_idx: Column index (0-based)
        layer_idx: Layer index in Fortran convention (-nlevsno+1 to nlevgrnd)
        nlevsno: Number of snow layers
        
    Returns:
        Temperature at specified location [K]
        
    Note:
        Converts Fortran layer indexing to JAX 0-based indexing:
        Fortran layer_idx -> JAX index = layer_idx + nlevsno - 1
    """
    jax_layer_idx = layer_idx + nlevsno - 1
    return state.t_soisno_col[col_idx, jax_layer_idx]


def update_soil_temperature(
    state: TemperatureState,
    col_idx: int,
    layer_idx: int,
    nlevsno: int,
    new_temp: float,
) -> TemperatureState:
    """Update soil temperature for a specific column and layer.
    
    Returns a new TemperatureState with updated temperature value.
    
    Args:
        state: Current temperature state
        col_idx: Column index (0-based)
        layer_idx: Layer index in Fortran convention (-nlevsno+1 to nlevgrnd)
        nlevsno: Number of snow layers
        new_temp: New temperature value [K]
        
    Returns:
        New TemperatureState with updated temperature
        
    Note:
        This creates a new state object (immutable update) for JAX compatibility.
    """
    jax_layer_idx = layer_idx + nlevsno - 1
    new_t_soisno = state.t_soisno_col.at[col_idx, jax_layer_idx].set(new_temp)
    
    return state._replace(t_soisno_col=new_t_soisno)


def get_surface_temperature(
    state: TemperatureState,
    col_idx: int,
    nlevsno: int,
    snl: int,
) -> float:
    """Get surface temperature (top snow or top soil layer).
    
    Args:
        state: Temperature state
        col_idx: Column index
        nlevsno: Maximum number of snow layers
        snl: Number of active snow layers (negative or zero)
        
    Returns:
        Surface temperature [K]
        
    Note:
        If snow present (snl < 0), returns top snow layer temperature.
        Otherwise returns top soil layer temperature.
    """
    # If snow present, get top snow layer (Fortran index = snl+1)
    # Otherwise get top soil layer (Fortran index = 1)
    fortran_idx = jnp.where(snl < 0, snl + 1, 1)
    jax_idx = fortran_idx + nlevsno - 1
    
    return state.t_soisno_col[col_idx, jax_idx]