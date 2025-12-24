"""
Soil State Type Definition and Initialization.

Translated from CTSM's SoilStateType.F90

This module defines the soil state variables and initialization routines for
the Community Terrestrial Systems Model (CTSM). It manages soil physical
properties including:

- Soil texture (sand, clay, organic matter content)
- Hydraulic properties (conductivity, matric potential, porosity)
- Thermal properties (conductivity, heat capacity)
- Root distribution

The original Fortran module uses a derived type with allocatable pointers.
In JAX, we use an immutable NamedTuple with fixed-size arrays for JIT
compatibility and functional programming patterns.

Key differences from Fortran:
- Immutable NamedTuple instead of mutable derived type
- Explicit array dimensions instead of dynamic allocation
- Pure functions instead of class methods
- NaN initialization for unset values (matching Fortran spval pattern)

Reference:
    SoilStateType.F90 (lines 1-106)
"""

from typing import NamedTuple
import jax.numpy as jnp


# ============================================================================
# Type Definitions
# ============================================================================


class BoundsType(NamedTuple):
    """Domain bounds for grid hierarchy.
    
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


class SoilStateType(NamedTuple):
    """Soil state variables.
    
    Translated from Fortran type soilstate_type (lines 20-48).
    
    All arrays are immutable and JIT-compatible. Dimensions follow CTSM conventions:
    - n_columns: Number of columns
    - n_patches: Number of patches
    - nlevsoi: Number of soil layers (typically 10)
    - nlevgrnd: Number of ground layers including bedrock (typically 15)
    - nlevsno: Number of snow layers (typically 5)
    
    Attributes:
        cellorg_col: Organic matter content [kg/m3] [n_columns, nlevsoi]
            (Fortran line 23)
        cellsand_col: Sand content [percent] [n_columns, nlevsoi]
            (Fortran line 24)
        cellclay_col: Clay content [percent] [n_columns, nlevsoi]
            (Fortran line 25)
        hksat_col: Hydraulic conductivity at saturation [mm H2O/s]
            [n_columns, nlevgrnd] (Fortran line 28)
        hk_l_col: Hydraulic conductivity [mm H2O/s]
            [n_columns, nlevgrnd] (Fortran line 29)
        smp_l_col: Soil matric potential [mm]
            [n_columns, nlevgrnd] (Fortran line 30)
        bsw_col: Clapp and Hornberger "b" parameter [-]
            [n_columns, nlevgrnd] (Fortran line 31)
        watsat_col: Volumetric soil water at saturation (porosity) [-]
            [n_columns, nlevgrnd] (Fortran line 32)
        sucsat_col: Minimum soil suction [mm]
            [n_columns, nlevgrnd] (Fortran line 33)
        dsl_col: Dry surface layer thickness [mm]
            [n_columns] (Fortran line 34)
        soilresis_col: Soil evaporative resistance (Swenson & Lawrence 2014) [s/m]
            [n_columns] (Fortran line 35)
        thk_col: Thermal conductivity of each layer [W/m/K]
            [n_columns, nlevgrnd+nlevsno] (Fortran line 38)
            Note: Includes snow layers (-nlevsno+1:nlevgrnd)
        tkmg_col: Thermal conductivity, soil minerals [W/m/K]
            [n_columns, nlevgrnd] (Fortran line 39)
        tkdry_col: Thermal conductivity, dry soil [W/m/K]
            [n_columns, nlevgrnd] (Fortran line 40)
        csol_col: Heat capacity, soil solids [J/m**3/K]
            [n_columns, nlevgrnd] (Fortran line 41)
        rootfr_patch: Effective fraction of roots in each soil layer [-]
            [n_patches, nlevgrnd] (Fortran line 44)
    """
    
    # Soil texture (lines 23-25)
    cellorg_col: jnp.ndarray  # [n_columns, nlevsoi]
    cellsand_col: jnp.ndarray  # [n_columns, nlevsoi]
    cellclay_col: jnp.ndarray  # [n_columns, nlevsoi]
    
    # Hydraulic properties (lines 28-35)
    hksat_col: jnp.ndarray  # [n_columns, nlevgrnd]
    hk_l_col: jnp.ndarray  # [n_columns, nlevgrnd]
    smp_l_col: jnp.ndarray  # [n_columns, nlevgrnd]
    bsw_col: jnp.ndarray  # [n_columns, nlevgrnd]
    watsat_col: jnp.ndarray  # [n_columns, nlevgrnd]
    sucsat_col: jnp.ndarray  # [n_columns, nlevgrnd]
    dsl_col: jnp.ndarray  # [n_columns]
    soilresis_col: jnp.ndarray  # [n_columns]
    
    # Thermal properties (lines 38-41)
    thk_col: jnp.ndarray  # [n_columns, nlevgrnd+nlevsno]
    tkmg_col: jnp.ndarray  # [n_columns, nlevgrnd]
    tkdry_col: jnp.ndarray  # [n_columns, nlevgrnd]
    csol_col: jnp.ndarray  # [n_columns, nlevgrnd]
    
    # Root distribution (line 44)
    rootfr_patch: jnp.ndarray  # [n_patches, nlevgrnd]


# ============================================================================
# Initialization Functions
# ============================================================================


def init_soil_state(
    bounds: BoundsType,
    nlevsoi: int = 10,
    nlevgrnd: int = 15,
    nlevsno: int = 5,
) -> SoilStateType:
    """Initialize soil state type.
    
    This is the main entry point for creating a SoilStateType instance.
    It allocates all necessary arrays based on the provided bounds and
    initializes them to NaN to catch uninitialized values.
    
    Fortran source: SoilStateType.F90, lines 57-64 (Init subroutine)
    
    Args:
        bounds: Domain bounds containing grid dimensions
            - begc, endc: Column index bounds
            - begp, endp: Patch index bounds
            - begg, endg: Grid cell index bounds
        nlevsoi: Number of soil layers (default: 10)
        nlevgrnd: Number of ground layers including bedrock (default: 15)
        nlevsno: Number of snow layers (default: 5)
            
    Returns:
        Initialized SoilStateType with NaN-filled arrays
        
    Note:
        This is a pure function that creates a new SoilStateType instance.
        In the original Fortran, this was a class method that modified
        the object in place. The JAX version returns a new immutable
        NamedTuple.
        
    Example:
        >>> bounds = BoundsType(begp=1, endp=100, begc=1, endc=50, 
        ...                     begg=1, endg=25)
        >>> soil_state = init_soil_state(bounds)
        >>> soil_state.cellsand_col.shape
        (50, 10)
    """
    # Call the allocation routine to create initialized soil state
    # (lines 62-63 in Fortran: call this%InitAllocate(bounds))
    return init_allocate(bounds, nlevsoi, nlevgrnd, nlevsno)


def init_allocate(
    bounds: BoundsType,
    nlevsoi: int,
    nlevgrnd: int,
    nlevsno: int,
) -> SoilStateType:
    """Initialize and allocate soil state arrays.
    
    Creates a SoilStateType with all arrays initialized to NaN.
    This follows the Fortran pattern of allocating arrays and setting
    them to NaN (spval) to catch uninitialized values.
    
    Fortran source: SoilStateType.F90, lines 67-104 (InitAllocate subroutine)
    
    Args:
        bounds: Bounds containing patch and column indices
        nlevsoi: Number of soil layers (typically 10)
        nlevgrnd: Number of ground layers including bedrock (typically 15)
        nlevsno: Number of snow layers (typically 5)
        
    Returns:
        Initialized SoilStateType with NaN-filled arrays
        
    Implementation Notes:
        - All arrays are initialized to NaN (matching Fortran spval)
        - Array dimensions are calculated from bounds
        - thk_col includes snow layers: shape is [n_columns, nlevgrnd+nlevsno]
        - This corresponds to Fortran indexing (-nlevsno+1:nlevgrnd)
        
    Reference:
        Lines 81-102 in SoilStateType.F90
    """
    # Extract bounds (lines 81-82)
    begp = bounds.begp
    endp = bounds.endp
    begc = bounds.begc
    endc = bounds.endc
    
    # Calculate array sizes
    # In Fortran: begc:endc means endc-begc+1 elements
    n_columns = endc - begc + 1
    n_patches = endp - begp + 1
    
    # Initialize soil texture arrays (lines 84-86)
    # allocate(this%cellorg_col  (begc:endc,1:nlevsoi))
    cellorg_col = jnp.full((n_columns, nlevsoi), jnp.nan)
    cellsand_col = jnp.full((n_columns, nlevsoi), jnp.nan)
    cellclay_col = jnp.full((n_columns, nlevsoi), jnp.nan)
    
    # Initialize hydraulic property arrays (lines 88-94)
    hksat_col = jnp.full((n_columns, nlevgrnd), jnp.nan)
    hk_l_col = jnp.full((n_columns, nlevgrnd), jnp.nan)
    smp_l_col = jnp.full((n_columns, nlevgrnd), jnp.nan)
    bsw_col = jnp.full((n_columns, nlevgrnd), jnp.nan)
    watsat_col = jnp.full((n_columns, nlevgrnd), jnp.nan)
    sucsat_col = jnp.full((n_columns, nlevgrnd), jnp.nan)
    dsl_col = jnp.full((n_columns,), jnp.nan)
    soilresis_col = jnp.full((n_columns,), jnp.nan)
    
    # Initialize thermal property arrays (lines 96-99)
    # Note: thk_col includes snow layers (-nlevsno+1:nlevgrnd)
    # In 0-based indexing, this is nlevgrnd + nlevsno total layers
    n_total_layers = nlevgrnd + nlevsno
    thk_col = jnp.full((n_columns, n_total_layers), jnp.nan)
    tkmg_col = jnp.full((n_columns, nlevgrnd), jnp.nan)
    tkdry_col = jnp.full((n_columns, nlevgrnd), jnp.nan)
    csol_col = jnp.full((n_columns, nlevgrnd), jnp.nan)
    
    # Initialize root fraction array (line 101)
    # allocate(this%rootfr_patch (begp:endp,1:nlevgrnd))
    rootfr_patch = jnp.full((n_patches, nlevgrnd), jnp.nan)
    
    # Create and return the SoilStateType instance
    return SoilStateType(
        cellorg_col=cellorg_col,
        cellsand_col=cellsand_col,
        cellclay_col=cellclay_col,
        hksat_col=hksat_col,
        hk_l_col=hk_l_col,
        smp_l_col=smp_l_col,
        bsw_col=bsw_col,
        watsat_col=watsat_col,
        sucsat_col=sucsat_col,
        dsl_col=dsl_col,
        soilresis_col=soilresis_col,
        thk_col=thk_col,
        tkmg_col=tkmg_col,
        tkdry_col=tkdry_col,
        csol_col=csol_col,
        rootfr_patch=rootfr_patch,
    )


# ============================================================================
# Utility Functions
# ============================================================================


def create_simple_bounds(
    n_patches: int,
    n_columns: int,
    n_gridcells: int,
) -> BoundsType:
    """Create simple bounds for testing.
    
    Helper function to create BoundsType with 1-based indexing
    (matching Fortran convention).
    
    Args:
        n_patches: Number of patches
        n_columns: Number of columns
        n_gridcells: Number of grid cells
        
    Returns:
        BoundsType with 1-based indices
        
    Example:
        >>> bounds = create_simple_bounds(100, 50, 25)
        >>> bounds.begp, bounds.endp
        (1, 100)
    """
    return BoundsType(
        begp=1,
        endp=n_patches,
        begc=1,
        endc=n_columns,
        begg=1,
        endg=n_gridcells,
    )


def get_soil_layer_indices(
    nlevsno: int = 5,
    nlevgrnd: int = 15,
) -> tuple[int, int, int]:
    """Get soil layer index ranges.
    
    Helper to understand the layer indexing convention.
    
    Args:
        nlevsno: Number of snow layers
        nlevgrnd: Number of ground layers
        
    Returns:
        Tuple of (snow_start_idx, ground_start_idx, ground_end_idx)
        
    Note:
        In Fortran, thk_col is indexed as (-nlevsno+1:nlevgrnd).
        In JAX with 0-based indexing:
        - Snow layers: indices 0 to nlevsno-1
        - Ground layers: indices nlevsno to nlevsno+nlevgrnd-1
    """
    snow_start_idx = 0
    ground_start_idx = nlevsno
    ground_end_idx = nlevsno + nlevgrnd
    
    return snow_start_idx, ground_start_idx, ground_end_idx


# Backward compatibility alias (Fortran naming convention)
soilstate_type = SoilStateType