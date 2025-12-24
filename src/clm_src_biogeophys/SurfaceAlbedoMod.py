"""Surface albedo calculations for CTSM.

This module provides functions for calculating surface albedo including
soil albedo based on soil moisture content, soil color classes, and
initialization routines for time-constant parameters.

Fortran source: SurfaceAlbedoMod.F90, lines 1-136

Key components:
- Soil color class management (8 or 20 classes)
- Saturated and dry soil albedo lookup tables
- Moisture-dependent albedo adjustment
- Visible and near-infrared waveband separation
"""

from typing import NamedTuple, Optional
import jax.numpy as jnp
from jax import Array, jit


# =============================================================================
# Type Definitions
# =============================================================================

class SurfaceAlbedoState(NamedTuple):
    """State container for surface albedo module data.
    
    Attributes:
        albsat: Wet soil albedo by color class and waveband (ncolor, 2).
                Waveband indices: 0=visible, 1=near-infrared.
                Fortran line 22.
        albdry: Dry soil albedo by color class and waveband (ncolor, 2).
                Waveband indices: 0=visible, 1=near-infrared.
                Fortran line 23.
        isoicol: Column soil color class indices (ncols,).
                 Fortran line 24.
    """
    albsat: Array  # shape: (ncolor, 2)
    albdry: Array  # shape: (ncolor, 2)
    isoicol: Array  # shape: (ncols,), dtype: int


class SurfaceAlbedoConstants(NamedTuple):
    """Time-constant parameters for surface albedo calculations.
    
    Attributes:
        isoicol: Soil color class index for each column (ncols,)
        albsat: Saturated soil albedo (mxsoil_color, numrad)
        albdry: Dry soil albedo (mxsoil_color, numrad)
        mxsoil_color: Maximum number of soil color classes
    """
    isoicol: Array  # shape: (n_columns,)
    albsat: Array   # shape: (mxsoil_color, numrad)
    albdry: Array   # shape: (mxsoil_color, numrad)
    mxsoil_color: int


class SoilAlbedoInputs(NamedTuple):
    """Input data for soil albedo calculation.
    
    Attributes:
        h2osoi_vol: Soil layer volumetric water content (m3/m3), shape (ncols, nlevgrnd)
        isoicol: Soil color class index for each column, shape (ncols,)
        albsat: Saturated soil albedo by color and waveband, shape (nsoilcol, numrad)
        albdry: Dry soil albedo by color and waveband, shape (nsoilcol, numrad)
        filter_nourbanc: Column indices for non-urban points, shape (num_nourbanc,)
    """
    h2osoi_vol: Array
    isoicol: Array
    albsat: Array
    albdry: Array
    filter_nourbanc: Array


class SoilAlbedoOutputs(NamedTuple):
    """Output data from soil albedo calculation.
    
    Attributes:
        albsoib: Direct beam albedo of ground (soil), shape (ncols, numrad)
        albsoid: Diffuse albedo of ground (soil), shape (ncols, numrad)
    """
    albsoib: Array
    albsoid: Array


class WaterStateType(NamedTuple):
    """Container for water state variables.
    
    Attributes:
        h2osoi_vol_col: Volumetric soil water content (m3/m3), shape (ncols, nlevgrnd)
    """
    h2osoi_vol_col: Array


class SurfAlbType(NamedTuple):
    """Container for surface albedo variables.
    
    Attributes:
        albgrd_col: Ground albedo (direct beam), shape (ncols, numrad)
        albgri_col: Ground albedo (diffuse), shape (ncols, numrad)
    """
    albgrd_col: Array
    albgri_col: Array


class BoundsType(NamedTuple):
    """Column bounds for CLM grid.
    
    Attributes:
        begc: Beginning column index
        endc: Ending column index
    """
    begc: int
    endc: int


# =============================================================================
# Constants
# =============================================================================

# Radiation waveband indices (Fortran clm_varpar module)
IVIS = 0  # Visible waveband index
INIR = 1  # Near-infrared waveband index
NUMRAD = 2  # Number of radiation wavebands

# Soil color class systems
MXSOIL_COLOR_8 = 8   # 8-class soil color system
MXSOIL_COLOR_20 = 20  # 20-class soil color system (default)


# =============================================================================
# Initialization Functions
# =============================================================================

def create_surface_albedo_state(
    ncolor: int,
    ncols: int,
    albsat_init: Optional[Array] = None,
    albdry_init: Optional[Array] = None,
    isoicol_init: Optional[Array] = None,
) -> SurfaceAlbedoState:
    """Create initial surface albedo state.
    
    Args:
        ncolor: Number of soil color classes.
        ncols: Number of columns.
        albsat_init: Optional initial wet soil albedo values (ncolor, 2).
        albdry_init: Optional initial dry soil albedo values (ncolor, 2).
        isoicol_init: Optional initial soil color class indices (ncols,).
    
    Returns:
        Initialized SurfaceAlbedoState.
        
    Note:
        Fortran lines 22-24: Module-level allocatable arrays.
    """
    if albsat_init is None:
        albsat_init = jnp.zeros((ncolor, 2), dtype=jnp.float64)
    if albdry_init is None:
        albdry_init = jnp.zeros((ncolor, 2), dtype=jnp.float64)
    if isoicol_init is None:
        isoicol_init = jnp.zeros(ncols, dtype=jnp.int32)
    
    return SurfaceAlbedoState(
        albsat=albsat_init,
        albdry=albdry_init,
        isoicol=isoicol_init,
    )


def surface_albedo_init_time_const(
    tower_isoicol: int,
    n_columns: int,
    numrad: int = NUMRAD,
    ivis: int = IVIS,
    inir: int = INIR,
    mxsoil_color: int = MXSOIL_COLOR_20
) -> SurfaceAlbedoConstants:
    """Initialize module time constant variables for surface albedo.
    
    This function sets up soil color classes and albedo lookup tables
    for saturated and dry soil conditions across visible and near-infrared
    wavebands.
    
    Fortran source: SurfaceAlbedoMod.F90, lines 29-83
    
    Args:
        tower_isoicol: Soil color class index from tower data
        n_columns: Number of columns (endc - begc + 1)
        numrad: Number of radiation wavebands (default: 2, vis and nir)
        ivis: Index for visible waveband (default: 0)
        inir: Index for near-infrared waveband (default: 1)
        mxsoil_color: Maximum number of soil color classes (default: 20)
        
    Returns:
        SurfaceAlbedoConstants containing initialized albedo parameters
        
    Raises:
        ValueError: If mxsoil_color is not 8 or 20
        
    Notes:
        - Lines 29-36: Subroutine declaration and documentation
        - Lines 37-55: Allocate and assign soil color for all columns
        - Lines 56-77: Set saturated and dry soil albedos based on color classes
        - Lines 78-80: Error handling for unsupported color classes
        
    Physics:
        Soil albedo lookup tables from Bonan (1996) and Lawrence & Chase (2007).
        - 8-class system: Simplified color classes
        - 20-class system: Extended color classes for better representation
        - Saturated albedo < Dry albedo (darker when wet)
        - NIR albedo > VIS albedo (soil reflects more in near-infrared)
    """
    # Lines 51-55: Allocate module variable for soil color and assign value
    # All columns get the same soil color from tower data
    isoicol = jnp.full((n_columns,), tower_isoicol, dtype=jnp.int32)
    
    # Lines 56-60: Set saturated and dry soil albedos for mxsoil_color color classes
    # and numrad wavebands (1=vis, 2=nir)
    
    # Initialize arrays
    albsat = jnp.zeros((mxsoil_color, numrad), dtype=jnp.float64)
    albdry = jnp.zeros((mxsoil_color, numrad), dtype=jnp.float64)
    
    # Lines 62-67: Handle mxsoil_color == 8 case
    if mxsoil_color == 8:
        # 8-class soil color system
        # Albedo decreases with increasing color class (darker soils)
        albsat_vis_8 = jnp.array(
            [0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05], 
            dtype=jnp.float64
        )
        albsat_nir_8 = jnp.array(
            [0.24, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10], 
            dtype=jnp.float64
        )
        albdry_vis_8 = jnp.array(
            [0.24, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10], 
            dtype=jnp.float64
        )
        albdry_nir_8 = jnp.array(
            [0.48, 0.44, 0.40, 0.36, 0.32, 0.28, 0.24, 0.20], 
            dtype=jnp.float64
        )
        
        albsat = albsat.at[:8, ivis].set(albsat_vis_8)
        albsat = albsat.at[:8, inir].set(albsat_nir_8)
        albdry = albdry.at[:8, ivis].set(albdry_vis_8)
        albdry = albdry.at[:8, inir].set(albdry_nir_8)
        
    # Lines 68-77: Handle mxsoil_color == 20 case
    elif mxsoil_color == 20:
        # 20-class soil color system (default)
        # Finer gradation of soil colors for improved accuracy
        albsat_vis_20 = jnp.array([
            0.25, 0.23, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14,
            0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04
        ], dtype=jnp.float64)
        
        albsat_nir_20 = jnp.array([
            0.50, 0.46, 0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28,
            0.26, 0.24, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08
        ], dtype=jnp.float64)
        
        albdry_vis_20 = jnp.array([
            0.36, 0.34, 0.32, 0.31, 0.30, 0.29, 0.28, 0.27, 0.26, 0.25,
            0.24, 0.23, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08
        ], dtype=jnp.float64)
        
        albdry_nir_20 = jnp.array([
            0.61, 0.57, 0.53, 0.51, 0.49, 0.48, 0.45, 0.43, 0.41, 0.39,
            0.37, 0.35, 0.33, 0.31, 0.29, 0.27, 0.25, 0.23, 0.21, 0.16
        ], dtype=jnp.float64)
        
        albsat = albsat.at[:20, ivis].set(albsat_vis_20)
        albsat = albsat.at[:20, inir].set(albsat_nir_20)
        albdry = albdry.at[:20, ivis].set(albdry_vis_20)
        albdry = albdry.at[:20, inir].set(albdry_nir_20)
        
    # Lines 78-80: Error handling
    else:
        raise ValueError(
            f"ERROR: SurfaceAlbedoInitTimeConst: maximum color class {mxsoil_color} "
            "is not supported. Only 8 or 20 color classes are supported."
        )
    
    return SurfaceAlbedoConstants(
        isoicol=isoicol,
        albsat=albsat,
        albdry=albdry,
        mxsoil_color=mxsoil_color
    )


# =============================================================================
# Soil Albedo Calculation
# =============================================================================

@jit
def soil_albedo(
    inputs: SoilAlbedoInputs,
    numrad: int = NUMRAD
) -> SoilAlbedoOutputs:
    """Calculate ground surface (soil) albedo.
    
    Computes soil albedo based on soil moisture content in the top layer,
    soil color class, and reference albedo values for saturated and dry conditions.
    
    The albedo is adjusted from the saturated value based on soil moisture,
    with a correction factor that increases albedo as soil dries, up to the
    dry soil albedo limit.
    
    Fortran source: SurfaceAlbedoMod.F90, lines 86-134
    
    Args:
        inputs: SoilAlbedoInputs containing soil moisture, color classes, and
                reference albedo values
        numrad: Number of radiation wavebands (default 2: visible and near-infrared)
    
    Returns:
        SoilAlbedoOutputs containing direct beam and diffuse soil albedo for
        each waveband
        
    Physics:
        The moisture correction follows:
        - inc = max(0.11 - 0.40 * h2osoi_vol, 0.0)  [Line 127]
        - albedo = min(albsat + inc, albdry)  [Line 128]
        
        This formulation:
        - Starts from saturated (dark) albedo
        - Increases linearly as soil dries (h2osoi_vol decreases)
        - Caps at dry albedo value
        - Coefficient 0.40 controls sensitivity to moisture
        - Offset 0.11 sets maximum brightness increase
        
    Note:
        - Direct beam and diffuse albedos are set equal (line 129)
        - Only top soil layer moisture affects albedo
        - Applied only to non-urban columns via filter
    """
    # Extract filter indices
    filter_cols = inputs.filter_nourbanc
    num_nourbanc = filter_cols.shape[0]
    
    # Get soil moisture for top layer (index 0) at filtered columns
    # Shape: (num_nourbanc,)
    h2osoi_vol_top = inputs.h2osoi_vol[filter_cols, 0]
    
    # Get soil color indices for filtered columns
    # Shape: (num_nourbanc,)
    soilcol = inputs.isoicol[filter_cols]
    
    # Calculate soil water correction factor (line 127)
    # inc increases as soil dries (lower h2osoi_vol)
    # Physical interpretation:
    # - Wet soil (h2osoi_vol ~ 0.275): inc ~ 0.0 (no correction)
    # - Dry soil (h2osoi_vol ~ 0.0): inc ~ 0.11 (maximum brightening)
    inc = jnp.maximum(0.11 - 0.40 * h2osoi_vol_top, 0.0)
    # Shape: (num_nourbanc,)
    
    # Initialize output arrays
    ncols = inputs.h2osoi_vol.shape[0]
    albsoib = jnp.zeros((ncols, numrad), dtype=jnp.float64)
    albsoid = jnp.zeros((ncols, numrad), dtype=jnp.float64)
    
    # Calculate albedo for each waveband (lines 124-130)
    for ib in range(numrad):
        # Get saturated and dry albedo for this waveband and soil colors
        # Shape: (num_nourbanc,)
        albsat_ib = inputs.albsat[soilcol, ib]
        albdry_ib = inputs.albdry[soilcol, ib]
        
        # Calculate albedo with moisture correction (line 128)
        # Bounded between saturated (wet) and dry values
        # albsat_ib: dark (wet) limit
        # albdry_ib: bright (dry) limit
        alb_ib = jnp.minimum(albsat_ib + inc, albdry_ib)
        
        # Update output arrays at filtered column indices (lines 128-129)
        # Direct beam and diffuse albedos are equal for soil
        albsoib = albsoib.at[filter_cols, ib].set(alb_ib)
        albsoid = albsoid.at[filter_cols, ib].set(alb_ib)
    
    return SoilAlbedoOutputs(
        albsoib=albsoib,
        albsoid=albsoid
    )


def soil_albedo_wrapper(
    bounds: BoundsType,
    num_nourbanc: int,
    filter_nourbanc: Array,
    waterstate_inst: WaterStateType,
    isoicol: Array,
    albsat: Array,
    albdry: Array,
    numrad: int = NUMRAD
) -> SurfAlbType:
    """Wrapper function for soil albedo calculation matching Fortran interface.
    
    This function provides a higher-level interface that matches the original
    Fortran subroutine signature more closely.
    
    Fortran source: SurfaceAlbedoMod.F90, lines 86-134
    
    Args:
        bounds: CLM column bounds
        num_nourbanc: Number of non-urban points in CLM column filter
        filter_nourbanc: CLM column filter for non-urban points
        waterstate_inst: Water state data including soil moisture
        isoicol: Soil color class indices
        albsat: Saturated soil albedo lookup table
        albdry: Dry soil albedo lookup table
        numrad: Number of radiation wavebands
    
    Returns:
        SurfAlbType containing updated albedo fields
        
    Note:
        This wrapper handles the interface between CLM data structures
        and the pure functional soil_albedo implementation.
    """
    inputs = SoilAlbedoInputs(
        h2osoi_vol=waterstate_inst.h2osoi_vol_col,
        isoicol=isoicol,
        albsat=albsat,
        albdry=albdry,
        filter_nourbanc=filter_nourbanc[:num_nourbanc]
    )
    
    outputs = soil_albedo(inputs, numrad=numrad)
    
    # Return updated surface albedo type
    return SurfAlbType(
        albgrd_col=outputs.albsoib,
        albgri_col=outputs.albsoid
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Types
    'SurfaceAlbedoState',
    'SurfaceAlbedoConstants',
    'SoilAlbedoInputs',
    'SoilAlbedoOutputs',
    'WaterStateType',
    'SurfAlbType',
    'BoundsType',
    # Constants
    'IVIS',
    'INIR',
    'NUMRAD',
    'MXSOIL_COLOR_8',
    'MXSOIL_COLOR_20',
    # Functions
    'create_surface_albedo_state',
    'surface_albedo_init_time_const',
    'soil_albedo',
    'soil_albedo_wrapper',
]# Backward compatibility alias (Fortran naming convention)
SurfaceAlbedoInitTimeConst = surface_albedo_init_time_const
SoilAlbedo = soil_albedo
