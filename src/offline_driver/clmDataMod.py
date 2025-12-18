"""
CLM Data Module for Reading Tower Site Forcing Data.

Translated from CTSM's clmDataMod.F90

This module provides functionality to read CLM forcing data for tower site
simulations. It handles reading single-level variables (LAI, SAI, coszen) and
soil water/temperature from CLM netCDF history files.

The module supports both CLM4.5 and CLM5.0 physics versions, which differ in
their soil layer configurations:
    - CLM4.5: Uses nlevgrnd soil layers (typically 15)
    - CLM5.0: Uses nlevsoi soil layers (typically 20)

Key operations:
    1. Read canopy state (LAI, SAI) from NetCDF files
    2. Read surface albedo data (coszen)
    3. Read volumetric soil moisture for appropriate CLM version
    4. Convert volumetric soil moisture to liquid water mass
    5. Apply saturation limits for CLM5.0

Physical constraints:
    - Soil moisture limited to watsat (saturation) for CLM5.0
    - Ice initialized to zero (liquid water only)
    - LAI and SAI must be non-negative
    - Coszen must be in range [-1, 1]

Reference: clmDataMod.F90 lines 1-278
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp


# =============================================================================
# Type Definitions
# =============================================================================


class CLMColumnState(NamedTuple):
    """Column-level state data from CLM.
    
    Corresponds to ColumnType::col from Fortran (line 8).
    """
    pass  # Placeholder for column state data


class CLMSoilState(NamedTuple):
    """Soil state data from CLM.
    
    Corresponds to SoilStateType::soilstate_type from Fortran (line 10).
    """
    pass  # Placeholder for soil state data


class CLMWaterState(NamedTuple):
    """Water state data from CLM.
    
    Corresponds to WaterStateType::waterstate_type from Fortran (line 11).
    """
    pass  # Placeholder for water state data


class CLMCanopyState(NamedTuple):
    """Canopy state data from CLM.
    
    Corresponds to CanopyStateType::canopystate_type from Fortran (line 12).
    """
    pass  # Placeholder for canopy state data


class CLMSurfaceAlbedo(NamedTuple):
    """Surface albedo data from CLM.
    
    Corresponds to SurfaceAlbedoType::surfalb_type from Fortran (line 13).
    """
    pass  # Placeholder for surface albedo data


class CLMDataInputs(NamedTuple):
    """Inputs for CLM data reading.
    
    Attributes:
        dz: Soil layer thickness [m] [n_columns, nlevgrnd]
        nbedrock: Depth to bedrock index [n_columns]
        watsat: Soil layer volumetric water content at saturation [m3/m3] [n_columns, nlevgrnd]
        elai_loc: Leaf area index from file [m2/m2] [scalar]
        esai_loc: Stem area index from file [m2/m2] [scalar]
        coszen_loc: Cosine solar zenith angle from file [scalar]
        h2osoi_clm45: Volumetric soil moisture for CLM4.5 [m3/m3] [nlevgrnd]
        h2osoi_clm50: Volumetric soil moisture for CLM5.0 [m3/m3] [nlevsoi]
        clm_phys: CLM physics version string ('CLM4_5' or 'CLM5_0')
        nlevgrnd: Number of ground layers
        nlevsoi: Number of soil layers (CLM5.0)
        denh2o: Density of liquid water [kg/m3]
    """
    dz: jnp.ndarray
    nbedrock: jnp.ndarray
    watsat: jnp.ndarray
    elai_loc: float
    esai_loc: float
    coszen_loc: float
    h2osoi_clm45: jnp.ndarray
    h2osoi_clm50: jnp.ndarray
    clm_phys: str
    nlevgrnd: int
    nlevsoi: int
    denh2o: float


class CLMDataOutputs(NamedTuple):
    """Outputs from CLM data reading.
    
    Attributes:
        elai: Leaf area index of canopy [m2/m2] [n_patches]
        esai: Stem area index of canopy [m2/m2] [n_patches]
        coszen: Cosine of solar zenith angle [n_columns]
        h2osoi_vol: Soil layer volumetric water content [m3/m3] [n_columns, nlevgrnd]
        h2osoi_liq: Soil layer liquid water [kg H2O/m2] [n_columns, nlevgrnd]
        h2osoi_ice: Soil layer ice lens [kg H2O/m2] [n_columns, nlevgrnd]
    """
    elai: jnp.ndarray
    esai: jnp.ndarray
    coszen: jnp.ndarray
    h2osoi_vol: jnp.ndarray
    h2osoi_liq: jnp.ndarray
    h2osoi_ice: jnp.ndarray


class CLMDataSlice(NamedTuple):
    """Data read from CLM history file for a single time slice.
    
    Attributes:
        elai: Leaf area index [m2/m2] [scalar]
        esai: Stem area index [m2/m2] [scalar]
        coszen: Cosine of solar zenith angle [dimensionless] [scalar]
    """
    elai: jnp.ndarray  # Shape: ()
    esai: jnp.ndarray  # Shape: ()
    coszen: jnp.ndarray  # Shape: ()


class SoilMoistureData(NamedTuple):
    """Container for soil moisture data from CLM history files.
    
    Attributes:
        h2osoi_clm45: Volumetric soil moisture for CLM4.5 [m3/m3] [1, 1, nlevgrnd]
        h2osoi_clm50: Volumetric soil moisture for CLM5.0 [m3/m3] [1, 1, nlevsoi]
    """
    h2osoi_clm45: jnp.ndarray
    h2osoi_clm50: jnp.ndarray


# =============================================================================
# Constants
# =============================================================================

# Physical constants
DENH2O_DEFAULT = 1000.0  # Density of liquid water [kg/m3] (line 9)
SPVAL_DEFAULT = 1.0e36  # Special value for missing data (line 12)

# CLM version identifiers
CLM45_VERSION = 'CLM4_5'
CLM50_VERSION = 'CLM5_0'

# Default layer counts
NLEVGRND_DEFAULT = 15  # Number of ground layers for CLM4.5 (line 10)
NLEVSOI_DEFAULT = 20   # Number of soil layers for CLM5.0 (line 10)


# =============================================================================
# Main Data Processing Functions
# =============================================================================


def clm_data(
    inputs: CLMDataInputs,
    n_patches: int,
    n_columns: int,
) -> CLMDataOutputs:
    """Read variables from CLM netCDF history file.
    
    This function processes CLM history file data and distributes it to all
    patches and columns. It handles different CLM physics versions (4.5 vs 5.0)
    and applies appropriate constraints on soil moisture.
    
    The function performs the following operations:
        1. Broadcast single-level canopy data (LAI, SAI) to all patches
        2. Broadcast surface data (coszen) to all columns
        3. Select appropriate soil moisture based on CLM version
        4. Apply saturation limits for CLM5.0
        5. Convert volumetric soil moisture to liquid water mass
        6. Initialize ice to zero
    
    Args:
        inputs: Input data structure containing soil properties and file data
        n_patches: Number of patches (begp to endp)
        n_columns: Number of columns (begc to endc)
        
    Returns:
        CLMDataOutputs containing canopy, surface, and water state variables
        
    Note:
        Reference: clmDataMod.F90 lines 32-140
        
        - All patches receive the same LAI and SAI values (lines 68-71)
        - All columns receive the same coszen value (lines 75-77)
        - Soil moisture is limited to watsat for CLM5.0 (lines 97-101)
        - Ice is initialized to zero (line 106)
        - Liquid water computed as: h2osoi_liq = h2osoi_vol * dz * denh2o
    """
    # Broadcast single-level data to all patches (lines 68-71)
    elai = jnp.full(n_patches, inputs.elai_loc)
    esai = jnp.full(n_patches, inputs.esai_loc)
    
    # Broadcast coszen to all columns (lines 75-77)
    coszen = jnp.full(n_columns, inputs.coszen_loc)
    
    # Initialize soil moisture array (lines 81-95)
    h2osoi_vol = jnp.zeros((n_columns, inputs.nlevgrnd))
    
    # Select appropriate soil moisture based on CLM physics version
    if inputs.clm_phys == CLM45_VERSION:
        # Use CLM4.5 soil layers (lines 84-86)
        h2osoi_vol = jnp.tile(inputs.h2osoi_clm45, (n_columns, 1))
    elif inputs.clm_phys == CLM50_VERSION:
        # Use CLM5.0 soil layers (lines 87-92)
        # First nlevsoi layers from CLM5.0 data
        h2osoi_vol_soil = jnp.tile(inputs.h2osoi_clm50, (n_columns, 1))
        # Remaining layers are zero
        h2osoi_vol_zero = jnp.zeros((n_columns, inputs.nlevgrnd - inputs.nlevsoi))
        h2osoi_vol = jnp.concatenate([h2osoi_vol_soil, h2osoi_vol_zero], axis=1)
    
    # Limit hydrologically active soil layers to <= watsat for CLM5.0 (lines 97-101)
    if inputs.clm_phys == CLM50_VERSION:
        # Apply saturation limit only up to bedrock depth
        # Create mask for layers above bedrock
        layer_indices = jnp.arange(inputs.nlevgrnd)
        # Broadcast nbedrock for comparison
        nbedrock_broadcast = inputs.nbedrock[:, jnp.newaxis]
        active_mask = layer_indices < nbedrock_broadcast
        
        # Apply minimum constraint where active
        h2osoi_vol = jnp.where(
            active_mask,
            jnp.minimum(h2osoi_vol, inputs.watsat),
            h2osoi_vol
        )
    
    # Set liquid water and ice (lines 105-107)
    # Convert volumetric water to mass [kg/m2]
    h2osoi_liq = h2osoi_vol * inputs.dz * inputs.denh2o
    
    # Initialize ice to zero (line 106)
    h2osoi_ice = jnp.zeros_like(h2osoi_liq)
    
    return CLMDataOutputs(
        elai=elai,
        esai=esai,
        coszen=coszen,
        h2osoi_vol=h2osoi_vol,
        h2osoi_liq=h2osoi_liq,
        h2osoi_ice=h2osoi_ice,
    )


# =============================================================================
# NetCDF Data Extraction Functions
# =============================================================================


def read_clm_data_slice(
    elai_data: jnp.ndarray,
    esai_data: jnp.ndarray,
    coszen_data: jnp.ndarray,
    time_index: int,
) -> CLMDataSlice:
    """Extract CLM data for a specific time slice.
    
    This function extracts data from pre-loaded NetCDF arrays for a given
    time index. The original Fortran subroutine performs NetCDF I/O operations,
    but in JAX we separate I/O from computation for JIT compatibility.
    
    Original Fortran behavior (lines 143-206):
        - Lines 169-171: Open NetCDF file
        - Lines 177-178: Set start2 = [1, strt], count2 = [1, 1]
        - Lines 182-186: Read ELAI variable
        - Lines 190-194: Read ESAI variable
        - Lines 198-202: Read COSZEN variable (with strt-1 for time index)
        - Line 206: Close file
    
    Args:
        elai_data: Full ELAI array from NetCDF [nlndgrid, ntime]
        esai_data: Full ESAI array from NetCDF [nlndgrid, ntime]
        coszen_data: Full COSZEN array from NetCDF [nlndgrid, ntime]
        time_index: Time slice to extract (1-based index as in Fortran)
        
    Returns:
        CLMDataSlice containing the extracted data for the time slice
        
    Note:
        - Original Fortran uses 1-based indexing for time (strt)
        - For COSZEN, Fortran uses strt-1 (line 200)
        - We convert to 0-based indexing for JAX arrays
        - Original arrays are (1,1,1) shaped outputs; we return scalars
        - Spatial dimension is 1 (single grid cell) in original code
    """
    # Convert 1-based Fortran index to 0-based Python index
    # Lines 177-178: start2 = [1, strt] means spatial index 0, time index strt-1
    time_idx_0based = time_index - 1
    
    # Extract data for first spatial point (index 0) and given time
    # Lines 182-186: Read ELAI with start2=[1, strt], count2=[1, 1]
    elai = elai_data[0, time_idx_0based]
    
    # Lines 190-194: Read ESAI with start2=[1, strt], count2=[1, 1]
    esai = esai_data[0, time_idx_0based]
    
    # Lines 198-202: Read COSZEN with start2=[1, strt-1], count2=[1, 1]
    # Note: Fortran uses strt-1 for COSZEN (line 200)
    coszen_idx = jnp.maximum(time_idx_0based - 1, 0)  # Prevent negative index
    coszen = coszen_data[0, coszen_idx]
    
    return CLMDataSlice(
        elai=elai,
        esai=esai,
        coszen=coszen,
    )


def validate_clm_data(data: CLMDataSlice) -> CLMDataSlice:
    """Validate CLM data values are physically reasonable.
    
    Applies physical constraints to ensure data quality:
        - ELAI, ESAI >= 0 (non-negative area indices)
        - COSZEN in [-1, 1] (valid cosine range)
    
    Args:
        data: CLM data slice to validate
        
    Returns:
        Validated CLM data slice with constraints applied
        
    Note:
        This function can be extended to add additional physical constraints
        as needed for specific applications.
    """
    # Ensure non-negative area indices
    elai = jnp.maximum(data.elai, 0.0)
    esai = jnp.maximum(data.esai, 0.0)
    
    # Ensure coszen is in valid range [-1, 1]
    coszen = jnp.clip(data.coszen, -1.0, 1.0)
    
    return CLMDataSlice(
        elai=elai,
        esai=esai,
        coszen=coszen,
    )


# =============================================================================
# Soil Moisture Reading (Non-JIT I/O Function)
# =============================================================================


def read_clm_soil(
    ncfilename: str,
    strt: int,
    nlevgrnd: int,
    nlevsoi: int,
    clm_phys: str,
    spval: float = SPVAL_DEFAULT,
) -> SoilMoistureData:
    """Read volumetric soil water from CLM netCDF history file.
    
    This function reads the H2OSOI variable from a CLM history file for a specific
    time slice. The data structure depends on the CLM physics version:
        - CLM4.5: Uses nlevgrnd vertical levels (typically 15)
        - CLM5.0: Uses nlevsoi vertical levels (typically 20)
    
    NetCDF dimensions are in row-major order (time, level, lat, lon) while
    the returned arrays follow Fortran column-major convention.
    
    WARNING: This function performs I/O and is NOT JIT-compatible. It should
    be called outside of JIT-compiled code to pre-load data.
    
    Args:
        ncfilename: Path to netCDF file
        strt: Time slice index to retrieve (1-based indexing)
        nlevgrnd: Number of ground layers for CLM4.5 (typically 15)
        nlevsoi: Number of soil layers for CLM5.0 (typically 20)
        clm_phys: CLM physics version ('CLM4_5' or 'CLM5_0')
        spval: Special value for missing data (default: 1.0e36)
        
    Returns:
        SoilMoistureData containing:
            - h2osoi_clm45: Soil moisture for CLM4.5 [m3/m3] [1, 1, nlevgrnd]
            - h2osoi_clm50: Soil moisture for CLM5.0 [m3/m3] [1, 1, nlevsoi]
            
    Raises:
        RuntimeError: If file cannot be opened or variable not found
        ValueError: If unknown CLM physics version specified
            
    Note:
        Reference: clmDataMod.F90 lines 209-278
        
        The function initializes both output arrays to spval, then populates
        only the array corresponding to the active CLM physics version.
        
        NetCDF indexing is 0-based, so strt-1 is used for the time dimension.
    """
    # Import netCDF4 here to avoid dependency in JIT-compiled code
    try:
        import netCDF4 as nc
    except ImportError:
        raise ImportError("netCDF4 package required for reading CLM soil data")
    
    # Initialize arrays to special value (lines 231-232)
    h2osoi_clm45 = jnp.full((1, 1, nlevgrnd), spval, dtype=jnp.float64)
    h2osoi_clm50 = jnp.full((1, 1, nlevsoi), spval, dtype=jnp.float64)
    
    # Open netCDF file (lines 234-236)
    try:
        dataset = nc.Dataset(ncfilename, 'r')
    except Exception as e:
        raise RuntimeError(f"Error opening netCDF file {ncfilename}: {str(e)}")
    
    try:
        # NetCDF uses 0-based indexing, Fortran uses 1-based
        # start3 = (/ 1, 1, strt /) in Fortran becomes (strt-1, 0, 0) in Python
        # due to reversed dimension order (lines 238-242)
        time_idx = strt - 1  # Convert from 1-based to 0-based
        
        if clm_phys == CLM45_VERSION:
            # Read h2osoi for CLM4.5 (lines 244-253)
            # NetCDF dimensions: (time, levgrnd, lat, lon)
            # We want: (lon, lat, levgrnd) = (1, 1, nlevgrnd)
            
            if 'H2OSOI' not in dataset.variables:
                raise RuntimeError(f"Variable H2OSOI not found in {ncfilename}")
            
            h2osoi_var = dataset.variables['H2OSOI']
            
            # Read data: time_idx, all levels, lat=0, lon=0
            # Result shape: (nlevgrnd,)
            h2osoi_data = h2osoi_var[time_idx, :nlevgrnd, 0, 0]
            
            # Reshape to (1, 1, nlevgrnd) to match Fortran convention
            h2osoi_clm45 = jnp.array(h2osoi_data, dtype=jnp.float64).reshape(1, 1, nlevgrnd)
            
        elif clm_phys == CLM50_VERSION:
            # Read h2osoi for CLM5.0 (lines 255-264)
            # NetCDF dimensions: (time, levsoi, lat, lon)
            # We want: (lon, lat, levsoi) = (1, 1, nlevsoi)
            
            if 'H2OSOI' not in dataset.variables:
                raise RuntimeError(f"Variable H2OSOI not found in {ncfilename}")
            
            h2osoi_var = dataset.variables['H2OSOI']
            
            # Read data: time_idx, all levels, lat=0, lon=0
            # Result shape: (nlevsoi,)
            h2osoi_data = h2osoi_var[time_idx, :nlevsoi, 0, 0]
            
            # Reshape to (1, 1, nlevsoi) to match Fortran convention
            h2osoi_clm50 = jnp.array(h2osoi_data, dtype=jnp.float64).reshape(1, 1, nlevsoi)
        
        else:
            raise ValueError(f"Unknown CLM physics version: {clm_phys}. "
                           f"Expected '{CLM45_VERSION}' or '{CLM50_VERSION}'")
            
    finally:
        # Close file (line 268)
        dataset.close()
    
    return SoilMoistureData(
        h2osoi_clm45=h2osoi_clm45,
        h2osoi_clm50=h2osoi_clm50,
    )


# =============================================================================
# Utility Functions
# =============================================================================


def create_clm_data_inputs(
    dz: jnp.ndarray,
    nbedrock: jnp.ndarray,
    watsat: jnp.ndarray,
    data_slice: CLMDataSlice,
    soil_moisture: SoilMoistureData,
    clm_phys: str,
    nlevgrnd: int = NLEVGRND_DEFAULT,
    nlevsoi: int = NLEVSOI_DEFAULT,
    denh2o: float = DENH2O_DEFAULT,
) -> CLMDataInputs:
    """Create CLMDataInputs from component data.
    
    Convenience function to assemble input data structure from individual
    components. This helps organize data flow from NetCDF reading to
    physics calculations.
    
    Args:
        dz: Soil layer thickness [m] [n_columns, nlevgrnd]
        nbedrock: Depth to bedrock index [n_columns]
        watsat: Soil saturation [m3/m3] [n_columns, nlevgrnd]
        data_slice: Single-level variables (LAI, SAI, coszen)
        soil_moisture: Soil moisture data for both CLM versions
        clm_phys: CLM physics version string
        nlevgrnd: Number of ground layers (default: 15)
        nlevsoi: Number of soil layers (default: 20)
        denh2o: Water density [kg/m3] (default: 1000.0)
        
    Returns:
        Complete CLMDataInputs structure ready for clm_data function
    """
    # Extract soil moisture for appropriate version
    if clm_phys == CLM45_VERSION:
        h2osoi_clm45 = soil_moisture.h2osoi_clm45[0, 0, :]  # Extract 1D array
        h2osoi_clm50 = jnp.zeros(nlevsoi)  # Placeholder
    elif clm_phys == CLM50_VERSION:
        h2osoi_clm45 = jnp.zeros(nlevgrnd)  # Placeholder
        h2osoi_clm50 = soil_moisture.h2osoi_clm50[0, 0, :]  # Extract 1D array
    else:
        raise ValueError(f"Unknown CLM physics version: {clm_phys}")
    
    return CLMDataInputs(
        dz=dz,
        nbedrock=nbedrock,
        watsat=watsat,
        elai_loc=float(data_slice.elai),
        esai_loc=float(data_slice.esai),
        coszen_loc=float(data_slice.coszen),
        h2osoi_clm45=h2osoi_clm45,
        h2osoi_clm50=h2osoi_clm50,
        clm_phys=clm_phys,
        nlevgrnd=nlevgrnd,
        nlevsoi=nlevsoi,
        denh2o=denh2o,
    )


def get_default_soil_properties(
    n_columns: int,
    nlevgrnd: int = NLEVGRND_DEFAULT,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Get default soil properties for testing.
    
    Provides reasonable default values for soil properties when actual
    data is not available. Useful for testing and initialization.
    
    Args:
        n_columns: Number of columns
        nlevgrnd: Number of ground layers (default: 15)
        
    Returns:
        Tuple of (dz, nbedrock, watsat):
            - dz: Layer thickness [m] [n_columns, nlevgrnd]
            - nbedrock: Bedrock depth index [n_columns]
            - watsat: Saturation [m3/m3] [n_columns, nlevgrnd]
    """
    # Default layer thicknesses (exponentially increasing with depth)
    dz_profile = jnp.array([
        0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20, 0.24,
        0.28, 0.32, 0.36, 0.40, 0.44, 0.54, 0.64
    ])
    dz = jnp.tile(dz_profile, (n_columns, 1))
    
    # Default bedrock at bottom layer
    nbedrock = jnp.full(n_columns, nlevgrnd)
    
    # Default saturation (typical loam soil)
    watsat = jnp.full((n_columns, nlevgrnd), 0.45)
    
    return dz, nbedrock, watsat