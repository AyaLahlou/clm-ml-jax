"""
Tower Data Module - Flux Tower Site Parameters.

Translated from CTSM's TowerDataMod.F90 (lines 1-100)

This module contains parameters for flux tower sites used in offline simulations.
Each tower site has associated metadata including location, vegetation type,
soil properties, and measurement characteristics.

The data is stored as immutable arrays that can be indexed by tower site number.
Tower sites are used for single-point offline simulations to validate model
physics against eddy covariance flux measurements.

Key tower parameters:
    - Location: Latitude/longitude coordinates
    - Vegetation: CLM PFT (Plant Functional Type) index
    - Soil: Texture class, sand/clay percentages, organic matter, color
    - Structure: Tower height, canopy height, bedrock depth
    - Forcing: Time step of meteorological forcing data

Usage:
    >>> tower_data = create_tower_data()
    >>> lat = tower_data.tower_lat[0]  # Get US-Ha1 latitude
    >>> pft = get_tower_parameter(tower_data, 3, 'tower_pft')  # Get US-UMB PFT
"""

from typing import NamedTuple, Dict, List
import jax.numpy as jnp


# =============================================================================
# Constants and Parameters
# =============================================================================

# Number of tower sites (line 16)
NTOWER: int = 15

# Tower site name mapping (lines 36-38)
# Encoded as integers for JAX compatibility
# Index mapping:
#   0: US-Ha1 (Harvard Forest, MA - deciduous broadleaf)
#   1: US-Ho1 (Howland Forest, ME - evergreen needleleaf)
#   2: US-MMS (Morgan Monroe State Forest, IN - deciduous broadleaf)
#   3: US-UMB (UMBS, MI - deciduous broadleaf)
#   4: US-Dk3 (Duke Forest, NC - evergreen needleleaf)
#   5: US-Me2 (Metolius, OR - evergreen needleleaf)
#   6: US-Var (Vaira Ranch, CA - grassland)
#   7: US-IB1 (Fermi Lab, IL - grassland)
#   8: US-Ne3 (Mead, NE - cropland)
#   9: US-ARM (ARM SGP, OK - cropland)
#  10: US-Bo1 (Bondville, IL - cropland)
#  11: US-Dk1 (Duke Forest, NC - grassland)
#  12: US-Dk2 (Duke Forest, NC - deciduous broadleaf)
#  13: CHATS7 (CHATS, CA - grassland)
#  14: UMBSmw (UMBS, MI - deciduous broadleaf)
TOWER_ID_NAMES: List[str] = [
    'US-Ha1', 'US-Ho1', 'US-MMS', 'US-UMB', 'US-Dk3', 'US-Me2',
    'US-Var', 'US-IB1', 'US-Ne3', 'US-ARM', 'US-Bo1', 'US-Dk1',
    'US-Dk2', 'CHATS7', 'UMBSmw'
]

# Soil texture class mapping (lines 50-56)
# Index mapping:
#   0: loam
#   1: sandy loam
#   2: clay
#   3: sand
#   4: silty loam
#   5: silty clay loam
#   6: clay loam
TOWER_TEX_NAMES: List[str] = [
    'loam', 'sandy loam', 'clay', 'sand', 'silty loam',
    'silty clay loam', 'clay loam'
]

# Missing value indicator for tower height (line 83)
MISSING_TOWER_HEIGHT: float = -999.0


# =============================================================================
# Type Definitions
# =============================================================================

class TowerData(NamedTuple):
    """Immutable container for flux tower site parameters.
    
    All arrays have shape [ntower] where ntower=15.
    
    Attributes:
        tower_id: Tower site names (encoded as integers for JAX compatibility)
                  [dimensionless] [ntower]
        tower_lat: Latitude of tower [degrees North] [ntower]
        tower_lon: Longitude of tower [degrees East] [ntower]
        tower_pft: CLM PFT (Plant Functional Type) index [dimensionless] [ntower]
        tower_tex: Soil texture class (encoded as integers) [dimensionless] [ntower]
        tower_sand: Percent sand (negative if not specified) [%] [ntower]
        tower_clay: Percent clay (negative if not specified) [%] [ntower]
        tower_organic: Soil organic matter content [kg/m3] [ntower]
        tower_isoicol: CLM soil color class [dimensionless] [ntower]
        tower_zbed: Depth to bedrock [m] [ntower]
        tower_ht: Flux tower measurement height [m] [ntower]
        tower_canht: Canopy height [m] [ntower]
        tower_time: Time step of forcing data [minutes] [ntower]
        
    Note:
        Negative values in tower_sand and tower_clay indicate that the
        tower_tex texture class should be used instead of explicit percentages.
    """
    tower_id: jnp.ndarray  # Shape: [15], dtype: int32
    tower_lat: jnp.ndarray  # Shape: [15], dtype: float64
    tower_lon: jnp.ndarray  # Shape: [15], dtype: float64
    tower_pft: jnp.ndarray  # Shape: [15], dtype: int32
    tower_tex: jnp.ndarray  # Shape: [15], dtype: int32
    tower_sand: jnp.ndarray  # Shape: [15], dtype: float64
    tower_clay: jnp.ndarray  # Shape: [15], dtype: float64
    tower_organic: jnp.ndarray  # Shape: [15], dtype: float64
    tower_isoicol: jnp.ndarray  # Shape: [15], dtype: int32
    tower_zbed: jnp.ndarray  # Shape: [15], dtype: float64
    tower_ht: jnp.ndarray  # Shape: [15], dtype: float64
    tower_canht: jnp.ndarray  # Shape: [15], dtype: float64
    tower_time: jnp.ndarray  # Shape: [15], dtype: int32


# =============================================================================
# Data Initialization Functions
# =============================================================================

def create_tower_data() -> TowerData:
    """Create immutable tower data structure with all site parameters.
    
    Initializes the complete set of flux tower site parameters from the
    original CTSM TowerDataMod.F90 data statements. All values are hardcoded
    to match the Fortran source exactly.
    
    Returns:
        TowerData: Immutable NamedTuple containing all tower site parameters
                   for 15 flux tower sites
        
    Note:
        Data values are from TowerDataMod.F90 lines 36-98.
        Negative sand/clay values (-1.0) indicate texture class should be
        used instead of explicit percentages.
        
    Reference:
        TowerDataMod.F90 lines 1-100
    """
    # Tower site IDs (lines 36-38) - encoded as indices
    tower_id = jnp.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        dtype=jnp.int32
    )
    
    # Latitude (lines 40-43) [degrees North]
    tower_lat = jnp.array([
        42.54,  # US-Ha1: Harvard Forest, MA
        45.20,  # US-Ho1: Howland Forest, ME
        39.32,  # US-MMS: Morgan Monroe, IN
        45.56,  # US-UMB: UMBS, MI
        35.98,  # US-Dk3: Duke Forest, NC
        44.45,  # US-Me2: Metolius, OR
        38.41,  # US-Var: Vaira Ranch, CA
        41.86,  # US-IB1: Fermi Lab, IL
        41.18,  # US-Ne3: Mead, NE
        36.61,  # US-ARM: ARM SGP, OK
        40.01,  # US-Bo1: Bondville, IL
        35.97,  # US-Dk1: Duke Forest, NC
        35.97,  # US-Dk2: Duke Forest, NC
        38.49,  # CHATS7: CHATS, CA
        45.56   # UMBSmw: UMBS, MI
    ], dtype=jnp.float64)
    
    # Longitude (lines 45-47) [degrees East]
    tower_lon = jnp.array([
        -72.17,   # US-Ha1
        -68.74,   # US-Ho1
        -86.41,   # US-MMS
        -84.71,   # US-UMB
        -79.09,   # US-Dk3
        -121.56,  # US-Me2
        -120.95,  # US-Var
        -88.22,   # US-IB1
        -96.44,   # US-Ne3
        -97.49,   # US-ARM
        -88.29,   # US-Bo1
        -79.09,   # US-Dk1
        -79.10,   # US-Dk2
        -121.84,  # CHATS7
        -84.71    # UMBSmw
    ], dtype=jnp.float64)
    
    # CLM PFT (Plant Functional Type) index (line 51)
    # PFT mapping:
    #   1: Evergreen needleleaf
    #   2: Evergreen broadleaf
    #   7: Deciduous broadleaf
    #  13: C3 grass
    #  15: Crop
    tower_pft = jnp.array(
        [7, 2, 7, 7, 1, 2, 13, 15, 15, 15, 15, 13, 7, 7, 7],
        dtype=jnp.int32
    )
    
    # Soil texture class (lines 57-60) - encoded as indices
    # 0: loam, 1: sandy loam, 2: clay, 3: sand, 4: silty loam,
    # 5: silty clay loam, 6: clay loam
    tower_tex = jnp.array(
        [0, 1, 2, 3, 1, 1, 4, 5, 6, 2, 4, 1, 1, 5, 3],
        dtype=jnp.int32
    )
    
    # Percent sand (lines 62-64) [%]
    # Negative values indicate texture class should be used
    tower_sand = jnp.array([
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        -1.0, 10.0, -1.0  # Only CHATS7 has explicit sand percentage
    ], dtype=jnp.float64)
    
    # Percent clay (lines 66-68) [%]
    # Negative values indicate texture class should be used
    tower_clay = jnp.array([
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        -1.0, 35.0, -1.0  # Only CHATS7 has explicit clay percentage
    ], dtype=jnp.float64)
    
    # Soil organic matter (lines 70-73) [kg/m3]
    tower_organic = jnp.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 50.0, 0.0  # Only CHATS7 has organic matter specified
    ], dtype=jnp.float64)
    
    # CLM soil color class (line 77) [dimensionless]
    # Range: 1-20, where lower numbers are lighter colors
    tower_isoicol = jnp.array(
        [18, 16, 15, 17, 15, 20, 17, 15, 13, 13, 15, 15, 15, 1, 17],
        dtype=jnp.int32
    )
    
    # Depth to bedrock (lines 79-81) [m]
    tower_zbed = jnp.array([
        50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
        50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
        2.0,   # CHATS7 has shallow bedrock
        50.0
    ], dtype=jnp.float64)
    
    # Flux tower measurement height (lines 83-86) [m]
    # -999.0 indicates missing value
    tower_ht = jnp.array([
        30.0,   # US-Ha1
        29.0,   # US-Ho1
        48.0,   # US-MMS
        46.0,   # US-UMB
        22.0,   # US-Dk3
        32.0,   # US-Me2
        2.5,    # US-Var
        4.0,    # US-IB1
        6.0,    # US-Ne3
        -999.0, # US-ARM (missing)
        6.0,    # US-Bo1
        5.0,    # US-Dk1
        42.0,   # US-Dk2
        23.0,   # CHATS7
        46.0    # UMBSmw
    ], dtype=jnp.float64)
    
    # Canopy height (lines 88-91) [m]
    tower_canht = jnp.array([
        23.0,  # US-Ha1: tall forest
        20.0,  # US-Ho1: tall forest
        27.0,  # US-MMS: tall forest
        21.0,  # US-UMB: tall forest
        17.0,  # US-Dk3: forest
        14.0,  # US-Me2: forest
        0.6,   # US-Var: grassland
        0.9,   # US-IB1: grassland
        0.9,   # US-Ne3: crop
        0.5,   # US-ARM: crop
        0.9,   # US-Bo1: crop
        0.5,   # US-Dk1: grassland
        25.0,  # US-Dk2: tall forest
        10.0,  # CHATS7: grassland
        21.0   # UMBSmw: tall forest
    ], dtype=jnp.float64)
    
    # Time step of forcing data (line 95) [minutes]
    tower_time = jnp.array(
        [60, 30, 60, 60, 30, 30, 30, 30, 60, 30, 30, 30, 30, 30, 60],
        dtype=jnp.int32
    )
    
    return TowerData(
        tower_id=tower_id,
        tower_lat=tower_lat,
        tower_lon=tower_lon,
        tower_pft=tower_pft,
        tower_tex=tower_tex,
        tower_sand=tower_sand,
        tower_clay=tower_clay,
        tower_organic=tower_organic,
        tower_isoicol=tower_isoicol,
        tower_zbed=tower_zbed,
        tower_ht=tower_ht,
        tower_canht=tower_canht,
        tower_time=tower_time,
    )


# =============================================================================
# Accessor Functions
# =============================================================================

def get_tower_parameter(
    tower_data: TowerData,
    tower_num: int,
    parameter: str
) -> jnp.ndarray:
    """Get a specific parameter for a given tower site.
    
    Convenience function for accessing individual tower parameters by name.
    For vectorized operations over multiple towers, access the arrays
    directly from the TowerData structure.
    
    Args:
        tower_data: TowerData structure containing all tower parameters
        tower_num: Tower site index [0 to 14]
        parameter: Name of parameter to retrieve (e.g., 'tower_lat', 'tower_pft')
        
    Returns:
        Parameter value for the specified tower site (scalar or array element)
        
    Raises:
        AttributeError: If parameter name is not valid
        IndexError: If tower_num is out of range [0, 14]
        
    Example:
        >>> tower_data = create_tower_data()
        >>> lat = get_tower_parameter(tower_data, 0, 'tower_lat')
        >>> print(f"US-Ha1 latitude: {lat}")
        US-Ha1 latitude: 42.54
        
    Note:
        This is a convenience function for accessing individual tower parameters.
        For vectorized operations, access the arrays directly from tower_data.
        
    Reference:
        TowerDataMod.F90 lines 1-100
    """
    param_array = getattr(tower_data, parameter)
    return param_array[tower_num]


def get_tower_name(tower_num: int) -> str:
    """Get the name of a tower site from its index.
    
    Args:
        tower_num: Tower site index [0 to 14]
        
    Returns:
        Tower site name string
        
    Raises:
        IndexError: If tower_num is out of range [0, 14]
        
    Example:
        >>> name = get_tower_name(0)
        >>> print(name)
        US-Ha1
    """
    return TOWER_ID_NAMES[tower_num]


def get_texture_name(texture_num: int) -> str:
    """Get the name of a soil texture class from its index.
    
    Args:
        texture_num: Texture class index [0 to 6]
        
    Returns:
        Texture class name string
        
    Raises:
        IndexError: If texture_num is out of range [0, 6]
        
    Example:
        >>> texture = get_texture_name(0)
        >>> print(texture)
        loam
    """
    return TOWER_TEX_NAMES[texture_num]


def get_tower_metadata(tower_data: TowerData, tower_num: int) -> Dict[str, any]:
    """Get all metadata for a tower site as a dictionary.
    
    Convenience function that returns all tower parameters in a
    human-readable dictionary format with proper units.
    
    Args:
        tower_data: TowerData structure containing all tower parameters
        tower_num: Tower site index [0 to 14]
        
    Returns:
        Dictionary containing all tower parameters with descriptive keys
        
    Example:
        >>> tower_data = create_tower_data()
        >>> metadata = get_tower_metadata(tower_data, 0)
        >>> print(metadata['name'])
        US-Ha1
        >>> print(f"{metadata['latitude']:.2f}°N, {metadata['longitude']:.2f}°E")
        42.54°N, -72.17°E
    """
    return {
        'name': get_tower_name(tower_num),
        'latitude': float(tower_data.tower_lat[tower_num]),
        'longitude': float(tower_data.tower_lon[tower_num]),
        'pft': int(tower_data.tower_pft[tower_num]),
        'texture_class': get_texture_name(int(tower_data.tower_tex[tower_num])),
        'sand_percent': float(tower_data.tower_sand[tower_num]),
        'clay_percent': float(tower_data.tower_clay[tower_num]),
        'organic_matter_kg_m3': float(tower_data.tower_organic[tower_num]),
        'soil_color': int(tower_data.tower_isoicol[tower_num]),
        'bedrock_depth_m': float(tower_data.tower_zbed[tower_num]),
        'tower_height_m': float(tower_data.tower_ht[tower_num]),
        'canopy_height_m': float(tower_data.tower_canht[tower_num]),
        'forcing_timestep_min': int(tower_data.tower_time[tower_num]),
    }