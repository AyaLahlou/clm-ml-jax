"""
Vertical structure initialization module for CLM.

This module handles the initialization of vertical components of column datatype,
including soil layer structure, depths, and bedrock configuration.

Translation of CLM-ml_v1/clm_src_main/initVerticalMod.F90 to Python/JAX.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Dict, Any, Union
import logging
from dataclasses import dataclass, field
from enum import Enum

# Import related modules (these would need to be implemented)
try:
    from .decompMod import BoundsType
    from .ColumnType import col
    from .abortutils import endrun, CLMError
    from .clm_varpar import nlevsoi, nlevgrnd
    from .clm_varcon import zmin_bedrock
    from ..offline_driver.TowerDataMod import tower_num, tower_zbed
    # Assuming MLclm_varctl provides physics version control
    clm_phys = "CLM5_0"  # Default physics version
    # Alias for backward compatibility
    bounds_type = BoundsType
except ImportError:
    # Provide fallback implementations for testing
    @dataclass
    class BoundsType:
        begc: int = 0
        endc: int = 0
    
    bounds_type = BoundsType
    
    @dataclass
    class ColType:
        dz: jnp.ndarray = field(default_factory=lambda: jnp.zeros((1, 50)))
        z: jnp.ndarray = field(default_factory=lambda: jnp.zeros((1, 50)))
        zi: jnp.ndarray = field(default_factory=lambda: jnp.zeros((1, 51)))
        nbedrock: jnp.ndarray = field(default_factory=lambda: jnp.zeros(1, dtype=int))
    
    col = ColType()
    
    def endrun(msg: str = ""):
        raise RuntimeError(msg)
    
    CLMError = RuntimeError
    
    # Default parameters
    nlevsoi = 20
    nlevgrnd = 50
    zmin_bedrock = 3.0
    tower_num = 0
    tower_zbed = jnp.array([10.0])  # Default bedrock depth
    clm_phys = "CLM5_0"

# Set up logger
logger = logging.getLogger(__name__)


class CLMPhysicsVersion(Enum):
    """Enumeration of supported CLM physics versions."""
    CLM4_5 = "CLM4_5"
    CLM5_0 = "CLM5_0"


@dataclass
class VerticalStructure:
    """
    Manages vertical structure configuration and calculations.
    
    This class handles the vertical discretization of soil and bedrock layers,
    including layer thicknesses, depths, and interface positions.
    
    Attributes:
        physics_version: CLM physics version (CLM4_5 or CLM5_0)
        scalez: Soil layer thickness discretization parameter (m)
        num_soil_layers: Number of soil layers
        num_ground_layers: Total number of ground layers (soil + bedrock)
        min_bedrock_depth: Minimum depth for bedrock (m)
        layer_depths: Array of layer center depths (m)
        layer_thicknesses: Array of layer thicknesses (m)
        interface_depths: Array of layer interface depths (m)
        bedrock_indices: Array of bedrock layer indices for each column
        metadata: Additional configuration metadata
    """
    physics_version: CLMPhysicsVersion = CLMPhysicsVersion.CLM5_0
    scalez: float = 0.025
    num_soil_layers: int = 20
    num_ground_layers: int = 50
    min_bedrock_depth: float = 3.0
    layer_depths: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    layer_thicknesses: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    interface_depths: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    bedrock_indices: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if vertical structure is valid."""
        # Check that arrays have consistent dimensions
        expected_shape = (1, self.num_ground_layers)  # Assuming single column for now
        
        if len(self.layer_depths.shape) == 2:
            if self.layer_depths.shape != expected_shape:
                return False
            if self.layer_thicknesses.shape != expected_shape:
                return False
            if self.interface_depths.shape != (expected_shape[0], expected_shape[1] + 1):
                return False
        
        # Check physics version
        if self.physics_version not in CLMPhysicsVersion:
            return False
        
        # Check parameter ranges
        if self.scalez <= 0 or self.min_bedrock_depth <= 0:
            return False
        
        return True
    
    def get_layer_info(self, layer_idx: int) -> Dict[str, float]:
        """Get information about a specific layer."""
        if layer_idx < 0 or layer_idx >= self.num_ground_layers:
            raise ValueError(f"Invalid layer index: {layer_idx}")
        
        if len(self.layer_depths.shape) == 2:
            return {
                'layer_index': layer_idx,
                'depth': float(self.layer_depths[0, layer_idx]),
                'thickness': float(self.layer_thicknesses[0, layer_idx]),
                'interface_top': float(self.interface_depths[0, layer_idx]),
                'interface_bottom': float(self.interface_depths[0, layer_idx + 1]),
                'is_soil': layer_idx < self.num_soil_layers,
                'is_bedrock': layer_idx >= self.num_soil_layers
            }
        
        return {'layer_index': layer_idx, 'error': 'Arrays not initialized'}


# Global vertical structure
_vertical_structure = VerticalStructure()


def initVertical(bounds: bounds_type) -> None:
    """
    Initialize vertical components of column datatype.
    
    This function sets up the soil layer structure including layer depths,
    thicknesses, interface positions, and bedrock configuration based on
    the specified CLM physics version.
    
    Args:
        bounds: Bounds type containing column index ranges
        
    Raises:
        CLMError: If physics version is invalid or initialization fails
        ValueError: If bounds are invalid
    """
    global _vertical_structure, col
    
    try:
        logger.info(f"Initializing vertical structure with physics version {clm_phys}")
        
        # Validate bounds
        if bounds.begc < 0 or bounds.endc < bounds.begc:
            raise ValueError(f"Invalid bounds: begc={bounds.begc}, endc={bounds.endc}")
        
        # Update global structure
        _vertical_structure.physics_version = CLMPhysicsVersion(clm_phys)
        _vertical_structure.num_soil_layers = nlevsoi
        _vertical_structure.num_ground_layers = nlevgrnd
        _vertical_structure.min_bedrock_depth = zmin_bedrock
        
        # Get column range
        begc, endc = bounds.begc, bounds.endc
        num_columns = endc - begc + 1
        
        # Initialize arrays if needed
        if hasattr(col, 'dz') and col.dz.shape[0] < num_columns:
            # Resize column arrays
            col.dz = jnp.zeros((num_columns, nlevgrnd))
            col.z = jnp.zeros((num_columns, nlevgrnd))
            col.zi = jnp.zeros((num_columns, nlevgrnd + 1))
            col.nbedrock = jnp.zeros(num_columns, dtype=int)
        
        # Define CLM layer structure for soil
        for c in range(begc, endc + 1):
            col_idx = c - begc  # Adjust for 0-based indexing
            
            if clm_phys == "CLM4_5":
                _init_clm45_layers(col_idx)
            elif clm_phys == "CLM5_0":
                _init_clm50_layers(col_idx)
            else:
                endrun(f"ERROR: initVertical: clm_phys '{clm_phys}' not valid")
        
        # Set column bedrock indices
        _set_bedrock_indices(begc, endc)
        
        # Update global structure arrays
        _vertical_structure.layer_depths = col.z
        _vertical_structure.layer_thicknesses = col.dz
        _vertical_structure.interface_depths = col.zi
        _vertical_structure.bedrock_indices = col.nbedrock
        
        # Store metadata
        _vertical_structure.metadata.update({
            'initialization_bounds': {'begc': begc, 'endc': endc},
            'num_columns': num_columns,
            'physics_version': clm_phys,
            'tower_bedrock_depth': float(tower_zbed[tower_num])
        })
        
        logger.info("Vertical structure initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Vertical structure initialization failed: {e}")
        raise CLMError(f"Failed to initialize vertical structure: {e}") from e


@jax.jit
def _calculate_clm45_depths(scalez: float, nlevgrnd: int) -> jnp.ndarray:
    """
    Calculate layer depths for CLM4.5 physics.
    
    Args:
        scalez: Scaling parameter for layer thickness
        nlevgrnd: Number of ground layers
        
    Returns:
        Array of layer depths
    """
    j_array = jnp.arange(1, nlevgrnd + 1)
    depths = scalez * (jnp.exp(0.5 * (j_array - 0.5)) - 1.0)
    return depths


@jax.jit
def _calculate_clm45_thicknesses(depths: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate layer thicknesses for CLM4.5 physics.
    
    Args:
        depths: Array of layer depths
        
    Returns:
        Array of layer thicknesses
    """
    nlevgrnd = len(depths)
    thicknesses = jnp.zeros(nlevgrnd)
    
    # First layer
    thicknesses = thicknesses.at[0].set(0.5 * (depths[0] + depths[1]))
    
    # Middle layers
    for j in range(1, nlevgrnd - 1):
        thicknesses = thicknesses.at[j].set(0.5 * (depths[j + 1] - depths[j - 1]))
    
    # Last layer
    thicknesses = thicknesses.at[nlevgrnd - 1].set(depths[nlevgrnd - 1] - depths[nlevgrnd - 2])
    
    return thicknesses


@jax.jit
def _calculate_clm45_interfaces(depths: jnp.ndarray, thicknesses: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate interface depths for CLM4.5 physics.
    
    Args:
        depths: Array of layer depths
        thicknesses: Array of layer thicknesses
        
    Returns:
        Array of interface depths
    """
    nlevgrnd = len(depths)
    interfaces = jnp.zeros(nlevgrnd + 1)
    
    # Surface interface
    interfaces = interfaces.at[0].set(0.0)
    
    # Middle interfaces
    for j in range(1, nlevgrnd):
        interfaces = interfaces.at[j].set(0.5 * (depths[j - 1] + depths[j]))
    
    # Bottom interface
    interfaces = interfaces.at[nlevgrnd].set(depths[nlevgrnd - 1] + 0.5 * thicknesses[nlevgrnd - 1])
    
    return interfaces


def _init_clm45_layers(col_idx: int) -> None:
    """
    Initialize CLM4.5 layer structure for a column.
    
    Args:
        col_idx: Column index (0-based)
    """
    global col, _vertical_structure
    
    # Calculate layer depths
    depths = _calculate_clm45_depths(_vertical_structure.scalez, nlevgrnd)
    col.z = col.z.at[col_idx, :].set(depths)
    
    # Calculate layer thicknesses
    thicknesses = _calculate_clm45_thicknesses(depths)
    col.dz = col.dz.at[col_idx, :].set(thicknesses)
    
    # Calculate interface depths
    interfaces = _calculate_clm45_interfaces(depths, thicknesses)
    col.zi = col.zi.at[col_idx, :].set(interfaces)


@jax.jit
def _calculate_clm50_thicknesses(nlevsoi: int, nlevgrnd: int) -> jnp.ndarray:
    """
    Calculate layer thicknesses for CLM5.0 physics.
    
    Args:
        nlevsoi: Number of soil layers
        nlevgrnd: Total number of ground layers
        
    Returns:
        Array of layer thicknesses
    """
    thicknesses = jnp.zeros(nlevgrnd)
    
    # Layers 1-4: increasing by 0.02 m
    for j in range(4):
        thicknesses = thicknesses.at[j].set((j + 1) * 0.02)
    
    # Layers 5-13: based on layer 4 + increments
    dz_4 = thicknesses[3]  # 4 * 0.02 = 0.08
    for j in range(4, 13):
        thicknesses = thicknesses.at[j].set(dz_4 + (j - 3) * 0.04)
    
    # Layers 14 to nlevsoi: based on layer 13 + increments
    dz_13 = thicknesses[12]
    for j in range(13, nlevsoi):
        thicknesses = thicknesses.at[j].set(dz_13 + (j - 12) * 0.10)
    
    # Bedrock layers: nlevsoi+1 to nlevgrnd
    dz_soil_last = thicknesses[nlevsoi - 1]
    for j in range(nlevsoi, nlevgrnd):
        bedrock_factor = ((j - nlevsoi + 1) * 25.0) ** 1.5 / 100.0
        thicknesses = thicknesses.at[j].set(dz_soil_last + bedrock_factor)
    
    return thicknesses


@jax.jit
def _calculate_clm50_interfaces(thicknesses: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate interface depths for CLM5.0 physics.
    
    Args:
        thicknesses: Array of layer thicknesses
        
    Returns:
        Array of interface depths
    """
    nlevgrnd = len(thicknesses)
    interfaces = jnp.zeros(nlevgrnd + 1)
    
    # Surface interface
    interfaces = interfaces.at[0].set(0.0)
    
    # Cumulative sum for interfaces
    for j in range(1, nlevgrnd + 1):
        interfaces = interfaces.at[j].set(jnp.sum(thicknesses[:j]))
    
    return interfaces


@jax.jit
def _calculate_clm50_depths(interfaces: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate layer center depths for CLM5.0 physics.
    
    Args:
        interfaces: Array of interface depths
        
    Returns:
        Array of layer center depths
    """
    nlevgrnd = len(interfaces) - 1
    depths = jnp.zeros(nlevgrnd)
    
    for j in range(nlevgrnd):
        depths = depths.at[j].set(0.5 * (interfaces[j] + interfaces[j + 1]))
    
    return depths


def _init_clm50_layers(col_idx: int) -> None:
    """
    Initialize CLM5.0 layer structure for a column.
    
    Args:
        col_idx: Column index (0-based)
    """
    global col
    
    # Calculate layer thicknesses
    thicknesses = _calculate_clm50_thicknesses(nlevsoi, nlevgrnd)
    col.dz = col.dz.at[col_idx, :].set(thicknesses)
    
    # Calculate interface depths
    interfaces = _calculate_clm50_interfaces(thicknesses)
    col.zi = col.zi.at[col_idx, :].set(interfaces)
    
    # Calculate layer center depths
    depths = _calculate_clm50_depths(interfaces)
    col.z = col.z.at[col_idx, :].set(depths)


def _set_bedrock_indices(begc: int, endc: int) -> None:
    """
    Set bedrock indices for all columns.
    
    Args:
        begc: Beginning column index
        endc: Ending column index
    """
    global col, _vertical_structure
    
    # Get depth to bedrock for the tower site
    zbedrock = float(tower_zbed[tower_num])
    
    logger.debug(f"Setting bedrock indices with zbedrock={zbedrock}")
    
    for c in range(begc, endc + 1):
        col_idx = c - begc
        
        # Determine minimum index for minimum soil depth
        jmin_bedrock = _find_minimum_bedrock_index(col_idx)
        
        # Determine actual bedrock index
        nbedrock_val = _find_bedrock_index(col_idx, zbedrock, jmin_bedrock)
        
        col.nbedrock = col.nbedrock.at[col_idx].set(nbedrock_val)
        
        logger.debug(f"Column {c}: bedrock index = {nbedrock_val}")


def _find_minimum_bedrock_index(col_idx: int) -> int:
    """
    Find minimum bedrock index based on minimum soil depth.
    
    Args:
        col_idx: Column index
        
    Returns:
        Minimum bedrock index
    """
    global col
    
    jmin_bedrock = 3  # Default minimum (1-based, so layer 2 in 0-based)
    
    for j in range(2, nlevsoi):  # Start from layer 2 (0-based)
        zi_prev = col.zi[col_idx, j]
        zi_curr = col.zi[col_idx, j + 1]
        
        if zi_prev < zmin_bedrock <= zi_curr:
            jmin_bedrock = j + 1  # Convert to 1-based for compatibility
            break
    
    return jmin_bedrock


def _find_bedrock_index(col_idx: int, zbedrock: float, jmin_bedrock: int) -> int:
    """
    Find bedrock index based on bedrock depth.
    
    Args:
        col_idx: Column index
        zbedrock: Depth to bedrock
        jmin_bedrock: Minimum bedrock index
        
    Returns:
        Bedrock index
    """
    global col
    
    nbedrock_val = nlevsoi  # Default to bottom of soil layers
    
    # Convert jmin_bedrock to 0-based for array indexing
    jmin_0based = jmin_bedrock - 1
    
    for j in range(jmin_0based, nlevsoi):
        zi_prev = col.zi[col_idx, j]
        zi_curr = col.zi[col_idx, j + 1]
        
        if zi_prev < zbedrock <= zi_curr:
            nbedrock_val = j + 1  # Convert to 1-based
            break
    
    return nbedrock_val


def get_vertical_structure() -> VerticalStructure:
    """
    Get the current vertical structure.
    
    Returns:
        Copy of current vertical structure
    """
    global _vertical_structure
    return _vertical_structure


def reset_vertical_structure(physics_version: str = "CLM5_0") -> None:
    """
    Reset the vertical structure.
    
    Args:
        physics_version: CLM physics version to use
    """
    global _vertical_structure
    
    try:
        version_enum = CLMPhysicsVersion(physics_version)
        _vertical_structure = VerticalStructure(physics_version=version_enum)
        logger.info(f"Vertical structure reset with physics version {physics_version}")
    except ValueError as e:
        raise ValueError(f"Invalid physics version: {physics_version}") from e


@jax.jit
def calculate_layer_statistics(depths: jnp.ndarray, 
                             thicknesses: jnp.ndarray) -> Dict[str, float]:
    """
    Calculate statistics for layer structure.
    
    Args:
        depths: Array of layer depths
        thicknesses: Array of layer thicknesses
        
    Returns:
        Dictionary with layer statistics
    """
    if len(depths) == 0:
        return {
            'mean_depth': 0.0,
            'max_depth': 0.0,
            'min_thickness': 0.0,
            'max_thickness': 0.0,
            'total_depth': 0.0
        }
    
    return {
        'mean_depth': float(jnp.mean(depths)),
        'max_depth': float(jnp.max(depths)),
        'min_thickness': float(jnp.min(thicknesses)),
        'max_thickness': float(jnp.max(thicknesses)),
        'total_depth': float(jnp.sum(thicknesses))
    }


def print_vertical_summary(column_idx: int = 0) -> None:
    """
    Print a summary of the vertical structure.
    
    Args:
        column_idx: Column index to summarize (0-based)
    """
    global _vertical_structure, col
    
    print(f"\n=== Vertical Structure Summary (Column {column_idx}) ===")
    print(f"Physics version: {_vertical_structure.physics_version.value}")
    print(f"Soil layers: {_vertical_structure.num_soil_layers}")
    print(f"Ground layers: {_vertical_structure.num_ground_layers}")
    print(f"Min bedrock depth: {_vertical_structure.min_bedrock_depth} m")
    
    if hasattr(col, 'z') and len(col.z.shape) == 2 and col.z.shape[0] > column_idx:
        depths = col.z[column_idx, :]
        thicknesses = col.dz[column_idx, :]
        bedrock_idx = col.nbedrock[column_idx]
        
        stats = calculate_layer_statistics(depths, thicknesses)
        
        print(f"\nLayer Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f} m")
        
        print(f"\nBedrock index: {bedrock_idx}")
        print(f"Bedrock depth: {col.zi[column_idx, bedrock_idx-1]:.4f} m")
    
    if _vertical_structure.metadata:
        print(f"\nMetadata:")
        for key, value in _vertical_structure.metadata.items():
            print(f"  {key}: {value}")
    
    print("=" * 50)


def validate_vertical_structure() -> Tuple[bool, str]:
    """
    Validate the vertical structure consistency.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    global _vertical_structure, col
    
    if not _vertical_structure.is_valid():
        return False, "Vertical structure basic validation failed"
    
    # Check column arrays if available
    if hasattr(col, 'z') and len(col.z.shape) == 2:
        num_cols, num_layers = col.z.shape
        
        # Check that depths are increasing
        for c in range(num_cols):
            depths = col.z[c, :]
            if not jnp.all(depths[1:] >= depths[:-1]):
                return False, f"Layer depths not increasing in column {c}"
            
            # Check positive thicknesses
            thicknesses = col.dz[c, :]
            if not jnp.all(thicknesses > 0):
                return False, f"Non-positive layer thickness found in column {c}"
            
            # Check bedrock index
            bedrock_idx = col.nbedrock[c]
            if bedrock_idx < 1 or bedrock_idx > _vertical_structure.num_soil_layers:
                return False, f"Invalid bedrock index {bedrock_idx} in column {c}"
    
    return True, "Vertical structure is valid and consistent"


# Factory functions for common configurations
def create_simple_vertical_structure(physics_version: str = "CLM5_0",
                                   bedrock_depth: float = 10.0) -> None:
    """
    Create a simple single-column vertical structure.
    
    Args:
        physics_version: CLM physics version
        bedrock_depth: Depth to bedrock (m)
    """
    global tower_zbed, tower_num
    
    # Set tower bedrock depth
    tower_zbed = jnp.array([bedrock_depth])
    tower_num = 0
    
    # Create simple bounds
    bounds = bounds_type()
    bounds.begc = 0
    bounds.endc = 0
    
    # Initialize
    reset_vertical_structure(physics_version)
    initVertical(bounds)
    
    logger.info(f"Created simple vertical structure: {physics_version}, bedrock at {bedrock_depth} m")


# Export interface
__all__ = [
    'initVertical',
    'VerticalStructure', 
    'CLMPhysicsVersion',
    'get_vertical_structure',
    'reset_vertical_structure',
    'calculate_layer_statistics',
    'print_vertical_summary',
    'validate_vertical_structure',
    'create_simple_vertical_structure'
]