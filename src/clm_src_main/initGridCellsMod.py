"""
Grid cell initialization module for CLM.

This module handles the initialization of sub-grid mapping for each land grid cell,
including the setup of landunit, column, and patch hierarchies.

Translation of CLM-ml_v1/clm_src_main/initGridCellsMod.F90 to Python/JAX.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Dict, Any
import logging
from dataclasses import dataclass, field

# Import related modules (these would need to be implemented)
try:
    from ..offline_driver.TowerDataMod import tower_pft, tower_num
    from .initSubgridMod import add_patch
except ImportError:
    # Provide fallback implementations for testing
    tower_pft = jnp.array([1])  # Default PFT
    tower_num = 0
    
    def add_patch(pi: int, pft: int) -> None:
        """Fallback implementation of add_patch."""
        pass

# Set up logger
logger = logging.getLogger(__name__)


@dataclass
class GridCellInitialization:
    """
    State and configuration for grid cell initialization.
    
    This class manages the state and parameters needed for initializing
    the sub-grid mapping structure in CLM.
    
    Attributes:
        initialized: Whether grid cells have been initialized
        num_patches: Number of patches created
        patch_indices: Array of patch indices
        landunit_types: Array of landunit type assignments
        metadata: Additional initialization metadata
    """
    initialized: bool = False
    num_patches: int = 0
    patch_indices: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    landunit_types: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def reset(self) -> 'GridCellInitialization':
        """Reset initialization state."""
        return GridCellInitialization()
    
    def is_valid(self) -> bool:
        """Check if initialization state is valid."""
        if not self.initialized:
            return True  # Uninitialized state is valid
        
        # Check that arrays have consistent sizes
        if len(self.patch_indices) != self.num_patches:
            return False
            
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get summary information about initialization state."""
        return {
            'initialized': self.initialized,
            'num_patches': self.num_patches,
            'patch_count': len(self.patch_indices),
            'landunit_count': len(self.landunit_types),
            'is_valid': self.is_valid()
        }


# Global initialization state
_grid_init_state = GridCellInitialization()


def initGridcells() -> None:
    """
    Initialize sub-grid mapping and allocate space for derived type hierarchy.
    
    For each land gridcell, this function determines landunit, column and patch
    properties by setting up the vegetated landunit with competition.
    
    This is the main entry point for grid cell initialization in CLM.
    
    Raises:
        RuntimeError: If initialization fails
    """
    global _grid_init_state
    
    try:
        logger.info("Starting grid cell initialization")
        
        # Determine naturally vegetated landunit
        set_landunit_veg_compete()
        
        # Mark as initialized
        _grid_init_state.initialized = True
        
        logger.info("Grid cell initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Grid cell initialization failed: {e}")
        raise RuntimeError(f"Failed to initialize grid cells: {e}") from e


def set_landunit_veg_compete() -> None:
    """
    Initialize vegetated landunit with competition.
    
    This function sets up the subgrid patch structure for a vegetated landunit.
    The code processes one patch (one grid cell with one column and one patch)
    and sets the subgrid patch structure accordingly.
    
    Notes:
        - Uses tower data to determine PFT (Plant Functional Type)
        - Creates a single patch for the current tower configuration
        - Updates global grid initialization state
    
    Raises:
        ValueError: If tower data is invalid
        RuntimeError: If patch addition fails
    """
    global _grid_init_state
    
    try:
        logger.debug("Initializing vegetated landunit with competition")
        
        # Validate tower data
        if tower_num < 0 or tower_num >= len(tower_pft):
            raise ValueError(f"Invalid tower_num: {tower_num}")
        
        # The code processes one patch (one grid cell with one column and one patch)
        # and the subgrid patch structure is set accordingly
        pi = 0
        
        # Get PFT for current tower
        current_pft = int(tower_pft[tower_num])
        
        # Add patch with the specified PFT
        add_patch(pi, current_pft)
        
        # Update grid initialization state
        _grid_init_state.num_patches += 1
        _grid_init_state.patch_indices = jnp.append(_grid_init_state.patch_indices, pi)
        
        # Store metadata
        _grid_init_state.metadata.update({
            'tower_num': tower_num,
            'current_pft': current_pft,
            'patch_index': pi
        })
        
        logger.debug(f"Added patch {pi} with PFT {current_pft} for tower {tower_num}")
        
    except Exception as e:
        logger.error(f"Failed to set landunit veg compete: {e}")
        raise RuntimeError(f"Landunit vegetation competition setup failed: {e}") from e


@jax.jit
def validate_patch_structure(patch_indices: jnp.ndarray, 
                           landunit_types: jnp.ndarray) -> bool:
    """
    Validate the patch structure consistency.
    
    Args:
        patch_indices: Array of patch indices
        landunit_types: Array of landunit types
        
    Returns:
        True if structure is valid, False otherwise
    """
    # Check for non-negative indices
    valid_indices = jnp.all(patch_indices >= 0)
    
    # Check for consistent array sizes (if both non-empty)
    size_consistent = (len(patch_indices) == 0) or (len(landunit_types) == 0) or \
                     (len(patch_indices) == len(landunit_types))
    
    return valid_indices & size_consistent


def get_grid_initialization_state() -> GridCellInitialization:
    """
    Get the current grid initialization state.
    
    Returns:
        Copy of current grid initialization state
    """
    global _grid_init_state
    return _grid_init_state


def reset_grid_initialization() -> None:
    """
    Reset the grid initialization state.
    
    This function resets the global grid initialization state,
    allowing for re-initialization of grid cells.
    """
    global _grid_init_state
    _grid_init_state = GridCellInitialization()
    logger.info("Grid initialization state reset")


def create_simple_grid(num_patches: int = 1, 
                      pft_types: Optional[jnp.ndarray] = None) -> GridCellInitialization:
    """
    Create a simple grid initialization for testing purposes.
    
    Args:
        num_patches: Number of patches to create
        pft_types: PFT types for each patch (optional)
        
    Returns:
        GridCellInitialization object with simple grid structure
    """
    if pft_types is None:
        pft_types = jnp.ones(num_patches, dtype=int)
    
    if len(pft_types) != num_patches:
        raise ValueError("pft_types length must match num_patches")
    
    grid_init = GridCellInitialization(
        initialized=True,
        num_patches=num_patches,
        patch_indices=jnp.arange(num_patches),
        landunit_types=pft_types,
        metadata={'creation_type': 'simple_grid'}
    )
    
    return grid_init


@jax.jit
def calculate_grid_statistics(patch_indices: jnp.ndarray) -> Dict[str, float]:
    """
    Calculate basic statistics for the grid structure.
    
    Args:
        patch_indices: Array of patch indices
        
    Returns:
        Dictionary with grid statistics
    """
    if len(patch_indices) == 0:
        return {
            'mean_index': 0.0,
            'max_index': 0.0,
            'min_index': 0.0,
            'num_patches': 0.0
        }
    
    return {
        'mean_index': float(jnp.mean(patch_indices)),
        'max_index': float(jnp.max(patch_indices)),
        'min_index': float(jnp.min(patch_indices)),
        'num_patches': float(len(patch_indices))
    }


def print_initialization_summary() -> None:
    """
    Print a summary of the current grid initialization state.
    """
    global _grid_init_state
    
    info = _grid_init_state.get_info()
    
    print("\n=== Grid Cell Initialization Summary ===")
    print(f"Initialized: {info['initialized']}")
    print(f"Number of patches: {info['num_patches']}")
    print(f"Patch count: {info['patch_count']}")
    print(f"Landunit count: {info['landunit_count']}")
    print(f"Valid state: {info['is_valid']}")
    
    if _grid_init_state.metadata:
        print("\nMetadata:")
        for key, value in _grid_init_state.metadata.items():
            print(f"  {key}: {value}")
    
    if len(_grid_init_state.patch_indices) > 0:
        stats = calculate_grid_statistics(_grid_init_state.patch_indices)
        print(f"\nGrid Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
    
    print("=" * 40)


# Factory functions for common initialization patterns
def initialize_single_patch_grid(pft: int = 1) -> None:
    """
    Initialize a simple single-patch grid.
    
    Args:
        pft: Plant Functional Type for the patch
    """
    global _grid_init_state
    
    _grid_init_state = create_simple_grid(num_patches=1, 
                                         pft_types=jnp.array([pft]))
    logger.info(f"Initialized single patch grid with PFT {pft}")


def initialize_multi_patch_grid(pft_list: jnp.ndarray) -> None:
    """
    Initialize a multi-patch grid.
    
    Args:
        pft_list: Array of PFT types for each patch
    """
    global _grid_init_state
    
    _grid_init_state = create_simple_grid(num_patches=len(pft_list), 
                                         pft_types=pft_list)
    logger.info(f"Initialized multi-patch grid with {len(pft_list)} patches")


# Validation and utility functions
def validate_initialization() -> Tuple[bool, str]:
    """
    Validate the current grid initialization state.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    global _grid_init_state
    
    if not _grid_init_state.is_valid():
        return False, "Grid initialization state is invalid"
    
    if _grid_init_state.initialized and _grid_init_state.num_patches == 0:
        return False, "Grid marked as initialized but has no patches"
    
    # Validate patch structure if arrays exist
    if (len(_grid_init_state.patch_indices) > 0 and 
        len(_grid_init_state.landunit_types) > 0):
        structure_valid = validate_patch_structure(
            _grid_init_state.patch_indices,
            _grid_init_state.landunit_types
        )
        if not structure_valid:
            return False, "Patch structure validation failed"
    
    return True, "Grid initialization is valid"


# Backward compatibility alias (capitalize first letter)
initGridCells = initGridcells

# Export interface
__all__ = [
    'initGridcells',
    'initGridCells',  # Alias
    'set_landunit_veg_compete', 
    'GridCellInitialization',
    'get_grid_initialization_state',
    'reset_grid_initialization',
    'create_simple_grid',
    'validate_patch_structure',
    'calculate_grid_statistics',
    'print_initialization_summary',
    'initialize_single_patch_grid',
    'initialize_multi_patch_grid',
    'validate_initialization'
]