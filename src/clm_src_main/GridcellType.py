"""
Gridcell data type module

This module defines the gridcell_type class for handling gridcell-level
data structures in CLM including latitude and longitude coordinates.
Translated from Fortran CLM code to Python JAX.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass

# Import dependencies
try:
    from ..cime_src_share_util.shr_kind_mod import r8
    from .clm_varcon import spval as nan
except ImportError:
    # Fallback for when running outside package context
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cime_src_share_util.shr_kind_mod import r8
    from clm_src_main.clm_varcon import spval as nan


@dataclass
class gridcell_type:
    """
    Gridcell data type for CLM
    
    This class contains gridcell-level variables including geographic
    coordinates and other grid-level properties.
    """
    
    # Geographic coordinate arrays
    latdeg: Optional[jnp.ndarray] = None    # latitude (degrees)
    londeg: Optional[jnp.ndarray] = None    # longitude (degrees)
    
    # Store bounds for reference
    begg: Optional[int] = None
    endg: Optional[int] = None
    
    def Init(self, begg: int, endg: int) -> None:
        """
        Initialize gridcell data structure
        
        Args:
            begg: Beginning gridcell index
            endg: Ending gridcell index
        """
        self.begg = begg
        self.endg = endg
        
        # Calculate array size
        grid_size = endg - begg + 1
        
        # Initialize arrays with NaN (missing data indicator)
        self.latdeg = jnp.full((grid_size,), nan, dtype=r8)
        self.londeg = jnp.full((grid_size,), nan, dtype=r8)
    
    def is_initialized(self) -> bool:
        """Check if the gridcell type has been initialized"""
        return all([
            self.latdeg is not None,
            self.londeg is not None,
            self.begg is not None,
            self.endg is not None
        ])
    
    def get_gridcell_count(self) -> int:
        """Get the number of gridcells"""
        if self.begg is not None and self.endg is not None:
            return self.endg - self.begg + 1
        return 0
    
    def get_coordinate_bounds(self) -> dict:
        """
        Get geographic coordinate bounds
        
        Returns:
            Dictionary with coordinate bounds
        """
        if not self.is_initialized():
            return {}
        
        # Filter out NaN values for bounds calculation
        valid_lat_mask = ~jnp.isnan(self.latdeg)
        valid_lon_mask = ~jnp.isnan(self.londeg)
        
        bounds = {}
        
        if jnp.any(valid_lat_mask):
            valid_lats = self.latdeg[valid_lat_mask]
            bounds['lat_min'] = float(jnp.min(valid_lats))
            bounds['lat_max'] = float(jnp.max(valid_lats))
            bounds['lat_mean'] = float(jnp.mean(valid_lats))
        
        if jnp.any(valid_lon_mask):
            valid_lons = self.londeg[valid_lon_mask]
            bounds['lon_min'] = float(jnp.min(valid_lons))
            bounds['lon_max'] = float(jnp.max(valid_lons))
            bounds['lon_mean'] = float(jnp.mean(valid_lons))
        
        return bounds
    
    def set_coordinates(self, latitudes: jnp.ndarray, longitudes: jnp.ndarray) -> None:
        """
        Set coordinate values for all gridcells
        
        Args:
            latitudes: Array of latitude values (degrees)
            longitudes: Array of longitude values (degrees)
        """
        if not self.is_initialized():
            raise RuntimeError("Gridcell type must be initialized before setting coordinates")
        
        grid_size = self.get_gridcell_count()
        
        if len(latitudes) != grid_size or len(longitudes) != grid_size:
            raise ValueError(f"Coordinate arrays must have length {grid_size}")
        
        self.latdeg = jnp.array(latitudes, dtype=r8)
        self.londeg = jnp.array(longitudes, dtype=r8)
    
    def set_single_coordinate(self, grid_idx: int, latitude: float, longitude: float) -> None:
        """
        Set coordinates for a single gridcell
        
        Args:
            grid_idx: Gridcell index (0-based Python indexing)
            latitude: Latitude value (degrees)
            longitude: Longitude value (degrees)
        """
        if not self.is_initialized():
            raise RuntimeError("Gridcell type must be initialized before setting coordinates")
        
        if grid_idx < 0 or grid_idx >= self.get_gridcell_count():
            raise IndexError(f"Grid index {grid_idx} out of range [0, {self.get_gridcell_count()-1}]")
        
        self.latdeg = self.latdeg.at[grid_idx].set(latitude)
        self.londeg = self.londeg.at[grid_idx].set(longitude)
    
    def get_coordinates(self, grid_idx: Optional[int] = None) -> Union[Tuple[float, float], Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Get coordinate values
        
        Args:
            grid_idx: Optional gridcell index (0-based). If None, return all coordinates
            
        Returns:
            Tuple of (latitude, longitude) - single values if grid_idx provided, arrays otherwise
        """
        if not self.is_initialized():
            raise RuntimeError("Gridcell type must be initialized before getting coordinates")
        
        if grid_idx is not None:
            if grid_idx < 0 or grid_idx >= self.get_gridcell_count():
                raise IndexError(f"Grid index {grid_idx} out of range [0, {self.get_gridcell_count()-1}]")
            return float(self.latdeg[grid_idx]), float(self.londeg[grid_idx])
        else:
            return self.latdeg, self.londeg
    
    def get_fortran_index(self, python_idx: int) -> int:
        """
        Convert Python index to equivalent Fortran index
        
        Args:
            python_idx: Python index (0-based)
            
        Returns:
            Fortran index (1-based, offset by begg)
        """
        return self.begg + python_idx
    
    def get_python_index(self, fortran_idx: int) -> int:
        """
        Convert Fortran index to equivalent Python index
        
        Args:
            fortran_idx: Fortran index (offset by begg)
            
        Returns:
            Python index (0-based)
        """
        return fortran_idx - self.begg
    
    def validate_arrays(self) -> bool:
        """Validate array dimensions and consistency"""
        if not self.is_initialized():
            return False
        
        try:
            grid_size = self.get_gridcell_count()
            
            # Check array shapes
            if (self.latdeg.shape != (grid_size,) or
                self.londeg.shape != (grid_size,)):
                return False
            
            # Check coordinate validity (basic range checks)
            if jnp.any(~jnp.isnan(self.latdeg)):
                valid_lats = self.latdeg[~jnp.isnan(self.latdeg)]
                if jnp.any(valid_lats < -90) or jnp.any(valid_lats > 90):
                    return False
            
            if jnp.any(~jnp.isnan(self.londeg)):
                valid_lons = self.londeg[~jnp.isnan(self.londeg)]
                if jnp.any(valid_lons < -180) or jnp.any(valid_lons > 360):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_grid_info(self) -> dict:
        """Get comprehensive gridcell information"""
        info = {
            'total_gridcells': self.get_gridcell_count(),
            'begg_index': self.begg,
            'endg_index': self.endg,
            'initialized': self.is_initialized()
        }
        
        if self.is_initialized():
            info.update(self.get_coordinate_bounds())
            
            # Count valid (non-NaN) coordinates
            valid_coords = ~(jnp.isnan(self.latdeg) | jnp.isnan(self.londeg))
            info['valid_coordinates'] = int(jnp.sum(valid_coords))
            info['missing_coordinates'] = int(jnp.sum(~valid_coords))
        
        return info


# Global gridcell instance (equivalent to Fortran module-level variable)
grc = gridcell_type()


def create_gridcell_instance(begg: int, endg: int) -> gridcell_type:
    """
    Factory function to create and initialize a gridcell_type instance
    
    Args:
        begg: Beginning gridcell index
        endg: Ending gridcell index
        
    Returns:
        Initialized gridcell_type instance
    """
    instance = gridcell_type()
    instance.Init(begg, endg)
    return instance


def reset_global_gridcell() -> None:
    """Reset the global gridcell instance"""
    global grc
    grc = gridcell_type()


def create_regular_grid(begg: int, endg: int, 
                       lat_range: Tuple[float, float], 
                       lon_range: Tuple[float, float]) -> gridcell_type:
    """
    Create a gridcell instance with regular lat/lon coordinates
    
    Args:
        begg: Beginning gridcell index
        endg: Ending gridcell index
        lat_range: Tuple of (min_lat, max_lat) in degrees
        lon_range: Tuple of (min_lon, max_lon) in degrees
        
    Returns:
        Gridcell instance with regular coordinate grid
    """
    instance = create_gridcell_instance(begg, endg)
    grid_size = endg - begg + 1
    
    # Create regular grid
    # For simplicity, create 1D arrays - in practice this might be 2D grid mapping
    lats = jnp.linspace(lat_range[0], lat_range[1], grid_size)
    lons = jnp.linspace(lon_range[0], lon_range[1], grid_size)
    
    instance.set_coordinates(lats, lons)
    return instance


# JAX utility functions for coordinate operations
@jax.jit
def calculate_distance_haversine(lat1: jnp.ndarray, lon1: jnp.ndarray,
                                lat2: jnp.ndarray, lon2: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate great circle distance between coordinate pairs using Haversine formula
    
    Args:
        lat1, lon1: First set of coordinates (degrees)
        lat2, lon2: Second set of coordinates (degrees)
        
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1_rad = jnp.radians(lat1)
    lon1_rad = jnp.radians(lon1)
    lat2_rad = jnp.radians(lat2)
    lon2_rad = jnp.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (jnp.sin(dlat/2)**2 + 
         jnp.cos(lat1_rad) * jnp.cos(lat2_rad) * jnp.sin(dlon/2)**2)
    
    c = 2 * jnp.arcsin(jnp.sqrt(a))
    
    # Earth radius in kilometers
    earth_radius = 6371.0
    
    return earth_radius * c


@jax.jit
def degrees_to_radians(degrees: jnp.ndarray) -> jnp.ndarray:
    """Convert degrees to radians"""
    return degrees * jnp.pi / 180.0


@jax.jit
def radians_to_degrees(radians: jnp.ndarray) -> jnp.ndarray:
    """Convert radians to degrees"""
    return radians * 180.0 / jnp.pi


@jax.jit
def normalize_longitude(longitude: jnp.ndarray, range_type: str = "180") -> jnp.ndarray:
    """
    Normalize longitude to standard range
    
    Args:
        longitude: Longitude values in degrees
        range_type: "180" for [-180, 180] or "360" for [0, 360]
        
    Returns:
        Normalized longitude values
    """
    if range_type == "180":
        # Normalize to [-180, 180]
        return ((longitude + 180) % 360) - 180
    elif range_type == "360":
        # Normalize to [0, 360]
        return longitude % 360
    else:
        raise ValueError("range_type must be '180' or '360'")


def validate_coordinates(latitudes: jnp.ndarray, longitudes: jnp.ndarray) -> bool:
    """
    Validate coordinate arrays
    
    Args:
        latitudes: Latitude values (degrees)
        longitudes: Longitude values (degrees)
        
    Returns:
        True if coordinates are valid
    """
    try:
        # Check latitude range
        if jnp.any(latitudes < -90) or jnp.any(latitudes > 90):
            return False
        
        # Check longitude range (allow both conventions)
        if jnp.any(longitudes < -180) or jnp.any(longitudes > 360):
            return False
        
        # Check for consistent array lengths
        if len(latitudes) != len(longitudes):
            return False
        
        return True
        
    except Exception:
        return False


def create_coordinate_mesh(lat_values: jnp.ndarray, lon_values: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create coordinate mesh from 1D lat/lon arrays
    
    Args:
        lat_values: 1D array of latitude values
        lon_values: 1D array of longitude values
        
    Returns:
        Tuple of (lat_mesh, lon_mesh) as flattened arrays for gridcell indexing
    """
    lat_mesh, lon_mesh = jnp.meshgrid(lat_values, lon_values, indexing='ij')
    return lat_mesh.flatten(), lon_mesh.flatten()


# Public interface
__all__ = [
    'gridcell_type', 'grc', 'create_gridcell_instance', 'reset_global_gridcell',
    'create_regular_grid', 'calculate_distance_haversine', 'degrees_to_radians',
    'radians_to_degrees', 'normalize_longitude', 'validate_coordinates',
    'create_coordinate_mesh'
]