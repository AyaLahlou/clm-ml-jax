"""
History file module for CLM

This module contains methods for CLM history file handling including
field registration and management for output files.
Translated from Fortran CLM code to Python JAX.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Union, Dict, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

# Import dependencies
try:
    from ..cime_src_share_util.shr_kind_mod import r8
    from .decompMod import BoundsType
except ImportError:
    # Fallback for when running outside package context
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cime_src_share_util.shr_kind_mod import r8
    from clm_src_main.decompMod import BoundsType

# Alias for backward compatibility
bounds_type = BoundsType


class FieldType(Enum):
    """Enumeration of field types for history output"""
    GRIDCELL = "gridcell"
    LANDUNIT = "landunit"
    COLUMN = "column"
    PATCH = "patch"
    LAND = "land"
    ATMOSPHERE = "atmosphere"


class AveragingFlag(Enum):
    """Enumeration of time averaging flags"""
    INSTANTANEOUS = "I"
    AVERAGE = "A"
    MINIMUM = "X"
    MAXIMUM = "M"
    SUM = "S"


class ScaleType(Enum):
    """Enumeration of scaling types for subgrid averaging"""
    UNITY = "unity"
    AREA = "area"
    MASS = "mass"
    VOLUME = "volume"


@dataclass
class HistoryField:
    """
    Data structure for a single history field
    
    Contains all metadata and configuration for a field that can be
    output to CLM history files.
    """
    
    # Basic field information
    fname: str                                  # field name
    units: str                                  # units of field
    avgflag: str                               # time averaging flag
    long_name: str                             # long name of field
    
    # Field type and dimensions
    field_type: FieldType = FieldType.COLUMN   # type of field
    is_2d: bool = False                        # whether field is 2D
    type1d_out: Optional[str] = None           # output type
    type2d: Optional[str] = None               # 2d output type
    
    # Data pointers (arrays)
    ptr_gcell: Optional[jnp.ndarray] = None    # pointer to gridcell array
    ptr_lunit: Optional[jnp.ndarray] = None    # pointer to landunit array
    ptr_col: Optional[jnp.ndarray] = None      # pointer to column array
    ptr_patch: Optional[jnp.ndarray] = None    # pointer to patch array
    ptr_lnd: Optional[jnp.ndarray] = None      # pointer to land array
    ptr_atm: Optional[jnp.ndarray] = None      # pointer to atmosphere array
    
    # Scaling configuration
    p2c_scale_type: Optional[str] = None       # patch to column scaling
    c2l_scale_type: Optional[str] = None       # column to landunit scaling
    l2g_scale_type: Optional[str] = None       # landunit to gridcell scaling
    
    # Special values for different surface types
    set_lake: Optional[float] = None           # value to set lakes to
    set_nolake: Optional[float] = None         # value to set non-lakes to
    set_urb: Optional[float] = None            # value to set urban to
    set_nourb: Optional[float] = None          # value to set non-urban to
    set_noglcmec: Optional[float] = None       # value to set non-glacier_mec to
    set_spec: Optional[float] = None           # value to set special to
    
    # Additional configuration
    no_snow_behavior: Optional[int] = None     # special behavior for multi-layer snow fields
    default: Optional[str] = None              # default tape behavior
    
    # Computed properties
    active: bool = True                        # whether field is active for output
    
    def get_data_pointer(self) -> Optional[jnp.ndarray]:
        """Get the appropriate data pointer based on field type"""
        pointer_map = {
            FieldType.GRIDCELL: self.ptr_gcell,
            FieldType.LANDUNIT: self.ptr_lunit,
            FieldType.COLUMN: self.ptr_col,
            FieldType.PATCH: self.ptr_patch,
            FieldType.LAND: self.ptr_lnd,
            FieldType.ATMOSPHERE: self.ptr_atm
        }
        return pointer_map.get(self.field_type)
    
    def validate(self) -> bool:
        """Validate field configuration"""
        try:
            # Check required fields
            if not all([self.fname, self.units, self.avgflag, self.long_name]):
                return False
            
            # Check that at least one data pointer is provided
            data_ptr = self.get_data_pointer()
            if data_ptr is None:
                return False
            
            # Validate averaging flag
            valid_avgflags = [flag.value for flag in AveragingFlag]
            if self.avgflag not in valid_avgflags:
                return False
            
            # For 2D fields, check type2d is provided
            if self.is_2d and self.type2d is None:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_field_info(self) -> Dict[str, Any]:
        """Get comprehensive field information"""
        return {
            'name': self.fname,
            'units': self.units,
            'long_name': self.long_name,
            'averaging': self.avgflag,
            'field_type': self.field_type.value,
            'is_2d': self.is_2d,
            'active': self.active,
            'has_data': self.get_data_pointer() is not None,
            'data_shape': self.get_data_pointer().shape if self.get_data_pointer() is not None else None
        }


class HistoryManager:
    """
    Manager for CLM history fields and output operations
    
    This class maintains a registry of all history fields and provides
    methods for field management and output coordination.
    """
    
    def __init__(self):
        self.fields: Dict[str, HistoryField] = {}
        self.field_order: List[str] = []
    
    def add_field_1d(self, fname: str, units: str, avgflag: str, long_name: str,
                    type1d_out: Optional[str] = None,
                    ptr_gcell: Optional[jnp.ndarray] = None,
                    ptr_lunit: Optional[jnp.ndarray] = None,
                    ptr_col: Optional[jnp.ndarray] = None,
                    ptr_patch: Optional[jnp.ndarray] = None,
                    ptr_lnd: Optional[jnp.ndarray] = None,
                    ptr_atm: Optional[jnp.ndarray] = None,
                    p2c_scale_type: Optional[str] = None,
                    c2l_scale_type: Optional[str] = None,
                    l2g_scale_type: Optional[str] = None,
                    set_lake: Optional[float] = None,
                    set_nolake: Optional[float] = None,
                    set_urb: Optional[float] = None,
                    set_nourb: Optional[float] = None,
                    set_noglcmec: Optional[float] = None,
                    set_spec: Optional[float] = None,
                    default: Optional[str] = None) -> bool:
        """
        Add a 1D single-level field to the master field list
        
        Args:
            fname: field name
            units: units of field
            avgflag: time averaging flag
            long_name: long name of field
            type1d_out: output type (from data type)
            ptr_gcell: pointer to gridcell array
            ptr_lunit: pointer to landunit array
            ptr_col: pointer to column array
            ptr_patch: pointer to patch array
            ptr_lnd: pointer to land array
            ptr_atm: pointer to atmosphere array
            p2c_scale_type: scale type for patch to column averaging
            c2l_scale_type: scale type for column to landunit averaging
            l2g_scale_type: scale type for landunit to gridcell averaging
            set_lake: value to set lakes to
            set_nolake: value to set non-lakes to
            set_urb: value to set urban to
            set_nourb: value to set non-urban to
            set_noglcmec: value to set non-glacier_mec to
            set_spec: value to set special to
            default: if set to 'inactive', field will not appear on primary tape
            
        Returns:
            True if field was successfully added
        """
        try:
            # Determine field type based on which pointer is provided
            field_type = self._determine_field_type(
                ptr_gcell, ptr_lunit, ptr_col, ptr_patch, ptr_lnd, ptr_atm
            )
            
            field = HistoryField(
                fname=fname,
                units=units,
                avgflag=avgflag,
                long_name=long_name,
                field_type=field_type,
                is_2d=False,
                type1d_out=type1d_out,
                ptr_gcell=ptr_gcell,
                ptr_lunit=ptr_lunit,
                ptr_col=ptr_col,
                ptr_patch=ptr_patch,
                ptr_lnd=ptr_lnd,
                ptr_atm=ptr_atm,
                p2c_scale_type=p2c_scale_type,
                c2l_scale_type=c2l_scale_type,
                l2g_scale_type=l2g_scale_type,
                set_lake=set_lake,
                set_nolake=set_nolake,
                set_urb=set_urb,
                set_nourb=set_nourb,
                set_noglcmec=set_noglcmec,
                set_spec=set_spec,
                default=default,
                active=default != 'inactive'
            )
            
            if field.validate():
                self.fields[fname] = field
                if fname not in self.field_order:
                    self.field_order.append(fname)
                return True
            return False
            
        except Exception:
            return False
    
    def add_field_2d(self, fname: str, type2d: str, units: str, avgflag: str, long_name: str,
                    type1d_out: Optional[str] = None,
                    ptr_gcell: Optional[jnp.ndarray] = None,
                    ptr_lunit: Optional[jnp.ndarray] = None,
                    ptr_col: Optional[jnp.ndarray] = None,
                    ptr_patch: Optional[jnp.ndarray] = None,
                    ptr_lnd: Optional[jnp.ndarray] = None,
                    ptr_atm: Optional[jnp.ndarray] = None,
                    p2c_scale_type: Optional[str] = None,
                    c2l_scale_type: Optional[str] = None,
                    l2g_scale_type: Optional[str] = None,
                    set_lake: Optional[float] = None,
                    set_nolake: Optional[float] = None,
                    set_urb: Optional[float] = None,
                    set_nourb: Optional[float] = None,
                    set_spec: Optional[float] = None,
                    no_snow_behavior: Optional[int] = None,
                    default: Optional[str] = None) -> bool:
        """
        Add a 2D multi-level field to the master field list
        
        Args:
            fname: field name
            type2d: 2D output type
            units: units of field
            avgflag: time averaging flag
            long_name: long name of field
            type1d_out: output type (from data type)
            ptr_gcell: pointer to gridcell array
            ptr_lunit: pointer to landunit array
            ptr_col: pointer to column array
            ptr_patch: pointer to patch array
            ptr_lnd: pointer to land array
            ptr_atm: pointer to atmosphere array
            p2c_scale_type: scale type for patch to column averaging
            c2l_scale_type: scale type for column to landunit averaging
            l2g_scale_type: scale type for landunit to gridcell averaging
            set_lake: value to set lakes to
            set_nolake: value to set non-lakes to
            set_urb: value to set urban to
            set_nourb: value to set non-urban to
            set_spec: value to set special to
            no_snow_behavior: special behavior for multi-layer snow fields
            default: if set to 'inactive', field will not appear on primary tape
            
        Returns:
            True if field was successfully added
        """
        try:
            # Determine field type based on which pointer is provided
            field_type = self._determine_field_type(
                ptr_gcell, ptr_lunit, ptr_col, ptr_patch, ptr_lnd, ptr_atm
            )
            
            field = HistoryField(
                fname=fname,
                units=units,
                avgflag=avgflag,
                long_name=long_name,
                field_type=field_type,
                is_2d=True,
                type1d_out=type1d_out,
                type2d=type2d,
                ptr_gcell=ptr_gcell,
                ptr_lunit=ptr_lunit,
                ptr_col=ptr_col,
                ptr_patch=ptr_patch,
                ptr_lnd=ptr_lnd,
                ptr_atm=ptr_atm,
                p2c_scale_type=p2c_scale_type,
                c2l_scale_type=c2l_scale_type,
                l2g_scale_type=l2g_scale_type,
                set_lake=set_lake,
                set_nolake=set_nolake,
                set_urb=set_urb,
                set_nourb=set_nourb,
                set_spec=set_spec,
                no_snow_behavior=no_snow_behavior,
                default=default,
                active=default != 'inactive'
            )
            
            if field.validate():
                self.fields[fname] = field
                if fname not in self.field_order:
                    self.field_order.append(fname)
                return True
            return False
            
        except Exception:
            return False
    
    def _determine_field_type(self, ptr_gcell, ptr_lunit, ptr_col, 
                             ptr_patch, ptr_lnd, ptr_atm) -> FieldType:
        """Determine field type based on which data pointer is provided"""
        if ptr_gcell is not None:
            return FieldType.GRIDCELL
        elif ptr_lunit is not None:
            return FieldType.LANDUNIT
        elif ptr_col is not None:
            return FieldType.COLUMN
        elif ptr_patch is not None:
            return FieldType.PATCH
        elif ptr_lnd is not None:
            return FieldType.LAND
        elif ptr_atm is not None:
            return FieldType.ATMOSPHERE
        else:
            return FieldType.COLUMN  # Default
    
    def get_field(self, fname: str) -> Optional[HistoryField]:
        """Get a field by name"""
        return self.fields.get(fname)
    
    def get_all_fields(self) -> Dict[str, HistoryField]:
        """Get all registered fields"""
        return self.fields.copy()
    
    def get_active_fields(self) -> Dict[str, HistoryField]:
        """Get only active fields"""
        return {name: field for name, field in self.fields.items() if field.active}
    
    def activate_field(self, fname: str) -> bool:
        """Activate a field for output"""
        if fname in self.fields:
            self.fields[fname].active = True
            return True
        return False
    
    def deactivate_field(self, fname: str) -> bool:
        """Deactivate a field from output"""
        if fname in self.fields:
            self.fields[fname].active = False
            return True
        return False
    
    def remove_field(self, fname: str) -> bool:
        """Remove a field from the registry"""
        if fname in self.fields:
            del self.fields[fname]
            if fname in self.field_order:
                self.field_order.remove(fname)
            return True
        return False
    
    def get_fields_by_type(self, field_type: FieldType) -> Dict[str, HistoryField]:
        """Get all fields of a specific type"""
        return {name: field for name, field in self.fields.items() 
                if field.field_type == field_type}
    
    def get_field_summary(self) -> Dict[str, Any]:
        """Get summary information about all fields"""
        active_fields = len(self.get_active_fields())
        total_fields = len(self.fields)
        
        type_counts = {}
        for field_type in FieldType:
            type_counts[field_type.value] = len(self.get_fields_by_type(field_type))
        
        return {
            'total_fields': total_fields,
            'active_fields': active_fields,
            'inactive_fields': total_fields - active_fields,
            'fields_by_type': type_counts,
            'field_names': list(self.field_order)
        }


# Global history manager instance
_history_manager = HistoryManager()


def hist_addfld1d(fname: str, units: str, avgflag: str, long_name: str,
                 type1d_out: Optional[str] = None,
                 ptr_gcell: Optional[jnp.ndarray] = None,
                 ptr_lunit: Optional[jnp.ndarray] = None,
                 ptr_col: Optional[jnp.ndarray] = None,
                 ptr_patch: Optional[jnp.ndarray] = None,
                 ptr_lnd: Optional[jnp.ndarray] = None,
                 ptr_atm: Optional[jnp.ndarray] = None,
                 p2c_scale_type: Optional[str] = None,
                 c2l_scale_type: Optional[str] = None,
                 l2g_scale_type: Optional[str] = None,
                 set_lake: Optional[float] = None,
                 set_nolake: Optional[float] = None,
                 set_urb: Optional[float] = None,
                 set_nourb: Optional[float] = None,
                 set_noglcmec: Optional[float] = None,
                 set_spec: Optional[float] = None,
                 default: Optional[str] = None) -> None:
    """
    Add a 1D single-level field to the master field list
    
    This function maintains the same signature as the original Fortran
    routine for compatibility.
    """
    _history_manager.add_field_1d(
        fname=fname, units=units, avgflag=avgflag, long_name=long_name,
        type1d_out=type1d_out, ptr_gcell=ptr_gcell, ptr_lunit=ptr_lunit,
        ptr_col=ptr_col, ptr_patch=ptr_patch, ptr_lnd=ptr_lnd, ptr_atm=ptr_atm,
        p2c_scale_type=p2c_scale_type, c2l_scale_type=c2l_scale_type,
        l2g_scale_type=l2g_scale_type, set_lake=set_lake, set_nolake=set_nolake,
        set_urb=set_urb, set_nourb=set_nourb, set_noglcmec=set_noglcmec,
        set_spec=set_spec, default=default
    )


def hist_addfld2d(fname: str, type2d: str, units: str, avgflag: str, long_name: str,
                 type1d_out: Optional[str] = None,
                 ptr_gcell: Optional[jnp.ndarray] = None,
                 ptr_lunit: Optional[jnp.ndarray] = None,
                 ptr_col: Optional[jnp.ndarray] = None,
                 ptr_patch: Optional[jnp.ndarray] = None,
                 ptr_lnd: Optional[jnp.ndarray] = None,
                 ptr_atm: Optional[jnp.ndarray] = None,
                 p2c_scale_type: Optional[str] = None,
                 c2l_scale_type: Optional[str] = None,
                 l2g_scale_type: Optional[str] = None,
                 set_lake: Optional[float] = None,
                 set_nolake: Optional[float] = None,
                 set_urb: Optional[float] = None,
                 set_nourb: Optional[float] = None,
                 set_spec: Optional[float] = None,
                 no_snow_behavior: Optional[int] = None,
                 default: Optional[str] = None) -> None:
    """
    Add a 2D multi-level field to the master field list
    
    This function maintains the same signature as the original Fortran
    routine for compatibility.
    """
    _history_manager.add_field_2d(
        fname=fname, type2d=type2d, units=units, avgflag=avgflag, 
        long_name=long_name, type1d_out=type1d_out, ptr_gcell=ptr_gcell,
        ptr_lunit=ptr_lunit, ptr_col=ptr_col, ptr_patch=ptr_patch,
        ptr_lnd=ptr_lnd, ptr_atm=ptr_atm, p2c_scale_type=p2c_scale_type,
        c2l_scale_type=c2l_scale_type, l2g_scale_type=l2g_scale_type,
        set_lake=set_lake, set_nolake=set_nolake, set_urb=set_urb,
        set_nourb=set_nourb, set_spec=set_spec, 
        no_snow_behavior=no_snow_behavior, default=default
    )


def get_history_manager() -> HistoryManager:
    """Get the global history manager instance"""
    return _history_manager


def reset_history_manager() -> None:
    """Reset the global history manager"""
    global _history_manager
    _history_manager = HistoryManager()


# Public interface
__all__ = [
    'hist_addfld1d', 'hist_addfld2d', 'HistoryManager', 'HistoryField',
    'FieldType', 'AveragingFlag', 'ScaleType',
    'get_history_manager', 'reset_history_manager'
]