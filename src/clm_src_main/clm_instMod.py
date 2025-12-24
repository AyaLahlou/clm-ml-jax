"""
CLM instance module

This module defines instances and initialization of CLM data types.
Translated from Fortran CLM code to Python JAX.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Any

# Import dependencies
try:
    from ..cime_src_share_util.shr_kind_mod import r8
    from .decompMod import BoundsType
    # Definition of component types
    from .atm2lndType import atm2lnd_type
    from ..clm_src_biogeophys.SoilStateType import soilstate_type
    from ..clm_src_biogeophys.WaterStateType import waterstate_type
    from ..clm_src_biogeophys.CanopyStateType import canopystate_type
    from ..clm_src_biogeophys.TemperatureType import temperature_type
    from ..clm_src_biogeophys.EnergyFluxType import energyflux_type
    from ..clm_src_biogeophys.WaterFluxType import waterflux_type
    from ..clm_src_biogeophys.FrictionVelocityMod import frictionvel_type
    from ..clm_src_biogeophys.SurfaceAlbedoType import surfalb_type
    from ..clm_src_biogeophys.SolarAbsorbedType import solarabs_type
    from ..clm_src_biogeophys.SurfaceAlbedoMod import SurfaceAlbedoInitTimeConst
    from ..clm_src_biogeophys.SoilStateInitTimeConstMod import SoilStateInitTimeConst
    from .initVerticalMod import initVertical
    from ..multilayer_canopy.MLCanopyFluxesType import mlcanopy_type
except ImportError:
    # Fallback for when running outside package context
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cime_src_share_util.shr_kind_mod import r8
    from clm_src_main.decompMod import BoundsType
    from clm_src_main.atm2lndType import atm2lnd_type
    from clm_src_biogeophys.SoilStateType import soilstate_type
    from clm_src_biogeophys.WaterStateType import waterstate_type
    from clm_src_biogeophys.CanopyStateType import canopystate_type
    from clm_src_biogeophys.TemperatureType import temperature_type
    from clm_src_biogeophys.EnergyFluxType import energyflux_type
    from clm_src_biogeophys.WaterFluxType import waterflux_type
    from clm_src_biogeophys.FrictionVelocityMod import frictionvel_type
    from clm_src_biogeophys.SurfaceAlbedoType import surfalb_type
    from clm_src_biogeophys.SolarAbsorbedType import solarabs_type
    from clm_src_biogeophys.SurfaceAlbedoMod import SurfaceAlbedoInitTimeConst
    from clm_src_biogeophys.SoilStateInitTimeConstMod import SoilStateInitTimeConst
    from clm_src_main.initVerticalMod import initVertical
    from multilayer_canopy.MLCanopyFluxesType import mlcanopy_type

# Alias for backward compatibility
bounds_type = BoundsType


class CLMInstances:
    """
    Container class for all CLM component type instances
    
    This class holds all the CLM component instances and provides
    methods for initialization and restart operations.
    """
    
    def __init__(self):
        # Initialize all component instances
        self.atm2lnd_inst: Optional[atm2lnd_type] = None
        self.soilstate_inst: Optional[soilstate_type] = None
        self.waterstate_inst: Optional[waterstate_type] = None
        self.canopystate_inst: Optional[canopystate_type] = None
        self.temperature_inst: Optional[temperature_type] = None
        self.energyflux_inst: Optional[energyflux_type] = None
        self.waterflux_inst: Optional[waterflux_type] = None
        self.frictionvel_inst: Optional[frictionvel_type] = None
        self.surfalb_inst: Optional[surfalb_type] = None
        self.solarabs_inst: Optional[solarabs_type] = None
        self.mlcanopy_inst: Optional[mlcanopy_type] = None
    
    def is_initialized(self) -> bool:
        """Check if all instances have been initialized"""
        return all([
            self.atm2lnd_inst is not None,
            self.soilstate_inst is not None,
            self.waterstate_inst is not None,
            self.canopystate_inst is not None,
            self.temperature_inst is not None,
            self.energyflux_inst is not None,
            self.waterflux_inst is not None,
            self.frictionvel_inst is not None,
            self.surfalb_inst is not None,
            self.solarabs_inst is not None,
            self.mlcanopy_inst is not None
        ])


# Global instances of component types (equivalent to Fortran module-level variables)
_clm_instances = CLMInstances()

# Public access to instances (maintaining Fortran naming)
atm2lnd_inst = _clm_instances.atm2lnd_inst
soilstate_inst = _clm_instances.soilstate_inst
waterstate_inst = _clm_instances.waterstate_inst
canopystate_inst = _clm_instances.canopystate_inst
temperature_inst = _clm_instances.temperature_inst
energyflux_inst = _clm_instances.energyflux_inst
waterflux_inst = _clm_instances.waterflux_inst
frictionvel_inst = _clm_instances.frictionvel_inst
surfalb_inst = _clm_instances.surfalb_inst
solarabs_inst = _clm_instances.solarabs_inst
mlcanopy_inst = _clm_instances.mlcanopy_inst


def clm_instInit(bounds: bounds_type) -> None:
    """
    Initialization of public data types
    
    This function initializes all CLM component type instances in the
    correct order, ensuring proper dependencies are satisfied.
    
    Args:
        bounds: CLM bounds structure containing grid indices
    """
    global _clm_instances
    
    # Initialize vertical coordinate system first
    initVertical(bounds)
    
    # Initialize component instances in dependency order
    _clm_instances.atm2lnd_inst = atm2lnd_type()
    _clm_instances.atm2lnd_inst.Init(bounds)
    
    _clm_instances.soilstate_inst = soilstate_type()
    _clm_instances.soilstate_inst.Init(bounds)
    SoilStateInitTimeConst(bounds, _clm_instances.soilstate_inst)
    
    _clm_instances.waterstate_inst = waterstate_type()
    _clm_instances.waterstate_inst.Init(bounds)
    
    _clm_instances.canopystate_inst = canopystate_type()
    _clm_instances.canopystate_inst.Init(bounds)
    
    _clm_instances.temperature_inst = temperature_type()
    _clm_instances.temperature_inst.Init(bounds)
    
    _clm_instances.energyflux_inst = energyflux_type()
    _clm_instances.energyflux_inst.Init(bounds)
    
    _clm_instances.waterflux_inst = waterflux_type()
    _clm_instances.waterflux_inst.Init(bounds)
    
    _clm_instances.frictionvel_inst = frictionvel_type()
    _clm_instances.frictionvel_inst.Init(bounds)
    
    _clm_instances.surfalb_inst = surfalb_type()
    _clm_instances.surfalb_inst.Init(bounds)
    
    _clm_instances.solarabs_inst = solarabs_type()
    _clm_instances.solarabs_inst.Init(bounds)
    
    # Initialize surface albedo time constants
    SurfaceAlbedoInitTimeConst(bounds)
    
    # Initialize multilayer canopy instance
    _clm_instances.mlcanopy_inst = mlcanopy_type()
    _clm_instances.mlcanopy_inst.Init(bounds)
    
    # Update global references
    update_global_instances()


def update_global_instances() -> None:
    """Update global instance references after initialization"""
    global atm2lnd_inst, soilstate_inst, waterstate_inst, canopystate_inst
    global temperature_inst, energyflux_inst, waterflux_inst, frictionvel_inst
    global surfalb_inst, solarabs_inst, mlcanopy_inst
    
    atm2lnd_inst = _clm_instances.atm2lnd_inst
    soilstate_inst = _clm_instances.soilstate_inst
    waterstate_inst = _clm_instances.waterstate_inst
    canopystate_inst = _clm_instances.canopystate_inst
    temperature_inst = _clm_instances.temperature_inst
    energyflux_inst = _clm_instances.energyflux_inst
    waterflux_inst = _clm_instances.waterflux_inst
    frictionvel_inst = _clm_instances.frictionvel_inst
    surfalb_inst = _clm_instances.surfalb_inst
    solarabs_inst = _clm_instances.solarabs_inst
    mlcanopy_inst = _clm_instances.mlcanopy_inst


def clm_instRest(bounds: bounds_type, ncid: Any, flag: str) -> None:
    """
    Define/write/read CLM restart file
    
    This function handles restart operations for all CLM component types.
    Currently only the mlcanopy_inst restart is active, following the
    original Fortran implementation.
    
    Args:
        bounds: CLM bounds structure
        ncid: NetCDF file descriptor
        flag: Operation flag ('define', 'write', 'read')
    """
    
    # Each CLM component type has a restart subroutine 
    # (shown here for example only - commented out as in original)
    
    # if hasattr(_clm_instances.atm2lnd_inst, 'restart'):
    #     _clm_instances.atm2lnd_inst.restart(bounds, ncid, flag=flag)
    # if hasattr(_clm_instances.soilstate_inst, 'restart'):
    #     _clm_instances.soilstate_inst.restart(bounds, ncid, flag=flag)
    # if hasattr(_clm_instances.canopystate_inst, 'restart'):
    #     _clm_instances.canopystate_inst.restart(bounds, ncid, flag=flag)
    # if hasattr(_clm_instances.waterflux_inst, 'restart'):
    #     _clm_instances.waterflux_inst.restart(bounds, ncid, flag=flag)
    
    # Only mlcanopy_inst restart is active (following original Fortran)
    if _clm_instances.mlcanopy_inst and hasattr(_clm_instances.mlcanopy_inst, 'restart'):
        _clm_instances.mlcanopy_inst.restart(bounds, ncid, flag=flag)


def get_instance(instance_name: str) -> Any:
    """
    Get a specific CLM component instance by name
    
    Args:
        instance_name: Name of the instance to retrieve
        
    Returns:
        The requested instance or None if not found
    """
    return getattr(_clm_instances, instance_name, None)


def reset_instances() -> None:
    """Reset all instances to None (for testing or reinitialization)"""
    global _clm_instances
    _clm_instances = CLMInstances()
    update_global_instances()


def validate_instances(bounds: bounds_type) -> bool:
    """
    Validate that all instances are properly initialized
    
    Args:
        bounds: CLM bounds structure for validation
        
    Returns:
        True if all instances are valid, False otherwise
    """
    try:
        if not _clm_instances.is_initialized():
            return False
        
        # Additional validation could be added here
        # e.g., checking array dimensions against bounds
        
        return True
        
    except Exception:
        return False


# JIT-compiled version of initialization for performance
clm_instInit_jit = jax.jit(clm_instInit, static_argnames=['bounds'])


# Public interface
__all__ = [
    'clm_instInit', 'clm_instRest', 'clm_instInit_jit',
    'atm2lnd_inst', 'soilstate_inst', 'waterstate_inst', 'canopystate_inst',
    'temperature_inst', 'energyflux_inst', 'waterflux_inst', 'frictionvel_inst',
    'surfalb_inst', 'solarabs_inst', 'mlcanopy_inst',
    'CLMInstances', 'get_instance', 'reset_instances', 'validate_instances'
]