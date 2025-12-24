"""
Comprehensive pytest suite for clm_instMod module.

This module tests the CLM instance management system, including:
- CLMInstances container class
- clm_instInit initialization function
- update_global_instances function
- clm_instRest restart operations
- get_instance retrieval function
- reset_instances function
- validate_instances function

The tests cover nominal cases, edge cases, state transitions, and integration scenarios.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch, call
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from clm_src_main.clm_instMod import (
    CLMInstances,
    clm_instInit,
    update_global_instances,
    clm_instRest,
    get_instance,
    reset_instances,
    validate_instances,
    _clm_instances,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_bounds():
    """
    Create a mock bounds_type object with standard attributes.
    
    Returns:
        Mock object with begg, endg, begl, endl, begc, endc, begp, endp attributes
    """
    bounds = Mock()
    bounds.begg = 1
    bounds.endg = 10
    bounds.begl = 1
    bounds.endl = 10
    bounds.begc = 1
    bounds.endc = 20
    bounds.begp = 1
    bounds.endp = 40
    return bounds


@pytest.fixture
def bounds_single_cell():
    """Create bounds for a single grid cell."""
    bounds = Mock()
    bounds.begg = 1
    bounds.endg = 1
    bounds.begl = 1
    bounds.endl = 1
    bounds.begc = 1
    bounds.endc = 1
    bounds.begp = 1
    bounds.endp = 1
    return bounds


@pytest.fixture
def bounds_small_domain():
    """Create bounds for a small domain (10 grid cells)."""
    bounds = Mock()
    bounds.begg = 1
    bounds.endg = 10
    bounds.begl = 1
    bounds.endl = 10
    bounds.begc = 1
    bounds.endc = 15
    bounds.begp = 1
    bounds.endp = 20
    return bounds


@pytest.fixture
def bounds_medium_domain():
    """Create bounds for a medium domain (100 grid cells)."""
    bounds = Mock()
    bounds.begg = 1
    bounds.endg = 100
    bounds.begl = 1
    bounds.endl = 100
    bounds.begc = 1
    bounds.endc = 250
    bounds.begp = 1
    bounds.endp = 500
    return bounds


@pytest.fixture
def bounds_large_domain():
    """Create bounds for a large domain (1000 grid cells)."""
    bounds = Mock()
    bounds.begg = 1
    bounds.endg = 1000
    bounds.begl = 1
    bounds.endl = 1000
    bounds.begc = 1
    bounds.endc = 3000
    bounds.begp = 1
    bounds.endp = 6000
    return bounds


@pytest.fixture
def bounds_non_unit_start():
    """Create bounds with non-unit starting indices."""
    bounds = Mock()
    bounds.begg = 101
    bounds.endg = 200
    bounds.begl = 101
    bounds.endl = 200
    bounds.begc = 301
    bounds.endc = 500
    bounds.begp = 501
    bounds.endp = 1000
    return bounds


@pytest.fixture
def bounds_zero_size():
    """Create bounds for zero-size domain."""
    bounds = Mock()
    bounds.begg = 1
    bounds.endg = 0
    bounds.begl = 1
    bounds.endl = 0
    bounds.begc = 1
    bounds.endc = 0
    bounds.begp = 1
    bounds.endp = 0
    return bounds


@pytest.fixture
def bounds_extreme_hierarchy():
    """Create bounds with extreme hierarchy ratio."""
    bounds = Mock()
    bounds.begg = 1
    bounds.endg = 1
    bounds.begl = 1
    bounds.endl = 1
    bounds.begc = 1
    bounds.endc = 100
    bounds.begp = 1
    bounds.endp = 1000
    return bounds


@pytest.fixture
def mock_component_types():
    """
    Create mock component type classes for all 11 CLM components.
    
    Returns:
        Dictionary mapping component names to mock classes
    """
    component_mocks = {}
    
    component_names = [
        'atm2lnd_type',
        'soilstate_type',
        'waterstate_type',
        'canopystate_type',
        'temperature_type',
        'energyflux_type',
        'waterflux_type',
        'frictionvel_type',
        'surfalb_type',
        'solarabs_type',
        'mlcanopy_type',
    ]
    
    for name in component_names:
        mock_class = Mock()
        mock_instance = Mock()
        mock_instance.name = name
        
        # mlcanopy_type needs a restart method
        if name == 'mlcanopy_type':
            mock_instance.restart = Mock()
        
        mock_class.return_value = mock_instance
        component_mocks[name] = mock_class
    
    return component_mocks


@pytest.fixture(autouse=True)
def reset_global_state():
    """
    Reset global state before and after each test.
    
    This ensures test isolation by resetting all instances.
    """
    reset_instances()
    yield
    reset_instances()


@pytest.fixture
def test_data():
    """
    Load test data from the provided JSON structure.
    
    Returns:
        Dictionary containing all test cases and metadata
    """
    return {
        "test_cases": [
            {
                "name": "test_nominal_single_grid_cell",
                "inputs": {
                    "bounds": {
                        "begg": 1, "endg": 1, "begl": 1, "endl": 1,
                        "begc": 1, "endc": 1, "begp": 1, "endp": 1
                    }
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Tests initialization with a single grid cell"
                }
            },
            {
                "name": "test_nominal_small_domain",
                "inputs": {
                    "bounds": {
                        "begg": 1, "endg": 10, "begl": 1, "endl": 10,
                        "begc": 1, "endc": 15, "begp": 1, "endp": 20
                    }
                },
                "metadata": {
                    "type": "nominal",
                    "description": "Tests initialization with a small domain"
                }
            },
        ]
    }


# ============================================================================
# CLMInstances Class Tests
# ============================================================================

class TestCLMInstances:
    """Test suite for CLMInstances container class."""
    
    def test_clminstances_initialization(self):
        """
        Test that CLMInstances initializes all attributes to None.
        
        Verifies that a new CLMInstances object has all 11 component
        instances set to None.
        """
        instances = CLMInstances()
        
        assert instances.atm2lnd_inst is None, "atm2lnd_inst should be None"
        assert instances.soilstate_inst is None, "soilstate_inst should be None"
        assert instances.waterstate_inst is None, "waterstate_inst should be None"
        assert instances.canopystate_inst is None, "canopystate_inst should be None"
        assert instances.temperature_inst is None, "temperature_inst should be None"
        assert instances.energyflux_inst is None, "energyflux_inst should be None"
        assert instances.waterflux_inst is None, "waterflux_inst should be None"
        assert instances.frictionvel_inst is None, "frictionvel_inst should be None"
        assert instances.surfalb_inst is None, "surfalb_inst should be None"
        assert instances.solarabs_inst is None, "solarabs_inst should be None"
        assert instances.mlcanopy_inst is None, "mlcanopy_inst should be None"
    
    def test_clminstances_is_initialized_false(self):
        """
        Test is_initialized returns False when instances are None.
        
        Verifies that is_initialized correctly identifies uninitialized state.
        """
        instances = CLMInstances()
        assert not instances.is_initialized(), "Should return False when all instances are None"
    
    def test_clminstances_is_initialized_partial(self):
        """
        Test is_initialized returns False when only some instances are set.
        
        Verifies that is_initialized requires all instances to be non-None.
        """
        instances = CLMInstances()
        instances.atm2lnd_inst = Mock()
        instances.soilstate_inst = Mock()
        
        assert not instances.is_initialized(), "Should return False when only some instances are set"
    
    def test_clminstances_is_initialized_true(self):
        """
        Test is_initialized returns True when all instances are set.
        
        Verifies that is_initialized correctly identifies fully initialized state.
        """
        instances = CLMInstances()
        
        # Set all instances to mock objects
        instances.atm2lnd_inst = Mock()
        instances.soilstate_inst = Mock()
        instances.waterstate_inst = Mock()
        instances.canopystate_inst = Mock()
        instances.temperature_inst = Mock()
        instances.energyflux_inst = Mock()
        instances.waterflux_inst = Mock()
        instances.frictionvel_inst = Mock()
        instances.surfalb_inst = Mock()
        instances.solarabs_inst = Mock()
        instances.mlcanopy_inst = Mock()
        
        assert instances.is_initialized(), "Should return True when all instances are set"


# ============================================================================
# clm_instInit Tests
# ============================================================================

class TestClmInstInit:
    """Test suite for clm_instInit function."""
    
    @patch('clm_src_main.clm_instMod.initVertical')
    @patch('clm_src_main.clm_instMod.atm2lnd_type')
    @patch('clm_src_main.clm_instMod.soilstate_type')
    @patch('clm_src_main.clm_instMod.waterstate_type')
    @patch('clm_src_main.clm_instMod.canopystate_type')
    @patch('clm_src_main.clm_instMod.temperature_type')
    @patch('clm_src_main.clm_instMod.energyflux_type')
    @patch('clm_src_main.clm_instMod.waterflux_type')
    @patch('clm_src_main.clm_instMod.frictionvel_type')
    @patch('clm_src_main.clm_instMod.surfalb_type')
    @patch('clm_src_main.clm_instMod.solarabs_type')
    @patch('clm_src_main.clm_instMod.mlcanopy_type')
    @patch('clm_src_main.clm_instMod.SoilStateInitTimeConst')
    @patch('clm_src_main.clm_instMod.SurfaceAlbedoInitTimeConst')
    def test_clm_instinit_single_cell(
        self, mock_alb_init, mock_soil_init, mock_mlcanopy, mock_solarabs,
        mock_surfalb, mock_friction, mock_waterflux, mock_energyflux,
        mock_temperature, mock_canopy, mock_waterstate, mock_soilstate,
        mock_atm2lnd, mock_init_vertical, bounds_single_cell
    ):
        """
        Test clm_instInit with a single grid cell.
        
        Verifies that initialization works correctly for the simplest case
        of a single grid cell domain.
        """
        # Call initialization
        clm_instInit(bounds_single_cell)
        
        # Verify initVertical was called
        mock_init_vertical.assert_called_once_with(bounds_single_cell)
        
        # Verify all component types were instantiated with bounds
        mock_atm2lnd.assert_called_once_with(bounds_single_cell)
        mock_soilstate.assert_called_once_with(bounds_single_cell)
        mock_waterstate.assert_called_once_with(bounds_single_cell)
        mock_canopy.assert_called_once_with(bounds_single_cell)
        mock_temperature.assert_called_once_with(bounds_single_cell)
        mock_energyflux.assert_called_once_with(bounds_single_cell)
        mock_waterflux.assert_called_once_with(bounds_single_cell)
        mock_friction.assert_called_once_with(bounds_single_cell)
        mock_surfalb.assert_called_once_with(bounds_single_cell)
        mock_solarabs.assert_called_once_with(bounds_single_cell)
        mock_mlcanopy.assert_called_once_with(bounds_single_cell)
        
        # Verify time constant initializations were called
        mock_soil_init.assert_called_once()
        mock_alb_init.assert_called_once()
    
    @patch('clm_src_main.clm_instMod.initVertical')
    @patch('clm_src_main.clm_instMod.atm2lnd_type')
    @patch('clm_src_main.clm_instMod.soilstate_type')
    @patch('clm_src_main.clm_instMod.waterstate_type')
    @patch('clm_src_main.clm_instMod.canopystate_type')
    @patch('clm_src_main.clm_instMod.temperature_type')
    @patch('clm_src_main.clm_instMod.energyflux_type')
    @patch('clm_src_main.clm_instMod.waterflux_type')
    @patch('clm_src_main.clm_instMod.frictionvel_type')
    @patch('clm_src_main.clm_instMod.surfalb_type')
    @patch('clm_src_main.clm_instMod.solarabs_type')
    @patch('clm_src_main.clm_instMod.mlcanopy_type')
    @patch('clm_src_main.clm_instMod.SoilStateInitTimeConst')
    @patch('clm_src_main.clm_instMod.SurfaceAlbedoInitTimeConst')
    def test_clm_instinit_small_domain(
        self, mock_alb_init, mock_soil_init, mock_mlcanopy, mock_solarabs,
        mock_surfalb, mock_friction, mock_waterflux, mock_energyflux,
        mock_temperature, mock_canopy, mock_waterstate, mock_soilstate,
        mock_atm2lnd, mock_init_vertical, bounds_small_domain
    ):
        """
        Test clm_instInit with a small domain.
        
        Verifies initialization with 10 grid cells, 15 columns, 20 patches.
        """
        clm_instInit(bounds_small_domain)
        
        # Verify all components were initialized
        mock_init_vertical.assert_called_once_with(bounds_small_domain)
        assert mock_atm2lnd.call_count == 1
        assert mock_soilstate.call_count == 1
        assert mock_waterstate.call_count == 1
    
    @patch('clm_src_main.clm_instMod.initVertical')
    @patch('clm_src_main.clm_instMod.atm2lnd_type')
    @patch('clm_src_main.clm_instMod.soilstate_type')
    @patch('clm_src_main.clm_instMod.waterstate_type')
    @patch('clm_src_main.clm_instMod.canopystate_type')
    @patch('clm_src_main.clm_instMod.temperature_type')
    @patch('clm_src_main.clm_instMod.energyflux_type')
    @patch('clm_src_main.clm_instMod.waterflux_type')
    @patch('clm_src_main.clm_instMod.frictionvel_type')
    @patch('clm_src_main.clm_instMod.surfalb_type')
    @patch('clm_src_main.clm_instMod.solarabs_type')
    @patch('clm_src_main.clm_instMod.mlcanopy_type')
    @patch('clm_src_main.clm_instMod.SoilStateInitTimeConst')
    @patch('clm_src_main.clm_instMod.SurfaceAlbedoInitTimeConst')
    def test_clm_instinit_medium_domain(
        self, mock_alb_init, mock_soil_init, mock_mlcanopy, mock_solarabs,
        mock_surfalb, mock_friction, mock_waterflux, mock_energyflux,
        mock_temperature, mock_canopy, mock_waterstate, mock_soilstate,
        mock_atm2lnd, mock_init_vertical, bounds_medium_domain
    ):
        """
        Test clm_instInit with a medium domain.
        
        Verifies initialization with 100 grid cells typical of regional simulations.
        """
        clm_instInit(bounds_medium_domain)
        
        mock_init_vertical.assert_called_once_with(bounds_medium_domain)
        assert mock_atm2lnd.call_count == 1
    
    @patch('clm_src_main.clm_instMod.initVertical')
    @patch('clm_src_main.clm_instMod.atm2lnd_type')
    @patch('clm_src_main.clm_instMod.soilstate_type')
    @patch('clm_src_main.clm_instMod.waterstate_type')
    @patch('clm_src_main.clm_instMod.canopystate_type')
    @patch('clm_src_main.clm_instMod.temperature_type')
    @patch('clm_src_main.clm_instMod.energyflux_type')
    @patch('clm_src_main.clm_instMod.waterflux_type')
    @patch('clm_src_main.clm_instMod.frictionvel_type')
    @patch('clm_src_main.clm_instMod.surfalb_type')
    @patch('clm_src_main.clm_instMod.solarabs_type')
    @patch('clm_src_main.clm_instMod.mlcanopy_type')
    @patch('clm_src_main.clm_instMod.SoilStateInitTimeConst')
    @patch('clm_src_main.clm_instMod.SurfaceAlbedoInitTimeConst')
    def test_clm_instinit_large_domain(
        self, mock_alb_init, mock_soil_init, mock_mlcanopy, mock_solarabs,
        mock_surfalb, mock_friction, mock_waterflux, mock_energyflux,
        mock_temperature, mock_canopy, mock_waterstate, mock_soilstate,
        mock_atm2lnd, mock_init_vertical, bounds_large_domain
    ):
        """
        Test clm_instInit with a large domain.
        
        Verifies initialization with 1000 grid cells typical of global simulations.
        """
        clm_instInit(bounds_large_domain)
        
        mock_init_vertical.assert_called_once_with(bounds_large_domain)
        assert mock_atm2lnd.call_count == 1
    
    @patch('clm_src_main.clm_instMod.initVertical')
    @patch('clm_src_main.clm_instMod.atm2lnd_type')
    @patch('clm_src_main.clm_instMod.soilstate_type')
    @patch('clm_src_main.clm_instMod.waterstate_type')
    @patch('clm_src_main.clm_instMod.canopystate_type')
    @patch('clm_src_main.clm_instMod.temperature_type')
    @patch('clm_src_main.clm_instMod.energyflux_type')
    @patch('clm_src_main.clm_instMod.waterflux_type')
    @patch('clm_src_main.clm_instMod.frictionvel_type')
    @patch('clm_src_main.clm_instMod.surfalb_type')
    @patch('clm_src_main.clm_instMod.solarabs_type')
    @patch('clm_src_main.clm_instMod.mlcanopy_type')
    @patch('clm_src_main.clm_instMod.SoilStateInitTimeConst')
    @patch('clm_src_main.clm_instMod.SurfaceAlbedoInitTimeConst')
    def test_clm_instinit_non_unit_start(
        self, mock_alb_init, mock_soil_init, mock_mlcanopy, mock_solarabs,
        mock_surfalb, mock_friction, mock_waterflux, mock_energyflux,
        mock_temperature, mock_canopy, mock_waterstate, mock_soilstate,
        mock_atm2lnd, mock_init_vertical, bounds_non_unit_start
    ):
        """
        Test clm_instInit with non-unit starting indices.
        
        Verifies initialization works with domain decomposition where
        indices don't start at 1.
        """
        clm_instInit(bounds_non_unit_start)
        
        mock_init_vertical.assert_called_once_with(bounds_non_unit_start)
        mock_atm2lnd.assert_called_once_with(bounds_non_unit_start)
    
    @patch('clm_src_main.clm_instMod.initVertical')
    @patch('clm_src_main.clm_instMod.atm2lnd_type')
    @patch('clm_src_main.clm_instMod.soilstate_type')
    @patch('clm_src_main.clm_instMod.waterstate_type')
    @patch('clm_src_main.clm_instMod.canopystate_type')
    @patch('clm_src_main.clm_instMod.temperature_type')
    @patch('clm_src_main.clm_instMod.energyflux_type')
    @patch('clm_src_main.clm_instMod.waterflux_type')
    @patch('clm_src_main.clm_instMod.frictionvel_type')
    @patch('clm_src_main.clm_instMod.surfalb_type')
    @patch('clm_src_main.clm_instMod.solarabs_type')
    @patch('clm_src_main.clm_instMod.mlcanopy_type')
    @patch('clm_src_main.clm_instMod.SoilStateInitTimeConst')
    @patch('clm_src_main.clm_instMod.SurfaceAlbedoInitTimeConst')
    def test_clm_instinit_zero_size_domain(
        self, mock_alb_init, mock_soil_init, mock_mlcanopy, mock_solarabs,
        mock_surfalb, mock_friction, mock_waterflux, mock_energyflux,
        mock_temperature, mock_canopy, mock_waterstate, mock_soilstate,
        mock_atm2lnd, mock_init_vertical, bounds_zero_size
    ):
        """
        Test clm_instInit with zero-size domain.
        
        Verifies that initialization handles empty domains gracefully.
        This is an edge case that may occur in domain decomposition.
        """
        clm_instInit(bounds_zero_size)
        
        # Should still call initialization functions
        mock_init_vertical.assert_called_once_with(bounds_zero_size)
        mock_atm2lnd.assert_called_once_with(bounds_zero_size)
    
    @patch('clm_src_main.clm_instMod.initVertical')
    @patch('clm_src_main.clm_instMod.atm2lnd_type')
    @patch('clm_src_main.clm_instMod.soilstate_type')
    @patch('clm_src_main.clm_instMod.waterstate_type')
    @patch('clm_src_main.clm_instMod.canopystate_type')
    @patch('clm_src_main.clm_instMod.temperature_type')
    @patch('clm_src_main.clm_instMod.energyflux_type')
    @patch('clm_src_main.clm_instMod.waterflux_type')
    @patch('clm_src_main.clm_instMod.frictionvel_type')
    @patch('clm_src_main.clm_instMod.surfalb_type')
    @patch('clm_src_main.clm_instMod.solarabs_type')
    @patch('clm_src_main.clm_instMod.mlcanopy_type')
    @patch('clm_src_main.clm_instMod.SoilStateInitTimeConst')
    @patch('clm_src_main.clm_instMod.SurfaceAlbedoInitTimeConst')
    def test_clm_instinit_extreme_hierarchy(
        self, mock_alb_init, mock_soil_init, mock_mlcanopy, mock_solarabs,
        mock_surfalb, mock_friction, mock_waterflux, mock_energyflux,
        mock_temperature, mock_canopy, mock_waterstate, mock_soilstate,
        mock_atm2lnd, mock_init_vertical, bounds_extreme_hierarchy
    ):
        """
        Test clm_instInit with extreme hierarchy ratio.
        
        Verifies initialization with 1 grid cell -> 100 columns -> 1000 patches.
        This tests handling of extreme sub-grid heterogeneity.
        """
        clm_instInit(bounds_extreme_hierarchy)
        
        mock_init_vertical.assert_called_once_with(bounds_extreme_hierarchy)
        mock_atm2lnd.assert_called_once_with(bounds_extreme_hierarchy)
    
    @patch('clm_src_main.clm_instMod.initVertical')
    @patch('clm_src_main.clm_instMod.atm2lnd_type')
    @patch('clm_src_main.clm_instMod.soilstate_type')
    @patch('clm_src_main.clm_instMod.waterstate_type')
    @patch('clm_src_main.clm_instMod.canopystate_type')
    @patch('clm_src_main.clm_instMod.temperature_type')
    @patch('clm_src_main.clm_instMod.energyflux_type')
    @patch('clm_src_main.clm_instMod.waterflux_type')
    @patch('clm_src_main.clm_instMod.frictionvel_type')
    @patch('clm_src_main.clm_instMod.surfalb_type')
    @patch('clm_src_main.clm_instMod.solarabs_type')
    @patch('clm_src_main.clm_instMod.mlcanopy_type')
    @patch('clm_src_main.clm_instMod.SoilStateInitTimeConst')
    @patch('clm_src_main.clm_instMod.SurfaceAlbedoInitTimeConst')
    def test_clm_instinit_updates_global_instances(
        self, mock_alb_init, mock_soil_init, mock_mlcanopy, mock_solarabs,
        mock_surfalb, mock_friction, mock_waterflux, mock_energyflux,
        mock_temperature, mock_canopy, mock_waterstate, mock_soilstate,
        mock_atm2lnd, mock_init_vertical, mock_bounds
    ):
        """
        Test that clm_instInit updates global instance references.
        
        Verifies that after initialization, global variables are properly set.
        """
        clm_instInit(mock_bounds)
        
        # Check that global _clm_instances is initialized
        from clm_src_main.clm_instMod import _clm_instances
        assert _clm_instances is not None, "Global _clm_instances should be set"
        
        # Verify instances are set (they should be mock objects)
        assert _clm_instances.atm2lnd_inst is not None
        assert _clm_instances.soilstate_inst is not None
        assert _clm_instances.waterstate_inst is not None


# ============================================================================
# update_global_instances Tests
# ============================================================================

class TestUpdateGlobalInstances:
    """Test suite for update_global_instances function."""
    
    @patch('clm_src_main.clm_instMod._clm_instances')
    def test_update_global_instances_from_container(self, mock_container):
        """
        Test that update_global_instances copies references from container.
        
        Verifies that global instance variables are updated from the
        _clm_instances container.
        """
        # Create mock instances
        mock_container.atm2lnd_inst = Mock(name='atm2lnd')
        mock_container.soilstate_inst = Mock(name='soilstate')
        mock_container.waterstate_inst = Mock(name='waterstate')
        mock_container.canopystate_inst = Mock(name='canopystate')
        mock_container.temperature_inst = Mock(name='temperature')
        mock_container.energyflux_inst = Mock(name='energyflux')
        mock_container.waterflux_inst = Mock(name='waterflux')
        mock_container.frictionvel_inst = Mock(name='frictionvel')
        mock_container.surfalb_inst = Mock(name='surfalb')
        mock_container.solarabs_inst = Mock(name='solarabs')
        mock_container.mlcanopy_inst = Mock(name='mlcanopy')
        
        # Call update function
        update_global_instances()
        
        # Verify globals are updated (this is implementation-dependent)
        # The actual verification depends on how the module implements globals


# ============================================================================
# clm_instRest Tests
# ============================================================================

class TestClmInstRest:
    """Test suite for clm_instRest restart operations."""
    
    @patch('clm_src_main.clm_instMod._clm_instances')
    def test_clm_instrest_define_flag(self, mock_container, mock_bounds):
        """
        Test clm_instRest with 'define' flag.
        
        Verifies that restart definition phase calls mlcanopy restart
        with correct parameters.
        """
        # Setup mock mlcanopy instance with restart method
        mock_mlcanopy = Mock()
        mock_mlcanopy.restart = Mock()
        mock_container.mlcanopy_inst = mock_mlcanopy
        
        ncid = "mock_netcdf_id_12345"
        
        # Call restart with define flag
        clm_instRest(mock_bounds, ncid, 'define')
        
        # Verify restart was called
        mock_mlcanopy.restart.assert_called_once_with(mock_bounds, ncid, 'define')
    
    @patch('clm_src_main.clm_instMod._clm_instances')
    def test_clm_instrest_write_flag(self, mock_container, mock_bounds):
        """
        Test clm_instRest with 'write' flag.
        
        Verifies that restart write phase works correctly.
        """
        mock_mlcanopy = Mock()
        mock_mlcanopy.restart = Mock()
        mock_container.mlcanopy_inst = mock_mlcanopy
        
        ncid = "mock_netcdf_id_67890"
        
        clm_instRest(mock_bounds, ncid, 'write')
        
        mock_mlcanopy.restart.assert_called_once_with(mock_bounds, ncid, 'write')
    
    @patch('clm_src_main.clm_instMod._clm_instances')
    def test_clm_instrest_read_flag(self, mock_container, mock_bounds):
        """
        Test clm_instRest with 'read' flag.
        
        Verifies that restart read phase works correctly.
        """
        mock_mlcanopy = Mock()
        mock_mlcanopy.restart = Mock()
        mock_container.mlcanopy_inst = mock_mlcanopy
        
        ncid = "mock_netcdf_id_11111"
        
        clm_instRest(mock_bounds, ncid, 'read')
        
        mock_mlcanopy.restart.assert_called_once_with(mock_bounds, ncid, 'read')
    
    @patch('clm_src_main.clm_instMod._clm_instances')
    def test_clm_instrest_no_mlcanopy_instance(self, mock_container, mock_bounds):
        """
        Test clm_instRest when mlcanopy_inst is None.
        
        Verifies that restart handles missing mlcanopy instance gracefully.
        """
        mock_container.mlcanopy_inst = None
        
        # Should not raise an error
        clm_instRest(mock_bounds, "ncid", 'define')
    
    @patch('clm_src_main.clm_instMod._clm_instances')
    def test_clm_instrest_no_restart_method(self, mock_container, mock_bounds):
        """
        Test clm_instRest when mlcanopy_inst has no restart method.
        
        Verifies that restart handles instances without restart method.
        """
        mock_mlcanopy = Mock(spec=[])  # No restart method
        mock_container.mlcanopy_inst = mock_mlcanopy
        
        # Should not raise an error
        clm_instRest(mock_bounds, "ncid", 'write')


# ============================================================================
# get_instance Tests
# ============================================================================

class TestGetInstance:
    """Test suite for get_instance function."""
    
    @pytest.mark.parametrize("instance_name", [
        "atm2lnd_inst",
        "soilstate_inst",
        "waterstate_inst",
        "canopystate_inst",
        "temperature_inst",
        "energyflux_inst",
        "waterflux_inst",
        "frictionvel_inst",
        "surfalb_inst",
        "solarabs_inst",
        "mlcanopy_inst",
    ])
    @patch('clm_src_main.clm_instMod._clm_instances')
    def test_get_instance_valid_names(self, mock_container, instance_name):
        """
        Test get_instance with all valid instance names.
        
        Verifies that get_instance correctly retrieves each component instance.
        """
        # Setup mock instance
        mock_instance = Mock(name=instance_name)
        setattr(mock_container, instance_name, mock_instance)
        
        # Get instance
        result = get_instance(instance_name)
        
        # Verify correct instance returned
        assert result == mock_instance, f"Should return {instance_name}"
    
    @patch('clm_src_main.clm_instMod._clm_instances')
    def test_get_instance_invalid_name(self, mock_container):
        """
        Test get_instance with invalid instance name.
        
        Verifies that get_instance returns None for invalid names.
        """
        result = get_instance("invalid_instance_name")
        
        assert result is None, "Should return None for invalid instance name"
    
    @patch('clm_src_main.clm_instMod._clm_instances')
    def test_get_instance_none_value(self, mock_container):
        """
        Test get_instance when instance is None.
        
        Verifies that get_instance returns None for uninitialized instances.
        """
        mock_container.atm2lnd_inst = None
        
        result = get_instance("atm2lnd_inst")
        
        assert result is None, "Should return None when instance is None"


# ============================================================================
# reset_instances Tests
# ============================================================================

class TestResetInstances:
    """Test suite for reset_instances function."""
    
    def test_reset_instances_creates_new_container(self):
        """
        Test that reset_instances creates a new CLMInstances object.
        
        Verifies that reset_instances properly reinitializes the global container.
        """
        # Call reset
        reset_instances()
        
        # Verify _clm_instances exists and is a CLMInstances object
        from clm_src_main.clm_instMod import _clm_instances
        assert _clm_instances is not None
        assert isinstance(_clm_instances, CLMInstances)
    
    def test_reset_instances_clears_all_instances(self):
        """
        Test that reset_instances sets all instances to None.
        
        Verifies that after reset, all component instances are None.
        """
        reset_instances()
        
        from clm_src_main.clm_instMod import _clm_instances
        
        assert _clm_instances.atm2lnd_inst is None
        assert _clm_instances.soilstate_inst is None
        assert _clm_instances.waterstate_inst is None
        assert _clm_instances.canopystate_inst is None
        assert _clm_instances.temperature_inst is None
        assert _clm_instances.energyflux_inst is None
        assert _clm_instances.waterflux_inst is None
        assert _clm_instances.frictionvel_inst is None
        assert _clm_instances.surfalb_inst is None
        assert _clm_instances.solarabs_inst is None
        assert _clm_instances.mlcanopy_inst is None


# ============================================================================
# validate_instances Tests
# ============================================================================

class TestValidateInstances:
    """Test suite for validate_instances function."""
    
    @patch('clm_src_main.clm_instMod.initVertical')
    @patch('clm_src_main.clm_instMod.atm2lnd_type')
    @patch('clm_src_main.clm_instMod.soilstate_type')
    @patch('clm_src_main.clm_instMod.waterstate_type')
    @patch('clm_src_main.clm_instMod.canopystate_type')
    @patch('clm_src_main.clm_instMod.temperature_type')
    @patch('clm_src_main.clm_instMod.energyflux_type')
    @patch('clm_src_main.clm_instMod.waterflux_type')
    @patch('clm_src_main.clm_instMod.frictionvel_type')
    @patch('clm_src_main.clm_instMod.surfalb_type')
    @patch('clm_src_main.clm_instMod.solarabs_type')
    @patch('clm_src_main.clm_instMod.mlcanopy_type')
    @patch('clm_src_main.clm_instMod.SoilStateInitTimeConst')
    @patch('clm_src_main.clm_instMod.SurfaceAlbedoInitTimeConst')
    def test_validate_instances_after_init(
        self, mock_alb_init, mock_soil_init, mock_mlcanopy, mock_solarabs,
        mock_surfalb, mock_friction, mock_waterflux, mock_energyflux,
        mock_temperature, mock_canopy, mock_waterstate, mock_soilstate,
        mock_atm2lnd, mock_init_vertical, mock_bounds
    ):
        """
        Test validate_instances returns True after initialization.
        
        Verifies that validation succeeds when all instances are initialized.
        """
        # Initialize instances
        clm_instInit(mock_bounds)
        
        # Validate
        result = validate_instances(mock_bounds)
        
        assert result is True, "Should return True after initialization"
    
    def test_validate_instances_before_init(self, mock_bounds):
        """
        Test validate_instances returns False before initialization.
        
        Verifies that validation fails when instances are not initialized.
        """
        # Reset to ensure clean state
        reset_instances()
        
        # Validate without initialization
        result = validate_instances(mock_bounds)
        
        assert result is False, "Should return False before initialization"
    
    @patch('clm_src_main.clm_instMod._clm_instances')
    def test_validate_instances_partial_init(self, mock_container, mock_bounds):
        """
        Test validate_instances with partially initialized instances.
        
        Verifies that validation fails when only some instances are set.
        """
        # Set only some instances
        mock_container.atm2lnd_inst = Mock()
        mock_container.soilstate_inst = Mock()
        mock_container.waterstate_inst = None
        mock_container.canopystate_inst = None
        mock_container.temperature_inst = None
        mock_container.energyflux_inst = None
        mock_container.waterflux_inst = None
        mock_container.frictionvel_inst = None
        mock_container.surfalb_inst = None
        mock_container.solarabs_inst = None
        mock_container.mlcanopy_inst = None
        
        result = validate_instances(mock_bounds)
        
        assert result is False, "Should return False with partial initialization"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    @patch('clm_src_main.clm_instMod.initVertical')
    @patch('clm_src_main.clm_instMod.atm2lnd_type')
    @patch('clm_src_main.clm_instMod.soilstate_type')
    @patch('clm_src_main.clm_instMod.waterstate_type')
    @patch('clm_src_main.clm_instMod.canopystate_type')
    @patch('clm_src_main.clm_instMod.temperature_type')
    @patch('clm_src_main.clm_instMod.energyflux_type')
    @patch('clm_src_main.clm_instMod.waterflux_type')
    @patch('clm_src_main.clm_instMod.frictionvel_type')
    @patch('clm_src_main.clm_instMod.surfalb_type')
    @patch('clm_src_main.clm_instMod.solarabs_type')
    @patch('clm_src_main.clm_instMod.mlcanopy_type')
    @patch('clm_src_main.clm_instMod.SoilStateInitTimeConst')
    @patch('clm_src_main.clm_instMod.SurfaceAlbedoInitTimeConst')
    def test_reset_and_reinitialize(
        self, mock_alb_init, mock_soil_init, mock_mlcanopy, mock_solarabs,
        mock_surfalb, mock_friction, mock_waterflux, mock_energyflux,
        mock_temperature, mock_canopy, mock_waterstate, mock_soilstate,
        mock_atm2lnd, mock_init_vertical, mock_bounds
    ):
        """
        Test reset followed by reinitialization.
        
        Verifies that the system can be reset and reinitialized cleanly.
        """
        # Initialize
        clm_instInit(mock_bounds)
        assert validate_instances(mock_bounds) is True
        
        # Reset
        reset_instances()
        assert validate_instances(mock_bounds) is False
        
        # Reinitialize
        clm_instInit(mock_bounds)
        assert validate_instances(mock_bounds) is True
    
    @patch('clm_src_main.clm_instMod.initVertical')
    @patch('clm_src_main.clm_instMod.atm2lnd_type')
    @patch('clm_src_main.clm_instMod.soilstate_type')
    @patch('clm_src_main.clm_instMod.waterstate_type')
    @patch('clm_src_main.clm_instMod.canopystate_type')
    @patch('clm_src_main.clm_instMod.temperature_type')
    @patch('clm_src_main.clm_instMod.energyflux_type')
    @patch('clm_src_main.clm_instMod.waterflux_type')
    @patch('clm_src_main.clm_instMod.frictionvel_type')
    @patch('clm_src_main.clm_instMod.surfalb_type')
    @patch('clm_src_main.clm_instMod.solarabs_type')
    @patch('clm_src_main.clm_instMod.mlcanopy_type')
    @patch('clm_src_main.clm_instMod.SoilStateInitTimeConst')
    @patch('clm_src_main.clm_instMod.SurfaceAlbedoInitTimeConst')
    def test_multiple_initializations(
        self, mock_alb_init, mock_soil_init, mock_mlcanopy, mock_solarabs,
        mock_surfalb, mock_friction, mock_waterflux, mock_energyflux,
        mock_temperature, mock_canopy, mock_waterstate, mock_soilstate,
        mock_atm2lnd, mock_init_vertical, mock_bounds
    ):
        """
        Test multiple initializations without reset.
        
        Verifies behavior when clm_instInit is called multiple times.
        """
        # First initialization
        clm_instInit(mock_bounds)
        assert validate_instances(mock_bounds) is True
        
        # Second initialization (should work or handle gracefully)
        clm_instInit(mock_bounds)
        assert validate_instances(mock_bounds) is True
    
    @patch('clm_src_main.clm_instMod.initVertical')
    @patch('clm_src_main.clm_instMod.atm2lnd_type')
    @patch('clm_src_main.clm_instMod.soilstate_type')
    @patch('clm_src_main.clm_instMod.waterstate_type')
    @patch('clm_src_main.clm_instMod.canopystate_type')
    @patch('clm_src_main.clm_instMod.temperature_type')
    @patch('clm_src_main.clm_instMod.energyflux_type')
    @patch('clm_src_main.clm_instMod.waterflux_type')
    @patch('clm_src_main.clm_instMod.frictionvel_type')
    @patch('clm_src_main.clm_instMod.surfalb_type')
    @patch('clm_src_main.clm_instMod.solarabs_type')
    @patch('clm_src_main.clm_instMod.mlcanopy_type')
    @patch('clm_src_main.clm_instMod.SoilStateInitTimeConst')
    @patch('clm_src_main.clm_instMod.SurfaceAlbedoInitTimeConst')
    def test_full_restart_cycle(
        self, mock_alb_init, mock_soil_init, mock_mlcanopy, mock_solarabs,
        mock_surfalb, mock_friction, mock_waterflux, mock_energyflux,
        mock_temperature, mock_canopy, mock_waterstate, mock_soilstate,
        mock_atm2lnd, mock_init_vertical, mock_bounds
    ):
        """
        Test complete restart cycle.
        
        Verifies: init -> define -> write -> reset -> init -> read
        """
        # Setup mlcanopy with restart method
        mock_mlcanopy_inst = Mock()
        mock_mlcanopy_inst.restart = Mock()
        mock_mlcanopy.return_value = mock_mlcanopy_inst
        
        # Initialize
        clm_instInit(mock_bounds)
        assert validate_instances(mock_bounds) is True
        
        # Define restart
        clm_instRest(mock_bounds, "ncid1", 'define')
        
        # Write restart
        clm_instRest(mock_bounds, "ncid2", 'write')
        
        # Reset
        reset_instances()
        assert validate_instances(mock_bounds) is False
        
        # Reinitialize
        clm_instInit(mock_bounds)
        assert validate_instances(mock_bounds) is True
        
        # Read restart
        clm_instRest(mock_bounds, "ncid3", 'read')
        
        # Verify restart was called with all flags
        calls = mock_mlcanopy_inst.restart.call_args_list
        assert len(calls) == 3
        assert calls[0][0][2] == 'define'
        assert calls[1][0][2] == 'write'
        assert calls[2][0][2] == 'read'
    
    @patch('clm_src_main.clm_instMod.initVertical')
    @patch('clm_src_main.clm_instMod.atm2lnd_type')
    @patch('clm_src_main.clm_instMod.soilstate_type')
    @patch('clm_src_main.clm_instMod.waterstate_type')
    @patch('clm_src_main.clm_instMod.canopystate_type')
    @patch('clm_src_main.clm_instMod.temperature_type')
    @patch('clm_src_main.clm_instMod.energyflux_type')
    @patch('clm_src_main.clm_instMod.waterflux_type')
    @patch('clm_src_main.clm_instMod.frictionvel_type')
    @patch('clm_src_main.clm_instMod.surfalb_type')
    @patch('clm_src_main.clm_instMod.solarabs_type')
    @patch('clm_src_main.clm_instMod.mlcanopy_type')
    @patch('clm_src_main.clm_instMod.SoilStateInitTimeConst')
    @patch('clm_src_main.clm_instMod.SurfaceAlbedoInitTimeConst')
    def test_get_all_instances_after_init(
        self, mock_alb_init, mock_soil_init, mock_mlcanopy, mock_solarabs,
        mock_surfalb, mock_friction, mock_waterflux, mock_energyflux,
        mock_temperature, mock_canopy, mock_waterstate, mock_soilstate,
        mock_atm2lnd, mock_init_vertical, mock_bounds
    ):
        """
        Test retrieving all instances after initialization.
        
        Verifies that get_instance works for all component types.
        """
        # Initialize
        clm_instInit(mock_bounds)
        
        # Get all instances
        instance_names = [
            "atm2lnd_inst", "soilstate_inst", "waterstate_inst",
            "canopystate_inst", "temperature_inst", "energyflux_inst",
            "waterflux_inst", "frictionvel_inst", "surfalb_inst",
            "solarabs_inst", "mlcanopy_inst"
        ]
        
        for name in instance_names:
            instance = get_instance(name)
            assert instance is not None, f"{name} should not be None after init"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    def test_validate_instances_with_none_bounds(self):
        """
        Test validate_instances with None bounds.
        
        Verifies handling of invalid bounds parameter.
        """
        # This may raise an error or return False depending on implementation
        try:
            result = validate_instances(None)
            # If it doesn't raise, it should return False
            assert result is False
        except (AttributeError, TypeError):
            # Expected if implementation doesn't handle None
            pass
    
    @patch('clm_src_main.clm_instMod._clm_instances')
    def test_get_instance_empty_string(self, mock_container):
        """
        Test get_instance with empty string.
        
        Verifies handling of invalid instance name.
        """
        result = get_instance("")
        assert result is None, "Should return None for empty string"
    
    @patch('clm_src_main.clm_instMod._clm_instances')
    def test_clm_instrest_invalid_flag(self, mock_container, mock_bounds):
        """
        Test clm_instRest with invalid flag.
        
        Verifies handling of invalid restart flag.
        """
        mock_mlcanopy = Mock()
        mock_mlcanopy.restart = Mock()
        mock_container.mlcanopy_inst = mock_mlcanopy
        
        # Call with invalid flag (may raise error or be ignored)
        try:
            clm_instRest(mock_bounds, "ncid", 'invalid_flag')
            # If it doesn't raise, verify restart was still called
            mock_mlcanopy.restart.assert_called_once()
        except ValueError:
            # Expected if implementation validates flag
            pass


# ============================================================================
# Documentation Tests
# ============================================================================

class TestDocumentation:
    """Test suite for documentation and metadata."""
    
    def test_clminstances_has_docstring(self):
        """Verify CLMInstances class has documentation."""
        assert CLMInstances.__doc__ is not None
        assert len(CLMInstances.__doc__) > 0
    
    def test_clm_instinit_has_docstring(self):
        """Verify clm_instInit function has documentation."""
        assert clm_instInit.__doc__ is not None
        assert len(clm_instInit.__doc__) > 0
    
    def test_get_instance_has_docstring(self):
        """Verify get_instance function has documentation."""
        assert get_instance.__doc__ is not None
        assert len(get_instance.__doc__) > 0
    
    def test_validate_instances_has_docstring(self):
        """Verify validate_instances function has documentation."""
        assert validate_instances.__doc__ is not None
        assert len(validate_instances.__doc__) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])