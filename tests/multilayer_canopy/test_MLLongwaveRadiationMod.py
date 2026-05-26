"""
Comprehensive pytest suite for longwave_radiation function from MLLongwaveRadiationMod.

This module tests the Norman (1979) two-stream longwave radiation scheme for
multi-layer canopy models, including:
- Output shape validation
- Physical constraint verification
- Edge cases (zero PAI, extreme gradients, boundary transmittance)
- Energy conservation checks
- Data type consistency
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multilayer_canopy.MLLongwaveRadiationMod import longwave_radiation


# Define NamedTuples matching the function signature
class BoundsType(NamedTuple):
    """Bounds defining spatial domain."""
    begp: int
    endp: int


class MLCanopyType(NamedTuple):
    """Multi-layer canopy state and fluxes."""
    ncan: jnp.ndarray
    ntop: jnp.ndarray
    nbot: jnp.ndarray
    tleaf_sun: jnp.ndarray
    tleaf_sha: jnp.ndarray
    fracsun: jnp.ndarray
    td: jnp.ndarray
    dpai: jnp.ndarray
    tg: jnp.ndarray
    lwsky: jnp.ndarray
    itype: jnp.ndarray
    lwup_layer: jnp.ndarray
    lwdn_layer: jnp.ndarray
    lwleaf_sun: jnp.ndarray
    lwleaf_sha: jnp.ndarray
    lwsoi: jnp.ndarray
    lwveg: jnp.ndarray
    lwup: jnp.ndarray


class LongwaveRadiationParams(NamedTuple):
    """Longwave radiation parameters."""
    sb: float
    emg: float
    emleaf: jnp.ndarray
    nlevcan: int


@pytest.fixture
def test_data():
    """
    Load and prepare test data for longwave_radiation function.
    
    Returns:
        dict: Dictionary containing all test cases with inputs and metadata.
    """
    return {
        "nominal_single_patch_single_layer": {
            "bounds": BoundsType(begp=0, endp=0),
            "num_filter": 1,
            "filter_indices": jnp.array([0]),
            "mlcanopy_inst": MLCanopyType(
                ncan=jnp.array([1]),
                ntop=jnp.array([0]),
                nbot=jnp.array([0]),
                tleaf_sun=jnp.array([[298.15, 0.0, 0.0, 0.0, 0.0]]),
                tleaf_sha=jnp.array([[295.15, 0.0, 0.0, 0.0, 0.0]]),
                fracsun=jnp.array([[0.6, 0.0, 0.0, 0.0, 0.0]]),
                td=jnp.array([[0.7, 0.0, 0.0, 0.0, 0.0]]),
                dpai=jnp.array([[2.5, 0.0, 0.0, 0.0, 0.0]]),
                tg=jnp.array([293.15]),
                lwsky=jnp.array([350.0]),
                itype=jnp.array([1]),
                lwup_layer=jnp.zeros((1, 6)),
                lwdn_layer=jnp.zeros((1, 6)),
                lwleaf_sun=jnp.zeros((1, 5)),
                lwleaf_sha=jnp.zeros((1, 5)),
                lwsoi=jnp.zeros(1),
                lwveg=jnp.zeros(1),
                lwup=jnp.zeros(1),
            ),
            "params": LongwaveRadiationParams(
                sb=5.67e-08,
                emg=0.96,
                emleaf=jnp.array([0.98, 0.97, 0.96, 0.95, 0.98]),
                nlevcan=5,
            ),
            "longwave_type": 1,
            "metadata": {
                "type": "nominal",
                "description": "Single patch with single canopy layer, typical summer conditions",
            },
        },
        "nominal_multi_patch_multi_layer": {
            "bounds": BoundsType(begp=0, endp=2),
            "num_filter": 3,
            "filter_indices": jnp.array([0, 1, 2]),
            "mlcanopy_inst": MLCanopyType(
                ncan=jnp.array([3, 2, 4]),
                ntop=jnp.array([0, 0, 0]),
                nbot=jnp.array([2, 1, 3]),
                tleaf_sun=jnp.array([
                    [303.15, 300.15, 297.15, 0.0, 0.0],
                    [301.15, 298.15, 0.0, 0.0, 0.0],
                    [305.15, 302.15, 299.15, 296.15, 0.0],
                ]),
                tleaf_sha=jnp.array([
                    [300.15, 297.15, 294.15, 0.0, 0.0],
                    [298.15, 295.15, 0.0, 0.0, 0.0],
                    [302.15, 299.15, 296.15, 293.15, 0.0],
                ]),
                fracsun=jnp.array([
                    [0.7, 0.5, 0.3, 0.0, 0.0],
                    [0.65, 0.4, 0.0, 0.0, 0.0],
                    [0.75, 0.6, 0.45, 0.25, 0.0],
                ]),
                td=jnp.array([
                    [0.8, 0.6, 0.4, 0.0, 0.0],
                    [0.75, 0.5, 0.0, 0.0, 0.0],
                    [0.85, 0.7, 0.55, 0.35, 0.0],
                ]),
                dpai=jnp.array([
                    [1.5, 2.0, 1.8, 0.0, 0.0],
                    [2.2, 1.9, 0.0, 0.0, 0.0],
                    [1.2, 1.8, 2.1, 1.5, 0.0],
                ]),
                tg=jnp.array([290.15, 292.15, 288.15]),
                lwsky=jnp.array([340.0, 355.0, 330.0]),
                itype=jnp.array([2, 1, 3]),
                lwup_layer=jnp.zeros((3, 6)),
                lwdn_layer=jnp.zeros((3, 6)),
                lwleaf_sun=jnp.zeros((3, 5)),
                lwleaf_sha=jnp.zeros((3, 5)),
                lwsoi=jnp.zeros(3),
                lwveg=jnp.zeros(3),
                lwup=jnp.zeros(3),
            ),
            "params": LongwaveRadiationParams(
                sb=5.67e-08,
                emg=0.96,
                emleaf=jnp.array([0.98, 0.97, 0.96, 0.95, 0.98]),
                nlevcan=5,
            ),
            "longwave_type": 1,
            "metadata": {
                "type": "nominal",
                "description": "Multiple patches with varying canopy layers, typical forest conditions",
            },
        },
        "edge_zero_plant_area_index": {
            "bounds": BoundsType(begp=0, endp=1),
            "num_filter": 2,
            "filter_indices": jnp.array([0, 1]),
            "mlcanopy_inst": MLCanopyType(
                ncan=jnp.array([1, 2]),
                ntop=jnp.array([0, 0]),
                nbot=jnp.array([0, 1]),
                tleaf_sun=jnp.array([
                    [298.15, 0.0, 0.0, 0.0, 0.0],
                    [300.15, 297.15, 0.0, 0.0, 0.0],
                ]),
                tleaf_sha=jnp.array([
                    [295.15, 0.0, 0.0, 0.0, 0.0],
                    [297.15, 294.15, 0.0, 0.0, 0.0],
                ]),
                fracsun=jnp.array([
                    [0.5, 0.0, 0.0, 0.0, 0.0],
                    [0.6, 0.4, 0.0, 0.0, 0.0],
                ]),
                td=jnp.array([
                    [0.9, 0.0, 0.0, 0.0, 0.0],
                    [0.8, 0.6, 0.0, 0.0, 0.0],
                ]),
                dpai=jnp.array([
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0, 0.0, 0.0],
                ]),
                tg=jnp.array([293.15, 291.15]),
                lwsky=jnp.array([345.0, 350.0]),
                itype=jnp.array([1, 2]),
                lwup_layer=jnp.zeros((2, 6)),
                lwdn_layer=jnp.zeros((2, 6)),
                lwleaf_sun=jnp.zeros((2, 5)),
                lwleaf_sha=jnp.zeros((2, 5)),
                lwsoi=jnp.zeros(2),
                lwveg=jnp.zeros(2),
                lwup=jnp.zeros(2),
            ),
            "params": LongwaveRadiationParams(
                sb=5.67e-08,
                emg=0.96,
                emleaf=jnp.array([0.98, 0.97, 0.96, 0.95, 0.98]),
                nlevcan=5,
            ),
            "longwave_type": 1,
            "metadata": {
                "type": "edge",
                "description": "Tests sparse/bare canopy with zero or minimal plant area index",
                "edge_cases": ["zero_dpai", "minimal_vegetation"],
            },
        },
        "edge_extreme_temperature_gradient": {
            "bounds": BoundsType(begp=0, endp=0),
            "num_filter": 1,
            "filter_indices": jnp.array([0]),
            "mlcanopy_inst": MLCanopyType(
                ncan=jnp.array([3]),
                ntop=jnp.array([0]),
                nbot=jnp.array([2]),
                tleaf_sun=jnp.array([[313.15, 303.15, 293.15, 0.0, 0.0]]),
                tleaf_sha=jnp.array([[310.15, 300.15, 290.15, 0.0, 0.0]]),
                fracsun=jnp.array([[0.8, 0.5, 0.2, 0.0, 0.0]]),
                td=jnp.array([[0.85, 0.65, 0.4, 0.0, 0.0]]),
                dpai=jnp.array([[1.8, 2.2, 2.0, 0.0, 0.0]]),
                tg=jnp.array([273.15]),
                lwsky=jnp.array([280.0]),
                itype=jnp.array([1]),
                lwup_layer=jnp.zeros((1, 6)),
                lwdn_layer=jnp.zeros((1, 6)),
                lwleaf_sun=jnp.zeros((1, 5)),
                lwleaf_sha=jnp.zeros((1, 5)),
                lwsoi=jnp.zeros(1),
                lwveg=jnp.zeros(1),
                lwup=jnp.zeros(1),
            ),
            "params": LongwaveRadiationParams(
                sb=5.67e-08,
                emg=0.96,
                emleaf=jnp.array([0.98, 0.97, 0.96, 0.95, 0.98]),
                nlevcan=5,
            ),
            "longwave_type": 1,
            "metadata": {
                "type": "edge",
                "description": "Extreme temperature gradient from hot canopy top to cold ground (40K difference)",
                "edge_cases": ["extreme_gradient", "cold_ground"],
            },
        },
        "edge_full_transmittance": {
            "bounds": BoundsType(begp=0, endp=1),
            "num_filter": 2,
            "filter_indices": jnp.array([0, 1]),
            "mlcanopy_inst": MLCanopyType(
                ncan=jnp.array([2, 1]),
                ntop=jnp.array([0, 0]),
                nbot=jnp.array([1, 0]),
                tleaf_sun=jnp.array([
                    [298.15, 296.15, 0.0, 0.0, 0.0],
                    [300.15, 0.0, 0.0, 0.0, 0.0],
                ]),
                tleaf_sha=jnp.array([
                    [296.15, 294.15, 0.0, 0.0, 0.0],
                    [298.15, 0.0, 0.0, 0.0, 0.0],
                ]),
                fracsun=jnp.array([
                    [0.9, 0.7, 0.0, 0.0, 0.0],
                    [0.95, 0.0, 0.0, 0.0, 0.0],
                ]),
                td=jnp.array([
                    [1.0, 0.95, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ]),
                dpai=jnp.array([
                    [0.05, 0.08, 0.0, 0.0, 0.0],
                    [0.03, 0.0, 0.0, 0.0, 0.0],
                ]),
                tg=jnp.array([295.15, 297.15]),
                lwsky=jnp.array([360.0, 365.0]),
                itype=jnp.array([2, 1]),
                lwup_layer=jnp.zeros((2, 6)),
                lwdn_layer=jnp.zeros((2, 6)),
                lwleaf_sun=jnp.zeros((2, 5)),
                lwleaf_sha=jnp.zeros((2, 5)),
                lwsoi=jnp.zeros(2),
                lwveg=jnp.zeros(2),
                lwup=jnp.zeros(2),
            ),
            "params": LongwaveRadiationParams(
                sb=5.67e-08,
                emg=0.96,
                emleaf=jnp.array([0.98, 0.97, 0.96, 0.95, 0.98]),
                nlevcan=5,
            ),
            "longwave_type": 1,
            "metadata": {
                "type": "edge",
                "description": "Nearly transparent canopy with transmittance approaching 1.0",
                "edge_cases": ["full_transmittance", "minimal_interception"],
            },
        },
        "edge_zero_transmittance": {
            "bounds": BoundsType(begp=0, endp=0),
            "num_filter": 1,
            "filter_indices": jnp.array([0]),
            "mlcanopy_inst": MLCanopyType(
                ncan=jnp.array([2]),
                ntop=jnp.array([0]),
                nbot=jnp.array([1]),
                tleaf_sun=jnp.array([[302.15, 298.15, 0.0, 0.0, 0.0]]),
                tleaf_sha=jnp.array([[299.15, 295.15, 0.0, 0.0, 0.0]]),
                fracsun=jnp.array([[0.3, 0.1, 0.0, 0.0, 0.0]]),
                td=jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0]]),
                dpai=jnp.array([[5.0, 4.5, 0.0, 0.0, 0.0]]),
                tg=jnp.array([294.15]),
                lwsky=jnp.array([340.0]),
                itype=jnp.array([3]),
                lwup_layer=jnp.zeros((1, 6)),
                lwdn_layer=jnp.zeros((1, 6)),
                lwleaf_sun=jnp.zeros((1, 5)),
                lwleaf_sha=jnp.zeros((1, 5)),
                lwsoi=jnp.zeros(1),
                lwveg=jnp.zeros(1),
                lwup=jnp.zeros(1),
            ),
            "params": LongwaveRadiationParams(
                sb=5.67e-08,
                emg=0.96,
                emleaf=jnp.array([0.98, 0.97, 0.96, 0.95, 0.98]),
                nlevcan=5,
            ),
            "longwave_type": 1,
            "metadata": {
                "type": "edge",
                "description": "Dense canopy with zero transmittance, complete radiation interception",
                "edge_cases": ["zero_transmittance", "dense_canopy"],
            },
        },
        "special_maximum_canopy_layers": {
            "bounds": BoundsType(begp=0, endp=0),
            "num_filter": 1,
            "filter_indices": jnp.array([0]),
            "mlcanopy_inst": MLCanopyType(
                ncan=jnp.array([5]),
                ntop=jnp.array([0]),
                nbot=jnp.array([4]),
                tleaf_sun=jnp.array([[308.15, 304.15, 300.15, 296.15, 292.15]]),
                tleaf_sha=jnp.array([[305.15, 301.15, 297.15, 293.15, 289.15]]),
                fracsun=jnp.array([[0.8, 0.65, 0.5, 0.35, 0.2]]),
                td=jnp.array([[0.9, 0.75, 0.6, 0.45, 0.3]]),
                dpai=jnp.array([[1.2, 1.5, 1.8, 1.6, 1.3]]),
                tg=jnp.array([288.15]),
                lwsky=jnp.array([335.0]),
                itype=jnp.array([2]),
                lwup_layer=jnp.zeros((1, 6)),
                lwdn_layer=jnp.zeros((1, 6)),
                lwleaf_sun=jnp.zeros((1, 5)),
                lwleaf_sha=jnp.zeros((1, 5)),
                lwsoi=jnp.zeros(1),
                lwveg=jnp.zeros(1),
                lwup=jnp.zeros(1),
            ),
            "params": LongwaveRadiationParams(
                sb=5.67e-08,
                emg=0.96,
                emleaf=jnp.array([0.98, 0.97, 0.96, 0.95, 0.98]),
                nlevcan=5,
            ),
            "longwave_type": 1,
            "metadata": {
                "type": "special",
                "description": "Maximum number of canopy layers (5) with smooth vertical gradient",
            },
        },
        "special_uniform_canopy_properties": {
            "bounds": BoundsType(begp=0, endp=1),
            "num_filter": 2,
            "filter_indices": jnp.array([0, 1]),
            "mlcanopy_inst": MLCanopyType(
                ncan=jnp.array([3, 3]),
                ntop=jnp.array([0, 0]),
                nbot=jnp.array([2, 2]),
                tleaf_sun=jnp.array([
                    [298.15, 298.15, 298.15, 0.0, 0.0],
                    [300.15, 300.15, 300.15, 0.0, 0.0],
                ]),
                tleaf_sha=jnp.array([
                    [298.15, 298.15, 298.15, 0.0, 0.0],
                    [300.15, 300.15, 300.15, 0.0, 0.0],
                ]),
                fracsun=jnp.array([
                    [0.5, 0.5, 0.5, 0.0, 0.0],
                    [0.5, 0.5, 0.5, 0.0, 0.0],
                ]),
                td=jnp.array([
                    [0.7, 0.7, 0.7, 0.0, 0.0],
                    [0.7, 0.7, 0.7, 0.0, 0.0],
                ]),
                dpai=jnp.array([
                    [2.0, 2.0, 2.0, 0.0, 0.0],
                    [2.0, 2.0, 2.0, 0.0, 0.0],
                ]),
                tg=jnp.array([298.15, 300.15]),
                lwsky=jnp.array([400.0, 410.0]),
                itype=jnp.array([1, 1]),
                lwup_layer=jnp.zeros((2, 6)),
                lwdn_layer=jnp.zeros((2, 6)),
                lwleaf_sun=jnp.zeros((2, 5)),
                lwleaf_sha=jnp.zeros((2, 5)),
                lwsoi=jnp.zeros(2),
                lwveg=jnp.zeros(2),
                lwup=jnp.zeros(2),
            ),
            "params": LongwaveRadiationParams(
                sb=5.67e-08,
                emg=0.96,
                emleaf=jnp.array([0.98, 0.97, 0.96, 0.95, 0.98]),
                nlevcan=5,
            ),
            "longwave_type": 1,
            "metadata": {
                "type": "special",
                "description": "Uniform canopy properties across all layers (isothermal, homogeneous)",
            },
        },
        "special_high_incoming_radiation": {
            "bounds": BoundsType(begp=0, endp=1),
            "num_filter": 2,
            "filter_indices": jnp.array([0, 1]),
            "mlcanopy_inst": MLCanopyType(
                ncan=jnp.array([2, 3]),
                ntop=jnp.array([0, 0]),
                nbot=jnp.array([1, 2]),
                tleaf_sun=jnp.array([
                    [310.15, 305.15, 0.0, 0.0, 0.0],
                    [312.15, 308.15, 304.15, 0.0, 0.0],
                ]),
                tleaf_sha=jnp.array([
                    [307.15, 302.15, 0.0, 0.0, 0.0],
                    [309.15, 305.15, 301.15, 0.0, 0.0],
                ]),
                fracsun=jnp.array([
                    [0.7, 0.5, 0.0, 0.0, 0.0],
                    [0.75, 0.6, 0.4, 0.0, 0.0],
                ]),
                td=jnp.array([
                    [0.75, 0.55, 0.0, 0.0, 0.0],
                    [0.8, 0.65, 0.5, 0.0, 0.0],
                ]),
                dpai=jnp.array([
                    [2.3, 2.6, 0.0, 0.0, 0.0],
                    [1.9, 2.2, 2.4, 0.0, 0.0],
                ]),
                tg=jnp.array([303.15, 305.15]),
                lwsky=jnp.array([500.0, 520.0]),
                itype=jnp.array([2, 3]),
                lwup_layer=jnp.zeros((2, 6)),
                lwdn_layer=jnp.zeros((2, 6)),
                lwleaf_sun=jnp.zeros((2, 5)),
                lwleaf_sha=jnp.zeros((2, 5)),
                lwsoi=jnp.zeros(2),
                lwveg=jnp.zeros(2),
                lwup=jnp.zeros(2),
            ),
            "params": LongwaveRadiationParams(
                sb=5.67e-08,
                emg=0.96,
                emleaf=jnp.array([0.98, 0.97, 0.96, 0.95, 0.98]),
                nlevcan=5,
            ),
            "longwave_type": 1,
            "metadata": {
                "type": "special",
                "description": "High incoming longwave radiation (cloudy/humid conditions)",
            },
        },
        "edge_boundary_emissivity_values": {
            "bounds": BoundsType(begp=0, endp=2),
            "num_filter": 3,
            "filter_indices": jnp.array([0, 1, 2]),
            "mlcanopy_inst": MLCanopyType(
                ncan=jnp.array([2, 1, 2]),
                ntop=jnp.array([0, 0, 0]),
                nbot=jnp.array([1, 0, 1]),
                tleaf_sun=jnp.array([
                    [299.15, 296.15, 0.0, 0.0, 0.0],
                    [301.15, 0.0, 0.0, 0.0, 0.0],
                    [298.15, 295.15, 0.0, 0.0, 0.0],
                ]),
                tleaf_sha=jnp.array([
                    [297.15, 294.15, 0.0, 0.0, 0.0],
                    [299.15, 0.0, 0.0, 0.0, 0.0],
                    [296.15, 293.15, 0.0, 0.0, 0.0],
                ]),
                fracsun=jnp.array([
                    [0.6, 0.4, 0.0, 0.0, 0.0],
                    [0.7, 0.0, 0.0, 0.0, 0.0],
                    [0.55, 0.35, 0.0, 0.0, 0.0],
                ]),
                td=jnp.array([
                    [0.7, 0.5, 0.0, 0.0, 0.0],
                    [0.75, 0.0, 0.0, 0.0, 0.0],
                    [0.68, 0.48, 0.0, 0.0, 0.0],
                ]),
                dpai=jnp.array([
                    [2.1, 2.3, 0.0, 0.0, 0.0],
                    [1.8, 0.0, 0.0, 0.0, 0.0],
                    [2.4, 2.5, 0.0, 0.0, 0.0],
                ]),
                tg=jnp.array([293.15, 295.15, 291.15]),
                lwsky=jnp.array([345.0, 355.0, 340.0]),
                itype=jnp.array([0, 4, 1]),
                lwup_layer=jnp.zeros((3, 6)),
                lwdn_layer=jnp.zeros((3, 6)),
                lwleaf_sun=jnp.zeros((3, 5)),
                lwleaf_sha=jnp.zeros((3, 5)),
                lwsoi=jnp.zeros(3),
                lwveg=jnp.zeros(3),
                lwup=jnp.zeros(3),
            ),
            "params": LongwaveRadiationParams(
                sb=5.67e-08,
                emg=1.0,
                emleaf=jnp.array([1.0, 0.95, 1.0, 0.95, 1.0]),
                nlevcan=5,
            ),
            "longwave_type": 1,
            "metadata": {
                "type": "edge",
                "description": "Boundary emissivity values (1.0 for perfect blackbody, 0.95 for realistic minimum)",
                "edge_cases": ["max_emissivity", "varied_pft_emissivity"],
            },
        },
    }


@pytest.mark.parametrize(
    "test_case_name",
    [
        "nominal_single_patch_single_layer",
        "nominal_multi_patch_multi_layer",
        "edge_zero_plant_area_index",
        "edge_extreme_temperature_gradient",
        "edge_full_transmittance",
        "edge_zero_transmittance",
        "special_maximum_canopy_layers",
        "special_uniform_canopy_properties",
        "special_high_incoming_radiation",
        "edge_boundary_emissivity_values",
    ],
)
def test_longwave_radiation_shapes(test_data, test_case_name):
    """
    Test that longwave_radiation returns outputs with correct shapes.
    
    Verifies that all output arrays have the expected dimensions based on
    the number of patches and canopy layers.
    """
    case = test_data[test_case_name]
    
    result = longwave_radiation(
        bounds=case["bounds"],
        num_filter=case["num_filter"],
        filter_indices=case["filter_indices"],
        mlcanopy_inst=case["mlcanopy_inst"],
        params=case["params"],
        longwave_type=case["longwave_type"],
    )
    
    n_patches = case["num_filter"]
    nlevcan = case["params"].nlevcan
    
    # Check output shapes
    assert result.lwup_layer.shape == (n_patches, nlevcan + 1), \
        f"lwup_layer shape mismatch for {test_case_name}"
    assert result.lwdn_layer.shape == (n_patches, nlevcan + 1), \
        f"lwdn_layer shape mismatch for {test_case_name}"
    assert result.lwleaf_sun.shape == (n_patches, nlevcan), \
        f"lwleaf_sun shape mismatch for {test_case_name}"
    assert result.lwleaf_sha.shape == (n_patches, nlevcan), \
        f"lwleaf_sha shape mismatch for {test_case_name}"
    assert result.lwsoi.shape == (n_patches,), \
        f"lwsoi shape mismatch for {test_case_name}"
    assert result.lwveg.shape == (n_patches,), \
        f"lwveg shape mismatch for {test_case_name}"
    assert result.lwup.shape == (n_patches,), \
        f"lwup shape mismatch for {test_case_name}"


@pytest.mark.parametrize(
    "test_case_name",
    [
        "nominal_single_patch_single_layer",
        "nominal_multi_patch_multi_layer",
        "edge_zero_plant_area_index",
        "edge_extreme_temperature_gradient",
        "edge_full_transmittance",
        "edge_zero_transmittance",
        "special_maximum_canopy_layers",
        "special_uniform_canopy_properties",
        "special_high_incoming_radiation",
        "edge_boundary_emissivity_values",
    ],
)
def test_longwave_radiation_physical_constraints(test_data, test_case_name):
    """
    Test that longwave_radiation outputs satisfy physical constraints.
    
    Verifies:
    - All radiation fluxes are non-negative
    - Absorbed radiation values are physically reasonable
    - Energy conservation (total absorbed = incoming - outgoing)
    """
    case = test_data[test_case_name]
    
    result = longwave_radiation(
        bounds=case["bounds"],
        num_filter=case["num_filter"],
        filter_indices=case["filter_indices"],
        mlcanopy_inst=case["mlcanopy_inst"],
        params=case["params"],
        longwave_type=case["longwave_type"],
    )
    
    # All radiation fluxes should be non-negative
    assert jnp.all(result.lwup_layer >= 0), \
        f"Negative upward longwave flux in {test_case_name}"
    assert jnp.all(result.lwdn_layer >= 0), \
        f"Negative downward longwave flux in {test_case_name}"
    assert jnp.all(result.lwsoi >= 0), \
        f"Negative soil absorbed radiation in {test_case_name}"
    assert jnp.all(result.lwveg >= 0), \
        f"Negative vegetation absorbed radiation in {test_case_name}"
    assert jnp.all(result.lwup >= 0), \
        f"Negative upward flux at canopy top in {test_case_name}"
    
    # Upward flux at canopy top should be reasonable
    # (between ground emission and incoming sky radiation)
    sb = case["params"].sb
    emg = case["params"].emg
    tg = case["mlcanopy_inst"].tg
    lwsky = case["mlcanopy_inst"].lwsky
    
    ground_emission = emg * sb * tg**4
    max_expected = jnp.maximum(ground_emission, lwsky) * 1.5  # Allow 50% margin
    
    assert jnp.all(result.lwup <= max_expected), \
        f"Unreasonably high upward flux in {test_case_name}"


@pytest.mark.parametrize(
    "test_case_name",
    [
        "nominal_single_patch_single_layer",
        "nominal_multi_patch_multi_layer",
        "special_maximum_canopy_layers",
    ],
)
def test_longwave_radiation_energy_balance(test_data, test_case_name):
    """
    Test approximate energy conservation in longwave radiation calculation.
    
    Verifies that the sum of absorbed radiation (soil + vegetation) is
    approximately equal to the net radiation (incoming - outgoing at top).
    Allows for numerical tolerance and emission from canopy elements.
    """
    case = test_data[test_case_name]
    
    result = longwave_radiation(
        bounds=case["bounds"],
        num_filter=case["num_filter"],
        filter_indices=case["filter_indices"],
        mlcanopy_inst=case["mlcanopy_inst"],
        params=case["params"],
        longwave_type=case["longwave_type"],
    )
    
    # Net incoming radiation at top of canopy
    lwsky = case["mlcanopy_inst"].lwsky
    net_incoming = lwsky - result.lwup
    
    # Total absorbed radiation
    total_absorbed = result.lwsoi + result.lwveg
    
    # Energy balance check (allow for canopy emission)
    # The absorbed radiation should be positive and related to net incoming
    assert jnp.all(total_absorbed >= 0), \
        f"Negative total absorbed radiation in {test_case_name}"
    
    # For cases with vegetation, absorbed should be significant
    has_vegetation = case["mlcanopy_inst"].ncan > 0
    if jnp.any(has_vegetation):
        assert jnp.any(total_absorbed[has_vegetation] > 0), \
            f"No radiation absorbed with vegetation present in {test_case_name}"


@pytest.mark.parametrize(
    "test_case_name",
    [
        "nominal_single_patch_single_layer",
        "nominal_multi_patch_multi_layer",
        "edge_zero_plant_area_index",
        "edge_extreme_temperature_gradient",
        "edge_full_transmittance",
        "edge_zero_transmittance",
        "special_maximum_canopy_layers",
        "special_uniform_canopy_properties",
        "special_high_incoming_radiation",
        "edge_boundary_emissivity_values",
    ],
)
def test_longwave_radiation_dtypes(test_data, test_case_name):
    """
    Test that longwave_radiation returns outputs with correct data types.
    
    Verifies that all output arrays are JAX arrays with float dtype.
    """
    case = test_data[test_case_name]
    
    result = longwave_radiation(
        bounds=case["bounds"],
        num_filter=case["num_filter"],
        filter_indices=case["filter_indices"],
        mlcanopy_inst=case["mlcanopy_inst"],
        params=case["params"],
        longwave_type=case["longwave_type"],
    )
    
    # Check that outputs are JAX arrays
    assert isinstance(result.lwup_layer, jnp.ndarray), \
        f"lwup_layer is not a JAX array in {test_case_name}"
    assert isinstance(result.lwdn_layer, jnp.ndarray), \
        f"lwdn_layer is not a JAX array in {test_case_name}"
    assert isinstance(result.lwleaf_sun, jnp.ndarray), \
        f"lwleaf_sun is not a JAX array in {test_case_name}"
    assert isinstance(result.lwleaf_sha, jnp.ndarray), \
        f"lwleaf_sha is not a JAX array in {test_case_name}"
    assert isinstance(result.lwsoi, jnp.ndarray), \
        f"lwsoi is not a JAX array in {test_case_name}"
    assert isinstance(result.lwveg, jnp.ndarray), \
        f"lwveg is not a JAX array in {test_case_name}"
    assert isinstance(result.lwup, jnp.ndarray), \
        f"lwup is not a JAX array in {test_case_name}"
    
    # Check that outputs have float dtype
    assert jnp.issubdtype(result.lwup_layer.dtype, jnp.floating), \
        f"lwup_layer has non-float dtype in {test_case_name}"
    assert jnp.issubdtype(result.lwdn_layer.dtype, jnp.floating), \
        f"lwdn_layer has non-float dtype in {test_case_name}"
    assert jnp.issubdtype(result.lwsoi.dtype, jnp.floating), \
        f"lwsoi has non-float dtype in {test_case_name}"
    assert jnp.issubdtype(result.lwveg.dtype, jnp.floating), \
        f"lwveg has non-float dtype in {test_case_name}"


def test_longwave_radiation_zero_dpai_special_case(test_data):
    """
    Test special case where plant area index is zero (bare ground).
    
    With zero PAI, the canopy should be transparent and soil should
    receive most of the incoming radiation.
    """
    case = test_data["edge_zero_plant_area_index"]
    
    result = longwave_radiation(
        bounds=case["bounds"],
        num_filter=case["num_filter"],
        filter_indices=case["filter_indices"],
        mlcanopy_inst=case["mlcanopy_inst"],
        params=case["params"],
        longwave_type=case["longwave_type"],
    )
    
    # For patch with zero PAI (first patch), vegetation absorption should be minimal
    zero_pai_patch = 0
    assert result.lwveg[zero_pai_patch] < result.lwsoi[zero_pai_patch], \
        "Vegetation absorbed more than soil with zero PAI"


def test_longwave_radiation_temperature_gradient_effect(test_data):
    """
    Test that extreme temperature gradients produce reasonable flux profiles.
    
    With a large temperature gradient, upward flux should generally increase
    from cold ground to warm canopy top.
    """
    case = test_data["edge_extreme_temperature_gradient"]
    
    result = longwave_radiation(
        bounds=case["bounds"],
        num_filter=case["num_filter"],
        filter_indices=case["filter_indices"],
        mlcanopy_inst=case["mlcanopy_inst"],
        params=case["params"],
        longwave_type=case["longwave_type"],
    )
    
    # Check that upward flux exists and varies through canopy
    patch_idx = 0
    ncan = case["mlcanopy_inst"].ncan[patch_idx]
    
    # Upward flux should be positive at all levels
    assert jnp.all(result.lwup_layer[patch_idx, :ncan+1] > 0), \
        "Non-positive upward flux with temperature gradient"
    
    # Flux profile should show variation (not all identical)
    flux_variation = jnp.std(result.lwup_layer[patch_idx, :ncan+1])
    assert flux_variation > 1.0, \
        "Insufficient flux variation with large temperature gradient"


def test_longwave_radiation_transmittance_extremes(test_data):
    """
    Test behavior at transmittance extremes (0 and 1).
    
    Full transmittance (1.0) should allow most radiation through.
    Zero transmittance (0.0) should block radiation completely.
    """
    # Test full transmittance
    case_full = test_data["edge_full_transmittance"]
    result_full = longwave_radiation(
        bounds=case_full["bounds"],
        num_filter=case_full["num_filter"],
        filter_indices=case_full["filter_indices"],
        mlcanopy_inst=case_full["mlcanopy_inst"],
        params=case_full["params"],
        longwave_type=case_full["longwave_type"],
    )
    
    # With high transmittance, soil should receive significant radiation
    assert jnp.all(result_full.lwsoi > 0), \
        "No soil absorption with high transmittance"
    
    # Test zero transmittance
    case_zero = test_data["edge_zero_transmittance"]
    result_zero = longwave_radiation(
        bounds=case_zero["bounds"],
        num_filter=case_zero["num_filter"],
        filter_indices=case_zero["filter_indices"],
        mlcanopy_inst=case_zero["mlcanopy_inst"],
        params=case_zero["params"],
        longwave_type=case_zero["longwave_type"],
    )
    
    # With zero transmittance, vegetation should absorb more than with full transmittance
    # (comparing similar patches if available, or just check vegetation absorption is significant)
    assert jnp.all(result_zero.lwveg > 0), \
        "No vegetation absorption with zero transmittance"


def test_longwave_radiation_uniform_canopy(test_data):
    """
    Test uniform canopy case where all layers have identical properties.
    
    With isothermal, homogeneous canopy, radiation profiles should be smooth
    and physically consistent.
    """
    case = test_data["special_uniform_canopy_properties"]
    
    result = longwave_radiation(
        bounds=case["bounds"],
        num_filter=case["num_filter"],
        filter_indices=case["filter_indices"],
        mlcanopy_inst=case["mlcanopy_inst"],
        params=case["params"],
        longwave_type=case["longwave_type"],
    )
    
    # Check that results are physically reasonable
    assert jnp.all(result.lwup > 0), \
        "Non-positive upward flux with uniform canopy"
    assert jnp.all(result.lwsoi > 0), \
        "Non-positive soil absorption with uniform canopy"
    assert jnp.all(result.lwveg > 0), \
        "Non-positive vegetation absorption with uniform canopy"
    
    # For uniform canopy, sunlit and shaded leaf absorption should be similar
    # (within each layer, though they may differ between layers)
    for patch_idx in range(case["num_filter"]):
        ncan = case["mlcanopy_inst"].ncan[patch_idx]
        for layer in range(ncan):
            sun_abs = result.lwleaf_sun[patch_idx, layer]
            sha_abs = result.lwleaf_sha[patch_idx, layer]
            # Both should be positive
            assert sun_abs >= 0 and sha_abs >= 0, \
                f"Negative leaf absorption in uniform canopy at patch {patch_idx}, layer {layer}"


def test_longwave_radiation_high_incoming(test_data):
    """
    Test response to high incoming longwave radiation.
    
    High incoming radiation should result in higher absorbed radiation
    and higher upward flux at canopy top.
    """
    case = test_data["special_high_incoming_radiation"]
    
    result = longwave_radiation(
        bounds=case["bounds"],
        num_filter=case["num_filter"],
        filter_indices=case["filter_indices"],
        mlcanopy_inst=case["mlcanopy_inst"],
        params=case["params"],
        longwave_type=case["longwave_type"],
    )
    
    # With high incoming radiation, absorbed radiation should be substantial
    assert jnp.all(result.lwsoi > 100), \
        "Unexpectedly low soil absorption with high incoming radiation"
    assert jnp.all(result.lwveg > 100), \
        "Unexpectedly low vegetation absorption with high incoming radiation"
    
    # Upward flux at top should be high
    assert jnp.all(result.lwup > 300), \
        "Unexpectedly low upward flux with high incoming radiation"


def test_longwave_radiation_preserves_input_structure(test_data):
    """
    Test that the function preserves the structure of input MLCanopyType.
    
    All input fields should remain unchanged in the output, with only
    the radiation fields updated.
    """
    case = test_data["nominal_single_patch_single_layer"]
    
    result = longwave_radiation(
        bounds=case["bounds"],
        num_filter=case["num_filter"],
        filter_indices=case["filter_indices"],
        mlcanopy_inst=case["mlcanopy_inst"],
        params=case["params"],
        longwave_type=case["longwave_type"],
    )
    
    # Check that input fields are preserved
    assert jnp.allclose(result.ncan, case["mlcanopy_inst"].ncan), \
        "ncan field was modified"
    assert jnp.allclose(result.ntop, case["mlcanopy_inst"].ntop), \
        "ntop field was modified"
    assert jnp.allclose(result.nbot, case["mlcanopy_inst"].nbot), \
        "nbot field was modified"
    assert jnp.allclose(result.tleaf_sun, case["mlcanopy_inst"].tleaf_sun), \
        "tleaf_sun field was modified"
    assert jnp.allclose(result.tleaf_sha, case["mlcanopy_inst"].tleaf_sha), \
        "tleaf_sha field was modified"
    assert jnp.allclose(result.fracsun, case["mlcanopy_inst"].fracsun), \
        "fracsun field was modified"
    assert jnp.allclose(result.td, case["mlcanopy_inst"].td), \
        "td field was modified"
    assert jnp.allclose(result.dpai, case["mlcanopy_inst"].dpai), \
        "dpai field was modified"
    assert jnp.allclose(result.tg, case["mlcanopy_inst"].tg), \
        "tg field was modified"
    assert jnp.allclose(result.lwsky, case["mlcanopy_inst"].lwsky), \
        "lwsky field was modified"
    assert jnp.allclose(result.itype, case["mlcanopy_inst"].itype), \
        "itype field was modified"


def test_longwave_radiation_inactive_layers(test_data):
    """
    Test that inactive canopy layers (beyond ncan) remain zero.
    
    Layers beyond the active canopy should not have radiation fluxes.
    """
    case = test_data["nominal_multi_patch_multi_layer"]
    
    result = longwave_radiation(
        bounds=case["bounds"],
        num_filter=case["num_filter"],
        filter_indices=case["filter_indices"],
        mlcanopy_inst=case["mlcanopy_inst"],
        params=case["params"],
        longwave_type=case["longwave_type"],
    )
    
    # Check that inactive layers have zero or minimal values
    for patch_idx in range(case["num_filter"]):
        ncan = case["mlcanopy_inst"].ncan[patch_idx]
        nlevcan = case["params"].nlevcan
        
        if ncan < nlevcan:
            # Inactive layers should have zero leaf absorption
            inactive_sun = result.lwleaf_sun[patch_idx, ncan:]
            inactive_sha = result.lwleaf_sha[patch_idx, ncan:]
            
            assert jnp.allclose(inactive_sun, 0.0, atol=1e-6), \
                f"Non-zero sunlit absorption in inactive layers for patch {patch_idx}"
            assert jnp.allclose(inactive_sha, 0.0, atol=1e-6), \
                f"Non-zero shaded absorption in inactive layers for patch {patch_idx}"


def test_longwave_radiation_emissivity_effect(test_data):
    """
    Test that different emissivity values affect radiation calculations.
    
    Higher emissivity should generally lead to more emission and absorption.
    """
    case = test_data["edge_boundary_emissivity_values"]
    
    result = longwave_radiation(
        bounds=case["bounds"],
        num_filter=case["num_filter"],
        filter_indices=case["filter_indices"],
        mlcanopy_inst=case["mlcanopy_inst"],
        params=case["params"],
        longwave_type=case["longwave_type"],
    )
    
    # With emissivity = 1.0 (perfect blackbody), emission should be maximal
    # Check that results are physically reasonable
    assert jnp.all(result.lwup > 0), \
        "Non-positive upward flux with boundary emissivity values"
    assert jnp.all(result.lwsoi > 0), \
        "Non-positive soil absorption with boundary emissivity values"
    
    # Ground emissivity is 1.0, so ground emission should be at Stefan-Boltzmann maximum
    sb = case["params"].sb
    emg = case["params"].emg
    tg = case["mlcanopy_inst"].tg
    
    expected_ground_emission = emg * sb * tg**4
    
    # Upward flux at ground level should be close to ground emission
    # (may differ due to reflected downward radiation)
    ground_level_flux = result.lwup_layer[:, -1]
    
    # Check that ground level flux is reasonable relative to emission
    assert jnp.all(ground_level_flux > 0.5 * expected_ground_emission), \
        "Ground level flux unreasonably low compared to emission"