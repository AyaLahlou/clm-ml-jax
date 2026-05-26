"""
Comprehensive pytest suite for MLLeafHeatCapacityMod module.

This module tests the leaf heat capacity calculation functions which compute
the thermal properties of canopy leaves based on their physical and chemical
composition.

Tests cover:
- Nominal cases with typical forest/grassland parameters
- Edge cases (zero PAI, extreme SLA values, boundary water fractions)
- Special cases (dense canopies, non-default parameters)
- Shape and dtype validation
- Physical constraint verification
- JIT-compiled versions
"""

import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multilayer_canopy.MLLeafHeatCapacityMod import (
    LeafHeatCapacityInput,
    LeafHeatCapacityParams,
    leaf_heat_capacity,
    leaf_heat_capacity_jit,
    leaf_heat_capacity_simple,
    leaf_heat_capacity_simple_jit,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_params():
    """Default physical parameters for leaf heat capacity calculations."""
    return LeafHeatCapacityParams(
        cpbio=1470.0, cpliq=4188.0, fcarbon=0.5, fwater=0.7
    )


@pytest.fixture
def test_data_nominal():
    """Nominal test cases with typical forest/grassland parameters."""
    return [
        {
            "name": "single_patch_single_layer",
            "slatop": jnp.array([0.015]),
            "ncan": jnp.array([1], dtype=jnp.int32),
            "dpai": jnp.array([[2.5]]),
            "cpbio": 1470.0,
            "cpliq": 4188.0,
            "fcarbon": 0.5,
            "fwater": 0.7,
            "description": "Single patch, single canopy layer with typical forest parameters",
        },
        {
            "name": "multi_patch_multi_layer",
            "slatop": jnp.array([0.012, 0.018, 0.02]),
            "ncan": jnp.array([3, 2, 4], dtype=jnp.int32),
            "dpai": jnp.array(
                [[1.5, 1.2, 0.8, 0.0], [2.0, 1.5, 0.0, 0.0], [1.8, 1.6, 1.4, 1.0]]
            ),
            "cpbio": 1470.0,
            "cpliq": 4188.0,
            "fcarbon": 0.5,
            "fwater": 0.7,
            "description": "Multiple patches with varying canopy layers",
        },
        {
            "name": "grassland_low_lai",
            "slatop": jnp.array([0.025, 0.03]),
            "ncan": jnp.array([2, 2], dtype=jnp.int32),
            "dpai": jnp.array([[0.5, 0.3], [0.8, 0.4]]),
            "cpbio": 1470.0,
            "cpliq": 4188.0,
            "fcarbon": 0.5,
            "fwater": 0.7,
            "description": "Grassland/crop canopy with high SLA and low LAI",
        },
    ]


@pytest.fixture
def test_data_edge():
    """Edge case test data covering boundary conditions."""
    return [
        {
            "name": "zero_dpai",
            "slatop": jnp.array([0.015, 0.02]),
            "ncan": jnp.array([3, 2], dtype=jnp.int32),
            "dpai": jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
            "cpbio": 1470.0,
            "cpliq": 4188.0,
            "fcarbon": 0.5,
            "fwater": 0.7,
            "expected_zero_mask": jnp.array(
                [[True, True, True], [False, True, True]]
            ),
            "description": "Zero plant area index in some or all layers",
        },
        {
            "name": "very_small_slatop",
            "slatop": jnp.array([0.001, 0.0005]),
            "ncan": jnp.array([2, 2], dtype=jnp.int32),
            "dpai": jnp.array([[1.0, 0.5], [1.2, 0.8]]),
            "cpbio": 1470.0,
            "cpliq": 4188.0,
            "fcarbon": 0.5,
            "fwater": 0.7,
            "description": "Very small SLA (thick leaves like succulents)",
        },
        {
            "name": "very_large_slatop",
            "slatop": jnp.array([0.05, 0.08]),
            "ncan": jnp.array([2, 3], dtype=jnp.int32),
            "dpai": jnp.array([[2.0, 1.5, 0.0], [1.8, 1.2, 0.6]]),
            "cpbio": 1470.0,
            "cpliq": 4188.0,
            "fcarbon": 0.5,
            "fwater": 0.7,
            "description": "Very large SLA (thin leaves)",
        },
        {
            "name": "extreme_fwater_high",
            "slatop": jnp.array([0.015]),
            "ncan": jnp.array([2], dtype=jnp.int32),
            "dpai": jnp.array([[1.5, 1.0]]),
            "cpbio": 1470.0,
            "cpliq": 4188.0,
            "fcarbon": 0.5,
            "fwater": 0.95,
            "description": "Very high water fraction near upper boundary",
        },
        {
            "name": "low_fwater",
            "slatop": jnp.array([0.015]),
            "ncan": jnp.array([2], dtype=jnp.int32),
            "dpai": jnp.array([[1.5, 1.0]]),
            "cpbio": 1470.0,
            "cpliq": 4188.0,
            "fcarbon": 0.5,
            "fwater": 0.1,
            "description": "Very low water fraction (desiccated leaves)",
        },
    ]


@pytest.fixture
def test_data_special():
    """Special test cases for unusual but valid scenarios."""
    return [
        {
            "name": "high_lai_dense_canopy",
            "slatop": jnp.array([0.01]),
            "ncan": jnp.array([5], dtype=jnp.int32),
            "dpai": jnp.array([[3.5, 3.0, 2.5, 2.0, 1.5]]),
            "cpbio": 1470.0,
            "cpliq": 4188.0,
            "fcarbon": 0.5,
            "fwater": 0.7,
            "description": "Dense tropical forest canopy with many layers",
        },
        {
            "name": "varying_physical_parameters",
            "slatop": jnp.array([0.015, 0.02]),
            "ncan": jnp.array([2, 2], dtype=jnp.int32),
            "dpai": jnp.array([[2.0, 1.5], [1.8, 1.2]]),
            "cpbio": 1200.0,
            "cpliq": 4188.0,
            "fcarbon": 0.45,
            "fwater": 0.65,
            "description": "Non-default physical parameters",
        },
        {
            "name": "single_value_broadcast",
            "slatop": jnp.array([0.015]),
            "ncan": jnp.array([1], dtype=jnp.int32),
            "dpai": jnp.array([[1.5]]),
            "cpbio": 1470.0,
            "cpliq": 4188.0,
            "fcarbon": 0.5,
            "fwater": 0.7,
            "description": "Minimal dimensions (1x1) to test broadcasting",
        },
    ]


# ============================================================================
# Helper Functions
# ============================================================================


def calculate_expected_heat_capacity(slatop, dpai, cpbio, cpliq, fcarbon, fwater):
    """
    Calculate expected leaf heat capacity using the documented equations.
    
    Args:
        slatop: Specific leaf area at top of canopy [m2/gC]
        dpai: Plant area index [m2/m2]
        cpbio: Heat capacity of dry biomass [J/kg/K]
        cpliq: Heat capacity of liquid water [J/kg/K]
        fcarbon: Carbon fraction of dry biomass [-]
        fwater: Water fraction of fresh biomass [-]
    
    Returns:
        Leaf heat capacity [J/m2 leaf/K]
    """
    # Convert to numpy for calculation
    slatop = np.asarray(slatop)
    dpai = np.asarray(dpai)
    
    # Leaf mass per area [kg C / m2]
    lma = 1.0 / slatop * 0.001
    
    # Dry weight [kg DM / m2]
    dry_weight = lma / fcarbon
    
    # Fresh weight [kg FM / m2]
    fresh_weight = dry_weight / (1.0 - fwater)
    
    # Leaf water content [kg H2O / m2]
    leaf_water = fwater * fresh_weight
    
    # Heat capacity per m2 leaf [J/K/m2 leaf]
    cpleaf = cpbio * dry_weight + cpliq * leaf_water
    
    return cpleaf


# ============================================================================
# Tests for leaf_heat_capacity (main function)
# ============================================================================


@pytest.mark.parametrize("use_jit", [False, True], ids=["regular", "jit"])
class TestLeafHeatCapacity:
    """Test suite for leaf_heat_capacity function."""

    def test_nominal_cases_shapes(self, test_data_nominal, use_jit):
        """Test that nominal cases produce correct output shapes."""
        func = leaf_heat_capacity_jit if use_jit else leaf_heat_capacity
        
        for case in test_data_nominal:
            inputs = LeafHeatCapacityInput(
                slatop=case["slatop"],
                ncan=case["ncan"],
                dpai=case["dpai"],
                cpbio=case["cpbio"],
                cpliq=case["cpliq"],
                fcarbon=case["fcarbon"],
                fwater=case["fwater"],
            )
            
            result = func(inputs)
            
            # Check shape matches dpai shape
            assert result.shape == case["dpai"].shape, (
                f"Shape mismatch for {case['name']}: "
                f"expected {case['dpai'].shape}, got {result.shape}"
            )

    def test_nominal_cases_values(self, test_data_nominal, use_jit):
        """Test that nominal cases produce physically reasonable values."""
        func = leaf_heat_capacity_jit if use_jit else leaf_heat_capacity
        
        for case in test_data_nominal:
            inputs = LeafHeatCapacityInput(
                slatop=case["slatop"],
                ncan=case["ncan"],
                dpai=case["dpai"],
                cpbio=case["cpbio"],
                cpliq=case["cpliq"],
                fcarbon=case["fcarbon"],
                fwater=case["fwater"],
            )
            
            result = func(inputs)
            
            # All values should be non-negative
            assert jnp.all(result >= 0), (
                f"Negative heat capacity found in {case['name']}"
            )
            
            # Where dpai > 0, heat capacity should be > 0
            positive_dpai_mask = case["dpai"] > 0
            if jnp.any(positive_dpai_mask):
                assert jnp.all(result[positive_dpai_mask] > 0), (
                    f"Zero heat capacity where dpai > 0 in {case['name']}"
                )
            
            # Where dpai == 0, heat capacity should be 0
            zero_dpai_mask = case["dpai"] == 0
            if jnp.any(zero_dpai_mask):
                assert jnp.allclose(result[zero_dpai_mask], 0.0, atol=1e-10), (
                    f"Non-zero heat capacity where dpai == 0 in {case['name']}"
                )

    def test_nominal_cases_manual_calculation(self, use_jit):
        """Test against manual calculation for a simple case."""
        func = leaf_heat_capacity_jit if use_jit else leaf_heat_capacity
        
        # Simple case: single patch, single layer
        slatop = jnp.array([0.015])
        dpai = jnp.array([[2.5]])
        cpbio = 1470.0
        cpliq = 4188.0
        fcarbon = 0.5
        fwater = 0.7
        
        inputs = LeafHeatCapacityInput(
            slatop=slatop,
            ncan=jnp.array([1], dtype=jnp.int32),
            dpai=dpai,
            cpbio=cpbio,
            cpliq=cpliq,
            fcarbon=fcarbon,
            fwater=fwater,
        )
        
        result = func(inputs)
        
        # Manual calculation
        expected = calculate_expected_heat_capacity(
            slatop[0], 1.0, cpbio, cpliq, fcarbon, fwater
        )
        
        assert jnp.allclose(result[0, 0], expected, rtol=1e-5, atol=1e-6), (
            f"Manual calculation mismatch: expected {expected}, got {result[0, 0]}"
        )

    def test_edge_cases_zero_dpai(self, test_data_edge, use_jit):
        """Test that zero dpai produces zero heat capacity."""
        func = leaf_heat_capacity_jit if use_jit else leaf_heat_capacity
        
        case = test_data_edge[0]  # zero_dpai case
        inputs = LeafHeatCapacityInput(
            slatop=case["slatop"],
            ncan=case["ncan"],
            dpai=case["dpai"],
            cpbio=case["cpbio"],
            cpliq=case["cpliq"],
            fcarbon=case["fcarbon"],
            fwater=case["fwater"],
        )
        
        result = func(inputs)
        
        # Where dpai is zero, result should be zero
        zero_mask = case["dpai"] == 0
        assert jnp.allclose(result[zero_mask], 0.0, atol=1e-10), (
            "Non-zero heat capacity where dpai is zero"
        )

    def test_edge_cases_extreme_sla(self, test_data_edge, use_jit):
        """Test extreme SLA values (very small and very large)."""
        func = leaf_heat_capacity_jit if use_jit else leaf_heat_capacity
        
        for case in test_data_edge[1:3]:  # small and large slatop cases
            inputs = LeafHeatCapacityInput(
                slatop=case["slatop"],
                ncan=case["ncan"],
                dpai=case["dpai"],
                cpbio=case["cpbio"],
                cpliq=case["cpliq"],
                fcarbon=case["fcarbon"],
                fwater=case["fwater"],
            )
            
            result = func(inputs)
            
            # Should still produce valid results
            assert jnp.all(jnp.isfinite(result)), (
                f"Non-finite values in {case['name']}"
            )
            assert jnp.all(result >= 0), (
                f"Negative values in {case['name']}"
            )
            
            # Small SLA (thick leaves) should give higher heat capacity
            # Large SLA (thin leaves) should give lower heat capacity
            if "small" in case["name"]:
                # For small SLA, heat capacity should be relatively large
                positive_dpai = case["dpai"] > 0
                if jnp.any(positive_dpai):
                    assert jnp.all(result[positive_dpai] > 1000), (
                        f"Unexpectedly low heat capacity for thick leaves in {case['name']}"
                    )

    def test_edge_cases_extreme_fwater(self, test_data_edge, use_jit):
        """Test extreme water fraction values."""
        func = leaf_heat_capacity_jit if use_jit else leaf_heat_capacity
        
        for case in test_data_edge[3:5]:  # high and low fwater cases
            inputs = LeafHeatCapacityInput(
                slatop=case["slatop"],
                ncan=case["ncan"],
                dpai=case["dpai"],
                cpbio=case["cpbio"],
                cpliq=case["cpliq"],
                fcarbon=case["fcarbon"],
                fwater=case["fwater"],
            )
            
            result = func(inputs)
            
            # Should produce finite, non-negative results
            assert jnp.all(jnp.isfinite(result)), (
                f"Non-finite values in {case['name']}"
            )
            assert jnp.all(result >= 0), (
                f"Negative values in {case['name']}"
            )
            
            # High fwater should give higher heat capacity (more water)
            # Low fwater should give lower heat capacity (less water)
            positive_dpai = case["dpai"] > 0
            if jnp.any(positive_dpai):
                if case["fwater"] > 0.9:
                    # Very high water content
                    assert jnp.all(result[positive_dpai] > 2000), (
                        f"Unexpectedly low heat capacity for high water content in {case['name']}"
                    )

    def test_special_cases(self, test_data_special, use_jit):
        """Test special cases like dense canopies and non-default parameters."""
        func = leaf_heat_capacity_jit if use_jit else leaf_heat_capacity
        
        for case in test_data_special:
            inputs = LeafHeatCapacityInput(
                slatop=case["slatop"],
                ncan=case["ncan"],
                dpai=case["dpai"],
                cpbio=case["cpbio"],
                cpliq=case["cpliq"],
                fcarbon=case["fcarbon"],
                fwater=case["fwater"],
            )
            
            result = func(inputs)
            
            # Basic validity checks
            assert result.shape == case["dpai"].shape
            assert jnp.all(jnp.isfinite(result))
            assert jnp.all(result >= 0)
            
            # For dense canopy case, check all layers have positive values
            if "dense" in case["name"]:
                assert jnp.all(result > 0), (
                    "Dense canopy should have positive heat capacity in all layers"
                )

    def test_dtype_consistency(self, test_data_nominal, use_jit):
        """Test that output dtype is consistent with input dtype."""
        func = leaf_heat_capacity_jit if use_jit else leaf_heat_capacity
        
        case = test_data_nominal[0]
        inputs = LeafHeatCapacityInput(
            slatop=case["slatop"],
            ncan=case["ncan"],
            dpai=case["dpai"],
            cpbio=case["cpbio"],
            cpliq=case["cpliq"],
            fcarbon=case["fcarbon"],
            fwater=case["fwater"],
        )
        
        result = func(inputs)
        
        # Result should be float32 or float64
        assert result.dtype in [jnp.float32, jnp.float64], (
            f"Unexpected dtype: {result.dtype}"
        )


# ============================================================================
# Tests for leaf_heat_capacity_simple
# ============================================================================


@pytest.mark.parametrize("use_jit", [False, True], ids=["regular", "jit"])
class TestLeafHeatCapacitySimple:
    """Test suite for leaf_heat_capacity_simple function."""

    def test_simple_nominal_default_params(self, default_params, use_jit):
        """Test simple function with default parameters."""
        func = leaf_heat_capacity_simple_jit if use_jit else leaf_heat_capacity_simple
        
        slatop = jnp.array([0.015, 0.02])
        dpai = jnp.array([[2.0, 1.5], [1.8, 1.2]])
        
        result = func(slatop, dpai, default_params)
        
        # Check shape
        assert result.shape == dpai.shape
        
        # Check values are positive where dpai > 0
        assert jnp.all(result[dpai > 0] > 0)
        
        # Check values are finite
        assert jnp.all(jnp.isfinite(result))

    def test_simple_zero_dpai(self, default_params, use_jit):
        """Test that zero dpai produces zero output."""
        func = leaf_heat_capacity_simple_jit if use_jit else leaf_heat_capacity_simple
        
        slatop = jnp.array([0.015])
        dpai = jnp.array([[0.0, 0.0, 0.0]])
        
        result = func(slatop, dpai, default_params)
        
        # All outputs should be zero
        assert jnp.allclose(result, 0.0, atol=1e-10)

    def test_simple_single_value(self, default_params, use_jit):
        """Test minimal dimensions (1x1)."""
        func = leaf_heat_capacity_simple_jit if use_jit else leaf_heat_capacity_simple
        
        slatop = jnp.array([0.015])
        dpai = jnp.array([[1.5]])
        
        result = func(slatop, dpai, default_params)
        
        # Check shape
        assert result.shape == (1, 1)
        
        # Check value is positive
        assert result[0, 0] > 0
        
        # Check against manual calculation
        expected = calculate_expected_heat_capacity(
            0.015, 1.0, 1470.0, 4188.0, 0.5, 0.7
        )
        assert jnp.allclose(result[0, 0], expected, rtol=1e-5, atol=1e-6)

    def test_simple_custom_params(self, use_jit):
        """Test with non-default parameters."""
        func = leaf_heat_capacity_simple_jit if use_jit else leaf_heat_capacity_simple
        
        slatop = jnp.array([0.015])
        dpai = jnp.array([[2.0, 1.5]])
        custom_params = LeafHeatCapacityParams(
            cpbio=1200.0, cpliq=4188.0, fcarbon=0.45, fwater=0.65
        )
        
        result = func(slatop, dpai, custom_params)
        
        # Should produce valid results
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result > 0)

    def test_simple_matches_full_function(self, default_params, use_jit):
        """Test that simple function matches full function with same inputs."""
        func_simple = (
            leaf_heat_capacity_simple_jit if use_jit else leaf_heat_capacity_simple
        )
        func_full = leaf_heat_capacity_jit if use_jit else leaf_heat_capacity
        
        slatop = jnp.array([0.015, 0.02])
        dpai = jnp.array([[2.0, 1.5], [1.8, 1.2]])
        ncan = jnp.array([2, 2], dtype=jnp.int32)
        
        # Simple function
        result_simple = func_simple(slatop, dpai, default_params)
        
        # Full function
        inputs_full = LeafHeatCapacityInput(
            slatop=slatop,
            ncan=ncan,
            dpai=dpai,
            cpbio=default_params.cpbio,
            cpliq=default_params.cpliq,
            fcarbon=default_params.fcarbon,
            fwater=default_params.fwater,
        )
        result_full = func_full(inputs_full)
        
        # Results should match
        assert jnp.allclose(result_simple, result_full, rtol=1e-6, atol=1e-8)


# ============================================================================
# Cross-validation Tests
# ============================================================================


class TestCrossValidation:
    """Cross-validation tests between different function versions."""

    def test_jit_vs_regular_leaf_heat_capacity(self, test_data_nominal):
        """Test that JIT and regular versions produce identical results."""
        for case in test_data_nominal:
            inputs = LeafHeatCapacityInput(
                slatop=case["slatop"],
                ncan=case["ncan"],
                dpai=case["dpai"],
                cpbio=case["cpbio"],
                cpliq=case["cpliq"],
                fcarbon=case["fcarbon"],
                fwater=case["fwater"],
            )
            
            result_regular = leaf_heat_capacity(inputs)
            result_jit = leaf_heat_capacity_jit(inputs)
            
            assert jnp.allclose(result_regular, result_jit, rtol=1e-6, atol=1e-8), (
                f"JIT and regular versions differ for {case['name']}"
            )

    def test_jit_vs_regular_simple(self, default_params):
        """Test that JIT and regular simple versions produce identical results."""
        slatop = jnp.array([0.015, 0.02, 0.018])
        dpai = jnp.array([[2.0, 1.5, 0.8], [1.8, 1.2, 0.0], [2.5, 2.0, 1.5]])
        
        result_regular = leaf_heat_capacity_simple(slatop, dpai, default_params)
        result_jit = leaf_heat_capacity_simple_jit(slatop, dpai, default_params)
        
        assert jnp.allclose(result_regular, result_jit, rtol=1e-6, atol=1e-8)


# ============================================================================
# Physical Constraint Tests
# ============================================================================


class TestPhysicalConstraints:
    """Test that physical constraints are properly enforced."""

    def test_monotonicity_with_dpai(self, default_params):
        """Test that heat capacity increases with dpai (for fixed per-area capacity)."""
        slatop = jnp.array([0.015])
        
        # Note: The output is per m2 leaf, not per m2 ground
        # So it should be constant regardless of dpai
        dpai1 = jnp.array([[1.0]])
        dpai2 = jnp.array([[2.0]])
        
        result1 = leaf_heat_capacity_simple(slatop, dpai1, default_params)
        result2 = leaf_heat_capacity_simple(slatop, dpai2, default_params)
        
        # Per m2 leaf should be the same
        assert jnp.allclose(result1, result2, rtol=1e-6)

    def test_monotonicity_with_sla(self, default_params):
        """Test that heat capacity decreases with increasing SLA (thinner leaves)."""
        dpai = jnp.array([[2.0]])
        
        slatop_low = jnp.array([0.01])  # Thick leaves
        slatop_high = jnp.array([0.03])  # Thin leaves
        
        result_low = leaf_heat_capacity_simple(slatop_low, dpai, default_params)
        result_high = leaf_heat_capacity_simple(slatop_high, dpai, default_params)
        
        # Thick leaves should have higher heat capacity
        assert result_low[0, 0] > result_high[0, 0]

    def test_water_content_effect(self):
        """Test that higher water content increases heat capacity."""
        slatop = jnp.array([0.015])
        dpai = jnp.array([[2.0]])
        
        params_low_water = LeafHeatCapacityParams(
            cpbio=1470.0, cpliq=4188.0, fcarbon=0.5, fwater=0.5
        )
        params_high_water = LeafHeatCapacityParams(
            cpbio=1470.0, cpliq=4188.0, fcarbon=0.5, fwater=0.8
        )
        
        result_low = leaf_heat_capacity_simple(slatop, dpai, params_low_water)
        result_high = leaf_heat_capacity_simple(slatop, dpai, params_high_water)
        
        # Higher water content should give higher heat capacity
        assert result_high[0, 0] > result_low[0, 0]

    def test_non_negative_output(self, test_data_nominal, test_data_edge):
        """Test that output is always non-negative."""
        all_cases = test_data_nominal + test_data_edge
        
        for case in all_cases:
            inputs = LeafHeatCapacityInput(
                slatop=case["slatop"],
                ncan=case["ncan"],
                dpai=case["dpai"],
                cpbio=case["cpbio"],
                cpliq=case["cpliq"],
                fcarbon=case["fcarbon"],
                fwater=case["fwater"],
            )
            
            result = leaf_heat_capacity(inputs)
            
            assert jnp.all(result >= 0), (
                f"Negative heat capacity found in {case['name']}"
            )


# ============================================================================
# Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_near_zero_denominator(self):
        """Test behavior when (1 - fwater) is very small."""
        slatop = jnp.array([0.015])
        dpai = jnp.array([[1.5]])
        
        # fwater = 0.999 means (1 - fwater) = 0.001
        params = LeafHeatCapacityParams(
            cpbio=1470.0, cpliq=4188.0, fcarbon=0.5, fwater=0.999
        )
        
        result = leaf_heat_capacity_simple(slatop, dpai, params)
        
        # Should still produce finite result
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result > 0)

    def test_very_small_values(self, default_params):
        """Test with very small but positive values."""
        slatop = jnp.array([1e-6])
        dpai = jnp.array([[1e-6]])
        
        result = leaf_heat_capacity_simple(slatop, dpai, default_params)
        
        # Should produce finite, non-negative result
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result >= 0)

    def test_large_values(self, default_params):
        """Test with large values."""
        slatop = jnp.array([0.001])  # Very thick leaves
        dpai = jnp.array([[10.0]])  # Very high LAI
        
        result = leaf_heat_capacity_simple(slatop, dpai, default_params)
        
        # Should produce finite result
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result > 0)

    def test_mixed_zero_nonzero(self, default_params):
        """Test with mix of zero and non-zero dpai values."""
        slatop = jnp.array([0.015, 0.02])
        dpai = jnp.array([[0.0, 1.5, 0.0, 2.0], [1.2, 0.0, 0.8, 0.0]])
        
        result = leaf_heat_capacity_simple(slatop, dpai, default_params)
        
        # Zero dpai should give zero result
        zero_mask = dpai == 0
        assert jnp.allclose(result[zero_mask], 0.0, atol=1e-10)
        
        # Non-zero dpai should give positive result
        nonzero_mask = dpai > 0
        assert jnp.all(result[nonzero_mask] > 0)


# ============================================================================
# Documentation Tests
# ============================================================================


class TestDocumentation:
    """Test that functions have proper documentation."""

    def test_function_docstrings(self):
        """Test that all functions have docstrings."""
        assert leaf_heat_capacity.__doc__ is not None
        assert leaf_heat_capacity_simple.__doc__ is not None

    def test_namedtuple_fields(self):
        """Test that NamedTuples have expected fields."""
        # Test LeafHeatCapacityInput
        input_fields = LeafHeatCapacityInput._fields
        assert "slatop" in input_fields
        assert "ncan" in input_fields
        assert "dpai" in input_fields
        assert "cpbio" in input_fields
        assert "cpliq" in input_fields
        assert "fcarbon" in input_fields
        assert "fwater" in input_fields
        
        # Test LeafHeatCapacityParams
        param_fields = LeafHeatCapacityParams._fields
        assert "cpbio" in param_fields
        assert "cpliq" in param_fields
        assert "fcarbon" in param_fields
        assert "fwater" in param_fields


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])