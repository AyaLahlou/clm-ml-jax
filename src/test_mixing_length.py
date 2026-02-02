"""
Standalone tests for mixing_length module.

This test file validates the JAX implementation of CLUBB mixing length calculations
without requiring full CLUBB module dependencies.
"""

import jax.numpy as jnp
import jax
import numpy as np
from typing import NamedTuple

# Enable 64-bit precision for accuracy
jax.config.update("jax_enable_x64", True)

# Import the module to test
from src.clm_src_main.mixing_length import (
    zm2zt_api,
    zt2zm_api,
    zm2zt2zm,
    zt2zm2zt,
    smooth_max,
    smooth_min,
    compute_mixing_length,
    calc_Lscale_directly,
    diagnose_Lscale_from_tau,
    CalcLscaleDirectlyInputs,
    DiagnoseLscaleFromTauInputs,
    ZLMIN,
    LSCALE_SFCLYR_DEPTH,
)


# ============================================================================
# Mock Classes for Testing
# ============================================================================

class MockGrid(NamedTuple):
    """Mock Grid object for testing."""
    dzm: jnp.ndarray
    zt: jnp.ndarray
    zm: jnp.ndarray
    
    
class MockErrInfo(NamedTuple):
    """Mock error info structure."""
    err_code: int = 0


class MockPDFParams(NamedTuple):
    """Mock PDF parameters."""
    rt_1: jnp.ndarray = None
    thl_1: jnp.ndarray = None


class MockStatsMetadata(NamedTuple):
    """Mock stats metadata."""
    l_stats_samp: bool = False


class MockStats(NamedTuple):
    """Mock stats structure."""
    pass


# ============================================================================
# Test Functions
# ============================================================================

def test_interpolation_functions():
    """Test zm2zt and zt2zm interpolation functions."""
    print("\n" + "="*70)
    print("TEST 1: Interpolation Functions")
    print("="*70)
    
    # Setup test data
    ngrdcol = 2
    nzm = 5
    nzt = 6
    
    # Create mock grid
    dzm = jnp.full((ngrdcol, nzm), 100.0)
    zt = jnp.arange(nzt) * 100.0
    zm = jnp.arange(nzm) * 100.0 + 50.0
    gr = MockGrid(dzm=dzm, zt=zt, zm=zm)
    
    # Test data: linear profile
    field_zm = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                          [2.0, 4.0, 6.0, 8.0, 10.0]])
    
    print(f"Input field (zm): shape={field_zm.shape}")
    print(field_zm)
    
    # Test zm2zt
    field_zt = zm2zt_api(nzm, nzt, ngrdcol, gr, field_zm)
    print(f"\nAfter zm2zt: shape={field_zt.shape}")
    print(field_zt)
    
    # Check boundaries
    assert jnp.allclose(field_zt[:, 0], field_zm[:, 0]), "Bottom boundary failed"
    assert jnp.allclose(field_zt[:, -1], field_zm[:, -1]), "Top boundary failed"
    
    # Test zt2zm
    field_zm_back = zt2zm_api(nzm, nzt, ngrdcol, gr, field_zt, 0.0)
    print(f"\nAfter zt2zm: shape={field_zm_back.shape}")
    print(field_zm_back)
    
    # Test roundtrip smoothing
    field_smoothed = zm2zt2zm(nzm, nzt, ngrdcol, gr, field_zm)
    print(f"\nAfter zm2zt2zm smoothing: shape={field_smoothed.shape}")
    print(field_smoothed)
    
    print("✓ Interpolation functions test PASSED")
    return True


def test_smooth_functions():
    """Test smooth max and min functions."""
    print("\n" + "="*70)
    print("TEST 2: Smooth Max/Min Functions")
    print("="*70)
    
    nzm = 5
    ngrdcol = 2
    
    field1 = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                        [0.5, 1.5, 2.5, 3.5, 4.5]])
    field2 = jnp.array([[2.0, 1.0, 3.0, 3.0, 4.0],
                        [1.0, 2.0, 2.0, 4.0, 4.0]])
    
    print(f"Field 1:\n{field1}")
    print(f"Field 2:\n{field2}")
    
    # Test smooth max
    smoothing_mag = 0.1
    result_max = smooth_max(nzm, ngrdcol, field1, field2, smoothing_mag)
    print(f"\nSmooth max (smoothing={smoothing_mag}):\n{result_max}")
    
    # Verify it's close to actual max
    actual_max = jnp.maximum(field1, field2)
    print(f"Actual max:\n{actual_max}")
    
    assert jnp.allclose(result_max, actual_max, atol=0.2), "Smooth max too different from actual max"
    assert jnp.all(result_max >= jnp.maximum(field1, field2) - 0.2), "Smooth max less than inputs"
    
    # Test smooth min
    upper_bound = 3.0
    smoothing_factor = 0.1
    result_min = smooth_min(nzm, ngrdcol, field1, upper_bound, smoothing_factor)
    print(f"\nSmooth min with bound={upper_bound}:\n{result_min}")
    
    assert jnp.all(result_min <= upper_bound + 0.2), "Smooth min exceeds upper bound"
    
    print("✓ Smooth functions test PASSED")
    return True


def test_compute_mixing_length():
    """Test compute_mixing_length function with synthetic data."""
    print("\n" + "="*70)
    print("TEST 3: Compute Mixing Length")
    print("="*70)
    
    # Setup synthetic atmospheric profile
    ngrdcol = 1
    nzm = 10
    nzt = 11
    
    # Create grid
    dzm = jnp.full((ngrdcol, nzm), 100.0)
    zt = jnp.arange(nzt) * 100.0
    zm = jnp.arange(nzm) * 100.0 + 50.0
    gr = MockGrid(dzm=dzm, zt=zt, zm=zm)
    
    # Create realistic atmospheric profile
    # Temperature decreases with height
    thvm = jnp.broadcast_to(300.0 - 0.006 * zt, (ngrdcol, nzt))
    thlm = jnp.broadcast_to(300.0 - 0.006 * zt, (ngrdcol, nzt))
    
    # Moisture decreases with height
    rtm = jnp.broadcast_to(0.015 - 0.001 * zt / 1000.0, (ngrdcol, nzt))
    
    # TKE profile (maximum in boundary layer)
    em_profile = 1.0 * jnp.exp(-zm / 500.0)
    em = jnp.broadcast_to(em_profile, (ngrdcol, nzm))
    
    # Pressure profile
    p_in_Pa = jnp.broadcast_to(101325.0 * jnp.exp(-zt / 8500.0), (ngrdcol, nzt))
    
    # Exner function
    exner = jnp.broadcast_to((p_in_Pa / 100000.0)**(287.0/1005.0), (ngrdcol, nzt))
    
    # Reference theta_v
    thv_ds = thvm
    
    # Maximum mixing length
    Lscale_max = jnp.full((ngrdcol,), 1000.0)
    
    # Parameters
    mu = 6e-4  # Entrainment rate
    lmin = 0.1  # Minimum length
    saturation_formula = 1
    l_implemented = True
    err_info = MockErrInfo()
    
    print(f"Grid: {nzm} momentum levels, {nzt} thermodynamic levels")
    print(f"Height range: {zt[0]:.1f} - {zt[-1]:.1f} m")
    print(f"Temperature range: {thvm[0,0]:.1f} - {thvm[0,-1]:.1f} K")
    print(f"TKE range: {em[0,0]:.3f} - {em[0,-1]:.3f} m²/s²")
    
    # Run the function
    print("\nRunning compute_mixing_length...")
    Lscale, Lscale_up, Lscale_down, err_info = compute_mixing_length(
        nzm, nzt, ngrdcol, gr, thvm, thlm, rtm, em,
        Lscale_max, p_in_Pa, exner, thv_ds, mu, lmin,
        saturation_formula, l_implemented, err_info
    )
    
    print(f"\nResults:")
    print(f"Lscale shape: {Lscale.shape}")
    print(f"Lscale_up shape: {Lscale_up.shape}")
    print(f"Lscale_down shape: {Lscale_down.shape}")
    
    print(f"\nLscale values (m):")
    print(f"  Min: {jnp.min(Lscale):.2f}")
    print(f"  Max: {jnp.max(Lscale):.2f}")
    print(f"  Mean: {jnp.mean(Lscale):.2f}")
    
    print(f"\nLscale_up values (m):")
    print(f"  Min: {jnp.min(Lscale_up):.2f}")
    print(f"  Max: {jnp.max(Lscale_up):.2f}")
    print(f"  Mean: {jnp.mean(Lscale_up):.2f}")
    
    print(f"\nLscale_down values (m):")
    print(f"  Min: {jnp.min(Lscale_down):.2f}")
    print(f"  Max: {jnp.max(Lscale_down):.2f}")
    print(f"  Mean: {jnp.mean(Lscale_down):.2f}")
    
    # Validation checks
    assert jnp.all(jnp.isfinite(Lscale)), "Lscale contains non-finite values"
    assert jnp.all(jnp.isfinite(Lscale_up)), "Lscale_up contains non-finite values"
    assert jnp.all(jnp.isfinite(Lscale_down)), "Lscale_down contains non-finite values"
    
    assert jnp.all(Lscale >= ZLMIN), f"Lscale below minimum ({ZLMIN})"
    assert jnp.all(Lscale_up >= ZLMIN), f"Lscale_up below minimum ({ZLMIN})"
    assert jnp.all(Lscale_down >= ZLMIN), f"Lscale_down below minimum ({ZLMIN})"
    
    assert jnp.all(Lscale <= Lscale_max[0]), f"Lscale exceeds maximum ({Lscale_max[0]})"
    
    print("✓ Compute mixing length test PASSED")
    return True


def test_calc_Lscale_directly():
    """Test calc_Lscale_directly function."""
    print("\n" + "="*70)
    print("TEST 4: Calculate Lscale Directly (PDF-based)")
    print("="*70)
    
    # Setup
    ngrdcol = 1
    nzm = 10
    nzt = 11
    
    # Create grid
    dzm = jnp.full((ngrdcol, nzm), 100.0)
    zt = jnp.arange(nzt) * 100.0
    zm = jnp.arange(nzm) * 100.0 + 50.0
    gr = MockGrid(dzm=dzm, zt=zt, zm=zm)
    
    # Atmospheric variables
    thvm = jnp.broadcast_to(300.0 - 0.006 * zt, (ngrdcol, nzt))
    thlm = jnp.broadcast_to(300.0 - 0.006 * zt, (ngrdcol, nzt))
    rtm = jnp.broadcast_to(0.015 - 0.001 * zt / 1000.0, (ngrdcol, nzt))
    
    # TKE
    em = jnp.broadcast_to(1.0 * jnp.exp(-zm / 500.0), (ngrdcol, nzm))
    
    # Pressure and exner
    p_in_Pa = jnp.broadcast_to(101325.0 * jnp.exp(-zt / 8500.0), (ngrdcol, nzt))
    exner = jnp.broadcast_to((p_in_Pa / 100000.0)**(287.0/1005.0), (ngrdcol, nzt))
    
    # PDF parameters (variances)
    rtp2_zt = jnp.broadcast_to(0.0001 * jnp.ones(nzt), (ngrdcol, nzt))
    thlp2_zt = jnp.broadcast_to(0.1 * jnp.ones(nzt), (ngrdcol, nzt))
    rtpthlp_zt = jnp.broadcast_to(0.001 * jnp.ones(nzt), (ngrdcol, nzt))
    
    # Create inputs
    inputs = CalcLscaleDirectlyInputs(
        ngrdcol=ngrdcol,
        nzm=nzm,
        nzt=nzt,
        gr=gr,
        l_implemented=True,
        p_in_Pa=p_in_Pa,
        exner=exner,
        rtm=rtm,
        thlm=thlm,
        thvm=thvm,
        newmu=6e-4,
        rtp2_zt=rtp2_zt,
        thlp2_zt=thlp2_zt,
        rtpthlp_zt=rtpthlp_zt,
        pdf_params=MockPDFParams(),
        em=em,
        thv_ds_zt=thvm,
        Lscale_max=1000.0,
        lmin=0.1,
        clubb_params=jnp.zeros(10),
        saturation_formula=1,
        l_Lscale_plume_centered=False,
        stats_metadata=MockStatsMetadata(),
        stats_zt=MockStats(),
        err_info=MockErrInfo()
    )
    
    print("Running calc_Lscale_directly...")
    outputs = calc_Lscale_directly(inputs)
    
    print(f"\nResults:")
    print(f"Lscale shape: {outputs.Lscale.shape}")
    print(f"Lscale values (m):")
    print(f"  Min: {jnp.min(outputs.Lscale):.2f}")
    print(f"  Max: {jnp.max(outputs.Lscale):.2f}")
    print(f"  Mean: {jnp.mean(outputs.Lscale):.2f}")
    
    print(f"\nLscale_up values (m):")
    print(f"  Min: {jnp.min(outputs.Lscale_up):.2f}")
    print(f"  Max: {jnp.max(outputs.Lscale_up):.2f}")
    print(f"  Mean: {jnp.mean(outputs.Lscale_up):.2f}")
    
    print(f"\nLscale_down values (m):")
    print(f"  Min: {jnp.min(outputs.Lscale_down):.2f}")
    print(f"  Max: {jnp.max(outputs.Lscale_down):.2f}")
    print(f"  Mean: {jnp.mean(outputs.Lscale_down):.2f}")
    
    # Validation
    assert jnp.all(jnp.isfinite(outputs.Lscale)), "Lscale contains non-finite values"
    assert jnp.all(outputs.Lscale >= inputs.lmin), "Lscale below minimum"
    assert jnp.all(outputs.Lscale <= inputs.Lscale_max), "Lscale exceeds maximum"
    
    print("✓ calc_Lscale_directly test PASSED")
    return True


def test_diagnose_Lscale_from_tau():
    """Test diagnose_Lscale_from_tau function."""
    print("\n" + "="*70)
    print("TEST 5: Diagnose Lscale from Tau (Timescale-based)")
    print("="*70)
    
    # Setup
    ngrdcol = 1
    nzm = 10
    nzt = 11
    
    # Create grid
    dzm = jnp.full((ngrdcol, nzm), 100.0)
    zt = jnp.arange(nzt) * 100.0
    zm = jnp.arange(nzm) * 100.0 + 50.0
    gr = MockGrid(dzm=dzm, zt=zt, zm=zm)
    
    # Surface momentum fluxes
    upwp_sfc = jnp.array([0.1])
    vpwp_sfc = jnp.array([0.05])
    
    # Wind shear
    ddzt_umvm_sqd = jnp.broadcast_to(0.0001 * jnp.ones(nzm), (ngrdcol, nzm))
    
    # Ice supersaturation fraction
    ice_supersat_frac = jnp.zeros((ngrdcol, nzt))
    
    # TKE
    em = jnp.broadcast_to(1.0 * jnp.exp(-zm / 500.0), (ngrdcol, nzm))
    sqrt_em_zt = jnp.sqrt(jnp.broadcast_to(1.0 * jnp.exp(-zt / 500.0), (ngrdcol, nzt)))
    
    # Brunt-Vaisala frequency squared
    brunt_vaisala_freq_sqd_smth = jnp.broadcast_to(0.0001 * jnp.ones(nzm), (ngrdcol, nzm))
    
    # Richardson number
    Ri_zm = jnp.broadcast_to(0.5 * jnp.ones(nzm), (ngrdcol, nzm))
    
    # Create inputs
    inputs = DiagnoseLscaleFromTauInputs(
        nzm=nzm,
        nzt=nzt,
        ngrdcol=ngrdcol,
        gr=gr,
        upwp_sfc=upwp_sfc,
        vpwp_sfc=vpwp_sfc,
        ddzt_umvm_sqd=ddzt_umvm_sqd,
        ice_supersat_frac=ice_supersat_frac,
        em=em,
        sqrt_em_zt=sqrt_em_zt,
        ufmin=0.01,
        tau_const=300.0,
        sfc_elevation=jnp.array([0.0]),
        Lscale_max=1000.0,
        clubb_params=jnp.zeros(50),
        stats_metadata=MockStatsMetadata(),
        l_e3sm_config=False,
        l_smooth_Heaviside_tau_wpxp=False,
        brunt_vaisala_freq_sqd_smth=brunt_vaisala_freq_sqd_smth,
        Ri_zm=Ri_zm,
        stats_zm=MockStats(),
        err_info=MockErrInfo()
    )
    
    print("Running diagnose_Lscale_from_tau...")
    outputs = diagnose_Lscale_from_tau(inputs)
    
    print(f"\nResults:")
    print(f"Inverse timescale (1/s):")
    print(f"  invrs_tau_zt min: {jnp.min(outputs.invrs_tau_zt):.6f}")
    print(f"  invrs_tau_zt max: {jnp.max(outputs.invrs_tau_zt):.6f}")
    
    print(f"\nTimescale tau (s):")
    print(f"  tau_zt min: {jnp.min(outputs.tau_zt):.2f}")
    print(f"  tau_zt max: {jnp.max(outputs.tau_zt):.2f}")
    print(f"  tau_zt mean: {jnp.mean(outputs.tau_zt):.2f}")
    
    print(f"\nMixing length Lscale (m):")
    print(f"  Min: {jnp.min(outputs.Lscale):.2f}")
    print(f"  Max: {jnp.max(outputs.Lscale):.2f}")
    print(f"  Mean: {jnp.mean(outputs.Lscale):.2f}")
    
    print(f"\nLscale_up (m):")
    print(f"  Min: {jnp.min(outputs.Lscale_up):.2f}")
    print(f"  Max: {jnp.max(outputs.Lscale_up):.2f}")
    
    print(f"\nLscale_down (m):")
    print(f"  Min: {jnp.min(outputs.Lscale_down):.2f}")
    print(f"  Max: {jnp.max(outputs.Lscale_down):.2f}")
    
    # Validation
    assert jnp.all(jnp.isfinite(outputs.invrs_tau_zt)), "invrs_tau_zt has non-finite values"
    assert jnp.all(jnp.isfinite(outputs.tau_zt)), "tau_zt has non-finite values"
    assert jnp.all(jnp.isfinite(outputs.Lscale)), "Lscale has non-finite values"
    
    assert jnp.all(outputs.invrs_tau_zt >= 0), "Negative inverse timescale"
    assert jnp.all(outputs.tau_zt > 0), "Non-positive timescale"
    assert jnp.all(outputs.Lscale >= ZLMIN), "Lscale below minimum"
    assert jnp.all(outputs.Lscale <= inputs.Lscale_max), "Lscale exceeds maximum"
    
    print("✓ diagnose_Lscale_from_tau test PASSED")
    return True


def test_physical_constraints():
    """Test that physical constraints are satisfied."""
    print("\n" + "="*70)
    print("TEST 6: Physical Constraints Validation")
    print("="*70)
    
    print(f"ZLMIN (minimum length scale): {ZLMIN} m")
    print(f"LSCALE_SFCLYR_DEPTH (surface layer depth): {LSCALE_SFCLYR_DEPTH} m")
    
    # Test that constants are reasonable
    assert ZLMIN > 0 and ZLMIN < 1.0, "ZLMIN should be small positive value"
    assert LSCALE_SFCLYR_DEPTH > 100 and LSCALE_SFCLYR_DEPTH < 1000, \
        "Surface layer depth should be O(100-1000m)"
    
    print("✓ Physical constraints test PASSED")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "#"*70)
    print("# MIXING LENGTH MODULE - STANDALONE TEST SUITE")
    print("#"*70)
    
    tests = [
        ("Interpolation Functions", test_interpolation_functions),
        ("Smooth Functions", test_smooth_functions),
        ("Compute Mixing Length", test_compute_mixing_length),
        ("Calc Lscale Directly", test_calc_Lscale_directly),
        ("Diagnose Lscale from Tau", test_diagnose_Lscale_from_tau),
        ("Physical Constraints", test_physical_constraints),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASSED", None))
        except Exception as e:
            results.append((test_name, "FAILED", str(e)))
            print(f"\n✗ {test_name} FAILED: {e}")
    
    # Summary
    print("\n" + "#"*70)
    print("# TEST SUMMARY")
    print("#"*70)
    
    passed = sum(1 for _, status, _ in results if status == "PASSED")
    failed = sum(1 for _, status, _ in results if status == "FAILED")
    
    for test_name, status, error in results:
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"{symbol} {test_name}: {status}")
        if error:
            print(f"  Error: {error}")
    
    print(f"\nTotal: {len(tests)} tests, {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        return True
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
