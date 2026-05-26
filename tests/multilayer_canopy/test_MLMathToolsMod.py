"""
Comprehensive pytest suite for MLMathToolsMod functions.

This module tests mathematical utilities for multilayer canopy modeling including:
- Root finding algorithms (hybrid method, Brent's method)
- Quadratic equation solver
- Tridiagonal matrix solvers (single and coupled systems)
- Special functions (log-gamma, beta function, beta distribution)

Test coverage includes:
- Nominal cases with known analytical solutions
- Edge cases (boundaries, extreme values, numerical stability)
- Array shape and dtype verification
- Physical realism constraints
"""

import sys
from pathlib import Path
from typing import Callable, Tuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multilayer_canopy.MLMathToolsMod import (
    beta_distribution_cdf,
    beta_distribution_pdf,
    beta_function,
    beta_function_incomplete_cf,
    hybrid_root_finder,
    log_gamma_function,
    quadratic,
    tridiag,
    tridiag_2eq,
    zbrent,
)


@pytest.fixture
def test_data():
    """
    Load test data for MLMathToolsMod functions.
    
    Returns:
        dict: Test cases with inputs and expected outputs for all functions.
    """
    return {
        "hybrid_root_finder": {
            "simple_quadratic": {
                "xa": jnp.array([1.0, 1.5, 1.8]),
                "xb": jnp.array([3.0, 2.5, 2.2]),
                "tol": 1e-6,
                "itmax": 40,
                "expected_root": jnp.array([2.0, 2.0, 2.0]),
                "tolerance": 1e-5,
            },
            "transcendental": {
                "xa": jnp.array([0.5, 1.0, 1.2]),
                "xb": jnp.array([2.0, 2.0, 2.0]),
                "tol": 1e-7,
                "itmax": 50,
                "expected_root": jnp.array([1.512134, 1.512134, 1.512134]),
                "tolerance": 1e-6,
            },
        },
        "zbrent": {
            "polynomial": {
                "xa": jnp.array([-3.0, 0.0, -5.0]),
                "xb": jnp.array([0.0, 3.0, 5.0]),
                "tol": 1e-8,
                "itmax": 50,
                "eps": 1e-8,
                "expected_root": jnp.array([-2.3, 1.5, -2.3]),
                "expected_converged": jnp.array([True, True, True]),
                "expected_error": jnp.array([False, False, False]),
                "tolerance": 1e-7,
            },
            "near_zero": {
                "xa": jnp.array([-1.0, 0.0]),
                "xb": jnp.array([0.0, 1.0]),
                "tol": 1e-10,
                "itmax": 100,
                "eps": 1e-12,
                "expected_root": jnp.array([0.1, 0.1]),
                "expected_converged": jnp.array([True, True]),
                "expected_error": jnp.array([False, False]),
                "tolerance": 1e-9,
            },
            "no_root": {
                "xa": jnp.array([-2.0, 0.0]),
                "xb": jnp.array([2.0, 3.0]),
                "tol": 1e-6,
                "itmax": 50,
                "eps": 1e-8,
                "expected_converged": jnp.array([False, False]),
                "expected_error": jnp.array([True, True]),
            },
        },
        "quadratic": {
            "standard": {
                "a": jnp.array([1.0, 2.0, -1.0, 1.0]),
                "b": jnp.array([0.0, -8.0, 0.0, -3.0]),
                "c": jnp.array([-4.0, 15.0, -9.0, 2.0]),
                "expected_r1": jnp.array([2.0, 2.5, 3.0, 2.0]),
                "expected_r2": jnp.array([-2.0, 1.5, -3.0, 1.0]),
                "tolerance": 1e-10,
            },
            "zero_discriminant": {
                "a": jnp.array([1.0, 4.0, 9.0]),
                "b": jnp.array([-4.0, -12.0, 6.0]),
                "c": jnp.array([4.0, 9.0, 1.0]),
                "expected_r1": jnp.array([2.0, 1.5, -0.333333]),
                "expected_r2": jnp.array([2.0, 1.5, -0.333333]),
                "tolerance": 1e-6,
            },
        },
        "tridiag": {
            "simple": {
                "a": jnp.array([0.0, 1.0, 1.0, 1.0]),
                "b": jnp.array([2.0, 2.0, 2.0, 2.0]),
                "c": jnp.array([1.0, 1.0, 1.0, 0.0]),
                "r": jnp.array([5.0, 6.0, 6.0, 5.0]),
                "expected_solution": jnp.array([2.0, 1.0, 1.0, 2.0]),
                "tolerance": 1e-10,
            },
            "diagonal_dominant": {
                "a": jnp.array([0.0, 0.1, 0.1]),
                "b": jnp.array([10.0, 10.0, 10.0]),
                "c": jnp.array([0.1, 0.1, 0.0]),
                "r": jnp.array([10.1, 10.2, 10.1]),
                "expected_solution": jnp.array([1.0, 1.0, 1.0]),
                "tolerance": 1e-9,
            },
        },
        "tridiag_2eq": {
            "coupled": {
                "a1": jnp.array([0.0, -0.5, -0.5, -0.5, -0.5]),
                "b11": jnp.array([2.0, 2.5, 2.5, 2.5, 2.0]),
                "b12": jnp.array([0.1, 0.1, 0.1, 0.1, 0.1]),
                "c1": jnp.array([-0.5, -0.5, -0.5, -0.5, 0.0]),
                "d1": jnp.array([300.0, 295.0, 290.0, 285.0, 280.0]),
                "a2": jnp.array([0.0, -0.3, -0.3, -0.3, -0.3]),
                "b21": jnp.array([0.05, 0.05, 0.05, 0.05, 0.05]),
                "b22": jnp.array([1.8, 2.0, 2.0, 2.0, 1.8]),
                "c2": jnp.array([-0.3, -0.3, -0.3, -0.3, 0.0]),
                "d2": jnp.array([0.015, 0.014, 0.013, 0.012, 0.011]),
                "n": 5,
                "expected_t": jnp.array([150.0, 147.5, 145.0, 142.5, 140.0]),
                "expected_q": jnp.array([0.0075, 0.007, 0.0065, 0.006, 0.0055]),
                "tolerance": 1e-6,
            },
        },
        "log_gamma": {
            "standard": {
                "x": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5]),
                "expected": jnp.array(
                    [0.0, 0.0, 0.693147, 1.791759, 3.178054, 0.572365, -0.120782]
                ),
                "tolerance": 1e-5,
            },
            "large_values": {
                "x": jnp.array([10.0, 50.0, 100.0, 0.01, 0.001]),
                "expected": jnp.array(
                    [12.801827, 144.565744, 359.134205, 4.599479, 6.907755]
                ),
                "tolerance": 1e-4,
            },
        },
        "beta_function": {
            "symmetric": {
                "a": jnp.array([1.0, 2.0, 3.0, 0.5, 5.0]),
                "b": jnp.array([1.0, 2.0, 3.0, 0.5, 5.0]),
                "expected": jnp.array([1.0, 0.166667, 0.033333, 3.141593, 0.001587]),
                "tolerance": 1e-5,
            },
            "asymmetric": {
                "a": jnp.array([0.1, 10.0, 0.5, 1.0]),
                "b": jnp.array([10.0, 0.1, 2.0, 100.0]),
                "expected": jnp.array([19.714639, 19.714639, 1.570796, 0.01]),
                "tolerance": 1e-4,
            },
        },
        "beta_pdf": {
            "uniform": {
                "a": 1.0,
                "b": 1.0,
                "x": jnp.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                "expected": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
                "tolerance": 1e-10,
            },
            "boundaries": {
                "a": jnp.array([2.0, 0.5, 3.0]),
                "b": jnp.array([5.0, 0.5, 2.0]),
                "x": jnp.array([0.0, 0.5, 1.0]),
                "expected": jnp.array([0.0, 0.63662, 0.0]),
                "tolerance": 1e-5,
            },
        },
        "beta_cdf": {
            "standard": {"a": 2.0, "b": 5.0, "x": 0.3, "expected": 0.47178, "tolerance": 1e-4},
            "symmetric": {"a": 0.5, "b": 0.5, "x": 0.5, "expected": 0.5, "tolerance": 1e-8},
        },
        "beta_incomplete_cf": {
            "convergence": {
                "a": jnp.array([2.0, 3.0, 5.0]),
                "b": jnp.array([3.0, 4.0, 2.0]),
                "x": jnp.array([0.3, 0.5, 0.7]),
                "expected": jnp.array([0.216, 0.5, 0.969]),
                "tolerance": 1e-3,
            },
            "near_boundaries": {
                "a": jnp.array([1.0, 10.0, 0.5]),
                "b": jnp.array([1.0, 10.0, 0.5]),
                "x": jnp.array([0.01, 0.99, 0.5]),
                "expected": jnp.array([0.01, 0.99, 0.5]),
                "tolerance": 1e-4,
            },
        },
    }


# ============================================================================
# Root Finding Tests
# ============================================================================


class TestHybridRootFinder:
    """Test suite for hybrid_root_finder function."""

    def test_simple_quadratic_shapes(self, test_data):
        """Verify output shape matches input shape for simple quadratic."""
        data = test_data["hybrid_root_finder"]["simple_quadratic"]
        func = lambda x: x**2 - 4
        
        result = hybrid_root_finder(func, data["xa"], data["xb"], data["tol"], data["itmax"])
        
        assert result.shape == data["xa"].shape, "Output shape should match input shape"

    def test_simple_quadratic_values(self, test_data):
        """Test convergence to known root x=2 for quadratic x²-4."""
        data = test_data["hybrid_root_finder"]["simple_quadratic"]
        func = lambda x: x**2 - 4
        
        result = hybrid_root_finder(func, data["xa"], data["xb"], data["tol"], data["itmax"])
        
        np.testing.assert_allclose(
            result,
            data["expected_root"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="Hybrid root finder should converge to x=2 for x²-4",
        )

    def test_transcendental_values(self, test_data):
        """Test convergence for transcendental equation exp(x) - 3x."""
        data = test_data["hybrid_root_finder"]["transcendental"]
        func = lambda x: jnp.exp(x) - 3 * x
        
        result = hybrid_root_finder(func, data["xa"], data["xb"], data["tol"], data["itmax"])
        
        np.testing.assert_allclose(
            result,
            data["expected_root"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="Should converge to root of exp(x) - 3x",
        )

    def test_dtype_preservation(self, test_data):
        """Verify output dtype matches input dtype."""
        data = test_data["hybrid_root_finder"]["simple_quadratic"]
        func = lambda x: x**2 - 4
        
        result = hybrid_root_finder(func, data["xa"], data["xb"], data["tol"], data["itmax"])
        
        assert result.dtype == data["xa"].dtype, "Output dtype should match input dtype"


class TestZbrent:
    """Test suite for zbrent (Brent's method) function."""

    def test_polynomial_shapes(self, test_data):
        """Verify output shapes for all three return values."""
        data = test_data["zbrent"]["polynomial"]
        func = lambda x: (x - 1.5) * (x + 2.3)
        
        root, converged, error = zbrent(
            func, data["xa"], data["xb"], data["tol"], data["itmax"], data["eps"]
        )
        
        assert root.shape == data["xa"].shape, "Root shape should match input"
        assert converged.shape == data["xa"].shape, "Converged shape should match input"
        assert error.shape == data["xa"].shape, "Error shape should match input"

    def test_polynomial_values(self, test_data):
        """Test convergence to known polynomial roots."""
        data = test_data["zbrent"]["polynomial"]
        func = lambda x: (x - 1.5) * (x + 2.3)
        
        root, converged, error = zbrent(
            func, data["xa"], data["xb"], data["tol"], data["itmax"], data["eps"]
        )
        
        np.testing.assert_allclose(
            root,
            data["expected_root"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="Should find correct polynomial roots",
        )
        np.testing.assert_array_equal(
            converged, data["expected_converged"], err_msg="Should converge for valid brackets"
        )
        np.testing.assert_array_equal(
            error, data["expected_error"], err_msg="Should not report errors for valid brackets"
        )

    def test_near_zero_root(self, test_data):
        """Test numerical precision for roots very close to zero."""
        data = test_data["zbrent"]["near_zero"]
        func = lambda x: x**3 - 0.001
        
        root, converged, error = zbrent(
            func, data["xa"], data["xb"], data["tol"], data["itmax"], data["eps"]
        )
        
        np.testing.assert_allclose(
            root,
            data["expected_root"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="Should handle roots near zero with high precision",
        )
        np.testing.assert_array_equal(
            converged, data["expected_converged"], err_msg="Should converge for near-zero roots"
        )

    def test_no_root_error_detection(self, test_data):
        """Test error detection when bracket doesn't contain root."""
        data = test_data["zbrent"]["no_root"]
        func = lambda x: x**2 + 1  # No real roots
        
        root, converged, error = zbrent(
            func, data["xa"], data["xb"], data["tol"], data["itmax"], data["eps"]
        )
        
        np.testing.assert_array_equal(
            converged,
            data["expected_converged"],
            err_msg="Should not converge when no root exists",
        )
        np.testing.assert_array_equal(
            error, data["expected_error"], err_msg="Should report bracketing error"
        )

    def test_dtype_consistency(self, test_data):
        """Verify dtype consistency across outputs."""
        data = test_data["zbrent"]["polynomial"]
        func = lambda x: (x - 1.5) * (x + 2.3)
        
        root, converged, error = zbrent(
            func, data["xa"], data["xb"], data["tol"], data["itmax"], data["eps"]
        )
        
        assert root.dtype == data["xa"].dtype, "Root dtype should match input"
        assert converged.dtype == jnp.bool_, "Converged should be boolean"
        assert error.dtype == jnp.bool_, "Error should be boolean"


# ============================================================================
# Quadratic Solver Tests
# ============================================================================


class TestQuadratic:
    """Test suite for quadratic equation solver."""

    def test_standard_cases_shapes(self, test_data):
        """Verify output shapes match input shapes."""
        data = test_data["quadratic"]["standard"]
        
        r1, r2 = quadratic(data["a"], data["b"], data["c"])
        
        assert r1.shape == data["a"].shape, "First root shape should match input"
        assert r2.shape == data["a"].shape, "Second root shape should match input"

    def test_standard_cases_values(self, test_data):
        """Test quadratic solver on standard cases with known roots."""
        data = test_data["quadratic"]["standard"]
        
        r1, r2 = quadratic(data["a"], data["b"], data["c"])
        
        np.testing.assert_allclose(
            r1,
            data["expected_r1"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="First root should match expected value",
        )
        np.testing.assert_allclose(
            r2,
            data["expected_r2"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="Second root should match expected value",
        )

    def test_zero_discriminant(self, test_data):
        """Test repeated roots when discriminant equals zero."""
        data = test_data["quadratic"]["zero_discriminant"]
        
        r1, r2 = quadratic(data["a"], data["b"], data["c"])
        
        np.testing.assert_allclose(
            r1,
            data["expected_r1"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="First root should match for zero discriminant",
        )
        np.testing.assert_allclose(
            r2,
            data["expected_r2"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="Roots should be equal for zero discriminant",
        )
        np.testing.assert_allclose(
            r1, r2, atol=data["tolerance"], err_msg="Both roots should be identical"
        )

    def test_root_verification(self, test_data):
        """Verify that computed roots satisfy the quadratic equation."""
        data = test_data["quadratic"]["standard"]
        
        r1, r2 = quadratic(data["a"], data["b"], data["c"])
        
        # Check that a*r1² + b*r1 + c ≈ 0
        residual1 = data["a"] * r1**2 + data["b"] * r1 + data["c"]
        residual2 = data["a"] * r2**2 + data["b"] * r2 + data["c"]
        
        np.testing.assert_allclose(
            residual1,
            jnp.zeros_like(residual1),
            atol=1e-8,
            err_msg="First root should satisfy equation",
        )
        np.testing.assert_allclose(
            residual2,
            jnp.zeros_like(residual2),
            atol=1e-8,
            err_msg="Second root should satisfy equation",
        )

    def test_dtype_preservation(self, test_data):
        """Verify output dtype matches input dtype."""
        data = test_data["quadratic"]["standard"]
        
        r1, r2 = quadratic(data["a"], data["b"], data["c"])
        
        assert r1.dtype == data["a"].dtype, "First root dtype should match input"
        assert r2.dtype == data["a"].dtype, "Second root dtype should match input"


# ============================================================================
# Tridiagonal Solver Tests
# ============================================================================


class TestTridiag:
    """Test suite for tridiagonal matrix solver."""

    def test_simple_system_shape(self, test_data):
        """Verify output shape matches input shape."""
        data = test_data["tridiag"]["simple"]
        
        solution = tridiag(data["a"], data["b"], data["c"], data["r"])
        
        assert solution.shape == data["a"].shape, "Solution shape should match input"

    def test_simple_system_values(self, test_data):
        """Test solver on simple symmetric tridiagonal system."""
        data = test_data["tridiag"]["simple"]
        
        solution = tridiag(data["a"], data["b"], data["c"], data["r"])
        
        np.testing.assert_allclose(
            solution,
            data["expected_solution"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="Solution should match expected values",
        )

    def test_diagonal_dominant_system(self, test_data):
        """Test numerical stability on diagonally dominant system."""
        data = test_data["tridiag"]["diagonal_dominant"]
        
        solution = tridiag(data["a"], data["b"], data["c"], data["r"])
        
        np.testing.assert_allclose(
            solution,
            data["expected_solution"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="Should handle diagonally dominant systems accurately",
        )

    def test_solution_verification(self, test_data):
        """Verify solution satisfies the tridiagonal system."""
        data = test_data["tridiag"]["simple"]
        
        solution = tridiag(data["a"], data["b"], data["c"], data["r"])
        
        # Compute A*x for tridiagonal matrix
        n = len(solution)
        residual = jnp.zeros(n)
        
        # First row: b[0]*x[0] + c[0]*x[1] = r[0]
        residual = residual.at[0].set(data["b"][0] * solution[0] + data["c"][0] * solution[1])
        
        # Middle rows: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = r[i]
        for i in range(1, n - 1):
            residual = residual.at[i].set(
                data["a"][i] * solution[i - 1]
                + data["b"][i] * solution[i]
                + data["c"][i] * solution[i + 1]
            )
        
        # Last row: a[n-1]*x[n-2] + b[n-1]*x[n-1] = r[n-1]
        residual = residual.at[n - 1].set(
            data["a"][n - 1] * solution[n - 2] + data["b"][n - 1] * solution[n - 1]
        )
        
        np.testing.assert_allclose(
            residual, data["r"], atol=1e-8, err_msg="Solution should satisfy A*x = r"
        )

    def test_dtype_preservation(self, test_data):
        """Verify output dtype matches input dtype."""
        data = test_data["tridiag"]["simple"]
        
        solution = tridiag(data["a"], data["b"], data["c"], data["r"])
        
        assert solution.dtype == data["a"].dtype, "Solution dtype should match input"


class TestTridiag2Eq:
    """Test suite for coupled tridiagonal system solver."""

    def test_coupled_system_shapes(self, test_data):
        """Verify output shapes match input shapes."""
        data = test_data["tridiag_2eq"]["coupled"]
        
        t, q = tridiag_2eq(
            data["a1"],
            data["b11"],
            data["b12"],
            data["c1"],
            data["d1"],
            data["a2"],
            data["b21"],
            data["b22"],
            data["c2"],
            data["d2"],
            data["n"],
        )
        
        assert t.shape == (data["n"],), "Temperature shape should match n"
        assert q.shape == (data["n"],), "Vapor shape should match n"

    def test_coupled_system_values(self, test_data):
        """Test solver on coupled temperature-vapor system."""
        data = test_data["tridiag_2eq"]["coupled"]
        
        t, q = tridiag_2eq(
            data["a1"],
            data["b11"],
            data["b12"],
            data["c1"],
            data["d1"],
            data["a2"],
            data["b21"],
            data["b22"],
            data["c2"],
            data["d2"],
            data["n"],
        )
        
        np.testing.assert_allclose(
            t,
            data["expected_t"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="Temperature solution should match expected values",
        )
        np.testing.assert_allclose(
            q,
            data["expected_q"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="Vapor solution should match expected values",
        )

    def test_physical_realism(self, test_data):
        """Verify solutions satisfy physical constraints."""
        data = test_data["tridiag_2eq"]["coupled"]
        
        t, q = tridiag_2eq(
            data["a1"],
            data["b11"],
            data["b12"],
            data["c1"],
            data["d1"],
            data["a2"],
            data["b21"],
            data["b22"],
            data["c2"],
            data["d2"],
            data["n"],
        )
        
        # Temperature should be positive (above absolute zero)
        assert jnp.all(t > 0), "Temperature should be positive"
        
        # Vapor mole fraction should be in [0, 1]
        assert jnp.all(q >= 0), "Vapor fraction should be non-negative"
        assert jnp.all(q <= 1), "Vapor fraction should not exceed 1"

    def test_dtype_consistency(self, test_data):
        """Verify output dtypes match input dtypes."""
        data = test_data["tridiag_2eq"]["coupled"]
        
        t, q = tridiag_2eq(
            data["a1"],
            data["b11"],
            data["b12"],
            data["c1"],
            data["d1"],
            data["a2"],
            data["b21"],
            data["b22"],
            data["c2"],
            data["d2"],
            data["n"],
        )
        
        assert t.dtype == data["a1"].dtype, "Temperature dtype should match input"
        assert q.dtype == data["a1"].dtype, "Vapor dtype should match input"


# ============================================================================
# Special Functions Tests
# ============================================================================


class TestLogGammaFunction:
    """Test suite for log-gamma function."""

    def test_standard_values_shape(self, test_data):
        """Verify output shape matches input shape."""
        data = test_data["log_gamma"]["standard"]
        
        result = log_gamma_function(data["x"])
        
        assert result.shape == data["x"].shape, "Output shape should match input"

    def test_standard_values(self, test_data):
        """Test log-gamma on standard values with known results."""
        data = test_data["log_gamma"]["standard"]
        
        result = log_gamma_function(data["x"])
        
        np.testing.assert_allclose(
            result,
            data["expected"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="Log-gamma should match known values",
        )

    def test_large_values(self, test_data):
        """Test numerical stability for large and small arguments."""
        data = test_data["log_gamma"]["large_values"]
        
        result = log_gamma_function(data["x"])
        
        np.testing.assert_allclose(
            result,
            data["expected"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="Should handle extreme values accurately",
        )

    def test_integer_property(self):
        """Verify Γ(n) = (n-1)! for positive integers."""
        n_values = jnp.array([2.0, 3.0, 4.0, 5.0, 6.0])
        factorial_values = jnp.array([1.0, 2.0, 6.0, 24.0, 120.0])
        
        result = log_gamma_function(n_values)
        expected = jnp.log(factorial_values)
        
        np.testing.assert_allclose(
            result, expected, atol=1e-10, err_msg="Should satisfy Γ(n) = (n-1)!"
        )

    def test_dtype_preservation(self, test_data):
        """Verify output dtype matches input dtype."""
        data = test_data["log_gamma"]["standard"]
        
        result = log_gamma_function(data["x"])
        
        assert result.dtype == data["x"].dtype, "Output dtype should match input"


class TestBetaFunction:
    """Test suite for beta function."""

    def test_symmetric_cases_shape(self, test_data):
        """Verify output shape matches input shape."""
        data = test_data["beta_function"]["symmetric"]
        
        result = beta_function(data["a"], data["b"])
        
        assert result.shape == data["a"].shape, "Output shape should match input"

    def test_symmetric_cases_values(self, test_data):
        """Test beta function on symmetric cases B(a,a)."""
        data = test_data["beta_function"]["symmetric"]
        
        result = beta_function(data["a"], data["b"])
        
        np.testing.assert_allclose(
            result,
            data["expected"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="Beta function should match known symmetric values",
        )

    def test_asymmetric_cases(self, test_data):
        """Test beta function on highly asymmetric parameters."""
        data = test_data["beta_function"]["asymmetric"]
        
        result = beta_function(data["a"], data["b"])
        
        np.testing.assert_allclose(
            result,
            data["expected"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="Should handle asymmetric parameters accurately",
        )

    def test_symmetry_property(self):
        """Verify B(a,b) = B(b,a) symmetry property."""
        a = jnp.array([1.5, 2.5, 3.5])
        b = jnp.array([2.0, 3.0, 4.0])
        
        result_ab = beta_function(a, b)
        result_ba = beta_function(b, a)
        
        np.testing.assert_allclose(
            result_ab, result_ba, atol=1e-10, err_msg="Beta function should be symmetric"
        )

    def test_dtype_preservation(self, test_data):
        """Verify output dtype matches input dtype."""
        data = test_data["beta_function"]["symmetric"]
        
        result = beta_function(data["a"], data["b"])
        
        assert result.dtype == data["a"].dtype, "Output dtype should match input"


class TestBetaDistributionPDF:
    """Test suite for beta distribution PDF."""

    def test_uniform_distribution_shape(self, test_data):
        """Verify output shape matches input shape."""
        data = test_data["beta_pdf"]["uniform"]
        
        result = beta_distribution_pdf(data["a"], data["b"], data["x"])
        
        assert result.shape == data["x"].shape, "Output shape should match input"

    def test_uniform_distribution_values(self, test_data):
        """Test Beta(1,1) uniform distribution."""
        data = test_data["beta_pdf"]["uniform"]
        
        result = beta_distribution_pdf(data["a"], data["b"], data["x"])
        
        np.testing.assert_allclose(
            result,
            data["expected"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="Beta(1,1) should be uniform on [0,1]",
        )

    def test_boundary_behavior(self, test_data):
        """Test PDF behavior at boundaries x=0 and x=1."""
        data = test_data["beta_pdf"]["boundaries"]
        
        result = beta_distribution_pdf(data["a"], data["b"], data["x"])
        
        np.testing.assert_allclose(
            result,
            data["expected"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="Should handle boundary values correctly",
        )

    def test_integration_property(self):
        """Verify PDF integrates to 1 (approximately)."""
        a, b = 2.0, 5.0
        x_vals = jnp.linspace(0.001, 0.999, 1000)
        pdf_vals = beta_distribution_pdf(a, b, x_vals)
        
        # Trapezoidal integration
        integral = jnp.trapz(pdf_vals, x_vals)
        
        np.testing.assert_allclose(
            integral, 1.0, atol=1e-3, err_msg="PDF should integrate to 1"
        )

    def test_dtype_preservation(self, test_data):
        """Verify output dtype matches input dtype."""
        data = test_data["beta_pdf"]["uniform"]
        
        result = beta_distribution_pdf(data["a"], data["b"], data["x"])
        
        # Note: scalar inputs may promote dtype
        assert jnp.issubdtype(result.dtype, jnp.floating), "Output should be floating point"


class TestBetaDistributionCDF:
    """Test suite for beta distribution CDF."""

    def test_standard_case_value(self, test_data):
        """Test CDF on standard case."""
        data = test_data["beta_cdf"]["standard"]
        
        result = beta_distribution_cdf(data["a"], data["b"], data["x"])
        
        np.testing.assert_allclose(
            result,
            data["expected"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="CDF should match expected value",
        )

    def test_symmetric_midpoint(self, test_data):
        """Test symmetric distribution at midpoint."""
        data = test_data["beta_cdf"]["symmetric"]
        
        result = beta_distribution_cdf(data["a"], data["b"], data["x"])
        
        np.testing.assert_allclose(
            result,
            data["expected"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="Symmetric CDF at x=0.5 should equal 0.5",
        )

    def test_boundary_values(self):
        """Test CDF at boundaries x=0 and x=1."""
        a, b = 2.0, 3.0
        
        cdf_0 = beta_distribution_cdf(a, b, 0.0)
        cdf_1 = beta_distribution_cdf(a, b, 1.0)
        
        np.testing.assert_allclose(cdf_0, 0.0, atol=1e-10, err_msg="CDF(0) should be 0")
        np.testing.assert_allclose(cdf_1, 1.0, atol=1e-10, err_msg="CDF(1) should be 1")

    def test_monotonicity(self):
        """Verify CDF is monotonically increasing."""
        a, b = 2.0, 5.0
        x_vals = jnp.linspace(0.0, 1.0, 11)
        
        cdf_vals = jnp.array([beta_distribution_cdf(a, b, x) for x in x_vals])
        
        # Check that each value is >= previous value
        diffs = jnp.diff(cdf_vals)
        assert jnp.all(diffs >= -1e-10), "CDF should be monotonically increasing"


class TestBetaFunctionIncompleteCF:
    """Test suite for incomplete beta function continued fraction."""

    def test_convergence_shape(self, test_data):
        """Verify output shape matches input shape."""
        data = test_data["beta_incomplete_cf"]["convergence"]
        
        result = beta_function_incomplete_cf(data["a"], data["b"], data["x"])
        
        assert result.shape == data["a"].shape, "Output shape should match input"

    def test_convergence_values(self, test_data):
        """Test continued fraction convergence at various points."""
        data = test_data["beta_incomplete_cf"]["convergence"]
        
        result = beta_function_incomplete_cf(data["a"], data["b"], data["x"])
        
        np.testing.assert_allclose(
            result,
            data["expected"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="Continued fraction should converge to expected values",
        )

    def test_near_boundaries(self, test_data):
        """Test numerical stability near boundaries x→0 and x→1."""
        data = test_data["beta_incomplete_cf"]["near_boundaries"]
        
        result = beta_function_incomplete_cf(data["a"], data["b"], data["x"])
        
        np.testing.assert_allclose(
            result,
            data["expected"],
            atol=data["tolerance"],
            rtol=data["tolerance"],
            err_msg="Should handle near-boundary values accurately",
        )

    def test_range_constraint(self, test_data):
        """Verify output is in valid range [0, 1]."""
        data = test_data["beta_incomplete_cf"]["convergence"]
        
        result = beta_function_incomplete_cf(data["a"], data["b"], data["x"])
        
        assert jnp.all(result >= 0), "Result should be non-negative"
        assert jnp.all(result <= 1), "Result should not exceed 1"

    def test_dtype_preservation(self, test_data):
        """Verify output dtype matches input dtype."""
        data = test_data["beta_incomplete_cf"]["convergence"]
        
        result = beta_function_incomplete_cf(data["a"], data["b"], data["x"])
        
        assert result.dtype == data["a"].dtype, "Output dtype should match input"


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_quadratic_near_zero_discriminant(self):
        """Test quadratic solver with discriminant very close to zero."""
        a = jnp.array([1.0])
        b = jnp.array([-2.0])
        c = jnp.array([1.0 - 1e-10])  # Discriminant ≈ 0
        
        r1, r2 = quadratic(a, b, c)
        
        # Roots should be very close to each other
        np.testing.assert_allclose(
            r1, r2, atol=1e-5, err_msg="Roots should be nearly equal for near-zero discriminant"
        )

    def test_tridiag_single_equation(self):
        """Test tridiagonal solver with n=1 (single equation)."""
        a = jnp.array([0.0])
        b = jnp.array([2.0])
        c = jnp.array([0.0])
        r = jnp.array([4.0])
        
        solution = tridiag(a, b, c, r)
        
        np.testing.assert_allclose(
            solution, jnp.array([2.0]), atol=1e-10, err_msg="Should solve single equation"
        )

    def test_log_gamma_half_integer(self):
        """Test log-gamma at half-integers using Γ(n+1/2) formula."""
        # Γ(3/2) = √π/2, Γ(5/2) = 3√π/4
        x = jnp.array([1.5, 2.5])
        expected = jnp.log(jnp.array([jnp.sqrt(jnp.pi) / 2, 3 * jnp.sqrt(jnp.pi) / 4]))
        
        result = log_gamma_function(x)
        
        np.testing.assert_allclose(
            result, expected, atol=1e-6, err_msg="Should handle half-integers correctly"
        )

    def test_beta_function_unit_parameters(self):
        """Test beta function with unit parameters B(1,1) = 1."""
        a = jnp.array([1.0])
        b = jnp.array([1.0])
        
        result = beta_function(a, b)
        
        np.testing.assert_allclose(
            result, jnp.array([1.0]), atol=1e-10, err_msg="B(1,1) should equal 1"
        )

    def test_beta_pdf_at_mode(self):
        """Test beta PDF at mode (maximum) for a>1, b>1."""
        a, b = 3.0, 5.0
        # Mode = (a-1)/(a+b-2) for a,b > 1
        mode = (a - 1) / (a + b - 2)
        
        # PDF at mode should be maximum
        x_vals = jnp.linspace(0.01, 0.99, 100)
        pdf_vals = beta_distribution_pdf(a, b, x_vals)
        pdf_at_mode = beta_distribution_pdf(a, b, mode)
        
        assert jnp.all(pdf_at_mode >= pdf_vals), "PDF at mode should be maximum"


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_root_finder_with_quadratic(self):
        """Test root finder using quadratic function."""
        # Find root of x² - 5x + 6 = 0 (roots at x=2 and x=3)
        def func(x):
            return x**2 - 5 * x + 6
        
        xa = jnp.array([1.5])
        xb = jnp.array([2.5])
        
        root = hybrid_root_finder(func, xa, xb, 1e-8, 40)
        
        # Should find root at x=2
        np.testing.assert_allclose(root, jnp.array([2.0]), atol=1e-6)

    def test_beta_pdf_cdf_consistency(self):
        """Test that PDF is derivative of CDF (approximately)."""
        a, b = 2.0, 5.0
        x = 0.3
        dx = 1e-6
        
        # Numerical derivative of CDF
        cdf_x = beta_distribution_cdf(a, b, x)
        cdf_x_plus_dx = beta_distribution_cdf(a, b, x + dx)
        numerical_pdf = (cdf_x_plus_dx - cdf_x) / dx
        
        # Analytical PDF
        analytical_pdf = beta_distribution_pdf(a, b, x)
        
        np.testing.assert_allclose(
            numerical_pdf,
            analytical_pdf,
            rtol=1e-4,
            err_msg="PDF should be derivative of CDF",
        )

    def test_tridiag_2eq_decoupled(self):
        """Test coupled solver reduces to two independent systems when uncoupled."""
        n = 3
        # Set coupling coefficients to zero
        a1 = jnp.array([0.0, -1.0, -1.0])
        b11 = jnp.array([2.0, 2.0, 2.0])
        b12 = jnp.array([0.0, 0.0, 0.0])  # No coupling
        c1 = jnp.array([-1.0, -1.0, 0.0])
        d1 = jnp.array([1.0, 0.0, 1.0])
        
        a2 = jnp.array([0.0, -1.0, -1.0])
        b21 = jnp.array([0.0, 0.0, 0.0])  # No coupling
        b22 = jnp.array([2.0, 2.0, 2.0])
        c2 = jnp.array([-1.0, -1.0, 0.0])
        d2 = jnp.array([2.0, 0.0, 2.0])
        
        t, q = tridiag_2eq(a1, b11, b12, c1, d1, a2, b21, b22, c2, d2, n)
        
        # Solve independently
        t_independent = tridiag(a1, b11, c1, d1)
        q_independent = tridiag(a2, b22, c2, d2)
        
        np.testing.assert_allclose(
            t, t_independent, atol=1e-10, err_msg="Decoupled T should match independent solve"
        )
        np.testing.assert_allclose(
            q, q_independent, atol=1e-10, err_msg="Decoupled q should match independent solve"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])