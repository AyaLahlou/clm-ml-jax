"""
Comprehensive pytest suite for MLclm_varpar module.

Tests the multilayer canopy parameter configuration system, including:
- Default parameter retrieval (get_mlcanopy_params)
- Parameter validation (validate_mlcanopy_params)
- Module constants consistency
- Integration with JAX array indexing conventions

The multilayer canopy model divides the canopy into vertical layers and
distinguishes between sunlit and shaded leaves for accurate radiative transfer,
photosynthesis, and energy balance calculations.
"""

import pytest
import numpy as np
from typing import NamedTuple

# Import the module under test
# Assuming the module is named MLclm_varpar and contains:
# - MLCanopyParams (NamedTuple)
# - get_mlcanopy_params() function
# - validate_mlcanopy_params(params) function
# - Module constants: DEFAULT_MLCANOPY_PARAMS, NLEVMLCAN, NLEAF, ISUN, ISHA
from multilayer_canopy.MLclm_varpar import (
    MLCanopyParams,
    get_mlcanopy_params,
    validate_mlcanopy_params,
    DEFAULT_MLCANOPY_PARAMS,
    NLEVMLCAN,
    NLEAF,
    ISUN,
    ISHA,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_params():
    """
    Fixture providing default MLCanopyParams instance.

    Returns:
        MLCanopyParams: Default parameter configuration with nlevmlcan=100,
                       nleaf=2, isun=1, isha=2
    """
    return get_mlcanopy_params()


@pytest.fixture
def valid_params_variants():
    """
    Fixture providing various valid parameter configurations.

    Returns:
        list: List of valid MLCanopyParams instances with different nlevmlcan values
    """
    return [
        MLCanopyParams(nlevmlcan=1, nleaf=2, isun=1, isha=2),  # Minimum layers
        MLCanopyParams(nlevmlcan=50, nleaf=2, isun=1, isha=2),  # Half default
        MLCanopyParams(nlevmlcan=100, nleaf=2, isun=1, isha=2),  # Default
        MLCanopyParams(nlevmlcan=200, nleaf=2, isun=1, isha=2),  # Double default
        MLCanopyParams(nlevmlcan=1000, nleaf=2, isun=1, isha=2),  # Large
    ]


@pytest.fixture
def invalid_params_variants():
    """
    Fixture providing various invalid parameter configurations.

    Returns:
        list: List of tuples (params, reason) for invalid configurations
    """
    return [
        (
            MLCanopyParams(nlevmlcan=0, nleaf=2, isun=1, isha=2),
            "zero layers",
        ),
        (
            MLCanopyParams(nlevmlcan=-10, nleaf=2, isun=1, isha=2),
            "negative layers",
        ),
        (
            MLCanopyParams(nlevmlcan=100, nleaf=1, isun=1, isha=2),
            "wrong nleaf (too few)",
        ),
        (
            MLCanopyParams(nlevmlcan=100, nleaf=3, isun=1, isha=2),
            "wrong nleaf (too many)",
        ),
        (
            MLCanopyParams(nlevmlcan=100, nleaf=2, isun=0, isha=2),
            "wrong isun index",
        ),
        (
            MLCanopyParams(nlevmlcan=100, nleaf=2, isun=1, isha=3),
            "wrong isha index",
        ),
        (
            MLCanopyParams(nlevmlcan=100, nleaf=2, isun=2, isha=1),
            "swapped indices",
        ),
    ]


# ============================================================================
# Tests for get_mlcanopy_params()
# ============================================================================


class TestGetMLCanopyParams:
    """Test suite for get_mlcanopy_params() function."""

    def test_returns_mlcanopy_params_type(self, default_params):
        """
        Verify get_mlcanopy_params() returns MLCanopyParams instance.

        Tests that the function returns the correct type, which is essential
        for type checking and downstream usage.
        """
        assert isinstance(
            default_params, MLCanopyParams
        ), f"Expected MLCanopyParams, got {type(default_params)}"

    def test_default_nlevmlcan_value(self, default_params):
        """
        Verify default nlevmlcan equals 100.

        The default 100 layers provides fine vertical resolution for
        radiative transfer and turbulence calculations.
        """
        assert (
            default_params.nlevmlcan == 100
        ), f"Expected nlevmlcan=100, got {default_params.nlevmlcan}"

    def test_default_nleaf_value(self, default_params):
        """
        Verify default nleaf equals 2 (sunlit and shaded).

        The model always uses exactly 2 leaf types: sunlit leaves that
        receive direct beam radiation, and shaded leaves that receive
        only diffuse radiation.
        """
        assert (
            default_params.nleaf == 2
        ), f"Expected nleaf=2, got {default_params.nleaf}"

    def test_default_isun_value(self, default_params):
        """
        Verify default isun equals 1 (sunlit index).

        Uses 1-based Fortran indexing convention. Must subtract 1 for
        0-based JAX/NumPy array indexing.
        """
        assert (
            default_params.isun == 1
        ), f"Expected isun=1, got {default_params.isun}"

    def test_default_isha_value(self, default_params):
        """
        Verify default isha equals 2 (shaded index).

        Uses 1-based Fortran indexing convention. Must subtract 1 for
        0-based JAX/NumPy array indexing.
        """
        assert (
            default_params.isha == 2
        ), f"Expected isha=2, got {default_params.isha}"

    def test_all_fields_are_integers(self, default_params):
        """
        Verify all parameter fields are integers.

        Integer types are required for array indexing and layer counting.
        """
        assert isinstance(
            default_params.nlevmlcan, int
        ), f"nlevmlcan should be int, got {type(default_params.nlevmlcan)}"
        assert isinstance(
            default_params.nleaf, int
        ), f"nleaf should be int, got {type(default_params.nleaf)}"
        assert isinstance(
            default_params.isun, int
        ), f"isun should be int, got {type(default_params.isun)}"
        assert isinstance(
            default_params.isha, int
        ), f"isha should be int, got {type(default_params.isha)}"

    def test_sunlit_shaded_indices_distinct(self, default_params):
        """
        Verify sunlit and shaded indices are different.

        The indices must be distinct to properly separate sunlit and
        shaded leaf arrays.
        """
        assert (
            default_params.isun != default_params.isha
        ), "Sunlit and shaded indices must be different"

    def test_nlevmlcan_positive(self, default_params):
        """
        Verify nlevmlcan is positive.

        At least one layer is required for the canopy model to function.
        """
        assert (
            default_params.nlevmlcan > 0
        ), f"nlevmlcan must be positive, got {default_params.nlevmlcan}"

    def test_function_is_deterministic(self):
        """
        Verify get_mlcanopy_params() returns consistent values.

        Multiple calls should return identical parameter values.
        """
        params1 = get_mlcanopy_params()
        params2 = get_mlcanopy_params()
        assert params1 == params2, "Function should return consistent values"

    def test_namedtuple_immutability(self, default_params):
        """
        Verify MLCanopyParams is immutable (NamedTuple property).

        Parameters should not be modifiable after creation to prevent
        accidental changes.
        """
        with pytest.raises(AttributeError):
            default_params.nlevmlcan = 50


# ============================================================================
# Tests for validate_mlcanopy_params()
# ============================================================================


class TestValidateMLCanopyParams:
    """Test suite for validate_mlcanopy_params() function."""

    def test_validate_default_params_returns_true(self, default_params):
        """
        Verify default parameters pass validation.

        The default configuration should always be valid.
        """
        result = validate_mlcanopy_params(default_params)
        assert result is True, "Default parameters should be valid"

    def test_validate_returns_boolean(self, default_params):
        """
        Verify validate_mlcanopy_params() returns boolean type.

        The function should return True or False, not other truthy/falsy values.
        """
        result = validate_mlcanopy_params(default_params)
        assert isinstance(
            result, bool
        ), f"Expected bool return type, got {type(result)}"

    @pytest.mark.parametrize(
        "nlevmlcan,expected",
        [
            (1, True),  # Minimum valid
            (10, True),  # Small but valid
            (100, True),  # Default
            (500, True),  # Large
            (1000, True),  # Very large
            (10000, True),  # Extremely large but still valid
        ],
    )
    def test_validate_various_nlevmlcan_values(self, nlevmlcan, expected):
        """
        Verify validation accepts any positive nlevmlcan value.

        Tests that the validator correctly handles various layer counts,
        from minimal (1) to very large (10000).

        Args:
            nlevmlcan: Number of canopy layers to test
            expected: Expected validation result (True for all positive values)
        """
        params = MLCanopyParams(nlevmlcan=nlevmlcan, nleaf=2, isun=1, isha=2)
        result = validate_mlcanopy_params(params)
        assert (
            result == expected
        ), f"nlevmlcan={nlevmlcan} should be {'valid' if expected else 'invalid'}"

    @pytest.mark.parametrize(
        "nlevmlcan,expected",
        [
            (0, False),  # Zero layers invalid
            (-1, False),  # Negative invalid
            (-10, False),  # Large negative invalid
            (-100, False),  # Very negative invalid
        ],
    )
    def test_validate_invalid_nlevmlcan_values(self, nlevmlcan, expected):
        """
        Verify validation rejects zero and negative nlevmlcan values.

        At least one layer is required for the canopy model.

        Args:
            nlevmlcan: Invalid number of layers to test
            expected: Expected validation result (False for all)
        """
        params = MLCanopyParams(nlevmlcan=nlevmlcan, nleaf=2, isun=1, isha=2)
        result = validate_mlcanopy_params(params)
        assert (
            result == expected
        ), f"nlevmlcan={nlevmlcan} should be invalid"

    @pytest.mark.parametrize(
        "nleaf,expected",
        [
            (0, False),  # Too few
            (1, False),  # Too few
            (2, True),  # Correct
            (3, False),  # Too many
            (10, False),  # Way too many
        ],
    )
    def test_validate_nleaf_must_equal_two(self, nleaf, expected):
        """
        Verify validation requires nleaf to be exactly 2.

        The model is designed for exactly 2 leaf types: sunlit and shaded.

        Args:
            nleaf: Number of leaf types to test
            expected: Expected validation result (True only for nleaf=2)
        """
        params = MLCanopyParams(nlevmlcan=100, nleaf=nleaf, isun=1, isha=2)
        result = validate_mlcanopy_params(params)
        assert (
            result == expected
        ), f"nleaf={nleaf} should be {'valid' if expected else 'invalid'}"

    @pytest.mark.parametrize(
        "isun,expected",
        [
            (0, False),  # Wrong index
            (1, True),  # Correct
            (2, False),  # Wrong (that's isha)
            (3, False),  # Wrong index
        ],
    )
    def test_validate_isun_must_equal_one(self, isun, expected):
        """
        Verify validation requires isun to be exactly 1.

        The sunlit index must be 1 per the Fortran indexing convention.

        Args:
            isun: Sunlit index to test
            expected: Expected validation result (True only for isun=1)
        """
        params = MLCanopyParams(nlevmlcan=100, nleaf=2, isun=isun, isha=2)
        result = validate_mlcanopy_params(params)
        assert (
            result == expected
        ), f"isun={isun} should be {'valid' if expected else 'invalid'}"

    @pytest.mark.parametrize(
        "isha,expected",
        [
            (0, False),  # Wrong index
            (1, False),  # Wrong (that's isun)
            (2, True),  # Correct
            (3, False),  # Wrong index
        ],
    )
    def test_validate_isha_must_equal_two(self, isha, expected):
        """
        Verify validation requires isha to be exactly 2.

        The shaded index must be 2 per the Fortran indexing convention.

        Args:
            isha: Shaded index to test
            expected: Expected validation result (True only for isha=2)
        """
        params = MLCanopyParams(nlevmlcan=100, nleaf=2, isun=1, isha=isha)
        result = validate_mlcanopy_params(params)
        assert (
            result == expected
        ), f"isha={isha} should be {'valid' if expected else 'invalid'}"

    def test_validate_swapped_indices_invalid(self):
        """
        Verify validation rejects swapped sunlit/shaded indices.

        isun=2 and isha=1 violates the indexing convention and should fail.
        """
        params = MLCanopyParams(nlevmlcan=100, nleaf=2, isun=2, isha=1)
        result = validate_mlcanopy_params(params)
        assert result is False, "Swapped indices should be invalid"

    def test_validate_all_valid_variants(self, valid_params_variants):
        """
        Verify validation accepts all valid parameter variants.

        Tests multiple valid configurations with different nlevmlcan values.
        """
        for params in valid_params_variants:
            result = validate_mlcanopy_params(params)
            assert (
                result is True
            ), f"Valid params {params} should pass validation"

    def test_validate_all_invalid_variants(self, invalid_params_variants):
        """
        Verify validation rejects all invalid parameter variants.

        Tests multiple invalid configurations covering different failure modes.
        """
        for params, reason in invalid_params_variants:
            result = validate_mlcanopy_params(params)
            assert (
                result is False
            ), f"Invalid params ({reason}) should fail validation: {params}"

    def test_validate_multiple_violations(self):
        """
        Verify validation rejects parameters with multiple violations.

        Tests that validation fails when multiple constraints are violated.
        """
        params = MLCanopyParams(nlevmlcan=0, nleaf=3, isun=0, isha=3)
        result = validate_mlcanopy_params(params)
        assert (
            result is False
        ), "Parameters with multiple violations should be invalid"


# ============================================================================
# Tests for Module Constants
# ============================================================================


class TestModuleConstants:
    """Test suite for module-level constants."""

    def test_nlevmlcan_constant_value(self):
        """
        Verify NLEVMLCAN constant equals 100.

        Module constant should match default parameter value.
        """
        assert NLEVMLCAN == 100, f"Expected NLEVMLCAN=100, got {NLEVMLCAN}"

    def test_nleaf_constant_value(self):
        """
        Verify NLEAF constant equals 2.

        Module constant should match default parameter value.
        """
        assert NLEAF == 2, f"Expected NLEAF=2, got {NLEAF}"

    def test_isun_constant_value(self):
        """
        Verify ISUN constant equals 1.

        Module constant should match default parameter value.
        """
        assert ISUN == 1, f"Expected ISUN=1, got {ISUN}"

    def test_isha_constant_value(self):
        """
        Verify ISHA constant equals 2.

        Module constant should match default parameter value.
        """
        assert ISHA == 2, f"Expected ISHA=2, got {ISHA}"

    def test_constants_match_default_params(self, default_params):
        """
        Verify module constants match default parameter values.

        Ensures consistency between convenience constants and parameter object.
        """
        assert (
            NLEVMLCAN == default_params.nlevmlcan
        ), "NLEVMLCAN should match default params"
        assert (
            NLEAF == default_params.nleaf
        ), "NLEAF should match default params"
        assert ISUN == default_params.isun, "ISUN should match default params"
        assert ISHA == default_params.isha, "ISHA should match default params"

    def test_default_mlcanopy_params_constant_type(self):
        """
        Verify DEFAULT_MLCANOPY_PARAMS is MLCanopyParams instance.

        The module-level default should be the correct type.
        """
        assert isinstance(
            DEFAULT_MLCANOPY_PARAMS, MLCanopyParams
        ), f"Expected MLCanopyParams, got {type(DEFAULT_MLCANOPY_PARAMS)}"

    def test_default_mlcanopy_params_equals_getter(self):
        """
        Verify DEFAULT_MLCANOPY_PARAMS equals get_mlcanopy_params() output.

        The module constant and function should return identical values.
        """
        params = get_mlcanopy_params()
        assert (
            DEFAULT_MLCANOPY_PARAMS == params
        ), "DEFAULT_MLCANOPY_PARAMS should equal get_mlcanopy_params()"

    def test_default_mlcanopy_params_is_valid(self):
        """
        Verify DEFAULT_MLCANOPY_PARAMS passes validation.

        The module-level default should always be valid.
        """
        result = validate_mlcanopy_params(DEFAULT_MLCANOPY_PARAMS)
        assert (
            result is True
        ), "DEFAULT_MLCANOPY_PARAMS should be valid"


# ============================================================================
# Integration Tests
# ============================================================================


class TestJAXArrayIndexing:
    """Test suite for JAX/NumPy array indexing integration."""

    def test_convert_fortran_to_python_indexing(self, default_params):
        """
        Verify 1-based Fortran indices convert correctly to 0-based Python.

        JAX and NumPy use 0-based indexing, so isun=1 -> index 0,
        isha=2 -> index 1.
        """
        sunlit_idx = default_params.isun - 1
        shaded_idx = default_params.isha - 1

        assert sunlit_idx == 0, f"Expected sunlit index 0, got {sunlit_idx}"
        assert (
            shaded_idx == 1
        ), f"Expected shaded index 1, got {shaded_idx}"

    def test_array_indexing_with_mock_data(self, default_params):
        """
        Verify parameter indices correctly access mock array data.

        Creates a mock [nlevmlcan, nleaf] array and verifies that
        the indices correctly separate sunlit and shaded data.
        """
        # Create mock array: [nlevmlcan, nleaf]
        mock_array = np.random.rand(default_params.nlevmlcan, default_params.nleaf)

        # Access using converted indices
        sunlit_data = mock_array[:, default_params.isun - 1]
        shaded_data = mock_array[:, default_params.isha - 1]

        # Verify shapes
        assert sunlit_data.shape == (
            default_params.nlevmlcan,
        ), f"Expected shape ({default_params.nlevmlcan},), got {sunlit_data.shape}"
        assert shaded_data.shape == (
            default_params.nlevmlcan,
        ), f"Expected shape ({default_params.nlevmlcan},), got {shaded_data.shape}"

        # Verify data is different (extremely unlikely to be identical)
        assert not np.allclose(
            sunlit_data, shaded_data
        ), "Sunlit and shaded data should be different"

    def test_single_layer_array_indexing(self):
        """
        Verify array indexing works with single-layer (big-leaf) model.

        Tests the minimal case of nlevmlcan=1.
        """
        params = MLCanopyParams(nlevmlcan=1, nleaf=2, isun=1, isha=2)
        mock_array = np.array([[1.0, 2.0]])  # Shape: [1, 2]

        sunlit_data = mock_array[:, params.isun - 1]
        shaded_data = mock_array[:, params.isha - 1]

        assert sunlit_data.shape == (1,), f"Expected shape (1,), got {sunlit_data.shape}"
        assert shaded_data.shape == (1,), f"Expected shape (1,), got {shaded_data.shape}"
        assert np.allclose(
            sunlit_data, [1.0]
        ), f"Expected [1.0], got {sunlit_data}"
        assert np.allclose(
            shaded_data, [2.0]
        ), f"Expected [2.0], got {shaded_data}"

    @pytest.mark.parametrize("nlevmlcan", [1, 10, 50, 100, 200, 1000])
    def test_array_dimensions_for_various_layer_counts(self, nlevmlcan):
        """
        Verify array dimensions scale correctly with nlevmlcan.

        Tests that arrays can be properly dimensioned for various layer counts.

        Args:
            nlevmlcan: Number of canopy layers to test
        """
        params = MLCanopyParams(nlevmlcan=nlevmlcan, nleaf=2, isun=1, isha=2)

        # Create typical canopy arrays
        radiation_array = np.zeros((params.nlevmlcan, params.nleaf))
        flux_array = np.zeros((params.nlevmlcan, params.nleaf))
        temperature_array = np.zeros((params.nlevmlcan, params.nleaf))

        # Verify shapes
        expected_shape = (nlevmlcan, 2)
        assert (
            radiation_array.shape == expected_shape
        ), f"Expected {expected_shape}, got {radiation_array.shape}"
        assert (
            flux_array.shape == expected_shape
        ), f"Expected {expected_shape}, got {flux_array.shape}"
        assert (
            temperature_array.shape == expected_shape
        ), f"Expected {expected_shape}, got {temperature_array.shape}"


class TestPhysicalConsistency:
    """Test suite for physical consistency checks."""

    def test_sunlit_fraction_decreases_with_depth(self, default_params):
        """
        Verify sunlit fraction decreases exponentially with canopy depth.

        Physical model: sunlit_fraction = exp(-K_beam * LAI_cumulative)
        This is a fundamental property of Beer's law radiative transfer.
        """
        # Mock cumulative LAI increasing with depth
        lai_cumulative = np.linspace(0, 6, default_params.nlevmlcan)
        k_beam = 0.5  # Typical beam extinction coefficient

        # Calculate sunlit fraction
        sunlit_fraction = np.exp(-k_beam * lai_cumulative)

        # Verify monotonic decrease
        assert np.all(
            np.diff(sunlit_fraction) <= 0
        ), "Sunlit fraction should decrease with depth"

        # Verify bounds
        assert np.all(
            (sunlit_fraction >= 0) & (sunlit_fraction <= 1)
        ), "Sunlit fraction should be in [0, 1]"

        # Verify top of canopy is fully sunlit
        assert np.isclose(
            sunlit_fraction[0], 1.0, atol=1e-6
        ), "Top of canopy should be fully sunlit"

    def test_shaded_fraction_complements_sunlit(self, default_params):
        """
        Verify shaded fraction = 1 - sunlit fraction.

        Every leaf is either sunlit or shaded, so fractions must sum to 1.
        """
        # Mock sunlit fractions
        sunlit_fraction = np.random.rand(default_params.nlevmlcan)
        shaded_fraction = 1.0 - sunlit_fraction

        # Verify sum equals 1
        total_fraction = sunlit_fraction + shaded_fraction
        assert np.allclose(
            total_fraction, 1.0, atol=1e-10
        ), "Sunlit + shaded fractions should equal 1"

        # Verify both are in valid range
        assert np.all(
            (sunlit_fraction >= 0) & (sunlit_fraction <= 1)
        ), "Sunlit fraction should be in [0, 1]"
        assert np.all(
            (shaded_fraction >= 0) & (shaded_fraction <= 1)
        ), "Shaded fraction should be in [0, 1]"

    def test_layer_count_affects_resolution(self):
        """
        Verify that more layers provide finer vertical resolution.

        More layers allow better representation of vertical gradients
        in radiation, temperature, and fluxes.
        """
        params_coarse = MLCanopyParams(nlevmlcan=10, nleaf=2, isun=1, isha=2)
        params_fine = MLCanopyParams(nlevmlcan=100, nleaf=2, isun=1, isha=2)

        # Mock LAI profile
        total_lai = 6.0
        lai_per_layer_coarse = total_lai / params_coarse.nlevmlcan
        lai_per_layer_fine = total_lai / params_fine.nlevmlcan

        # Finer resolution has smaller LAI per layer
        assert (
            lai_per_layer_fine < lai_per_layer_coarse
        ), "Finer resolution should have smaller LAI per layer"

        # Verify resolution ratio
        expected_ratio = params_fine.nlevmlcan / params_coarse.nlevmlcan
        actual_ratio = lai_per_layer_coarse / lai_per_layer_fine
        assert np.isclose(
            actual_ratio, expected_ratio, atol=1e-10
        ), f"Resolution ratio should be {expected_ratio}, got {actual_ratio}"


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_minimum_valid_configuration(self):
        """
        Verify minimum valid configuration (1 layer) works correctly.

        Single-layer model is the big-leaf approximation.
        """
        params = MLCanopyParams(nlevmlcan=1, nleaf=2, isun=1, isha=2)
        assert validate_mlcanopy_params(params), "Minimum config should be valid"

        # Verify can create arrays
        test_array = np.zeros((params.nlevmlcan, params.nleaf))
        assert test_array.shape == (1, 2), f"Expected shape (1, 2), got {test_array.shape}"

    def test_very_large_layer_count(self):
        """
        Verify very large layer count (10000) is accepted.

        Tests computational scalability, though this may be impractical
        for actual simulations.
        """
        params = MLCanopyParams(nlevmlcan=10000, nleaf=2, isun=1, isha=2)
        assert validate_mlcanopy_params(params), "Large layer count should be valid"

        # Verify can create arrays (may be slow/memory-intensive)
        test_array = np.zeros((params.nlevmlcan, params.nleaf))
        assert test_array.shape == (
            10000,
            2,
        ), f"Expected shape (10000, 2), got {test_array.shape}"

    def test_boundary_nlevmlcan_values(self):
        """
        Verify boundary values for nlevmlcan (0 and 1).

        0 should be invalid, 1 should be valid.
        """
        params_zero = MLCanopyParams(nlevmlcan=0, nleaf=2, isun=1, isha=2)
        params_one = MLCanopyParams(nlevmlcan=1, nleaf=2, isun=1, isha=2)

        assert not validate_mlcanopy_params(
            params_zero
        ), "Zero layers should be invalid"
        assert validate_mlcanopy_params(
            params_one
        ), "One layer should be valid"

    def test_all_wrong_indices_combination(self):
        """
        Verify validation fails when all indices are wrong.

        Tests the worst-case scenario where multiple constraints fail.
        """
        params = MLCanopyParams(nlevmlcan=100, nleaf=5, isun=3, isha=4)
        assert not validate_mlcanopy_params(
            params
        ), "All wrong indices should be invalid"

    def test_params_with_extreme_nlevmlcan(self):
        """
        Verify validation handles extreme nlevmlcan values correctly.

        Tests very large positive and negative values.
        """
        # Very large positive (should be valid)
        params_large = MLCanopyParams(
            nlevmlcan=1000000, nleaf=2, isun=1, isha=2
        )
        assert validate_mlcanopy_params(
            params_large
        ), "Very large nlevmlcan should be valid"

        # Very large negative (should be invalid)
        params_negative = MLCanopyParams(
            nlevmlcan=-1000000, nleaf=2, isun=1, isha=2
        )
        assert not validate_mlcanopy_params(
            params_negative
        ), "Very negative nlevmlcan should be invalid"


# ============================================================================
# Documentation and Metadata Tests
# ============================================================================


class TestDocumentationAndMetadata:
    """Test suite for documentation and metadata consistency."""

    def test_mlcanopy_params_has_docstring(self):
        """
        Verify MLCanopyParams has documentation.

        NamedTuples should have docstrings explaining their purpose.
        """
        assert (
            MLCanopyParams.__doc__ is not None
        ), "MLCanopyParams should have docstring"

    def test_get_mlcanopy_params_has_docstring(self):
        """
        Verify get_mlcanopy_params() has documentation.

        Functions should have docstrings explaining their purpose and returns.
        """
        assert (
            get_mlcanopy_params.__doc__ is not None
        ), "get_mlcanopy_params should have docstring"

    def test_validate_mlcanopy_params_has_docstring(self):
        """
        Verify validate_mlcanopy_params() has documentation.

        Functions should have docstrings explaining their purpose and parameters.
        """
        assert (
            validate_mlcanopy_params.__doc__ is not None
        ), "validate_mlcanopy_params should have docstring"

    def test_mlcanopy_params_field_names(self):
        """
        Verify MLCanopyParams has expected field names.

        Ensures the NamedTuple structure matches specification.
        """
        expected_fields = {"nlevmlcan", "nleaf", "isun", "isha"}
        actual_fields = set(MLCanopyParams._fields)
        assert (
            actual_fields == expected_fields
        ), f"Expected fields {expected_fields}, got {actual_fields}"

    def test_mlcanopy_params_field_count(self):
        """
        Verify MLCanopyParams has exactly 4 fields.

        Ensures no extra or missing fields in the NamedTuple.
        """
        assert (
            len(MLCanopyParams._fields) == 4
        ), f"Expected 4 fields, got {len(MLCanopyParams._fields)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])