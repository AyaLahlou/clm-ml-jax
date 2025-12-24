"""
Comprehensive pytest suite for decompMod module.

This module tests the domain decomposition functionality for CTSM spatial hierarchy,
including gridcells, landunits, columns, and patches.

Tests cover:
- get_clump_bounds: Returns default single-cell bounds regardless of processor index
- create_bounds: Creates BoundsType instances with specified spatial decomposition
- BoundsType namedtuple: Validates structure and field access
- Edge cases: Single elements, large domains, subdomain offsets, boundary conditions
"""

import pytest
from typing import NamedTuple
import sys
from pathlib import Path

# Import the module under test
# Adjust the import path as needed for your project structure
try:
    from clm_src_main.decompMod import get_clump_bounds, create_bounds, BoundsType
except ImportError:
    # Fallback for different import contexts
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
    from clm_src_main.decompMod import get_clump_bounds, create_bounds, BoundsType


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def single_cell_bounds():
    """
    Fixture providing bounds for a single grid cell.
    
    Returns a BoundsType with all indices set to 1, representing
    the minimal spatial decomposition unit.
    """
    return BoundsType(
        begg=1, endg=1,
        begl=1, endl=1,
        begc=1, endc=1,
        begp=1, endp=1
    )


@pytest.fixture
def small_domain_bounds():
    """
    Fixture providing bounds for a small domain.
    
    Represents a typical small simulation with:
    - 10 gridcells
    - 15 landunits (1.5 per gridcell average)
    - 30 columns (2 per landunit average)
    - 45 patches (1.5 per column average)
    """
    return BoundsType(
        begg=1, endg=10,
        begl=1, endl=15,
        begc=1, endc=30,
        begp=1, endp=45
    )


@pytest.fixture
def large_domain_bounds():
    """
    Fixture providing bounds for a large global domain.
    
    Represents a high-resolution global simulation with:
    - 1000 gridcells
    - 2500 landunits
    - 5000 columns
    - 10000 patches
    """
    return BoundsType(
        begg=1, endg=1000,
        begl=1, endl=2500,
        begc=1, endc=5000,
        begp=1, endp=10000
    )


@pytest.fixture
def subdomain_bounds():
    """
    Fixture providing bounds for a subdomain with offset indices.
    
    Simulates MPI domain decomposition where this processor handles
    a subset of the global domain with non-unit starting indices.
    """
    return BoundsType(
        begg=100, endg=200,
        begl=250, endl=500,
        begc=600, endc=1200,
        begp=1500, endp=3000
    )


# ============================================================================
# BoundsType Structure Tests
# ============================================================================

class TestBoundsTypeStructure:
    """Tests for the BoundsType namedtuple structure."""
    
    def test_boundstype_is_namedtuple(self):
        """Verify BoundsType is a proper namedtuple."""
        assert issubclass(BoundsType, tuple), "BoundsType should be a namedtuple"
        assert hasattr(BoundsType, '_fields'), "BoundsType should have _fields attribute"
    
    def test_boundstype_fields(self):
        """Verify BoundsType has all required fields."""
        expected_fields = ('begg', 'endg', 'begl', 'endl', 'begc', 'endc', 'begp', 'endp')
        assert BoundsType._fields == expected_fields, \
            f"BoundsType fields should be {expected_fields}, got {BoundsType._fields}"
    
    def test_boundstype_field_access(self, single_cell_bounds):
        """Verify all fields are accessible by name."""
        assert single_cell_bounds.begg == 1, "begg field should be accessible"
        assert single_cell_bounds.endg == 1, "endg field should be accessible"
        assert single_cell_bounds.begl == 1, "begl field should be accessible"
        assert single_cell_bounds.endl == 1, "endl field should be accessible"
        assert single_cell_bounds.begc == 1, "begc field should be accessible"
        assert single_cell_bounds.endc == 1, "endc field should be accessible"
        assert single_cell_bounds.begp == 1, "begp field should be accessible"
        assert single_cell_bounds.endp == 1, "endp field should be accessible"
    
    def test_boundstype_immutability(self, single_cell_bounds):
        """Verify BoundsType instances are immutable."""
        with pytest.raises(AttributeError):
            single_cell_bounds.begg = 2


# ============================================================================
# get_clump_bounds Tests
# ============================================================================

class TestGetClumpBounds:
    """Tests for get_clump_bounds function."""
    
    @pytest.mark.parametrize("processor_index", [
        0,      # Default/first processor
        1,      # Second processor
        42,     # Arbitrary processor
        100,    # Large processor index
        999,    # Very large processor index
    ])
    def test_get_clump_bounds_returns_single_cell(self, processor_index):
        """
        Test that get_clump_bounds always returns single cell bounds.
        
        The processor index parameter is unused in the current implementation,
        so all calls should return identical bounds with all values set to 1.
        """
        result = get_clump_bounds(processor_index)
        
        assert isinstance(result, BoundsType), \
            f"Result should be BoundsType, got {type(result)}"
        
        assert result.begg == 1, f"begg should be 1, got {result.begg}"
        assert result.endg == 1, f"endg should be 1, got {result.endg}"
        assert result.begl == 1, f"begl should be 1, got {result.begl}"
        assert result.endl == 1, f"endl should be 1, got {result.endl}"
        assert result.begc == 1, f"begc should be 1, got {result.begc}"
        assert result.endc == 1, f"endc should be 1, got {result.endc}"
        assert result.begp == 1, f"begp should be 1, got {result.begp}"
        assert result.endp == 1, f"endp should be 1, got {result.endp}"
    
    def test_get_clump_bounds_negative_index(self):
        """
        Test get_clump_bounds with negative processor index.
        
        Verifies that the processor index parameter is truly unused,
        even with invalid negative values.
        """
        result = get_clump_bounds(-1)
        
        assert isinstance(result, BoundsType), \
            "Result should be BoundsType even with negative index"
        assert result == BoundsType(1, 1, 1, 1, 1, 1, 1, 1), \
            "Should return single cell bounds even with negative index"
    
    def test_get_clump_bounds_consistency(self):
        """
        Test that multiple calls return consistent results.
        
        Verifies that the function is deterministic and returns
        the same bounds regardless of input.
        """
        result1 = get_clump_bounds(0)
        result2 = get_clump_bounds(100)
        result3 = get_clump_bounds(-5)
        
        assert result1 == result2 == result3, \
            "All calls should return identical bounds"
    
    def test_get_clump_bounds_return_type(self):
        """Verify get_clump_bounds returns correct type."""
        result = get_clump_bounds(0)
        assert type(result).__name__ == 'BoundsType', \
            f"Should return BoundsType, got {type(result).__name__}"


# ============================================================================
# create_bounds Tests - Nominal Cases
# ============================================================================

class TestCreateBoundsNominal:
    """Tests for create_bounds function with nominal/typical inputs."""
    
    def test_create_bounds_single_cell(self, single_cell_bounds):
        """
        Test creating bounds for a single cell at all hierarchy levels.
        
        This represents the minimal spatial decomposition unit.
        """
        result = create_bounds(
            begg=1, endg=1,
            begl=1, endl=1,
            begc=1, endc=1,
            begp=1, endp=1
        )
        
        assert result == single_cell_bounds, \
            f"Single cell bounds mismatch: expected {single_cell_bounds}, got {result}"
    
    def test_create_bounds_small_domain(self, small_domain_bounds):
        """
        Test creating bounds for a small domain.
        
        Typical hierarchy: 10 gridcells, 15 landunits, 30 columns, 45 patches.
        """
        result = create_bounds(
            begg=1, endg=10,
            begl=1, endl=15,
            begc=1, endc=30,
            begp=1, endp=45
        )
        
        assert result == small_domain_bounds, \
            f"Small domain bounds mismatch: expected {small_domain_bounds}, got {result}"
    
    def test_create_bounds_large_domain(self, large_domain_bounds):
        """
        Test creating bounds for a large global domain.
        
        Typical of high-resolution global simulations with thousands of elements.
        """
        result = create_bounds(
            begg=1, endg=1000,
            begl=1, endl=2500,
            begc=1, endc=5000,
            begp=1, endp=10000
        )
        
        assert result == large_domain_bounds, \
            f"Large domain bounds mismatch: expected {large_domain_bounds}, got {result}"
    
    def test_create_bounds_hierarchical_consistency(self):
        """
        Test realistic hierarchical decomposition.
        
        Verifies ~3 landunits/gridcell, ~3 columns/landunit, ~3 patches/column.
        """
        result = create_bounds(
            begg=10, endg=50,
            begl=20, endl=150,
            begc=40, endc=450,
            begp=80, endp=1350
        )
        
        # Verify gridcell range
        assert result.begg == 10 and result.endg == 50, \
            "Gridcell bounds should match input"
        
        # Verify landunit range
        assert result.begl == 20 and result.endl == 150, \
            "Landunit bounds should match input"
        
        # Verify column range
        assert result.begc == 40 and result.endc == 450, \
            "Column bounds should match input"
        
        # Verify patch range
        assert result.begp == 80 and result.endp == 1350, \
            "Patch bounds should match input"
        
        # Verify hierarchical expansion
        gridcells = result.endg - result.begg + 1
        landunits = result.endl - result.begl + 1
        columns = result.endc - result.begc + 1
        patches = result.endp - result.begp + 1
        
        assert landunits > gridcells, \
            "Should have more landunits than gridcells"
        assert columns > landunits, \
            "Should have more columns than landunits"
        assert patches > columns, \
            "Should have more patches than columns"


# ============================================================================
# create_bounds Tests - Edge Cases
# ============================================================================

class TestCreateBoundsEdgeCases:
    """Tests for create_bounds function with edge cases."""
    
    def test_create_bounds_equal_begin_end(self):
        """
        Test boundary case where begin equals end at each level.
        
        Each level represents a single element range.
        """
        result = create_bounds(
            begg=5, endg=5,
            begl=12, endl=12,
            begc=25, endc=25,
            begp=50, endp=50
        )
        
        assert result.begg == result.endg == 5, \
            "Gridcell begin should equal end"
        assert result.begl == result.endl == 12, \
            "Landunit begin should equal end"
        assert result.begc == result.endc == 25, \
            "Column begin should equal end"
        assert result.begp == result.endp == 50, \
            "Patch begin should equal end"
    
    def test_create_bounds_maximum_reasonable_indices(self):
        """
        Test very large but reasonable domain sizes.
        
        Represents high-resolution global simulations with millions of elements.
        """
        result = create_bounds(
            begg=1, endg=100000,
            begl=1, endl=500000,
            begc=1, endc=1000000,
            begp=1, endp=2000000
        )
        
        assert result.endg == 100000, "Should handle 100k gridcells"
        assert result.endl == 500000, "Should handle 500k landunits"
        assert result.endc == 1000000, "Should handle 1M columns"
        assert result.endp == 2000000, "Should handle 2M patches"
    
    def test_create_bounds_subdomain_offset(self, subdomain_bounds):
        """
        Test subdomain with non-unit starting indices.
        
        Simulates MPI domain decomposition where this processor handles
        a subset of the global domain.
        """
        result = create_bounds(
            begg=100, endg=200,
            begl=250, endl=500,
            begc=600, endc=1200,
            begp=1500, endp=3000
        )
        
        assert result == subdomain_bounds, \
            f"Subdomain bounds mismatch: expected {subdomain_bounds}, got {result}"
        
        # Verify non-unit starting indices
        assert result.begg > 1, "Gridcell should start above 1"
        assert result.begl > 1, "Landunit should start above 1"
        assert result.begc > 1, "Column should start above 1"
        assert result.begp > 1, "Patch should start above 1"
    
    def test_create_bounds_minimum_valid_indices(self):
        """
        Test with minimum valid indices (all set to 1).
        
        Verifies that the minimum constraint (indices >= 1) is respected.
        """
        result = create_bounds(
            begg=1, endg=1,
            begl=1, endl=1,
            begc=1, endc=1,
            begp=1, endp=1
        )
        
        # All indices should be 1
        for field in result._fields:
            assert getattr(result, field) == 1, \
                f"Field {field} should be 1, got {getattr(result, field)}"


# ============================================================================
# create_bounds Tests - Value Validation
# ============================================================================

class TestCreateBoundsValues:
    """Tests for create_bounds function value preservation."""
    
    @pytest.mark.parametrize("begg,endg,begl,endl,begc,endc,begp,endp", [
        (1, 1, 1, 1, 1, 1, 1, 1),           # Single cell
        (1, 10, 1, 15, 1, 30, 1, 45),       # Small domain
        (1, 100, 1, 250, 1, 500, 1, 1000),  # Medium domain
        (50, 100, 125, 250, 300, 600, 700, 1400),  # Offset domain
        (1, 1000, 1, 2500, 1, 5000, 1, 10000),  # Large domain
    ])
    def test_create_bounds_preserves_all_values(
        self, begg, endg, begl, endl, begc, endc, begp, endp
    ):
        """
        Test that create_bounds preserves all input values exactly.
        
        Parametrized test covering various domain sizes and configurations.
        """
        result = create_bounds(
            begg=begg, endg=endg,
            begl=begl, endl=endl,
            begc=begc, endc=endc,
            begp=begp, endp=endp
        )
        
        assert result.begg == begg, f"begg mismatch: expected {begg}, got {result.begg}"
        assert result.endg == endg, f"endg mismatch: expected {endg}, got {result.endg}"
        assert result.begl == begl, f"begl mismatch: expected {begl}, got {result.begl}"
        assert result.endl == endl, f"endl mismatch: expected {endl}, got {result.endl}"
        assert result.begc == begc, f"begc mismatch: expected {begc}, got {result.begc}"
        assert result.endc == endc, f"endc mismatch: expected {endc}, got {result.endc}"
        assert result.begp == begp, f"begp mismatch: expected {begp}, got {result.begp}"
        assert result.endp == endp, f"endp mismatch: expected {endp}, got {result.endp}"


# ============================================================================
# create_bounds Tests - Data Types
# ============================================================================

class TestCreateBoundsDtypes:
    """Tests for create_bounds function data type handling."""
    
    def test_create_bounds_returns_correct_type(self):
        """Verify create_bounds returns BoundsType instance."""
        result = create_bounds(1, 1, 1, 1, 1, 1, 1, 1)
        assert isinstance(result, BoundsType), \
            f"Should return BoundsType, got {type(result)}"
    
    def test_create_bounds_field_types(self):
        """Verify all fields in returned BoundsType are integers."""
        result = create_bounds(1, 10, 1, 15, 1, 30, 1, 45)
        
        for field in result._fields:
            value = getattr(result, field)
            assert isinstance(value, int), \
                f"Field {field} should be int, got {type(value)}"
    
    def test_create_bounds_accepts_int_inputs(self):
        """Verify create_bounds accepts integer inputs."""
        # Should not raise any exceptions
        result = create_bounds(
            begg=int(1), endg=int(10),
            begl=int(1), endl=int(15),
            begc=int(1), endc=int(30),
            begp=int(1), endp=int(45)
        )
        assert isinstance(result, BoundsType), "Should accept explicit int types"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_get_clump_bounds_matches_create_bounds_single_cell(self):
        """
        Test that get_clump_bounds output matches create_bounds for single cell.
        
        Both functions should produce identical results for the single cell case.
        """
        clump_result = get_clump_bounds(0)
        create_result = create_bounds(1, 1, 1, 1, 1, 1, 1, 1)
        
        assert clump_result == create_result, \
            "get_clump_bounds and create_bounds should produce identical single cell bounds"
    
    def test_bounds_equality_comparison(self):
        """Test that BoundsType instances can be compared for equality."""
        bounds1 = create_bounds(1, 10, 1, 15, 1, 30, 1, 45)
        bounds2 = create_bounds(1, 10, 1, 15, 1, 30, 1, 45)
        bounds3 = create_bounds(1, 10, 1, 15, 1, 30, 1, 46)
        
        assert bounds1 == bounds2, "Identical bounds should be equal"
        assert bounds1 != bounds3, "Different bounds should not be equal"
    
    def test_bounds_can_be_used_in_collections(self):
        """Test that BoundsType instances can be used in sets and dicts."""
        bounds1 = create_bounds(1, 10, 1, 15, 1, 30, 1, 45)
        bounds2 = create_bounds(1, 10, 1, 15, 1, 30, 1, 45)
        bounds3 = create_bounds(1, 20, 1, 30, 1, 60, 1, 90)
        
        # Test in set (requires hashability)
        bounds_set = {bounds1, bounds2, bounds3}
        assert len(bounds_set) == 2, "Duplicate bounds should not create separate set entries"
        
        # Test as dict key
        bounds_dict = {bounds1: "domain1", bounds3: "domain2"}
        assert bounds_dict[bounds2] == "domain1", "Should be able to use as dict key"


# ============================================================================
# Constraint Validation Tests
# ============================================================================

class TestConstraintValidation:
    """Tests for physical constraint validation."""
    
    def test_begin_less_than_or_equal_end_gridcells(self):
        """Verify that begin <= end for gridcells in typical usage."""
        result = create_bounds(1, 100, 1, 200, 1, 400, 1, 800)
        assert result.begg <= result.endg, \
            "Beginning gridcell index should be <= ending gridcell index"
    
    def test_begin_less_than_or_equal_end_landunits(self):
        """Verify that begin <= end for landunits in typical usage."""
        result = create_bounds(1, 100, 1, 200, 1, 400, 1, 800)
        assert result.begl <= result.endl, \
            "Beginning landunit index should be <= ending landunit index"
    
    def test_begin_less_than_or_equal_end_columns(self):
        """Verify that begin <= end for columns in typical usage."""
        result = create_bounds(1, 100, 1, 200, 1, 400, 1, 800)
        assert result.begc <= result.endc, \
            "Beginning column index should be <= ending column index"
    
    def test_begin_less_than_or_equal_end_patches(self):
        """Verify that begin <= end for patches in typical usage."""
        result = create_bounds(1, 100, 1, 200, 1, 400, 1, 800)
        assert result.begp <= result.endp, \
            "Beginning patch index should be <= ending patch index"
    
    def test_all_indices_positive(self):
        """Verify all indices are positive (>= 1) in typical usage."""
        result = create_bounds(1, 100, 1, 200, 1, 400, 1, 800)
        
        for field in result._fields:
            value = getattr(result, field)
            assert value >= 1, \
                f"Field {field} should be >= 1 (Fortran convention), got {value}"


# ============================================================================
# Documentation and Metadata Tests
# ============================================================================

class TestDocumentation:
    """Tests for function documentation and metadata."""
    
    def test_get_clump_bounds_has_docstring(self):
        """Verify get_clump_bounds has documentation."""
        assert get_clump_bounds.__doc__ is not None, \
            "get_clump_bounds should have a docstring"
    
    def test_create_bounds_has_docstring(self):
        """Verify create_bounds has documentation."""
        assert create_bounds.__doc__ is not None, \
            "create_bounds should have a docstring"
    
    def test_boundstype_has_docstring(self):
        """Verify BoundsType has documentation."""
        assert BoundsType.__doc__ is not None, \
            "BoundsType should have a docstring"


# ============================================================================
# Test Summary
# ============================================================================

def test_suite_summary():
    """
    Summary of test coverage.
    
    This test suite provides comprehensive coverage of the decompMod module:
    
    1. BoundsType Structure (5 tests):
       - Namedtuple validation
       - Field access and immutability
    
    2. get_clump_bounds (5 tests):
       - Default behavior with various processor indices
       - Edge cases (negative indices)
       - Consistency and return type validation
    
    3. create_bounds Nominal Cases (4 tests):
       - Single cell, small, large, and hierarchical domains
    
    4. create_bounds Edge Cases (4 tests):
       - Equal begin/end, maximum indices, subdomains, minimum indices
    
    5. create_bounds Value Validation (5 parametrized tests):
       - Value preservation across various configurations
    
    6. create_bounds Data Types (3 tests):
       - Return type and field type validation
    
    7. Integration Tests (3 tests):
       - Function interoperability and collection usage
    
    8. Constraint Validation (5 tests):
       - Physical constraint verification (begin <= end, positive indices)
    
    9. Documentation Tests (3 tests):
       - Docstring presence validation
    
    Total: 37+ individual test cases covering nominal, edge, and special cases.
    """
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])