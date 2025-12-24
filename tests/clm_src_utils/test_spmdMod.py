"""
Comprehensive pytest suite for spmdMod.is_master_proc function.

This module tests the is_master_proc() function which serves as a compatibility
layer for SPMD (Single Program Multiple Data) patterns in JAX. In single-process
mode, it always returns True. Tests verify behavior across different JAX execution
contexts and ensure consistency with the MASTERPROC constant.

Test Categories:
- Nominal cases: Default behavior and repeated calls
- JAX context tests: jit, vmap, pmap execution
- Edge cases: Type validation and constant alignment
- Concurrent execution: Thread-safe behavior
"""

import sys
from pathlib import Path
from typing import Any, Dict, List
import threading

import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from clm_src_utils.spmdMod import is_master_proc, MASTERPROC


@pytest.fixture
def test_data() -> Dict[str, Any]:
    """
    Fixture providing test data for is_master_proc tests.
    
    Returns:
        Dictionary containing test cases with inputs, expected outputs,
        and metadata for various execution contexts.
    """
    return {
        "test_cases": [
            {
                "name": "test_single_process_default",
                "inputs": {},
                "expected_output": True,
                "metadata": {
                    "type": "nominal",
                    "description": "Tests default behavior in single-process JAX execution"
                }
            },
            {
                "name": "test_repeated_calls_consistency",
                "inputs": {},
                "expected_output": True,
                "metadata": {
                    "type": "nominal",
                    "call_count": 5
                }
            },
            {
                "name": "test_within_jit_compilation",
                "inputs": {},
                "expected_output": True,
                "metadata": {
                    "type": "nominal",
                    "jit_context": True
                }
            },
            {
                "name": "test_within_vmap_context",
                "inputs": {},
                "expected_output": True,
                "metadata": {
                    "type": "special",
                    "vmap_context": True,
                    "vmap_size": 4
                }
            },
            {
                "name": "test_within_pmap_single_device",
                "inputs": {},
                "expected_output": True,
                "metadata": {
                    "type": "special",
                    "pmap_context": True,
                    "device_count": 1
                }
            },
            {
                "name": "test_type_consistency",
                "inputs": {},
                "expected_output": True,
                "metadata": {
                    "type": "edge",
                    "strict_type_check": True
                }
            },
            {
                "name": "test_constant_alignment",
                "inputs": {},
                "expected_output": True,
                "metadata": {
                    "type": "edge",
                    "validate_against_constant": "MASTERPROC"
                }
            }
        ]
    }


class TestIsMasterProcBasic:
    """Basic functionality tests for is_master_proc."""
    
    def test_single_process_default(self, test_data):
        """
        Test default behavior in single-process JAX execution.
        
        Verifies that is_master_proc returns True in the standard
        single-process execution mode, matching the MASTERPROC constant.
        """
        result = is_master_proc()
        
        assert isinstance(result, bool), \
            f"Expected bool type, got {type(result)}"
        assert result is True, \
            "is_master_proc should return True in single-process mode"
    
    def test_repeated_calls_consistency(self, test_data):
        """
        Test that multiple calls return consistent results.
        
        Verifies that is_master_proc returns the same value across
        multiple invocations, ensuring deterministic behavior.
        """
        call_count = 5
        results = [is_master_proc() for _ in range(call_count)]
        
        assert all(r is True for r in results), \
            f"All {call_count} calls should return True, got {results}"
        assert len(set(results)) == 1, \
            "All calls should return identical values"
    
    def test_cold_start_initialization(self):
        """
        Test first call after module import (cold start scenario).
        
        Verifies that the function works correctly on first invocation
        without requiring any initialization.
        """
        # This test runs early in the suite to simulate cold start
        result = is_master_proc()
        
        assert result is True, \
            "Cold start call should return True"
        assert isinstance(result, bool), \
            "Cold start should return proper bool type"


class TestIsMasterProcJAXContexts:
    """Tests for is_master_proc behavior in various JAX execution contexts."""
    
    def test_within_jit_compilation(self, test_data):
        """
        Test behavior when called within a JAX jit-compiled function.
        
        Verifies that is_master_proc works correctly inside JIT-compiled
        code, which is a common pattern in JAX applications.
        """
        @jax.jit
        def jitted_check():
            return is_master_proc()
        
        result = jitted_check()
        
        assert isinstance(result, (bool, np.bool_, jnp.bool_)), \
            f"JIT result should be bool-like, got {type(result)}"
        assert bool(result) is True, \
            "is_master_proc should return True within JIT context"
    
    def test_within_vmap_context(self, test_data):
        """
        Test behavior when called within a JAX vmap (vectorized map) context.
        
        Verifies that is_master_proc returns consistent values when
        called within vectorized operations.
        """
        vmap_size = 4
        
        def vmapped_check(x):
            # x is just a dummy input for vmap
            return is_master_proc()
        
        # Create dummy input array for vmap
        dummy_input = jnp.arange(vmap_size)
        
        # Apply vmap
        vmapped_fn = jax.vmap(vmapped_check)
        results = vmapped_fn(dummy_input)
        
        # All results should be True
        assert results.shape == (vmap_size,), \
            f"Expected shape ({vmap_size},), got {results.shape}"
        assert jnp.all(results), \
            "All vmapped calls should return True"
    
    def test_within_pmap_single_device(self, test_data):
        """
        Test behavior within pmap context on single device.
        
        Verifies that is_master_proc works correctly in pmap context
        when only one device is available.
        """
        device_count = jax.device_count()
        
        if device_count < 1:
            pytest.skip("No JAX devices available")
        
        def pmapped_check(x):
            return is_master_proc()
        
        # Create input for pmap (one per device)
        dummy_input = jnp.arange(min(device_count, 1))
        
        try:
            pmapped_fn = jax.pmap(pmapped_check)
            results = pmapped_fn(dummy_input.reshape(-1, 1))
            
            # In single-process mode, should return True
            assert jnp.all(results), \
                "pmap calls should return True in single-process mode"
        except Exception as e:
            # pmap might not be available in all environments
            pytest.skip(f"pmap not available: {e}")
    
    def test_after_device_operations(self, test_data):
        """
        Test that is_master_proc returns correct value after JAX device operations.
        
        Verifies that prior JAX computations don't affect the return value
        of is_master_proc.
        """
        # Perform some device operations
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x ** 2)
        z = jax.jit(lambda a: a * 2)(y)
        
        # Now check is_master_proc
        result = is_master_proc()
        
        assert result is True, \
            "is_master_proc should return True after device operations"


class TestIsMasterProcEdgeCases:
    """Edge case tests for is_master_proc."""
    
    def test_type_consistency(self, test_data):
        """
        Verify return type is exactly bool (not int or other truthy value).
        
        Ensures strict type checking - the function must return a proper
        Python bool, not just a truthy value.
        """
        result = is_master_proc()
        
        # Check exact type (not just isinstance)
        assert type(result) is bool or isinstance(result, (np.bool_, jnp.bool_)), \
            f"Expected bool type, got {type(result).__name__}"
        
        # Verify it's not an integer masquerading as bool
        assert not isinstance(result, (int, np.integer)) or isinstance(result, (bool, np.bool_)), \
            "Result should be bool, not integer"
        
        # Verify boolean operations work correctly
        assert (result and True) is True, \
            "Boolean AND operation should work correctly"
        assert (result or False) is True, \
            "Boolean OR operation should work correctly"
    
    def test_constant_alignment(self, test_data):
        """
        Verify function return matches MASTERPROC constant value.
        
        Ensures that is_master_proc() returns the same value as the
        MASTERPROC constant, maintaining consistency in the module.
        """
        result = is_master_proc()
        
        assert result == MASTERPROC, \
            f"is_master_proc() should match MASTERPROC constant: {MASTERPROC}"
        assert type(result) == type(MASTERPROC), \
            f"Types should match: {type(result)} vs {type(MASTERPROC)}"
    
    def test_no_side_effects(self):
        """
        Verify that calling is_master_proc has no side effects.
        
        Ensures the function is pure and doesn't modify any global state.
        """
        # Get initial state
        initial_result = is_master_proc()
        
        # Call multiple times
        for _ in range(10):
            is_master_proc()
        
        # Verify state unchanged
        final_result = is_master_proc()
        
        assert initial_result == final_result, \
            "Multiple calls should not change the return value"
    
    def test_hashable_return(self):
        """
        Verify that the return value is hashable.
        
        Ensures the return value can be used in sets, as dict keys, etc.
        """
        result = is_master_proc()
        
        # Should be able to hash it
        try:
            hash_value = hash(result)
            assert isinstance(hash_value, int), \
                "Hash should return an integer"
        except TypeError as e:
            pytest.fail(f"Return value should be hashable: {e}")
        
        # Should be able to use in a set
        result_set = {result, result, True}
        assert len(result_set) <= 2, \
            "Should be usable in sets"


class TestIsMasterProcConcurrency:
    """Concurrency and thread-safety tests for is_master_proc."""
    
    def test_concurrent_calls(self, test_data):
        """
        Test behavior with concurrent/parallel calls in single-process mode.
        
        Verifies that is_master_proc is thread-safe and returns consistent
        results when called from multiple threads simultaneously.
        """
        thread_count = 3
        results = []
        errors = []
        
        def thread_worker():
            try:
                result = is_master_proc()
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create and start threads
        threads = [threading.Thread(target=thread_worker) for _ in range(thread_count)]
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check for errors
        assert len(errors) == 0, \
            f"Threads encountered errors: {errors}"
        
        # Verify all results are True
        assert len(results) == thread_count, \
            f"Expected {thread_count} results, got {len(results)}"
        assert all(r is True for r in results), \
            f"All concurrent calls should return True, got {results}"
    
    def test_rapid_sequential_calls(self):
        """
        Test rapid sequential calls to verify no race conditions.
        
        Ensures that calling is_master_proc in rapid succession
        doesn't cause any issues or inconsistencies.
        """
        call_count = 1000
        results = [is_master_proc() for _ in range(call_count)]
        
        assert len(results) == call_count, \
            f"Expected {call_count} results"
        assert all(r is True for r in results), \
            "All rapid sequential calls should return True"
        assert len(set(results)) == 1, \
            "All results should be identical"


class TestIsMasterProcIntegration:
    """Integration tests combining multiple aspects of is_master_proc."""
    
    def test_mixed_context_usage(self):
        """
        Test using is_master_proc in mixed JAX contexts.
        
        Verifies correct behavior when the function is called in
        various combinations of JAX transformations.
        """
        # Regular call
        result1 = is_master_proc()
        
        # JIT call
        @jax.jit
        def jitted():
            return is_master_proc()
        result2 = jitted()
        
        # Nested in computation
        @jax.jit
        def complex_computation():
            x = jnp.array([1.0, 2.0])
            y = jnp.sum(x)
            is_master = is_master_proc()
            return is_master, y
        result3, _ = complex_computation()
        
        # All should be True
        assert bool(result1) is True, "Regular call should return True"
        assert bool(result2) is True, "JIT call should return True"
        assert bool(result3) is True, "Nested call should return True"
    
    def test_conditional_logic_usage(self):
        """
        Test using is_master_proc in conditional logic.
        
        Verifies that the return value works correctly in if statements
        and other conditional contexts, which is its primary use case.
        """
        # Direct conditional
        if is_master_proc():
            executed = True
        else:
            executed = False
        
        assert executed is True, \
            "Conditional should execute True branch"
        
        # Ternary operator
        value = "master" if is_master_proc() else "worker"
        assert value == "master", \
            "Ternary should select master branch"
        
        # Boolean logic
        result = is_master_proc() and True
        assert result is True, \
            "Boolean AND should work correctly"
        
        result = is_master_proc() or False
        assert result is True, \
            "Boolean OR should work correctly"
    
    def test_function_composition(self):
        """
        Test composing is_master_proc with other functions.
        
        Verifies that the function works correctly when composed
        with other operations in a functional programming style.
        """
        # Compose with lambda
        check_and_return = lambda: (is_master_proc(), "checked")
        result, msg = check_and_return()
        
        assert result is True, "Composed function should return True"
        assert msg == "checked", "Composed function should return both values"
        
        # Use in map
        results = list(map(lambda _: is_master_proc(), range(3)))
        assert all(r is True for r in results), \
            "Mapped calls should all return True"
        
        # Use in filter (though unusual for this function)
        filtered = list(filter(lambda x: is_master_proc(), [1, 2, 3]))
        assert len(filtered) == 3, \
            "Filter with is_master_proc should pass all items"


@pytest.mark.parametrize("test_case", [
    {"name": "call_1", "expected": True},
    {"name": "call_2", "expected": True},
    {"name": "call_3", "expected": True},
    {"name": "call_4", "expected": True},
    {"name": "call_5", "expected": True},
])
def test_is_master_proc_parametrized(test_case):
    """
    Parametrized test for is_master_proc consistency.
    
    Tests that is_master_proc returns True consistently across
    multiple parametrized test cases.
    
    Args:
        test_case: Dictionary with test name and expected output
    """
    result = is_master_proc()
    
    assert result == test_case["expected"], \
        f"Test {test_case['name']}: expected {test_case['expected']}, got {result}"
    assert isinstance(result, bool), \
        f"Test {test_case['name']}: result should be bool type"


def test_is_master_proc_documentation():
    """
    Test that is_master_proc has proper documentation.
    
    Verifies that the function has a docstring and is properly
    documented for users.
    """
    assert is_master_proc.__doc__ is not None, \
        "is_master_proc should have a docstring"
    
    # Function should be callable
    assert callable(is_master_proc), \
        "is_master_proc should be callable"


def test_masterproc_constant_exists():
    """
    Test that MASTERPROC constant is properly defined.
    
    Verifies that the MASTERPROC constant exists and has the
    correct value and type.
    """
    assert MASTERPROC is True, \
        "MASTERPROC constant should be True"
    assert isinstance(MASTERPROC, bool), \
        "MASTERPROC should be bool type"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])