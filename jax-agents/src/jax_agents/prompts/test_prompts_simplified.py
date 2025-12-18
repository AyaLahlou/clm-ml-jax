"""
Simplified Test Prompts for Python/JAX test generation only.

These prompts focus on generating comprehensive tests for Python/JAX code
without requiring Fortran validation.
"""

TEST_PROMPTS = {
    "system": """You are an expert Python/JAX Testing Agent specializing in scientific computing.

Your expertise includes:
- Python, JAX, and NumPy testing best practices
- pytest framework and fixtures
- Parametrized testing and property-based testing
- Scientific computing edge cases and numerical stability
- Test data generation for physics simulations

Your responsibilities:
1. Analyze Python/JAX function signatures
2. Generate comprehensive test data covering:
   - Nominal/typical cases
   - Edge cases (zeros, negatives, boundaries, NaN/Inf)
   - Array dimension variations
   - Physical realism (temps > 0K, fractions in [0,1])
3. Create pytest files with:
   - Fixtures for test data
   - Parametrized tests
   - Clear assertions with good error messages
   - Docstrings explaining test purpose
4. Generate test documentation

You follow pytest best practices and write clear, maintainable tests.""",

    "analyze_python_signature": """Analyze the following Python/JAX function to extract its complete signature.

Module: {module_name}

Python Code:
```python
{python_code}
```

Extract:
1. Function name
2. All parameters with:
   - Name
   - Type hint (jnp.ndarray, float, NamedTuple, etc.)
   - Array shape if applicable
   - Default values
   - Description from docstring
3. Return type and structure
4. Any NamedTuple or dataclass definitions used
5. Physical constraints (e.g., temperature > 0, fractions in [0,1])

Return as JSON:
```json
{{
  "name": "function_name",
  "parameters": [
    {{
      "name": "param_name",
      "python_type": "jnp.ndarray",
      "shape": "(n_columns, n_levels)" or null,
      "default": null or value,
      "description": "parameter description",
      "constraints": {{"min": 0, "max": 1}} or null
    }}
  ],
  "returns": {{
    "type": "jnp.ndarray" or "Tuple" or "NamedTuple",
    "description": "return value description",
    "components": ["field1", "field2"] or null
  }},
  "namedtuples": [
    {{
      "name": "ResultType",
      "fields": ["field1", "field2"]
    }}
  ]
}}
```""",

    "generate_test_data": """Generate comprehensive synthetic test data for a Python/JAX function.

Python Signature:
```json
{python_signature}
```

Number of test cases: {num_cases}
Include edge cases: {include_edge_cases}

Generate {num_cases} diverse test cases:

**Test Types:**
1. **Nominal cases** (50%): Typical operating conditions
2. **Edge cases** (30%): If include_edge_cases is true
   - Zero values
   - Negative values (where physically valid)
   - Boundary conditions
   - Very small/large magnitudes
3. **Special cases** (20%):
   - Different array sizes/dimensions
   - Extreme but valid physical conditions

**Requirements:**
- Physically realistic (temps in Kelvin > 0, etc.)
- Consistent dimensions within each test
- Cover parameter space thoroughly
- Include descriptive metadata

Return as JSON:
```json
{{
  "function_name": "function_name",
  "test_cases": [
    {{
      "name": "test_nominal_conditions",
      "inputs": {{
        "param1": value_or_array,
        "param2": value_or_array
      }},
      "metadata": {{
        "type": "nominal" or "edge" or "special",
        "description": "what this tests",
        "edge_cases": ["zero_flux", "boundary"] or []
      }}
    }}
  ],
  "notes": "How data was generated and assumptions made"
}}
```

For arrays, use nested lists. Ensure all dimensions match within each test case.""",

    "generate_pytest": """Generate a comprehensive pytest file for the Python/JAX function.

Module: {module_name}
Source Directory: {source_directory}

Python Signature:
```json
{python_signature}
```

Test Data:
```json
{test_data}
```

Include Performance Tests: {include_performance}

Create a pytest file with:

1. **Imports**: 
   - Add the src directory to sys.path: `sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))`
   - Import the actual function from {source_directory}.{module_name}
   - For example: `from {source_directory}.{module_name} import function_name`
   - IMPORTANT: Do NOT create mock functions. Import the real translated functions.
   - pytest, numpy/jax, etc.
2. **Fixtures**: 
   - `test_data()` fixture loading test data
   - Any other needed setup
3. **Parametrized Tests**:
   - Use `@pytest.mark.parametrize` for test cases
   - Clear test names
4. **Test Functions**:
   - `test_<function>_shapes()`: Verify output shapes
   - `test_<function>_values()`: Verify output values
   - `test_<function>_edge_cases()`: Test edge conditions
   - `test_<function>_dtypes()`: Verify data types
   - `test_<function>_performance()`: If include_performance is true
5. **Assertions**:
   - Use `np.allclose` for arrays (atol=1e-6, rtol=1e-6)
   - Check shapes with `assert array.shape == expected_shape`
   - Good error messages
6. **Docstrings**: Explain what each test does

CRITICAL: The tests must import and use the actual translated functions from {source_directory}.{module_name}, not mock implementations. Always add 'src' to sys.path first.

Return only the complete Python pytest code.""",

    "generate_documentation": """Generate test documentation for the module.

Module: {module_name}

Python Signature:
```json
{python_signature}
```

Test Data Summary:
```json
{test_data_summary}
```

Create a markdown documentation file explaining:

## Test Suite Overview
- What functions are tested
- Number and types of test cases
- Coverage areas

## Running the Tests
```bash
# Basic run
pytest test_{module_name}.py

# With coverage
pytest test_{module_name}.py --cov={module_name}

# Verbose
pytest test_{module_name}.py -v
```

## Test Cases
Brief description of each test type:
- Nominal cases: ...
- Edge cases: ...
- Special cases: ...

## Test Data
How test data was generated and what it covers

## Expected Behavior
What should pass/fail and why

## Extending Tests
How to add new test cases

## Common Issues
Potential problems and solutions

Return complete markdown documentation.""",
}

