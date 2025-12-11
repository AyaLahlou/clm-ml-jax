"""Condensed prompt templates for Translator Agent."""

TRANSLATION_PROMPTS = {
    "system": """You are a Fortran-to-JAX translator specializing in CTSM code.

Core Principles:
- Pure functions, immutable NamedTuples, no side effects
- JIT-compatible: use jnp.where not Python if, vectorize loops
- Preserve exact physics equations
- Full type hints and Google-style docstrings
- Reference Fortran source line numbers""",

    "translate_module": """Translate Fortran module to JAX.

MODULE: {module_name}

FORTRAN CODE:
```fortran
{fortran_code}
```

ANALYSIS:
```json
{module_info}
```

CONTEXT (dependencies, translation units, complexity):
```json
{enhanced_context}
```

REFERENCE PATTERN:
```python
{reference_pattern}
```

REQUIREMENTS:
1. **Structure**: NamedTuples for data, separate params file if needed
2. **Functions**: Pure with type hints, preserve physics exactly
3. **Arrays**: Vectorized ops (no Python loops), document shapes # [n_patches, n_layers]
4. **Conditionals**: jnp.where for JIT compatibility
5. **Docs**: Reference Fortran line numbers from translation units, explain translations

Translation units guide:
- "module": header/declarations
- "root": complete function
- "inner": part of split function (note parent)
- Use line_start/line_end for references
- High complexity_score = careful vectorization

Output:
1. Main physics module
2. Parameters file (if needed)
3. Translation notes""",

    "translate_function": """Translate Fortran subroutine to JAX function.

FORTRAN:
```fortran
{fortran_code}
```

CONTEXT:
```json
{context}
```

Requirements: Pure function, type hints, docstring with Fortran reference, preserve physics, JIT-compatible.""",

    "convert_data_structure": """Convert Fortran type to JAX NamedTuple.

FORTRAN:
```fortran
{fortran_type}
```

Requirements: NamedTuple, map types to jnp.ndarray, document shapes, field descriptions.""",

    "vectorize_loop": """Vectorize Fortran loop to JAX.

FORTRAN:
```fortran
{loop_code}
```

ANALYSIS: {loop_analysis}

Requirements: Eliminate loop, use jnp operations/vmap, preserve computation order.""",

    "handle_conditional": """Convert Fortran conditional to JIT-compatible JAX.

FORTRAN:
```fortran
{conditional_code}
```

Requirements: jnp.where for arrays, preserve logic, JIT-compatible.""",

    "create_parameters": """Create JAX parameter class.

FORTRAN:
```fortran
{parameters}
```

Requirements: NamedTuple, default values, document sources.""",

    "translate_unit": """Translate this translation unit to JAX.

MODULE: {module_name}
UNIT: {unit_id} ({unit_type})
LINES: {line_start}-{line_end}

FORTRAN CODE:
```fortran
{fortran_code}
```

UNIT INFO:
```json
{unit_info}
```

CONTEXT (module dependencies, previously translated units):
```json
{context}
```

REFERENCE PATTERN:
```python
{reference_pattern}
```

REQUIREMENTS:
- Pure functions with type hints
- Preserve physics exactly (lines {line_start}-{line_end} from original)
- Vectorize loops, use jnp.where for conditionals
- If unit_type is "inner", this is part of parent: {parent_id}
- Document with Fortran line reference

Output ONLY the translated code for this unit.""",

    "assemble_module": """Assemble complete JAX module from translated units.

MODULE: {module_name}

TRANSLATED UNITS:
```json
{translated_units}
```

MODULE INFO:
```json
{module_info}
```

REFERENCE PATTERN:
```python
{reference_pattern}
```

REQUIREMENTS:
1. Combine all units into cohesive module
2. Add imports (jax, jax.numpy as jnp, typing, NamedTuple)
3. Organize: imports → types → params → functions
4. Ensure consistency across units
5. Generate params file if needed
6. Add module-level docstring

Output:
1. Main physics module (complete, executable)
2. Parameters file (if needed)
3. Brief assembly notes""",
}

