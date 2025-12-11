# JSON-Based Fortran to JAX Translation

## Quick Start

The translator uses static analysis JSON files and processes translation units **iteratively** for better translations.

### Setup

JSON files should be at:
```
jax-agents/
  static_analysis_output/
    analysis_results.json      # From Fortran-Analyzer
    translation_units.json     # From Fortran-Analyzer
```

Fortran source files at:
```
CLM-ml_v1/                     # Your Fortran code location
  clm_src_biogeophys/
  clm_src_main/
  ...
```

### Usage

```python
from pathlib import Path
from jax_agents.translator import TranslatorAgent

translator = TranslatorAgent(
    analysis_results_path=Path("static_analysis_output/analysis_results.json"),
    translation_units_path=Path("static_analysis_output/translation_units.json"),
    jax_ctsm_dir=Path("../jax-ctsm"),
    fortran_root=Path("../CLM-ml_v1"),  # Path to Fortran source files
)

# Translate by module name (file path automatic)
result = translator.translate_module(
    module_name="SoilStateType",
    output_dir=Path("translated_modules/SoilStateType")
)
```

### Test It

```bash
# 1. Verify setup
python examples/verify_json_integration.py

# 2. Translate samples (iterative approach)
python examples/translate_with_json.py

# 3. Batch translate
python examples/batch_translate_modules.py
```

See [TESTING.md](TESTING.md) for detailed testing instructions.

## What Changed

**Before**: Manual file paths, limited context
```python
translator.translate_module(
    fortran_file=Path("SoilStateType.F90"),
    analysis=manual_analysis,
)
```

**After**: Module name, rich context from JSON
```python
translator.translate_module(
    module_name="SoilStateType",  # Automatic path resolution
)
```

## Translation Approach

**Iterative Unit-by-Unit Processing**:
1. Each translation unit translated separately (N LLM calls)
2. Each unit sees previously translated units for context
3. All units assembled into final module (1 assembly LLM call)
4. Total: N+1 LLM calls per module

**Benefits**:
- Smaller, focused prompts per unit
- Better for large/complex modules
- Previous context available to subsequent units
- Clearer debugging (identify problematic units)

## Benefits

- **Dependency aware**: Knows what modules depend on each other
- **Complexity guided**: LLM sees difficulty scores and effort estimates  
- **Line precise**: References exact Fortran source lines
- **Iterative translation**: Unit-by-unit with context accumulation
- **Batch ready**: Translate entire project systematically

## JSON Structure

**analysis_results.json**: Module metadata, dependencies, entities
**translation_units.json**: 120 units with complexity scores, split functions

## Complexity Guide

- **Low** (< 5): ~5-10 min - Simple modules like `clm_varctl`
- **Medium** (5-10): ~15-30 min - Data structures like `SoilStateType`
- **High** (≥ 10): ~45+ min - Physics like `SoilTemperatureMod`

## Output

Each module generates:
```
translated_modules/
  SoilStateType/
    SoilStateType.py                    # Main physics
    SoilStateType_params.py             # Parameters (if needed)
    test_SoilStateType.py               # Tests (if generated)
    SoilStateType_translation_notes.md  # Translation notes
```

## Agents Overview

This system includes multiple specialized agents:

### 1. **Static Analysis Agent**
Analyzes Fortran code structure, dependencies, and complexity.

### 2. **Translator Agent**
Converts Fortran code to JAX with full context awareness.

### 3. **Test Agent**
Generates comprehensive test suites for translated code:
- Analyzes Python function signatures
- Creates synthetic test data with edge cases
- Generates pytest files
- Provides test documentation

```python
from jax_agents import TestAgent

test_agent = TestAgent()
result = test_agent.generate_tests(
    module_name="SoilTemperatureMod",
    python_code=translated_code,
    output_dir=Path("tests"),
)
```

### 4. **Repair Agent** ✨ NEW
Automatically debugs and fixes failed translations:
- Analyzes test failures and error messages
- Compares with original Fortran to identify root causes
- Generates corrected Python code
- Runs tests iteratively until they pass
- Creates comprehensive root cause analysis reports

```python
from jax_agents import RepairAgent

repair_agent = RepairAgent(max_repair_iterations=5)
result = repair_agent.repair_translation(
    module_name="SoilTemperatureMod",
    fortran_code=original_fortran,
    failed_python_code=failed_translation,
    test_report=pytest_output,
    test_file_path=Path("tests/test_module.py"),
    output_dir=Path("repair_outputs"),
)
```

**Repair Agent Features:**
- 🔍 Root cause identification
- 🔧 Automatic code fixing
- 🔄 Iterative refinement
- ✅ Test verification
- 📝 Detailed RCA reports

### 5. **Orchestrator Agent**
Manages the entire conversion pipeline automatically.

## Complete Translation Pipeline

```bash
# 1. Translate module
python examples/translate_with_context.py

# 2. Generate tests
python examples/generate_tests.py

# 3. Run tests
pytest output/test_ModuleName.py -v

# 4. If tests fail, repair automatically
python examples/repair_agent_example.py

# 5. Verify fix
pytest output/test_ModuleName.py -v
```

## Files Modified

- `src/jax_agents/translator.py` - Enhanced with JSON support
- `src/jax_agents/prompts/translation_prompts.py` - Enhanced prompt
- `src/jax_agents/repair_agent.py` - **NEW**: Automatic bug fixing
- `src/jax_agents/prompts/repair_prompts.py` - **NEW**: Repair prompts
- `examples/repair_agent_example.py` - **NEW**: Repair example
- `docs/repair_agent.md` - **NEW**: Comprehensive documentation
- `examples/` - New verification and batch translation scripts
