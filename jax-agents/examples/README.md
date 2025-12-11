# JAX-CTSM Translation Examples

This directory contains examples demonstrating different workflows for converting Fortran CTSM code to JAX.

## Examples Overview

### 1. `analyze_module.py` - Static Analysis Only
**Purpose:** Analyze a Fortran module to understand its structure without translating.

```bash
python examples/analyze_module.py
```

**Output:**
- `analysis_result.json` - Complete analysis that can be reused for translation

**Use when:**
- You want to understand code structure first
- You need to review analysis before translation
- You want to cache analysis for multiple translation attempts

---

### 2. `translate_from_analysis.py` - Translation from Saved Analysis
**Purpose:** Translate a Fortran module using a pre-saved analysis file.

```bash
# First, run analysis to create analysis_result.json
python examples/analyze_module.py

# Then translate using the saved analysis
python examples/translate_from_analysis.py
```

**Workflow:**
1. Loads `analysis_result.json` 
2. Translates to JAX using the loaded analysis
3. Saves output to `./output/` directory

**Benefits:**
- **Separation of concerns:** Analysis and translation are separate steps
- **Reusability:** Run translation multiple times without re-analyzing
- **Debugging:** Easier to debug translation issues independently
- **Cost efficiency:** Don't pay for repeated analysis

---

### 3. `translate_with_context.py` - Two-Step Translation
**Purpose:** Perform analysis and translation in one script (no file saving between steps).

```bash
python examples/translate_with_context.py
```

**Workflow:**
1. Analyze the Fortran module
2. Immediately translate using the analysis
3. Save both analysis and translation

**Use when:**
- You want a streamlined workflow
- You don't need to inspect analysis first
- You want both steps in one command

---

### 4. `convert_single_module.py` - Full Orchestration
**Purpose:** Use the orchestrator to manage the entire conversion process automatically.

```bash
python examples/convert_single_module.py
```

**Features:**
- Creates a conversion plan
- Manages analysis and translation automatically
- Generates comprehensive reports
- Handles complex dependencies

**Use when:**
- You want the highest-level interface
- You need conversion reports
- You want automatic dependency management

---

### 5. `generate_tests.py` - Test Generation
**Purpose:** Generate comprehensive test suites for translated JAX code.

```bash
python examples/generate_tests.py
```

**Features:**
- Analyzes Python function signatures
- Generates synthetic test data covering edge cases
- Creates pytest files with parametrized tests
- Produces test documentation

**Use when:**
- You've translated a module and need tests
- You want comprehensive test coverage
- You need test data for verification

---

### 6. `repair_agent_example.py` - Automatic Bug Fixing
**Purpose:** Automatically debug and fix failed Python/JAX translations.

```bash
python examples/repair_agent_example.py
```

**Features:**
- Analyzes test failures and error messages
- Identifies root causes by comparing with Fortran
- Generates corrected Python code
- Runs tests iteratively until they pass
- Creates comprehensive root cause analysis reports

**Workflow:**
1. Takes failed Python code and test report
2. Analyzes what went wrong
3. Generates a fix
4. Runs tests (if pytest file provided)
5. If tests still fail, iterates with new analysis
6. Produces corrected code + root cause report

**Use when:**
- Your translation has failing tests
- You need to understand why tests failed
- You want automated debugging assistance
- You need root cause analysis documentation

---

## Recommended Workflow

### For Development/Debugging

```bash
# Step 1: Analyze
python examples/analyze_module.py

# Step 2: Review analysis_result.json manually

# Step 3: Translate (can repeat without re-analyzing)
python examples/translate_from_analysis.py
```

### For Production

```bash
# All-in-one orchestrated conversion
python examples/convert_single_module.py
```

### Complete Workflow with Testing & Repair

```bash
# Step 1: Translate the module
python examples/translate_with_context.py

# Step 2: Generate comprehensive tests
python examples/generate_tests.py

# Step 3: Run the generated tests
pytest output/test_ModuleName.py -v

# Step 4: If tests fail, use repair agent
python examples/repair_agent_example.py

# Step 5: Verify the fix
pytest output/test_ModuleName.py -v
```

This workflow ensures:
1. ✓ Proper translation from Fortran
2. ✓ Comprehensive test coverage
3. ✓ Automated bug detection
4. ✓ Automatic debugging and repair
5. ✓ Root cause analysis documentation

---

## Configuration

All examples use settings from `../config.yaml`:

```yaml
llm:
  model: "claude-sonnet-4-5"
  temperature: 0.0
  max_tokens: 50000  # High limit to avoid truncation
  timeout: 600
```

### Important Settings

- **max_tokens:** Set high (16000+) for translation to avoid code truncation
- **timeout:** Increase for large modules
- **temperature:** Keep at 0.0 for deterministic code generation

---

## Output Structure

All examples save output to `./output/` directory:

```
output/
├── ModuleName.py              # Main physics module
├── ModuleName_params.py       # Parameters (if generated)
├── test_ModuleName.py         # Tests (if generated)
└── ModuleName_translation_notes.md  # Translation notes
```

---

## Troubleshooting

### Rate Limits
If you hit API rate limits:
1. Wait 60 seconds between attempts
2. Reduce `max_tokens` in config.yaml
3. Use cached analysis with `translate_from_analysis.py`

### Truncated Output
If generated code is truncated:
1. Increase `max_tokens` in config.yaml (try 16000+)
2. The translator automatically uses minimum 16000 tokens

### JSON Parsing Errors
If analysis fails with JSON errors:
1. Check `logs/json_error_*.txt` for details
2. Increase `max_tokens` in config.yaml
3. The code now handles truncation gracefully

---

## File Locations

Configure paths in `../config.yaml`:

```yaml
paths:
  ctsm_root: "../CTSM"
  jax_ctsm_root: "../jax-ctsm"
  output_dir: "../jax-ctsm/src/jax_ctsm"
```

Or override in the script itself.

