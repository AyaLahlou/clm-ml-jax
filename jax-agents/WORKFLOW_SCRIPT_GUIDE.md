# Translation Workflow Script Guide

## Overview

The `run_translation_workflow.sh` script provides a unified interface to run the complete JAX-CTSM translation workflow:

1. **Translate** - Convert Fortran modules to JAX Python
2. **Test** - Generate comprehensive test suites  
3. **Repair** - Automatically fix any failing tests

---

## Quick Start

### Run Everything (One Command!)

```bash
cd /burg-archive/home/mck2199/jax-agents
./run_translation_workflow.sh --all
```

This will:
- ✅ Translate all default modules (clm_varctl, SoilStateType, SoilTemperatureMod)
- ✅ Generate tests for each module
- ✅ Run tests
- ✅ Prompt to repair if any tests fail

---

## Usage Modes

### 1. Complete Workflow (Recommended)

Run everything in sequence:

```bash
# Default modules with auto-repair
./run_translation_workflow.sh --all --auto-repair

# Just the workflow (no auto-repair)
./run_translation_workflow.sh --all
```

### 2. Individual Steps

Run specific parts of the workflow:

```bash
# Only translate
./run_translation_workflow.sh --translate

# Only generate tests (requires translated modules)
./run_translation_workflow.sh --test

# Only repair (requires test failures)
./run_translation_workflow.sh --repair
```

### 3. Interactive Mode

Let the script guide you:

```bash
./run_translation_workflow.sh --interactive
```

You'll see a menu:
```
What would you like to do?
1) Run complete workflow (translate → test → repair)
2) Translate modules only
3) Generate tests only
4) Run tests only
5) Repair failed tests
6) Exit
```

### 4. Custom Modules

Translate specific modules:

```bash
# Single module
./run_translation_workflow.sh --translate --modules "WaterFluxType"

# Multiple modules
./run_translation_workflow.sh --all --modules "clm_varctl,SoilStateType,CanopyFluxesMod"
```

---

## Command-Line Options

### Main Actions

| Option | Description |
|--------|-------------|
| `--all` | Run complete workflow (translate → test → repair) |
| `--translate` | Run translation only |
| `--test` | Run test generation only |
| `--repair` | Run repair agent only |
| `--interactive` | Interactive mode with menu |

### Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `--modules "m1,m2,m3"` | Comma-separated module list | clm_varctl, SoilStateType, SoilTemperatureMod |
| `--output DIR` | Output directory | ./translated_modules |
| `--auto-repair` | Automatically repair failures | false (prompts user) |
| `--skip-tests` | Skip test generation | false |

### Help

| Option | Description |
|--------|-------------|
| `-h`, `--help` | Show help message |

---

## Examples

### Example 1: Quick Test with Default Modules

```bash
./run_translation_workflow.sh --all
```

**What happens:**
1. Translates clm_varctl, SoilStateType, SoilTemperatureMod
2. Generates tests for each
3. Runs tests
4. Asks if you want to repair failures

### Example 2: Translate Specific Module

```bash
./run_translation_workflow.sh --translate --modules "WaterFluxType"
```

**What happens:**
- Only translates WaterFluxType
- Saves to `translated_modules/WaterFluxType/`

### Example 3: Complete Workflow with Auto-Repair

```bash
./run_translation_workflow.sh --all --auto-repair
```

**What happens:**
1. Translates all modules
2. Generates tests
3. Runs tests
4. Automatically repairs any failures (no prompt)

### Example 4: Just Generate Tests

```bash
# Assuming you already have translated modules
./run_translation_workflow.sh --test
```

**What happens:**
- Finds all modules in `translated_modules/`
- Generates test files for each
- Saves tests in `translated_modules/<module>/tests/`

### Example 5: Just Repair Failures

```bash
# Assuming you have test failures
./run_translation_workflow.sh --repair
```

**What happens:**
- Runs all tests
- Identifies failures
- Repairs each failed module
- Saves repairs to `repair_outputs/`

### Example 6: Custom Module List

```bash
./run_translation_workflow.sh --all \
  --modules "clm_varctl,WaterFluxType,CanopyFluxesMod,SoilTemperatureMod" \
  --auto-repair
```

**What happens:**
- Translates the 4 specified modules
- Generates tests for each
- Runs tests
- Auto-repairs failures

---

## Workflow Steps Explained

### Step 1: Translation

**Uses:** `examples/translate_with_json.py`

```bash
./run_translation_workflow.sh --translate
```

**What it does:**
- Reads `static_analysis_output/analysis_results.json`
- Reads `static_analysis_output/translation_units.json`
- Translates Fortran modules to JAX Python
- Saves to `translated_modules/<module>/<module>.py`

**Output:**
```
translated_modules/
├── clm_varctl/
│   └── clm_varctl.py
├── SoilStateType/
│   └── SoilStateType.py
└── SoilTemperatureMod/
    └── SoilTemperatureMod.py
```

### Step 2: Test Generation

**Uses:** `examples/generate_tests.py --all`

```bash
./run_translation_workflow.sh --test
```

**What it does:**
- Finds all translated modules
- Analyzes Python function signatures
- Generates comprehensive test suites
- Creates test data and documentation

**Output:**
```
translated_modules/
└── SoilTemperatureMod/
    ├── SoilTemperatureMod.py
    └── tests/
        ├── test_SoilTemperatureMod.py
        ├── test_data_SoilTemperatureMod.json
        └── test_documentation_SoilTemperatureMod.md
```

### Step 3: Testing

**Uses:** `pytest`

**What it does:**
- Runs pytest on each test file
- Captures pass/fail status
- Saves test output for repair agent

**Output:**
```
test_SoilTemperatureMod_output.txt  # Test results
```

### Step 4: Repair (if needed)

**Uses:** `examples/repair_agent_example.py`

```bash
./run_translation_workflow.sh --repair
```

**What it does:**
- Analyzes test failures
- Compares with Fortran reference (if available)
- Generates corrected code
- Creates root cause analysis

**Output:**
```
repair_outputs/
└── SoilTemperatureMod/
    ├── SoilTemperatureMod_corrected.py
    ├── root_cause_analysis_SoilTemperatureMod.md
    ├── failure_analysis_SoilTemperatureMod.json
    └── final_test_report_SoilTemperatureMod.txt
```

---

## Requirements

### Prerequisites

1. **Python 3.9+** with jax-agents installed
2. **Anthropic API Key**:
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"
   ```
3. **JSON Analysis Files**:
   - `static_analysis_output/analysis_results.json`
   - `static_analysis_output/translation_units.json`

### Check Requirements

The script automatically checks requirements when you run it. To manually verify:

```bash
# Check Python
python --version

# Check API key
echo $ANTHROPIC_API_KEY

# Check JSON files
ls -l static_analysis_output/
```

---

## Output Structure

After running the complete workflow:

```
jax-agents/
├── translated_modules/          # Translated Python code
│   ├── clm_varctl/
│   │   ├── clm_varctl.py
│   │   └── tests/
│   │       ├── test_clm_varctl.py
│   │       ├── test_data_clm_varctl.json
│   │       └── test_clm_varctl_output.txt
│   ├── SoilStateType/
│   └── SoilTemperatureMod/
│
└── repair_outputs/              # Repaired code (if needed)
    ├── SoilTemperatureMod/
    │   ├── SoilTemperatureMod_corrected.py
    │   ├── root_cause_analysis_SoilTemperatureMod.md
    │   └── failure_analysis_SoilTemperatureMod.json
    └── ...
```

---

## Advanced Usage

### Translate Without Testing

```bash
./run_translation_workflow.sh --translate --skip-tests
```

### Test Only Specific Module

```bash
# First translate
./run_translation_workflow.sh --translate --modules "SoilTemperatureMod"

# Then test just that module
python examples/generate_tests.py \
  --module SoilTemperatureMod \
  --python translated_modules/SoilTemperatureMod/SoilTemperatureMod.py \
  --output translated_modules/SoilTemperatureMod/tests
```

### Repair Specific Module

```bash
python examples/repair_agent_example.py \
  --module SoilTemperatureMod \
  --fortran translated_modules/SoilTemperatureMod/tests/FORTRAN_REFERENCE.F90 \
  --python translated_modules/SoilTemperatureMod/SoilTemperatureMod.py \
  --test-report translated_modules/SoilTemperatureMod/tests/test_SoilTemperatureMod_output.txt \
  --max-iterations 10
```

---

## Troubleshooting

### Issue: "ANTHROPIC_API_KEY not set"

**Solution:**
```bash
export ANTHROPIC_API_KEY="your-key-here"
# Or add to ~/.bashrc for permanent
echo 'export ANTHROPIC_API_KEY="your-key"' >> ~/.bashrc
```

### Issue: "analysis_results.json not found"

**Solution:**
Ensure JSON files exist:
```bash
ls -l static_analysis_output/
# Should show:
# - analysis_results.json
# - translation_units.json
```

### Issue: "Python not found"

**Solution:**
```bash
# Activate your Python environment
source /path/to/venv/bin/activate

# Or use specific Python version
alias python=python3.9
```

### Issue: Tests fail but repair doesn't work

**Solution:**
- Check if `FORTRAN_REFERENCE.F90` exists for the module
- Increase max iterations: `--max-iterations 10`
- Review test output manually
- Run repair interactively for better control

### Issue: Translation fails for a module

**Solution:**
```bash
# Check if module exists in JSON
grep "ModuleName" static_analysis_output/analysis_results.json

# Run with verbose output
python examples/translate_with_json.py 2>&1 | tee translation.log
```

---

## Tips & Best Practices

### 1. Start Small

```bash
# First try one simple module
./run_translation_workflow.sh --all --modules "clm_varctl"

# Then expand to more
./run_translation_workflow.sh --all --modules "clm_varctl,SoilStateType"
```

### 2. Use Auto-Repair for Production

```bash
# For batch processing
./run_translation_workflow.sh --all --auto-repair
```

### 3. Review Repairs

```bash
# Always check what was fixed
cat repair_outputs/SoilTemperatureMod/root_cause_analysis_SoilTemperatureMod.md
```

### 4. Iterative Development

```bash
# Translate once
./run_translation_workflow.sh --translate

# Iterate on tests and repairs
./run_translation_workflow.sh --test
./run_translation_workflow.sh --repair

# Test again after repair
pytest translated_modules/*/tests/test_*.py -v
```

### 5. Save Logs

```bash
# Save complete run log
./run_translation_workflow.sh --all 2>&1 | tee workflow_log.txt
```

---

## Comparison with Manual Process

### Manual (Old Way)

```bash
# Step 1: Translate
python examples/translate_with_json.py

# Step 2: Generate tests for each module
python examples/generate_tests.py --module clm_varctl --python ...
python examples/generate_tests.py --module SoilStateType --python ...
python examples/generate_tests.py --module SoilTemperatureMod --python ...

# Step 3: Run tests
pytest translated_modules/clm_varctl/tests/test_clm_varctl.py -v
pytest translated_modules/SoilStateType/tests/test_SoilStateType.py -v
pytest translated_modules/SoilTemperatureMod/tests/test_SoilTemperatureMod.py -v

# Step 4: Repair each failure
python examples/repair_agent_example.py --module ... --fortran ... --python ...
# ... repeat for each failure
```

**Time:** ~30-60 minutes (manual work)

### With Script (New Way)

```bash
./run_translation_workflow.sh --all --auto-repair
```

**Time:** ~5-10 minutes (automated)

---

## Script Features

### ✅ Automatic

- Checks requirements before running
- Handles errors gracefully
- Provides clear progress indicators
- Saves outputs automatically

### ✅ Flexible

- Run all steps or individual steps
- Customize modules list
- Configure output directories
- Interactive or command-line mode

### ✅ User-Friendly

- Colored output for clarity
- Progress indicators
- Error messages with suggestions
- Help text with examples

### ✅ Production-Ready

- Exit on error (safe defaults)
- Validates inputs
- Logs test results
- Organizes outputs clearly

---

## Quick Reference

```bash
# Most common commands

# Run everything
./run_translation_workflow.sh --all

# Run everything with auto-repair
./run_translation_workflow.sh --all --auto-repair

# Interactive mode
./run_translation_workflow.sh --interactive

# Specific modules
./run_translation_workflow.sh --all --modules "module1,module2"

# Just translate
./run_translation_workflow.sh --translate

# Just test
./run_translation_workflow.sh --test

# Just repair
./run_translation_workflow.sh --repair

# Help
./run_translation_workflow.sh --help
```

---

## Summary

The `run_translation_workflow.sh` script provides a complete, automated solution for:

✅ **Translating** Fortran modules to JAX Python  
✅ **Testing** translated code comprehensively  
✅ **Repairing** any failures automatically  

It saves time, reduces errors, and makes the entire workflow accessible through simple commands.

**Get started:**
```bash
./run_translation_workflow.sh --all
```

🎉 **That's it! One command to rule them all!**

