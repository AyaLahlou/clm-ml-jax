# Getting Started with JAX-CTSM Translation Agents

**5-Minute Quick Start Guide**

## What You Need

- [x] Python 3.9+
- [x] Anthropic API key ([get free trial here](https://console.anthropic.com/))
- [x] 5 minutes

## Three Steps to Your First Conversion

### Step 1: Install (1 minute)

```bash
cd /burg-archive/home/mck2199/jax-agents
pip install -e .
```

### Step 2: Configure (1 minute)

```bash
# Copy environment template
cp .env.example .env

# Add your API key (get from console.anthropic.com)
echo "ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY-HERE" > .env
```

### Step 3: Run Your First Example (3 minutes)

```bash
# Analyze a Fortran module (costs ~$0.05)
python examples/analyze_module.py
```

**Expected Output:**
```
📊 Analyzing Fortran module: CNPhenologyMod.F90
🤖 Static Analysis is thinking...
✓ Analysis complete!
  - Found 12 subroutines
  - Found 5 data types
  - Found 15 module dependencies

Analysis Cost: $0.0487
```

## What Just Happened?

The **Static Analysis Agent** just:
1. Read a 1,500-line Fortran module
2. Extracted all subroutines, types, and dependencies
3. Identified spatial hierarchy patterns
4. Suggested JAX translation approaches
5. Saved complete analysis to JSON

All in about 30 seconds! 🚀

## Next: Convert a Full Module

Ready to convert Fortran to JAX?

```bash
# Full conversion (costs ~$0.50)
python examples/convert_single_module.py
```

This will:
1. ✅ Plan the conversion strategy
2. ✅ Analyze Fortran structure
3. ✅ Generate JAX code with type hints
4. ✅ Create parameter classes
5. ✅ Generate documentation
6. ✅ Produce a detailed report

**Output Files:**
```
jax-ctsm/src/jax_ctsm/physics/
├── cngresp.py                    # JAX physics module
├── cngresp_params.py             # Parameter class
├── cngresp_translation_notes.md  # Translation decisions
└── cngresp_conversion_report.md  # Full report
```

## Understanding the Agents

### 🎯 Orchestrator Agent
**Job**: Project manager

Coordinates everything:
- Plans conversion approach
- Delegates to other agents
- Synthesizes results
- Tracks costs

### 🔍 Static Analysis Agent
**Job**: Code archaeologist

Analyzes Fortran:
- Extracts structure
- Maps dependencies
- Identifies patterns
- Finds vectorization opportunities

### ⚡ Translator Agent
**Job**: Code converter

Converts to JAX:
- Pure functions
- Immutable state
- Type hints
- Vectorized operations
- Documentation

## Example Workflows

### Workflow 1: Just Analyze (Learning Phase)

```python
from pathlib import Path
from jax_agents import StaticAnalysisAgent

analyzer = StaticAnalysisAgent()
analysis = analyzer.analyze_module(
    Path("CTSM/src/biogeochem/CNPhenologyMod.F90")
)

print(f"Module: {analysis.module_name}")
print(f"Subroutines: {len(analysis.subroutines)}")
print(f"Complexity: {analysis.jax_translation_notes['challenges']}")
```

**Use when**: You want to understand a module before converting

### Workflow 2: Full Conversion (Production)

```python
from pathlib import Path
from jax_agents import OrchestratorAgent

orchestrator = OrchestratorAgent(
    ctsm_dir=Path("CTSM"),
    jax_ctsm_dir=Path("jax-ctsm"),
)

result = orchestrator.convert_module(
    fortran_file="src/biogeochem/CNGRespMod.F90",
)

print(f"Generated: {result.saved_files}")
print(f"Cost: ${result.cost_summary['total_cost_usd']:.2f}")
```

**Use when**: Ready to convert a module to JAX

### Workflow 3: Custom Two-Step (Advanced)

```python
from jax_agents import StaticAnalysisAgent, TranslatorAgent

# Step 1: Analyze
analyzer = StaticAnalysisAgent()
analysis = analyzer.analyze_module(fortran_file)
analysis.save("my_analysis.json")

# Review analysis, make notes...

# Step 2: Translate
translator = TranslatorAgent(jax_ctsm_dir=jax_dir)
translation = translator.translate_module(
    fortran_file, 
    analysis,
    output_dir="custom_output/"
)
```

**Use when**: You want control over each step

## Costs

**Typical Costs** (using Claude Sonnet 4.5):

| Task | Cost | Time |
|------|------|------|
| Analyze only | $0.05-0.15 | 30s |
| Convert simple module | $0.10-0.30 | 1-2 min |
| Convert medium module | $0.30-1.00 | 2-4 min |
| Convert complex module | $1.00-3.00 | 4-8 min |

**Cost Tracking**:
```python
cost = orchestrator.get_cost_estimate()
print(f"Spent so far: ${cost['total_cost_usd']:.4f}")
```

## What Gets Generated

### Physics Module (`cngresp.py`)
```python
from typing import NamedTuple
import jax.numpy as jnp

class GRespParams(NamedTuple):
    """Growth respiration parameters."""
    growth_resp_factor: float = 0.3  # Construction cost

def calculate_growth_respiration(
    patch_state: PatchState,
    params: GRespParams,
) -> CarbonFlux:
    """Calculate growth respiration.
    
    Translated from: CTSM/src/biogeochem/CNGRespMod.F90
    
    Args:
        patch_state: Patch state with allocation info
        params: Growth respiration parameters
        
    Returns:
        Growth respiration fluxes
    """
    # Pure function, vectorized, type-hinted!
    gresp = patch_state.allocation * params.growth_resp_factor
    return CarbonFlux(gresp=gresp)
```

### Parameters (`cngresp_params.py`)
```python
class GRespParams(NamedTuple):
    """Parameters for growth respiration."""
    growth_resp_factor: float = 0.3
    # ... with helper methods
```

### Report (`conversion_report.md`)
- Conversion strategy
- Analysis findings
- Translation decisions
- Testing recommendations
- Cost breakdown

## Configuration

Edit `config.yaml`:

```yaml
# Control agent behavior
jax_patterns:
  add_type_hints: true      # Always type-hint
  use_pure_functions: true  # No side effects
  use_immutable_state: true # NamedTuples only

# Set limits
cost_management:
  max_cost_per_module: 5.0  # USD
  warn_on_large_context: true
```

## Troubleshooting

### "API Key not found"
```bash
# Make sure .env exists and has your key
cat .env
# Should show: ANTHROPIC_API_KEY=sk-ant-...

# If not, create it:
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

### "Module not found"
```bash
# Verify installation
pip install -e .

# Check import works
python -c "from jax_agents import OrchestratorAgent; print('OK')"
```

### "Costs too high"
```python
# Start with smaller modules
# Check token usage
analysis = analyzer.analyze_module(small_file)
cost = analyzer.get_cost_estimate()
print(f"Analysis cost: ${cost['total_cost_usd']:.4f}")
```

## Learning Path

1. **Day 1**: Run `analyze_module.py` on different Fortran files
2. **Day 2**: Run `convert_single_module.py` on a simple module
3. **Day 3**: Review generated code, understand patterns
4. **Day 4**: Try `translate_with_context.py` for custom workflow
5. **Day 5**: Convert a medium-complexity module
6. **Week 2**: Start batch conversions

## Tips for Success

✅ **Start Small**: Begin with simple modules (< 500 lines)  
✅ **Review Output**: Always inspect generated code  
✅ **Validate**: Compare with Fortran behavior  
✅ **Track Costs**: Monitor spending per module  
✅ **Read Reports**: The conversion reports have valuable insights  
✅ **Iterate**: Refine prompts if needed  

## Quick Commands

```bash
# Install
pip install -e .

# Configure
cp .env.example .env && nano .env

# Analyze
python examples/analyze_module.py

# Convert
python examples/convert_single_module.py

# Batch
python examples/batch_conversion.py

# Check logs
tail -f logs/*.log
```

## Where to Go from Here

- 📖 **Detailed Guide**: Read `README.md`
- 🏗️ **Architecture**: See `ARCHITECTURE.md`
- 🚀 **Full Walkthrough**: Check `QUICKSTART.md`
- 💻 **Examples**: Explore `examples/` directory

## Ready to Start?

```bash
# Your first conversion in 3 commands:
pip install -e .
echo "ANTHROPIC_API_KEY=your-key" > .env
python examples/convert_single_module.py
```

**That's it! You're converting Fortran to JAX with AI! 🎉**

---

Questions? Check the logs in `logs/` or review the documentation.

Happy converting! 🚀

