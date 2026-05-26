# CLM-JAX Tests

Comprehensive test suite for all CLM-JAX modules. Tests are organized to mirror the source
code hierarchy in `src/`, using JSON reference data generated from the Fortran reference
implementation for numerical validation.

## Directory Structure

```
tests/
├── conftest.py                         # Shared pytest fixtures and JAX configuration
├── test_differentiability.py           # Integration tests for jax.grad through full model
├── test_data/                          # JSON reference datasets (61 files, ~1.3 MB)
│
├── multilayer_canopy/                  # Multi-layer canopy physics (19 test files)
│   ├── test_MLCanopyFluxesMod.py       # Top-level canopy physics integration
│   ├── test_MLCanopyFluxesType.py      # Canopy state data structures
│   ├── test_MLCanopyNitrogenProfileMod.py
│   ├── test_MLCanopyTurbulenceMod.py   # Turbulence closure (Harman-Finnigan RSL)
│   ├── test_MLCanopyWaterMod.py
│   ├── test_MLLeafBoundaryLayerMod.py
│   ├── test_MLLeafFluxesMod.py
│   ├── test_MLLeafHeatCapacityMod.py
│   ├── test_MLLeafPhotosynthesisMod.py # Medlyn/Ball-Berry/WUE stomatal models
│   ├── test_MLLongwaveRadiationMod.py
│   ├── test_MLMathToolsMod.py
│   ├── test_MLPlantHydraulicsMod.py
│   ├── test_MLSoilFluxesMod.py
│   ├── test_MLSolarRadiationMod.py     # Norman / two-stream radiation
│   ├── test_MLWaterVaporMod.py
│   ├── test_MLclm_varcon.py
│   ├── test_MLclm_varctl.py
│   ├── test_MLclm_varpar.py
│   └── test_MLinitVerticalMod.py       # Vertical grid initialization
│
├── offline_driver/                     # Tower-site driver (7 test files)
│   ├── test_CLMml.py                   # Clump management and decomposition
│   ├── test_CLMml_driver.py            # Full tower driver integration
│   ├── test_SoilTexMod.py              # Soil texture properties
│   ├── test_TowerDataMod.py            # Tower site metadata (15 AmeriFlux sites)
│   ├── test_TowerMetMod.py             # Tower meteorology ingest
│   ├── test_clmDataMod.py              # CLM history file I/O
│   └── test_controlMod.py             # Namelist parsing
│
├── clm_src_utils/                      # Core utilities (5 test files)
│   ├── test_clm_time_manager.py        # Calendar, time-stepping, date arithmetic
│   ├── test_clm_varorb.py             # Orbital parameters
│   ├── test_fileutils.py              # File I/O helpers
│   ├── test_restUtilMod.py            # Restart file utilities
│   └── test_spmdMod.py               # Distributed computing (pmap)
│
├── cime_src_share_util/               # Placeholder — no tests yet
└── clm_src_cpl/                       # Placeholder — no tests yet
```

## Quick Start

```bash
# Run the full suite
pytest

# Run one module's tests
pytest tests/multilayer_canopy/test_MLLeafPhotosynthesisMod.py

# Run a single test
pytest tests/multilayer_canopy/test_MLCanopyFluxesMod.py::test_canopy_energy_balance

# Skip slow tests (GPU runs, full-day simulations)
pytest -m "not slow"

# With coverage
pytest --cov=src --cov-report=html
```

## Test Configuration

| File | Purpose |
|------|---------|
| `pytest.ini` | Test discovery, markers, warning filters |
| `conftest.py` | Shared fixtures, JAX 64-bit setup, test data builders |

### Markers

| Marker | Usage |
|--------|-------|
| `@pytest.mark.slow` | GPU runs, full timestep loops; excluded by `-m "not slow"` |
| `@pytest.mark.unit` | Pure-function tests with no I/O |
| `@pytest.mark.integration` | Multi-module tests requiring full model state |

### Key Fixtures (`conftest.py`)

| Fixture | Returns | Description |
|---------|---------|-------------|
| `jax_config` | — | Enables 64-bit floats; applied session-wide |
| `sample_grid` | `GridInfo` | Single-patch grid (begg=endg=begp=endp=0) |
| `sample_arrays` | dict of JAX arrays | Random test inputs with correct shapes |
| `test_data_dir` | `Path` | Path to `tests/test_data/` |
| `clmData_wrapper` | callable | Wraps `clmDataMod` functions for testing |
| `MLCanopyFluxes_wrapper` | callable | Wraps top-level canopy physics |
| `SoilTemperature_wrapper` | callable | Wraps soil temperature solver |
| `MutableWrapper` | class | Wraps immutable NamedTuples for in-place update during test setup |

## Test Data

Reference data lives in `tests/test_data/` as 61 JSON files generated from the Fortran
reference implementation. Each file contains input arrays and expected output arrays for
one module. Tests compare JAX output against these values within a tolerance (typically
`atol=1e-6` for physics quantities).

To regenerate reference data (requires the Fortran submodule):
```bash
python clm-ml-fortran/tests/python/generate_golden.py
```

## Test Types

**Value tests** — Compare output arrays against Fortran reference data. This is the primary
correctness check: a passing value test means the Python/JAX implementation is numerically
equivalent to the Fortran original.

**Gradient tests** — Compare `jax.grad` output against finite differences. These verify
that the JAX implementation is fully differentiable, which is required for gradient-based
parameter calibration and sensitivity analysis.

**Integration tests** — Drive multiple modules together (e.g., `test_CLMml_driver.py`,
`test_differentiability.py`) to check end-to-end behavior.

## Writing New Tests

1. Create `tests/<module_dir>/test_<ModuleName>.py`
2. Load reference data with the `test_data_dir` fixture:
   ```python
   def test_my_function(test_data_dir):
       data = json.loads((test_data_dir / "test_data_MyMod.json").read_text())
   ```
3. Use `@pytest.mark.slow` for any test that takes more than a few seconds
4. For gradient tests, use `jax.grad` and `jnp.allclose` with generous tolerance:
   ```python
   grad_ad = jax.grad(fn)(x)
   grad_fd = finite_difference(fn, x, eps=1e-5)
   assert jnp.allclose(grad_ad, grad_fd, rtol=1e-3)
   ```

## Coverage Gaps

These source directories have no dedicated tests (covered indirectly through integration tests):

| Source directory | Indirect test coverage |
|-----------------|----------------------|
| `clm_src_biogeophys/` | via `test_MLCanopyFluxesMod.py`, `test_CLMml_driver.py` |
| `clm_src_main/` | via `test_CLMml_driver.py`, `test_differentiability.py` |
| `clm_src_cpl/` | via `test_differentiability.py` |

## Differentiability Tests (`test_differentiability.py`)

Tests end-to-end `jax.grad` through the full model forward pass. Requires a valid namelist
and input files; tests that can't find the files are skipped with a clear message. Run these
on GPU for reasonable speed:

```bash
pytest tests/test_differentiability.py -v
```
