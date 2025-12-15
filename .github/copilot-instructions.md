<!-- Copilot / AI agent instructions for the clm-ml-jax repository -->
# Quickly get productive in this repo

This project translates and tests a Fortran-based canopy/climate model (CLM-ml) to JAX/Python. Use these focused notes when generating, editing, or testing code.

- Big picture: Fortran source lives under `CLM-ml_v1/` (many stubbed CLM directories and `multilayer_canopy/`). The translation pipeline lives in `jax-agents/` and produces `translated_modules/`.
- Key components:
  - `CLM-ml_v1/` — original Fortran sources: `clm_src_biogeophys/`, `clm_src_main/`, `multilayer_canopy/`, `offline_driver/`, etc. Example: `MLCanopyFluxesType.F90`, `SoilStateType.F90`.
  - `jax-agents/` — translator, test, and repair agents. Important files: `jax-agents/translator.py`, `static_analysis_output/` (expected JSON inputs), `examples/` for usage patterns.
  - `translated_modules/` — outputs produced by translator (module-per-directory, Python + tests + notes).
  - `tests/` — pytest-based tests that mirror `src/` structure; `conftest.py` contains shared fixtures. Tests use the JAX CPU backend by default.

- Developer workflows (commands you can rely on):
  - Run full test suite: `pytest` from repo root.
  - Run module tests: `pytest tests/clm_src_biogeophys/test_SoilTemperatureMod.py` (adjust path/name).
  - Run translation examples: `python jax-agents/examples/translate_with_json.py` or `python jax-agents/examples/batch_translate_modules.py` (see `jax-agents/README.md`).
  - Verify static-analysis inputs: `python jax-agents/examples/verify_json_integration.py`.

- Project-specific conventions and patterns:
  - Translation is iterative and unit-based: translator splits Fortran modules into translation units (see `translation_units.json`) and translates unit-by-unit, then assembles (N+1 LLM calls). Respect and preserve that approach when generating prompts.
  - Tests mirror `src/` paths; new tests should follow `tests/<component>/test_<Module>.py` and use shared fixtures in `tests/conftest.py`.
  - Slow/performance tests are marked `@pytest.mark.slow` — use `-m "not slow"` in CI if needed.
  - Fortran module -> Python module mapping: prefer module name (e.g., `SoilStateType`) rather than raw file paths when invoking translator helpers.

- Integration points and dependencies to watch for:
  - `jax-agents` expects static analysis JSON in `jax-agents/static_analysis_output/` named `analysis_results.json` and `translation_units.json`.
  - Translated code often relies on small hand-authored runtime helpers in `src/` — inspect `src/` before large refactors.
  - Offline executable: for running the standalone Fortran offline case, see `CLM-ml_v1/offline_executable` and the run command in that README (`./prgm.exe < nl.US-UMB.2006`).

- Prompting / code generation guidance (do this in your prompts):
  1. Reference the exact Fortran module and file (path under `CLM-ml_v1/`) and the relevant translation unit id from `translation_units.json` when available.
  2. Keep translations unit-scoped; do not attempt whole-module rewrites in one pass. Use the existing iterative assembly flow.
  3. When changing generated Python, also update or add a pytest in `tests/` mirroring the module path and use fixtures from `tests/conftest.py`.

- Helpful examples to cite in prompts:
  - Fortran source: `CLM-ml_v1/multilayer_canopy/MLCanopyFluxesMod.F90` (data types in `MLCanopyFluxesType.F90`).
  - Translator usage: `jax-agents/README.md` shows `TranslatorAgent(...).translate_module(module_name="SoilStateType")`.
  - Tests: `tests/clm_src_biogeophys/test_SoilTemperatureMod.py` (pattern: `test_<Module>.py`).

If anything here looks incomplete or you want more examples (specific module translation prompt templates, sample test fixture usage, or CI test filters), tell me which module or flow and I'll add it.
