# Contributing to CLM-ML-JAX

This guide is for land-surface scientists and developers who want to add a process or a
scheme, fix a bug, or improve the documentation. The model is a scientific tool: it must
reproduce the Fortran CLM-ml v2 results, so changes are tested against them.

## Development setup

```bash
git clone https://github.com/linniahawkins/clm-ml-jax.git
cd clm-ml-jax
pip install -e ".[dev]"      # dev extras: pytest, pytest-cov, black, ruff

clm-ml --help
clm-ml run configs/chats7_1day          # one day at CHATS7 (about a minute on a CPU)
python -m pytest tests/clm_ml tests/fortran_validation -m "not slow" -q
```

## Repository layout

```
src/clm_ml/
  processes/    the processes, pure functions on 0-based arrays   <- most contributions go here
  step.py       one time step of one patch (State + inputs + constants -> StepState)
  state.py      State (prognostic), Diagnostics, PatchLayers
  setup/        every patch's constants and initial state (pure functions)
  driver.py     compiled chunks over all patches, output streams, checks, restarts
  experiment.py experiment folders: load, new, upgrade, run
  run.py, domain.py, config.py, params.py   the four files' settings (run, domain, model, params)
  compsets/      what `clm-ml new` copies (towers, parameter sets, experiments)
  io/           tower and CLM history readers, restarts (netCDF), legacy .out writer, YAML schema
configs/        the compsets written out as experiment folders
data/           CHATS7 forcing and the CLM-ml namelists (test and example data)
tests/
  clm_ml/              unit tests (fast)
  fortran_validation/  160+ routine tests against Fortran golden values (rtol 1e-9)
  regression/          end-to-end runs against committed references
docs/design/    decisions.md (decision register), open_items.md, model_overview.md, plan
```

## Adding a process or a scheme

1. **Write it by process** in `src/clm_ml/processes/`: pure functions on 0-based arrays (layers
   from the ground up, leaves sunlit/shaded, bands visible/near-infrared); everything read is an
   argument and everything changed is returned. No hidden globals.
2. **Name the Fortran source** in the docstring (`Fortran reference: MLFooMod.F90::Bar`) and the
   paper for any choice not in Fortran.
3. **Masks, not loops:** compute on every layer slot and keep results with the masks of
   `clm_ml.processes.layers`; no Python loops over layers inside the step, no `if` on traced
   values (`jnp.where`, `lax.cond`), no host callbacks in compiled code.
4. **Static choices vs numbers:** a scheme choice is a model.yaml option; every number that could
   be calibrated is a per-PFT parameter in params.yaml (`clm_ml.params.PARAM_SPECS`) and stays
   differentiable.
5. **64-bit precision:** `clm_ml` enables float64; constants are Python floats.

## Adding or changing a setting

Every setting of `run.yaml`, `domain.yaml` and `model.yaml` is declared once in the code with
`clm_ml.io.yaml_schema.option(...)`: its default (or required), its explanation, the Fortran
name it replaces, and when it applies (`applies="towers"`, `when={"source": "file"}`). The
generated files, the error messages and the reference documentation come from these
declarations. After adding or changing one:

- give the old form to `clm-ml upgrade` if a setting was renamed or regrouped
  (`clm_ml.experiment._migrate`);
- regenerate the folders in `configs/`:
  `clm-ml new configs/NAME --from NAME --force` for each compset
  (`tests/clm_ml/test_configs.py` checks they are current);
- record the decision in `docs/design/decisions.md`.

## Tests and the regression policy

End-to-end runs cost about a minute each, so test in tiers:

- **after each edit:** `python -m pytest tests/clm_ml tests/fortran_validation -m "not slow" -q`
  plus the test being written;
- **before each commit:** the bit-for-bit and noise-twin tests
  (`python -m pytest tests/regression/test_end_to_end_regression.py
  tests/regression/test_clm_ml_equivalence.py -m regression -q`), plus the changed feature's own
  end-to-end test;
- **at the end of a phase:** all of `tests/regression -m regression`, the gradient tier
  (`python tests/gradient_tier.py`) and the slow tests.

A refactor that should not change the arithmetic must pass **bit for bit**. Every change must
pass the **noise-twin** test (within the model's own round-off sensitivity, twins perturbed by
1e-12). A science change is the maintainer's decision; it regenerates the reference
(`python tests/regression/record_run.py --update-reference`) explicitly, in its own commit,
recorded in `docs/design/decisions.md`. Never edit the Fortran golden JSON files to make a test
pass.

Fortran validation tests: capture golden I/O from the Fortran build, add the JSON file to
`tests/fortran_validation/golden_IO/` and a case in `tests/fortran_validation/test_golden_jax.py`.

## Code style

```bash
black --check --line-length 100 src tests timing && ruff check src tests timing
```

Both run in CI.

## Workflow

Work happens on one branch per phase (e.g. `phase8a/housekeeping`); commits are made when the
tests of the tiers above pass. Record design decisions in `docs/design/decisions.md` and open
questions in `docs/design/open_items.md` in the same commit as the change.

## Reporting bugs

Open an issue at https://github.com/linniahawkins/clm-ml-jax/issues with:
- a minimal experiment folder or script that reproduces it;
- the Python and JAX versions (`python --version`, `python -c "import jax; print(jax.__version__)"`);
- the expected and actual output.
