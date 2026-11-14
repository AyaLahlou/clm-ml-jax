# CLM-ML-JAX: the CLM-ml v2 multilayer canopy model in JAX

[![CI](https://github.com/linniahawkins/clm-ml-jax/actions/workflows/ci.yml/badge.svg)](https://github.com/linniahawkins/clm-ml-jax/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-green.svg)](LICENSE)

CLM-ml v2 is the multilayer canopy model developed for the Community Land Model by Gordon
Bonan (Bonan et al. 2021, 2026). This repository is a JAX version of it. It gives the same
results as the Fortran model ([gbonan/CLM-ml_v2.CHATS](https://github.com/gbonan/CLM-ml_v2.CHATS)),
and adds GPU execution, gradients for calibration and sensitivity analysis, and runs from one
flux tower up to the whole globe.

How the model is organized, what runs when, and how it differs from the Fortran:
[`docs/how_the_model_works.md`](docs/how_the_model_works.md).

---

## Install

You need Python 3.10 or newer and git.

```bash
git clone https://github.com/linniahawkins/clm-ml-jax.git
cd clm-ml-jax
pip install -e .            # the clm_ml package and the clm-ml command
clm-ml --help
```

A fresh conda environment avoids clashes with other packages:

```bash
conda create -n clm-ml-jax python=3.11
conda activate clm-ml-jax
pip install -e .
```

This installs JAX for the CPU. To run on an NVIDIA GPU, also install the CUDA build of JAX
(`pip install "jax[cuda12]==0.10.2"`, the version the model is tested with; see the
[JAX install guide](https://docs.jax.dev/en/latest/installation.html)), or create the
environment from `environment.yml`. With the CUDA build installed, set `JAX_PLATFORMS=cpu` on
machines without a GPU (otherwise JAX prints a harmless CUDA error at start-up).
For development (tests, linting) install the extras: `pip install -e ".[dev]"`.

---

## Run the CHATS tower (the default test case)

CHATS is the walnut orchard in Dixon, California, used to test CLM-ml v2 (Bonan, Burns &
Patton 2026). The forcing and CLM input files for May 2007 are in this repository
(`data/input_files/`), so this runs right after installing.

```bash
clm-ml new chats7 --from chats7_1day    # write an experiment folder from the compset
clm-ml check chats7                      # check the settings and print a summary (no run)
clm-ml run chats7                        # run 1 May 2007 (under a minute on a CPU)
```

The output is in `chats7/output/`:

- `CHATS7_halfhourly_2007-05-01_patch.nc`: every output variable, every 30 minutes (fluxes,
  profiles through the canopy, sunlit and shaded leaves, soil temperature), with units;
- `config/`: a copy of the four settings files exactly as the run read them;
- `restart/`: the state at the end of the run, to continue from.

Or run it from Python, which also prints the mean fluxes and saves a figure:

```bash
python examples/run_tower.py            # writes chats7_example/ and runs it
```

Read the output with xarray:

```python
import xarray as xr

ds = xr.open_dataset("chats7/output/CHATS7_halfhourly_2007-05-01_patch.nc")
ds.latent_heat_flux.isel(patch=0).plot()          # through the day
ds.air_temperature.isel(patch=0, time=24)         # profile: one value per canopy layer
```

The first run compiles the model (most of that minute); compiled programs are cached on disk
(in `$SCRATCH/jax_compile_cache` where `$SCRATCH` is defined, as on NCAR's machines, otherwise
`~/.cache/jax_compile_cache`; set `JAX_COMPILATION_CACHE_DIR` to choose), so later runs with the
same settings start faster. Times are UTC, at the end of each output
interval. Canopy layers are numbered from the ground;
slots above the top layer are NaN.

### Change the run

An experiment is a folder of four files. Every setting is listed in them with an explanation:

| File | What it sets |
|---|---|
| `run.yaml` | when and with what: dates, forcing and land-data files, initial conditions, output, restarts |
| `domain.yaml` | where: towers (location, forcing height, soil, patches) or a grid |
| `model.yaml` | how: the scheme for each process, the model step, canopy layers |
| `params.yaml` | which plants: every parameter of every plant functional type |

Edit the files, or change a setting for one run with `--set`:

```bash
clm-ml run chats7 --set 'run.time.length=31 days'                                   # all of May 2007
clm-ml run chats7 --set model.processes.stomatal_conductance.scheme=medlyn
clm-ml run chats7 --set params.broadleaf_deciduous_temperate_tree.vcmax25=100
```

### Other towers and global runs

- **Your own tower:** copy a folder (`clm-ml new my_tower --from chats7_1day`), then edit
  `domain.yaml` (location, forcing height, soil, patches) and `run.yaml` (dates, forcing and
  CLM history files). `python examples/run_tower.py --folder my_tower` runs it.
- **Global:** `examples/run_global.py` sets up a case on the 1° CLM grid from the
  compset `grid_1deg_2000` (CRUJRA forcing, CLM6 parameters; its input files are on NCAR's
  Glade storage): edit the settings at its top (case folder, dates, region, machine, GPUs,
  account) and run `python examples/run_global.py`. On Derecho the case folder gets a batch
  job, submitted with `qsub CASE_DIR/submit.pbs`; elsewhere run `clm-ml run CASE_DIR`
  (`clm-ml new --machine derecho --account ...` writes the same job). The whole globe needs
  a GPU; a small region (e.g. `REGION = "{lat: [35, 40], lon: [-100, -95]}"`) runs on a CPU.
- **Ready-made folders:** `configs/` (`configs/README.md`), including May 2007 set up for the
  comparison with the Fortran model.
- **From a Fortran CLM-ml namelist:** `clm-ml convert data/namelists/nl.CHATS7.1day -o chats7/`.

---

## Tests

The JAX model is checked against the Fortran model routine by routine (160+ tests) and end to
end (CHATS7, May 2007). Every change must also pass a bit-for-bit regression test:

```bash
pip install -e ".[dev]"
python -m pytest tests/clm_ml tests/fortran_validation -m "not slow" -q   # fast
python -m pytest tests/regression -m regression -q                         # end to end (~20 min)
```

How to add a process or scheme, and the testing rules: [`CONTRIBUTING.md`](CONTRIBUTING.md).
The development record (decisions, open issues): [`docs/design/`](docs/design/).

---

## Citation

Please cite the software (`CITATION.cff`) and the model it translates:

```bibtex
@software{clm_ml_jax,
  title  = {CLM-ML-JAX: the CLM-ml v2 multilayer canopy model in JAX},
  author = {Hawkins, Linnia and Lahlou, Aya},
  year   = {2026},
  url    = {https://github.com/linniahawkins/clm-ml-jax},
  note   = {JAX translation of CLM-ml v2 (Bonan et al.)}
}
```

- Bonan, G. B., Burns, S. P., and Patton, E. G. (2026), Beyond surface fluxes: Observational
  and computational needs of multilayer canopy models – A walnut orchard test case,
  *Agricultural and Forest Meteorology*, 378, 110960, doi:10.1016/j.agrformet.2025.110960.
- Bonan, G. B., Patton, E. G., Finnigan, J. J., Baldocchi, D. D., and Harman, I. N. (2021),
  Moving beyond the incorrect but useful paradigm: reevaluating big-leaf and multilayer plant
  canopies to model biosphere-atmosphere fluxes – a review, *Agricultural and Forest
  Meteorology*, 306, 108435, doi:10.1016/j.agrformet.2021.108435.

---

## License

BSD-3-Clause. See [LICENSE](LICENSE). Bugs and questions:
[open an issue](https://github.com/linniahawkins/clm-ml-jax/issues).
