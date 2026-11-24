# Getting started

This page installs the model, runs one day at the CHATS walnut orchard, and reads the output.
It takes a few minutes on a laptop.

## Install

You need Python 3.10 or newer and git. A fresh conda environment avoids clashes with other
packages:

```bash
git clone https://github.com/linniahawkins/clm-ml-jax.git
cd clm-ml-jax
conda create -n clm-ml-jax python=3.11
conda activate clm-ml-jax
pip install -e .            # the clm_ml package and the clm-ml command
clm-ml --help
```

If `pip` cannot build `netCDF4` (on some Macs), install it from conda-forge first:
`conda install -c conda-forge netcdf4`.

This installs JAX for the CPU, which is all a tower run needs. To run on an NVIDIA GPU, also
install the CUDA build of JAX (`pip install "jax[cuda12]==0.10.2"`, the version the model is
tested with; see the [JAX install guide](https://docs.jax.dev/en/latest/installation.html)), or
create the environment from [`environment.yml`](repo:environment.yml). With the CUDA build
installed, set `JAX_PLATFORMS=cpu` on machines without a GPU (otherwise JAX prints a harmless
CUDA error at start-up).

## Run one day at CHATS

CHATS is the walnut orchard in Dixon, California, used to test CLM-ml v2 (Bonan, Burns &
Patton 2026). Its forcing and CLM input files for April and May 2007 are in the repository
(`data/input_files/`), so this runs right after installing:

```bash
clm-ml new chats7 --from chats7_1day    # write an experiment folder from a ready-made setup
clm-ml check chats7                      # check the settings and print a summary (no run)
clm-ml run chats7                        # run 1 May 2007
```

The first run compiles the model, which takes most of a minute. Compiled programs are cached
on disk (`$SCRATCH/jax_compile_cache` where `$SCRATCH` is defined, as on NCAR's machines,
otherwise `~/.cache/jax_compile_cache`; set `JAX_COMPILATION_CACHE_DIR` to choose), so later
runs with the same model settings start in seconds.

The output is in `chats7/output/`:

- `CHATS7_halfhourly_2007-05-01_patch.nc`: every output variable every 30 minutes (fluxes,
  profiles through the canopy, sunlit and shaded leaves, soil temperature), with units;
- `config/`: a copy of the four settings files exactly as the run read them;
- `restart/`: the state at the end of the run, to continue from.

## Read the output

```python
import xarray as xr

ds = xr.open_dataset("chats7/output/CHATS7_halfhourly_2007-05-01_patch.nc")
ds.latent_heat_flux.isel(patch=0).plot()          # through the day
ds.air_temperature.isel(patch=0, time=24)         # a profile: one value per canopy layer
```

Times are UTC, at the end of each output interval. Canopy layers are numbered from the
ground, and `layer_height` gives their heights; slots above a patch's top layer are NaN.

`python examples/run_tower.py` does the same run from Python, prints the day's mean fluxes and
saves a figure (see [Examples](examples.md)).

## The four files

An experiment is a folder of four files. Every setting is written in them with an explanation
(the same text as the [settings reference](reference/settings.md)):

| File | What it sets |
|---|---|
| `run.yaml` | when and with what: dates, forcing and land-data files, initial conditions, output, restarts |
| `domain.yaml` | where: towers (location, forcing height, soil, patches) or a grid |
| `model.yaml` | how: the scheme for each process, the model step, canopy layers |
| `params.yaml` | which plants: every parameter of every plant functional type (PFT) |

The files are complete and strict: a missing, misspelled or misplaced key is an error that
names the file and key, never a silent default. `clm-ml check` prints where each value comes
from.

## Change a setting

Edit the files, or change a setting for one run with `--set` (`FILE.KEY=VALUE`):

```bash
clm-ml run chats7 --set 'run.time.length=31 days'                     # all of May 2007
clm-ml run chats7 --set model.processes.stomatal_conductance.scheme=medlyn
clm-ml run chats7 --set params.broadleaf_deciduous_temperate_tree.vcmax25=100
```

Output is configured in `run.yaml` under `output.streams`: each stream has its own variables,
frequency (every step to yearly), statistic (mean, instantaneous, minimum, maximum) and files.
A run continues from a restart with `run.yaml` `initial_conditions`.

## Ready-made setups

`clm-ml new FOLDER --from NAME` writes a folder from one of these:

| Name | What it runs |
|---|---|
| `chats7_1day` | One day (1 May 2007) at CHATS: the quickest complete run |
| `chats7_may2007_fortran` | May 2007 at CHATS set up as the Fortran model runs it, for comparison with Fortran output |
| `grid_1deg_2000` | A 1° global run of 2000 (CRUJRA forcing, CLM6 parameters); its input files are on NCAR's Glade storage |

The same setups written out are in [`configs/`](repo:configs/README.md).

## From a Fortran CLM-ml namelist

If you have run the Fortran CLM-ml v2, `clm-ml convert` turns its namelist (`&clmML_inparm`)
into an experiment folder, translating the integer option codes into named settings:

```bash
clm-ml convert data/namelists/nl.CHATS7.1day -o chats7_from_namelist/
clm-ml run chats7_from_namelist
```

The Fortran name of each setting is given in the [settings reference](reference/settings.md).
How the JAX version differs from the Fortran is in
[How the model works](how_the_model_works.md#5-how-the-jax-version-differs-from-fortran).

## Next

- [Your own tower](your_own_tower.md): run the model with your own forcing.
- [Examples](examples.md): a tower run with figures, parameter sensitivity by gradients, a
  global run.
- [How the model works](how_the_model_works.md): the physics, the code and what differs from
  Fortran.
