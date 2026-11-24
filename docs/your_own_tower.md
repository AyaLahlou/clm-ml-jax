# Your own tower

A tower run needs three things: meteorological forcing at the tower, the leaf and stem area
and soil water through the run, and a description of the tower (location, forcing height,
soil, vegetation). Start from the CHATS folder and change it:

```bash
clm-ml new my_tower --from chats7_1day
```

then edit `my_tower/domain.yaml` (the tower) and `my_tower/run.yaml` (dates and files), and
run `clm-ml check my_tower` until it is happy. Paths in the files are relative to the folder.

## 1. The tower: `domain.yaml`

```yaml
type: towers
towers:
  MY_TOWER:                      # any name; it appears in the output file names
    latitude: 38.49              # degrees north
    longitude: -121.84           # degrees east
    measurement_height: 23.0     # forcing (reference) height, m
    soil:
      texture:
        given_as: percentages    # or: class (with texture_class: silty clay loam, ...)
        sand_percent: 10.0
        clay_percent: 35.0
      organic_matter: 50.0       # kg m-3, top 0.5 m
      color_class: 15            # CLM soil colour class, 1 (light) to 20 (dark)
      bedrock_depth: 2.0         # m; null: no bedrock
    patches:
    - pft: broadleaf_deciduous_temperate_tree   # a PFT name from params.yaml
      weight: 1.0                # fraction of the tower's area; weights sum to 1
      canopy_height: 10.0        # m, or pft_default
      root_biomass: 500.0        # fine roots, g m-2
      area_profiles:             # beta distributions of leaf and stem area with height
        leaf_beta: [2.6, 1.3]
        stem_beta: [1.8, 1.3]
      canopy_layers: from_model_yaml
```

A tower can have several patches (for example trees and grass), each with its own PFT,
canopy and soil column; the tower's output then has one entry per patch, and a gridcell value
weighted by area. Several towers can run together in one experiment: list them all.

**Canopy layers.** By default the canopy air is divided into layers of 0.5 m (canopies taller
than 2 m) or 0.1 m, from the ground up to the forcing height, with the same spacing above
the canopy as within it (`model.yaml` `canopy_layers`). A short canopy under a tall tower then
gets very many layers (a 0.5 m grass under a 30 m tower: 300). Give such a patch explicit
counts instead:

```yaml
      canopy_layers: {within: 10, above: 10}
```

Bonan et al. (2021) found 5–10 layers within the canopy enough to reproduce the fluxes of a
finely resolved canopy at forest towers.

## 2. The forcing: `run.yaml`

```yaml
time:
  start: 2007-05-01T00:00
  length: 31 days                # or 1488 steps, or until 2007-06-01T00:00
forcing:
  meteorology:
    files: [forcing/{year}-{month:02d}.nc]   # one file, a list, or a pattern
    interval_seconds: 1800       # time between records
  clm_history:
    file: forcing/my_tower.clm2.h1.nc
    start: 2007-04-01T00:00      # date-time of the history file's first record
```

**Meteorology** is a netCDF file (or files split in time) in the format of CLM's tower
forcing, with dimensions `(time, lat, lon)` of size `(n, 1, 1)` and a `time` variable in
`days since <date>` with a constant interval:

| Variable | Units | |
|---|---|---|
| `TBOT` | K | air temperature at the forcing height |
| `RH` or `QBOT` | % or kg kg-1 | relative or specific humidity |
| `WIND` | m s-1 | wind speed |
| `PSRF` | Pa | surface pressure (101325 Pa if missing) |
| `FSDS` | W m-2 | incident shortwave radiation (or `FSDS_DIRECT` and `FSDS_DIFFUSE`) |
| `FLDS` | W m-2 | incident longwave radiation (computed from air temperature and humidity if missing) |
| `PRECTmms` | mm s-1 | precipitation |
| `ZBOT` | m | forcing height (optional; `domain.yaml` `measurement_height` otherwise) |

The CHATS files in `data/input_files/tower-forcing/CHATS7/` are an example.

**Leaf area, stem area and soil water** are prescribed. By default they come from a CLM
history file of a CLM run at the tower (one gridcell, at the forcing interval) with `ELAI`,
`ESAI`, `H2OSOI` (20 soil levels, CLM5's layer structure) and `TSOI` (the initial soil
temperature). Alternatively, give daily or monthly files (`vegetation: {source: file, files:
...}` with `LAI` and `SAI`; `soil_water: {source: file, files: ...}` with `H2OSOI`), which
are interpolated between the middles of their intervals. The initial state can also come from
a restart file of an earlier run (`initial_conditions: {source: restart, ...}`).

**The model step.** The model runs at 30 minutes by default, split into six 5-minute
sub-steps. If your forcing is hourly, set `interval_seconds: 3600`; the model step
(`model.yaml` `numerics.step_seconds`) must divide the forcing interval, and the forcing is
interpolated to it.

## 3. The plants: `params.yaml`

`params.yaml` holds every parameter of every PFT. Change the values of your tower's PFT there,
or replace the file with another parameter set from
[`src/clm_ml/compsets/params/`](repo:src/clm_ml/compsets/params):

| Parameter set | |
|---|---|
| `chats.yaml` | CLM5 with the walnut values of Bonan, Burns & Patton (2026, Table 2); what `chats7_1day` uses |
| `clm5_defaults.yaml` | CLM5 PFT parameters (the CLM-ml v2 defaults) |
| `clm6.yaml` | CTSM 6.0 parameters (what the 1° grid run uses) |

`clm-ml params ctsm60_params.nc -o params.yaml` writes a complete `params.yaml` from a CLM
parameter file. Every parameter is listed in the [settings reference](reference/settings.md).

## 4. Run

```bash
clm-ml check my_tower
clm-ml run my_tower
python examples/run_tower.py --folder my_tower     # the same, with mean fluxes and a figure
```

Most problems show up in `clm-ml check` (a missing file or variable, a date outside the
forcing). During a run, energy and water conservation are checked after each chunk; a
failure stops the run with a message naming the check.
