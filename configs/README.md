# Configurations

Ready-to-run experiment folders, one per compset, written by `clm-ml new` (decision 91;
`docs/design/experiment_folders.md`). Each folder holds the four files of an experiment, every
setting listed with its explanation:

- `run.yaml`: when and with what: dates, forcing and land-data files, initial conditions,
  output streams, chunks and restarts;
- `domain.yaml`: where: towers (location, forcing height, soil, patches) or a grid;
- `model.yaml`: how: physics schemes and numerics (a change recompiles the model);
- `params.yaml`: which plants: every parameter of every PFT.

| Folder | What it runs |
|---|---|
| `chats7_1day/` | One day (1 May 2007) at the CHATS walnut orchard, the CLM-ml v2 test case (Bonan, Burns & Patton 2026). The quickest complete run. |
| `chats7_may2007_fortran/` | May 2007 at CHATS with the Fortran acclimation temperature and the legacy `.out` files: the run compared with Fortran CLM-ml v2 (`tests/test_fortran_python_comparison.py`). |
| `grid_1deg_2000/` | A 1° grid run of 2000 (CRUJRA forcing, CLM6 parameters, CTSM 5.4 soil moisture) globally; inputs on NCAR storage. |

```bash
clm-ml check configs/chats7_1day     # validate and summarize, no run
clm-ml run configs/chats7_1day       # output in configs/chats7_1day/output/
clm-ml run configs/chats7_1day --set 'run.time.length=3 days' --set model.processes.stomatal_conductance.scheme=medlyn
clm-ml new my_site --from chats7_1day   # your own copy to edit
```

Do not edit these folders by hand: they are the compsets in `src/clm_ml/compsets/` written
out, and `tests/clm_ml/test_configs.py` checks that they match (regenerate one with
`clm-ml new configs/NAME --from NAME --force`). To add a tower or patches, edit `domain.yaml`
of your own folder; for your own parameters, edit its `params.yaml`.
