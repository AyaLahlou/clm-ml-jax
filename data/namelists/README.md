# CLM-ml namelists

The Fortran CLM-ml v2 offline driver (`prgm.exe < nl.CHATS7.05.2007`) reads a `&clmML_inparm`
namelist. These files are those namelists, kept as test and example data and as the record of
the Fortran runs the model is compared with. The JAX model does not read them directly: convert
one to an experiment folder (run.yaml, domain.yaml, model.yaml, params.yaml) and run that:

```bash
clm-ml convert data/namelists/nl.CHATS7.1day -o chats7/
clm-ml check chats7
clm-ml run chats7
```

The converter reads the paths as the Fortran driver did (`../input_files/...` relative to this
directory, i.e. `data/input_files/`), takes the tower from its compset (CHATS7), the parameters
from `pftcon_val` (1: the `chats` compset, 0: `clm5_defaults`) and refuses settings it cannot
translate (e.g. `clm_phys = 'CLM4_5'`). The ready-made equivalents of the CHATS7 runs are in
`configs/` (`configs/chats7_1day`, `configs/chats7_may2007_fortran`).

| Namelist | Run |
|---|---|
| `nl.CHATS7.1day` | 1 May 2007, one day |
| `nl.CHATS7.50steps` | 1 May 2007, 50 steps (the regression reference) |
| `nl.CHATS7.05.2007` | May 2007 (the Fortran comparison) |

The former `clm-ml-offline` driver was retired in Phase 7.
