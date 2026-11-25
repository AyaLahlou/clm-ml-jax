# Fortran provenance and numerical parity audit

## Repository and candidate

The audit traversed 311 reachable commits across local and cached GitHub refs and inspected test suites, fixtures, workflows, and translation notes. The live GitHub repositories had no tags or releases. The configured origin (AryaAvery) publishes `main` and `haoqi-biogeophy--main`; the AyaLahlou repository publishes eight branches spanning main, development, differentiable physics, CLUBB, packaging, profiling, and diagnostics. The latter main is a 236-commit descendant of the configured origin's main, which remains at `f5e606c8c751412edb0e7839c5a39f7c4b246bda`. The connected GitHub account has write/admin permissions on AyaLahlou and read access on AryaAvery. Select the canonical destination before any deployment.

| Revision | Commit |
| --- | --- |
| Fortran CLM-ml v1, gitlink in the configured origin's main | [`5c8d01dd2062430d09343d240f4ba07f320dff8e`](https://github.com/gbonan/CLM-ml_v1/commit/5c8d01dd2062430d09343d240f4ba07f320dff8e) |
| Original Bonan CLM-ml v2 CHATS, merge base of the instrumented fork | [`8d1cc40f210c82b2be9f2ba80c10f3325270b928`](https://github.com/gbonan/CLM-ml_v2.CHATS/commit/8d1cc40f210c82b2be9f2ba80c10f3325270b928) |
| Fortran test fork pinned by historical JAX candidate | [`9f7e9caadbd6e8584e0c8fc9bf58536869b5480c`](https://github.com/AyaLahlou/CLM-ml_v2.CHATS/commit/9f7e9caadbd6e8584e0c8fc9bf58536869b5480c) |
| Fortran fork revision carrying the actual archived JSON fixture bytes and generating harness | [`213e2cba3cded1fd4149a56f2a940dc5b333273e`](https://github.com/AyaLahlou/CLM-ml_v2.CHATS/commit/213e2cba3cded1fd4149a56f2a940dc5b333273e) |
| Historical JAX reference starting point | `7e7da4716ff5caf20c6dc212d0a31318cfd0f961` |
| Later JAX replay snapshot and source of vendored fixture bytes | `fe4a5563fec7eacaba86537e82157f866917dfd8` |

The provisional local `reference/fortran-parity` points to `7e7da47`, the last revision with the v2 gitlink and corrected golden comparisons before submodule removal. This is a candidate for fidelity repairs. Earlier local main pins v1 and contains incomplete translation paths. Later upstream main carries valuable scientific developments and should remain part of the active development track. Its history includes a removed uppercase `Paper/jaxes_paper` gitlink. The active lower-case `paper/` manuscript directory is ignored and stays local.

All 26 archived JAX JSON blob IDs match the Fortran fork at `213e2cb`. The older gitlink has 15 JSON fixtures with different names. The intervening change updates harness, Makefiles and fixtures. The JSONs record executable names and generation time, with no build-log linkage. The newer committed test Makefile names NVIDIA `nvfortran` with `-C -Minform=inform -Ktrap=fp` and links NetCDF Fortran and BLAS. The older revision adds a commented GNU alternative with `-O2 -fno-range-check -ffree-line-length-none`. Actual compiler, library versions, invocation and executable digests for the archived values remain unknown.

## Component evidence

Both JAX snapshots passed 160 archived golden cases on CPU with JAX float64 enabled. The historical candidate used the identical later fixture archive in its expected submodule directory. The observed runtime used Python 3.14.9, JAX/JAXlib 0.7.1, NumPy 2.5.4, SciPy 1.18.1 and netCDF4 1.7.5.1.post1. The CPU CI dependency file specifies a separate Python 3.12.12 Linux environment, with Linux wheel resolution checked. A live Fortran build was unavailable in this workspace.

The replay compares 306 scalars under `|JAX − Fortran| ≤ 1e−15 + 1e−9 × |Fortran|`. The largest absolute difference was 3.9961e−10 in leaf boundary-layer conductance `gbv` (reference 1.59625398). The largest fraction of the allowed tolerance was 85.76% in `GetPsiRSL.psim2` (absolute 2.0510e−10, relative 8.5760e−10). A log-gamma case had 100% relative error at a reference magnitude of 4.44e−16, where its absolute difference was 4.44e−16.

The full canopy nitrogen-profile fixture has five uncollected cases. Wetted fraction and nitrogen scale adapters duplicate formulas outside the production calculation. Boundary layer and leaf heat capacity checks use private single-layer kernels. Runge–Kutta checks default coefficients. Canopy/soil coupling, multilayer water and energy balances, initialization, full photosynthesis, and longer runs require independent coverage. The default turbulence solve already departs from the original Fortran algorithm. Local feature commit `6803748` repairs exact dry-canopy wetted fraction and passes 161 cases with the archived fixtures, including a production-path regression that fails at the historical candidate. See the [divergence register](../MODEL_DIVERGENCE.md).

The older local main's ordinary suite yielded **2,286 passed, 704 failed, 4 skipped and 14 errors** under the available Python 3.14.9 environment when run with collection errors allowed. This is an environment-specific regression baseline. It includes traced Python conditionals in the time manager, fixture and API issues, missing dependencies and warning-policy failures. The scoped CI regression selection passed 137 cases locally.

## CHATS7 archived trajectory evidence

The one-day JAX archive was compared with the first 48 steps of the 31-day Fortran archive. The 31-day pair has 1,488 steps in each archive. The comparator checks file dimensions and keyed timestamps and heights, aligns untimestamped `aux` and `fsun` positionally, and excludes shared −999 missing values. The original upstream test used a stale root path, could truncate or drop rows, misinterpreted solar zenith as time, and could skip incomplete comparisons. Its 2.5% normalized MAE and 0.9999 correlation thresholds give no per-variable pointwise bound.

| CHATS7 output | One-day prefix max absolute | One-day worst column MAE / peak | 31-day max absolute | 31-day worst column MAE / peak |
| --- | ---: | ---: | ---: | ---: |
| Flux | 4.953 | 0.5190% | 7.468 | 0.02979% |
| Auxiliary | 0.011 | 0.2017% | 0.101 | 0.01552% |
| Sunlit canopy | 2.449 | 0.8278% | 8.149 | 0.02328% |
| Soil temperature output | 0.005 | 0.000423% | 0.008 | 0.0000634% |
| State profile | 7.856 | 0.5426% | 50.753 | 0.00691% |
| Flux profile | 3.206 | 0.3210% | 7.556 | 0.01367% |

Each row aggregates maxima across columns with potentially different physical units. MAE / peak divides the column's MAE by its peak reference magnitude with a 1e−6 floor. From the flux writer's column mapping, one-day prefix max absolute / RMSE in W m⁻² was 4.953 / 0.742 for canopy-air heat storage, 2.303 / 0.936 for sensible heat, and 3.206 / 1.133 for latent heat. The corresponding monthly errors were 1.402 / 0.063, 7.468 / 0.458 and 6.769 / 0.432. A historical report mislabeled some columns. The one-day run-length difference is confounded with build and configuration provenance. These data cannot certify a parity threshold.

A fresh CPU run of JAX `fe4a556` completed all 48 steps in 262 s. Against the archived Fortran month prefix, max absolute / RMSE values in W m⁻² were 4.950 / 0.741 for canopy-air heat storage, 2.284 / 0.965 for sensible heat, and 3.211 / 1.152 for latent heat. The fresh trajectory also differed from the archived one-day JAX trajectory by up to 1.260 W m⁻² in sensible heat. The driver emitted a float64-to-float32 scatter warning with x64 enabled.

## Reproduce the archived evidence

Create a CPU environment with CPython 3.12.12, install [pinned validation dependencies](../requirements/validation-cpu.txt), and run from a complete repository history with the audited commits available.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements/validation-cpu.txt
scripts/reproduce-parity-candidate.sh "$PWD/.venv/bin/python"
```

For the trajectory tables, extract the fixture directory and both archived JAX output directories. Run the comparator for each JAX archive, using `--jax-steps 48` for `1day` and `--jax-steps 1488` for `31days`. Untimestamped outputs require equal ordering and lack independent row keys.

```sh
archive=$(mktemp -d)
git archive fe4a5563fec7eacaba86537e82157f866917dfd8 \
  tests/fortran_output_files \
  src/output_files/JAX_outputs_05_2007_1day \
  src/output_files/JAX_outputs_05_2007_31days | tar -xf - -C "$archive"
.venv/bin/python scripts/compare-chats7-archives.py \
  "$archive/tests/fortran_output_files" \
  "$archive/src/output_files/JAX_outputs_05_2007_31days" \
  --reference-steps 1488 --jax-steps 1488
```

For a fresh JAX run, extract the entire `fe4a556` tree into a separate scratch directory. In `src/offline_executable` of that tree, run the driver with `PYTHONPATH=.. JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m offline_executable.main nl.CHATS7.1day`. This run also uses `f90nml` (version 1.5.0 locally). Preserve the output and log from the scratch tree and compare the generated `src/output_files` against the archived reference prefix.

## Publication gates

1. Pin Fortran compiler and flags, BLAS and NetCDF libraries, JAX and Python, CPU architecture, and fixture/input checksums. Archive the environment with a container digest or platform-specific lockfile. In clean Fortran checkouts regenerate fixtures using the fork's `offline_executable/Makefile`, `tests/Makefile` and `tests/python/generate_golden.py`. Cross-check instrumented production runs against Bonan's original source.
2. Close the known production discrepancies. Build production-entrypoint comparisons, including canopy nitrogen, dry interception, hybrid turbulence, and precision at the float32 scatter observed with x64 enabled. Translate original solver semantics where source and tests isolate a discrepancy. Document every intended departure and validate its own formulation.
3. Run identical forcing, initial states, timesteps and rounding in both implementations, with matched single-step, daily, monthly and longer horizons. Include stable/unstable conditions, day/night, dry/wet canopies, soil moisture extremes, PFT and site variation, and relevant supported solver switches. Compare all time–layer keys, finiteness, valid-value masks, and per-variable absolute and relative error. Set physically justified tolerances and near-zero scales before evaluation. Validate conservation and restart behavior in supported configurations.
4. Have an independent reviewer rerun both implementations from clean checkouts and audit the first discrepancy, cases, thresholds and unsupported settings. Present evidence before marking GPU, float32, differentiable pathways, CLUBB, alternate integrators, or coupled configurations as supported by the parity release.
5. Add the workflow and verification assets to the reference workstream, confirm hosted CI and rulesets, and assemble the input/output and publication-figure reproduction bundle. Review data redistribution rights, metadata, checksums and author list. Seek approval before pushing, tagging `v1.0.0-parity`, and publishing an immutable GitHub and Zenodo release. Verify the versioned DOI and asset hashes.

**Decision log.** Preserve both model histories and keep the reference branch provisional. Use the CHATS v2 lineage for reference validation, track the three Fortran provenance identities, and treat archived component and trajectory results as evidence with different scopes. No publication release or DOI has been created.
