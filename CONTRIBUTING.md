# Contributing to CLM-ML-JAX

Start a `feature/<topic>` branch from the intended target. Open a pull request with the source equation or scientific rationale, affected variables and options, and validation commands and results.

- Target `reference/fortran-parity` for a demonstrated translation correction, a reproducibility change, or a test. Cite the pinned Fortran file and routine, add a test that exposes the discrepancy, and preserve the original formulation. Fixture changes require reproducible regeneration and review of the changed values.
- Target `main` for model changes. Keep the Fortran tests for the unchanged pathways, and update [MODEL_DIVERGENCE.md](MODEL_DIVERGENCE.md) with the change, its justification, affected modes, and evidence. Validate altered equations against an analytic, observational, or independently computed reference. Check relevant energy, water, or carbon budgets and include timestep refinement or solver convergence where applicable.
- For differentiable pathways, compare directional derivatives to centered finite differences over a range of step sizes in CPU float64. Record parameters, scales, tolerances, and treatment of nonsmooth boundaries. Include a forward comparison under identical forcing and initial states.
- Move validated reference fixes to `main` by reviewed merge or cherry-pick as appropriate. Preserve commit history and attribution. Record both commits when cherry-picking.

## Repository rules

After approval, configure rulesets on `main` and `reference/fortran-parity`. Require pull requests, independent review, successful track-specific CI checks, and resolved review conversations. Block force pushes and deletion. Require a scientific reviewer for reference equations, input fixtures, tolerances, and release scripts. Set the required checks from actual workflow runs and confirm both allowed and rejected PR scenarios before relying on them.

## Releases

Use semantic versioning. Bump major for incompatible interfaces or changes to default scientific behavior that invalidate expected results, minor for backward-compatible capabilities, and patch for backward-compatible corrections. Document scientific and numerical effects in every release. Use `v1.0.0-parity` for the first independently verified reference, per the publication plan. Treat that tag and its archived files as permanent. Corrections require a new version and new archive record.

The [validation report](docs/FORTRAN_PARITY.md) lists the release criteria. After approval, create an annotated tag on the verified reference commit, enable immutable GitHub releases, protect release tags from updates and deletion, and publish the release with checksummed assets. Connect the canonical repository to Zenodo before publication. Confirm the deposit includes the reference data and reproducibility assets, record the version-specific DOI, and verify the archive against the release manifest. Set citation metadata from the author-approved author list and archival record.
