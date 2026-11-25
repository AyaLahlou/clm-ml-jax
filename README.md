# CLM-ML-JAX

A JAX implementation of the Community Land Model with Multi-Layer Canopy (CLM-ML).

| Track | Purpose | Status |
| --- | --- | --- |
| `reference/fortran-parity` | Preserve the equations, algorithms, and parameterizations of a pinned Fortran configuration. | Local candidate under validation. |
| `main` | Develop the JAX model, including differentiable physics, ML integration, and numerical optimization. | Existing development history is preserved. |
| `feature/*` | Review and validate individual changes against their intended track. | Branch from the pull request target. |

The two audited histories use different Fortran baselines. See the [provenance and validation report](docs/FORTRAN_PARITY.md) for exact revisions, measured errors, and the remaining publication gates. Full model parity has not been established.

Run the archived CPU component comparisons with `scripts/reproduce-parity-candidate.sh`. The [validation report](docs/FORTRAN_PARITY.md) describes the environment and scope of this run. See the [contribution guide](CONTRIBUTING.md) for branch and test policy, and the [divergence register](MODEL_DIVERGENCE.md) for changes that require separate scientific validation.
