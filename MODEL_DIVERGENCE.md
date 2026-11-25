# Model divergence from CLM-ML Fortran

This initial register covers departures identified in the audit. It includes default-path changes, optional JAX pathways, and branch-only developments. Complete the inventory before certifying the reference.

| Change and location | Scientific or numerical reason | Evidence and status |
| --- | --- | --- |
| Default Monin–Obukhov solve, `MLCanopyTurbulenceMod._GetObu`, introduced with the differentiable path in `7149932` | JAX uses a 25-iteration solve even outside differentiable mode. Original Fortran invokes the hybrid solver with +100 m and −100 m seeds, a 0.1 m convergence tolerance, and callback state writes. | Confirmed source departure. Restore original stopping and state semantics on the reference after capturing live Fortran cases. Validate stability regimes, convergence and atmospheric states. |
| Dry-canopy floor, `MLCanopyWaterMod.CanopyWettedFraction` | A positive floor of 1e−30 inside the fractional power avoids singular autodiff at zero. Active dry canopy returns a small nonzero wetted fraction in JAX and zero in Fortran. | Local `feature/reference-dry-canopy-fidelity` commit `6803748` restores exact zero and preserves the positive-water formula. Its production-path test failed before the fix and passed after it. At zero the implementation uses the dry branch; the wet-side derivative is singular. Reference review remains pending. |
| Differentiable canopy integration, `7149932`, `68de426`, `b277440` | Optional functional time stepping, `lax.scan`, and checkpointing support autodiff and reduce overhead. Fixed-iteration solvers and alternate integration paths can affect trajectory and sensitivities. | Historical diagnostics exist. Require forward agreement where intended, convergence, budgets, and derivative tests on matched configurations. |
| Differentiable internal leaf CO₂ solve, `9e35b8d`, `b196ac0`, `41dfea4` | Bisection and Newton refinement with implicit differentiation change root solving and define a fallback derivative for degenerate roots. | Verify residual and forward root, then test derivative accuracy and behavior near degeneracy. |
| Optional Medlyn slope parameter, `4fcaae8` | Exposes a parameter for calibration. | Require default recovery and validate altered stomatal conductance, assimilation and transpiration responses. |
| Differentiable diagnostics, `2ffaee8` | Populates canopy-integrated diagnostics in differentiable mode and bypasses host-side balance aborts. | Compare diagnostic states and explicitly evaluate conservation residuals. |
| CLUBB mixing length, `upstream/CLUBB` at `1037f91`, including smoothing at `d2934a1` | Adds alternate turbulence development and smoothing, including saturation mixing ratio. | Preserved as branch history. Validate stable and unstable cases, transitions and gradients before integration. |

## Translation corrections and unresolved differences

- `f9541ed` corrects Python soil texture indexing while keeping Fortran texture identifiers one-based. Verify through initialization across texture IDs and NetCDF inputs before backporting. `b183687` is the historical column-index repair on its side branch. The older local main already computes indices from `begc`.
- `12940e5` replaced stubs with real physics and NetCDF I/O and changed suction sign representation. Verify the initialization and downstream equations together before classifying all changes as equivalent.
- Historical CHATS7 trajectories show heat storage and turbulent flux differences. A run-length-dependent next-forcing clamp is a hypothesis. Reproduce both models at identical horizons, including the terminal step, and trace first disagreement. See the [validation report](docs/FORTRAN_PARITY.md).
