# Agentic Pipeline Analysis: Differentiating a Legacy Fortran ESM with Claude Code

> **Scope:** All CHANGELOG sessions (Sessions 1–46, April 1 – May 8 2026) that involve model
> debugging, differentiability fixes, or performance optimization of the CLM-ML-JAX codebase.
> Every finding is derived directly from CHANGELOG.md or session memory files — no estimates
> are stated as facts unless the primary source contains them.

---

## 1. Project Overview

| Metric | Value | Source |
|--------|-------|--------|
| Fortran modules mirrored | 28 | CHANGELOG session 33–34 paper table |
| Python source files | 99 | CHANGELOG session 33–34 |
| Total translated LOC | 31,661 | CHANGELOG session 33–34 |
| Multilayer canopy LOC | 15,419 | CHANGELOG session 33–34 |
| Test modules | 32 | CHANGELOG session 33–34 |
| Ralph loop sessions (translation) | 31 | CHANGELOG session 33–34 |
| Translation period | Dec 2025 – Apr 2026 (~5 months) | CHANGELOG |
| Differentiability debugging period | April 1 – May 8 2026 (~5 weeks) | CHANGELOG sessions 1–46 |
| Human supervision (stated) | ~2–4 hours/week | CHANGELOG session 33–34 paper table |

---

## 2. Session Timeline and Classification

### 2.1 Session Density by Week

| Week | Date range | Sessions | Dominant work type |
|------|-----------|----------|-------------------|
| W1 | Apr 1–3 | 1–11 | NaN gradient fixes + D→H sync elimination + XLA recompile fixes |
| W2 | Apr 6–8 | 12–16 | NaN continuation + zero-gradient root-cause isolation |
| W3 | Apr 9–10 | 17–28 | IFT fixes + lax.scan + parameter differentiability (6+ sessions per day on Apr 10) |
| W4 | Apr 11–15 | 29–31 | LE/H grad checks + Jacobian fixes + paper writing |
| W5 | Apr 20–24 | 33–38 | Paper rewrite + multi-param experiments + pbot NaN fix |
| W6 | Apr 27–29 | 39–43 | g1_MED confirmed + calibration jobs + equifinality |
| W7 | May 8 | 44–46 | Multi-step calibration + Adam fix + nighttime step fix |

**Note:** Sessions 22–28 all fall on April 10 — this reflects multiple rapid iteration loops within a single day.

### 2.2 SLURM Job Count (Differentiability Phase)

At least **47 distinct SLURM jobs** were submitted across sessions 17–46. Job IDs extracted from CHANGELOG:

| Session range | Job IDs cited | Count |
|---|---|---|
| Sessions 17–20 | 7314440, 7314582/83, 7314294, 7314377, 7314396, 7314421, 7314439, 7314322, 7314403, 7315181, 7315861 | 11 |
| Sessions 22–28 | 7328915, 7329012/52/181/322/437, 7342742/43/44, 7343159/434/825, 7344537/39/785, 7345152 | 16 |
| Sessions 30–34 | 7403949, 7447256, 7450291, 7527834/893/971, 7552426, 7577146/180, 7578601/655/886, 7579151 | 13 |
| Sessions 36–43 | 7589868, 7590268, 7677819/849, 7681126/127/249/907 | 8 |
| Sessions 45–46 | 7870919–922, 7870940, 7870948/49/952 | 8 |
| **Total** | | **≥ 56** |

**Job failure/retry rate:** 19 jobs either FAILED, TIMED OUT, CANCELLED, or were superseded by a corrected resubmission. That is **≥34% job failure rate** (19/56+).

---

## 3. Bug Taxonomy (Codified Types)

### 3.1 Type Codes

| Code | Bug Type | Root Mechanism |
|------|----------|----------------|
| **T1** | NaN Gradient — jnp.where both-branch | JAX differentiates both branches of `jnp.where`; inactive branch containing 1/0 or x**n at x=0 produces `0 × inf = NaN` |
| **T2** | Zero/Wrong Gradient — path broken | Parameter value not reaching the JAX trace (Python float, overwritten state, or JIT constant baking) |
| **T3** | XLA Recompilation | JIT cache miss due to Python closure capturing varying scalars or lack of `lru_cache` |
| **T4** | Memory / OOM | Graph too large for device (grad unrolling, CPU vmap unrolling, large tensors) |
| **T5** | Gradient Explosion | Jacobian accumulation through iterative solvers (secant/bisection/scan) |
| **T6** | Device-Host Sync | `np.asarray(jax_array)` or `float(jax_array)` inside hot loops → GPU→CPU copies |
| **T7** | Optimization Algorithm | Adam hyperparameter failure, step-index arithmetic, underdetermined system |
| **T8** | Crash / Compile Failure | XLA backend optimization pass bug; GPU contention; SLURM resource exhaustion |
| **T9** | Diagnostic Reliability | Finite-difference epsilon instability; spval contamination in loss; wrong timing barrier |

### 3.2 Bug Inventory (All Documented Bugs)

| # | Code | Module(s) | Description | Session first seen | Session fixed | Attempts |
|---|------|-----------|-------------|-------------------|--------------|----------|
| B1 | T8 | 9 modules | XLA `select_divide_fusion` compilation crash on `jax.grad` | 1 | 1 | 1 |
| B2 | T6 | MLCanopyFluxesMod | D→H syncs in sun/shade leaf merge (~4 round-trips/step) | 1 | 1 | 1 |
| B3 | T6 | MLCanopyFluxesMod | D→H syncs in patch hierarchy arrays | 1 | 1 | 1 |
| B4 | T1 | MLSolarRadiationMod | `dpai_safe`, `kb_ic`, `p1/p2`, `fs/fsha` jnp.where-over-denom (~8 patterns) | 1 | 1 | 1 |
| B5 | T1 | MLCanopyNitrogenProfileMod | `dpai_safe`, `fs/fsha` patterns (~3) | 1 | 1 | 1 |
| B6 | T1 | MLFluxProfileSolutionMod | `gs_gbv_denom`, `den_l`, `den_lf` patterns (~4) | 1 | 1 | 1 |
| B7 | T1 | MLLeafFluxesMod | `gleaf_denom`, `tleaf_denom` | 1 | 1 | 1 |
| B8 | T1 | MLLeafPhotosynthesisMod | `gs_safe`, `gbc_safe`, secant denom | 1 | 1 | 1 |
| B9 | T1 | MLPlantHydraulicsMod | `totevap_safe`, `lsc_safe` | 1 | 1 | 1 |
| B10 | T1 | MLCanopyWaterMod | `total_safe`, `dpai_safe` | 1 | 1 | 1 |
| B11 | T1 | MLCanopyTurbulenceMod | `obu_cur_safe`, `tvstar_safe`, secant denom (~12 patterns) | 1 | 1 | 1 |
| B12 | T1 | MLMathToolsMod | `a_safe`, `q_safe` in `quadratic()` | 1 | 1 | 1 |
| B13 | T1 | MLMathToolsMod | `tridiag` (bet) and `tridiag_2eq` (det) near-zero denominators | 2 | 2 | 1 |
| B14 | T1 | MLCanopyTurbulenceMod | `_phim/_phic`: sqrt(negative) — `jnp.abs` not enough | 2/5 | 5 | 1 |
| B15 | T6 | MLLeafPhotosynthesisMod | 22 D→H syncs in first+second loops | 2 | 2 | 1 |
| B16 | T6 | MLCanopyTurbulenceMod | 12 D→H syncs in `_GetObu` re-extraction | 2 | 2 | 1 |
| B17 | T1 | MLCanopyTurbulenceMod | `_obu_writeback_jax` — jnp.where denominator → inf in grad | 3 | 3 | 1 |
| B18 | T1 | MLCanopyTurbulenceMod | `_AerodynamicConductance_jax` × 3 jnp.where denominators | 3 | 3 | 1 |
| B19 | T1 | MLPlantHydraulicsMod | `SoilResistance` frozen soil `hk_v=0` | 3 | 3 | 1 |
| B20 | T4 | MLCanopyFluxesMod | `jax.grad` on 30-step Python for-loop → 103 GB OOM | 4 | 4 | 1 |
| B21 | T1 | MLSolarRadiationMod | `cos_zen = 0` division → inf gradient at solar zenith | 4 | 4 | 1 |
| B22 | T3 | MLLeafPhotosynthesisMod | No `lru_cache` on kernel factory → 60 XLA recompiles/step | 8/9 | 9 | 1 |
| B23 | T3 | MLCanopyTurbulenceMod | `_obu_fixed_iter` lambda closes over float kwargs → 90 recompiles/step | 10/11 | 11 | 1 |
| B24 | T1 | MLLeafBoundaryLayerMod | `re**0.5`, `re**0.8`, `gr**0.25` at zero wind | 12 | 12 | 1 |
| B25 | T1 | MLCanopyWaterMod | `(h2ocan/h2ocanmx)**0.67` base=0 | 12 | 12 | 1 |
| B26 | T1 | MLMathToolsMod | `sqrt(max(disc,0))` — 0 floor not 1e-30 | 12 | 12 | 1 |
| B27 | T1 | MLCanopyTurbulenceMod | `sqrt(max(...,0))` in `_GetBeta_jax` — both jnp.where branches | 12 | 12 | 1 |
| B28 | T1 | MLPlantHydraulicsMod | `1/soilr_v`, `1/nlayers_f`, `1/rld_v` unguarded | 12 | 12 | 1 |
| B29 | T2 | MLCanopyFluxesMod | Radiation overwrite: `__init__` overwrites forcing fields from atm2lnd_inst | 13 | 13 | 1 |
| B30 | T2 | MLCanopyFluxesMod | diff-mode skips `CanopyFluxesDiagnostics` → `gppveg_canopy` always stale | 13 | 13 | 1 |
| B31 | T2 | MLCanopyNitrogenProfileMod | `vcmax25_profile`/`dpai_profile` recomputed inside physics step → overwritten | 14 | 14/15 | 1 |
| B32 | T5 | MLLeafPhotosynthesisMod | WUE bisection `jnp.where` gradient → 15% GPP gradient error | 16/17 | 17 | **3** |
| B33 | T9 | diags/isolate_grad_path.py | `spval=1e36` contamination cancels FD signal at `jnp.sum` | 16 | 16 | 1 |
| B34 | T5 | MLCanopyTurbulenceMod | Obukhov secant solver: 25-iter accumulation → `9.95e+144` gradient | 24 | 24 | **2** |
| B35 | T2 | MLLeafPhotosynthesisMod | Stomatal params (`g1_MED`, `iota_SPA`, etc.) as `float()` in `lru_cache` key | 25 | 25 | 1 |
| B36 | T2 | MLpftconMod + physics modules | MLpftcon injection: module-local binding not updated by module-var replace | 26 | 26 | 1 |
| B37 | T2 | MLCanopyNitrogenProfileMod | `@jax.jit` bakes `MLpftcon.vcmaxpft` as XLA constant → f(+eps)=f(-eps) | 26 | 26 | **2** |
| B38 | T5 | MLLeafPhotosynthesisMod | `_ci_solver_scan` NaN in backward pass (Medlyn mode) | 33 | 33 | 1 |
| B39 | T2 | MLLeafPhotosynthesisMod | g1_MED IFT: wrong sign + 100× magnitude error after session-25 fix | 36 | 37 | **3** |
| B40 | T1 | MLLeafPhotosynthesisMod | `alpha_pbot` NaN: `_ko_sl=0`, `_gbc_sl=0`, `_lesat_sl=0` in inactive layers | 38 | 38/39 | 1 |
| B41 | T7 | diags/multipar_calibration | Multi-step backward JIT: 8.6h for T=8 steps (XLA full graph unroll) | 44 | 45 | **2** |
| B42 | T7 | diags/multipar_calibration | Nighttime step index: `[24 + k×192]` = UTC 11:30 = 04:30 PDT (dark) | 44 | 45 | 1 |
| B43 | T7 | diags/multipar_calibration | Adam β₂=0.999 stalls: phase-1 gradient scale locked in v accumulator | 43 | 43/44 | 1 |
| B44 | T7 | all diags calibration scripts | Equifinality: 3 outputs × 10 params → underdetermined, ‖Δθ‖≈0.75 | 42 | 45 | **2** |
| B45 | T9 | diags/precision_roofline.py | Wrong timing barrier: `jax.effects_barrier()` measures dispatch, not compute | 46 | 46 | 1 |
| B46 | T8 | SLURM GPU nodes | GPU contention → XLA core dump (`CompileExecutables()` crash) | 46 | 46 | 1 |
| B47 | T8 | diags/multipar_calibration_laxscan | Parallel agent race: laxscan copied file before nighttime fix applied | 46 | 46 | 1 |
| B48 | T4 | CPU vmap compilation | CPU XLA vmap N≥128 → LLVM VA space exhaustion (not RAM) | 33–34 | N/A (infeasible) | 3 |

**Total bugs documented: 48** (individual instances, some batched where multiple patterns fixed in same session)

---

## 4. Bug Frequency Analysis

### 4.1 Bug Count by Type

| Code | Type | Count | % of total |
|------|------|-------|------------|
| T1 | NaN Gradient (jnp.where) | **17** | 35.4% |
| T2 | Zero/Wrong Gradient (path broken) | **9** | 18.8% |
| T3 | XLA Recompilation | **2** | 4.2% |
| T4 | Memory / OOM | **3** | 6.3% |
| T5 | Gradient Explosion | **3** | 6.3% |
| T6 | Device-Host Sync | **3** (batched, ~90+ individual syncs) | 6.3% |
| T7 | Optimization Algorithm | **4** | 8.3% |
| T8 | Crash / Compile Failure | **3** | 6.3% |
| T9 | Diagnostic Reliability | **3** | 6.3% |
| **Total** | | **48** | 100% |

### 4.2 Bug Count by Module

| Module | Bugs (primary) | Dominant type |
|--------|---------------|---------------|
| MLCanopyTurbulenceMod | 8 | T1, T3, T5, T6 |
| MLLeafPhotosynthesisMod | 8 | T1, T2, T3, T5 |
| MLCanopyFluxesMod | 4 | T2, T4 |
| MLPlantHydraulicsMod | 3 | T1 |
| MLMathToolsMod | 3 | T1 |
| MLSolarRadiationMod | 3 | T1 |
| MLLeafBoundaryLayerMod | 2 | T1 |
| MLCanopyNitrogenProfileMod | 2 | T2 |
| MLCanopyWaterMod | 2 | T1 |
| MLFluxProfileSolutionMod | 1 | T1 |
| MLLeafFluxesMod | 1 | T1 |
| diags scripts | 6 | T7, T8, T9 |
| Multi-module (MLpftcon injection) | 2 | T2 |

### 4.3 Bug Discovery Rate Over Time

| Phase | Sessions | Bugs discovered | Rate (bugs/session) |
|-------|----------|----------------|---------------------|
| NaN blast (batch) | 1 | B1–B12 (12 fixes in one session) | 12.0 |
| NaN continuation | 2–5 | B13–B21 (9 bugs) | ~2.2 |
| Recompile/JIT | 6–11 | B22–B23 (2 major bugs) | ~0.3 |
| NaN residual | 12 | B24–B28 (5 bugs) | 5.0 |
| Zero-gradient | 13–16 | B29–B33 (5 bugs) | ~1.3 |
| IFT+lax.scan | 17–24 | B32–B34 (2–3 bugs) | ~0.4 |
| Param differentiability | 25–28 | B35–B37 (3 bugs) | 0.75 |
| Paper experiments | 29–39 | B38–B40 (3 bugs) | ~0.3 |
| Calibration | 40–46 | B41–B48 (8 bugs) | ~1.1 |

---

## 5. Failed Attempts Analysis

### 5.1 All Documented Failed Attempts

| # | Bug targeted | Failed approach | Reason for failure | Session |
|---|--------------|----------------|-------------------|---------|
| F1 | B34 (gradient explosion) | Outer `jax.jit` on `jax.grad` | OOM + select_divide_fusion crash | 1 |
| F2 | B34 (gradient explosion) | `jax.lax.stop_gradient` on denominators | Breaks gradient flow | 1 |
| F3 | B14 (JIT feasibility) | GridInfo JIT in non-diff mode | XLA CUDA OOM (32 GB V100) | 7 |
| F4 | B32 (WUE bisection) | `jax.lax.custom_root` IFT | JAX version incompatibility: `Partial` not scalar | 17 |
| F5 | B34 (Obukhov explosion) | `|df/dgs| > 1e-6` denominator guard | Guard never triggers (df=0.16, well-conditioned) | 19 |
| F6 | B36 (MLpftcon injection) | `MLpftconMod.MLpftcon = new_inst` alone | Physics modules hold direct ref to old object | 26 |
| F7 | B37 (vcmaxpft JIT) | `_set_pftcon` mutation for @jax.jit function | JIT bakes value at first trace; mutation ignored | 26 |
| F8 | B39 (g1_MED wrong sign) | IFT pattern from session 25 | Gradient +5.2 vs FD −743 (wrong sign, 100× off) | 36 |
| F9 | B34 (Obukhov) | `jacfwd` to bypass custom_vjp | JAX hard constraint: can't apply jvp to custom_vjp | 14 |
| F10 | B48 (CPU OOM) | 128G RAM node for CPU vmap N=512 | LLVM VA space, not RAM — 768G node also OOM'd | 33–34 |
| F11 | T7 (Adam calibration) | NM with maxiter=300 after 14h Adam run | SLURM time limit; NM killed at eval 32 of 300 | 33 |
| F12 | B41 (multi-step JIT) | 6h SLURM wall time for multi-step backward | Killed mid-compilation (needed 8.6h) | 42–43 |
| F13 | B45 (timing barrier) | `jax.effects_barrier()` for GPU timing | Waits for Python callbacks only, not GPU compute | 46 |
| F14 | B47 (laxscan race) | Parallel agent copy of calibration script | Copied before nighttime fix applied → GPP=0 | 46 |
| F15 | B44 (equifinality) | Single-step loss (3 outputs, 10 params) | Underdetermined: infinite equifinal solutions | 42 |
| F16 | T9 (FD reliability) | FD at eps=1e-4 for g1_MED | Ci-solver chaotic at small eps — solver branch flip | 36/39 |

**Total failed attempts: 16**

### 5.2 Attempts per Bug

| Category | # bugs | # requiring >1 attempt | Average attempts |
|----------|--------|------------------------|------------------|
| T1 (NaN gradient) | 17 | 0 | **1.0** |
| T2 (Zero/wrong gradient) | 9 | 2 (B37, B39) | **1.3** |
| T3 (Recompilation) | 2 | 0 | **1.0** |
| T4 (OOM) | 3 | 1 (B48) | **1.7** |
| T5 (Gradient explosion) | 3 | 2 (B32, B34) | **2.0** |
| T6 (D→H sync) | 3 | 0 | **1.0** |
| T7 (Algorithm) | 4 | 2 (B41, B44) | **1.5** |
| T8 (Crash) | 3 | 1 (B48 adjacent) | **1.3** |
| T9 (Diagnostic) | 3 | 1 (F16) | **1.3** |
| **Overall** | **48** | **9** | **1.33 attempts/bug** |

---

## 6. Implicit Function Theorem (IFT) Usage

A central differentiability strategy was the Newton-refinement IFT identity, applied to 4 iterative solvers:

| Solver | Bug | Session fixed | Pattern |
|--------|-----|--------------|---------|
| WUE stomatal bisection (`_bisect_gs_jax`) | B32 | 17 | `gs_ift = gs0 - f(gs0)/stop_grad(df/dgs)` |
| Obukhov length secant solver (`_GetObu`) | B34 | 24 | `obu_ift = obu0 - f(obu0)/stop_grad(df/dobu)` |
| Medlyn ci-solver scan (`_ci_solver_scan`) | B38 | 33 | `ci_ift = ci0 - F(ci0;θ)/stop_grad(∂F/∂ci)` |
| g1_MED ci-solver scan (same code) | B39 | 37 | explicit `g1_MED_jax` arg bypasses module-global |

**Why IFT was needed:** All four solvers converge via iterative bisection/secant methods. At convergence `f(x*) ≈ 0`, but JAX differentiates through every iteration. With N iterations, Jacobian accumulation gives `|J|^N` — catastrophically large at N=20–25.

---

## 7. Key Architecture Changes (Differentiability)

| Change | Sessions | Impact |
|--------|----------|--------|
| `jnp.maximum(x, eps)` guards — 35+ sites across 9 modules | 1–12 | Eliminates T1 NaN pattern universally |
| `jax.checkpoint` on physics step | 4 | Reduces backward memory from O(30×step_mem) to O(step_mem) |
| `lru_cache` on kernel factories | 9 | Eliminates 60 XLA recompiles/timestep |
| `_obu_body_pure` at module level | 11 | Eliminates 90 recompiles/timestep from Obukhov solver |
| Unified diff/non-diff code paths | 1–2 | Eliminates ~90+ D→H syncs per sub-step |
| `jax.lax.scan` over ML sub-steps | 22 | 77–818× speedup vs Python for-loop (Euler→RK4) |
| Explicit `vcmaxpft_jax` argument | 26 | Bypasses @jax.jit constant baking for Vcmax25 |
| `_set_pftcon` 3-module update pattern | 26 | Correct NamedTuple injection for iota_SPA, g1_MED |
| Explicit `g1_MED_jax` argument | 37 | Correct gradient sign and magnitude for Medlyn g1 |
| `bracket_ok` gate in IFT Newton step | 21 | Prevents IFT from extrapolating when no root exists |
| `jnp.maximum(inactive_denom, 1e-30)` — 3 sites in second loop | 38 | Fixes alpha_pbot NaN for inactive canopy layers |

---

## 8. Gradient Verification Progression

All gradients verified via JAX vs FD (central differences), reported as relative error.

| Parameter | Sessions to verify | Final rel. error | Status |
|-----------|-------------------|-----------------|--------|
| alpha_sw | 16–17 (fix), 17 (confirm), 28 (GPU) | 3.7e-7 | PASS |
| alpha_tref | 24 (IFT), 28 (GPU FD match) | 1.3e-4 | PASS |
| alpha_iota | 25 (fix), 27 (CPU verify), 28 (GPU) | 1.1e-6 | PASS |
| alpha_vcmax | 26 (fix), 27 (CPU+GPU) | 1.8e-8 | PASS |
| alpha_g1 | 25 (fix), 37 (sign fix), 39 (confirmed correct) | ~0.8% (at eps=0.1) | PASS |
| alpha_pbot | 38 (fix), 40 (confirmed) | PASS | PASS |
| dLE/dα, dH/dα (all 5 params) | 30 | All PASS | PASS |
| 7-param Jacobian (all columns non-zero) | 40 | Non-zero confirmed | PASS |

---

## 9. Performance Results

### 9.1 XLA Compile Time

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| LeafPhotosynthesis (steady-state/step) | 1942 s | 146 s | 13.2× |
| _obu_fixed_iter (steady-state/step) | 146 s | ~120 s | ~1.2× |
| lax.scan Euler (first compile) | 290 s | 290 s (cached 0s) | ∞ after 1st run |
| Full forward per-step (GPU, A40) | ~100 s | ~38 ms | ~2600× |

### 9.2 lax.scan Speedups (job 7344537, A100)

| Mode | diff (lax.scan) | non-diff (Python loop) | Speedup |
|------|----------------|----------------------|---------|
| Euler (1 sub-step) | 38.1 ms/step | 2,941 ms/step | **77×** |
| RK4 (6×4 stages) | 37.1 ms/step | 30,391 ms/step | **818×** |

### 9.3 Multi-site vmap Speedup (job 7342743, A100)

| N sites | Speedup | ms/site/step |
|---------|---------|-------------|
| 1 | 0.92× | 722 ms |
| 8 | 1.75× | 374 ms |
| 32 | 1.89× | 354 ms |

---

## 10. Calibration Experiment Results

### 10.1 AD vs FD Efficiency (confirmed)

From job 7681249 (single-step, p=10, job 7681249):

| Method | Final loss | ‖Δθ‖₂ | Time |
|--------|-----------|-------|------|
| Adam + AD | 1.06e-12 | 0.749 | 79 s |
| L-BFGS-B + AD | 1.03e-20 | 0.913 | 1.9 s |
| L-BFGS-B + FD | 1.24e-15 | 0.913 | 6.2 s |
| Nelder-Mead | 1.09e-14 | 0.478 | 19.8 s |

**AD speedup at p=10: 7.3× faster than FD** (T_backward/T_forward = 2.73; breakeven at p=1.4).

### 10.2 Minimal Calibration (paper-ready, p=3)

From job 7870922:

| Method | Loss | ‖Δθ‖₂ | Evals | Time |
|--------|------|-------|-------|------|
| Adam + AD | 3.60e-02 | 0.679 | 100 | 7.5 s |
| L-BFGS-B + AD | 5.38e-19 | **0.000** | 49 | 3.1 s |
| Nelder-Mead | 8.62e-16 | **0.000** | 414 | 8.8 s |

L-BFGS-B+AD recovers all 3 parameters exactly in 3.1s (49 gradient evaluations).

---

## 11. Ralph Loop Analysis

### 11.1 What Is Known

- The paper explicitly states **31 Ralph loop sessions** (Dec 2025 – Apr 2026) for translation + repair
- CHANGELOG sessions 1–46 document the differentiability work (April–May 2026)
- Sessions 22–28 all occur on April 10 (6 sessions in one day) suggesting rapid loop iterations
- Sessions 1–11 on April 1–3 (11 sessions in 3 days) similarly dense

### 11.2 Loop Duration Indicators

| Date | Sessions | Work density |
|------|----------|-------------|
| Apr 1 | 1 | 1 session (large batch NaN fixes: 12 bugs in 1 session) |
| Apr 2–3 | 2–11 | 10 sessions (NaN fixes + benchmarks) |
| Apr 8 | 12–16 | 5 sessions (zero-gradient isolation) |
| Apr 9 | 17–23 | 7 sessions (IFT + lax.scan) |
| Apr 10 | 24–28 | 5 sessions (Obukhov IFT + param diff + GPU confirm) |

### 11.3 When Loops Did NOT Resolve Issues

| Issue | Outcome |
|-------|---------|
| B48: CPU vmap XLA OOM | Never resolved; confirmed infeasible after 3 job attempts (128G node, 768G node, 72h run) |
| B44: Equifinality | Not a code bug — requires more timesteps or regularization; addressed with Tikhonov |
| B41: Multi-step JIT 8.6h | Workaround found (lax.scan) but root cause (XLA full graph unroll) is a JAX limitation |
| g1_MED gradient (B39) | Required 3 sessions across 2 weeks to isolate correct fix |
| Fortran timing baseline | Multiple SLURM job failures (nvfortran missing → gfortran → arithmetic overflow) |

---

## 12. Human Oversight Estimates

Based on the CHANGELOG's stated "~2–4 hours/week" supervision rate:

| Period | Duration | Estimated oversight |
|--------|----------|--------------------| 
| Translation phase (Dec 2025 – Mar 2026, 31 Ralph loops) | ~16 weeks | ~32–64 hours |
| Differentiability phase (Apr 1 – May 8, sessions 1–46) | ~5 weeks | ~10–20 hours |
| **Total project** | **~21 weeks** | **~42–84 hours** |

**Nature of oversight (observed from CHANGELOG):**
- Reviewing SLURM job outputs and confirming which direction to pursue next
- Deciding between multiple fix strategies when agents proposed alternatives (e.g., Option A/B/C in session 6)
- Authorizing paper section rewrites and verifying scientific framing
- Providing domain knowledge (e.g., confirming FD epsilon should be eps=0.1 for g1_MED)

---

## 13. Summary Statistics for Conference Presentation

```
Total CLM-ML-JAX differentiability debugging sessions:      46
Total SLURM jobs submitted:                               ≥56
SLURM job failure/retry rate:                            ≥34%
Total distinct bugs documented:                            48
Bugs requiring >1 debugging attempt:                        9 (18.8%)
Total failed debug attempts:                               16
Average debugging attempts per bug:                       1.33
Most common bug type:         T1 (NaN gradient) — 35.4% of all bugs
Most buggy module:     MLLeafPhotosynthesisMod + MLCanopyTurbulenceMod (8 bugs each)
Fastest bug class to fix:            T1 (always 1 attempt — pattern is mechanical)
Hardest bug class to fix:         T5 (gradient explosion) — avg 2.0 attempts
Time from first jax.grad to all 5 params verified:    ~10 days (Apr 1–10)
Time to full 7-param Jacobian:                        ~27 days (Apr 1–27)
lax.scan speedup over Python loop:              77× (Euler) to 818× (RK4)
AD speedup over FD at p=10 parameters:                      7.3×
L-BFGS-B+AD parameter recovery (p=3):          exact (‖Δθ‖=0.000) in 3.1s
Human oversight (estimated):                ~42–84 hours total (2–4 hr/wk)
```

---

## 14. Key Findings for ML Community

1. **NaN gradients dominate early work (35.4% of bugs):** All follow one root pattern — JAX evaluates both branches of `jnp.where`; a closed-form fix (`jnp.maximum(x, eps)`) resolves every instance mechanically. This suggests that automated pre-commit NaN gradient checks (scanning for `jnp.where` over potential denominators) could proactively prevent this entire class.

2. **Parameter injection is the hardest problem class (requires deep JAX knowledge):** Three distinct failure modes were encountered: (a) Python `float()` casting breaks autodiff tape silently, (b) module-level NamedTuple replacement doesn't propagate to imported local bindings, (c) `@jax.jit`-decorated functions bake module globals as XLA constants. Each required understanding JAX's tracing model at a deep level. Average: 1.6 attempts for T2 bugs vs 1.0 for T1 bugs.

3. **Iterative solvers require IFT — not AD-through-the-loop:** Three physical solvers (WUE bisection, Obukhov secant, ci-scan) accumulated Jacobians catastrophically. The Newton-refinement IFT identity (`x_ift = stop_grad(x*) - f(x*;θ)/stop_grad(df/dx)`) resolved all three cleanly with minimal code change and zero impact on forward accuracy.

4. **`lax.scan` is non-optional for differentiable ESM loops:** 77–818× speedups by replacing a Python `for` loop with `jax.lax.scan`. The non-diff Python loop was slower by the ratio of Python overhead × N sub-steps × RK stages. This is a universal finding for any JAX port of time-stepped models.

5. **GPU compile times dominate clock time, not physics:** A single XLA compilation (280–1225s per parameter) vastly exceeds actual per-step compute time (38ms). The persistent JAX compile cache (`JAX_COMPILATION_CACHE_DIR`) is essential for multi-job experimental campaigns.

6. **AD advantage over FD materializes at p≥2 parameters:** Breakeven is p=1.4 (T_backward/T_forward=2.73). At p=10, AD is 7.3× faster. This makes gradient-based calibration economically compelling for any ESM with more than 2 tunable parameters.

7. **Agentic parallelism introduced new failure modes:** Session 46 documents a parallel-agent race condition: one agent copied a source file before another agent finished fixing it. This required a resubmission and 24h wall-clock delay. Mutex/lock mechanisms for file-level parallelism are absent in current Ralph loop infrastructure.

8. **Equifinality is a physics problem, not a code bug:** Single-step calibration with 3 scalar outputs × 10 parameters is fundamentally underdetermined. The agent pipeline correctly identified this and proposed Tikhonov regularization — but required 3 sessions (42, 43, 45) and multiple SLURM jobs to arrive at the solution.
