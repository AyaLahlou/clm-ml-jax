# Bug Taxonomy Reference Card — CLM-ML-JAX Differentiability Work

> Source: CHANGELOG.md sessions 1–46 (April 1 – May 8 2026).
> For conference presentation use. All numbers traceable to CHANGELOG.

---

## Bug Type Codes (9 categories)

| Code | Name | One-line definition |
|------|------|---------------------|
| T1 | NaN Gradient (jnp.where) | JAX differentiates both branches; inactive branch with 0/0 or x^n\|_{x=0} gives `0×inf=NaN` |
| T2 | Zero/Wrong Gradient (path broken) | Parameter value never reaches JAX trace (Python cast, overwritten state, JIT constant) |
| T3 | XLA Recompilation | JIT cache miss per-call due to unstable closure or missing `lru_cache` |
| T4 | Memory / OOM | Trace graph too large for device (gradient unrolling, CPU vmap, large tensor) |
| T5 | Gradient Explosion | Jacobian accumulation through N-iteration solver: \|J\|^N → ∞ |
| T6 | Device–Host Sync | `np.asarray(jax_arr)` or `float(jax_arr)` inside hot loops |
| T7 | Optimization Algorithm | Adam hyperparameters, step-index arithmetic, underdetermined system |
| T8 | Crash / Compile Failure | XLA backend bug, GPU contention, parallel agent race |
| T9 | Diagnostic Reliability | FD epsilon instability, spval contamination, wrong timing barrier |

---

## Counts

| Code | Bugs | % | Avg attempts to fix | Always 1-shot? |
|------|------|---|---------------------|---------------|
| T1 | 17 | 35.4% | **1.0** | Yes |
| T2 | 9 | 18.8% | 1.3 | No (2 multi-attempt) |
| T3 | 2 | 4.2% | 1.0 | Yes |
| T4 | 3 | 6.3% | 1.7 | No |
| T5 | 3 | 6.3% | **2.0** | No |
| T6 | 3 | 6.3% | 1.0 | Yes |
| T7 | 4 | 8.3% | 1.5 | No |
| T8 | 3 | 6.3% | 1.3 | No |
| T9 | 3 | 6.3% | 1.3 | No |
| **All** | **48** | 100% | **1.33** | — |

- Total failed attempts: **16** out of 64 total attempts
- 18.8% of bugs required more than 1 debugging attempt
- Hardest class: T5 (Gradient Explosion) — avg 2.0 attempts; requires IFT insight

---

## T1 Fix Template (mechanical, always 1-shot)

```python
# Any jnp.where over a denominator:
# WRONG: jnp.where(cond, a / x, fallback)  →  inactive branch grad: -a/x² = inf at x=0
# RIGHT:
x_safe = jnp.maximum(x, 1e-30)                 # non-negative quantities
x_safe = jnp.sign(x) * jnp.maximum(jnp.abs(x), 1e-30)  # signed quantities
result = a / x_safe                              # safe everywhere
```

Applied at **35+ sites** across 9 modules. Every T1 bug follows this same pattern.

## T5 Fix Template (requires IFT insight, avg 2.0 attempts)

```python
# Any iterative solver via lax.scan / lax.fori_loop:
# WRONG: differentiate through all N iterations → |J|^N catastrophic
# RIGHT: Newton-refinement IFT
x0 = jax.lax.stop_gradient(solve(f, theta))   # bisect/secant forward result
f0 = f(x0, theta)                              # JAX traces through theta here
df = jax.lax.stop_gradient(FD(f, x0))         # denominator frozen
x_ift = x0 - f0 / df                          # IFT: dx/dtheta = -(∂f/∂theta)/(∂f/∂x)
```

Applied to: WUE bisection (B32), Obukhov secant (B34), Medlyn ci-scan (B38).

## T2 Sub-patterns (requires JAX tracing knowledge)

| Sub-type | Root cause | Fix |
|----------|-----------|-----|
| T2a: Python cast | `float(MLpftcon.iota)` in `lru_cache` key → JAX sees constant | Pass as JAX broadcast scalar (`in_axes=None`) |
| T2b: State overwrite | Physics step overwrites scaled mlcanopy_inst fields | Scale upstream forcing; pass as explicit traced arg |
| T2c: JIT constant | `@jax.jit` function captures module global as XLA constant | Add explicit `param_jax=None` non-static arg |
| T2d: Local binding | `from module import X` holds reference to original NamedTuple | Update ALL local bindings, not just module variable |

T2c and T2d together required **3 separate sessions** (25, 26 part 1, 26 part 2) and 2 failed attempts.

---

## Module Bug Density

```
MLLeafPhotosynthesisMod:   ████████ 8 bugs  (most diverse: T1,T2,T3,T5)
MLCanopyTurbulenceMod:     ████████ 8 bugs  (T1,T3,T5,T6)
MLCanopyFluxesMod:         ████     4 bugs  (T2,T4)
MLMathToolsMod:            ███      3 bugs  (T1)
MLSolarRadiationMod:       ███      3 bugs  (T1)
MLPlantHydraulicsMod:      ███      3 bugs  (T1)
MLCanopyNitrogenProfileMod:██       2 bugs  (T2)
MLCanopyWaterMod:          ██       2 bugs  (T1)
MLLeafBoundaryLayerMod:    ██       2 bugs  (T1)
diags scripts:             ██████   6 bugs  (T7,T8,T9)
```

---

## Timeline Condensed

```
Apr 1        → First jax.grad attempt: crash (B1/T8) + 12 NaN bugs (B2-B12/T1) fixed
Apr 2–3      → 9 more NaN bugs (B13-B21/T1); 2 recompile bugs (B22-B23/T3); OOM fix (B20/T4)
Apr 6        → 5 fractional-power NaN bugs (B24-B28/T1) — all modules now NaN-clean
Apr 8        → Zero-gradient root cause found (3-layer: B29-B31/T2); gradient still 15% off
Apr 9        → WUE IFT fix (B32/T5, 3 attempts); lax.scan 77×-818× speedup
Apr 10       → Obukhov IFT (B34/T5, 2 attempts); stomatal float→JAX (B35/T2)
Apr 10       → MLpftcon injection fix (B36-B37/T2, 2 attempts); all 5 params CPU-verified
Apr 10       → All 5 JAX grads confirmed on GPU A40 (rel err ≤ 1.3e-4)
Apr 14       → LE/H grad check PASS; Jacobian Vcmax25 non-zero confirmed
Apr 22–24    → Medlyn ci-scan NaN (B38/T5); g1_MED sign wrong → explicit arg (B39/T2, 3 attempts)
Apr 24       → alpha_pbot NaN inactive layer (B40/T1) fixed
Apr 27       → 7-param Jacobian all non-zero, all PASS
Apr 28–May 8 → Calibration: equifinality (B44), nighttime step (B42), Adam β₂ (B43) — fixed
```

**Time to full differentiability: 27 days (Apr 1 → Apr 27)**
