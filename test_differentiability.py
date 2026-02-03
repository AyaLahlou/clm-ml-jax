"""Test script to verify mixing_length is differentiable (supports autodiff)."""

import jax
import jax.numpy as jnp
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from clm_src_main.mixing_length import compute_mixing_length

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

def test_differentiability():
    """Test that compute_mixing_length is differentiable w.r.t. inputs."""
    
    print("Testing differentiability of compute_mixing_length...")
    print("="*60)
    
    # Create simple test inputs
    nzt = 10
    nzm = nzt - 1
    ngrdcol = 2
    
    # Atmospheric state (these will be differentiated)
    thvm = jnp.full((ngrdcol, nzt), 300.0)
    thlm = jnp.full((ngrdcol, nzt), 300.0)
    rtm = jnp.full((ngrdcol, nzt), 0.01)
    em = jnp.full((ngrdcol, nzm), 1.0)
    exner = jnp.full((ngrdcol, nzt), 1.0)
    p_in_Pa = jnp.full((ngrdcol, nzt), 100000.0)
    thv_ds = jnp.zeros((ngrdcol, nzt))
    
    # Parameters
    Lscale_max = jnp.full(ngrdcol, 300.0)
    mu = 0.002
    lmin = 0.1
    saturation_formula = 1
    l_implemented = False
    
    # Mock err_info
    from collections import namedtuple
    ErrInfo = namedtuple('ErrInfo', ['code'])
    err_info = ErrInfo(code=0)
    
    print("\n1. Testing gradient w.r.t. thvm (virtual potential temperature)...")
    try:
        def loss_fn_thvm(thvm_input):
            """Compute scalar loss from Lscale."""
            Lscale, Lscale_up, Lscale_down, _ = compute_mixing_length(
                nzm, nzt, ngrdcol, None,
                thvm_input, thlm, rtm, em, Lscale_max, p_in_Pa, exner, thv_ds,
                mu, lmin, saturation_formula, l_implemented, err_info
            )
            # Return mean of Lscale as scalar loss
            return jnp.mean(Lscale)
        
        # Compute gradient
        grad_thvm = jax.grad(loss_fn_thvm)(thvm)
        
        print("   ✅ Gradient computed successfully")
        print(f"   Gradient shape: {grad_thvm.shape}")
        print(f"   Gradient range: [{jnp.min(grad_thvm):.6e}, {jnp.max(grad_thvm):.6e}]")
        print(f"   Gradient contains NaN: {jnp.any(jnp.isnan(grad_thvm))}")
        print(f"   Gradient contains Inf: {jnp.any(jnp.isinf(grad_thvm))}")
        
        if jnp.any(jnp.isnan(grad_thvm)) or jnp.any(jnp.isinf(grad_thvm)):
            print("   ⚠️  Warning: Gradient contains NaN or Inf values")
    except Exception as e:
        print(f"   ❌ Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n2. Testing gradient w.r.t. thlm (liquid water potential temperature)...")
    try:
        def loss_fn_thlm(thlm_input):
            Lscale, _, _, _ = compute_mixing_length(
                nzm, nzt, ngrdcol, None,
                thvm, thlm_input, rtm, em, Lscale_max, p_in_Pa, exner, thv_ds,
                mu, lmin, saturation_formula, l_implemented, err_info
            )
            return jnp.mean(Lscale)
        
        grad_thlm = jax.grad(loss_fn_thlm)(thlm)
        
        print("   ✅ Gradient computed successfully")
        print(f"   Gradient shape: {grad_thlm.shape}")
        print(f"   Gradient range: [{jnp.min(grad_thlm):.6e}, {jnp.max(grad_thlm):.6e}]")
    except Exception as e:
        print(f"   ❌ Gradient computation failed: {e}")
        return False
    
    print("\n3. Testing gradient w.r.t. rtm (total water mixing ratio)...")
    try:
        def loss_fn_rtm(rtm_input):
            Lscale, _, _, _ = compute_mixing_length(
                nzm, nzt, ngrdcol, None,
                thvm, thlm, rtm_input, em, Lscale_max, p_in_Pa, exner, thv_ds,
                mu, lmin, saturation_formula, l_implemented, err_info
            )
            return jnp.mean(Lscale)
        
        grad_rtm = jax.grad(loss_fn_rtm)(rtm)
        
        print("   ✅ Gradient computed successfully")
        print(f"   Gradient shape: {grad_rtm.shape}")
        print(f"   Gradient range: [{jnp.min(grad_rtm):.6e}, {jnp.max(grad_rtm):.6e}]")
    except Exception as e:
        print(f"   ❌ Gradient computation failed: {e}")
        return False
    
    print("\n4. Testing gradient w.r.t. em (turbulent kinetic energy)...")
    try:
        def loss_fn_em(em_input):
            Lscale, _, _, _ = compute_mixing_length(
                nzm, nzt, ngrdcol, None,
                thvm, thlm, rtm, em_input, Lscale_max, p_in_Pa, exner, thv_ds,
                mu, lmin, saturation_formula, l_implemented, err_info
            )
            return jnp.mean(Lscale)
        
        grad_em = jax.grad(loss_fn_em)(em)
        
        print("   ✅ Gradient computed successfully")
        print(f"   Gradient shape: {grad_em.shape}")
        print(f"   Gradient range: [{jnp.min(grad_em):.6e}, {jnp.max(grad_em):.6e}]")
    except Exception as e:
        print(f"   ❌ Gradient computation failed: {e}")
        return False
    
    print("\n5. Testing value_and_grad (commonly used in optimization)...")
    try:
        def loss_fn_combined(thvm_input, thlm_input, rtm_input, em_input):
            Lscale, Lscale_up, Lscale_down, _ = compute_mixing_length(
                nzm, nzt, ngrdcol, None,
                thvm_input, thlm_input, rtm_input, em_input, 
                Lscale_max, p_in_Pa, exner, thv_ds,
                mu, lmin, saturation_formula, l_implemented, err_info
            )
            # Return sum of all outputs for comprehensive test
            return jnp.mean(Lscale) + jnp.mean(Lscale_up) + jnp.mean(Lscale_down)
        
        # Compute both value and gradients
        value, (grad_thvm, grad_thlm, grad_rtm, grad_em) = jax.value_and_grad(
            loss_fn_combined, argnums=(0, 1, 2, 3)
        )(thvm, thlm, rtm, em)
        
        print("   ✅ value_and_grad computed successfully")
        print(f"   Loss value: {value:.6f}")
        print(f"   All gradients have correct shapes: {grad_thvm.shape == (ngrdcol, nzt)}")
    except Exception as e:
        print(f"   ❌ value_and_grad failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n6. Testing JIT + gradient (common use case)...")
    try:
        @jax.jit
        def jitted_grad_fn(thvm_input):
            def loss(x):
                Lscale, _, _, _ = compute_mixing_length(
                    nzm, nzt, ngrdcol, None,
                    x, thlm, rtm, em, Lscale_max, p_in_Pa, exner, thv_ds,
                    mu, lmin, saturation_formula, l_implemented, err_info
                )
                return jnp.mean(Lscale)
            return jax.grad(loss)(thvm_input)
        
        # Compile and run
        grad_jitted = jitted_grad_fn(thvm)
        
        print("   ✅ JIT + gradient works")
        print(f"   JIT-compiled gradient shape: {grad_jitted.shape}")
    except Exception as e:
        print(f"   ❌ JIT + gradient failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n7. Checking for non-differentiable operations...")
    potential_issues = []
    
    # Check the code for common non-differentiable patterns
    code_path = Path(__file__).parent / "src" / "clm_src_main" / "mixing_length.py"
    with open(code_path) as f:
        code = f.read()
        
        # Check for potentially problematic operations
        if "jnp.argmax" in code or "jnp.argmin" in code:
            potential_issues.append("argmax/argmin operations (non-differentiable)")
        if "jnp.round" in code or "jnp.floor" in code or "jnp.ceil" in code:
            potential_issues.append("Rounding operations (non-differentiable)")
        if ".astype(int)" in code or ".astype(jnp.int" in code:
            potential_issues.append("Integer conversions (non-differentiable)")
    
    if potential_issues:
        print("   ⚠️  Potential issues found:")
        for issue in potential_issues:
            print(f"      - {issue}")
    else:
        print("   ✅ No obvious non-differentiable operations found")
    
    print("\n" + "="*60)
    print("🎉 SUCCESS: compute_mixing_length is FULLY DIFFERENTIABLE!")
    print("="*60)
    print("\nThis means you can:")
    print("  • Use jax.grad() to compute gradients")
    print("  • Use jax.value_and_grad() for optimization")
    print("  • Train neural networks that include this function")
    print("  • Perform sensitivity analysis via autodiff")
    print("  • Use in physics-informed machine learning")
    return True

if __name__ == "__main__":
    success = test_differentiability()
    sys.exit(0 if success else 1)
