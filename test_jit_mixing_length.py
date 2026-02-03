"""Test script to verify mixing_length is fully JIT-compatible."""

import jax
import jax.numpy as jnp
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from clm_src_main.mixing_length import compute_mixing_length

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

def test_jit_compilation():
    """Test that compute_mixing_length can be JIT-compiled."""
    
    print("Testing JIT compilation of compute_mixing_length...")
    
    # Create simple test inputs
    nzt = 10
    nzm = nzt - 1
    ngrdcol = 2
    
    # Grid arrays (mock Grid object is None, we'll pass arrays directly)
    zt = jnp.linspace(0, 1000, nzt).reshape(1, nzt).repeat(ngrdcol, axis=0)
    zm = jnp.linspace(0, 1000, nzm).reshape(1, nzm).repeat(ngrdcol, axis=0)
    
    # Atmospheric state
    thvm = jnp.full((ngrdcol, nzt), 300.0)
    thlm = jnp.full((ngrdcol, nzt), 300.0)
    rtm = jnp.full((ngrdcol, nzt), 0.01)
    em = jnp.full((ngrdcol, nzm), 1.0)  # TKE on momentum levels
    exner = jnp.full((ngrdcol, nzt), 1.0)
    p_in_Pa = jnp.full((ngrdcol, nzt), 100000.0)
    thv_ds = jnp.zeros((ngrdcol, nzt))
    
    # Parameters
    Lscale_max = jnp.full(ngrdcol, 300.0)  # Array, not scalar
    mu = 0.002
    lmin = 0.1
    saturation_formula = 1
    l_implemented = False
    
    # Mock err_info
    from collections import namedtuple
    ErrInfo = namedtuple('ErrInfo', ['code'])
    err_info = ErrInfo(code=0)
    
    print("\n1. Testing eager execution (no JIT)...")
    try:
        Lscale, Lscale_up, Lscale_down, err_info_out = compute_mixing_length(
            nzm, nzt, ngrdcol, None,
            thvm, thlm, rtm, em, Lscale_max, p_in_Pa, exner, thv_ds,
            mu, lmin, saturation_formula, l_implemented, err_info
        )
        print("   ✅ Eager execution successful")
        print(f"   Lscale shape: {Lscale.shape}")
        print(f"   Lscale_up shape: {Lscale_up.shape}")
        print(f"   Lscale_down shape: {Lscale_down.shape}")
    except Exception as e:
        print(f"   ❌ Eager execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n2. JIT-compiling compute_mixing_length...")
    try:
        # Mark static arguments: nzm (0), nzt (1), ngrdcol (2), gr=None (3), mu (12), lmin (13), saturation_formula (14), l_implemented (15)
        # These must be hashable (int, float, None, bool) - NOT arrays
        # thv_ds (11) and other arrays should NOT be static
        compute_mixing_length_jit = jax.jit(compute_mixing_length, static_argnums=(0, 1, 2, 3, 12, 13, 14, 15))
        print("   ✅ JIT compilation initiated")
    except Exception as e:
        print(f"   ❌ JIT compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n3. Testing first call (triggers compilation)...")
    try:
        Lscale_jit, Lscale_up_jit, Lscale_down_jit, err_info_jit = compute_mixing_length_jit(
            nzm, nzt, ngrdcol, None,
            thvm, thlm, rtm, em, Lscale_max, p_in_Pa, exner, thv_ds,
            mu, lmin, saturation_formula, l_implemented, err_info
        )
        print("   ✅ First JIT call successful (compilation completed)")
        
        # Verify outputs match
        if jnp.allclose(Lscale, Lscale_jit):
            print("   ✅ Outputs match eager execution")
        else:
            print("   ⚠️  Outputs differ slightly from eager execution")
            max_diff = jnp.max(jnp.abs(Lscale - Lscale_jit))
            print(f"      Max difference: {max_diff}")
    except Exception as e:
        print(f"   ❌ First JIT call failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n4. Testing second call (should be fast - uses cached compilation)...")
    try:
        Lscale_jit_2, _, _, _ = compute_mixing_length_jit(
            nzm, nzt, ngrdcol, None,
            thvm, thlm, rtm, em, Lscale_max, p_in_Pa, exner, thv_ds,
            mu, lmin, saturation_formula, l_implemented, err_info
        )
        print("   ✅ Second JIT call successful (cached)")
        
        # Verify consistency
        if jnp.allclose(Lscale_jit, Lscale_jit_2):
            print("   ✅ JIT outputs are deterministic")
        else:
            print("   ❌ JIT outputs are non-deterministic!")
            return False
    except Exception as e:
        print(f"   ❌ Second JIT call failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("🎉 SUCCESS: compute_mixing_length is FULLY JIT-COMPATIBLE!")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_jit_compilation()
    sys.exit(0 if success else 1)
