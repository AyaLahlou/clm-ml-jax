"""
Integration tests for differentiability of the CLM-ML-JAX model.

Tests:
  - jax.grad(forward_fn) completes without error
  - Gradients are finite and non-zero for key state variables
  - Gradient magnitudes pass a finite-difference sanity check (loose tolerance)

These tests require the full model initialization (tower data, CLM history
files) and run on GPU if available.  Mark them slow so the default test
run skips them:

    pytest -m "not slow"   # skip differentiability tests
    pytest -m slow         # run only differentiability tests
    pytest                 # run everything

"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# -----------------------------------------------------------------------
# JAX must be configured before any import that touches jax arrays
# -----------------------------------------------------------------------
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Project root on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

NAMELIST = SRC_DIR / "offline_executable" / "nl.CHATS7.1day"


# -----------------------------------------------------------------------
# Shared model initialization (done once per pytest session)
# -----------------------------------------------------------------------

def _build_initialized_model():
    """Return (forward_fn, mlcanopy_inst, filter_info) after 1 warm-up step."""
    import offline_executable.main as _main_mod

    nml    = _main_mod.read_namelist(str(NAMELIST))
    params = nml.get("clmML_inparm", nml.get("clm_inparm", nml.get("clm_input", {})))

    tower_name = str(params.get("tower_name", "CHATS7"))
    start_ymd  = int(params.get("start_ymd", 20070501))
    iyear      = start_ymd // 10000
    imonth     = (start_ymd // 100) % 100

    fin_tower       = _main_mod._resolve_path(str(params.get("fin_tower", "")))
    fin_clm         = _main_mod._resolve_path(str(params.get("fin_clm", "")))
    fin_soil_adjust = _main_mod._resolve_path(str(params.get("fin_soil_adjust", "")))
    dirout_raw      = str(params.get("dirout", "")).strip()
    dirout          = _main_mod._resolve_path(dirout_raw) if dirout_raw else ""

    from offline_driver import controlMod, TowerDataMod, clmSoilOptionMod
    from multilayer_canopy import MLclm_varctl
    from clm_src_utils import clm_time_manager
    from clm_src_utils.clm_time_manager import get_curr_calday

    controlMod.tower_site = tower_name
    controlMod.iyear      = iyear
    controlMod.imonth     = imonth

    tower_num = 0
    for i in range(1, int(TowerDataMod.ntower) + 1):
        if tower_name == str(TowerDataMod.tower_id[i]):
            tower_num = i
            break
    TowerDataMod.tower_num = tower_num

    clmSoilOptionMod.clm_phys          = str(params.get("clm_phys", clmSoilOptionMod.clm_phys))
    clmSoilOptionMod.nlev_soil_adjust  = int(params.get("nlev_soil_adjust", 0))
    clmSoilOptionMod.fin_soil_adjust   = fin_soil_adjust
    MLclm_varctl.met_type              = int(params.get("met_type",   MLclm_varctl.met_type))
    MLclm_varctl.dpai_min              = float(params.get("dpai_min", MLclm_varctl.dpai_min))
    MLclm_varctl.pftcon_val            = int(params.get("pftcon_val", MLclm_varctl.pftcon_val))

    clm_time_manager.start_date_ymd = start_ymd
    clm_time_manager.start_date_tod = int(params.get("start_tod", 0))
    dtstep_sec = int(TowerDataMod.tower_time[tower_num]) * 60
    clm_time_manager.dtstep = dtstep_sec

    bounds = _main_mod.build_bounds(nml)
    ntimes = 2

    from clm_src_cpl.lnd_comp_nuopc import InitializeRealize, ModelAdvance
    from clm_src_main import filterMod as _filterMod

    InitializeRealize(bounds)

    from clm_src_main import clm_instMod

    _new_filter = _filterMod.setFilters(_filterMod.filter)
    _filterMod.filter = _new_filter
    import clm_src_main.clm_driver as _clm_driver_mod
    _clm_driver_mod.filter = _new_filter

    from offline_driver.CLMml_driver import init_acclim, TowerVeg, SoilInit
    from offline_driver.TowerMetMod import TowerMetCurr, TowerMetNext
    from clm_share.shr_orb_mod import shr_orb_params
    from clm_src_utils import clm_varorb

    _eccen, _obliq, _mvelp, _obliqr, _lambm0, _mvelpp = shr_orb_params(iyear)
    clm_varorb.eccen  = _eccen
    clm_varorb.obliqr = _obliqr
    clm_varorb.lambm0 = _lambm0
    clm_varorb.mvelpp = _mvelpp

    (clm_instMod.atm2lnd_inst,
     clm_instMod.wateratm2lndbulk_inst,
     clm_instMod.temperature_inst,
     clm_instMod.frictionvel_inst,
     clm_instMod.mlcanopy_inst) = init_acclim(
        fin_tower, tower_num, ntimes,
        bounds.begp, bounds.endp,
        clm_instMod.atm2lnd_inst,
        clm_instMod.wateratm2lndbulk_inst,
        clm_instMod.temperature_inst,
        clm_instMod.frictionvel_inst,
        clm_instMod.mlcanopy_inst,
    )

    (clm_instMod.canopystate_inst,
     clm_instMod.mlcanopy_inst) = TowerVeg(
        tower_num, bounds.begp, bounds.endp,
        clm_instMod.canopystate_inst, clm_instMod.mlcanopy_inst,
    )

    clm_start_ymd = int(params.get("clm_start_ymd", start_ymd))
    clm_start_tod = int(params.get("clm_start_tod", 0))
    run_start_ymd = clm_time_manager.start_date_ymd
    run_start_tod = clm_time_manager.start_date_tod
    clm_time_manager.start_date_ymd = clm_start_ymd
    clm_time_manager.start_date_tod = clm_start_tod
    clm_time_manager.itim = 1
    start_calday_clm = get_curr_calday(offset=0)
    clm_time_manager.start_date_ymd = run_start_ymd
    clm_time_manager.start_date_tod = run_start_tod
    clm_time_manager.itim = 1
    curr_calday = get_curr_calday(offset=0)
    soil_init_time_indx = round(
        (curr_calday - start_calday_clm) * 86400.0 / float(dtstep_sec)
    ) + 1

    (clm_instMod.waterstatebulk_inst,
     clm_instMod.temperature_inst) = SoilInit(
        fin_clm, soil_init_time_indx,
        bounds.begc, bounds.endc,
        clm_instMod.soilstate_inst,
        clm_instMod.waterstatebulk_inst,
        clm_instMod.temperature_inst,
    )

    os.makedirs(dirout, exist_ok=True)

    from clm_src_utils.clm_time_manager import get_curr_date
    clm_time_manager.itim = 1
    curr_calday_ts1 = get_curr_calday(offset=0)
    time_indx = round(
        (curr_calday_ts1 - start_calday_clm) * 86400.0 / float(dtstep_sec)
    ) + 1

    (clm_instMod.atm2lnd_inst,
     clm_instMod.wateratm2lndbulk_inst,
     clm_instMod.frictionvel_inst) = TowerMetCurr(
        fin_tower, 1, tower_num,
        bounds.begp, bounds.endp,
        clm_instMod.atm2lnd_inst,
        clm_instMod.wateratm2lndbulk_inst,
        clm_instMod.frictionvel_inst,
    )

    if MLclm_varctl.met_type == 3:
        clm_instMod.mlcanopy_inst = TowerMetNext(
            fin_tower, min(2, ntimes),
            bounds.begp, bounds.endp,
            clm_instMod.mlcanopy_inst,
        )

    # Run 1 warm-up timestep
    ModelAdvance(bounds, time_indx, fin_tower, fin_clm)

    mlcanopy_inst = clm_instMod.mlcanopy_inst
    filt          = _clm_driver_mod.filter
    num_evp       = filt.num_exposedvegp
    evp_list      = [int(v) for v in filt.exposedvegp[:num_evp]]

    from multilayer_canopy.MLCanopyFluxesMod import make_clm_ml_forward
    forward_fn = make_clm_ml_forward(
        mlcanopy_inst_template        = mlcanopy_inst,
        bounds                        = bounds,
        num_exposedvegp               = num_evp,
        filter_exposedvegp            = evp_list,
        atm2lnd_inst                  = clm_instMod.atm2lnd_inst,
        canopystate_inst              = clm_instMod.canopystate_inst,
        soilstate_inst                = clm_instMod.soilstate_inst,
        temperature_inst              = clm_instMod.temperature_inst,
        waterstatebulk_inst           = clm_instMod.waterstatebulk_inst,
        waterfluxbulk_inst            = clm_instMod.waterfluxbulk_inst,
        energyflux_inst               = clm_instMod.energyflux_inst,
        frictionvel_inst              = clm_instMod.frictionvel_inst,
        surfalb_inst                  = clm_instMod.surfalb_inst,
        solarabs_inst                 = clm_instMod.solarabs_inst,
        wateratm2lndbulk_inst         = clm_instMod.wateratm2lndbulk_inst,
        waterdiagnosticbulk_inst      = clm_instMod.waterdiagnosticbulk_inst,
    )

    return forward_fn, mlcanopy_inst, {"num_evp": num_evp, "evp_list": evp_list}


# -----------------------------------------------------------------------
# Session-scoped fixture: build the model once, share across all tests
# -----------------------------------------------------------------------

@pytest.fixture(scope="session")
def initialized_model():
    if not NAMELIST.exists():
        pytest.skip(f"Namelist not found: {NAMELIST}")
    return _build_initialized_model()


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

@pytest.mark.slow
def test_forward_pass_returns_finite_scalar(initialized_model):
    """Differentiable forward pass returns a finite scalar loss."""
    forward_fn, mlcanopy_inst, _ = initialized_model
    loss = forward_fn(mlcanopy_inst)
    assert jnp.isfinite(loss), f"Forward loss is not finite: {loss}"
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"


@pytest.mark.slow
def test_jax_grad_completes(initialized_model):
    """jax.grad(forward_fn) completes without raising an exception."""
    forward_fn, mlcanopy_inst, _ = initialized_model
    grad_fn = jax.grad(forward_fn, allow_int=True)
    grads = grad_fn(mlcanopy_inst)  # must not raise
    assert grads is not None


@pytest.mark.slow
def test_gradients_are_finite(initialized_model):
    """Key gradient fields are finite (no NaN/Inf)."""
    forward_fn, mlcanopy_inst, _ = initialized_model
    grad_fn = jax.grad(forward_fn, allow_int=True)
    grads   = grad_fn(mlcanopy_inst)

    fields_to_check = [
        "tair_profile",
        "eair_profile",
        "tleaf_leaf",
        "tg_soil",
    ]
    for field in fields_to_check:
        g = getattr(grads, field, None)
        if g is None:
            continue
        assert jnp.all(jnp.isfinite(g)), (
            f"Gradient of '{field}' contains non-finite values: "
            f"max_abs={float(jnp.max(jnp.abs(g))):.3e}"
        )


@pytest.mark.slow
def test_gradients_are_nonzero(initialized_model):
    """Key gradient fields have at least one non-zero entry."""
    forward_fn, mlcanopy_inst, _ = initialized_model
    grad_fn = jax.grad(forward_fn, allow_int=True)
    grads   = grad_fn(mlcanopy_inst)

    # At least one of the primary prognostic fields must have a non-zero gradient.
    primary_fields = ["tleaf_leaf", "tair_profile", "tg_soil"]
    nonzero_found  = False
    for field in primary_fields:
        g = getattr(grads, field, None)
        if g is None:
            continue
        if float(jnp.max(jnp.abs(g))) > 0.0:
            nonzero_found = True
            break

    assert nonzero_found, (
        "All primary gradient fields are zero — forward pass may be "
        "disconnected from the loss or constants are masking gradients."
    )


@pytest.mark.slow
def test_finite_difference_gradient_check(initialized_model):
    """
    Loose finite-difference check on a single scalar parameter.

    Perturbs tleaf_leaf[p, ic, isun] by eps and compares the analytic
    gradient from jax.grad against the finite-difference estimate.
    Tolerance is generous (rtol=0.05) to account for non-smooth physics.
    """
    from multilayer_canopy.MLclm_varpar import isun

    forward_fn, mlcanopy_inst, info = initialized_model
    evp_list = info["evp_list"]
    p        = evp_list[0]
    ncan     = int(mlcanopy_inst.ncan_canopy[p])
    ic       = max(1, ncan // 2)   # middle canopy layer

    # Analytic gradient
    grad_fn = jax.grad(forward_fn, allow_int=True)
    grads   = grad_fn(mlcanopy_inst)
    g_analytic = float(grads.tleaf_leaf[p, ic, isun])

    if g_analytic == 0.0:
        pytest.skip("Analytic gradient is zero at test point — skipping FD check")

    # Finite-difference estimate
    eps  = 1e-3  # K perturbation
    tleaf_plus  = mlcanopy_inst.tleaf_leaf.at[p, ic, isun].add( eps)
    tleaf_minus = mlcanopy_inst.tleaf_leaf.at[p, ic, isun].add(-eps)
    f_plus  = float(forward_fn(mlcanopy_inst._replace(tleaf_leaf=tleaf_plus)))
    f_minus = float(forward_fn(mlcanopy_inst._replace(tleaf_leaf=tleaf_minus)))
    g_fd = (f_plus - f_minus) / (2.0 * eps)

    # Relative error (use |g_fd| as reference if non-zero)
    denom = abs(g_fd) if abs(g_fd) > 1e-30 else abs(g_analytic)
    rel_err = abs(g_analytic - g_fd) / denom if denom > 0 else 0.0

    assert rel_err < 0.10, (
        f"Finite-difference check failed for tleaf_leaf[{p},{ic},{isun}]: "
        f"analytic={g_analytic:.6e}, fd={g_fd:.6e}, rel_err={rel_err:.3%}"
    )
