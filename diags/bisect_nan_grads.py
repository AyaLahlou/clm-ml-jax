"""
Bisect NaN gradients by applying jax.lax.stop_gradient after each physics module.

Strategy: Monkey-patch each physics module's return value to stop gradient flow.
If stopping gradient after module X makes gradients finite, X is the NaN source.

Run from project root:
    cd src && python ../diags/bisect_nan_grads.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

NAMELIST = SRC_DIR / "offline_executable" / "nl.CHATS7.1day"

# ── Initialization (same as test_grad.py) ─────────────────────────────────
sys.argv = ['bisect_nan_grads.py', str(NAMELIST)]
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

clmSoilOptionMod.clm_phys         = str(params.get("clm_phys", clmSoilOptionMod.clm_phys))
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
print("Running warmup timestep...", flush=True)
ModelAdvance(bounds, time_indx, fin_tower, fin_clm)
print("Warmup done.", flush=True)

# ── Set up diff mode (Euler, 1 sub-step) ─────────────────────────────────
from multilayer_canopy import MLclm_varctl as _ml_ctl
_orig_rk    = _ml_ctl.runge_kutta_type
_orig_dtime = _ml_ctl.dtime_ml
_ml_ctl.runge_kutta_type = 10                # Euler: nrk_steps=0
_ml_ctl.dtime_ml         = float(clm_time_manager.dtstep)  # 1 sub-step

mlcanopy_inst = clm_instMod.mlcanopy_inst
filt          = _clm_driver_mod.filter
num_evp       = filt.num_exposedvegp
evp_list      = [int(v) for v in filt.exposedvegp[:num_evp]]

from multilayer_canopy.MLCanopyFluxesMod import make_clm_ml_forward, GridInfo

_p = evp_list[0]
grid = GridInfo(
    p=_p,
    ncan=int(mlcanopy_inst.ncan_canopy[_p]),
    ntop=int(mlcanopy_inst.ntop_canopy[_p]),
    nbot=int(mlcanopy_inst.nbot_canopy[_p]),
)
print(f"Grid: p={grid.p}, ncan={grid.ncan}, ntop={grid.ntop}, nbot={grid.nbot}")

# Create forward function
forward_fn = make_clm_ml_forward(
    mlcanopy_inst_template=mlcanopy_inst,
    bounds=bounds,
    num_exposedvegp=num_evp,
    filter_exposedvegp=evp_list,
    atm2lnd_inst=clm_instMod.atm2lnd_inst,
    canopystate_inst=clm_instMod.canopystate_inst,
    soilstate_inst=clm_instMod.soilstate_inst,
    temperature_inst=clm_instMod.temperature_inst,
    waterstatebulk_inst=clm_instMod.waterstatebulk_inst,
    waterfluxbulk_inst=clm_instMod.waterfluxbulk_inst,
    energyflux_inst=clm_instMod.energyflux_inst,
    frictionvel_inst=clm_instMod.frictionvel_inst,
    surfalb_inst=clm_instMod.surfalb_inst,
    solarabs_inst=clm_instMod.solarabs_inst,
    wateratm2lndbulk_inst=clm_instMod.wateratm2lndbulk_inst,
    waterdiagnosticbulk_inst=clm_instMod.waterdiagnosticbulk_inst,
)

# ── Baseline: verify NaN ─────────────────────────────────────────────────
print("\n=== Baseline (no stop_gradient) ===")
t0 = time.time()
loss = forward_fn(mlcanopy_inst)
print(f"  Forward loss = {float(loss):.6f}  (finite={bool(jnp.isfinite(loss))})")

try:
    grad_fn = jax.grad(forward_fn, allow_int=True)
    with jax.disable_jit():
        grads = grad_fn(mlcanopy_inst)
    # Check a few key fields
    for fname in ['tair_profile', 'eair_profile', 'tleaf_leaf', 'tg_soil']:
        gv = getattr(grads, fname)
        if gv is not None:
            is_fin = bool(jnp.all(jnp.isfinite(gv)))
            mx = float(jnp.max(jnp.abs(gv))) if is_fin else float('nan')
            print(f"  grad({fname}): finite={is_fin}, max_abs={mx:.4e}")
except Exception as e:
    print(f"  jax.grad FAILED: {e}")
dt_baseline = time.time() - t0
print(f"  Time: {dt_baseline:.1f}s")

# ── Bisection: stop_gradient after each module ───────────────────────────
# We monkey-patch the module functions to apply stop_gradient to the
# mlcanopy_inst they return. This cuts gradient flow at that point.
# If stopping after module X gives finite grads, X is the NaN source.

import multilayer_canopy.MLCanopyTurbulenceMod as _turb_mod
import multilayer_canopy.MLLeafPhotosynthesisMod as _photo_mod
import multilayer_canopy.MLFluxProfileSolutionMod as _fps_mod
import multilayer_canopy.MLLongwaveRadiationMod as _lw_mod
import multilayer_canopy.MLSolarRadiationMod as _solar_mod
import multilayer_canopy.MLCanopyWaterMod as _water_mod

def _make_sg_wrapper(orig_fn, name):
    """Wrap a physics function to apply stop_gradient to its output."""
    def wrapped(*args, **kwargs):
        result = orig_fn(*args, **kwargs)
        # Apply stop_gradient to every field of the NamedTuple
        result = jax.lax.stop_gradient(result)
        return result
    wrapped.__name__ = f"{name}_sg"
    return wrapped

# Modules to test, in physics call order.
# Each entry: (module_obj, function_name, display_name)
modules_to_test = [
    # CanopyTurbulence is called via CanopyTurbulence() which dispatches to _HF2008_diff
    (_turb_mod, 'CanopyTurbulence', 'CanopyTurbulence'),
    (_photo_mod, 'LeafPhotosynthesis', 'LeafPhotosynthesis'),
    (_fps_mod, 'FluxProfileSolution', 'FluxProfileSolution'),
    (_lw_mod, 'LongwaveRadiation', 'LongwaveRadiation'),
    (_solar_mod, 'SolarRadiation', 'SolarRadiation'),
]

# We also need to patch at the call-site level.
# The physics_step_fn in MLCanopyFluxesMod imports these functions at module level.
import multilayer_canopy.MLCanopyFluxesMod as _flux_mod

print("\n=== Bisection: stop_gradient after each module ===")
print("(If 'finite=True' after stopping module X, then X introduces NaN)\n")

for mod_obj, fn_name, display_name in modules_to_test:
    # Save originals
    orig_fn = getattr(mod_obj, fn_name)
    orig_flux_fn = getattr(_flux_mod, fn_name, None)

    # Patch
    sg_fn = _make_sg_wrapper(orig_fn, fn_name)
    setattr(mod_obj, fn_name, sg_fn)
    if orig_flux_fn is not None:
        setattr(_flux_mod, fn_name, sg_fn)

    # Rebuild forward_fn with the patched module
    forward_fn_patched = make_clm_ml_forward(
        mlcanopy_inst_template=mlcanopy_inst,
        bounds=bounds,
        num_exposedvegp=num_evp,
        filter_exposedvegp=evp_list,
        atm2lnd_inst=clm_instMod.atm2lnd_inst,
        canopystate_inst=clm_instMod.canopystate_inst,
        soilstate_inst=clm_instMod.soilstate_inst,
        temperature_inst=clm_instMod.temperature_inst,
        waterstatebulk_inst=clm_instMod.waterstatebulk_inst,
        waterfluxbulk_inst=clm_instMod.waterfluxbulk_inst,
        energyflux_inst=clm_instMod.energyflux_inst,
        frictionvel_inst=clm_instMod.frictionvel_inst,
        surfalb_inst=clm_instMod.surfalb_inst,
        solarabs_inst=clm_instMod.solarabs_inst,
        wateratm2lndbulk_inst=clm_instMod.wateratm2lndbulk_inst,
        waterdiagnosticbulk_inst=clm_instMod.waterdiagnosticbulk_inst,
    )

    t0 = time.time()
    try:
        grad_fn_p = jax.grad(forward_fn_patched, allow_int=True)
        with jax.disable_jit():
            grads_p = grad_fn_p(mlcanopy_inst)

        any_nan = False
        for fname in ['tair_profile', 'eair_profile', 'tleaf_leaf', 'tg_soil']:
            gv = getattr(grads_p, fname)
            if gv is not None and not bool(jnp.all(jnp.isfinite(gv))):
                any_nan = True
                break

        result = "FINITE (NaN source!)" if not any_nan else "still NaN"
        dt = time.time() - t0
        print(f"  stop_gradient after {display_name:30s} → {result}  ({dt:.1f}s)")

        if not any_nan:
            # Print gradient details
            for fname in ['tair_profile', 'eair_profile', 'tleaf_leaf', 'tg_soil']:
                gv = getattr(grads_p, fname)
                if gv is not None:
                    mx = float(jnp.max(jnp.abs(gv)))
                    print(f"    grad({fname}): max_abs={mx:.4e}")

    except Exception as e:
        dt = time.time() - t0
        print(f"  stop_gradient after {display_name:30s} → ERROR: {e}  ({dt:.1f}s)")

    # Restore originals
    setattr(mod_obj, fn_name, orig_fn)
    if orig_flux_fn is not None:
        setattr(_flux_mod, fn_name, orig_flux_fn)

# Restore settings
_ml_ctl.runge_kutta_type = _orig_rk
_ml_ctl.dtime_ml         = _orig_dtime

print("\nBisection complete.")
