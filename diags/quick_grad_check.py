"""
Quick JIT-based gradient NaN check.

Runs jax.grad with JIT enabled (fast) and checkpoint disabled.
Usage: cd src && python ../diags/quick_grad_check.py
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

os.environ['CLM_ML_NO_CHECKPOINT'] = '1'  # must be before JAX imports

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
NAMELIST = SRC_DIR / "offline_executable" / "nl.CHATS7.1day"

# ── Init (same as test_grad.py) ──
sys.argv = ['quick_grad_check.py', str(NAMELIST)]
import offline_executable.main as _main_mod
nml = _main_mod.read_namelist(str(NAMELIST))
params = nml.get("clmML_inparm", nml.get("clm_inparm", nml.get("clm_input", {})))

tower_name = str(params.get("tower_name", "CHATS7"))
start_ymd = int(params.get("start_ymd", 20070501))
iyear, imonth = start_ymd // 10000, (start_ymd // 100) % 100
fin_tower = _main_mod._resolve_path(str(params.get("fin_tower", "")))
fin_clm = _main_mod._resolve_path(str(params.get("fin_clm", "")))
fin_soil_adjust = _main_mod._resolve_path(str(params.get("fin_soil_adjust", "")))
dirout_raw = str(params.get("dirout", "")).strip()
dirout = _main_mod._resolve_path(dirout_raw) if dirout_raw else ""

from offline_driver import controlMod, TowerDataMod, clmSoilOptionMod
from multilayer_canopy import MLclm_varctl
from clm_src_utils import clm_time_manager
from clm_src_utils.clm_time_manager import get_curr_calday

controlMod.tower_site = tower_name; controlMod.iyear = iyear; controlMod.imonth = imonth
tower_num = 0
for i in range(1, int(TowerDataMod.ntower) + 1):
    if tower_name == str(TowerDataMod.tower_id[i]):
        tower_num = i; break
TowerDataMod.tower_num = tower_num
clmSoilOptionMod.clm_phys = str(params.get("clm_phys", clmSoilOptionMod.clm_phys))
clmSoilOptionMod.nlev_soil_adjust = int(params.get("nlev_soil_adjust", 0))
clmSoilOptionMod.fin_soil_adjust = fin_soil_adjust
MLclm_varctl.met_type = int(params.get("met_type", MLclm_varctl.met_type))
MLclm_varctl.dpai_min = float(params.get("dpai_min", MLclm_varctl.dpai_min))
MLclm_varctl.pftcon_val = int(params.get("pftcon_val", MLclm_varctl.pftcon_val))
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
clm_varorb.eccen = _eccen; clm_varorb.obliqr = _obliqr
clm_varorb.lambm0 = _lambm0; clm_varorb.mvelpp = _mvelpp

(clm_instMod.atm2lnd_inst, clm_instMod.wateratm2lndbulk_inst,
 clm_instMod.temperature_inst, clm_instMod.frictionvel_inst,
 clm_instMod.mlcanopy_inst) = init_acclim(
    fin_tower, tower_num, ntimes, bounds.begp, bounds.endp,
    clm_instMod.atm2lnd_inst, clm_instMod.wateratm2lndbulk_inst,
    clm_instMod.temperature_inst, clm_instMod.frictionvel_inst, clm_instMod.mlcanopy_inst)
(clm_instMod.canopystate_inst, clm_instMod.mlcanopy_inst) = TowerVeg(
    tower_num, bounds.begp, bounds.endp,
    clm_instMod.canopystate_inst, clm_instMod.mlcanopy_inst)

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
soil_init_time_indx = round((curr_calday - start_calday_clm) * 86400.0 / float(dtstep_sec)) + 1

(clm_instMod.waterstatebulk_inst, clm_instMod.temperature_inst) = SoilInit(
    fin_clm, soil_init_time_indx, bounds.begc, bounds.endc,
    clm_instMod.soilstate_inst, clm_instMod.waterstatebulk_inst, clm_instMod.temperature_inst)

os.makedirs(dirout, exist_ok=True)
clm_time_manager.itim = 1
curr_calday_ts1 = get_curr_calday(offset=0)
time_indx = round((curr_calday_ts1 - start_calday_clm) * 86400.0 / float(dtstep_sec)) + 1

(clm_instMod.atm2lnd_inst, clm_instMod.wateratm2lndbulk_inst,
 clm_instMod.frictionvel_inst) = TowerMetCurr(
    fin_tower, 1, tower_num, bounds.begp, bounds.endp,
    clm_instMod.atm2lnd_inst, clm_instMod.wateratm2lndbulk_inst, clm_instMod.frictionvel_inst)
if MLclm_varctl.met_type == 3:
    clm_instMod.mlcanopy_inst = TowerMetNext(
        fin_tower, min(2, ntimes), bounds.begp, bounds.endp, clm_instMod.mlcanopy_inst)

print("Running warmup...", flush=True)
ModelAdvance(bounds, time_indx, fin_tower, fin_clm)
print("Warmup done.", flush=True)

# ── Set Euler, 1 sub-step ──
from multilayer_canopy import MLclm_varctl as _ml_ctl
_ml_ctl.runge_kutta_type = 10
_ml_ctl.dtime_ml = float(clm_time_manager.dtstep)

mlcanopy_inst = clm_instMod.mlcanopy_inst
filt = _clm_driver_mod.filter
num_evp = filt.num_exposedvegp
evp_list = [int(v) for v in filt.exposedvegp[:num_evp]]

from multilayer_canopy.MLCanopyFluxesMod import make_clm_ml_forward
forward_fn = make_clm_ml_forward(
    mlcanopy_inst_template=mlcanopy_inst, bounds=bounds,
    num_exposedvegp=num_evp, filter_exposedvegp=evp_list,
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

# ── Forward ──
print("\n=== Forward pass ===")
t0 = time.time()
loss = forward_fn(mlcanopy_inst)
jax.block_until_ready(loss)
print(f"Loss = {float(loss):.6f}  ({time.time()-t0:.1f}s)")

# ── Grad (JIT, no checkpoint) ──
print("\n=== Computing jax.grad (JIT, no checkpoint) ===")
t0 = time.time()
grad_fn = jax.jit(jax.grad(forward_fn, allow_int=True))
grads = grad_fn(mlcanopy_inst)
# Force materialization
for fname in ['tair_profile', 'eair_profile', 'tleaf_leaf', 'tg_soil']:
    gv = getattr(grads, fname)
    if gv is not None:
        jax.block_until_ready(gv)
dt = time.time() - t0
print(f"Grad completed in {dt:.1f}s")

# ── Check results ──
print("\n=== Gradient check ===")
all_finite = True
for fname in ['tair_profile', 'eair_profile', 'tleaf_leaf', 'tg_soil',
              'wind_profile', 'pref_forcing', 'tref_forcing', 'rhomol_forcing']:
    gv = getattr(grads, fname, None)
    if gv is not None:
        is_fin = bool(jnp.all(jnp.isfinite(gv)))
        mx = float(jnp.max(jnp.abs(gv))) if is_fin else float(jnp.nanmax(jnp.abs(gv)))
        status = "FINITE" if is_fin else "*** NaN ***"
        print(f"  grad({fname:25s}): {status}  max_abs={mx:.4e}")
        if not is_fin:
            all_finite = False
            # Show where NaN occurs
            nan_mask = ~jnp.isfinite(gv)
            nan_count = int(jnp.sum(nan_mask))
            print(f"    NaN count: {nan_count}/{gv.size}")

if all_finite:
    print("\n*** ALL GRADIENTS FINITE — differentiability achieved! ***")
else:
    print("\n*** Some gradients still NaN — more fixes needed ***")
