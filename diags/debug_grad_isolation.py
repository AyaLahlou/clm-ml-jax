"""
Isolate NaN gradient source by testing each physics module independently.

Run with:
    cd /burg-archive/home/al4385/clm-ml-jax
    python diags/debug_grad_isolation.py
"""
import os, sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ── replicate test_grad.py init ───────────────────────────────────────────────
namelist_file = str(Path(__file__).parent.parent / "src/offline_executable/nl.CHATS7.1day")
sys.argv = ['debug_grad_isolation.py', namelist_file]

import offline_executable.main as _main_mod
nml    = _main_mod.read_namelist(namelist_file)
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
controlMod.iyear = iyear
controlMod.imonth = imonth

tower_num = 0
for i in range(1, int(TowerDataMod.ntower) + 1):
    if tower_name == str(TowerDataMod.tower_id[i]):
        tower_num = i
        break
TowerDataMod.tower_num = tower_num

clmSoilOptionMod.clm_phys  = str(params.get("clm_phys", clmSoilOptionMod.clm_phys))
clmSoilOptionMod.nlev_soil_adjust = int(params.get("nlev_soil_adjust", 0))
clmSoilOptionMod.fin_soil_adjust  = fin_soil_adjust
MLclm_varctl.met_type   = int(params.get("met_type",   MLclm_varctl.met_type))
MLclm_varctl.dpai_min   = float(params.get("dpai_min", MLclm_varctl.dpai_min))
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
clm_varorb.eccen  = _eccen
clm_varorb.obliqr = _obliqr
clm_varorb.lambm0 = _lambm0
clm_varorb.mvelpp = _mvelpp

(clm_instMod.atm2lnd_inst, clm_instMod.wateratm2lndbulk_inst,
 clm_instMod.temperature_inst, clm_instMod.frictionvel_inst,
 clm_instMod.mlcanopy_inst) = init_acclim(
    fin_tower, tower_num, ntimes, bounds.begp, bounds.endp,
    clm_instMod.atm2lnd_inst, clm_instMod.wateratm2lndbulk_inst,
    clm_instMod.temperature_inst, clm_instMod.frictionvel_inst,
    clm_instMod.mlcanopy_inst,
)
(clm_instMod.canopystate_inst, clm_instMod.mlcanopy_inst) = TowerVeg(
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
soil_init_time_indx = round((curr_calday - start_calday_clm) * 86400.0 / float(dtstep_sec)) + 1

(clm_instMod.waterstatebulk_inst, clm_instMod.temperature_inst) = SoilInit(
    fin_clm, soil_init_time_indx, bounds.begc, bounds.endc,
    clm_instMod.soilstate_inst, clm_instMod.waterstatebulk_inst,
    clm_instMod.temperature_inst,
)

os.makedirs(dirout, exist_ok=True)

met_type = MLclm_varctl.met_type
clm_time_manager.itim = 1
from clm_src_utils.clm_time_manager import get_curr_date
curr_calday_ts1 = get_curr_calday(offset=0)
time_indx = round((curr_calday_ts1 - start_calday_clm) * 86400.0 / float(dtstep_sec)) + 1

(clm_instMod.atm2lnd_inst, clm_instMod.wateratm2lndbulk_inst,
 clm_instMod.frictionvel_inst) = TowerMetCurr(
    fin_tower, 1, tower_num, bounds.begp, bounds.endp,
    clm_instMod.atm2lnd_inst, clm_instMod.wateratm2lndbulk_inst,
    clm_instMod.frictionvel_inst,
)
if met_type == 3:
    clm_instMod.mlcanopy_inst = TowerMetNext(
        fin_tower, min(2, ntimes), bounds.begp, bounds.endp,
        clm_instMod.mlcanopy_inst,
    )

ModelAdvance(bounds, time_indx, fin_tower, fin_clm)
print("First timestep complete.", flush=True)

# ── Reduce to Euler / 1 sub-step ──────────────────────────────────────────────
_orig_rk    = MLclm_varctl.runge_kutta_type
_orig_dtime = MLclm_varctl.dtime_ml
MLclm_varctl.runge_kutta_type = 10
MLclm_varctl.dtime_ml         = float(clm_time_manager.dtstep)

mlcanopy_inst = clm_instMod.mlcanopy_inst
filt = _clm_driver_mod.filter
num_evp = filt.num_exposedvegp
evp_list = [int(v) for v in filt.exposedvegp[:num_evp]]
p = int(evp_list[0])
ncan = int(mlcanopy_inst.ncan_canopy[p])

print(f"p={p}, ncan={ncan}, num_evp={num_evp}, filter={evp_list}")

# ── Build a "pre-run" state: run one step in non-diff mode to get a warm state ─
from multilayer_canopy.MLCanopyFluxesMod import GridInfo, make_clm_ml_forward
grid = GridInfo(p=p, ncan=ncan,
                ntop=int(mlcanopy_inst.ntop_canopy[p]),
                nbot=int(mlcanopy_inst.nbot_canopy[p]))

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

print("\nForward pass:", flush=True)
loss = forward_fn(mlcanopy_inst)
print(f"  loss = {float(loss):.4f}", flush=True)

# ── Test 1: Full gradient (same as test_grad.py) ───────────────────────────────
print("\n=== TEST 1: Full forward_fn gradient ===", flush=True)
try:
    grads = jax.grad(forward_fn, allow_int=True)(mlcanopy_inst)
    nan_fields = [f for f in grads._fields
                  if getattr(grads, f) is not None and
                  bool(jnp.any(jnp.isnan(getattr(grads, f))))]
    print(f"  NaN fields: {nan_fields[:10]}" + ("..." if len(nan_fields) > 10 else ""))
    for fn in ['tair_profile', 'eair_profile', 'gac_profile', 'ustar_canopy']:
        g = getattr(grads, fn)
        if g is not None:
            print(f"  grad({fn}): NaN={bool(jnp.any(jnp.isnan(g)))}, "
                  f"max={float(jnp.nanmax(jnp.abs(g))):.3e}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── Test 2: Loss = sum(gac_profile) after CanopyTurbulence only ───────────────
print("\n=== TEST 2: Gradient through CanopyTurbulence only ===", flush=True)
from multilayer_canopy.MLCanopyFluxesMod import _copy_bef_state
from multilayer_canopy.MLGetAtmForcingMod import GetAtmForcing
from multilayer_canopy.MLSolarRadiationMod import SolarRadiation
from multilayer_canopy.MLCanopyNitrogenProfileMod import CanopyNitrogenProfile
from multilayer_canopy.MLCanopyWaterMod import CanopyWettedFraction
from multilayer_canopy.MLLongwaveRadiationMod import LongwaveRadiation
from multilayer_canopy.MLCanopyTurbulenceMod import CanopyTurbulence
from multilayer_canopy.MLLeafBoundaryLayerMod import LeafBoundaryLayer
from multilayer_canopy.MLLeafPhotosynthesisMod import LeafPhotosynthesis
from multilayer_canopy.MLFluxProfileSolutionMod import ImplicitFluxProfileSolution as FPS
from multilayer_canopy.MLclm_varctl import isun, isha, ivis, inir

filter_mlcan = tuple(evp_list)
ncan_vals    = tuple(int(mlcanopy_inst.ncan_canopy[q]) for q in evp_list)
calday_ml    = float(get_curr_calday(offset=0))

def _build_state_up_to_turb(m):
    """Run everything up to and including CanopyTurbulence."""
    m = _copy_bef_state(filter_mlcan, ncan_vals, m)
    m = GetAtmForcing(calday_ml, calday_ml, calday_ml, calday_ml,
                      num_evp, evp_list, m, grid=grid)
    m = SolarRadiation(bounds, num_evp, evp_list, m, grid=grid)
    m = CanopyNitrogenProfile(num_evp, evp_list, m)
    m = CanopyWettedFraction(num_evp, evp_list, m)
    m = LongwaveRadiation(bounds, num_evp, evp_list, m, grid=grid)
    m = m._replace(
        rnleaf_leaf=(m.swleaf_leaf[..., ivis] + m.swleaf_leaf[..., inir] + m.lwleaf_leaf),
        rnsoi_soil =(m.swsoi_soil[:, ivis]    + m.swsoi_soil[:, inir]    + m.lwsoi_soil),
    )
    m = CanopyTurbulence(1, num_evp, evp_list, m, grid=grid)
    return m

try:
    def loss_turb(m):
        m2 = _build_state_up_to_turb(m)
        return jnp.sum(m2.gac_profile[p, 1:ncan+1])

    grads2 = jax.grad(loss_turb, allow_int=True)(mlcanopy_inst)
    nan_fields2 = [f for f in grads2._fields
                   if getattr(grads2, f) is not None and
                   bool(jnp.any(jnp.isnan(getattr(grads2, f))))]
    print(f"  loss = sum(gac): {float(loss_turb(mlcanopy_inst)):.4f}")
    print(f"  NaN fields: {nan_fields2[:10]}" + ("..." if len(nan_fields2) > 10 else ""))
except Exception as e:
    print(f"  ERROR: {e}")

# ── Test 3: Loss = sum(gac_profile) after LeafPhotosynthesis ─────────────────
print("\n=== TEST 3: Gradient through LeafPhotosynthesis ===", flush=True)
try:
    def loss_photo(m):
        m2 = _build_state_up_to_turb(m)
        m2 = LeafBoundaryLayer(num_evp, evp_list, isun, m2)
        m2 = LeafBoundaryLayer(num_evp, evp_list, isha, m2)
        m2 = LeafPhotosynthesis(num_evp, evp_list, isun, m2, grid=grid)
        m2 = LeafPhotosynthesis(num_evp, evp_list, isha, m2, grid=grid)
        return jnp.sum(m2.gs_leaf[p, 1:ncan+1, :])

    grads3 = jax.grad(loss_photo, allow_int=True)(mlcanopy_inst)
    nan_fields3 = [f for f in grads3._fields
                   if getattr(grads3, f) is not None and
                   bool(jnp.any(jnp.isnan(getattr(grads3, f))))]
    print(f"  loss = sum(gs): {float(loss_photo(mlcanopy_inst)):.4f}")
    print(f"  NaN fields: {nan_fields3[:10]}" + ("..." if len(nan_fields3) > 10 else ""))
except Exception as e:
    print(f"  ERROR: {e}")

# ── Test 4: Loss = sum(shair) after FPS ──────────────────────────────────────
print("\n=== TEST 4: Gradient through FluxProfileSolution ===", flush=True)
try:
    def loss_fps(m):
        m2 = _build_state_up_to_turb(m)
        m2 = LeafBoundaryLayer(num_evp, evp_list, isun, m2)
        m2 = LeafBoundaryLayer(num_evp, evp_list, isha, m2)
        m2 = LeafPhotosynthesis(num_evp, evp_list, isun, m2, grid=grid)
        m2 = LeafPhotosynthesis(num_evp, evp_list, isha, m2, grid=grid)
        for _p in evp_list:
            m2 = FPS(_p, m2, n=int(m2.ncan_canopy[_p]))
        return (jnp.sum(m2.shair_profile[p, 1:ncan+1])
                + jnp.sum(m2.etair_profile[p, 1:ncan+1]))

    grads4 = jax.grad(loss_fps, allow_int=True)(mlcanopy_inst)
    nan_fields4 = [f for f in grads4._fields
                   if getattr(grads4, f) is not None and
                   bool(jnp.any(jnp.isnan(getattr(grads4, f))))]
    print(f"  loss = sum(shair+etair): {float(loss_fps(mlcanopy_inst)):.4f}")
    print(f"  NaN fields: {nan_fields4[:10]}" + ("..." if len(nan_fields4) > 10 else ""))
    for fn in ['tair_profile', 'tair_bef_profile', 'gac_profile', 'gs_leaf', 'ustar_canopy']:
        g = getattr(grads4, fn)
        if g is not None:
            print(f"  grad({fn}): NaN={bool(jnp.any(jnp.isnan(g)))}, "
                  f"max={float(jnp.nanmax(jnp.abs(g))):.3e}")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()

MLclm_varctl.runge_kutta_type = _orig_rk
MLclm_varctl.dtime_ml         = _orig_dtime
print("\nDone.", flush=True)
