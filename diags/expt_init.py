"""
Shared initialization for differentiability experiments (Exp 2, 3, 4).

Imports this module to get:
  forward_fn       — scalar forward pass: mlcanopy_inst → loss (H+LE sum)
  mlcanopy_inst    — initialized mlcanopy_type NamedTuple (post-warmup)
  grid             — GridInfo namedtuple (p, ncan, ntop, nbot)

Usage (from project root, inside src/):
    cd src
    import sys; sys.path.insert(0, '.')
    from diags.expt_init import forward_fn, mlcanopy_inst, grid

Note: importing this module has side effects — it initializes all CLM
module-level singletons and runs one warmup timestep.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ['CLM_ML_NO_CHECKPOINT'] = '1'

import jax
jax.config.update("jax_enable_x64", os.environ.get("CLM_ML_X64", "1") != "0")
import jax.numpy as jnp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
NAMELIST = SRC_DIR / "offline_executable" / "nl.CHATS7.1day"

# ── Namelist ──────────────────────────────────────────────────────────────────
sys.argv = ['expt_init.py', str(NAMELIST)]
import offline_executable.main as _main_mod
nml = _main_mod.read_namelist(str(NAMELIST))
params = nml.get("clmML_inparm", nml.get("clm_inparm", nml.get("clm_input", {})))

tower_name    = str(params.get("tower_name", "CHATS7"))
start_ymd     = int(params.get("start_ymd", 20070501))
iyear, imonth = start_ymd // 10000, (start_ymd // 100) % 100
fin_tower     = _main_mod._resolve_path(str(params.get("fin_tower", "")))
fin_clm       = _main_mod._resolve_path(str(params.get("fin_clm", "")))
fin_soil_adj  = _main_mod._resolve_path(str(params.get("fin_soil_adjust", "")))
dirout        = _main_mod._resolve_path(str(params.get("dirout", "")).strip()) or ""

# ── Module globals ────────────────────────────────────────────────────────────
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
clmSoilOptionMod.fin_soil_adjust   = fin_soil_adj
MLclm_varctl.met_type              = int(params.get("met_type", MLclm_varctl.met_type))
MLclm_varctl.dpai_min              = float(params.get("dpai_min", MLclm_varctl.dpai_min))
MLclm_varctl.pftcon_val            = int(params.get("pftcon_val", MLclm_varctl.pftcon_val))
clm_time_manager.start_date_ymd    = start_ymd
clm_time_manager.start_date_tod    = int(params.get("start_tod", 0))

dtstep_sec = int(TowerDataMod.tower_time[tower_num]) * 60
clm_time_manager.dtstep = dtstep_sec
bounds = _main_mod.build_bounds(nml)
ntimes = 2

# ── Initialize ────────────────────────────────────────────────────────────────
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
clm_varorb.eccen   = _eccen
clm_varorb.obliqr  = _obliqr
clm_varorb.lambm0  = _lambm0
clm_varorb.mvelpp  = _mvelpp

(clm_instMod.atm2lnd_inst, clm_instMod.wateratm2lndbulk_inst,
 clm_instMod.temperature_inst, clm_instMod.frictionvel_inst,
 clm_instMod.mlcanopy_inst) = init_acclim(
    fin_tower, tower_num, ntimes, bounds.begp, bounds.endp,
    clm_instMod.atm2lnd_inst, clm_instMod.wateratm2lndbulk_inst,
    clm_instMod.temperature_inst, clm_instMod.frictionvel_inst,
    clm_instMod.mlcanopy_inst)

(clm_instMod.canopystate_inst, clm_instMod.mlcanopy_inst) = TowerVeg(
    tower_num, bounds.begp, bounds.endp,
    clm_instMod.canopystate_inst, clm_instMod.mlcanopy_inst)

clm_start_ymd  = int(params.get("clm_start_ymd", start_ymd))
clm_start_tod  = int(params.get("clm_start_tod", 0))
run_start_ymd  = clm_time_manager.start_date_ymd
run_start_tod  = clm_time_manager.start_date_tod
clm_time_manager.start_date_ymd = clm_start_ymd
clm_time_manager.start_date_tod = clm_start_tod
clm_time_manager.itim = 1
start_calday_clm = get_curr_calday(offset=0)
clm_time_manager.start_date_ymd = run_start_ymd
clm_time_manager.start_date_tod = run_start_tod
clm_time_manager.itim = 1
curr_calday = get_curr_calday(offset=0)
soil_init_time_indx = round(
    (curr_calday - start_calday_clm) * 86400.0 / float(dtstep_sec)) + 1

(clm_instMod.waterstatebulk_inst, clm_instMod.temperature_inst) = SoilInit(
    fin_clm, soil_init_time_indx, bounds.begc, bounds.endc,
    clm_instMod.soilstate_inst, clm_instMod.waterstatebulk_inst,
    clm_instMod.temperature_inst)

os.makedirs(dirout, exist_ok=True)
clm_time_manager.itim = 1
curr_calday_ts1 = get_curr_calday(offset=0)
time_indx = round(
    (curr_calday_ts1 - start_calday_clm) * 86400.0 / float(dtstep_sec)) + 1

(clm_instMod.atm2lnd_inst, clm_instMod.wateratm2lndbulk_inst,
 clm_instMod.frictionvel_inst) = TowerMetCurr(
    fin_tower, 1, tower_num, bounds.begp, bounds.endp,
    clm_instMod.atm2lnd_inst, clm_instMod.wateratm2lndbulk_inst,
    clm_instMod.frictionvel_inst)

if MLclm_varctl.met_type == 3:
    clm_instMod.mlcanopy_inst = TowerMetNext(
        fin_tower, min(2, ntimes), bounds.begp, bounds.endp,
        clm_instMod.mlcanopy_inst)

# ── Warmup ────────────────────────────────────────────────────────────────────
print("expt_init: running warmup timestep...", flush=True)
ModelAdvance(bounds, time_indx, fin_tower, fin_clm)
print("expt_init: warmup done.", flush=True)

# ── Set Euler, 1 sub-step ─────────────────────────────────────────────────────
MLclm_varctl.runge_kutta_type = 10
MLclm_varctl.dtime_ml = float(clm_time_manager.dtstep)

# ── Build forward_fn ─────────────────────────────────────────────────────────
mlcanopy_inst = clm_instMod.mlcanopy_inst
filt          = _clm_driver_mod.filter
num_evp       = filt.num_exposedvegp
evp_list      = [int(v) for v in filt.exposedvegp[:num_evp]]

from multilayer_canopy.MLCanopyFluxesMod import make_clm_ml_forward, MLCanopyFluxes
from multilayer_canopy.MLclm_varctl import GridInfo

forward_fn = make_clm_ml_forward(
    mlcanopy_inst_template      = mlcanopy_inst,
    bounds                      = bounds,
    num_exposedvegp             = num_evp,
    filter_exposedvegp          = evp_list,
    atm2lnd_inst                = clm_instMod.atm2lnd_inst,
    canopystate_inst            = clm_instMod.canopystate_inst,
    soilstate_inst              = clm_instMod.soilstate_inst,
    temperature_inst            = clm_instMod.temperature_inst,
    waterstatebulk_inst         = clm_instMod.waterstatebulk_inst,
    waterfluxbulk_inst          = clm_instMod.waterfluxbulk_inst,
    energyflux_inst             = clm_instMod.energyflux_inst,
    frictionvel_inst            = clm_instMod.frictionvel_inst,
    surfalb_inst                = clm_instMod.surfalb_inst,
    solarabs_inst               = clm_instMod.solarabs_inst,
    wateratm2lndbulk_inst       = clm_instMod.wateratm2lndbulk_inst,
    waterdiagnosticbulk_inst    = clm_instMod.waterdiagnosticbulk_inst,
)

_p    = int(evp_list[0])
grid  = GridInfo(
    p    = _p,
    ncan = int(mlcanopy_inst.ncan_canopy[_p]),
    ntop = int(mlcanopy_inst.ntop_canopy[_p]),
    nbot = int(mlcanopy_inst.nbot_canopy[_p]),
)

# Expose MLCanopyFluxes constructor arguments for multi-output wrappers
_mlcf_kwargs = dict(
    bounds                   = bounds,
    num_exposedvegp          = num_evp,
    filter_exposedvegp       = evp_list,
    atm2lnd_inst             = clm_instMod.atm2lnd_inst,
    canopystate_inst         = clm_instMod.canopystate_inst,
    soilstate_inst           = clm_instMod.soilstate_inst,
    temperature_inst         = clm_instMod.temperature_inst,
    waterstatebulk_inst      = clm_instMod.waterstatebulk_inst,
    waterfluxbulk_inst       = clm_instMod.waterfluxbulk_inst,
    energyflux_inst          = clm_instMod.energyflux_inst,
    frictionvel_inst         = clm_instMod.frictionvel_inst,
    surfalb_inst             = clm_instMod.surfalb_inst,
    solarabs_inst            = clm_instMod.solarabs_inst,
    wateratm2lndbulk_inst    = clm_instMod.wateratm2lndbulk_inst,
    waterdiagnosticbulk_inst = clm_instMod.waterdiagnosticbulk_inst,
    grid                     = grid,
    _o2ref_py                = float(mlcanopy_inst.o2ref_forcing[_p]),
)

print(f"expt_init: ready. patch={_p}, ncan={grid.ncan}", flush=True)

# Export atmospheric forcing instances so experiments can scale them directly.
# NOTE: scaling mlcanopy_inst.swskyb_forcing does NOT affect the radiation
# computation because MLCanopyFluxes.__init__ overwrites swskyb_cur_forcing from
# atm2lnd_inst.forc_solad_downscaled_col (lines 910-921 of MLCanopyFluxesMod).
# Correct gradients require scaling atm2lnd_inst and wateratm2lndbulk_inst.
atm2lnd_inst          = clm_instMod.atm2lnd_inst
wateratm2lndbulk_inst = clm_instMod.wateratm2lndbulk_inst

from multilayer_canopy.MLclm_varpar import isun, isha


def compute_gpp(inst, p: int, ncan: int) -> jnp.ndarray:
    """Compute canopy GPP from agross_leaf (differentiable in diff mode).

    In diff mode, MLCanopyFluxes skips _CanopyFluxesDiagnostics, so
    gppveg_canopy is never updated.  agross_leaf IS updated by
    LeafPhotosynthesis inside _physics_step_fn and is the correct
    differentiable proxy for GPP.

    Units: proportional to umol CO2 m-2 s-1 (sum over layers, sun+shade).
    """
    agross_sun = inst.agross_leaf[p, 1:ncan + 1, isun]
    agross_sha = inst.agross_leaf[p, 1:ncan + 1, isha]
    fracsun    = inst.fracsun_profile[p, 1:ncan + 1]
    dpai       = inst.dpai_profile[p, 1:ncan + 1]
    return jnp.sum(
        (agross_sun * fracsun + agross_sha * (1.0 - fracsun)) * dpai
    )


def compute_le(inst, p: int, ncan: int) -> jnp.ndarray:
    """Compute canopy LE proxy from lhleaf_leaf (differentiable in diff mode).

    lhleaf_leaf is set by FluxProfileSolution (MLFluxProfileSolutionMod.py:453)
    inside the RK inner loop — available in diff mode even though
    _CanopyFluxesDiagnostics is skipped.

    Units: W m-2 (weighted sum over layers, sun+shade).
    """
    lhleaf_sun = inst.lhleaf_leaf[p, 1:ncan + 1, isun]
    lhleaf_sha = inst.lhleaf_leaf[p, 1:ncan + 1, isha]
    fracsun    = inst.fracsun_profile[p, 1:ncan + 1]
    dpai       = inst.dpai_profile[p, 1:ncan + 1]
    return jnp.sum(
        (lhleaf_sun * fracsun + lhleaf_sha * (1.0 - fracsun)) * dpai
    )


def compute_h(inst, p: int, ncan: int) -> jnp.ndarray:
    """Compute canopy sensible heat proxy from shleaf_leaf (differentiable in diff mode).

    shleaf_leaf is set by LeafFluxes (MLLeafFluxesMod.py:177) and
    FluxProfileSolution (MLFluxProfileSolutionMod.py:452) inside the RK
    inner loop — available in diff mode even though _CanopyFluxesDiagnostics
    is skipped.

    Gradient path: shleaf = 2 * cpair * (T_leaf - T_air) * g_bh
    T_leaf is solved by the leaf energy balance; g_bh depends on wind speed
    and leaf boundary layer conductance. Both flow through the physics step.

    Units: W m-2 (weighted sum over canopy layers, sun+shade).
    """
    shleaf_sun = inst.shleaf_leaf[p, 1:ncan + 1, isun]
    shleaf_sha = inst.shleaf_leaf[p, 1:ncan + 1, isha]
    fracsun    = inst.fracsun_profile[p, 1:ncan + 1]
    dpai       = inst.dpai_profile[p, 1:ncan + 1]
    return jnp.sum(
        (shleaf_sun * fracsun + shleaf_sha * (1.0 - fracsun)) * dpai
    )
