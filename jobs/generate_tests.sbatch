#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=generate_tests_clm-ml-v1
#SBATCH --output=/burg-archive/home/mck2199/clm-ml-jax/jobs/logs/generate_tests_clm-ml-v1_%j.out
#SBATCH --error=/burg-archive/home/mck2199/clm-ml-jax/jobs/logs/generate_tests_clm-ml-v1_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mck2199@columbia.edu

source /burg-archive/home/mck2199/clm-ml-jax/jax-agents/.venv/bin/activate
cd /burg-archive/home/mck2199/clm-ml-jax/jax-agents/

./run_translation_workflow.sh --test --modules "shr_file_mod,shr_orb_mod,shr_kind_mod,WaterStateType,SoilStateType,FrictionVelocityMod,SurfaceAlbedoType,WaterFluxType,TemperatureType,SolarAbsorbedType,EnergyFluxType,SoilStateInitTimeConstMod,SurfaceAlbedoMod,SoilWaterMovementMod,SurfaceResistanceMod,histFileMod,pftconMod,PatchType,ColumnType,atm2lndType,GridcellType,initVerticalMod,initSubgridMod,filterMod,clm_instMod,initGridCellsMod,clm_driver,clm_initializeMod,clm_varpar,clm_varcon,restUtilMod,spmdMod"

echo "Test generation completed"