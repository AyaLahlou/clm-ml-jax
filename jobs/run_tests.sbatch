#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=run_all_tests_clm-ml-v1
#SBATCH --output=/burg-archive/home/mck2199/clm-ml-jax/jobs/logs/run_all_tests_clm-ml-v1_%j.out
#SBATCH --error=/burg-archive/home/mck2199/clm-ml-jax/jobs/logs/run_all_tests_clm-ml-v1_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mck2199@columbia.edu

# Activate virtual environment
source /burg-archive/home/mck2199/clm-ml-jax/jax-agents/.venv/bin/activate

# Navigate to project root
cd /burg-archive/home/mck2199/clm-ml-jax

# Set PYTHONPATH to include src directory
export PYTHONPATH=/burg-archive/home/mck2199/clm-ml-jax/src:$PYTHONPATH

# Run all tests with verbose output
pytest tests/ -v --tb=short --color=yes --maxfail=5

echo "All tests completed"