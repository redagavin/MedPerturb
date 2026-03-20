#!/bin/bash
#SBATCH --job-name=simulation
#SBATCH --partition=177huntington
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/simulation_%j.out
#SBATCH --error=logs/simulation_%j.err
#SBATCH --signal=INT@300

# ABOUTME: SLURM script for Monte Carlo power simulation of all 5 metrics
# ABOUTME: CPU-only job, estimates false positive rate and power

echo "========================================"
echo "MI Power Simulation (Monte Carlo)"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

source ~/.bashrc
conda activate cot

MEDPERTURB_DIR="/scratch/yang.zih/cot_faithfulness/MedPerturb"
cd "${MEDPERTURB_DIR}"

mkdir -p results/simulation

python -u code/simulation.py \
    --n-simulations 1000 \
    --n-bootstrap 1000 \
    --beta1-values 1.0 2.0 3.0 \
    --beta-gender-max 4.0 \
    --beta-gender-step 0.1 \
    --sigma-values 0.0 0.1 0.3 0.5 \
    --n-cases 100 \
    --seed 42 \
    --n-workers 32 \
    --output-dir results/simulation

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}
