#!/bin/bash
#SBATCH --job-name=medperturb_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=8:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/medperturb_eval_%A_%a.out
#SBATCH --error=logs/medperturb_eval_%A_%a.err
#SBATCH --signal=INT@300

# ABOUTME: SLURM script for MedPerturb batch evaluation
# ABOUTME: 4 data-parallel array tasks with FP8 quantization for 70B

MEDPERTURB_DIR="/scratch/yang.zih/cot_faithfulness/MedPerturb"

echo "========================================"
echo "MedPerturb Model Evaluation"
echo "========================================"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Total Tasks: ${SLURM_ARRAY_TASK_COUNT}"
echo "Start time: $(date)"
echo "Node: $(hostname)"

nvidia-smi

# Activate conda environment
source ~/.bashrc
conda activate cot

# Set model (passed as argument or default)
MODEL=${1:-"meta-llama/Llama-3.1-70B-Instruct"}

echo "Model: ${MODEL}"
echo ""

cd ${MEDPERTURB_DIR}

# Run batch evaluation
python code/batch_evaluate.py \
    --model "${MODEL}" \
    --dataset "data.csv" \
    --perturbations_dir "results/perturbations" \
    --baselines "results/baselines.json" \
    --output "results/evaluation_${SLURM_ARRAY_TASK_ID}.json" \
    --checkpoint_dir "checkpoints" \
    --checkpoint_freq 10

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}
