#!/bin/bash
#SBATCH --job-name=sanity_check
#SBATCH --partition=177huntington
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/sanity_check_%A_%a.out
#SBATCH --error=logs/sanity_check_%A_%a.err
#SBATCH --signal=INT@300

# ABOUTME: SLURM script for sanity check gender question evaluation
# ABOUTME: 4 data-parallel array tasks with Llama-3-8B

MEDPERTURB_DIR="/scratch/yang.zih/cot_faithfulness/MedPerturb"

echo "========================================"
echo "Sanity Check: Gender Question Evaluation"
echo "========================================"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Total Tasks: ${SLURM_ARRAY_TASK_COUNT}"
echo "Start time: $(date)"
echo "Node: $(hostname)"

nvidia-smi

source ~/.bashrc
conda activate cot

MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}

echo "Model: ${MODEL}"
echo ""

cd ${MEDPERTURB_DIR}

python code/sanity_check_evaluate.py \
    --model "${MODEL}" \
    --dataset "data_with_baselines.csv" \
    --output "results/sanity_check_eval_${SLURM_ARRAY_TASK_ID}.json" \
    --checkpoint_dir "checkpoints/sanity_check" \
    --checkpoint_freq 10

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}
