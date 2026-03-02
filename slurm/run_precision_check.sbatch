#!/bin/bash
#SBATCH --job-name=precision_check
#SBATCH --partition=177huntington
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/precision_check_%A_%a.out
#SBATCH --error=logs/precision_check_%A_%a.err
#SBATCH --signal=INT@300

# ABOUTME: SLURM script for precision check evaluation with age swap negative control
# ABOUTME: 4 data-parallel array tasks with Llama-3-8B

MEDPERTURB_DIR="/scratch/yang.zih/cot_faithfulness/MedPerturb"

echo "========================================"
echo "Precision Check: Age Swap Evaluation"
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

python code/precision_check_evaluate.py \
    --model "${MODEL}" \
    --dataset "data_with_baselines.csv" \
    --age_swap "results/precision_check_age_swap.json" \
    --baselines "results/precision_check_baselines.json" \
    --output "results/precision_check_eval_${SLURM_ARRAY_TASK_ID}.json" \
    --checkpoint_dir "checkpoints/precision_check"

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}
