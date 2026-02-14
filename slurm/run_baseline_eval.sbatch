#!/bin/bash
#SBATCH --job-name=baseline_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=8:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/baseline_eval_%A_%a.out
#SBATCH --error=logs/baseline_eval_%A_%a.err
#SBATCH --signal=INT@300

# ABOUTME: SLURM script for baseline evaluation with data-parallel array jobs
# ABOUTME: 4 array tasks, each with 1 H200 GPU, 8h time limit

MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
DATASET=${2:-"data_with_baselines.csv"}

echo "======================================"
echo "Baseline Evaluation"
echo "======================================"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Total Tasks: ${SLURM_ARRAY_TASK_COUNT}"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Started at: $(date)"
echo ""

nvidia-smi

# Setup
source ~/.bashrc
conda activate cot

# SLURM copies scripts to temp location, so BASH_SOURCE won't work
MEDPERTURB_DIR="/scratch/yang.zih/cot_faithfulness/MedPerturb"
cd "${MEDPERTURB_DIR}"

mkdir -p logs
mkdir -p checkpoints/baseline_eval

# Run evaluation for this GPU's shard (4 GPUs matches --array=0-3)
python code/evaluate_baselines.py \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --gpu_id ${SLURM_ARRAY_TASK_ID} \
    --total_gpus 4 \
    --checkpoint_dir checkpoints/baseline_eval \
    --checkpoint_freq 10

EXIT_CODE=$?

# Create completion marker (4 matches --array=0-3 and orchestration TOTAL_GPUS)
if [ $EXIT_CODE -eq 0 ]; then
    MODEL_SHORT=$(echo $MODEL | sed 's/.*\///' | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    touch "checkpoints/baseline_eval/${MODEL_SHORT}_gpu${SLURM_ARRAY_TASK_ID}_of_4_COMPLETE"
    echo "Created completion marker"
fi

echo ""
echo "======================================"
echo "Completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "======================================"

exit ${EXIT_CODE}
