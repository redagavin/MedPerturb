#!/bin/bash
#SBATCH --job-name=baseline_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --output=logs/baseline_eval_%j.out
#SBATCH --error=logs/baseline_eval_%j.err
#SBATCH --signal=INT@300

# ABOUTME: SLURM script for baseline evaluation with 4 H200 GPUs
# ABOUTME: Runs data parallel evaluation across GPUs

MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
DATASET=${2:-"data_with_baselines.csv"}

echo "======================================"
echo "Baseline Evaluation"
echo "======================================"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "GPUs: 4x H200"
echo "Started at: $(date)"
echo ""

# Setup
source ~/.bashrc
conda activate cot
cd /scratch/yang.zih/cot_faithfulness/MedPerturb

mkdir -p logs
mkdir -p checkpoints/baseline_eval

# Launch 4 parallel processes
for GPU_ID in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python code/evaluate_baselines.py \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --gpu_id ${GPU_ID} \
        --total_gpus 4 \
        --checkpoint_dir checkpoints/baseline_eval \
        --checkpoint_freq 10 &
done

# Wait for all to complete
wait

echo ""
echo "======================================"
echo "All GPUs complete!"
echo "Completed at: $(date)"
echo "======================================"

# Merge results
python << EOF
import pandas as pd
import glob

# Load original dataset
df = pd.read_csv('${DATASET}')

# Load checkpoint files
checkpoint_files = glob.glob('checkpoints/baseline_eval/baseline_eval_gpu*.csv')
print(f"Found {len(checkpoint_files)} checkpoint files")

# Merge results
for f in checkpoint_files:
    results = pd.read_csv(f)
    for _, row in results.iterrows():
        idx = df[df['Index'] == row['Index']].index[0]
        for col in results.columns:
            if col != 'Index':
                df.loc[idx, col] = row[col]

# Save
df.to_csv('${DATASET}', index=False)
print(f"Updated {DATASET} with {len(results)} baseline evaluations")
EOF
