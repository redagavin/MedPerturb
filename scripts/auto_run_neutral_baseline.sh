#!/bin/bash
# ABOUTME: Orchestration script for neutral baseline pipeline
# ABOUTME: Runs Steps 1-4: generate dataset, evaluate model, merge results, run analysis

set -e

# Defaults
MODEL="meta-llama/Llama-3.1-8B-Instruct"
TEST_MODE=false

# Parse flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            shift
            ;;
        *)
            MODEL="$1"
            shift
            ;;
    esac
done

# Use script's directory as base (works from worktree or main repo)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDPERTURB_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${MEDPERTURB_DIR}"

# Configuration
TOTAL_GPUS=4
CHECK_INTERVAL=300  # 5 minutes
JOB_NAME="neutral_bl_eval"
MODEL_SHORT=$(echo $MODEL | sed 's/.*\///' | tr '[:upper:]' '[:lower:]' | tr '-' '_')

echo "======================================"
echo "Neutral Baseline Pipeline"
echo "======================================"
echo "Model: ${MODEL}"
echo "Test mode: ${TEST_MODE}"
echo "Started at: $(date)"
echo ""

source ~/.bashrc
conda activate cot

mkdir -p results
mkdir -p logs
mkdir -p checkpoints/neutral_baseline_eval

# ==========================================
# STEP 1: Generate Neutral Baseline Dataset
# ==========================================
echo "======================================"
echo "STEP 1: Generate Neutral Baseline Dataset"
echo "======================================"

if [ -f "data_with_neutral_baseline.csv" ]; then
    echo "data_with_neutral_baseline.csv already exists, skipping generation"
else
    echo "Generating neutral baseline dataset..."
    srun --partition=frink --cpus-per-task=4 --mem=32G --time=1:00:00 \
        bash -c "source ~/.bashrc && conda activate cot && python code/generate_neutral_baseline.py \
            --dataset data.csv \
            --output data_with_neutral_baseline.csv"
fi

echo ""
echo "Step 1 complete!"
echo ""

# ==========================================
# STEP 2: Evaluate Model on Neutral Baselines
# ==========================================
echo "======================================"
echo "STEP 2: Evaluate Model on Neutral Baselines"
echo "======================================"

if [ "$TEST_MODE" = true ]; then
    # Test mode: single GPU, synchronous
    srun --partition=177huntington --gres=gpu:a100:1 --cpus-per-task=8 --mem=80G --time=2:00:00 \
        bash -c "source ~/.bashrc && conda activate cot && python code/evaluate_neutral_baseline.py \
            --model ${MODEL} \
            --dataset data_with_neutral_baseline.csv \
            --gpu_id 0 \
            --total_gpus 1"
else
    # Production mode: 4x H200 array jobs with auto-relaunch

    # Function to check if all GPUs complete
    check_completion() {
        COMPLETE_COUNT=$(ls -1 checkpoints/neutral_baseline_eval/${MODEL_SHORT}_gpu*_of_${TOTAL_GPUS}_COMPLETE 2>/dev/null | wc -l)
        if [ "$COMPLETE_COUNT" -eq $TOTAL_GPUS ]; then
            return 0  # All complete
        fi
        return 1  # Not complete
    }

    # Function to check if jobs are running
    check_jobs_running() {
        RUNNING_JOBS=$(squeue -u $USER -n ${JOB_NAME} -h 2>/dev/null | wc -l)
        if [ "$RUNNING_JOBS" -gt 0 ]; then
            return 0  # Jobs running
        fi
        return 1  # No jobs running
    }

    # Main evaluation loop
    ITERATION=1
    while true; do
        echo "[$(date)] Evaluation iteration ${ITERATION}"

        # Check if all GPUs complete
        if check_completion; then
            COMPLETE_COUNT=$(ls -1 checkpoints/neutral_baseline_eval/${MODEL_SHORT}_gpu*_of_${TOTAL_GPUS}_COMPLETE 2>/dev/null | wc -l)
            echo "All ${COMPLETE_COUNT}/${TOTAL_GPUS} GPUs complete!"
            break
        fi

        # Check progress
        COMPLETE_COUNT=$(ls -1 checkpoints/neutral_baseline_eval/${MODEL_SHORT}_gpu*_of_${TOTAL_GPUS}_COMPLETE 2>/dev/null | wc -l)
        echo "Progress: ${COMPLETE_COUNT}/${TOTAL_GPUS} GPUs complete"

        # Check if jobs are running
        if check_jobs_running; then
            echo "Jobs currently running:"
            squeue -u $USER -n ${JOB_NAME}
            echo "Waiting ${CHECK_INTERVAL}s before next check..."
            sleep $CHECK_INTERVAL
        else
            echo "No jobs running - launching/relaunching..."

            # Submit array job
            SUBMITTED_JOB_ID=$(sbatch slurm/run_neutral_baseline_eval.sbatch "${MODEL}" "data_with_neutral_baseline.csv" | awk '{print $NF}')
            echo "Submitted job: ${SUBMITTED_JOB_ID}"

            # Wait for jobs to start
            sleep 60

            # Check if jobs started
            if check_jobs_running; then
                echo "Jobs started successfully"
                squeue -u $USER -n ${JOB_NAME}
            else
                echo "Warning: Jobs may not have started"
                squeue -u $USER -n ${JOB_NAME}
            fi

            echo "Waiting ${CHECK_INTERVAL}s before next check..."
            sleep $CHECK_INTERVAL
        fi

        ITERATION=$((ITERATION + 1))
    done
fi

# ==========================================
# STEP 3: Merge Results
# ==========================================
echo "======================================"
echo "STEP 3: Merge Results"
echo "======================================"

python << EOF
import pandas as pd
import glob
import tempfile
import shutil
import os

df = pd.read_csv('data_with_neutral_baseline.csv')
checkpoint_files = glob.glob('checkpoints/neutral_baseline_eval/neutral_baseline_eval_gpu*.csv')
print(f"Found {len(checkpoint_files)} checkpoint files")

total_merged = 0
for f in checkpoint_files:
    results = pd.read_csv(f)
    for _, row in results.iterrows():
        matches = df[df['Index'] == row['Index']].index
        if len(matches) > 0:
            idx = matches[0]
            for col in results.columns:
                if col != 'Index':
                    df.loc[idx, col] = row[col]
            total_merged += 1

# Atomic write to prevent corruption on crash
output_path = 'data_with_neutral_baseline.csv'
with tempfile.NamedTemporaryFile('w', delete=False, dir='.', suffix='.tmp') as f:
    df.to_csv(f, index=False)
    temp_path = f.name
shutil.move(temp_path, output_path)
print(f"Merged {total_merged} neutral baseline evaluations into data_with_neutral_baseline.csv")
EOF

echo ""

# ==========================================
# STEP 4: Run Analysis
# ==========================================
echo "======================================"
echo "STEP 4: Run Analysis"
echo "======================================"

python case_studies/neutral_baseline_analysis.py \
    --dataset data_with_neutral_baseline.csv \
    --output results/neutral_baseline_analysis.xlsx

echo ""
echo "======================================"
echo "Pipeline Complete!"
echo "======================================"
echo "Dataset: data_with_neutral_baseline.csv"
echo "Analysis: results/neutral_baseline_analysis.xlsx"
echo "Completed at: $(date)"
