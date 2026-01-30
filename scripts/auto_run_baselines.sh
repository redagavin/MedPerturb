#!/bin/bash
# ABOUTME: Orchestration script for calibrated baseline pipeline
# ABOUTME: Runs Steps 1-3: generate baselines, create dataset, evaluate model

set -e

MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
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

MEDPERTURB_DIR="/scratch/yang.zih/cot_faithfulness/MedPerturb"
cd "${MEDPERTURB_DIR}"

echo "======================================"
echo "Calibrated Baseline Pipeline"
echo "======================================"
echo "Model: ${MODEL}"
echo "Test mode: ${TEST_MODE}"
echo "Started at: $(date)"
echo ""

source ~/.bashrc
conda activate cot

mkdir -p results
mkdir -p logs
mkdir -p checkpoints

# ==========================================
# STEP 1: Generate Calibrated Baselines
# ==========================================
echo "======================================"
echo "STEP 1: Generate Calibrated Baselines"
echo "======================================"

BASELINE_ARGS="--dataset data.csv --output results/baselines_v2.json --model ${MODEL}"
if [ "$TEST_MODE" = true ]; then
    BASELINE_ARGS="${BASELINE_ARGS} --sample_size 5"
fi

if [ -f "results/baselines_v2.json" ]; then
    echo "Baselines file exists, checking completeness..."
    BASELINE_COUNT=$(python -c "import json; print(len(json.load(open('results/baselines_v2.json'))))")
    EXPECTED=600
    if [ "$TEST_MODE" = true ]; then
        EXPECTED=5
    fi
    if [ "$BASELINE_COUNT" -ge "$EXPECTED" ]; then
        echo "Baselines complete (${BASELINE_COUNT}/${EXPECTED})"
    else
        echo "Baselines incomplete (${BASELINE_COUNT}/${EXPECTED}), resuming..."
        srun --partition=177huntington --cpus-per-task=8 --mem=64G --time=14-00:00:00 \
            bash -c "source ~/.bashrc && conda activate cot && python code/generate_baselines_v2.py ${BASELINE_ARGS}"
    fi
else
    echo "Generating calibrated baselines..."
    srun --partition=177huntington --cpus-per-task=8 --mem=64G --time=14-00:00:00 \
        bash -c "source ~/.bashrc && conda activate cot && python code/generate_baselines_v2.py ${BASELINE_ARGS}"
fi

echo ""
echo "Step 1 complete!"
echo ""

# ==========================================
# STEP 2: Create Extended Dataset
# ==========================================
echo "======================================"
echo "STEP 2: Create Extended Dataset"
echo "======================================"

srun --partition=177huntington --cpus-per-task=4 --mem=32G --time=1:00:00 \
    bash -c "source ~/.bashrc && conda activate cot && python code/create_extended_dataset.py \
        --dataset data.csv \
        --baselines results/baselines_v2.json \
        --output data_with_baselines.csv"

echo ""
echo "Step 2 complete!"
echo ""

# ==========================================
# STEP 3: Evaluate Model on Baselines
# ==========================================
echo "======================================"
echo "STEP 3: Evaluate Model on Baselines"
echo "======================================"

if [ "$TEST_MODE" = true ]; then
    # Test mode: single GPU
    srun --partition=177huntington --gres=gpu:a100:1 --cpus-per-task=8 --mem=80G --time=2:00:00 \
        bash -c "source ~/.bashrc && conda activate cot && python code/evaluate_baselines.py \
            --model ${MODEL} \
            --dataset data_with_baselines.csv \
            --gpu_id 0 \
            --total_gpus 1"
else
    # Production: 4x H200 via sbatch
    sbatch slurm/run_baseline_eval.sbatch "${MODEL}" "data_with_baselines.csv"
    echo "Submitted evaluation job. Monitor with: squeue -u \$USER"
fi

echo ""
echo "======================================"
echo "Pipeline launched!"
echo "======================================"
echo "Completed at: $(date)"
