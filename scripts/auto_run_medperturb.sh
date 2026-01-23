#!/bin/bash
# ABOUTME: Automated MedPerturb replication pipeline with calibrated baselines
# ABOUTME: Runs perturbation generation, baseline generation, and evaluation with auto-relaunch

set -e

# Parse arguments
TEST_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            shift
            ;;
        *)
            MODEL_ARG="$1"
            shift
            ;;
    esac
done

# Configuration
MEDPERTURB_DIR="/scratch/yang.zih/cot_faithfulness/MedPerturb"
DATASET="${MEDPERTURB_DIR}/data.csv"
PERTURBATIONS_DIR="${MEDPERTURB_DIR}/results/perturbations"
BASELINES_FILE="${MEDPERTURB_DIR}/results/baselines.json"
CHECKPOINT_DIR="${MEDPERTURB_DIR}/checkpoints"
SBATCH_SCRIPT="${MEDPERTURB_DIR}/slurm/run_evaluation.sbatch"
JOB_NAME="medperturb_eval"
CHECK_INTERVAL=300  # Check every 5 minutes

# Test mode settings
if [ "$TEST_MODE" = true ]; then
    MODEL=${MODEL_ARG:-"meta-llama/Llama-3.1-8B-Instruct"}
    SAMPLE_SIZE=2
    TOTAL_GPUS=1
    PARTITION="177huntington"
    GPU_TYPE="a100"
    echo "=========================================="
    echo "TEST MODE ENABLED"
    echo "=========================================="
    echo "  Model: ${MODEL} (8B for testing)"
    echo "  Sample size: ${SAMPLE_SIZE}"
    echo "  Partition: ${PARTITION}"
    echo "  GPU: ${GPU_TYPE}"
    echo ""
else
    MODEL=${MODEL_ARG:-"meta-llama/Llama-3.1-70B-Instruct"}
    SAMPLE_SIZE=""
    TOTAL_GPUS=4
    PARTITION="gpu"
    GPU_TYPE="h200"
fi

MODEL_SHORT=$(echo $MODEL | sed 's/.*\///' | tr '[:upper:]' '[:lower:]' | tr '-' '_')

# Perturbation types
PERTURBATION_TYPES=("gender:swap" "gender:remove" "stylistic:uncertain" "stylistic:colorful")

echo "=========================================="
echo "MedPerturb Replication Pipeline"
echo "=========================================="
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Started at: $(date)"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate cot

# Create directories
mkdir -p "${PERTURBATIONS_DIR}"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${MEDPERTURB_DIR}/results"
mkdir -p "${MEDPERTURB_DIR}/logs"

# ==========================================
# STEP 1: Generate Perturbations
# ==========================================
echo "=========================================="
echo "STEP 1: Generating Perturbations"
echo "=========================================="

generate_perturbations() {
    local ptype=$1
    local variant=$2
    local output_file="${PERTURBATIONS_DIR}/${ptype}_${variant}.json"

    # Skip if already done
    if [ -f "${output_file}" ]; then
        echo "  ${ptype}:${variant} - Already exists, skipping"
        return 0
    fi

    echo "  Generating ${ptype}:${variant}..."

    # Create temporary Python script on shared filesystem (follows perturb_data.py main() pattern)
    local tmp_script="${MEDPERTURB_DIR}/tmp_perturb_${ptype}_${variant}.py"
    cat > "${tmp_script}" << 'PYTHON_EOF'
import pandas as pd
import json
import sys
import os

medperturb_dir = os.environ.get('MEDPERTURB_DIR')
sys.path.insert(0, f'{medperturb_dir}/code')
from perturb_data import ClinicalContextPerturber
from utils import setup_logging

logger = setup_logging()

# Get parameters from environment
dataset_path = os.environ.get('DATASET')
output_file = os.environ.get('OUTPUT_FILE')
model_name = os.environ.get('MODEL')
ptype = os.environ.get('PTYPE')
variant = os.environ.get('VARIANT')
sample_size = int(os.environ.get('SAMPLE_SIZE', 0))

# Load dataset
df = pd.read_csv(dataset_path)
logger.info(f"Loaded {len(df)} samples")

# Limit samples in test mode
if sample_size > 0:
    df = df.head(sample_size)
    logger.info(f"Limited to {len(df)} samples (test mode)")

# Initialize perturber (follows perturb_data.py ClinicalContextPerturber init)
perturber = ClinicalContextPerturber(model_name=model_name)

# Generate perturbations
results = {}
for idx, row in df.iterrows():
    if idx % 10 == 0:
        logger.info(f"Processing {idx}/{len(df)}...")

    context_id = str(row['context_id'])
    original = row['clinical_context']

    try:
        # Call perturb_context same as perturb_data.py main()
        perturbed = perturber.perturb_context(
            text=original,
            dataset_type='askadocs',
            perturbation_type=ptype,
            variant=variant
        )
        # Output format follows perturb_data.py save format
        results[context_id] = {
            'original': original,
            'perturbed': perturbed,
            'dataset': 'askadocs',
            'perturbation_type': ptype,
            'variant': variant
        }
    except Exception as e:
        logger.error(f"Failed for {context_id}: {e}")
        results[context_id] = {
            'original': original,
            'perturbed': original,
            'dataset': 'askadocs',
            'perturbation_type': ptype,
            'variant': variant
        }

# Save results
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

logger.info(f"Saved {len(results)} perturbations to {output_file}")
PYTHON_EOF

    # Export environment variables for the script
    export MEDPERTURB_DIR="${MEDPERTURB_DIR}"
    export DATASET="${DATASET}"
    export OUTPUT_FILE="${output_file}"
    export MODEL="${MODEL}"
    export PTYPE="${ptype}"
    export VARIANT="${variant}"
    export SAMPLE_SIZE="${SAMPLE_SIZE:-0}"

    # Run via srun in test mode (needs GPU), directly otherwise
    if [ "$TEST_MODE" = true ]; then
        srun --partition=${PARTITION} \
             --gres=gpu:${GPU_TYPE}:1 \
             --cpus-per-task=8 \
             --mem=80G \
             --time=1:00:00 \
             bash -c "source ~/.bashrc && conda activate cot && python ${tmp_script}"
    else
        python3 "${tmp_script}"
    fi

    rm -f "${tmp_script}"
}

# Generate all perturbation types
for ptype_variant in "${PERTURBATION_TYPES[@]}"; do
    IFS=':' read -r ptype variant <<< "$ptype_variant"
    generate_perturbations "$ptype" "$variant"
done

echo ""
echo "Perturbation generation complete!"
echo ""

# ==========================================
# STEP 2: Generate Calibrated Baselines
# ==========================================
echo "=========================================="
echo "STEP 2: Generating Calibrated Baselines"
echo "=========================================="

BASELINE_ARGS="--dataset ${DATASET} --perturbations_dir ${PERTURBATIONS_DIR} --output ${BASELINES_FILE} --model ${MODEL}"
if [ -n "$SAMPLE_SIZE" ]; then
    BASELINE_ARGS="${BASELINE_ARGS} --sample_size ${SAMPLE_SIZE}"
fi

if [ -f "${BASELINES_FILE}" ]; then
    echo "Baselines file exists: ${BASELINES_FILE}"
    echo "Checking if complete..."

    BASELINE_COUNT=$(python3 -c "import json; d=json.load(open('${BASELINES_FILE}')); print(len(d))")
    if [ -n "$SAMPLE_SIZE" ]; then
        DATASET_COUNT=$SAMPLE_SIZE
    else
        DATASET_COUNT=$(python3 -c "import pandas as pd; print(len(pd.read_csv('${DATASET}')))")
    fi

    if [ "$BASELINE_COUNT" -ge "$DATASET_COUNT" ]; then
        echo "Baselines complete (${BASELINE_COUNT}/${DATASET_COUNT})"
    else
        echo "Baselines incomplete (${BASELINE_COUNT}/${DATASET_COUNT}), resuming..."
        python3 ${MEDPERTURB_DIR}/code/generate_baselines.py ${BASELINE_ARGS}
    fi
else
    echo "Generating calibrated baselines..."
    python3 ${MEDPERTURB_DIR}/code/generate_baselines.py ${BASELINE_ARGS}
fi

echo ""
echo "Baseline generation complete!"
echo ""

# ==========================================
# STEP 3: Run Model Evaluation
# ==========================================
echo "=========================================="
echo "STEP 3: Running Model Evaluation"
echo "=========================================="

if [ "$TEST_MODE" = true ]; then
    # Test mode: Use srun for interactive testing
    echo "Running evaluation with srun (test mode)..."

    EVAL_ARGS="--model ${MODEL} --dataset data.csv --perturbations_dir results/perturbations --baselines results/baselines.json --output results/evaluation_test.json --checkpoint_dir checkpoints --checkpoint_freq 1 --gpu_id 0 --total_gpus 1 --sample_size ${SAMPLE_SIZE}"

    cd ${MEDPERTURB_DIR}

    srun --partition=${PARTITION} \
         --gres=gpu:${GPU_TYPE}:1 \
         --cpus-per-task=8 \
         --mem=80G \
         --time=1:00:00 \
         bash -c "source ~/.bashrc && conda activate cot && python code/batch_evaluate.py ${EVAL_ARGS}"

    echo ""
    echo "=========================================="
    echo "TEST MODE Complete!"
    echo "=========================================="
    echo "Results: ${MEDPERTURB_DIR}/results/evaluation_test.json"
    echo "Completed at: $(date)"
    echo "=========================================="
    exit 0
fi

# Production mode: Use sbatch with auto-relaunch

# Function to check if evaluation is complete
check_completion() {
    # Check if final merged output file exists
    if [ -f "${MEDPERTURB_DIR}/results/evaluation_merged_${MODEL_SHORT}.json" ]; then
        return 0  # Complete
    fi

    # Check if all GPU completion markers exist
    COMPLETE_COUNT=$(ls -1 ${CHECKPOINT_DIR}/${MODEL_SHORT}_gpu*_of_${TOTAL_GPUS}_COMPLETE 2>/dev/null | wc -l)
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
    else
        return 1  # No jobs running
    fi
}

# Main evaluation loop
ITERATION=1
while true; do
    echo "[$(date)] Evaluation iteration ${ITERATION}"

    # Check if all GPUs complete
    if check_completion; then
        COMPLETE_COUNT=$(ls -1 ${CHECKPOINT_DIR}/${MODEL_SHORT}_gpu*_of_${TOTAL_GPUS}_COMPLETE 2>/dev/null | wc -l)
        echo "All ${COMPLETE_COUNT}/${TOTAL_GPUS} GPUs complete!"
        break
    fi

    # Check progress
    COMPLETE_COUNT=$(ls -1 ${CHECKPOINT_DIR}/${MODEL_SHORT}_gpu*_of_${TOTAL_GPUS}_COMPLETE 2>/dev/null | wc -l)
    echo "Progress: ${COMPLETE_COUNT}/${TOTAL_GPUS} GPUs complete"

    # Check if jobs are running
    if check_jobs_running; then
        echo "Jobs currently running:"
        squeue -u $USER -n ${JOB_NAME}
        echo "Waiting ${CHECK_INTERVAL}s before next check..."
        sleep $CHECK_INTERVAL
    else
        echo "No jobs running - launching/relaunching..."

        # Submit job
        SUBMITTED_JOB_ID=$(sbatch ${SBATCH_SCRIPT} "${MODEL}" | awk '{print $NF}')
        echo "Submitted job: ${SUBMITTED_JOB_ID}"

        # Wait for jobs to start
        sleep 60

        # Check if jobs started successfully
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

# ==========================================
# STEP 4: Merge Results
# ==========================================
echo "=========================================="
echo "STEP 4: Merging Results"
echo "=========================================="

python3 << EOF
import json
import glob

# Find all evaluation result files
result_files = glob.glob('${MEDPERTURB_DIR}/results/evaluation_*.json')
result_files = [f for f in result_files if 'merged' not in f]

print(f"Found {len(result_files)} result files to merge")

# Merge all results
merged = []
for f in sorted(result_files):
    with open(f, 'r') as fp:
        data = json.load(fp)
        if isinstance(data, list):
            merged.extend(data)
        else:
            merged.append(data)

# Save merged results
output_path = '${MEDPERTURB_DIR}/results/evaluation_merged_${MODEL_SHORT}.json'
with open(output_path, 'w') as f:
    json.dump(merged, f, indent=2)

print(f"Merged {len(merged)} results to {output_path}")
EOF

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Perturbations: ${PERTURBATIONS_DIR}/"
echo "Baselines: ${BASELINES_FILE}"
echo "Evaluation: ${MEDPERTURB_DIR}/results/evaluation_merged_${MODEL_SHORT}.json"
echo "Completed at: $(date)"
echo "=========================================="
echo ""
echo "Next: Run the analysis notebook:"
echo "  jupyter notebook ${MEDPERTURB_DIR}/case_studies/case_study1_with_baseline.ipynb"
