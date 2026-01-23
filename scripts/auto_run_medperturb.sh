#!/bin/bash
# ABOUTME: Automated MedPerturb replication pipeline with calibrated baselines
# ABOUTME: Runs perturbation generation, baseline generation, and evaluation with auto-relaunch

set -e

# Configuration
MEDPERTURB_DIR="/scratch/yang.zih/cot_faithfulness/MedPerturb"
DATASET="${MEDPERTURB_DIR}/data.csv"
PERTURBATIONS_DIR="${MEDPERTURB_DIR}/results/perturbations"
BASELINES_FILE="${MEDPERTURB_DIR}/results/baselines.json"
CHECKPOINT_DIR="${MEDPERTURB_DIR}/checkpoints"
MODEL=${1:-"meta-llama/Llama-3.1-70B-Instruct"}
MODEL_SHORT=$(echo $MODEL | sed 's/.*\///' | tr '[:upper:]' '[:lower:]' | tr '-' '_')
TOTAL_GPUS=4
SBATCH_SCRIPT="${MEDPERTURB_DIR}/slurm/run_evaluation.sbatch"
JOB_NAME="medperturb_eval"
CHECK_INTERVAL=300  # Check every 5 minutes

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

    # Create Python script for batch perturbation
    python3 << EOF
import pandas as pd
import json
import sys
sys.path.insert(0, '${MEDPERTURB_DIR}/code')
from perturb_data import ClinicalContextPerturber

# Load dataset
df = pd.read_csv('${DATASET}')
print(f"  Loaded {len(df)} samples")

# Initialize perturber
perturber = ClinicalContextPerturber()

# Generate perturbations
results = {}
for idx, row in df.iterrows():
    if idx % 10 == 0:
        print(f"  Processing {idx}/{len(df)}...")

    context_id = str(row['context_id'])
    original = row['clinical_context']

    try:
        perturbed = perturber.perturb_context(
            text=original,
            dataset_type='askadocs',
            perturbation_type='${ptype}',
            variant='${variant}'
        )
        results[context_id] = {
            'original': original,
            'perturbed_text': perturbed
        }
    except Exception as e:
        print(f"  Warning: Failed for {context_id}: {e}")
        results[context_id] = {
            'original': original,
            'perturbed_text': original  # Fallback to original
        }

# Save results
with open('${output_file}', 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Saved {len(results)} perturbations to ${output_file}")
EOF
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

if [ -f "${BASELINES_FILE}" ]; then
    echo "Baselines file exists: ${BASELINES_FILE}"
    echo "Checking if complete..."

    BASELINE_COUNT=$(python3 -c "import json; d=json.load(open('${BASELINES_FILE}')); print(len(d))")
    DATASET_COUNT=$(python3 -c "import pandas as pd; print(len(pd.read_csv('${DATASET}')))")

    if [ "$BASELINE_COUNT" -ge "$DATASET_COUNT" ]; then
        echo "Baselines complete (${BASELINE_COUNT}/${DATASET_COUNT})"
    else
        echo "Baselines incomplete (${BASELINE_COUNT}/${DATASET_COUNT}), resuming..."
        python3 ${MEDPERTURB_DIR}/code/generate_baselines.py \
            --dataset "${DATASET}" \
            --perturbations_dir "${PERTURBATIONS_DIR}" \
            --output "${BASELINES_FILE}" \
            --model "${MODEL}"
    fi
else
    echo "Generating calibrated baselines..."
    python3 ${MEDPERTURB_DIR}/code/generate_baselines.py \
        --dataset "${DATASET}" \
        --perturbations_dir "${PERTURBATIONS_DIR}" \
        --output "${BASELINES_FILE}" \
        --model "${MODEL}"
fi

echo ""
echo "Baseline generation complete!"
echo ""

# ==========================================
# STEP 3: Run Model Evaluation (with auto-relaunch)
# ==========================================
echo "=========================================="
echo "STEP 3: Running Model Evaluation"
echo "=========================================="

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
