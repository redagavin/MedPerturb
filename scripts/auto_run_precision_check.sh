#!/bin/bash
# ABOUTME: Automated precision check pipeline with auto-relaunch
# ABOUTME: Runs age swap evaluation with model-tagged shards and merge step

set -euo pipefail

MAX_ITERATIONS=50

# Defaults
MODEL="meta-llama/Llama-3.1-8B-Instruct"
TEST_MODE=false

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDPERTURB_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${MEDPERTURB_DIR}"

TOTAL_GPUS=4
CHECK_INTERVAL=300
JOB_NAME="precision_check"
MODEL_SHORT=$(echo $MODEL | sed 's/.*\///' | tr '[:upper:]' '[:lower:]' | tr '-' '_')

echo "======================================"
echo "Precision Check Pipeline"
echo "======================================"
echo "Model: ${MODEL}"
echo "Test mode: ${TEST_MODE}"
echo "Started at: $(date)"
echo ""

source ~/.bashrc
conda activate cot

mkdir -p results logs checkpoints/precision_check

# ==========================================
# STEP 1: Generate Calibrated Baselines
# ==========================================
echo "======================================"
echo "STEP 1: Generate Calibrated Baselines"
echo "======================================"

BASELINES_PATH="results/precision_check_baselines.json"
AGE_SWAP_PATH="results/precision_check_age_swap.json"

if [ -f "${BASELINES_PATH}" ]; then
    BASELINE_COUNT=$(python -c "import json; print(len(json.load(open('${BASELINES_PATH}'))))")
    echo "Baselines file exists with ${BASELINE_COUNT} entries (will resume if incomplete)"
fi

srun --partition=frink --cpus-per-task=4 --mem=16G --time=2:00:00 \
    bash -c "source ~/.bashrc && conda activate cot && cd ${MEDPERTURB_DIR} && python code/precision_check_baselines.py \
        --age_swap '${AGE_SWAP_PATH}' \
        --dataset data_with_baselines.csv \
        --output '${BASELINES_PATH}' \
        --model '${MODEL}'"

BASELINE_COUNT=$(python -c "import json; print(len(json.load(open('${BASELINES_PATH}'))))")
echo "Baselines complete: ${BASELINE_COUNT} entries"
echo ""

# ==========================================
# STEP 2: Run Evaluation
# ==========================================
echo "======================================"
echo "STEP 2: Age Swap Evaluation"
echo "======================================"

if [ "$TEST_MODE" = true ]; then
    srun --partition=frink --gres=gpu:1 --cpus-per-task=8 --mem=80G --time=2:00:00 \
        bash -c "source ~/.bashrc && conda activate cot && python code/precision_check_evaluate.py \
            --model ${MODEL} \
            --dataset data_with_baselines.csv \
            --age_swap results/precision_check_age_swap.json \
            --baselines results/precision_check_baselines.json \
            --output_dir results \
            --checkpoint_dir checkpoints/precision_check \
            --gpu_id 0 \
            --total_gpus 1 \
            --sample_size 5"
else
    # Production: sbatch with auto-relaunch
    check_completion() {
        COMPLETE_COUNT=$(ls -1 checkpoints/precision_check/precision_check_eval_${MODEL_SHORT}_gpu*_of_${TOTAL_GPUS}_COMPLETE 2>/dev/null | wc -l)
        [ "$COMPLETE_COUNT" -eq $TOTAL_GPUS ]
    }

    check_jobs_running() {
        RUNNING_JOBS=$(squeue -u $USER -n ${JOB_NAME} -h 2>/dev/null | wc -l)
        [ "$RUNNING_JOBS" -gt 0 ]
    }

    ITERATION=1
    while [ $ITERATION -le $MAX_ITERATIONS ]; do
        echo "[$(date)] Iteration ${ITERATION}"

        if check_completion; then
            echo "All ${TOTAL_GPUS}/${TOTAL_GPUS} GPUs complete!"
            break
        fi

        COMPLETE_COUNT=$(ls -1 checkpoints/precision_check/precision_check_eval_${MODEL_SHORT}_gpu*_of_${TOTAL_GPUS}_COMPLETE 2>/dev/null | wc -l)
        echo "Progress: ${COMPLETE_COUNT}/${TOTAL_GPUS} GPUs complete"

        if check_jobs_running; then
            squeue -u $USER -n ${JOB_NAME}
            echo "Waiting ${CHECK_INTERVAL}s..."
            sleep $CHECK_INTERVAL
        else
            echo "Launching jobs..."
            SUBMITTED=$(sbatch slurm/run_precision_check.sbatch "${MODEL}" | awk '{print $NF}')
            echo "Submitted job: ${SUBMITTED}"
            sleep 60

            if check_jobs_running; then
                echo "Jobs started"
                squeue -u $USER -n ${JOB_NAME}
            else
                echo "Warning: Jobs may not have started"
            fi

            echo "Waiting ${CHECK_INTERVAL}s..."
            sleep $CHECK_INTERVAL
        fi

        ITERATION=$((ITERATION + 1))
    done

    if ! check_completion; then
        echo "FATAL: exceeded ${MAX_ITERATIONS} iterations without completion"
        exit 1
    fi
fi

# ==========================================
# STEP 3: Merge Results
# ==========================================
echo "======================================"
echo "STEP 3: Merge Results"
echo "======================================"

MERGED_OUTPUT="results/precision_check_evaluation_${MODEL_SHORT}.json"

if [ "$TEST_MODE" = true ]; then
    TEST_FILE=$(ls -1 results/precision_check_eval_${MODEL_SHORT}_gpu0_of_1.json 2>/dev/null | head -1)
    if [ -n "${TEST_FILE}" ]; then
        cp "${TEST_FILE}" "${MERGED_OUTPUT}"
        echo "Test mode: copied to ${MERGED_OUTPUT}"
    else
        echo "Warning: No test result file found"
    fi
else
    python << EOF
import json
import glob

result_files = glob.glob('results/precision_check_eval_${MODEL_SHORT}_gpu*_of_*.json')

print(f"Found {len(result_files)} result files")

seen_ids = set()
merged = []
for f in sorted(result_files):
    with open(f, 'r') as fp:
        data = json.load(fp)
    for item in data:
        cid = item['context_id']
        if cid not in seen_ids:
            seen_ids.add(cid)
            merged.append(item)

output = '${MERGED_OUTPUT}'
with open(output, 'w') as f:
    json.dump(merged, f, indent=2)

print(f"Merged {len(merged)} unique results to {output}")
EOF
fi

echo ""
echo "======================================"
echo "Pipeline Complete!"
echo "======================================"
echo "Evaluation: ${MERGED_OUTPUT}"
echo "Completed at: $(date)"
echo "======================================"
