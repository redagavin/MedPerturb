#!/bin/bash
# ABOUTME: Automated sanity check pipeline with auto-relaunch
# ABOUTME: Runs gender question evaluation and MI analysis with merge step

set -e

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
JOB_NAME="sanity_check"
MODEL_SHORT=$(echo $MODEL | sed 's/.*\///' | tr '[:upper:]' '[:lower:]' | tr '-' '_')

echo "======================================"
echo "Sanity Check Pipeline"
echo "======================================"
echo "Model: ${MODEL}"
echo "Test mode: ${TEST_MODE}"
echo "Started at: $(date)"
echo ""

source ~/.bashrc
conda activate cot

mkdir -p results logs checkpoints/sanity_check

# ==========================================
# STEP 1: Run Evaluation
# ==========================================
echo "======================================"
echo "STEP 1: Gender Question Evaluation"
echo "======================================"

if [ "$TEST_MODE" = true ]; then
    srun --partition=177huntington --gres=gpu:a100:1 --cpus-per-task=8 --mem=80G --time=2:00:00 \
        bash -c "source ~/.bashrc && conda activate cot && python code/sanity_check_evaluate.py \
            --model ${MODEL} \
            --dataset data_with_baselines.csv \
            --output results/sanity_check_eval_test.json \
            --checkpoint_dir checkpoints/sanity_check \
            --checkpoint_freq 1 \
            --gpu_id 0 \
            --total_gpus 1 \
            --sample_size 5"
else
    # Production: sbatch with auto-relaunch
    check_completion() {
        COMPLETE_COUNT=$(ls -1 checkpoints/sanity_check/${MODEL_SHORT}_gpu*_of_${TOTAL_GPUS}_COMPLETE 2>/dev/null | wc -l)
        [ "$COMPLETE_COUNT" -eq $TOTAL_GPUS ]
    }

    check_jobs_running() {
        RUNNING_JOBS=$(squeue -u $USER -n ${JOB_NAME} -h 2>/dev/null | wc -l)
        [ "$RUNNING_JOBS" -gt 0 ]
    }

    ITERATION=1
    while true; do
        echo "[$(date)] Iteration ${ITERATION}"

        if check_completion; then
            echo "All ${TOTAL_GPUS}/${TOTAL_GPUS} GPUs complete!"
            break
        fi

        COMPLETE_COUNT=$(ls -1 checkpoints/sanity_check/${MODEL_SHORT}_gpu*_of_${TOTAL_GPUS}_COMPLETE 2>/dev/null | wc -l)
        echo "Progress: ${COMPLETE_COUNT}/${TOTAL_GPUS} GPUs complete"

        if check_jobs_running; then
            squeue -u $USER -n ${JOB_NAME}
            echo "Waiting ${CHECK_INTERVAL}s..."
            sleep $CHECK_INTERVAL
        else
            echo "Launching jobs..."
            SUBMITTED=$(sbatch slurm/run_sanity_check.sbatch "${MODEL}" | awk '{print $NF}')
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
fi

# ==========================================
# STEP 2: Merge Results
# ==========================================
echo "======================================"
echo "STEP 2: Merge Results"
echo "======================================"

if [ "$TEST_MODE" = true ]; then
    MERGED_OUTPUT="results/sanity_check_evaluation_llama3.json"
    cp results/sanity_check_eval_test.json "${MERGED_OUTPUT}"
    echo "Test mode: copied to ${MERGED_OUTPUT}"
else
    python << 'EOF'
import json
import glob

result_files = glob.glob('results/sanity_check_eval_*.json')
result_files = [f for f in result_files if 'test' not in f]

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

output = 'results/sanity_check_evaluation_llama3.json'
with open(output, 'w') as f:
    json.dump(merged, f, indent=2)

print(f"Merged {len(merged)} unique results to {output}")
EOF
fi

# ==========================================
# STEP 3: Run Analysis
# ==========================================
echo "======================================"
echo "STEP 3: MI Analysis"
echo "======================================"

python case_studies/sanity_check_analysis.py \
    --evaluation results/sanity_check_evaluation_llama3.json \
    --output results/sanity_check_analysis.xlsx

echo ""
echo "======================================"
echo "Pipeline Complete!"
echo "======================================"
echo "Evaluation: results/sanity_check_evaluation_llama3.json"
echo "Analysis: results/sanity_check_analysis.xlsx"
echo "Completed at: $(date)"
echo "======================================"
