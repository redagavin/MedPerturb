#!/bin/bash
# ABOUTME: Auto-relaunch wrapper for SC main-experiment evaluation
# ABOUTME: Loops sbatch until all 100 cases complete; aborts if progress stalls

set -eo pipefail
MAX_ITERATIONS=10
MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDPERTURB_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${MEDPERTURB_DIR}"
MODEL_SHORT=$(echo "$MODEL" | sed 's/.*\///' | tr '[:upper:]' '[:lower:]' | tr '-' '_')
CHECKPOINT="checkpoints/main_experiment_sc/partial_${MODEL_SHORT}.json"
OUTPUT="results/main_evaluation_sc_${MODEL_SHORT}.json"

source ~/.bashrc
conda activate cot

count_cases() {
    if [ -f "$1" ]; then
        python -c "import json; print(len(json.load(open('$1'))))" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

PREV_COUNT=-1
for ITER in $(seq 1 $MAX_ITERATIONS); do
    CURRENT_COUNT=$(count_cases "${CHECKPOINT}")
    echo "[$(date)] Iter ${ITER}: ${CURRENT_COUNT} cases in checkpoint"

    if [ "${CURRENT_COUNT}" -ge 100 ]; then
        echo "All 100 cases complete."
        cp "${CHECKPOINT}" "${OUTPUT}"
        exit 0
    fi

    # Infinite-loop guard: abort if no progress across two consecutive iterations.
    if [ "$ITER" -gt 1 ] && [ "${CURRENT_COUNT}" -le "${PREV_COUNT}" ]; then
        echo "FATAL: case count stalled at ${CURRENT_COUNT} across two iterations."
        exit 1
    fi

    JOB=$(sbatch slurm/run_main_experiment_sc.sbatch "${MODEL}" | awk '{print $NF}')
    echo "[$(date)] Iter ${ITER}: submitted ${JOB}; waiting for completion..."

    # Wait for the job to finish
    while squeue -j "${JOB}" -h 2>/dev/null | grep -q .; do
        sleep 60
    done

    PREV_COUNT="${CURRENT_COUNT}"
done

echo "FATAL: exceeded ${MAX_ITERATIONS} iterations without completion."
exit 1
