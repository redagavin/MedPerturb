#!/bin/bash
# ABOUTME: Auto-relaunch wrapper for SC main-experiment evaluation (4-way array)
# ABOUTME: Submits array job, waits, merges shards, relaunches if incomplete

set -eo pipefail
MAX_ITERATIONS=5
TOTAL_GPUS=4
MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDPERTURB_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${MEDPERTURB_DIR}"
MODEL_SHORT=$(echo "$MODEL" | sed 's/.*\///' | tr '[:upper:]' '[:lower:]' | tr '-' '_')

source ~/.bashrc
conda activate cot

check_all_shards_done() {
    for gpu in $(seq 0 $((TOTAL_GPUS - 1))); do
        SHARD="results/main_evaluation_sc_${MODEL_SHORT}_gpu${gpu}_of_${TOTAL_GPUS}.json"
        [ ! -f "${SHARD}" ] && return 1
    done
    return 0
}

merge_shards() {
    MERGED="results/main_evaluation_sc_${MODEL_SHORT}.json"
    python3 << EOF
import json, glob
shards = sorted(glob.glob('results/main_evaluation_sc_${MODEL_SHORT}_gpu*_of_${TOTAL_GPUS}.json'))
print(f"Merging {len(shards)} shards...")
seen = set()
merged = []
for f in shards:
    for item in json.load(open(f)):
        cid = item['context_id']
        if cid not in seen:
            seen.add(cid)
            merged.append(item)
with open('${MERGED}', 'w') as f:
    json.dump(merged, f, indent=2)
errors = sum(1 for c in merged if 'error' in c)
print(f"Merged {len(merged)} cases ({errors} errors) to ${MERGED}")
EOF
}

PREV_SHARD_COUNT=-1
for ITER in $(seq 1 $MAX_ITERATIONS); do
    echo "[$(date)] Iter ${ITER}"

    if check_all_shards_done; then
        echo "All ${TOTAL_GPUS} shards complete. Merging..."
        merge_shards
        echo "Done!"
        exit 0
    fi

    # Count how many shards exist
    CURRENT_SHARD_COUNT=$(ls -1 results/main_evaluation_sc_${MODEL_SHORT}_gpu*_of_${TOTAL_GPUS}.json 2>/dev/null | wc -l)
    echo "  Shards complete: ${CURRENT_SHARD_COUNT}/${TOTAL_GPUS}"

    if [ "$ITER" -gt 1 ] && [ "${CURRENT_SHARD_COUNT}" -le "${PREV_SHARD_COUNT}" ]; then
        echo "FATAL: shard count stalled at ${CURRENT_SHARD_COUNT}."
        exit 1
    fi

    JOB=$(sbatch slurm/run_main_experiment_sc.sbatch "${MODEL}" | awk '{print $NF}')
    echo "  Submitted array job ${JOB}; waiting..."

    while squeue -j "${JOB}" -h 2>/dev/null | grep -q .; do
        sleep 60
    done

    PREV_SHARD_COUNT="${CURRENT_SHARD_COUNT}"
done

echo "FATAL: exceeded ${MAX_ITERATIONS} iterations."
exit 1
