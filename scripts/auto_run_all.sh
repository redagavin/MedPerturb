#!/bin/bash
# ABOUTME: Master orchestration for full experiment matrix
# ABOUTME: Runs simulations, evaluations (3 scenarios x 2 models), and analysis

set -euo pipefail

TEST_MODE=false
if [ "${1:-}" = "--test" ]; then
    TEST_MODE=true
    TEST_FLAG="--test"
fi

source ~/.bashrc
conda activate cot

cd /scratch/yang.zih/cot_faithfulness/MedPerturb

# Step 1: Launch simulations in parallel (CPU, fire-and-forget)
echo "=== Launching simulations ==="
sbatch slurm/run_simulation.sbatch

# Step 2: Llama 8B evaluations (sequential)
MODEL_8B="meta-llama/Llama-3.1-8B-Instruct"
echo "=== Llama 8B: Main experiment ==="
bash scripts/auto_run_main_experiment.sh "$MODEL_8B" ${TEST_FLAG:-}
echo "=== Llama 8B: Sanity check ==="
bash scripts/auto_run_sanity_check.sh "$MODEL_8B" ${TEST_FLAG:-}
echo "=== Llama 8B: Precision check ==="
bash scripts/auto_run_precision_check.sh "$MODEL_8B" ${TEST_FLAG:-}

# Step 3: Llama 70B evaluations (sequential)
MODEL_70B="meta-llama/Llama-3.1-70B-Instruct"
echo "=== Llama 70B: Main experiment ==="
bash scripts/auto_run_main_experiment.sh "$MODEL_70B" ${TEST_FLAG:-}
echo "=== Llama 70B: Sanity check ==="
bash scripts/auto_run_sanity_check.sh "$MODEL_70B" ${TEST_FLAG:-}
echo "=== Llama 70B: Precision check ==="
bash scripts/auto_run_precision_check.sh "$MODEL_70B" ${TEST_FLAG:-}

# Step 4: Run analysis per model (CPU)
echo "=== Running analysis ==="
for MODEL in "$MODEL_8B" "$MODEL_70B"; do
    MS=$(echo "$MODEL" | sed 's/.*\///' | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    echo "Analysis for ${MS}..."
    python case_studies/baseline_analysis.py \
        --evaluation "results/main_evaluation_${MS}.json" \
        --output "results/baseline_analysis_${MS}.xlsx"
    python case_studies/sanity_check_analysis.py \
        --evaluation "results/sanity_check_evaluation_${MS}.json" \
        --output "results/sanity_check_analysis_${MS}.xlsx"
    python case_studies/precision_check_analysis.py \
        --evaluation "results/precision_check_evaluation_${MS}.json" \
        --output "results/precision_check_analysis_${MS}.xlsx"
done

echo "=== All complete ==="
