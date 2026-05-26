#!/bin/bash
#SBATCH --job-name=sc_main_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=8:00:00
#SBATCH --output=logs/sc_main_eval_%j.out
#SBATCH --error=logs/sc_main_eval_%j.err
#SBATCH --signal=INT@300

# ABOUTME: SC main-experiment evaluation: batched inference at N=10 samples per prompt
# ABOUTME: Single GPU; per-case atomic checkpoints; auto-resumable

MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
MEDPERTURB_DIR="${SLURM_SUBMIT_DIR}"
MODEL_SHORT=$(echo "$MODEL" | sed 's/.*\///' | tr '[:upper:]' '[:lower:]' | tr '-' '_')

source ~/.bashrc
conda activate cot
cd "${MEDPERTURB_DIR}"
mkdir -p results checkpoints/main_experiment_sc logs

python code/main_evaluate_sc.py \
    --model "${MODEL}" \
    --dataset data_with_baselines.csv \
    --output "results/main_evaluation_sc_${MODEL_SHORT}.json" \
    --checkpoint "checkpoints/main_experiment_sc/partial_${MODEL_SHORT}.json" \
    --n_samples 10 \
    --batch_size 4 \
    --max_new_tokens 512 \
    --rng_seed 42

exit $?
