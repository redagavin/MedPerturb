# ABOUTME: Evaluate models on neutral sentence baseline rows only (dataset_id=6)
# ABOUTME: Supports data-parallel sharding across multiple GPUs

import pandas as pd
import argparse
import json
import os
import tempfile
import shutil
from tqdm import tqdm

from evaluate_models import ModelEvaluator


def atomic_save_csv(df: pd.DataFrame, path: str) -> None:
    """Write CSV atomically to prevent corruption on crash."""
    dir_path = os.path.dirname(path) or '.'
    with tempfile.NamedTemporaryFile('w', delete=False, dir=dir_path, suffix='.tmp') as f:
        df.to_csv(f, index=False)
        temp_path = f.name
    shutil.move(temp_path, path)


def atomic_save_json(data, path: str) -> None:
    """Write JSON atomically to prevent corruption on crash."""
    dir_path = os.path.dirname(path) or '.'
    with tempfile.NamedTemporaryFile('w', delete=False, dir=dir_path, suffix='.tmp') as f:
        json.dump(data, f, indent=2)
        temp_path = f.name
    shutil.move(temp_path, path)


def get_neutral_baseline_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to neutral baseline rows only (dataset_id=6)."""
    return df[df['dataset_id'] == 6].copy()


def shard_data(df: pd.DataFrame, gpu_id: int, total_gpus: int) -> pd.DataFrame:
    """Shard data for data-parallel processing."""
    return df.iloc[gpu_id::total_gpus].copy()


def get_column_prefix(model_name: str) -> str:
    """Map model name to column prefix."""
    if 'Llama-3' in model_name and '8B' in model_name:
        return 'LLAMA3'
    elif 'Llama-3' in model_name and '70B' in model_name:
        return 'LLAMA3-70'
    else:
        raise ValueError(f"Unknown model: {model_name}")


def majority_vote(responses: list) -> int:
    """Return majority vote from list of 0/1 responses."""
    return 1 if sum(responses) > len(responses) / 2 else 0


def evaluate_neutral_baseline(
    df: pd.DataFrame,
    model_name: str,
    gpu_id: int = 0,
    total_gpus: int = 1,
    checkpoint_dir: str = 'checkpoints/neutral_baseline_eval',
    checkpoint_freq: int = 10
) -> pd.DataFrame:
    """
    Evaluate model on neutral baseline rows (dataset_id=6).

    Args:
        df: Dataset containing neutral baseline rows
        model_name: Model to evaluate
        gpu_id: This GPU's ID (for sharding)
        total_gpus: Total GPUs (for sharding)
        checkpoint_dir: Directory for checkpoints
        checkpoint_freq: Save every N rows

    Returns:
        pd.DataFrame: Results with model outputs for neutral baseline rows
    """
    # Get neutral baseline rows and shard
    neutral_df = get_neutral_baseline_rows(df)
    shard = shard_data(neutral_df, gpu_id, total_gpus)

    print(f"GPU {gpu_id}/{total_gpus}: Evaluating {len(shard)} neutral baseline rows")

    # Load model
    evaluator = ModelEvaluator(model_name)
    prefix = get_column_prefix(model_name)

    # Checkpoint path
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/neutral_baseline_eval_gpu{gpu_id}.csv"

    # Load checkpoint if exists (preserving previous results on resume)
    completed_indices = set()
    results = []
    trace_data = []
    if os.path.exists(checkpoint_path):
        checkpoint_df = pd.read_csv(checkpoint_path)
        # Explicit int() coercion for type safety (CSV can read as float/string)
        completed_indices = {int(idx) for idx in checkpoint_df['Index'].tolist()}
        results = checkpoint_df.to_dict('records')
        trace_path = checkpoint_path.replace('.csv', '_trace.json')
        if os.path.exists(trace_path):
            with open(trace_path) as f:
                trace_data = json.load(f)
        print(f"  Resuming from checkpoint: {len(completed_indices)} completed")

    # Evaluate
    for idx, row in tqdm(shard.iterrows(), total=len(shard), desc=f"GPU {gpu_id}"):
        if int(row['Index']) in completed_indices:
            continue

        # Evaluate triage questions
        triage_results = evaluator.evaluate_triage(row['clinical_context'])

        result = {
            'Index': int(row['Index']),
            f'{prefix}_MANAGE': majority_vote(triage_results['MANAGE']['binary_answers']),
            f'{prefix}_VISIT': majority_vote(triage_results['VISIT']['binary_answers']),
            f'{prefix}_RESOURCE': majority_vote(triage_results['RESOURCE']['binary_answers']),
        }
        results.append(result)

        trace_entry = {'Index': int(row['Index'])}
        for qt in ['MANAGE', 'VISIT', 'RESOURCE']:
            trace_entry[qt] = triage_results[qt]
        trace_data.append(trace_entry)

        # Checkpoint (atomic to prevent corruption on crash)
        if len(results) % checkpoint_freq == 0:
            results_df = pd.DataFrame(results)
            atomic_save_csv(results_df, checkpoint_path)
            trace_path = checkpoint_path.replace('.csv', '_trace.json')
            atomic_save_json(trace_data, trace_path)

    # Final save (atomic)
    if results:
        results_df = pd.DataFrame(results)
        atomic_save_csv(results_df, checkpoint_path)
        trace_path = checkpoint_path.replace('.csv', '_trace.json')
        atomic_save_json(trace_data, trace_path)

    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model on neutral sentence baseline rows"
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Model to evaluate')
    parser.add_argument('--dataset', type=str, default='data_with_neutral_baseline.csv',
                        help='Dataset path with neutral baseline rows')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID for this process')
    parser.add_argument('--total_gpus', type=int, default=1,
                        help='Total number of GPUs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/neutral_baseline_eval',
                        help='Checkpoint directory')
    parser.add_argument('--checkpoint_freq', type=int, default=10,
                        help='Save checkpoint every N rows')

    args = parser.parse_args()

    # Auto-detect SLURM array job parameters
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        args.gpu_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        args.total_gpus = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        print(f"Detected SLURM array job: GPU {args.gpu_id} of {args.total_gpus}")

    print(f"Loading dataset: {args.dataset}")
    df = pd.read_csv(args.dataset)

    print(f"Model: {args.model}")
    print(f"GPU: {args.gpu_id}/{args.total_gpus}")

    results_df = evaluate_neutral_baseline(
        df=df,
        model_name=args.model,
        gpu_id=args.gpu_id,
        total_gpus=args.total_gpus,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq
    )

    print(f"\nCompleted {len(results_df)} evaluations")
