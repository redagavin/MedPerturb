# ABOUTME: Generate calibrated baseline paraphrases for MedPerturb perturbations
# ABOUTME: Creates token-matched paraphrases to test perturbation-specific vs general sensitivity

import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/src')

from token_edit_distance import levenshtein_distance


def compute_token_change_percent(original, perturbed, tokenizer=None, use_words=False):
    """
    Compute token change percentage between original and perturbed text.

    Args:
        original: Original text
        perturbed: Perturbed text
        tokenizer: HuggingFace tokenizer (optional if use_words=True)
        use_words: If True, use word-level comparison (for testing)

    Returns:
        float: Percentage of tokens changed
    """
    if use_words:
        orig_tokens = original.split()
        pert_tokens = perturbed.split()
    else:
        orig_tokens = tokenizer.encode(original, add_special_tokens=False)
        pert_tokens = tokenizer.encode(perturbed, add_special_tokens=False)

    if len(orig_tokens) == 0:
        return 0.0 if len(pert_tokens) == 0 else 100.0

    edit_dist = levenshtein_distance(orig_tokens, pert_tokens)
    return (edit_dist / len(orig_tokens)) * 100
