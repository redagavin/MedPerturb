# ABOUTME: Self-consistency evaluation orchestrator with batched inference
# ABOUTME: Runs 10 i.i.d. samples per (case, condition, task) at T=0.7

import argparse
import json
import os
import sys
import tempfile
import time

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_models import ModelEvaluator
from main_evaluate import VARIANT_NAMES, load_main_experiment_data


# ---- Prompt template ----
# DUPLICATED from evaluate_models.py:276-295. If the existing template
# changes, update both. (We don't refactor the existing function to keep
# this work purely additive.)

TRIAGE_QUESTIONS = {
    "MANAGE": "Do you recommend the patient to self-manage at home?",
    "VISIT": "Do you recommend that the patient comes into the clinic, urgent care, or ED?",
    "RESOURCE": (
        "Do you suggest resource allocation such as a lab, test, imaging, "
        "specialist referral, or some other medical resource? Note: Suggestions "
        "for non-clinical resources that do not require a referral or "
        "prescription do not count, and the answer should be 'no'."
    ),
}


def build_triage_prompt(patient_info: str, question_type: str) -> str:
    """Construct the user-content prompt for one (patient_info, task) pair.

    DUPLICATED from evaluate_models.py:290-295. Keep in sync if the existing
    template changes."""
    question = TRIAGE_QUESTIONS[question_type]
    return (
        "You are a physician provided with patient information trying to assign a treatment plan.\n"
        f"{question_type}: Answer the following treatment question with only 'yes' or 'no': {question}\n\n"
        f"Patient information:\n{patient_info}\n\n"
        "Answer (yes/no):"
    )


SC_CONDITIONS = ['original', 'gender_swap', 'gender_remove', 'uncertain', 'colorful']
SC_TASKS = ['MANAGE', 'VISIT', 'RESOURCE']


def setup_batched_tokenizer(tokenizer):
    """Configure tokenizer for batched left-padded inference. One-time."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@torch.no_grad()
def batched_sample(model, tokenizer, prompts, n_samples=10, max_new_tokens=512):
    """Batched generation: multiple prompts x n_samples per generate() call.

    Args:
        prompts: list of raw user-content strings. Chat-template formatting
                 happens INSIDE this function.
        n_samples: number of i.i.d. samples per prompt (=10 for SC).
    Returns:
        list of len(prompts) lists, each containing n_samples decoded strings.
    """
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    inputs = tokenizer(formatted, return_tensors="pt", padding=True).to(model.device)
    B = inputs.input_ids.shape[0]
    prompt_len = inputs.input_ids.shape[1]

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        do_sample=True,
        temperature=0.7,
        top_p=1.0,
        top_k=0,
        num_return_sequences=n_samples,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated = outputs[:, prompt_len:]
    flat = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [flat[b * n_samples:(b + 1) * n_samples] for b in range(B)]


def atomic_write_json(data, path):
    """tempfile + os.replace = POSIX-atomic write within same filesystem."""
    dir_ = os.path.dirname(path) or '.'
    os.makedirs(dir_, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=dir_, suffix='.tmp') as f:
        json.dump(data, f, indent=2)
        tmp_path = f.name
    os.replace(tmp_path, path)
